"""The three supervised screens: regressors that predict a code's score so only the best-looking codes are paid.

A screen is the plain supervised alternative to a generative policy. It never proposes codes itself: it ranks
uniform draws from the generating arena, and only the top of the ranking reaches the oracle.

Matched screen, n = 180. `ScreenTransformer` is GFlowQEC-offline's transformer body turned into a regressor
(no causal mask, one scalar head), so it is matched in size to the policy. It trains offline on the policy's
own data stream (the 116,972 stored forms, `policy.augment`) and is deployed like the policy's second rule:
ten (or a hundred) uniform candidates per paid call, ranked, the top 1,000 paid. Four configurations were
tried: target T1 (the stored score, z-scored per bias) or T2 (the policy's log-reward), each with ("tfr") or
without ("plain") the relational counts added to the start vector. One was chosen by `choose_rule` on a fixed
held-out tenth of the stored forms, then retrained on all of them.

Earlier screen, n = 180. A two-hidden-layer perceptron (`screen_net`) on the bag encoding plus the relational
counts plus the condition vector, trained on the same stored forms (`train_mlp_screen`) and deployed on the
same uniform candidates (`propose`).

Self-screen, n = 300. `arm_selfscreen` is a search arm: the same perceptron, with no condition, refitted on
the run's own paid calls every 20 calls, picking the next 20 from a fresh slate of 2,000 uniform candidates.
"""

import hashlib
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn

from gflowqec.codes import (GROUP_180, _REL_ENTRIES, RelTables, canonicalise, conj_table, draw_gen, generates_g,
                            group_for, mirror, orbit_key, orbit_rep, rel_features)
from gflowqec.counts import N180, value
from gflowqec.oracle import TRAIN_LAM, bias_score, cond_vec, make_oracle
from gflowqec.policy import BASE_STEPS, BLOCK_OF, LAMS, SCALE_STEPS, augment


# =============================================================================================
# Matched screen, n = 180
# =============================================================================================

CONFIGS = ("T1plain", "T2plain", "T1tfr", "T2tfr")
MAX_STEPS = SCALE_STEPS["x20"]          # 315,280: the policy's update count, the most the chosen screen may train
EVAL_EVERY = BASE_STEPS                 # 15,764: twenty selection points up to MAX_STEPS
BATCH = 256
TIE = 0.01                              # selection points within 0.01 of the best mean are ties
# the earlier, unscreened ranking cell of the policy: its candidates are the ones the matched screen must rank
U10 = "r177_samples_transformer_w18_b256_x20_e10Z_n1000_u10_s{seed}.json"
# the earlier screen's paid picks: they too must lie inside the matched screen's candidates
MLP_PICKS = "r174_proxy_pool116972_MR_e10{sign}_n1000_o{over}_h64_st315280_s{seed}.json"


def rel_batch(trajs, T):
    """codes.rel_features of B full ten-tuples at once, element for element: per-entry ratio histograms, the
    two class histograms and their outer product (the joint A x B class histogram counts every (A ratio,
    B ratio) pair, which is exactly hist_A[a] * hist_B[b])."""
    trajs = np.asarray(trajs, dtype=np.int64)
    B, n, nc = len(trajs), T.n, T.n_cls
    out = np.zeros((B, T.dim), dtype=np.float32)
    hist = {"A": np.zeros((B, nc), dtype=np.float32), "B": np.zeros((B, nc), dtype=np.float32)}
    rows = np.arange(B)
    for k, (ent, side, has_e) in enumerate(_REL_ENTRIES):
        terms = [trajs[:, p] for p in ent] + ([np.full(B, T.e)] if has_e else [])
        for i, t in enumerate(terms):
            for j, u in enumerate(terms):
                if i == j:
                    continue
                r = T.mul[t, T.inv[u]] if side == "A" else T.mul[T.inv[t], u]
                w = (t != u).astype(np.float32)          # rel_features skips a pair of equal values
                np.add.at(out, (rows, k * n + r), w)
                np.add.at(hist[side], (rows, T.cls[r]), w)
    base = 4 * n
    out[:, base:base + nc] = hist["A"]
    out[:, base + nc:base + 2 * nc] = hist["B"]
    out[:, base + 2 * nc:] = (hist["A"][:, :, None] * hist["B"][:, None, :]).reshape(B, nc * nc)
    return out


class ScreenTransformer(nn.Module):
    """The policy's TupleTransformer body as a regressor: no causal mask, the start position plus all ten tokens
    (eleven positions), one scalar head on the start position. With `rel_dim` (the "tfr" configurations) a
    Linear(504, 128) of the relational counts is added to the start vector."""

    def __init__(self, n, rel_dim=0, d=128, layers=3, heads=4):
        super().__init__()
        self.tok = nn.Embedding(n, d)
        self.start = nn.Parameter(torch.zeros(d))
        self.cond = nn.Linear(2, d)
        self.pos = nn.Embedding(11, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        # enable_nested_tensor=False only silences torch's warning: with norm_first=True the
        # nested-tensor fast path is switched off anyway, so the computation is unchanged.
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.rel = nn.Linear(rel_dim, d) if rel_dim else None
        self.head = nn.Linear(d, 1)

    def forward(self, tokens, cond, rel=None):
        """tokens (B, 10) long, cond (B, 2), rel (B, 504) or None -> (B,) predicted target."""
        x0 = self.start + self.cond(cond)
        if self.rel is not None:
            x0 = x0 + self.rel(rel)
        x = torch.cat([x0.unsqueeze(1), self.tok(tokens)], dim=1)
        x = x + self.pos(torch.arange(11, device=tokens.device)).unsqueeze(0)
        return self.head(self.body(x)[:, 0]).squeeze(-1)


def build_net(config, n, T):
    """The network of one configuration ("tfr" configurations also read the relational counts)."""
    return ScreenTransformer(n, rel_dim=T.dim if config.endswith("tfr") else 0)


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def predict(net, trajs, lam, config, T, device="cpu", bs=4096):
    """The model's prediction for stored-form or candidate tuples at ONE bias, in batches, as numpy.

    The net is never put in eval mode: dropout is 0 and there is no batch normalisation, so train and eval mode
    compute the same function, and training, selection and deployment share one forward pass."""
    out = []
    with torch.no_grad():
        for a in range(0, len(trajs), bs):
            chunk = trajs[a:a + bs]
            tok = torch.tensor(chunk, dtype=torch.long, device=device)
            cond = torch.tensor([cond_vec(lam)] * len(chunk), dtype=torch.float32, device=device)
            rel = torch.from_numpy(rel_batch(chunk, T)).to(device) if config.endswith("tfr") else None
            out.append(net(tok, cond, rel).cpu().numpy())
    return np.concatenate(out)


def zstats(rows, lams):
    """{lam: (mean, std)} of the stored score over `rows` (population std plus 1e-9)."""
    out = {}
    for lam in lams:
        s = np.array([value(r, lam) for r in rows])
        out[lam] = (float(s.mean()), float(s.std() + 1e-9))
    return out


def target_of(r, lam, kind, stats, beta):
    """T1: the stored score at lam, z-scored per bias; T2: the policy's log-reward beta log max(score, 0.01)."""
    s = value(r, lam)
    if kind == "T1":
        return (s - stats[lam][0]) / stats[lam][1]
    return beta[lam] * math.log(max(s, 1e-2))


def split_rows(pool):
    """The fixed 90/10 split by orbit key used to choose the configuration: perm = default_rng(0).permutation of
    the sorted keys, the first round(0.1 m) keys held out. Returns (training rows in pool order, held-out rows in
    perm order, md5 of the sorted held-out keys)."""
    keys = sorted(pool)
    perm = np.random.default_rng(0).permutation(len(keys))
    held_keys = [keys[i] for i in perm[:round(0.1 * len(keys))]]
    held_set = set(held_keys)
    train = [r for k, r in pool.items() if k not in held_set]
    held = [pool[k] for k in held_keys]
    return train, held, hashlib.md5(repr(sorted(held_keys)).encode()).hexdigest()


def precision_at_top(pred, labels):
    """Share of true labels among the ceil(1 % of m) highest predictions (stable sort, ties keep input order)."""
    k = (len(pred) + 99) // 100
    order = np.argsort(-np.asarray(pred), kind="stable")[:k]
    return float(np.mean(np.asarray(labels, dtype=float)[order]))


def policy_batch(rows, rng, kind, stats, beta, conj, T):
    """One step of GFlowQEC-offline's training stream, draw for draw: the picks, the bias gamma0, the target at
    gamma0 BEFORE augment(), then augment(). Returns (augmented trajectories, their bias, targets)."""
    picks = rng.integers(len(rows), size=BATCH)
    trajs, lams, ys = [], [], []
    for i in picks:
        lam0 = float(rng.choice(LAMS))
        ys.append(target_of(rows[i], lam0, kind, stats, beta))
        traj, lam = augment(rows[i]["traj"], lam0, rng, conj, T)
        trajs.append(traj)
        lams.append(lam)
    return trajs, lams, ys


def fit(net, rows, config, steps, rng, device, stats, beta, conj, T, on_eval=None, eval_every=EVAL_EVERY):
    """Squared error on the policy's stream for `steps` Adam updates; on_eval(step, mean loss since the last
    call) every eval_every steps. Returns the per-200-step mean losses."""
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    kind = config[:2]
    losses, trace, t = [], [], time.time()
    for step in range(1, steps + 1):
        trajs, lams, ys = policy_batch(rows, rng, kind, stats, beta, conj, T)
        tok = torch.tensor(trajs, dtype=torch.long).to(device)
        cond = torch.tensor([cond_vec(l) for l in lams], dtype=torch.float32).to(device)
        rel = torch.from_numpy(rel_batch(trajs, T)).to(device) if config.endswith("tfr") else None
        y = torch.tensor(ys, dtype=torch.float32).to(device)
        loss = ((net(tok, cond, rel) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % 200 == 0:
            trace.append(float(np.mean(losses[-200:])))
        if step % 2000 == 0:
            print(f"  step {step}  MSE {np.mean(losses[-2000:]):.4f}  {time.time() - t:.0f}s", flush=True)
        if on_eval is not None and step % eval_every == 0:
            on_eval(step, float(np.mean(losses[-eval_every:])))
    return trace


def choose_rule(curves):
    """The configuration choice: the (configuration, step) with the highest two-bias mean of held-out precision;
    every point within TIE of the best is a tie, broken by plain over TF+R, then T1 over T2, then the later step.
    curves = {config: [{"step", "mean"}, ...]}. Returns (chosen point, best mean, every tied point), points as
    (config, step, mean)."""
    points = [(cfg, p["step"], p["mean"]) for cfg, curve in curves.items() for p in curve]
    best = max(m for _, _, m in points)
    ties = [p for p in points if p[2] >= best - TIE - 1e-12]
    chosen = max(ties, key=lambda p: (p[0].endswith("plain"), p[0].startswith("T1"), p[1]))
    return chosen, best, ties


def mscreen_stem(config, steps, seed):
    """The checkpoint stem, which every downstream samples / screen / certify file of the cell takes. It ends
    with _s{seed} and holds no other '_s<digit>'."""
    return f"r177_mscreen_{config}_st{steps}_s{seed}"


def candidate_check(cands, seed, over, n):
    """The identity check of the matched screen's candidates: it must rank exactly the codes the earlier cells
    ranked. The orbit-key set of the first 10,000 candidates must equal the policy's unscreened u10 ranking
    cell's 10,000 (rows + unpaid), and at n = 1,000 every pick of the earlier screen at this over-sampling must
    lie inside the candidates. Raises on any mismatch; returns the md5 of the sorted candidate keys."""
    keys = [orbit_key(GROUP_180, c) for c in cands]
    md5 = lambda ks: hashlib.md5(repr(sorted(ks)).encode()).hexdigest()
    u = json.load(open(N180 / U10.format(seed=seed)))
    uk = [orbit_key(GROUP_180, r["exps"]) for r in u["rows"] + u["unpaid"]]
    assert len(set(keys)) == len(keys), "a repeated orbit among the candidates"
    assert len(keys) >= len(uk), f"seed {seed}: {len(keys):,} candidates, fewer than the u10 set's {len(uk):,}"
    assert set(keys[:len(uk)]) == set(uk), f"seed {seed}: the first {len(uk):,} candidates are not the u10 set"
    print(f"  check-cands s{seed}: the first {len(uk):,} of {len(keys):,} candidates = the u10 ranker's "
          f"(md5 {md5(keys[:len(uk)])} = {md5(uk)})", flush=True)
    if n == 1000:
        inside = set(keys)
        for sign in ("Z", "X"):
            f = N180 / MLP_PICKS.format(sign=sign, over=over, seed=seed)
            if not f.exists():
                # the earlier screen ran over-sampling 100 at seeds 1-2 only, as the policy's over-100 cells did
                assert over == 100 and seed == 0, f"{f.name} missing: the check cannot run"
                print(f"  check-cands s{seed}: no earlier over-{over} MLP screen at seed {seed} (seeds 1-2 only)")
                continue
            picks = [orbit_key(GROUP_180, r["rep"]) for r in json.load(open(f))["screen"]]
            assert all(k in inside for k in picks), f"{f.name}: an earlier MLP pick lies outside the candidates"
            print(f"  check-cands s{seed}: the {len(picks):,} picks of {f.name} lie inside the candidates", flush=True)
    return md5(keys)


def draw_candidates(seed, over, n):
    """The earlier screen's exact candidate call: over x n distinct generating uniform orbits."""
    return propose(over * n, np.random.default_rng(10_000 + seed))


def rank_rows(cands, pred, novel, n):
    """The ranked deployment: candidates in descending prediction (a stable sort, so equal predictions keep draw
    order), the top n the paid rows, the rest unpaid, both in rank order and in the samples-file schema the paid
    legs read (rows[i]["exps"], rows[i]["novel"])."""
    order = sorted(range(len(cands)), key=lambda i: -pred[i])
    out = [{"exps": [int(v) for v in cands[i]], "pred": float(pred[i]), "novel": bool(novel[i])} for i in order]
    return out[:n], out[n:]


# =============================================================================================
# Earlier screen, n = 180
# =============================================================================================

def propose(m, uniform_rng, cap_mult=50):
    """m DISTINCT generating orbits at zero oracle cost, drawn uniformly, as canonical forms in draw order.
    Returns (codes, attempts)."""
    seen, out, attempts = set(), [], 0
    while len(out) < m and attempts < cap_mult * m:
        attempts += 1
        exps = list(draw_gen(uniform_rng, GROUP_180))
        exps = list(canonicalise(GROUP_180, exps))
        if not generates_g(GROUP_180, exps):
            continue
        k = orbit_key(GROUP_180, exps)
        if k not in seen:
            seen.add(k)
            out.append(exps)
    return out, attempts


def score_rows(cands, lam, budget, kind):
    """Pay every distinct candidate once at lam on the given oracle kind; returns the oracle."""
    oracle = make_oracle(kind, GROUP_180, budget, lam)
    for c in cands:
        oracle.pay(c)
    return oracle


def rows_of(oracle, cands):
    """Every stored field of each paid candidate plus its value at the oracle's condition (already paid, so free)."""
    return [{**{k: v for k, v in oracle.meta[orbit_key(GROUP_180, c)].items() if k != "src"},
             "score": oracle.value(c, oracle.lam)} for c in cands]


def encode(chosen, n, T):
    """The perceptron screens' input: the bag encoding (a one per (entry, element) present) followed by the
    relational counts."""
    x = np.zeros(4 * n, dtype=np.float32)
    for i, a in enumerate(chosen):
        x[BLOCK_OF[i] * n + a] = 1.0
    return np.concatenate([x, rel_features(list(chosen), T)])


def train_mlp_screen(rows, steps, batch, hidden, seed):
    """The earlier n = 180 screen: an MLP on (bag encoding, relational counts, condition) trained on the stored
    forms `rows` (counts.stored_forms().values()) with the same augmentations as the policy (the reordering
    is implicit in the bag, conjugation and mirror explicit). Target: the score at a training bias drawn per
    example, z-scored per bias, using the exact sector distances where the form was certified and the bound
    distances elsewhere. Returns (net, training CPU seconds)."""
    n = len(group_for(GROUP_180)[1])
    T = RelTables(GROUP_180)
    conj = conj_table(GROUP_180)
    rows = [(list(r["traj"]), r["dX"] if r["cert"] else r["dA"], r["dZ"] if r["cert"] else r["dB"]) for r in rows]
    stats = {}
    for l in TRAIN_LAM:
        s = np.array([bias_score(dX, dZ, l) for _, dX, dZ in rows])
        stats[l] = (float(s.mean()), float(s.std() + 1e-9))
    print(f"  {len(rows)} stored forms; targets z-scored per condition", flush=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = screen_net(4 * n + 2 + T.dim, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    t, c0 = time.time(), time.process_time()
    for step in range(steps):
        xs, ys = [], []
        for i in rng.choice(len(rows), batch):
            traj, dX, dZ = rows[i]
            l = float(rng.choice(TRAIN_LAM))
            y = (bias_score(dX, dZ, l) - stats[l][0]) / stats[l][1]
            # a conjugation per side (the residual gauge), then the mirror with probability 1/2, flipping the bias
            gA, gB = rng.integers(n), rng.integers(n)
            traj = [int(conj[gA, v]) for v in traj[:5]] + [int(conj[gB, v]) for v in traj[5:]]
            if rng.random() < 0.5:
                traj, l = mirror(traj, T), -l
            xs.append(np.concatenate([encode(traj, n, T), np.float32(cond_vec(l))]))
            ys.append(y)
        loss = ((net(torch.tensor(np.stack(xs))).squeeze(1) - torch.tensor(ys, dtype=torch.float32)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 5000 == 0:
            print(f"  step {step}: loss {loss.item():.3f} ({time.time() - t:.0f}s)", flush=True)
    return net, time.process_time() - c0


# =============================================================================================
# Self-screen, n = 300
# =============================================================================================

# 200 uniform calibration calls, then rounds: refit on every paid form, rank a fresh slate of 2,000 unpaid uniform
# candidates, pay the top 20. 160 warm-started Adam steps per refit: 1,990 rounds x 160 = 318,400 updates over a
# 40,000-call run, about the earlier n = 180 screen's 315,280.
SS_CALIB, SS_SLATE, SS_BATCH, SS_STEPS = 200, 2000, 20, 160


def screen_net(in_dim, hidden=64):
    """The perceptron screens' network: two hidden layers of 64 with ReLU and a scalar output."""
    return torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
                               torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                               torch.nn.Linear(hidden, 1))


def screen_steps(net, opt, rows, steps, rng, n, T, conj, batch=64):
    """`steps` Adam updates of the self-screen. rows = [(trajectory, y)], y the row's standardised score; there
    is no condition, and the mirror flips none.

    Per example: a row drawn with replacement, a conjugation per side (the residual gauge, codes.conj_table),
    then the mirror (h^-1, g^-1) with probability 1/2."""
    for _ in range(steps):
        xs, ys = [], []
        for i in rng.choice(len(rows), batch):
            traj, y = rows[i]
            gA, gB = rng.integers(n), rng.integers(n)
            traj = [int(conj[gA, v]) for v in traj[:5]] + [int(conj[gB, v]) for v in traj[5:]]
            if rng.random() < 0.5:
                traj = mirror(traj, T)
            xs.append(encode(traj, n, T))
            ys.append(y)
        loss = ((net(torch.tensor(np.stack(xs))).squeeze(1) - torch.tensor(ys, dtype=torch.float32)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def arm_selfscreen(oracle, group, rng, seed, ho):
    """A supervised screen that pays for its own labels.

    SS_CALIB uniform paid calls, drawn exactly as the uniform and BO arms draw them (so the first 200 paid orbits
    and values equal the uniform arm's at the same seed), then rounds of
      (i)   refit the screen (screen_net on bag + relational counts) by regression of the in-run score s on every
            paid canonical form so far, z-scored over the paid rows, SS_STEPS warm-started Adam steps;
      (ii)  draw SS_SLATE fresh uniform generating candidates not yet paid (free, de-duplicated);
      (iii) pay the SS_BATCH with the highest predicted score.
    `rng` stays the calibration and candidate stream; the training draws use their own stream, so the candidate
    stream is never shifted by the fit. `ho` is the dump's extra-field dict; the round counters are updated in
    place so every 1,000-call checkpoint carries them."""
    n = len(group_for(group)[1])
    T = RelTables(group)
    conj = conj_table(group)
    torch.manual_seed(seed)
    fit_rng = np.random.default_rng(10_000 + seed)
    net = screen_net(4 * n + T.dim)
    # the stamp the script wrote before the run must describe the network that actually runs
    assert ho["ss_in_dim"] == 4 * n + T.dim and ho["ss_params"] == sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    paid = {}       # canonical rep (orbit_rep) -> in-run score s, in call order

    def rep_of(exps):
        return tuple(int(v) for v in orbit_rep(group, exps))

    def pay(exps):
        paid[rep_of(exps)] = oracle.score(exps)

    while oracle.calls < SS_CALIB:
        pay(draw_gen(rng, group))
    while oracle.calls < oracle.budget:
        # (i) refit on every paid form so far; the net and Adam's state carry over between rounds
        c0 = time.process_time()
        s = np.array(list(paid.values()))
        mu, sd = s.mean(), s.std() + 1e-9
        rows = [(list(r), (v - mu) / sd) for r, v in paid.items()]
        screen_steps(net, opt, rows, SS_STEPS, fit_rng, n, T, conj)
        c1 = time.process_time()
        # (ii) a fresh slate of unpaid uniform generating candidates
        cand, seen = [], set()
        while len(cand) < SS_SLATE:
            e = draw_gen(rng, group)
            rep = rep_of(e)
            if rep in paid or rep in seen:
                continue
            seen.add(rep)
            cand.append((e, rep))
        with torch.no_grad():
            pred = net(torch.tensor(np.stack([encode(list(r), n, T) for _, r in cand]))).squeeze(1).numpy()
        c2 = time.process_time()
        # (iii) pay the top SS_BATCH by predicted score; ties break by draw order
        for i in np.argsort(-pred, kind="stable")[:SS_BATCH]:
            if oracle.calls >= oracle.budget:
                break
            pay(cand[i][0])
        ho["n_rounds"] += 1
        ho["fit_cpu_s"] += c1 - c0
        ho["cand_cpu_s"] += c2 - c1
        if ho["n_rounds"] <= 3 or ho["n_rounds"] % 50 == 0:
            print(f"  round {ho['n_rounds']}: {oracle.calls} calls, refit {c1 - c0:.2f} s, "
                  f"slate {c2 - c1:.2f} s CPU", flush=True)
