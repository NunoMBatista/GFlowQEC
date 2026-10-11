"""GFlowQEC-offline at n = 180: train the policy on the stored forms, deploy it, and pay what it picks.

The offline policy of the paper (Sec. 6; App. B.1) is a small decoder-only transformer that writes a
two-block lifted-product code over C6 x S3 one group element at a time (the ten slots of codes.ENTRIES). It never
calls the oracle while it trains: its training data are the 116,972 distinct canonical forms that the 42 stored
search runs already paid for (counts.stored_forms), each rewarded with what the certifying oracle returned for it.
It is trained by trajectory balance (GFlowQEC-offline) or, for its maximum-entropy rival, by the soft Bellman loss of
offline soft Q-learning on the same network, data stream and step budget. Then it is deployed: it draws
over x n distinct generating codes, keeps the n it gives the highest log-probability, and only those n are paid,
the two classical bound ladders on every one (screen), the exact sector ladders on those whose bounds reach the
bar (certify).

Four legs, each writing one file under results/ (the shipped copies sit in data/models/ and data/n180/):

    train    r177_transformer_w18_b256_{x20 | stN}[_softq]_s{seed}.pt           the trained policy
    sample   r177_samples_transformer_..._e10{sign}_n{n}[_o{over} | _u{over}]_s{seed}.json    the n picks
    screen   r177_screen_..._s{seed}.json        the bound ladders (dA, dB) on every pick
    certify  r177_certified_..._s{seed}.json     the exact ladders (dX, dZ) on the picks whose bounds reach the bar

"r177", "w18" and "e10" are fixed tokens of the shipped file names (e10 = bias strength eta 10; sign Z is gamma = +1,
sign X is gamma = -1). The readouts count these files into the n = 180 rows of GFlowQEC-offline, offline soft Q, the
policy as a ranker of uniform candidates and the matched screen (Table III; App. D.1, Table IX).

Commands of record (seeds s = 0, 1, 2; signs S = Z, X):

    python -m scripts.n180_offline train --scale x20 --seed s                                    GFlowQEC-offline
    python -m scripts.n180_offline train --scale x20 --objective softq --device cuda --seed s    offline soft Q

    then sample, screen and certify, in that order, with the same flags:
    python -m scripts.n180_offline sample --scale x20 --sign S --seed s --n 1000 --over 10       second deployment (mu = 10)
    python -m scripts.n180_offline sample --scale x20 --sign S --seed s --n 1000 --over 100      hundredfold (seeds 1, 2)
    python -m scripts.n180_offline sample --scale x20 --sign S --seed s --n 1000 --over 10 --source uniform
                                                                    the policy as a ranker of uniform candidates
    python -m scripts.n180_offline sample --scale x20 --sign S --seed s --n 2000                 first, unranked deployment
    python -m scripts.n180_offline sample --scale x20 --objective softq --sign S --seed s --n 1000 --over 10   offline soft Q

    the matched screen's paid legs (its samples file is written by scripts.n180_screens rank; only the stem of
    --model-path is read here, to name the files):
    python -m scripts.n180_offline screen --model-path data/models/r177_mscreen_T1tfr_st252224_s{s}.pt \\
        --source uniform --over 10 --sign S --n 1000 --seed s          (and certify; --over 100 at seeds 1, 2)

    To deploy a shipped checkpoint instead of retraining, add --model-path data/models/<model file>.pt to sample,
    screen and certify; the output names are then exactly the shipped ones. --steps N trains a step-capped cell
    (the shipped st1699 checkpoints of App. B.1's transformer-vs-perceptron comparison; that comparison is printed
    from the shipped files only).

Compute (from the stamps of the paper's runs):
    train      34.4 to 34.6 CPU-hours per seed on 4 threads (9.1 to 9.6 h wall; 315,280 steps at batch 256);
               offline soft Q 3,728 to 3,766 s wall per seed on one GPU
    sample     about 24 s per 10,000 draws
    screen + certify, CPU-hours summed over the cells of one row (3 seeds x 2 signs unless stated):
               mu = 10: 8.3 + 96.3; hundredfold (4 cells): 5.5 + 77.9; ranker: 4.0 + 22.1; unranked: 6.7 + 34.3;
               offline soft Q: 8.4 + 100.2; matched screen mu = 10: 5.2 + 33.0, mu = 100 (4 cells): 9.5 + 135.3
               (one exact certification takes about 2 to 4 minutes)

Random streams: train seeds numpy (the data stream) and torch with --seed; sample re-seeds torch with --seed before
building the network, and the uniform candidates of --source uniform come from numpy default_rng(10,000 + seed),
the same stream the uniform search arm uses, so a re-run reproduces every row.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from gflowqec.certify import bound_ladder, exact_sectors, lifted_bases
from gflowqec.codes import (GROUP_180, RelTables, build, canonicalise, conj_table, draw_gen, generates_g, group_for,
                            identity_index, orbit_key)
from gflowqec.counts import stored_forms, value
from gflowqec.oracle import bias_score, cond_vec, key_beta, key_top, lam_of
from gflowqec.paths import OUT
from gflowqec.policy import (LAMS, SCALE_STEPS, TARGET_EVERY, OnlinePolicy, TupleTransformer, augment, legal_mask,
                             logp_of_trajs, soft_bellman_loss)

WALL_MIN = 18                 # a file-name token only ("w18"): the first runs were capped at 18 minutes of wall clock
BATCH = 256                   # train: rows per gradient step, every run of the paper (the file-name token "b256")
T0 = time.time()
torch.set_num_threads(4)


# ----------------------------------------------------------------------------- file names
def model_path(args):
    """The model file, without ".pt". Every knob that distinguishes a cell is in the name, so a model and all its
    sample / screen / certify files are unambiguous."""
    if args.model_path:
        # load THIS checkpoint (a shipped one in data/models/, say) and name every downstream file by its stem,
        # so those files carry exactly the shipped names. The stem must carry the run's seed, or samples_path
        # would keep two seeds in one name.
        stem = Path(args.model_path)
        if stem.suffix == ".pt":
            stem = stem.with_suffix("")
        assert stem.name.endswith(f"_s{args.seed}"), f"--model-path stem {stem.name} does not end with _s{args.seed}"
        return stem
    bud = f"_{args.scale}" if args.scale else f"_st{args.steps}"      # the training budget
    obj = {"tb": "", "softq": "_softq"}[args.objective]               # soft Q cells carry _softq after the budget
    return OUT / f"r177_transformer_w{WALL_MIN}_b{BATCH}{bud}{obj}_s{args.seed}"


def samples_path(args):
    """The samples file of a deployment. A ranked deployment (over > 1) of the policy's own draws carries _o{over};
    a deployment whose candidates are uniform draws carries _u{over} instead, at every over, so the two can never
    be confused; over 1 (the first, unranked deployment) carries neither."""
    if args.source == "uniform":
        over = f"_u{args.over}"
    elif args.over > 1:
        over = f"_o{args.over}"
    else:
        over = ""
    name = (model_path(args).name.replace("r177_", "r177_samples_").replace(f"_s{args.seed}", "")
            + f"_e10{args.sign}_n{args.n}{over}_s{args.seed}.json")
    return OUT / name


# ----------------------------------------------------------------------------- train
def train_leg(args):
    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    rows = list(stored_forms().values())
    n, ident = len(group_for(GROUP_180)[1]), identity_index(GROUP_180)
    conj, T = conj_table(GROUP_180), RelTables(GROUP_180)
    beta = {lam: key_beta(lam) for lam in LAMS}
    # The closed-form log Z warm start: under the untrained (uniform) policy, trajectory balance wants
    # log Z = mean log R + log(number of legal ordered tuples). The count comes from the online policy's
    # log_n_ordered(), so an OnlinePolicy twin is built here and nothing else of it is used.
    # KEEP THE TWIN, AND KEEP IT HERE: its constructor re-seeds torch and draws its own weights (with its
    # relational inputs), so the transformer built below starts from weights that depend on the twin having been
    # built first. Removing it would change every trained policy.
    twin = OnlinePolicy(n, hidden=64, seed=args.seed, identity=ident, conj=conj, cond_dim=2, rel=T)
    logz0 = {lam: float(np.mean([beta[lam] * math.log(max(value(r, lam), 1e-2)) for r in rows]))
             + twin.log_n_ordered() for lam in LAMS}
    print(f"transformer: {len(rows)} distinct paid orbits, beta {beta}, logZ0 {logz0}", flush=True)

    # initialised on the CPU, then moved: the initial weights are the same on every device
    net = TupleTransformer(n).to(dev)
    net.logz0 = float(np.mean(list(logz0.values())))
    opt = torch.optim.Adam([{"params": [p for name, p in net.named_parameters() if not name.startswith("zc")],
                             "lr": 1e-3}, {"params": net.zc.parameters(), "lr": 1e-2}])

    if args.objective == "softq":
        # The soft Q-learning extras. Same network, same optimizer groups (zc gets no gradient under this loss, so
        # Adam skips it and it stays zero), same data stream; two additions:
        # (1) a TARGET COPY of the network, hard-refreshed every TARGET_EVERY updates, read for the bootstrap
        #     targets V(s_t+1). deepcopy draws no random numbers, so the data stream and the initial weights are
        #     those of the trajectory-balance run of the same seed (a paired stream).
        # (2) a FIXED per-slot offset c_t: Q(s_t, a) = logits[:, t, a] + c_t, with c_t = mean log R + sum over
        #     p > t of log |A_p|, the soft value of the uniform policy (the analogue of the log Z warm start;
        #     c_0 = logz0). It is constant across the actions of a slot, so it changes neither the policy
        #     softmax(logits) = exp(Q - V) nor the fixed point, and the sample / screen legs, which read the bare
        #     logits, rank and draw a soft Q model with no change.
        tnet = copy.deepcopy(net)
        n_legal = legal_mask(np.array([rows[0]["traj"]]), n, ident)[0].sum(1).numpy()   # |A_p|, the same for every row
        mean_logr = net.logz0 - twin.log_n_ordered()            # mean log R over the rows, both conditions
        q_offset = [mean_logr + float(np.sum(np.log(n_legal[p + 1:]))) for p in range(10)]
        c = torch.tensor(q_offset, dtype=torch.float32, device=dev).view(1, 10, 1)
        print(f"softq: |A_p| {n_legal.tolist()}, c_t {[round(v, 3) for v in q_offset]} (logz0 {net.logz0:.3f}), "
              f"target refresh every {TARGET_EVERY} steps, device {dev}", flush=True)

    losses, steps, visits = [], 0, 0
    t_train = time.time()
    lab = {"tb": "TB", "softq": "SQ"}[args.objective]   # the loss label in the progress lines

    def ckpt():
        """The saved dict, one copy for the periodic checkpoint and the final save. cpuTrain is the training's
        processor time (process_time sums this process's threads)."""
        state = net.state_dict()
        if dev.type != "cpu":
            # a GPU run saves CPU tensors, so every reader opens it on a CPU-only box
            state = {k: v.detach().cpu() for k, v in state.items()}
        out = {"arch": "transformer", "state": state, "logz0": logz0, "beta": beta, "steps": steps, "visits": visits,
               "rows": len(rows), "losses": losses, "wall_min": WALL_MIN, "batch": BATCH,
               "seed": args.seed, "cpuTrain": time.process_time()}
        if args.objective == "softq":
            # v0 = V(s0, z) = Q(s0, e; z) (slot 0 is forced to e): the soft Q estimate of log Z(z), reported beside
            # the trajectory-balance model's logz0 + zc(z). "logz0" above keeps its warm-start meaning. On a GPU
            # run cpuTrain misses the GPU's work, so the wall clock and the device are stamped too.
            with torch.no_grad():
                tok = torch.zeros(1, 10, dtype=torch.long, device=dev)
                v0 = {lam: float(c[0, 0, 0] + net.logits(tok, torch.tensor([cond_vec(lam)], dtype=torch.float32,
                                                                              device=dev))[0, 0, ident])
                      for lam in LAMS}
            out.update({"objective": "softq", "q_offset": q_offset, "target_every": TARGET_EVERY,
                        "device": args.device, "wallTrain": time.time() - t_train, "v0": v0})
        return out

    while steps < args.steps:
        batch = []
        picks = rng.integers(len(rows), size=BATCH)
        for i in picks:
            lam0 = float(rng.choice(LAMS))
            # the reward is the row's value at the DRAWN condition; the mirror inside augment() may flip the
            # condition the net sees, and (M x, -lam0) carries exactly that reward
            logr = beta[lam0] * math.log(max(value(rows[i], lam0), 1e-2))
            traj, lam = augment(rows[i]["traj"], lam0, rng, conj, T)
            batch.append((traj, lam, logr))
        tokens = torch.tensor([b[0] for b in batch], dtype=torch.long)
        cond = torch.tensor([cond_vec(b[1]) for b in batch], dtype=torch.float32)
        logr = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        mask = legal_mask(tokens.numpy(), n, ident)
        # built on the CPU, moved once (a no-op on the CPU)
        tokens, cond, logr, mask = tokens.to(dev), cond.to(dev), logr.to(dev), mask.to(dev)
        if args.objective == "softq":
            # the actions ARE the tokens: output t predicts token t from the tokens before it
            q = net.logits(tokens, cond) + c
            with torch.no_grad():
                q_next = tnet.logits(tokens, cond) + c
            loss = soft_bellman_loss(q, q_next, mask, tokens, logr)
        else:
            # trajectory balance: (log Z(z) + log p(x | z) - log R(x))^2, the backward policy being 1 on a tree
            loss = ((net.logZ(cond) + net.logp(tokens, cond, mask) - logr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        steps += 1
        visits += BATCH
        if args.objective == "softq" and steps % TARGET_EVERY == 0:
            tnet.load_state_dict(net.state_dict())              # the hard target refresh
        if steps % 200 == 0:
            print(f"  step {steps}  rows visited {visits}  {lab} loss {np.mean(losses[-200:]):.3f}  "
                  f"{time.time() - T0:.0f}s", flush=True)
        # a full run is hours long with no resume, so checkpoint: a killed job still leaves a usable model whose
        # "steps" field says how far it got (the sample leg refuses it under --scale)
        if steps % 2000 == 0:
            torch.save(ckpt(), str(model_path(args)) + ".pt")
    torch.save(ckpt(), str(model_path(args)) + ".pt")
    first = np.mean(losses[:200])
    last = np.mean(losses[-200:])
    print(f"done: {steps} steps, {visits} row visits, {lab} loss first/last-200 {first:.3f} -> "
          f"{last:.3f}, {time.time() - T0:.0f}s -> {model_path(args).name}.pt", flush=True)


def load_model(args):
    d = torch.load(str(model_path(args)) + ".pt", weights_only=False, map_location="cpu")
    n = len(group_for(GROUP_180)[1])
    net = TupleTransformer(n)
    net.load_state_dict(d["state"])
    net.logz0 = float(np.mean(list(d["logz0"].values())))
    net.eval()
    return net, d


# ----------------------------------------------------------------------------- sample
def sample_leg(args):
    lam = lam_of(10, args.sign)
    # the certify leg joins to this file BY ROW INDEX, so a re-run must reproduce the same orbit list
    torch.manual_seed(args.seed)
    model, d = load_model(args)
    if args.scale:
        # a --scale cell samples its FINISHED checkpoint only: train_leg rewrites the final file every 2,000 steps
        # while it runs, so a chain that starts on the file's existence would otherwise rank a partly trained model
        assert d["steps"] == args.steps and d["batch"] == BATCH, \
            f"{model_path(args).name}.pt stamps {d['steps']} steps at batch {d['batch']}, not the registered " \
            f"{args.steps} at {BATCH}: still training, or not this cell's model"
    train_keys = set(stored_forms().keys())
    n, ident = len(group_for(GROUP_180)[1]), identity_index(GROUP_180)
    seen, out, attempts, nongen = set(), [], 0, 0
    want = args.over * args.n                # draw over x n distinct orbits, keep the top n by the policy's logp
    # THE RANKER: with --source uniform the candidates are UNIFORM draws instead of the policy's own, drawn exactly
    # as the uniform search arm draws them (codes.draw_gen: redraw until both sides generate G) and scored by the
    # SAME model's teacher-forced logp, so the two deployments differ only in where the candidates came from. The
    # stream is default_rng(10,000 + seed), the uniform candidate pool of the earlier supervised screen, so at
    # over 10 / n 1,000 it draws the very codes that screen ranked. draw_gen absorbs the non-generating
    # rejections itself, so a uniform cell's `attempts` counts candidates that already generate and its
    # `nongenerating` is 0 by construction.
    unirng = np.random.default_rng(10_000 + args.seed) if args.source == "uniform" else None
    t = time.time()
    while len(out) < want and attempts < 50 * want:
        if args.source == "uniform":
            # canonicalise BEFORE scoring: the policy reads trajectories in its own gauge (g1 = h1 = e),
            # which is where its own draws already live, so this is the same quantity on both arms
            raw = [list(canonicalise(GROUP_180, draw_gen(unirng, GROUP_180))) for _ in range(256)]
            drawn = list(zip(raw, logp_of_trajs(model, raw, lam, ident)))
        else:
            cond = torch.tensor([cond_vec(lam)] * 256, dtype=torch.float32)
            toks, lps = model.sample(256, cond, ident)
            drawn = [(list(map(int, tk)), float(lp)) for tk, lp in zip(toks, lps)]
        for exps, lp in drawn:
            attempts += 1
            exps = list(canonicalise(GROUP_180, exps))
            if not generates_g(GROUP_180, exps):
                nongen += 1                  # discarded without an oracle call
                continue
            k = orbit_key(GROUP_180, exps)
            if k in seen:
                continue                     # a repeated canonical form is discarded too
            seen.add(k)
            out.append({"exps": exps, "logp": lp, "novel": k not in train_keys})
            if len(out) >= want:
                break
    extra = {"source": "uniform"} if args.source == "uniform" else {}
    if args.over > 1:
        # THE RANKED DEPLOYMENT: the model is the ranker, of its own draws or, under --source uniform, of uniform
        # ones. The top n by stamped logp become the paid rows (the screen leg pays the bound ladders on every row,
        # the certify leg the exact ladders on the rows whose bounds reach the bar); the other (over - 1) x n draws
        # were free and are kept, unpaid, for disclosure only. Stable sort, so equal logp keeps sampling order and a
        # re-run reproduces the row list.
        order = sorted(range(len(out)), key=lambda i: -out[i]["logp"])
        kept = [out[i] for i in order[:args.n]]
        extra.update({"over": args.over, "ranked_by": "logp", "sampled": len(out),
                      "unpaid": [out[i] for i in order[args.n:]]})
        out = kept
    novel = sum(r["novel"] for r in out)
    json.dump({"experiment": "r177", "arch": "transformer", "sign": args.sign, "lam": lam, "n": len(out),
               "attempts": attempts, "nongenerating": nongen, "novel": novel, "model": model_path(args).name,
               "seed": args.seed, "wall": time.time() - t, "rows": out, **extra}, open(samples_path(args), "w"))
    print(f"transformer ({args.source} draws) at {args.sign}: {len(out)} distinct generating orbits in {attempts} "
          f"attempts ({nongen} non-generating), {novel} novel, {time.time() - t:.0f}s -> {samples_path(args).name}"
          + (f" (top {args.n} of {extra['sampled']} by logp)" if args.over > 1 else ""), flush=True)


# ----------------------------------------------------------------------------- screen
def screen_leg(args):
    lam = lam_of(10, args.sign)
    bar = key_top(lam)[0]
    s = json.load(open(samples_path(args)))
    n = len(group_for(GROUP_180)[1])
    path = OUT / samples_path(args).name.replace("r177_samples", "r177_screen")
    t = time.time()
    for i, r in enumerate(s["rows"]):
        A, B = lifted_bases(build(GROUP_180, np.array(r["exps"])), n)
        # the FULL ladder (every rung), the one the search runs paid, so a policy cell's screening seconds may be
        # compared with theirs; the odd-rung skip would give the same verdicts and different seconds
        dA, exA, _, cA = bound_ladder(A, skip_odd=False)
        dB, exB, _, cB = bound_ladder(B, skip_odd=False)
        proxy = bias_score(dA, dB, lam)
        r.update({"dA": dA, "dB": dB, "exactA": exA, "exactB": exB, "cpuBound": cA + cB, "proxy": proxy,
                  "top": bool(proxy >= bar), "censored": not (exA and exB)})
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(s['rows'])}  {time.time() - t:.0f}s", flush=True)
    tops = [r for r in s["rows"] if r["top"]]
    nov = [r for r in s["rows"] if r["novel"]]
    s.update({"leg": "screen", "ladder": "full", "bar": bar, "proxy_top": len(tops),
              "proxy_top_novel": sum(r["novel"] for r in tops), "n_novel": len(nov),
              "n_censored": sum(r["censored"] for r in s["rows"]),
              "cpu": sum(r["cpuBound"] for r in s["rows"]), "screen_wall": time.time() - t})
    json.dump(s, open(path, "w"))
    print(f"{s['arch']} at {args.sign}: proxy-top {len(tops)}/{len(s['rows'])} "
          f"({100 * len(tops) / len(s['rows']):.2f} %), novel-only {s['proxy_top_novel']}/{len(nov)}, "
          f"{time.time() - t:.0f}s -> {path.name}", flush=True)


# ----------------------------------------------------------------------------- certify
def certify_leg(args):
    """The exact two-sector ladders on every screened row whose bounds reach the bar (about 2 to 4 minutes each);
    writes the exact-top count and the cell's precision. Resumable: rows already certified are skipped."""
    lam = lam_of(10, args.sign)
    bar = key_top(lam)[0]
    path = OUT / samples_path(args).name.replace("r177_samples", "r177_screen")
    s = json.load(open(path))
    out = OUT / path.name.replace("r177_screen", "r177_certified")
    done = json.load(open(out)) if out.exists() else {"rows": {}}          # resume-safe
    t = time.time()
    for i, r in enumerate(s["rows"]):
        if not r["top"] or str(i) in done["rows"]:
            continue
        dX, exX, dZ, exZ, secs, cpu = exact_sectors(build(GROUP_180, np.array(r["exps"])))
        # "exps" stamps the identity of the row this entry certifies, so a reader can assert that the by-index
        # join to the screen file is still valid
        done["rows"][str(i)] = {"exps": r["exps"], "dX": dX, "dZ": dZ, "exactX": exX, "exactZ": exZ, "cpu": cpu,
                                "exact_top": bool(exX and exZ and bias_score(dX, dZ, lam) >= bar), "novel": r["novel"]}
        # written whole, then renamed, so a resume never meets half a file
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(done, f)
        tmp.replace(out)
        print(f"  {len(done['rows'])}/{s['proxy_top']} certified, {time.time() - t:.0f}s", flush=True)
    rows = list(done["rows"].values())
    n_ex = sum(r["exact_top"] for r in rows)
    done.update({"experiment": "r177", "leg": "certify", "arch": s["arch"], "sign": args.sign, "model": s["model"],
                 "n_samples": len(s["rows"]), "n_novel": s["n_novel"], "proxy_top": s["proxy_top"], "exact_top": n_ex,
                 "exact_top_novel": sum(r["exact_top"] and r["novel"] for r in rows),
                 "precision": n_ex / max(1, len(rows)), "cpu": sum(r["cpu"] for r in rows),
                 "complete": len(rows) == s["proxy_top"]})
    tmp = out.with_name(out.name + ".tmp")        # the final write too: whole, then renamed
    with open(tmp, "w") as f:
        json.dump(done, f)
    tmp.replace(out)
    print(f"{s['arch']} at {args.sign}: exact-top {n_ex}/{len(s['rows'])} ({1000 * n_ex / len(s['rows']):.1f} per 1,000), "
          f"precision {done['precision']:.2f} ({n_ex}/{len(rows)}), {done['cpu']:.0f} s CPU -> {out.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("leg", choices=["train", "sample", "screen", "certify"])
    ap.add_argument("--sign", default="Z", choices=["Z", "X"], help="the held-out bias: Z = gamma +1, X = gamma -1")
    ap.add_argument("--n", type=int, default=1000, help="sample: distinct generating orbits paid per cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="train: gradient steps (names the cell _st<steps>)")
    ap.add_argument("--scale", choices=sorted(SCALE_STEPS), default=None,
                    help="the training budget as a step count (x20 = 315,280, the policy of record); names the cell, "
                         "so sample / screen / certify need it too")
    ap.add_argument("--source", default="policy", choices=["policy", "uniform"],
                    help="sample: where the candidates come from. policy = the model's own draws; uniform = uniform "
                         "draws scored by the same model's logp (the ranker; names the cell _u<over>). Screen / "
                         "certify need it too")
    ap.add_argument("--model-path", default=None,
                    help="load THIS checkpoint (path; .pt optional) instead of the name model_path() builds, and name "
                         "the downstream files by its stem (a shipped model in data/models/, or the matched screen)")
    ap.add_argument("--over", type=int, default=1,
                    help="sample: draw over x n distinct orbits and keep the top n by the policy's own logp (names "
                         "the cell _o<over>). Screen / certify need it too")
    ap.add_argument("--objective", default="tb", choices=["tb", "softq"],
                    help="train: the loss. tb = trajectory balance; softq = offline soft Q-learning (names the cell "
                         "_softq). Sample / screen / certify need it too")
    ap.add_argument("--device", default="cpu",
                    help="train: torch device of the transformer (cpu or cuda); the data pipeline stays on the CPU")
    args = ap.parse_args()
    if args.scale:
        args.steps = SCALE_STEPS[args.scale]
    assert args.steps or args.model_path, "give --scale or --steps (they name the model), or --model-path"
    OUT.mkdir(exist_ok=True)
    {"train": train_leg, "sample": sample_leg, "screen": screen_leg, "certify": certify_leg}[args.leg](args)


if __name__ == "__main__":
    main()
