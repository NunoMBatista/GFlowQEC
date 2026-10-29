"""The two supervised screens at n = 180: the matched screen and the earlier, smaller screen.

A supervised screen is the plain alternative to a generative policy. It is a regressor trained on the 116,972
stored forms (every code the 42 stored runs paid for, with its stored distances), the same data GFlowQEC-offline
learns from. It never proposes codes: it ranks uniform draws from the arena, and only the top of its ranking is
paid. Both screens rank the same uniform candidates, the first over x 1,000 distinct generating orbits of
default_rng(10,000 + seed), so their rows compare with the policy's on equal information.

Regenerates the supervised-screen rows of the n = 180 summary table ("Supervised screen") and its per-seed table, and the results
sentences on them:

    matched screen   "Supervised screen, 10 times" and "Supervised screen, 100 times" (the latter pending in the
                     paper): GFlowQEC-offline's transformer turned into a regressor
                     (gflowqec.screens.ScreenTransformer). Legs, in order:
      select --config C         one configuration (T1plain, T2plain, T1tfr, T2tfr) trained at seed 0 on a fixed
                                90 % of the stored forms; its held-out precision at the top 1 % per bias every
                                15,764 steps -> results/screen180match_select_{C}_s0.json
      choose                    the configuration and step count by gflowqec.screens.choose_rule
                                -> results/screen180match_choice.json
      train --seed s            the choice retrained on all 116,972 stored forms
                                -> results/r177_mscreen_{C}_st{steps}_s{s}.pt (the paper's: T1tfr, 252,224 steps)
      rank --seed s --sign S --over mu
                                the mu x 1,000 uniform candidates ranked by the model at the deployment bias; the top
                                1,000 are written as the paid rows of a samples file, the rest kept unpaid in rank
                                order -> results/r177_samples_mscreen_{C}_st{steps}_e10{S}_n1000_u{mu}_s{s}.json.
                                scripts.n180_offline's screen and certify legs then pay it (the command is printed).
      check-cands --seed s --over mu
                                the identity check of the candidates alone (rank runs it first)
    earlier, smaller screen
                     "Earlier, smaller screen, 10 times", and at 100 times (seeds 1 and 2) the ratios
                     1.32 and 1.57 (not printed in the paper): a two-hidden-layer perceptron on the bag encoding, the
                     relational counts and the bias. One leg:
      mlp --seed s --sign S --over mu
                                trains it (gflowqec.screens.train_mlp_screen), ranks the candidates, pays the top
                                1,000 on the certifying oracle
                                -> results/r174_proxy_pool116972_MR_e10{S}_n1000_o{mu}_h64_st315280_s{s}.json

Commands (run from the repo root):
    python -m scripts.n180_screens select --config T1plain --device cuda       (and T2plain, T1tfr, T2tfr)
    python -m scripts.n180_screens choose
    python -m scripts.n180_screens train --seed 0 --device cuda                (seeds 0, 1, 2)
    python -m scripts.n180_screens rank --seed 0 --sign Z --over 10            (signs Z, X; over 10 at seeds 0-2,
                                                                                over 100 at seeds 1-2)
    python -m scripts.n180_offline screen --model-path results/r177_mscreen_T1tfr_st252224_s0.pt --source uniform
        --over 10 --sign Z --n 1000 --seed 0                                   (then certify, same flags)
    python -m scripts.n180_screens mlp --seed 0 --sign Z --over 10             (signs Z, X; over 10 at seeds 0-2,
                                                                                over 100 at seeds 1-2)

Every leg writes to results/, and each later leg reads the earlier legs' files from results/. To rank with the
shipped model instead of retraining it, copy data/n180/screen180match_choice.json and
data/models/r177_mscreen_T1tfr_st252224_s{s}.pt into results/ first.

Compute (the runs behind the paper): selection 2,792 to 2,952 s of GPU wall per configuration and final training 2,366 to
2,443 s of GPU wall per seed, on one GPU; rank takes a few minutes on one CPU, and its paid legs belong to
scripts.n180_offline. Earlier, smaller screen: training 20.4 CPU-h over the six over-10 cells and 13.4 CPU-h over
the four over-100 cells, paying 21.2 and 46.7 CPU-h on the certifying oracle.
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from gflowqec.codes import GROUP_180, RelTables, conj_table, group_for, orbit_key
from gflowqec.counts import stored_forms, value
from gflowqec.oracle import cond_vec, key_beta, key_top, lam_of
from gflowqec.paths import OUT
from gflowqec.policy import LAMS
from gflowqec.screens import (BATCH, CONFIGS, EVAL_EVERY, MAX_STEPS, TIE, ScreenTransformer, build_net,
                              candidate_check, choose_rule, draw_candidates, encode, fit, mscreen_stem, n_params,
                              precision_at_top, predict, propose, rank_rows, rows_of, score_rows, split_rows,
                              train_mlp_screen, zstats)

# The matched screen ran one process per GPU job with OMP_NUM_THREADS=1; on a CPU the default is 4 threads
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

N = 1000                                # paid calls per deployment cell
ETA = 10.0                              # the deployment bias: eta = 10 at either sign (lam = +1 or -1)
CHOICE = OUT / "screen180match_choice.json"

# The earlier, smaller screen's recipe: hidden width 64, batch 64, and the policy's 315,280 updates
MLP_HIDDEN = 64
MLP_BATCH = 64
MLP_STEPS = MAX_STEPS


def setup():
    """The stored forms, their target statistics and the tables every training leg shares."""
    pool = stored_forms()
    n = len(group_for(GROUP_180)[1])
    conj, T = conj_table(GROUP_180), RelTables(GROUP_180)
    # T1 targets are z-scored with the mean and sd over ALL 116,972 forms at each bias (gamma = +1, -1)
    stats = zstats(list(pool.values()), LAMS)
    # T2 targets use the policy's reward exponent beta at each bias; the bar is the top-scoring threshold (10.0)
    beta = {lam: key_beta(lam) for lam in LAMS}
    bar = {lam: key_top(lam)[0] for lam in LAMS}
    print(f"{len(pool):,} stored forms; T1 mean / sd {', '.join(f'{lam:+.0f}: {m:.4f} / {s:.4f}' for lam, (m, s) in stats.items())}; "
          f"T2 beta {beta}; bar {bar}", flush=True)
    return pool, n, conj, T, stats, beta, bar


def write(path, obj):
    json.dump(obj, open(path, "w"), indent=1)


# ----------------------------------------------------------------------------- select
def select_leg(args):
    """One configuration trained at seed 0 on the 90 % training rows; its held-out precision at the top 1 per cent
    per bias every EVAL_EVERY steps. The curve is rewritten after every point (complete False until the last)."""
    cfg, dev = args.config, torch.device(args.device)
    pool, n, conj, T, stats, beta, bar = setup()
    train, held, md5 = split_rows(pool)
    held_trajs = [list(r["traj"]) for r in held]
    # a held-out form is a "top" at a bias when its stored score reaches the bar
    labels = {lam: [value(r, lam) >= bar[lam] for r in held] for lam in LAMS}
    print(f"split: {len(train):,} training / {len(held):,} held-out forms (md5 {md5}); held-out tops "
          f"{', '.join(f'{lam:+.0f}: {sum(v)}' for lam, v in labels.items())}; precision at the top "
          f"{(len(held) + 99) // 100} per bias", flush=True)
    # every configuration starts from the same seed 0 streams
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    net = build_net(cfg, n, T).to(dev)
    path = OUT / f"screen180match_select_{cfg}_s0.json"
    rec = {"experiment": "screen180match", "leg": "select", "config": cfg, "seed": 0, "params": n_params(net),
           "device": args.device, "batch": BATCH, "steps": MAX_STEPS, "eval_every": EVAL_EVERY, "split_md5": md5,
           "n_train": len(train), "n_held": len(held), "k_top": (len(held) + 99) // 100,
           "held_tops": {str(lam): int(sum(v)) for lam, v in labels.items()}, "curve": [], "complete": False}
    t, c0 = time.time(), time.process_time()

    def on_eval(step, mse):
        prec = {lam: precision_at_top(predict(net, held_trajs, lam, cfg, T, dev), labels[lam]) for lam in LAMS}
        rec["curve"].append({"step": step, "Z": prec[1.0], "X": prec[-1.0], "mean": (prec[1.0] + prec[-1.0]) / 2,
                             "mse": mse})
        rec.update({"wall": time.time() - t, "cpu": time.process_time() - c0})
        write(path, rec)
        print(f"  EVAL step {step}: held-out precision Z {prec[1.0]:.4f} X {prec[-1.0]:.4f} mean "
              f"{(prec[1.0] + prec[-1.0]) / 2:.4f}", flush=True)

    rec["loss_trace"] = fit(net, train, cfg, MAX_STEPS, rng, dev, stats, beta, conj, T, on_eval, EVAL_EVERY)
    rec.update({"wall": time.time() - t, "cpu": time.process_time() - c0, "complete": True})
    write(path, rec)
    # the curve has MAX_STEPS / EVAL_EVERY = 20 points, so it is never empty
    best = max(rec["curve"], key=lambda p: p["mean"])
    print(f"done: {cfg} -> {path.name}; best mean {best['mean']:.4f} at step {best['step']}", flush=True)


# ----------------------------------------------------------------------------- choose
def choose_leg(args):
    """Reads the four complete selection curves and writes the choice (configuration and step count)."""
    curves, md5s, splits = {}, {}, set()
    for cfg in CONFIGS:
        path = OUT / f"screen180match_select_{cfg}_s0.json"
        d = json.load(open(path))
        assert d["complete"], f"{path.name} is not complete: the choice waits for all four selection runs"
        curves[cfg] = d["curve"]
        md5s[cfg] = hashlib.md5(path.read_bytes()).hexdigest()
        splits.add(d["split_md5"])
        best = max(d["curve"], key=lambda p: p["mean"])
        print(f"{cfg:8s} {d['params']:,} parameters: best mean {best['mean']:.4f} (Z {best['Z']:.4f}, X {best['X']:.4f}) "
              f"at step {best['step']:,}; last {d['curve'][-1]['mean']:.4f} at {d['curve'][-1]['step']:,}")
    assert len(splits) == 1, f"the selection runs used different splits: {splits}"
    (cfg, step, score), best, ties = choose_rule(curves)
    out = {"config": cfg, "step": step, "score": score, "best": best,
           "ties": [{"config": c, "step": s, "mean": m} for c, s, m in ties], "select_md5": md5s,
           "split_md5": splits.pop(),
           "rule": "highest two-bias mean of held-out precision at the top 1 per cent; within 0.01 a tie, broken by "
                   "plain over TF+R, then T1 over T2, then the later step"}
    write(CHOICE, out)
    print(f"CHOICE: {cfg} at step {step:,} (mean {score:.4f}; best {best:.4f}; {len(ties)} point(s) within {TIE}) "
          f"-> {CHOICE.name}")


# ----------------------------------------------------------------------------- train
def train_leg(args):
    """The choice retrained on ALL 116,972 stored forms for the chosen step count, seed s (the policy's stream at
    seed s, draw for draw)."""
    choice = json.load(open(CHOICE))
    cfg, steps = choice["config"], choice["step"]
    assert steps <= MAX_STEPS, f"{steps} steps: the choice may use at most the policy's {MAX_STEPS:,}"
    dev = torch.device(args.device)
    pool, n, conj, T, stats, beta, _ = setup()
    rows = list(pool.values())
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    net = build_net(cfg, n, T).to(dev)
    t = time.time()
    trace = fit(net, rows, cfg, steps, rng, dev, stats, beta, conj, T)
    path = OUT / (mscreen_stem(cfg, steps, args.seed) + ".pt")
    # CPU tensors, so the rank leg (and any reader) opens a GPU-trained model on a CPU-only machine
    torch.save({"arch": "screen-transformer", "config": cfg, "target": cfg[:2], "params": n_params(net),
                "state": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "steps": steps, "batch": BATCH,
                "seed": args.seed, "rows": len(rows), "device": args.device, "wallTrain": time.time() - t,
                "cpuTrain": time.process_time(), "t1_stats": {str(k): v for k, v in stats.items()},
                "choice_md5": hashlib.md5(open(CHOICE, "rb").read()).hexdigest(), "loss_trace": trace}, path)
    print(f"done: {cfg} seed {args.seed}, {steps:,} steps, {time.time() - t:.0f}s -> {path.name}", flush=True)


# ----------------------------------------------------------------------------- check-cands
def check_cands_leg(args):
    """Draws the candidates and runs the identity check alone (they must be the policy's u10 ranking cell's
    candidates, and hold every pick of the earlier, smaller screen)."""
    cands, attempts = draw_candidates(args.seed, args.over, N)
    print(f"{len(cands):,} candidates in {attempts:,} attempts", flush=True)
    candidate_check(cands, args.seed, args.over, N)
    print("check-cands PASS", flush=True)


# ----------------------------------------------------------------------------- rank (the deployment)
def rank_leg(args):
    """The second deployment rule with the model as the ranker: the mu x N uniform candidates, ranked by the
    prediction at the deployment bias (stable descending, ties keep draw order), the top N written as the paid rows
    of a samples file (scripts.n180_offline's screen and certify legs pay them by row index), the other (mu - 1) N
    kept, unpaid, in rank order. The candidate check runs first and aborts before any prediction or file."""
    choice = json.load(open(CHOICE))
    cfg, steps = choice["config"], choice["step"]
    model = OUT / (mscreen_stem(cfg, steps, args.seed) + ".pt")
    # the samples file takes the model's stem, so every downstream file of the cell names the model
    # (r177_mscreen_... -> r177_samples_mscreen_..., the seed moved to the end; "_u" marks uniform candidates)
    name = model.stem.replace("r177_", "r177_samples_").replace(f"_s{args.seed}", "")
    path = OUT / f"{name}_e10{args.sign}_n{N}_u{args.over}_s{args.seed}.json"
    d = torch.load(model, weights_only=False, map_location="cpu")
    assert (d["config"], d["steps"], d["seed"]) == (cfg, steps, args.seed), f"{model.name} stamps {d['config']}, " \
        f"{d['steps']} steps, seed {d['seed']}"
    n_el, T = len(group_for(GROUP_180)[1]), RelTables(GROUP_180)
    net = ScreenTransformer(n_el, rel_dim=T.dim if cfg.endswith("tfr") else 0)
    net.load_state_dict(d["state"])
    lam = lam_of(ETA, args.sign)
    # the deployment's charged time: drawing the candidates and ranking them (the identity check is not charged)
    t, c0 = time.time(), time.process_time()
    cands, attempts = draw_candidates(args.seed, args.over, N)
    draw_wall, draw_cpu = time.time() - t, time.process_time() - c0
    md5 = candidate_check(cands, args.seed, args.over, N)         # aborts on any mismatch, before any file
    t, c0 = time.time(), time.process_time()
    pred = predict(net, cands, lam, cfg, T)
    rank_wall, rank_cpu = draw_wall + time.time() - t, draw_cpu + time.process_time() - c0
    # a candidate is novel when it is not one of the stored forms the screen was trained on
    pool_keys = set(stored_forms().keys())
    rows, unpaid = rank_rows(cands, pred, [orbit_key(GROUP_180, c) not in pool_keys for c in cands], N)
    novel = sum(r["novel"] for r in rows)
    # nongenerating is 0 by construction: the uniform draw redraws until both sides generate the group
    write_rec = {"experiment": "r177", "arch": "screen-transformer", "config": cfg, "sign": args.sign, "lam": lam,
                 "n": len(rows), "attempts": attempts, "nongenerating": 0, "novel": novel, "model": model.stem,
                 "seed": args.seed, "wall": rank_wall, "rows": rows, "source": "uniform", "over": args.over,
                 "ranked_by": "pred", "sampled": len(rows) + len(unpaid), "unpaid": unpaid, "rank_wall": rank_wall,
                 "rank_cpu": rank_cpu, "cand_md5": md5}
    json.dump(write_rec, open(path, "w"))
    out = rows + unpaid
    print(f"{cfg} at {args.sign}: top {len(rows):,} of {len(out):,} uniform candidates by prediction, {novel} novel, "
          f"{len(rows) - novel} stored forms among the paid rows ({sum(not r['novel'] for r in out)} among all "
          f"candidates); rank {rank_wall:.0f}s wall -> {path.name}", flush=True)
    print(f"paid legs: python -m scripts.n180_offline screen --model-path results/{model.name} --source uniform "
          f"--over {args.over} --sign {args.sign} --n {N} --seed {args.seed}   (then certify, same flags)", flush=True)


# ----------------------------------------------------------------------------- mlp (the earlier, smaller screen)
def mlp_leg(args):
    """The earlier, smaller screen: the perceptron trained on all stored forms for the policy's 315,280 updates,
    then the mu x N uniform candidates ranked at the deployment bias and the top N paid on the certifying oracle
    (its bound ladders use the parity rule, the oracle's default)."""
    lam = lam_of(ETA, args.sign)
    n = len(group_for(GROUP_180)[1])
    T = RelTables(GROUP_180)
    pool = stored_forms()
    path = OUT / (f"r174_proxy_pool{len(pool)}_MR_e{ETA:g}{args.sign}_n{N}_o{args.over}"
                  f"_h{MLP_HIDDEN}_st{MLP_STEPS}_s{args.seed}.json")
    # restart-safe: a finished cell is never paid twice
    if path.exists() and json.load(open(path)).get("complete"):
        print(f"already written, skipping: {path.name}", flush=True)
        return
    t = time.time()
    net, cpu_train = train_mlp_screen(pool.values(), MLP_STEPS, MLP_BATCH, MLP_HIDDEN, args.seed)
    with torch.no_grad():
        # the same uniform candidates the matched screen ranks: default_rng(10,000 + seed), first mu x N distinct
        upool, _ = propose(args.over * N, np.random.default_rng(10_000 + args.seed))
        feats = np.stack([np.concatenate([encode(c, n, T), np.float32(cond_vec(lam))]) for c in upool])
        pred = net(torch.tensor(feats)).squeeze(1).numpy()
    screen = [upool[i] for i in np.argsort(-pred)[:N]]
    oracle = score_rows(screen, lam, len(screen), "cert")
    json.dump({"experiment": "r174", "leg": "proxy", "complete": True, "dump": "pool", "seed": args.seed, "eta": ETA,
               "sign": args.sign, "lam": lam, "n": N, "over": args.over, "hidden": MLP_HIDDEN,
               "relational": True, "oracle": "cert", "certified": oracle.certified,
               "train_rows": len(pool), "train_source": "pool", "cpuTrain": cpu_train,
               "steps": MLP_STEPS, "batch": MLP_BATCH, "calls": oracle.calls, "wall": time.time() - t,
               "screen": rows_of(oracle, screen)}, open(path, "w"))
    print(f"  wrote {path.name} ({oracle.calls} calls, {time.time() - t:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("leg", choices=["select", "choose", "train", "rank", "check-cands", "mlp"])
    ap.add_argument("--config", choices=CONFIGS, help="select: the configuration")
    ap.add_argument("--device", default="cpu", help="select / train: torch device (cuda for the runs behind the paper)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sign", default="Z", choices=["Z", "X"])
    ap.add_argument("--over", type=int, default=10, choices=[10, 100],
                    help="rank / check-cands / mlp: uniform candidates drawn per paid call")
    args = ap.parse_args()
    if args.leg == "select":
        assert args.config, "select needs --config"
    OUT.mkdir(exist_ok=True)
    {"select": select_leg, "choose": choose_leg, "train": train_leg, "rank": rank_leg,
     "check-cands": check_cands_leg, "mlp": mlp_leg}[args.leg](args)


if __name__ == "__main__":
    main()
