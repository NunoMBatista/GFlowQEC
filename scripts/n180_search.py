"""The n = 180 searches: one method, one seed, one noise condition, a fixed number of paid oracle calls.

Regenerates the run files data/n180/r174_*, r200_* and r209_*.json behind every n = 180 search number of the
paper: the n = 180 summary table and its per-seed table, the method choices of the methods table (the better of
two PT ranges), the per-call costs of the oracle-cost table, and the 42 stored runs that GFlowQEC-offline trains on.
The readouts read the files; nothing here prints a paper number.

The arena. A code is ten group elements of G = C6 x S3 (order 36): A = [g1+g2+g3, e+g4+g5] and
B = [h1+h2+h3, e+h4+h5] define the lifted product, n = 5|G| = 180 (gflowqec.codes.build). Only tuples whose
elements generate all of G are legal. The oracle caches by the code's gauge orbit, so the budget counts distinct
orbits: a revisit or a gauge copy of a paid code is free.

The score (the oracle section of the paper) is the leading-order failure exponent under biased noise, min(c_X d_X, c_Z d_Z), at the
condition set by --eta (p_noisy / p_quiet; 10 in every printed comparison) and --sign (the noisy sector:
Z is gamma = +1, X is gamma = -1). Two oracles (--oracle):
    cert    the certifying oracle: two classical bound ladders per paid code (d_A >= d_X, d_B >= d_Z), and the
            exact sector ladders only when the bound score reaches the reference sample's top score, so the top
            set is exact. Every printed cell paid the FULL bound ladder (no parity skip), pinned below.
    exact   exact sector distances for every paid code (the default; the reference cells of the oracle-cost table).

The methods (--arm):
    random      uniform legal draws
    pt          parallel tempering, 8 chains on a geometric ladder [--tmin, --tmax]; --warm-pool starts the
                chains from the stored runs' certified tops (tuned PT started from the stored forms)
    gflowqec    GFlowQEC-online at depth 7 (gflowqec.policy). Written "lsgfn" in the file names and in the
                run's "arm" field, the name the stored runs carry
    mdqn, sdqn  Munchausen DQN (alpha 0.15) and soft Q-learning (alpha 0): the online maximum-entropy learners
    ppo         PPO, entropy weight --ppo-ent
    me          MAP-Elites on the descriptor grid --me-grid (g5: d_A - d_B bins; g20: k bins x d_A - d_B bins)
    mdqnls      Munchausen DQN inside GFlowQEC-online's local search (the same local move, a different learner)
    bo          Bayesian optimisation, a one-hot Matern-5/2 GP with expected improvement, cold start

--prefix only names the file, so a regenerated run lands under the name its shipped file carries: r174 for the
stored runs and the exact-oracle block, r200 for the later runs kept out of the stored pool (tuned PT started
from the stored forms, uniform at gamma = -1, BO), r209 for the remaining rivals.

Commands, each for seeds s = 0, 1, 2 and both --sign Z and --sign X unless stated (--eta 10 and
--budget 3000 are the defaults):
  stored runs (the pool)
    tuned PT                   python -m scripts.n180_search --arm pt --tmin 0.25 --tmax 2.0 --oracle cert --sign S --seed s
    PT at the other range      python -m scripts.n180_search --arm pt --tmin 0.5 --tmax 4.0 --oracle cert --sign S --seed s
    GFlowQEC-online, depth 7   python -m scripts.n180_search --arm gflowqec --oracle cert --sign S --seed s
    Munchausen DQN             python -m scripts.n180_search --arm mdqn --oracle cert --sign S --seed s
    uniform, gamma = +1        python -m scripts.n180_search --arm random --sign Z --oracle cert --seed s
  later runs
    uniform, gamma = -1        python -m scripts.n180_search --arm random --sign X --oracle cert --prefix r200 --seed s
    tuned PT started from the stored forms
                               python -m scripts.n180_search --arm pt --tmin 0.25 --tmax 2.0 --warm-pool --budget 1000 --oracle cert --prefix r200 --sign S --seed s
    one-hot BO                 python -m scripts.n180_search --arm bo --budget 1000 --oracle cert --prefix r200 --sign S --seed s
    MAP-Elites                 python -m scripts.n180_search --arm me --me-grid g5 --oracle cert --prefix r209 --sign S --seed s   (and --me-grid g20)
    PPO                        python -m scripts.n180_search --arm ppo --ppo-ent 0.1 --oracle cert --prefix r209 --sign S --seed s   (and --ppo-ent 0.01)
    soft Q-learning            python -m scripts.n180_search --arm sdqn --oracle cert --prefix r209 --sign S --seed s
    MDQN in the local search   python -m scripts.n180_search --arm mdqnls --oracle cert --prefix r209 --sign S --seed s
  exact-oracle block
    both PT ranges             python -m scripts.n180_search --arm pt --tmin 0.25 --tmax 2.0 --sign S --seed s   (and --tmin 0.5 --tmax 4.0)
    uniform, unbiased          python -m scripts.n180_search --arm random --eta 1 --sign Z --seed s
The other 15 stored runs (a conditional sampler with and without relational features, a relational variant of
GFlowQEC-online, and GFlowQEC-online at eta 1) are shipped as data only.

Compute, CPU-hours per cell (all its seeds and signs), network training included: tuned PT 113.6, PT 0.5-4.0
54.1, GFlowQEC-online 25.9, Munchausen DQN 16.9, uniform at +1 12.0 (the 42 stored runs: 320.4); tuned PT
started from the stored forms 49.2; uniform at -1 10.2; BO 16.6; MAP-Elites g5 87.9, g20 91.2; PPO 0.1 23.0, 0.01
20.3; soft Q-learning 18.6; MDQN in the local search 31.0; exact block: PT 526.9, uniform at eta 1 56.6. So
between about 3 CPU-hours (one BO run) and 44 (one exact-oracle PT run) per run.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from gflowqec.codes import GROUP_180
from gflowqec.oracle import HELD_OUT, T0, TRAIN_LAM, dump_rows, key_beta, key_top, lam_of, make_oracle
from gflowqec.paths import OUT
from gflowqec.policy import arm_gflowqec_online
from gflowqec.rivals import (ME_ASYM_EDGES_N180, ME_K_BIN_N180, arm_bo, arm_maxent, arm_mdqnls, arm_me, arm_ppo,
                             arm_pt, arm_random, me_cell_of, warm_pool_states)

# torch's thread count changes the order of floating-point sums, so a learner's run reproduces only at a fixed
# count: the runs behind the paper set OMP_NUM_THREADS and torch follows it
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

# the uniform calibration draws every learner pays first (they are also MAP-Elites' and BO's initial draws)
CALIB = 200


def dump_path(args, arm):
    """The run file: <prefix>_<arm>_C6xS3_e<eta><sign>_b<budget><cell>_s<seed>.json, where <cell> holds the
    knobs that distinguish one configuration of a method from another. It is the name of the shipped file of the
    same run under data/n180/."""
    cell = {"pt": f"_t{args.tmin}-{args.tmax}",
            # k1 = one slot masked per local move, i7 = depth 7, det = keep a child on a strict improvement only
            "lsgfn": "_k1i7det", "mdqnls": "_k1i7det",
            "ppo": f"_ent{args.ppo_ent:g}", "me": f"_{args.me_grid}", "bo": "_onehot"}.get(arm, "")
    if args.warm_pool:
        cell += "_Wpool"
    if args.oracle == "cert":
        cell += "_cert"
    return OUT / f"{args.prefix}_{arm}_{GROUP_180}_e{args.eta:g}{args.sign}_b{args.budget}{cell}_s{args.seed}.json"


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="random",
                    choices=["random", "pt", "gflowqec", "mdqn", "sdqn", "ppo", "me", "mdqnls", "bo"])
    ap.add_argument("--oracle", default="exact", choices=["exact", "cert"],
                    help="exact sector ladders for every code, or the certifying oracle (_cert in the file name)")
    ap.add_argument("--eta", type=float, default=10.0, help="p_noisy / p_quiet")
    ap.add_argument("--sign", default="Z", choices=["Z", "X"], help="which sector is the noisy one")
    ap.add_argument("--budget", type=int, default=3000, help="distinct orbits paid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmin", type=float, default=0.25, help="pt: ladder bottom, in score units")
    ap.add_argument("--tmax", type=float, default=2.0, help="pt: ladder top")
    ap.add_argument("--warm-pool", action="store_true",
                    help="pt: start the 8 chains from the stored runs' certified tops at this condition")
    ap.add_argument("--ppo-ent", type=float, default=0.01, help="ppo: the entropy coefficient")
    ap.add_argument("--me-grid", default=None, choices=["g20", "g5"], help="me: the descriptor grid")
    ap.add_argument("--prefix", default="r174", choices=["r174", "r200", "r209"], help="names the file only")
    return ap


def main():
    args = build_parser().parse_args()
    arm = "lsgfn" if args.arm == "gflowqec" else args.arm       # the name the run files carry
    assert not args.warm_pool or arm == "pt", "--warm-pool is a PT option"
    assert (args.me_grid is not None) == (arm == "me"), "--me-grid is required by, and only by, --arm me"
    OUT.mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cert = args.oracle == "cert"
    lam = lam_of(args.eta, args.sign)
    oracle = make_oracle(args.oracle, GROUP_180, args.budget, lam)
    path = dump_path(args, arm)
    if cert:
        oracle.skip_odd = False      # the full classical bound ladder, which every printed cell paid
    # a finished run is never re-run; a checkpoint file (no "complete" flag) is overwritten by a full re-run
    if path.exists() and json.load(open(path)).get("complete"):
        print(f"already complete, skipping: {path.name}", flush=True)
        return
    # tuned PT started from the stored forms: the chain starts are drawn from THIS run's rng before the arm
    # consumes it, so the run is reproducible from (seed, stored runs); they are paid like any other orbit, so
    # the first 8 rows are the starts
    warm, warm_info = (warm_pool_states(lam, args.sign, rng) if args.warm_pool else (None, None))
    # the reward temperature of every learner: one beta per condition, from the reference sample (oracle.key_beta)
    beta = key_beta(lam)

    # the run's header. Fields fixed by this release (no conditional sampler, no relational features, the
    # depth-7 strict-improvement schedule, 200 calibration draws, no tag) keep the values the shipped files carry,
    # so a regenerated file has the fields of a shipped one (the r174 files lack five of them: ladder, ls_iters,
    # accept, warm_pool, warm_paid; a missing ladder field means the full ladder)
    header = {"experiment": args.prefix, "arm": arm, "group": GROUP_180, "eta": args.eta,
              "ladder": "full" if cert else None,
              "sign": args.sign, "lam": lam, "train_lam": None,
              "budget": args.budget, "seed": args.seed, "tmin": args.tmin, "tmax": args.tmax,
              "mirror": None,
              "relational": False if arm == "lsgfn" else None,
              "calib": CALIB, "beta": beta,
              "ls_iters": 7 if arm == "lsgfn" else None,
              "accept": "strict" if arm == "lsgfn" else None,
              "oracle": args.oracle,
              # the certification bar (the reference sample's top score) at every condition the runs are read at
              "bar": {f"{l:+.2f}": key_top(l)[0] for l in TRAIN_LAM + HELD_OUT} if cert else None,
              "warm_pool": warm_info, "warm_paid": len(warm) if warm else None,
              "tag": ""}
    if args.prefix == "r209":
        # the r209 runs also record each rival's configuration in the header
        me_spec = {"g20": f"(k bin {ME_K_BIN_N180} else 3, searchsorted({ME_ASYM_EDGES_N180}, dA - dB))",
                   "g5": f"searchsorted({ME_ASYM_EDGES_N180}, dA - dB)"}
        header.update({"ppo_ent": args.ppo_ent if arm == "ppo" else None,
                       "me_grid": args.me_grid, "me_spec": me_spec.get(args.me_grid),
                       "alpha_m": {"sdqn": 0.0, "mdqnls": 0.15}.get(arm),
                       "harness": "k1i7det" if arm == "mdqnls" else None})
    # a partial file every 500 paid calls
    oracle.checkpoint = lambda: dump_rows(oracle, header, path)
    print(f"{arm} on the {args.oracle} oracle at lam {lam:+.2f} (e{args.eta:g}{args.sign}), budget {args.budget}, "
          f"seed {args.seed}, key beta {beta} -> {path.name}", flush=True)

    save = str(path).replace(".json", ".pt")        # a learner's final network, beside the run file
    if arm == "random":
        arm_random(oracle, GROUP_180, rng)
    elif arm == "pt":
        # every shipped PT run tags all its rows src 0 and arm_pt sets no tag, so a regenerated file matches the
        # shipped one column for column; the draws are the same
        arm_pt(oracle, GROUP_180, rng, args.tmin, args.tmax, warm=warm)
    elif arm == "lsgfn":
        arm_gflowqec_online(oracle, GROUP_180, rng, args.seed, calib=CALIB, beta_mass=0.8, beta=beta, iters=7,
                            save_path=save)
    elif arm == "mdqn":
        arm_maxent(oracle, GROUP_180, rng, args.seed, alpha_m=0.15, calib=CALIB, beta_mass=0.8, beta=beta,
                   save_path=save)
    elif arm == "sdqn":
        arm_maxent(oracle, GROUP_180, rng, args.seed, alpha_m=0.0, calib=CALIB, beta_mass=0.8, beta=beta,
                   save_path=save)
    elif arm == "ppo":
        arm_ppo(oracle, GROUP_180, rng, args.seed, calib=CALIB, beta_mass=0.8, save_path=save,
                ent_coef=args.ppo_ent, beta=beta)
    elif arm == "me":
        arm_me(oracle, GROUP_180, rng, init=CALIB, cell_of=me_cell_of(oracle, args.me_grid))
    elif arm == "mdqnls":
        arm_mdqnls(oracle, GROUP_180, rng, args.seed, alpha_m=0.15, calib=CALIB, beta=beta, save_path=save)
    elif arm == "bo":
        # cold: CALIB uniform initial draws, then the exact GP on every paid row, refit every step, expected
        # improvement over a fresh slate of 2,000 unpaid uniform candidates
        arm_bo(oracle, GROUP_180, rng, init=CALIB, slate=2000, xi=0.01, window=0, elite=0, refit=1)

    # the closing pass on the certifying oracle: a run that does not depend on the sign (uniform draws, or the
    # unbiased condition) is read at both signs, so every paid code that could be a top at eta 10 is certified
    # there; a biased run is read only at its own sign, where every paid code was already valued when paid
    if cert and (arm == "random" or lam == 0.0):
        oracle.rescore()
    dump_rows(oracle, {**header, "complete": True}, path)
    best = max(oracle.value(m["rep"], lam) for m in oracle.meta.values())
    print(f"done: {oracle.calls} calls, {oracle.hits} hits, {oracle.censored} censored, "
          f"certified {getattr(oracle, 'certified', 'n/a')}, best score {best:.2f}, "
          f"{time.time() - T0:.0f}s -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
