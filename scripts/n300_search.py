"""The n = 300 searches: one method, one seed, 40,000 paid oracle calls, every paid code kept in call order.

Regenerates the run files data/n300/mitten_lersearch_*.json behind every n = 300 number of the paper: the
n = 300 summary table and figure, the table of further n = 300 runs and their text (the readouts
sum300.py, fig_main300.py, text300.py, res300.py, more300.py). Those readouts never use a run's in-run scores
as the result: scripts/n300_rescore.py re-estimates 800 of each run's visits on fresh noise first.

The arena. A code is ten group elements g1..g5, h1..h5 of G = Z_10 x S_3 (order 60); A = [g1+g2+g3, e+g4+g5]
and B = [h1+h2+h3, e+h4+h5] define the lifted product LP(A, B), n = 5|G| = 300 (gflowqec.codes.build). Only
tuples whose elements generate all of G are legal. A paid call estimates the code's per-logical-qubit logical
error rate ler_pq with BP-OSD at 1,000 shots of depolarizing code-capacity noise, p = 0.07, and returns the
score s = -log(ler_pq + 1/(k * shots)) (gflowqec.oracle.LEROracle). The oracle caches by the code's gauge
orbit, so the budget counts distinct orbits: a revisit or a gauge copy of a paid code is free.

Noise streams. By default every orbit is scored on its own noise stream, a pure function of (seed, orbit):
two methods at the same seed score a common code identically. With --crn every orbit of the run is scored on
ONE stream seeded by the run alone (common random numbers across codes). The two are different oracles, so a
--crn run is its own cell and its file name carries the letter C.

The methods (--arm):
    gflowqec    GFlowQEC-online: the online GFlowNet inside a local search (gflowqec.policy). Its depth L is
                --ls-iters (31 or 127). Written "lsgfn" in the file names and in the run's "arm" field, the
                name the stored runs carry. Reference and ablation cells of the paper:
                --ls-uniform (no GFlowNet: uniform draws and uniform local moves), --ls-uniform-local
                (the GFlowNet's draws, uniform local moves), --ls-paired (a child is scored on its incumbent's
                noise stream, an accepted child re-scored on its own), --ls-objective reg (the objective swap:
                per-step regression onto log R instead of trajectory balance).
    pt          parallel tempering, --pt-chains chains on a geometric ladder [--tmin, --tmax].
    gfn         the plain sampler: the online GFlowNet without local search.
    sdqn, mdqn  soft Q-learning and Munchausen DQN (alpha 0.15): the online maximum-entropy learners.
    ppo         PPO, entropy weight --ppo-ent.
    me          MAP-Elites on the k x sector-asymmetry grid.
    random      uniform legal draws.
    bo          Bayesian optimisation, a one-hot Matern-5/2 GP with expected improvement.
    selfscreen  a supervised screen trained on the run's own paid calls (gflowqec.screens.arm_selfscreen).

Commands, each for seeds s = 0-5 and 7-10 (seed 6 was the tuning seed).
--budget 40000, --p 0.07 and --shots 1000 are the defaults, and
every command starts with `python -m scripts.n300_search`:
    GFlowQEC-online at L = 127, one stream per code, under --crn, and the local search alone:
        --arm gflowqec --ls-iters 127 --tag _d127 --seed s
        --arm gflowqec --ls-iters 127 --crn --tag _c127 --seed s
        --arm gflowqec --ls-iters 127 --ls-uniform --tag _u127 --seed s
    the five reference cells whose pooled median fixes the bar y*: four at L = 31, and tuned PT:
        --arm gflowqec --ls-iters 31 --tag _d31 --seed s
        --arm gflowqec --ls-iters 31 --ls-uniform --tag _u31 --seed s
        --arm gflowqec --ls-iters 31 --ls-uniform-local --tag _g31 --seed s
        --arm gflowqec --ls-iters 31 --ls-paired --tag _f31 --seed s
        --arm pt --tmin 0.01 --tmax 0.15 --beta-mass 0.35 --seed s
    the objective swap at L = 31 and at L = 127 (the latter under --crn):
        --arm gflowqec --ls-iters 31 --ls-batch 4 --ls-objective reg --tag _p31 --seed s
        --arm gflowqec --ls-iters 127 --crn --ls-objective reg --tag _p127 --seed s
    tuned PT under --crn:
        --arm pt --tmin 0.01 --tmax 0.15 --beta-mass 0.35 --crn --tag _crn --seed s
    the PT ladders tried (tuned PT is the best of these five): --arm pt --beta-mass 0.35 with --tmin / --tmax
    0.01 / 0.15, 0.02 / 0.3, 0.05 / 0.67, 0.05 / 0.67 with --pt-chains 32, and 0.3 / 4.0
    uniform, MAP-Elites, soft Q-learning, Munchausen DQN, the plain sampler, PPO at entropy weight 0.01 and 0.1:
        --arm random --seed s
        --arm me --seed s
        --arm sdqn --beta-mass 0.8 --seed s
        --arm mdqn --beta-mass 0.8 --seed s
        --arm gfn --beta-mass 0.8 --seed s
        --arm ppo --beta-mass 0.8 --seed s
        --arm ppo --beta-mass 0.8 --ppo-ent 0.1 --tag _ent01 --seed s
    BO at 40,000 calls and at 200 calls:
        --arm bo --bo-init 200 --bo-window 2000 --bo-elite 200 --bo-refit 50 --crn --tag _bo40k-onehot --seed s
        --arm bo --bo-init 20 --crn --tag _bo-onehot --budget 200 --seed s
    the supervised screen:
        --arm selfscreen --tag _ss300 --seed s

Compute: one paid call costs 1.2 to 2.5 s on one core, so a 40,000-call run spends about 14 to 28 CPU-hours in
the oracle (the GFlowQEC-online runs at L = 127 took 89,600 to 94,700 s each on a loaded server; BO at 40,000
calls 389,000 to 394,000 s per seed, most of it GP refits; the supervised screen adds about 10,300 s of fitting
and 9,100 s of candidate drawing). The full grid of 20 cells x 10 seeds is about 3,300 to 5,800 CPU-hours.
BO at 200 calls takes about 15 minutes per run.

Output: results/<the same file name as the stored run>.json, rewritten every 1,000 paid calls so a run that is
cut short still leaves an analysable partial file. The learning methods also write their final network to the
same name with .pt.
"""

import argparse
import json
import time

import numpy as np

from gflowqec.oracle import T0, LEROracle
from gflowqec.paths import OUT
from gflowqec.policy import arm_gflowqec_online, arm_gfn
from gflowqec.rivals import arm_bo, arm_maxent, arm_me, arm_ppo, arm_pt, arm_random
from gflowqec.screens import SS_BATCH, SS_CALIB, SS_SLATE, SS_STEPS, arm_selfscreen

GROUP = "C10xS3"     # Z_10 x S_3, n = 300; the only group of this script, kept in every file name

# Settings every run of the paper used. They are constants here, written into the file names ("k1", "det") and
# the run files ("k_mask", "filt", "ls_temp", "bo_slate", "bo_xi") exactly as the stored runs carry them.
K_MASK = 1           # GFlowQEC-online: how many random slots a local move re-emits
FILT = "det"         # GFlowQEC-online: a local move is kept on strict improvement only
LS_TEMP = 1.0        # the objective swap: the sampler temperature (logits / T)
BO_SLATE = 2000      # bo: fresh unpaid candidates the expected-improvement argmax is taken over, each step
BO_XI = 0.01         # bo: the expected-improvement offset


def build_parser():
    ap = argparse.ArgumentParser(description="One n = 300 search run (see the module docstring).")
    ap.add_argument("--arm", default="random",
                    choices=["random", "pt", "gfn", "sdqn", "mdqn", "me", "ppo", "gflowqec", "bo", "selfscreen"])
    ap.add_argument("--budget", type=int, default=40_000, help="paid oracle calls (distinct orbits)")
    ap.add_argument("--seed", type=int, default=0,
                    help="the run's seed: its random stream and the oracle's noise streams")
    ap.add_argument("--p", type=float, default=0.07, help="depolarizing error rate")
    ap.add_argument("--shots", type=int, default=1000, help="BP-OSD shots per paid call")
    ap.add_argument("--tmin", type=float, default=0.05, help="pt: temperature ladder bottom, in nats of s")
    ap.add_argument("--tmax", type=float, default=0.67, help="pt: temperature ladder top")
    ap.add_argument("--pt-chains", type=int, default=8, help="pt: number of chains")
    ap.add_argument("--beta-mass", type=float, default=0.35,
                    help="the learners' reward is s^beta, with beta the smallest that puts this share of the reward "
                         "on the top tenth of the calibration draws; pinned to 0.8 for gflowqec, sdqn, mdqn, ppo")
    # GFlowQEC-online's knobs
    ap.add_argument("--ls-iters", type=int, default=7, help="gflowqec: the depth L, local moves per draw")
    ap.add_argument("--ls-batch", type=int, default=4, help="gflowqec: fresh draws per round")
    ap.add_argument("--ls-uniform", action="store_true",
                    help="gflowqec: no GFlowNet, uniform global draws and uniform local moves (needs --tag)")
    ap.add_argument("--ls-uniform-local", action="store_true",
                    help="gflowqec: the GFlowNet's draws and training, but uniform local moves (needs --tag)")
    ap.add_argument("--ls-paired", action="store_true",
                    help="gflowqec: score each local child on its incumbent's noise stream and re-score an "
                         "accepted child once on its own (needs --tag)")
    ap.add_argument("--ls-objective", default="tb", choices=["tb", "reg"],
                    help="gflowqec: trajectory balance, or the per-step regression of the taken action's logit "
                         "onto log R (the objective swap; needs --tag)")
    ap.add_argument("--crn", action="store_true",
                    help="score every orbit on one noise stream per run (common random numbers; needs --tag)")
    ap.add_argument("--ppo-ent", type=float, default=0.01, help="ppo: entropy weight (a non-default needs --tag)")
    # BO's knobs
    ap.add_argument("--bo-init", type=int, default=20, help="bo: uniform paid calls before the first GP fit")
    ap.add_argument("--bo-window", type=int, default=0,
                    help="bo: condition the GP on the W most recent paid calls only; 0 = on every paid call")
    ap.add_argument("--bo-elite", type=int, default=0,
                    help="bo: also keep the E best paid calls in the GP's window whatever their age")
    ap.add_argument("--bo-refit", type=int, default=1,
                    help="bo: re-optimise the GP hyper-parameters every K steps (the posterior is refreshed "
                         "every step either way)")
    ap.add_argument("--tag", default="", help="file-name suffix that names the cell (e.g. _d127)")
    return ap


def check_flags(args):
    """Refuse knob combinations no cell of the paper used, and pin the learners' reward exponent.

    The tags matter because the file name does not carry every knob (the budget, the PPO entropy, the BO
    settings): two cells that differ only there would overwrite each other without one."""
    if args.arm == "selfscreen":
        # the supervised screen runs the main n = 300 oracle only: one stream per code, p 0.07, 1,000 shots
        assert not args.crn and args.p == 0.07 and args.shots == 1000 and args.tag.startswith("_ss"), \
            "selfscreen runs the main n = 300 oracle (no --crn, p 0.07, 1,000 shots) with a tag _ss..."
    if args.ppo_ent != 0.01:
        assert args.arm == "ppo" and args.tag, "a non-default PPO entropy needs --tag"
    if args.ls_uniform:
        assert args.arm == "gflowqec" and args.tag, "--ls-uniform is the gflowqec arm with a --tag"
    if args.ls_uniform_local:
        assert args.arm == "gflowqec" and args.tag, "--ls-uniform-local is the gflowqec arm with a --tag"
    if args.ls_paired:
        assert args.arm == "gflowqec" and args.tag, "--ls-paired is the gflowqec arm with a --tag"
    if args.arm in ("sdqn", "mdqn", "ppo", "gflowqec"):
        # Pinned here, before the file name is built, so the name, the stored beta_mass and the training agree.
        # 0.8 makes the maximum-entropy learners target the same distribution as the GFlowNets; for PPO it is
        # inert under advantage normalisation and pinned for parity; GFlowQEC-online keeps the plain sampler's 0.8.
        args.beta_mass = 0.8
    if (args.arm == "bo" or args.bo_init != 20 or args.bo_window != 0 or args.bo_elite != 0
            or args.bo_refit != 1):
        # the 200-call and the 40,000-call BO cells differ only by knobs that are not in the file name
        assert args.arm == "bo" and args.tag, \
            "--arm bo needs a --tag; --bo-init / --bo-window / --bo-elite / --bo-refit are its knobs"
        assert args.bo_elite == 0 or args.bo_window > 0, "--bo-elite is inert without --bo-window"
        assert args.bo_refit >= 1, "--bo-refit is a step count"
    if args.ls_objective != "tb":
        # the objective swap is GFlowQEC-online with the loss swapped and nothing else (--crn allowed)
        assert args.arm == "gflowqec" and args.tag, "--ls-objective reg is the gflowqec arm with a --tag"
        assert not (args.ls_uniform or args.ls_uniform_local or args.ls_paired), \
            "the objective swap changes the loss only"
    if args.crn:
        # a different oracle is a new cell; it cannot combine with the per-orbit pairing of --ls-paired
        assert args.tag and not args.ls_paired, "--crn needs a --tag and excludes --ls-paired"


def file_arm(args):
    """The arm's name in file names and in the run's "arm" field: GFlowQEC-online is stored as "lsgfn"."""
    return "lsgfn" if args.arm == "gflowqec" else args.arm


def dump_path(args):
    """The run's file name. Every knob that makes a new cell reaches it, and the stored runs' names come out
    unchanged (the w token once held a cut method's knob and is always empty here)."""
    cell = ""
    if args.arm == "gflowqec":
        cell = (f"_k{K_MASK}i{args.ls_iters}{FILT}"
                + ("u" if args.ls_uniform else "")
                + ("g" if args.ls_uniform_local else "")
                + ("P" if args.ls_paired else "")
                + ("S" if args.ls_objective == "reg" else ""))
    chains = f"_c{args.pt_chains}" if (args.arm == "pt" and args.pt_chains != 8) else ""
    crn = "C" if args.crn else ""
    name = (f"mitten_lersearch_{file_arm(args)}_{GROUP}_b{args.beta_mass}_t{args.tmin}-{args.tmax}{chains}"
            f"_w{cell}{crn}_s{args.seed}{args.tag}.json")
    return OUT / name


def main():
    args = build_parser().parse_args()
    check_flags(args)

    OUT.mkdir(exist_ok=True)
    # one generator for the whole run: every uniform draw, chain move and policy sample comes from it, in order
    rng = np.random.default_rng(args.seed)
    # the oracle's noise streams are seeded by the run's seed as well
    oracle = LEROracle(GROUP, args.budget, p=args.p, shots=args.shots, seed=args.seed)
    oracle.global_stream = args.crn

    path = dump_path(args)
    model_path = str(path).replace(".json", ".pt")     # where the learning methods save their final network
    extra = {}      # method-specific fields of the run file (the supervised screen's stamps and counters)

    def dump():
        # Every paid orbit in CALL ORDER, so a readout can split a run into phases. Called every 1,000 paid
        # calls and once at the end. The field set is the stored runs' field set: warm, ls_full, any_order,
        # ls_repulse and bias_w are knobs of methods outside the paper, always at these values here.
        order = list(oracle.meta.values())
        d = {"arm": file_arm(args), "arena": "generating", "group": GROUP,
             "seed": args.seed, "budget": args.budget, "p": args.p,
             "shots": args.shots, "tmin": args.tmin, "tmax": args.tmax,
             "pt_chains": args.pt_chains,
             "beta_mass": args.beta_mass, "warm": 4000,
             "k_mask": K_MASK, "filt": FILT, "ls_iters": args.ls_iters,
             "ls_uniform": args.ls_uniform, "ls_full": False,
             "ls_uniform_local": args.ls_uniform_local, "any_order": False,
             "ls_repulse": False,
             "ls_batch": args.ls_batch, "tag": args.tag,
             "ppo_ent": args.ppo_ent,
             "bias_w": None, "run_seed": oracle.seed,
             "calls": oracle.calls, "hits": oracle.hits,
             "distinct": len(oracle.cache),
             # per paid orbit: the in-run rate estimates, k, the stream tag (0 global draw, 1 local move)
             "ler_pq": [m["ler_pq"] for m in order],
             "k": [m["k"] for m in order],
             "ler_x": [m["ler_x"] for m in order],
             "ler_z": [m["ler_z"] for m in order],
             "src": [m.get("src", 0) for m in order],
             # and the code itself: the orbit's canonical ten-index representative
             "rep": [m["rep"] for m in order]}
        if args.crn:
            d["crn"] = True
        if args.ls_paired:
            # which orbits were valued on an incumbent's stream (aligned with the lists above), and how many
            # accepted children were re-scored on their own stream
            d["ls_paired"] = True
            d["paired"] = [k in oracle.paired_keys for k in oracle.meta]
            d["n_rescored"] = oracle.n_rescored
        if args.ls_objective == "reg":
            d["ls_objective"] = "reg"
            d["ls_temp"] = LS_TEMP
        if args.arm == "bo":
            # the BO knobs, and the shot count its best codes are re-estimated at by the readouts
            d["bo_encoding"], d["bo_init"] = "onehot", args.bo_init
            d["bo_slate"], d["bo_xi"] = BO_SLATE, BO_XI
            d["bo_rescore_shots"] = 4000
            if args.bo_window:
                d["bo_window"], d["bo_elite"] = args.bo_window, args.bo_elite
            if args.bo_refit != 1:
                d["bo_refit"] = args.bo_refit
        d.update(extra)
        with open(path, "w") as f:
            json.dump(d, f)

    oracle.checkpoint = dump

    if args.arm == "random":
        arm_random(oracle, GROUP, rng)
    elif args.arm == "pt":
        arm_pt(oracle, GROUP, rng, args.tmin, args.tmax, args.pt_chains)
    elif args.arm == "selfscreen":
        # The screen's fixed constants, stamped BEFORE the run so every partial file carries them; the arm
        # asserts that the network it builds matches them and updates the three counters in place each round.
        extra.update({"ss_calib": SS_CALIB, "ss_slate": SS_SLATE, "ss_batch": SS_BATCH, "ss_steps": SS_STEPS,
                      "ss_fit_batch": 64, "ss_lr": 1e-3, "ss_hidden": 64, "ss_in_dim": 1440, "ss_params": 96449,
                      "ss_mlp": ("bag 240 + relational 1200 -> 64 -> 64 -> 1, ReLU, Adam 1e-3, batch 64, squared "
                                 "error on s z-scored over the paid rows, conjugation per side + mirror p 1/2 "
                                 "(the n = 180 screen's recipe), warm start"),
                      "ss_rescore_shots": 4000, "n_rounds": 0, "fit_cpu_s": 0.0, "cand_cpu_s": 0.0})
        arm_selfscreen(oracle, GROUP, rng, args.seed, extra)
    elif args.arm == "gfn":
        arm_gfn(oracle, GROUP, rng, args.seed, beta_mass=args.beta_mass, save_path=model_path)
    elif args.arm == "me":
        arm_me(oracle, GROUP, rng)
    elif args.arm == "gflowqec":
        arm_gflowqec_online(oracle, GROUP, rng, args.seed, batch=args.ls_batch, iters=args.ls_iters,
                            k_mask=K_MASK, uniform=args.ls_uniform, uniform_local=args.ls_uniform_local,
                            paired=args.ls_paired, beta_mass=args.beta_mass, save_path=model_path,
                            objective=args.ls_objective, temp=LS_TEMP)
    elif args.arm == "ppo":
        arm_ppo(oracle, GROUP, rng, args.seed, beta_mass=args.beta_mass, save_path=model_path,
                ent_coef=args.ppo_ent)
    elif args.arm in ("sdqn", "mdqn"):
        # alpha_m = 0.15 is Tiapkin et al.'s Munchausen weight; 0 is plain soft Q-learning
        arm_maxent(oracle, GROUP, rng, args.seed, alpha_m=0.15 if args.arm == "mdqn" else 0.0,
                   beta_mass=args.beta_mass, save_path=model_path)
    elif args.arm == "bo":
        arm_bo(oracle, GROUP, rng, init=args.bo_init, slate=BO_SLATE, xi=BO_XI,
               window=args.bo_window, elite=args.bo_elite, refit=args.bo_refit)

    pqs = np.array([m["ler_pq"] for m in oracle.meta.values()])
    print(f"\narm {args.arm}, {GROUP}, seed {args.seed}: {oracle.calls} calls, "
          f"{len(oracle.cache)} distinct orbits, {oracle.hits} cache hits, "
          f"{time.time() - T0:.0f}s")
    print(f"  ler_pq: min {pqs.min():.6f}  median {np.median(pqs):.6f}")
    dump()
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
