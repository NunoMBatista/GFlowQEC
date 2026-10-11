"""Re-estimate stored n = 300 search runs on fresh, independent shots.

An n = 300 search run (scripts/n300_search.py) scores every paid code on 1,000 shots of BP-OSD. Those
in-run scores are what the search selected on, so they are biased low at the top: a lucky draw looks
like a good code. The paper never counts a run from its own in-run scores. It re-estimates codes of
the run on 4,000 independent shots, on a noise stream the search never saw. Two legs:

  visited  a uniform random subsample of 800 of the run's visits (counts.visit_indices: the calls its
           proposal mechanism made, after any uniform calibration draws), each re-estimated on stream
           run_seed = 20,000 + seed. There is no selection on the in-run score, so the good-proposal
           count W (Eq. 20) reads the same way for every method. These files are behind every
           n = 300 good-proposal count: Table IV and the text of Sec. 8, the text and Fig. 7 of
           Sec. 9.2, Table X and the text of App. D.2, and App. C.
           Writes results/g1_repay_visited_i4000_<arm>_<cell><tag>_s<seed>.json.
  elites   the run's 200 best codes by in-run score, each re-estimated on stream run_seed = 10,000 + seed:
           the best-code comparison of App. D.2 (readouts/appg300.py).
           Writes results/g1_repay_elites_i4000_<arm><tag>_s<seed>_top200.json.

Every re-estimate is oracle.ler_score: a pure function of (run_seed, the code's orbit), so a re-run on
the same ldpc build reproduces every value exactly. The search runs are read from data/n300 by default;
--data points at another folder (for instance results/, to re-score runs you regenerated).

Which run a command picks: the file mitten_lersearch_<arm>_C10xS3_*<cell>_s<seed><tag>.json must match
exactly one stored run (--src-cell plays the role of --cell in the elites leg). The 20 cells of the
paper (counts.CELLS), each for --seed 0..5, with --workers 2 as run:

    cell    --arm       --cell               --tag
    d31     gflowqec    det                  _d31
    f31     gflowqec    detP                 _f31
    g31     gflowqec    detg                 _g31
    u31     gflowqec    detu                 _u31
    d127    gflowqec    k1i127det            _d127
    c127    gflowqec    k1i127detC           _c127
    u127    gflowqec    k1i127detu           _u127
    p31     gflowqec    k1i31detS            _p31
    p127    gflowqec    k1i127detSC          _p127
    pt      pt          b0.35_t0.01-0.15_w   (empty: --tag=)
    ptc     pt          b0.35_t0.01-0.15_wC  _crn
    rnd     random      w                    (empty)
    me      me          w                    (empty)
    sdqn    sdqn        w                    (empty)
    mdqn    mdqn        w                    (empty)
    ppo     ppo         w                    (empty)
    ppo1    ppo         w                    _ent01
    gfn     gfn         b0.8_t0.05-0.67_w    (empty)
    bo40k   bo          wC                   _bo40k-onehot
    ss300   selfscreen  w                    _ss300

    python -m scripts.n300_rescore visited --arm gflowqec --cell det --tag _d31 --seed 0 --workers 2

The d31 and f31 visited files in data/n300 were written before the arm entered the file name, so they
are named det_d31 and detP_f31; this script writes lsgfn_det_d31 and lsgfn_detP_f31 for them.

The six best-code runs, each for --seed 0..5:

    python -m scripts.n300_rescore elites --arm gflowqec --tag _d127 --workers 2   (also _u127, _c127)
    python -m scripts.n300_rescore elites --arm pt --tag= --src-cell t0.01-0.15_w --workers 2
    python -m scripts.n300_rescore elites --arm pt --tag _crn --workers 2
    python -m scripts.n300_rescore elites --arm selfscreen --tag _ss300 --workers 2

Compute: a visited run is 800 codes x 4,000 shots, 1,360 to 1,392 s with two workers on one server
(about 0.4 CPU-h), about 46 CPU-h over the 120 runs; an elites run is about a quarter of that.
"""

import argparse
import glob
import json
import time
from multiprocessing import Pool

import numpy as np

from gflowqec.counts import visit_indices
from gflowqec.oracle import ler_score
from gflowqec.paths import DATA, OUT

GROUP = "C10xS3"            # the n = 300 arena
P = 0.07                    # depolarizing rate of every n = 300 run
INDEP_SHOTS = 4000          # shots per re-estimate (the search scored on 1,000)
INDEP_OFFSET = 10_000       # elites leg: run_seed of the re-score = offset + seed
TOP = 200                   # elites leg: the run's 200 best codes by in-run score

VISITED_N = 800             # visited leg: uniform subsample size per run
VISITED_OFFSET = 20_000     # visited leg: run_seed of the re-score = offset + seed (a stream of its own)
VISITED_DRAW = 20260904     # visited leg: numpy seed of the subsample draw = this + seed

# The command line says gflowqec for GFlowQEC-online; its stored runs carry the token lsgfn.
ARM_TOKEN = {"gflowqec": "lsgfn"}


def score_orbit(task):
    """Elites leg: one code re-estimated on the elites stream. The field ler_pq_booked (the name the stored
    files use) is the code's in-run score."""
    seed, idx, rep, pq_booked = task
    r = ler_score(GROUP, rep, P, INDEP_SHOTS, run_seed=INDEP_OFFSET + seed)
    return {"idx": idx, "rep": rep, "k": r["k"], "ler_pq_booked": pq_booked,
            "fails": r["fails"][2], "ler_pq": r["ler_pq"]}


def score_visited(task):
    """Visited leg: one code re-estimated on the visited stream."""
    seed, idx, rep, pq_booked = task
    r = ler_score(GROUP, rep, P, INDEP_SHOTS, run_seed=VISITED_OFFSET + seed)
    return {"idx": idx, "rep": rep, "k": r["k"], "ler_pq_booked": pq_booked,
            "fails": r["fails"][2], "ler_pq": r["ler_pq"]}


def elites_glob(data, arm, seed, tag, src_cell=""):
    # --src-cell is a file-name infix that picks ONE run when several variants of an arm share the seed
    # and the tag (five PT runs end in _w_s{seed}.json); the elites leg's twin of the visited leg's --cell
    return f"{data}/mitten_lersearch_{arm}_C10xS3_*{src_cell}_s{seed}{tag}.json"


def run_elites(args, arm):
    files = glob.glob(elites_glob(args.data, arm, args.seed, args.tag, args.src_cell))
    assert len(files) == 1, files
    d = json.load(open(files[0]))
    assert d["shots"] == 1000 and d["p"] == P and d.get("run_seed", d["seed"]) == args.seed
    # the TOP best codes by in-run score, then put back in call order
    order = sorted(range(len(d["ler_pq"])), key=lambda i: d["ler_pq"][i])[:TOP]
    tasks = [(args.seed, i, d["rep"][i], d["ler_pq"][i]) for i in sorted(order)]
    path = OUT / f"g1_repay_elites_i{INDEP_SHOTS}_{arm}{args.tag}_s{args.seed}_top{TOP}.json"
    meta = {"source": files[0], "tag": args.tag, "seed": args.seed, "top": TOP,
            "indep_shots": INDEP_SHOTS, "run_seed_rescore": INDEP_OFFSET + args.seed,
            "n_elites": len(tasks)}
    # the file is rewritten every 200 codes, so a killed run keeps what it paid
    done, start = [], time.time()
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(score_orbit, tasks, chunksize=8):
            done.append(r)
            if len(done) % 200 == 0 or len(done) == len(tasks):
                done.sort(key=lambda r: r["idx"])
                json.dump({**meta, "n_done": len(done), "orbits": done}, open(path, "w"))
                print(f"{len(done)}/{len(tasks)} orbits, {time.time() - start:.0f}s", flush=True)
    print(f"wrote {path}")


def run_visited(args, arm):
    """Re-score a UNIFORM random subsample of one run's visits, independent noise. Selection-free, so it
    reads the same way for methods whose rules for choosing what to pay differ."""
    # --cell is the file-name block that tells the arm's variants apart (det / detP at depth 31, the
    # baseline arms' "w" block; the plain sampler needs b0.8_t0.05-0.67_w)
    files = glob.glob(f"{args.data}/mitten_lersearch_{arm}_{GROUP}_*{args.cell}_s{args.seed}{args.tag}.json")
    assert len(files) == 1, files
    d = json.load(open(files[0]))
    assert d["shots"] == 1000 and d["p"] == P and d.get("run_seed", d["seed"]) == args.seed
    # the visit rule per arm lives in counts.visit_indices
    local = visit_indices(d)
    pick = np.random.default_rng(VISITED_DRAW + args.seed).choice(local, VISITED_N, replace=False)
    tasks = [(args.seed, int(i), d["rep"][i], d["ler_pq"][i]) for i in sorted(pick)]
    path = OUT / f"g1_repay_visited_i{INDEP_SHOTS}_{arm}_{args.cell}{args.tag}_s{args.seed}.json"
    meta = {"source": files[0], "tag": args.tag, "seed": args.seed, "n_local": len(local),
            "n_sub": VISITED_N, "indep_shots": INDEP_SHOTS,
            "run_seed_rescore": VISITED_OFFSET + args.seed, "draw_seed": VISITED_DRAW + args.seed}
    # the file is rewritten every 100 codes, so a killed run keeps what it paid
    done, start = [], time.time()
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(score_visited, tasks, chunksize=8):
            done.append(r)
            if len(done) % 100 == 0 or len(done) == len(tasks):
                done.sort(key=lambda r: r["idx"])
                json.dump({**meta, "n_done": len(done), "orbits": done}, open(path, "w"))
                print(f"{len(done)}/{len(tasks)} orbits, {time.time() - start:.0f}s", flush=True)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("leg", choices=["elites", "visited"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="_d31")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell", default="det", help="visited leg: file-name block of the run, e.g. det or detP")
    ap.add_argument("--src-cell", default="", help="elites leg: file-name infix that picks one run, e.g. t0.01-0.15_w")
    ap.add_argument("--arm", default="gflowqec",
                    choices=["gflowqec", "pt", "random", "me", "sdqn", "mdqn", "ppo", "gfn", "bo", "selfscreen"],
                    help="the search method of the stored run")
    ap.add_argument("--data", default=DATA / "n300", help="folder holding the search runs")
    args = ap.parse_args()
    arm = ARM_TOKEN.get(args.arm, args.arm)
    OUT.mkdir(exist_ok=True)
    {"elites": run_elites, "visited": run_visited}[args.leg](args, arm)


if __name__ == "__main__":
    main()
