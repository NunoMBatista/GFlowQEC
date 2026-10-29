"""The n = 180 reference sample: 1,000 uniform draws over C6 x S3 with exact distances.

Regenerates the four files of data/keys that the paper's n = 180 answer key reads (the answer-key table, the
text on the oracle and references, and the reference sample behind the n = 180 tables):

    draw             r169_headroom_C6xS3_n1000.json    1,000 uniform legal draws of the lifted-product
                                                       arena over C6 x S3, each scored [[n, k, d]] by
                                                       the SAT ladder (min over the two sectors)
    resolve          r169_resolve_C6xS3_c5M.json       the draws whose ladder ran out of budget,
                                                       re-run from their proven lower bound with 25x
                                                       the budget
    read             r174_bias_read_C6xS3_c200k.json   per-sector distances d_X and d_Z of the draws
                                                       whose two sides generate the whole group (666)
    resolve-sectors  r174_bias_resolve_C6xS3_c5M.json  the sectors the read left undecided, at 25x the
                                                       budget

Commands, in order (run from the repo root):
    python -m scripts.n180_reference draw --draws 1000 --workers 4
    python -m scripts.n180_reference resolve --workers 2
    python -m scripts.n180_reference read
    python -m scripts.n180_reference resolve-sectors

Compute: about 4.0 CPU-hours in total (draw 1.15 h, resolve 0.60 h, read 0.87 h, resolve-sectors
1.37 h, on one server CPU).

Every leg writes to results/, and each later leg reads the earlier legs' files from results/. To run
a later leg on the shipped files instead, copy them from data/keys/ into results/ first. The read and
resolve-sectors legs are resume-safe: they skip the codes already in their output file.

How to read a distance in these files. The SAT ladder asks "is there a logical of weight <= w?" for
w = 1, 2, ... up to the cap 12. Every rung answered "no" is a proof, so when a rung runs out of its
conflict budget the ladder stops and the reported d is a LOWER BOUND (d >= that rung), flagged by
exact = False. Only rows with exact = True are exact distances. Censored rows are never read as point
values; the resolve legs exist to close as many of them as a larger budget can.
"""

import argparse
import json
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

from gflowqec.certify import _kernel_basis, ladder_from
from gflowqec.codes import build, generates_g, random_exps, score
from gflowqec.paths import OUT

GROUP = "C6xS3"                 # Z_6 x S_3, order 36, so n = 5 * 36 = 180 qubits
# The draws were made by a script that looped over five groups and seeded each one with
# 20260905 + its index in that list; C6xS3 was index 3, so its stream is 20260905 + 3 = 20260908.
# (The shipped draw file records the base 20260905 in its draw_seed field.)
DRAW_SEED = 20260908
MAX_WEIGHT = 12                 # the ladder's cap: a code with no logical of weight <= 12 reads d >= 13
CONFLICTS = 200_000             # CaDiCaL conflict budget per rung, first pass (draw and read)
RESOLVE_CONFLICTS = 5_000_000   # per rung on the censored ladders only, 25x the first pass

DRAW_PATH = OUT / f"r169_headroom_{GROUP}_n1000.json"
RESOLVE_PATH = OUT / f"r169_resolve_{GROUP}_c5M.json"
READ_PATH = OUT / f"r174_bias_read_{GROUP}_c200k.json"
SECTOR_RESOLVE_PATH = OUT / f"r174_bias_resolve_{GROUP}_c5M.json"


# ---------------------------------------------------------------------------------------------
# Legs draw and resolve: [[n, k, d]] with d the minimum over the X and Z sectors
# ---------------------------------------------------------------------------------------------

def _score(task):
    """Score one draw from scratch (ladder from weight 1). Runs in a worker process."""
    name, i, exps = task
    t = time.perf_counter()
    n, k, d, exact, w = score(name, exps, max_weight=MAX_WEIGHT, conflict_budget=CONFLICTS)
    return {"i": i, "exps": exps, "n": n, "k": k, "d": d, "exact": exact, "w": w,
            "seconds": time.perf_counter() - t}


def draw(args):
    # The whole task list is drawn from the rng BEFORE any scoring, so the first N draws of a
    # shorter run are exactly the first N draws of the 1,000-draw run.
    rng = np.random.default_rng(DRAW_SEED)
    tasks = [(GROUP, i, [int(v) for v in random_exps(rng, GROUP)]) for i in range(args.draws)]
    path = OUT / f"r169_headroom_{GROUP}_n{args.draws}.json"
    meta = {"group": GROUP, "draws": args.draws, "draw_seed": DRAW_SEED, "max_weight": MAX_WEIGHT,
            "conflicts": CONFLICTS}
    OUT.mkdir(exist_ok=True)
    done, start = [], time.time()
    with Pool(args.workers) as pool:
        # Workers finish out of order; rows are sorted by draw index before every save.
        for r in pool.imap_unordered(_score, tasks, chunksize=8):
            done.append(r)
            if len(done) % 200 == 0 or len(done) == args.draws:
                done.sort(key=lambda r: r["i"])
                json.dump({**meta, "n_done": len(done), "codes": done}, open(path, "w"))
                print(f"{GROUP}: {len(done)}/{args.draws}, {time.time() - start:.0f}s", flush=True)


def _resolve(task):
    """Re-run one censored ladder from its proven lower bound with the big budget. Starting at the
    lower bound is sound: every rung below it was already proved to hold no logical."""
    name, i, exps, lower = task
    t = time.perf_counter()
    n, k, d, exact, w = score(name, exps, max_weight=MAX_WEIGHT, conflict_budget=RESOLVE_CONFLICTS,
                              min_weight=lower)
    return {"i": i, "exps": exps, "lower": lower, "n": n, "k": k, "d": d, "exact": exact, "w": w,
            "seconds": time.perf_counter() - t}


def resolve(args):
    """The censored ladders of the 1,000-draw file, warm-started at their proven lower bound."""
    src = json.load(open(DRAW_PATH))
    censored = [c for c in src["codes"] if not c["exact"]]
    tasks = [(GROUP, c["i"], c["exps"], c["d"]) for c in censored]
    meta = {"group": GROUP, "source": DRAW_PATH.name, "conflicts": RESOLVE_CONFLICTS,
            "max_weight": MAX_WEIGHT, "n_censored": len(censored)}
    done, start = [], time.time()
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(_resolve, tasks, chunksize=1):
            done.append(r)
            done.sort(key=lambda r: r["i"])
            json.dump({**meta, "n_done": len(done), "codes": done}, open(RESOLVE_PATH, "w"))
            print(f"{GROUP} resolve: {len(done)}/{len(censored)}, i={r['i']} d={r['d']} exact={r['exact']} "
                  f"{r['seconds']:.0f}s, {time.time() - start:.0f}s total", flush=True)
    closed = [r for r in done if r["exact"]]
    print(f"closed {len(closed)}/{len(done)}; d among closed: {sorted(Counter(r['d'] for r in closed).items())}")


# ---------------------------------------------------------------------------------------------
# Legs read and resolve-sectors: d_X and d_Z separately
# ---------------------------------------------------------------------------------------------
# In a non-abelian lifted product the two sectors can have different distances (inside abelian
# BB codes they cannot), which is what biased noise rewards. d = min(d_X, d_Z) is already known
# from the draw legs, so each sector's ladder starts at d.

def generating_draws():
    """The 1,000 draws with the resolved ladders folded in, kept only when BOTH sides' elements
    generate the whole group (a draw inside a subgroup is a disjoint union of smaller codes, not a
    member of the family), highest d first."""
    rows = json.load(open(DRAW_PATH))["codes"]
    resolved = {r["i"]: r for r in json.load(open(RESOLVE_PATH))["codes"]}
    rows = [r if r["exact"] else resolved.get(r["i"], r) for r in rows]
    rows = [r for r in rows if generates_g(GROUP, np.array(r["exps"]))]
    rows.sort(key=lambda r: -r["d"])      # stable: equal d keeps the draw order
    return rows


def sectors(exps):
    """(Hx, X-logical basis, Hz, Z-logical basis) of the code; d_X is the X-sector distance.
    A logical of one sector lies in the kernel of the OTHER sector's check matrix."""
    code = build(GROUP, np.array(exps))
    Hx = np.asarray(code.matrix_x) % 2
    Hz = np.asarray(code.matrix_z) % 2
    return Hx, _kernel_basis(Hz), Hz, _kernel_basis(Hx)


def load(path):
    """The rows already saved by an interrupted run, or none."""
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return []


def read(args):
    rows = generating_draws()
    out = load(READ_PATH)
    done = {o["i"] for o in out}
    print(f"{len(rows)} generating draws, {len(out)} already read", flush=True)
    started = time.time()
    for r in rows:
        if r["i"] in done:
            continue
        t = time.time()
        Hx, lx, Hz, lz = sectors(r["exps"])
        dX, exX = ladder_from(Hx, lx, r["d"], CONFLICTS, cap=MAX_WEIGHT)
        dZ, exZ = ladder_from(Hz, lz, r["d"], CONFLICTS, cap=MAX_WEIGHT)
        out.append({"i": r["i"], "exps": r["exps"], "d": r["d"], "dX": dX, "exactX": exX,
                    "dZ": dZ, "exactZ": exZ, "seconds": time.time() - t})
        if len(out) % 10 == 0:
            json.dump(out, open(READ_PATH, "w"))
            print(f"[{time.time() - started:5.0f}s] {len(out)} codes; last d {r['d']}: "
                  f"dX {dX}{'' if exX else '+'} dZ {dZ}{'' if exZ else '+'}", flush=True)
    json.dump(out, open(READ_PATH, "w"))
    print("done", len(out), flush=True)


def resolve_sectors(args):
    rows = load(READ_PATH)
    # Censored rungs >= 8 first (the only ones that can lift a code above the balanced top),
    # then the rung-7 censors.
    todo = []
    for sector in ["Z", "X"]:
        todo += [(r, sector) for r in rows if not r["exact" + sector] and r["d" + sector] >= 8]
    for sector in ["Z", "X"]:
        todo += [(r, sector) for r in rows if not r["exact" + sector] and r["d" + sector] == 7]
    out = load(SECTOR_RESOLVE_PATH)
    done = {(o["i"], o["sector"]) for o in out}
    print(f"{len(todo)} censored sectors, {len(out)} resolved", flush=True)
    started = time.time()
    for r, sector in todo:
        if (r["i"], sector) in done:
            continue
        t = time.time()
        Hx, lx, Hz, lz = sectors(r["exps"])
        check, logicals = (Hx, lx) if sector == "X" else (Hz, lz)
        # Restart at the censored rung: every rung below it was proved empty by the read.
        w, exact = ladder_from(check, logicals, r["d" + sector], RESOLVE_CONFLICTS, cap=MAX_WEIGHT)
        out.append({"i": r["i"], "d": r["d"], "sector": sector, "start": r["d" + sector],
                    "resolved": w if exact else None, "censored_at": None if exact else w,
                    "seconds": time.time() - t})
        json.dump(out, open(SECTOR_RESOLVE_PATH, "w"))
        print(f"[{time.time() - started:5.0f}s] i {r['i']} d {r['d']} {sector}: {r['d' + sector]} -> "
              f"{w if exact else f'censored at {w}'} in {time.time() - t:.0f}s", flush=True)
    print("done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("leg", choices=["draw", "resolve", "read", "resolve-sectors"])
    ap.add_argument("--draws", type=int, default=1000, help="draw: number of uniform draws")
    ap.add_argument("--workers", type=int, default=4, help="draw and resolve: worker processes")
    args = ap.parse_args()
    legs = {"draw": draw, "resolve": resolve, "read": read, "resolve-sectors": resolve_sectors}
    legs[args.leg](args)


if __name__ == "__main__":
    main()
