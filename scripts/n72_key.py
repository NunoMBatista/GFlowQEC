"""The n = 72 answer key: an exact distance for every code with k > 0 of the Z_6 x Z_6 two-block space.

Regenerates data/keys/certified_l6m6_w3.csv, escalated_l6m6_w3.csv and closed_l6m6_w3.csv, which
together are the n = 72 row of the answer-key table: 9,280 forms with k > 0, every distance exact,
the [[72,12,6]] summit (48 forms) and the 1,440 [[72,4,8]] forms of largest d.

The key is built in three passes, each aimed at the codes the pass before it could not close:

  base      the two-sided squeeze of certify.certify_distance on all 9,280 forms: an exhaustive
            sweep up to weight 5 (a proof of d > 5 when it finds nothing) against qldpc's
            randomized decoder bound (a logical it actually built, so d <= its weight). Exact when
            the two meet; 1,802 forms stay open as honest intervals [6, upper].
  escalate  the same squeeze on the open forms with the sweep raised to weight 6. The rule is to
            raise the sweep, never the decoder trials: the randomized bound can stall above the
            true d, while the sweep always tells the truth. A weight-6 logical closes d = 6
            exactly (362 forms); otherwise d >= 7 is proved and the interval stays (1,440).
  close     the SAT decision ladder (certify.sat_distance_bounded) on every escalated form,
            warm-started at the sweep's lower bound: closes all 1,440 at d = 8.

Each pass reads the ARCHIVED output of the pass before it from data/keys (escalate reads the
certified file, close the escalated file), not a fresh one from results/. The decoder bound draws
fresh OS randomness, so a re-run base pass may leave a slightly different set of open forms;
reading the archive lets each pass be checked against its own stored file.

Run from the repo root:
  python -m scripts.n72_key base                              # all 9,280 forms (cluster-sized)
  python -m scripts.n72_key base --sample 100                 # costing pilot: 100 random forms
  python -m scripts.n72_key escalate                          # all 1,802 open forms in one process
  python -m scripts.n72_key escalate --chunk 0 --chunks 16    # one task of a 16-task cluster array
  python -m scripts.n72_key merge                             # join the array's part files
  python -m scripts.n72_key close --workers 4

Compute: 31.30 CPU-hours in all, summed from the files' seconds columns: base 6.28 (one cluster
job), escalation 24.62 (a 16-task cluster array; the weight-6 layer is about 156 million supports
per sector, tens of seconds per form), SAT close 0.40 (359 s wall on 4 laptop cores).

Outputs in results/: certified_l6m6_w3.csv, escalated_l6m6_w3.csv (or _part{i}.csv per array
task), closed_l6m6_w3.csv. Columns key, k, d_lower, d_upper, exact, seconds; the closed file adds
old_lower, old_upper (the escalated interval it started from).
"""

import argparse
import ast
import csv
import multiprocessing as mp
import time
from collections import Counter

import numpy as np

from gflowqec.certify import certify_distance, sat_distance_bounded
from gflowqec.codes import build_code
from gflowqec.paths import DATA, OUT

KEYS = DATA / "keys"
SWEEP_CAP = 5                 # base pass: exhaustive floor; d <= 5 forms close by the sweep alone
ESCALATED_CAP = 6             # escalation: closes every true d = 6, proves d >= 7 otherwise
TRIALS = 1000                 # randomized upper bound budget (its cutoff exits early)
CONFLICT_BUDGET = 5_000_000   # SAT close: per-rung budget; a rung that blows it stays an interval


def write_rows(rows, path):
    """Write a list of dicts as a CSV, columns in the first row's order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def squeeze(key, k, cap):
    """One row of the key: the two-sided squeeze of the form `key` with the sweep capped at `cap`."""
    l, m, a, b = key
    code = build_code(l, m, a, b)
    t0 = time.time()
    cert = certify_distance(code, max_sweep_weight=cap, trials=TRIALS)
    return {"key": repr(key), "k": k,
            "d_lower": cert.lower, "d_upper": cert.upper,
            "exact": int(cert.lower == cert.upper),
            "seconds": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------- base pass

def base(sample):
    """Squeeze every k > 0 form (or a random sample of them) with the sweep capped at 5."""
    viable = []
    with open(KEYS / "candidates_l6m6_w3.csv") as f:      # scripts/key_forms.py 6 6
        for row in csv.DictReader(f):
            if int(row["k"]) > 0:
                viable.append((ast.literal_eval(row["key"]), int(row["k"])))
    if sample is not None:
        # The costing pilot: a fixed random subset, kept in file order.
        rng = np.random.default_rng(0)
        picks = rng.choice(len(viable), size=sample, replace=False)
        viable = [viable[i] for i in sorted(picks)]
    print(f"certifying {len(viable)} viable codes of (6,6) "
          f"(sweep cap {SWEEP_CAP}, {TRIALS} trials)", flush=True)

    rows = []
    start = time.time()
    for done, (key, k) in enumerate(viable, 1):
        rows.append(squeeze(key, k, SWEEP_CAP))
        if done % 25 == 0:
            print(f"  {done}/{len(viable)} done, {time.time() - start:.0f}s", flush=True)
    out = OUT / "certified_l6m6_w3.csv"
    write_rows(rows, out)

    # Intervals are counted, never silently escalated: they are the escalation pass's work list.
    exact = [r for r in rows if r["exact"]]
    open_rows = [r for r in rows if not r["exact"]]
    spectrum = Counter(r["d_lower"] for r in exact)
    total = time.time() - start
    print(f"\n{len(exact)}/{len(rows)} exact; distance spectrum: "
          + "  ".join(f"d={d}: {c}" for d, c in sorted(spectrum.items())))
    if open_rows:
        intervals = Counter((r["d_lower"], r["d_upper"]) for r in open_rows)
        print("open squeezes (need escalation): "
              + "  ".join(f"[{lo},{hi}]: {c}" for (lo, hi), c in sorted(intervals.items())))
    mean_cost = total / len(rows)
    print(f"cost: {total:.0f}s total, {mean_cost:.2f}s/code "
          f"-> full space of 9,280 viable ~ {9280 * mean_cost / 3600:.1f} CPU-hours")
    print(f"wrote {out}")


# ---------------------------------------------------------------------------- escalation pass

def escalate(chunk, chunks):
    """Re-squeeze the base key's open forms with the sweep raised to weight 6.

    A cluster array splits the work into `chunks` tasks; task `chunk` takes every chunks-th open
    form starting at `chunk` (interleaved, so every task gets a similar mix of costs) and writes
    its own part file.
    """
    open_codes = []
    with open(KEYS / "certified_l6m6_w3.csv") as f:       # the base key as shipped
        for row in csv.DictReader(f):
            if row["exact"] == "0":
                open_codes.append((ast.literal_eval(row["key"]), int(row["k"])))
    open_codes = open_codes[chunk::chunks]
    print(f"escalating {len(open_codes)} open codes of (6,6) "
          f"(chunk {chunk}/{chunks}, sweep cap {ESCALATED_CAP})", flush=True)

    rows = []
    start = time.time()
    for done, (key, k) in enumerate(open_codes, 1):
        rows.append(squeeze(key, k, ESCALATED_CAP))
        if done % 10 == 0:
            print(f"  {done}/{len(open_codes)} done, {time.time() - start:.0f}s", flush=True)
    part = "" if chunks == 1 else f"_part{chunk}"
    out = OUT / f"escalated_l6m6_w3{part}.csv"
    write_rows(rows, out)

    closed = [r for r in rows if r["exact"]]
    still_open = [r for r in rows if not r["exact"]]
    spectrum = Counter(r["d_lower"] for r in closed)
    print(f"\n{len(closed)}/{len(rows)} closed; spectrum: "
          + "  ".join(f"d={d}: {c}" for d, c in sorted(spectrum.items())))
    if still_open:
        intervals = Counter((r["d_lower"], r["d_upper"]) for r in still_open)
        print("still open (for the SAT close): "
              + "  ".join(f"[{lo},{hi}]: {c}" for (lo, hi), c in sorted(intervals.items())))
    print(f"cost: {time.time() - start:.0f}s, wrote {out}")


def merge():
    """Concatenate the array's part files into escalated_l6m6_w3.csv.

    Parts are taken in file-name order, as the shell lists them (part0, part1, part10, ...,
    part15, part2, ..., part9): that is the row order of the archived escalated file.
    """
    parts = sorted(OUT.glob("escalated_l6m6_w3_part*.csv"))
    rows = []
    for path in parts:
        rows += list(csv.DictReader(open(path)))
    out = OUT / "escalated_l6m6_w3.csv"
    write_rows(rows, out)
    print(f"merged {len(parts)} part files, {len(rows)} rows, wrote {out}")


# ---------------------------------------------------------------------------- SAT close

def close_one(row):
    """SAT-certify one escalated form, warm-started at its certified lower bound.

    The ladder asks "is there a logical of weight <= w?" for w = d_lower, d_lower + 1, ... and the
    first yes is d. Starting at d_lower skips rungs already known to be empty. That chains two
    instruments, so it is sound only because d_lower comes from the EXHAUSTIVE sweep (a proof),
    never from the randomized decoder bound (a guess).
    """
    l, m, a, b = ast.literal_eval(row["key"])
    code = build_code(l, m, a, b)
    lower = int(row["d_lower"])

    t0 = time.time()
    d, exact = sat_distance_bounded(code.matrix_x, code.matrix_z,
                                    conflict_budget=CONFLICT_BUDGET,
                                    min_weight=lower)
    return {
        "key": row["key"],
        "k": row["k"],
        "d_lower": d,
        "d_upper": d if exact else row["d_upper"],
        "exact": int(exact),
        "seconds": round(time.time() - t0, 2),
        "old_lower": row["d_lower"],
        "old_upper": row["d_upper"],
    }


def close(workers):
    """SAT-certify every escalated form (the 362 already exact are re-proved at no real cost)."""
    rows = list(csv.DictReader(open(KEYS / "escalated_l6m6_w3.csv")))   # the escalation pass as shipped

    # Cheapest first, so a stalled tail never blocks the bulk of the progress. A rung's cost grows
    # about 10x per distance step, and the per-rung budget keeps one hard form from stalling a
    # worker: it is reported as an interval, never dropped.
    rows.sort(key=lambda r: int(r["d_lower"]))
    print(f"{len(rows)} open intervals, {workers} workers", flush=True)

    t0 = time.time()
    done = []
    with mp.Pool(workers) as pool:
        # imap_unordered: rows come back in completion order, so compare files by key.
        for i, res in enumerate(pool.imap_unordered(close_one, rows, chunksize=4), 1):
            done.append(res)
            if i % 100 == 0:
                elapsed = time.time() - t0
                print(f"  {i}/{len(rows)}  {elapsed:.0f}s elapsed, "
                      f"{elapsed / i:.2f}s/code", flush=True)
    out = OUT / "closed_l6m6_w3.csv"
    write_rows(done, out)

    still_open = [r for r in done if not r["exact"]]
    spectrum = {}
    for r in done:
        if r["exact"]:
            spectrum[r["d_lower"]] = spectrum.get(r["d_lower"], 0) + 1
    print(f"\nwrote {out}", flush=True)
    print(f"closed exactly: {len(done) - len(still_open)}/{len(done)}", flush=True)
    print(f"distance spectrum of the closed set: {sorted(spectrum.items())}", flush=True)
    print(f"total wall time: {time.time() - t0:.0f}s", flush=True)
    if still_open:
        print(f"STILL OPEN ({len(still_open)}), reported not dropped:", flush=True)
        for r in still_open[:20]:
            print(f"  {r['key']}  d >= {r['d_lower']}  ({r['seconds']}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the n = 72 answer key, one pass at a time.")
    ap.add_argument("leg", choices=["base", "escalate", "merge", "close"])
    ap.add_argument("--sample", type=int, default=None,
                    help="base: certify only this many random forms (the costing pilot)")
    ap.add_argument("--chunk", type=int, default=0, help="escalate: this task's index")
    ap.add_argument("--chunks", type=int, default=1, help="escalate: number of array tasks")
    ap.add_argument("--workers", type=int, default=4, help="close: worker processes")
    args = ap.parse_args()

    if args.leg == "base":
        base(args.sample)
    elif args.leg == "escalate":
        escalate(args.chunk, args.chunks)
    elif args.leg == "merge":
        merge()
    else:
        close(args.workers)
