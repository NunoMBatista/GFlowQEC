"""Time paid oracle calls at n = 36 and n = 300 on one CPU core (two cells of Table V, App. A.4).

The stored runs at n = 72 and n = 180 stamp the seconds of every paid call, so readouts/oracle_costs.py
reads those cells from data/. Nothing at n = 36 or n = 300 stores a time, so this script measures them
once and writes data/timing/oracle_times.json, which the readout prints from.

  n 36   codes.score_code (exact n, k, d by enumeration) on every one of the 2,304 canonical forms of
         the complete key, data/keys/candidates_l6m3_w3.csv: every call an n = 36 search can pay,
         the 1,600 forms with k = 0 included (their call is paid like any other). Each k is checked
         against the key, and each d of a k > 0 form against data/keys/datasheet_l6m3_w3.csv.
  n 300  oracle.ler_score (BP-OSD, p = 0.07, 1,000 shots) on the first ten paid calls of the stored
         uniform-draw run at seed 0, data/n300/mitten_lersearch_random_C10xS3_b0.35_t0.05-0.67_w_s0.json,
         with that run's noise stream (run seed 0, one stream per code). Each value is checked against
         the stored one, so the timed call is the call the run paid.

Every call is timed with time.process_time (CPU seconds of this process) and time.perf_counter (wall)
in three separate passes over the whole set; the readout keeps each call's median over the passes. On a
shared laptop the CPU time of a call jumps for about a second at a time (a k = 4 code that costs 7 ms
measured 0.3 s on three consecutive calls), and such a burst hits a given call in one pass only. One
untimed call per size goes first, so building the group tables and importing the decoders is not
charged to the first timed call. Run single-threaded, from the repo root:
    OMP_NUM_THREADS=1 python -m scripts.time_oracle
Compute: about two minutes on a laptop. The times depend on the machine; the file records which one.
"""

import ast
import csv
import datetime
import json
import os
import platform
import statistics
import time
from importlib.metadata import version

from gflowqec import codes, oracle
from gflowqec.paths import DATA

N36_KEY = DATA / "keys" / "candidates_l6m3_w3.csv"
N36_DATASHEET = DATA / "keys" / "datasheet_l6m3_w3.csv"
N300_RUN = DATA / "n300" / "mitten_lersearch_random_C10xS3_b0.35_t0.05-0.67_w_s0.json"
N300_CALLS = 10
PASSES = 3                                            # each call timed once per pass; the readout keeps the median
OUT_FILE = DATA / "timing" / "oracle_times.json"


def timed(fn, *args, **kwargs):
    """Call fn once; return (result, CPU seconds, wall seconds)."""
    c0, w0 = time.process_time(), time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.process_time() - c0, time.perf_counter() - w0


# =============================================================================================
# n 36: every canonical form of the complete key.
# =============================================================================================
forms = list(csv.DictReader(open(N36_KEY)))
assert len(forms) == 2304
stored_d = {r["key"]: int(r["d"]) for r in csv.DictReader(open(N36_DATASHEET))}
assert len(stored_d) == 704

l, m, a, b = ast.literal_eval(forms[0]["key"])
codes.score_code(l, m, a, b)                          # untimed first call

cpu36 = [[] for _ in forms]      # per form, one CPU time per pass
wall36 = [[] for _ in forms]
for _ in range(PASSES):
    for i, row in enumerate(forms):
        l, m, a, b = ast.literal_eval(row["key"])
        scored, cpu, wall = timed(codes.score_code, l, m, a, b)
        assert scored.k == int(row["k"])
        if scored.k > 0:
            assert scored.d == stored_d[row["key"]]
        cpu36[i].append(cpu)
        wall36[i].append(wall)
print(f"n 36: {len(cpu36)} calls, slowest median {max(statistics.median(t) for t in cpu36):.3f} CPU s")

# =============================================================================================
# n 300: the first paid calls of a stored uniform-draw run, on that run's noise stream.
# =============================================================================================
run = json.load(open(N300_RUN))
assert run["arm"] == "random" and run["seed"] == 0 and run["p"] == 0.07 and run["shots"] == 1000

oracle.ler_score("C10xS3", run["rep"][0], run["p"], run["shots"], run_seed=run["seed"])   # untimed first call

cpu300 = [[] for _ in range(N300_CALLS)]      # per call, one CPU time per pass
wall300 = [[] for _ in range(N300_CALLS)]
for _ in range(PASSES):
    for i in range(N300_CALLS):
        out, cpu, wall = timed(oracle.ler_score, "C10xS3", run["rep"][i], run["p"], run["shots"],
                               run_seed=run["seed"])
        assert out["ler_pq"] == run["ler_pq"][i] and out["k"] == run["k"][i]
        cpu300[i].append(cpu)
        wall300[i].append(wall)
for i in range(N300_CALLS):
    print(f"n 300 call {i}: k {run['k'][i]}, CPU s per pass " + " ".join(f"{t:.2f}" for t in cpu300[i]))

OUT_FILE.parent.mkdir(exist_ok=True)
json.dump({
    "made_by": "python -m scripts.time_oracle",
    "date": datetime.date.today().isoformat(),
    "machine": platform.platform(),
    "python": platform.python_version(),
    "ldpc": version("ldpc"),
    "qldpc": version("qldpc"),
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    "passes": PASSES,
    "n36": {"call": "codes.score_code(l, m, a, b)", "source": str(N36_KEY.relative_to(DATA.parent)),
            "calls": len(cpu36), "cpu_s": cpu36, "wall_s": wall36},
    "n300": {"call": "oracle.ler_score('C10xS3', rep, 0.07, 1000, run_seed=0)",
             "source": str(N300_RUN.relative_to(DATA.parent)),
             "calls": len(cpu300), "cpu_s": cpu300, "wall_s": wall300},
}, open(OUT_FILE, "w"), indent=1)
print(f"wrote {OUT_FILE}")
