"""Table V (App. A.4): the cost of one paid call on one CPU core, from stored time stamps.

n 72   every form of the complete key carries the seconds each pass spent on it (first pass, wider
       sweep, SAT ladder); a form's cost is the sum over the passes that touched it.
n 180  every paid call of the 42 stored search runs stamps the CPU seconds of its two classical bound
       ladders (cpuBound); a certified call also stamps the CPU seconds of its exact ladders (cpu).
       The three stored runs of uniform draws on the exact oracle stamp what an exact score costs.

n 36   no stored run stamps a time, so scripts/time_oracle.py timed the oracle once, single-threaded,
n 300  into data/timing/oracle_times.json: at n 36 every one of the 2,304 canonical forms of the
       complete key, at n 300 the first ten paid calls of a stored uniform-draw run; each call in three
       separate passes, of which a call's cost is the median. The n 36 cell reads "below 0.1 s" only if
       every call is under 0.1 s; the n 300 cell reads "1 to 2 s" only if the median call is.

Reads data/keys, data/n180, data/timing and one data/n300 header; no oracle call; about 1 s on a laptop.
Run from the repo root:
    python -m readouts.oracle_costs
"""

import ast
import csv
import json
import statistics
import time

from gflowqec.paths import DATA

T0 = time.time()
KEYS = DATA / "keys"
N180 = DATA / "n180"

print("Table V, cost per paid call on one CPU core")

# The Instrument column at n 300 is a setting, not a measurement: check it against a stored search run.
head = json.load(open(DATA / "n300" / "mitten_lersearch_pt_C10xS3_b0.35_t0.01-0.15_wC_s0_crn.json"))
assert head["p"] == 0.07 and head["shots"] == 1000

# The two timed sizes: one cost per call, the median of its three passes, in CPU seconds on one core.
timing = json.load(open(DATA / "timing" / "oracle_times.json"))
assert timing["OMP_NUM_THREADS"] == "1"
cost36 = [statistics.median(t) for t in timing["n36"]["cpu_s"]]
cost300 = [statistics.median(t) for t in timing["n300"]["cpu_s"]]
assert len(cost36) == 2304 and len(cost300) == 10

# =============================================================================================
# n 36: every call an n = 36 search can pay, timed.
# =============================================================================================
if max(cost36) < 0.1:
    print("n 36, Z6 x Z3: below 0.1 s")
else:
    print(f"n 36, Z6 x Z3: {max(cost36):.2f} s (slowest call)")

# =============================================================================================
# n 72: merge the three passes as the answer key does (the latest pass wins per form, and every
# merged form is exact), then add up the seconds every pass spent on each form.
# =============================================================================================
passes = {}
for name in ("certified_l6m6_w3.csv", "escalated_l6m6_w3.csv", "closed_l6m6_w3.csv"):
    passes[name] = {ast.literal_eval(r["key"]): r for r in csv.DictReader(open(KEYS / name))}
final = {}
for rows in passes.values():
    final.update(rows)
assert len(final) == 9280 and all(r["exact"] == "1" for r in final.values())

per_code = []
for key in final:
    seconds = 0.0
    for rows in passes.values():
        if key in rows:
            seconds += float(rows[key]["seconds"])
    per_code.append(seconds)
print(f"n 72, Z6 x Z6: {statistics.median(per_code):.2f} s median, {statistics.mean(per_code):.0f} s mean")

# =============================================================================================
# n 180, the bound score: every paid call of the 42 stored runs (3,000 calls each). The certified
# calls among them also paid the exact SAT ladders of both sectors.
# =============================================================================================
run_files = sorted(N180.glob("r174_*_cert_s[0-9].json"))
assert len(run_files) == 42
bound, certified = [], []
for f in run_files:
    d = json.load(open(f))
    assert d["complete"]
    bound += d["cpuBound"]
    for i, c in enumerate(d["cert"]):
        if c is not None:
            certified.append(d["cpu"][i])
assert len(bound) == 126000 and len(certified) == 6893

# n 180, the exact score of a uniform draw: three runs of 3,000 uniform draws on the exact oracle
# at gamma = 0. Every call was paid (no cache hits), so every call carries a stamp.
exact = []
for seed in range(3):
    d = json.load(open(N180 / f"r174_random_C6xS3_e1Z_b3000_s{seed}.json"))
    assert d["complete"] and d["oracle"] == "exact" and d["hits"] == 0
    exact += d["cpu"]
assert len(exact) == 9000

print(f"n 180, Z6 x S3: {statistics.median(bound):.1f} s median, {statistics.mean(bound):.1f} s mean;"
      f" exact {statistics.median(exact):.0f} s (uniform draw),"
      f" about {statistics.median(certified) / 60:.0f} min (certified)")

# =============================================================================================
# n 300: ten paid calls of a stored uniform-draw run, timed.
# =============================================================================================
if 1 <= statistics.median(cost300) <= 2:
    print("n 300, Z10 x S3: 1 to 2 s")
else:
    print(f"n 300, Z10 x S3: {statistics.median(cost300):.1f} s median")

print(f"basis: n 180 bounds = {len(bound):,} bound stamps of the {len(run_files)} stored runs,"
      f" median {statistics.median(bound):.3f} s, mean {statistics.mean(bound):.3f} s")
print(f"basis: n 180 certified = {len(certified):,} certifications, median {statistics.median(certified):.1f} s;"
      f" uniform exact = {len(exact):,} stamps, median {statistics.median(exact):.2f} s")
print(f"basis: n 36 = {len(cost36):,} timed calls, median {1000 * statistics.median(cost36):.1f} ms,"
      f" mean {1000 * statistics.mean(cost36):.1f} ms, slowest {1000 * max(cost36):.0f} ms;"
      f" n 300 = {len(cost300)} timed calls, median {statistics.median(cost300):.2f} s,"
      f" range {min(cost300):.2f} to {max(cost300):.2f} s")
print(f"basis: timed on {timing['machine']}, {timing['date']}, OMP_NUM_THREADS {timing['OMP_NUM_THREADS']}")
print(f"timed: wall {time.time() - T0:.0f} s")
