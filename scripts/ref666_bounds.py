"""The classical bounds d_A, d_B of the 666 codes of the n = 180 reference sample.

The appendix on search spaces, oracle and distances states that on every reference code the quantum
distances sit under two classical ones: d_Z <= d_A and d_X <= d_B. Here A = [L(a0) | L(a1)] and
B = [L(b0) | L(b1)] are the two 36 x 72 lifted bases of the code (certify.lifted_bases), and d_A, d_B
are the minimum weights of their kernels, computed by the same SAT bound ladder the n = 180 oracle pays
before any exact ladder (certify.bound_ladder: 1,000,000 conflicts per rung, cap 12).

Naming: the reference sample stores the per-sector distances under the keys dX and dZ. The key dX is
the paper's d_Z and the key dZ is the paper's d_X, so the file pairs key dX with d_A and key dZ with
d_B.

Regenerates data/n180/ref666_bounds.csv. Output: results/ref666_bounds.csv, one row per code in the
order of oracle.reference_sample():
    index    the code's index i in the reference sample
    d_Z      its exact Z-sector distance (key dX)
    d_A      the bound ladder's value on A; exactA = 1 when the ladder closed (a proven minimum)
    d_X      its exact X-sector distance (key dZ)
    d_B      the bound ladder's value on B; exactB likewise
A ladder that does not close stops at its undecided rung w, which is then a lower bound on d_A (or
d_B): every rung below w was proven empty.

Run from the repo root:
    python -m scripts.ref666_bounds

Compute (laptop, one core): about 3 minutes. Deterministic; the file carries no timing column, and its
md5 sum is deaa7b646968486e3be8e3bee5ecc666. readouts/family_counts.py reads the shipped copy in data/n180/.
"""

import csv
import time

import numpy as np

from gflowqec import certify, codes, oracle
from gflowqec.paths import OUT

GROUP = "C6xS3"   # Z_6 x S_3, order 36: the lifted bases are 36 x 72
OUT_PATH = OUT / "ref666_bounds.csv"

t0 = time.time()
ref = oracle.reference_sample()
rows = []
for r in ref:
    A, B = certify.lifted_bases(codes.build(GROUP, np.array(r["exps"])), 36)
    dA, exA, *_ = certify.bound_ladder(A)
    dB, exB, *_ = certify.bound_ladder(B)
    rows.append((r["i"], r["dX"], dA, int(exA), r["dZ"], dB, int(exB)))

with open(OUT_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "d_Z", "d_A", "exactA", "d_X", "d_B", "exactB"])
    writer.writerows(rows)

# The summary the appendix sentence rests on: how many codes satisfy both inequalities, and how often
# the classical bound is attained exactly.
held = sum(1 for o in rows if o[1] <= o[2] and o[4] <= o[5])
both_equal = sum(1 for o in rows if o[1] == o[2] and o[4] == o[5])
inexact = sum(1 for o in rows if not (o[3] and o[6]))
print(f"{len(rows)} codes: d_Z <= d_A and d_X <= d_B on {held}; both equal on {both_equal}; "
      f"inexact ladders {inexact}")
print(f"wrote {OUT_PATH} in {time.time() - t0:.0f} s")
