"""The statistical procedures appendix: the seed-count row of its table and the threshold of the packed-yield
paragraph above it.

Seed count    Under a fair coin per seed, ten of ten has probability 0.001 and at least eight of ten 0.055.
Packed yield  Packed yield counted paid codes whose rate was at most 0.003567, the first percentile of
              26,281 earlier uniform draws.

The first part is arithmetic. The second reads the uniform dataset at n = 300 (data/n300, 40 files of
1,000 draws, each re-estimated at p = 0.07: "ler_pq" is the per-qubit logical error rate). No oracle call;
about 34 s on a laptop. Run from the repo root:
    python -m readouts.stats_appendix
"""

import glob
import json
from math import comb

import numpy as np

from gflowqec import counts
from gflowqec.codes import generates_g
from gflowqec.paths import DATA


def at_least(j, n):
    """P(at least j of n seeds lead) when each seed is a fair coin flip."""
    return sum(comb(n, i) for i in range(j, n + 1)) / 2 ** n


# ---------------------------------------------------------------------------------------------
# The statistics table, the seed count, for ten seeds (the count the paper's seed comparisons use). The paper
# prints three decimals.
# ---------------------------------------------------------------------------------------------
p10of10 = at_least(10, 10)           # 1 / 1,024
p8of10 = at_least(8, 10)             # 56 / 1,024
print(f"Table C.2 seed count under a fair coin: 10 of 10 {p10of10:.3f}; at least 8 of 10 {p8of10:.3f}")

# ---------------------------------------------------------------------------------------------
# The packed-yield paragraph, tau_1%. The threshold is defined on the generating draws only: a draw
# whose A side or B side lives in a proper subgroup is not a genuine member of the family
# (codes.generates_g), so it is dropped before the quantile is taken. tau_1% is the 1st percentile of
# ler_pq over the rest (numpy's default linear interpolation). The value was fixed before the
# n = 300 searches and is the constant counts.TAU1 the readouts use; this check stops if the data
# disagree with it.
# ---------------------------------------------------------------------------------------------
files = sorted(glob.glob(str(DATA / "n300" / "mitten_ler_data_C10xS3_p0.07_s*_t*.json")))
assert len(files) == 40, f"expected the 40 files of the uniform dataset, found {len(files)}"

rates = []
n_draws = 0
for f in files:
    for row in json.load(open(f))["rows"]:
        n_draws += 1
        if generates_g("C10xS3", row["exps"]):
            rates.append(row["ler_pq"])

tau = np.quantile(rates, 0.01)
print(f"basis: {n_draws:,} uniform draws in {len(files)} files, {len(rates):,} of them generate the group")
assert round(tau, 6) == counts.TAU1, f"tau_1% from the data is {tau:.6f}, the stored constant {counts.TAU1}"
print(f"App. C.2: tau_1% {tau:.6f}, the first percentile of {len(rates):,} generating uniform draws")
