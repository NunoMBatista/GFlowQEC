"""App. C.2 (statistical procedures): the seed-count row of Table VIII and the threshold of the packed-yield
paragraph above it.

Seed count    Under a fair coin per seed, at least five of six has probability 0.109, six of six 0.016,
              at least four of six 0.344, at least two of three at each of two biases 0.25.
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
# Table VIII, the seed count. The last case is two independent biases with three seeds each, and
# at least two of three must lead at both: (4 / 8) ** 2. The paper prints three decimals, and two
# for the last value.
# ---------------------------------------------------------------------------------------------
p5of6 = at_least(5, 6)               # 7 / 64
p6of6 = at_least(6, 6)               # 1 / 64
p4of6 = at_least(4, 6)               # 22 / 64
p2of3_twice = at_least(2, 3) ** 2    # (4 / 8) ** 2
print(f"Table VIII seed count under a fair coin: at least 5 of 6 {p5of6:.3f}; 6 of 6 {p6of6:.3f}; "
      f"at least 4 of 6 {p4of6:.3f}; at least 2 of 3 at each of two biases {p2of3_twice:.2f}")

# ---------------------------------------------------------------------------------------------
# The packed-yield paragraph, tau_1%. The threshold is defined on the generating draws only: a draw
# whose A side or B side lives in a proper subgroup is not a genuine member of the family
# (codes.generates_g), so it is dropped before the quantile is taken. tau_1% is the 1st percentile of
# ler_pq over the rest (numpy's default linear interpolation). The value was registered before the
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
assert round(tau, 6) == counts.TAU1, f"tau_1% from the data is {tau:.6f}, the registered constant {counts.TAU1}"
print(f"App. C.2: tau_1% {tau:.6f}, the first percentile of {len(rates):,} generating uniform draws")
