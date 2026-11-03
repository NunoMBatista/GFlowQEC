"""The protocol appendix (its Budgets and Measures paragraphs): the two measured numbers of its text.

Budgets   Ten runs of Bayesian optimisation at n = 300 have a budget of 200 calls.
Measures  The reference rate y* is set per seed as the median of the re-estimated rates pooled over the
          five reference runs of that seed; the ten values lie between 0.004228 and 0.004245.

Reads data/n300 only; no oracle call; about 3 s on a laptop. Run from the repo root:
    python -m readouts.protocol
"""

import json

from gflowqec import counts
from gflowqec.paths import DATA

N300 = DATA / "n300"

# ---------------------------------------------------------------------------------------------
# Budgets. The ten BO runs at 200 calls are seeds 0-5 and 7-10 of the one-hot BO arm with common random
# numbers. The seed 6 file belongs to the tuning seed, so it is not read.
# A run counts only if it was given 200 calls ("budget") and paid all of them ("calls").
# ---------------------------------------------------------------------------------------------
n_bo200 = 0
for seed in counts.SEEDS:
    d = json.load(open(N300 / f"mitten_lersearch_bo_C10xS3_b0.35_t0.05-0.67_wC_s{seed}_bo-onehot.json"))
    if d["budget"] == 200 and d["calls"] == 200:
        n_bo200 += 1
print(f"App. C: BO runs at n = 300 with a budget of 200 calls: {n_bo200}")

# ---------------------------------------------------------------------------------------------
# Measures. y* per seed. The five reference cells (counts.REFERENCE_CELLS) are GFlowQEC-online at depth 31, its
# two depth-31 variants, local search alone at depth 31 and tuned PT. Each cell's visited file holds
# 800 visits sampled from that seed's search run, each re-estimated on 4,000 fresh shots ("ler_pq").
# y* is the exact median of the 5 x 800 pooled values, never rounded (counts.y_star).
# counts.load_visited stops if any visit leg is short or any search run is short of its budget.
# ---------------------------------------------------------------------------------------------
runs = counts.load_visited(counts.REFERENCE_CELLS)
ys = [counts.y_star(runs, seed) for seed in counts.SEEDS]
print("basis: y* per seed 0-5 and 7-10 = " + ", ".join(f"{y:.6f}" for y in ys))
print(f"App. C: y* per seed between {min(ys):.6f} and {max(ys):.6f}")
