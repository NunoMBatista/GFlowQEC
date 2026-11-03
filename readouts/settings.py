"""Two measured settings that no table and no other readout prints.

1. The reference rate y* of the n = 300 grading. It is fixed per seed as the exact median of the
   re-estimated logical error rates of five earlier reference runs at n = 300, pooled (counts.y_star).
   The text gives one value, "about 0.00424": the mean of the ten per-seed values to five decimals.
   (A single seed's value would not do: they range from 0.004228 to 0.004245.)

2. The matched screen's training length at n = 180 (the networks table), as a share of the policy's gradient updates.
   The screen's step count was chosen on a held-out tenth of the stored forms and is stored in
   data/n180/screen180match_choice.json; the policy the paper reports trains for policy.SCALE_STEPS["x20"].

Reads data/n300 and one data/n180 file; no oracle call; about 4 s on a laptop. Run from the repo root:
    python -m readouts.settings
"""

import json

import numpy as np

from gflowqec import counts, policy
from gflowqec.paths import DATA

# 1. y*: load the five reference cells (seeds 0-5 and 7-10) and take the pooled median per seed.
runs = counts.load_visited(list(counts.REFERENCE_CELLS))
y_stars = [counts.y_star(runs, seed) for seed in counts.SEEDS]
print(f"App. E.1: y* about {np.mean(y_stars):.5f}, the median re-estimated rate of "
      f"{len(counts.REFERENCE_CELLS)} reference runs")

# 2. The screen's chosen step count over the policy's step count.
choice = json.load(open(DATA / "n180" / "screen180match_choice.json"))
share = choice["step"] / policy.SCALE_STEPS["x20"]
print(f"Not in the paper: the screen's training length (Table B.1 prints its updates): "
      f"{share:.1f} of the policy's updates")

print("basis: y* per seed " + ", ".join(f"{q:.6f}" for q in y_stars)
      + f"; screen step {choice['step']:,} ({choice['config']}) / policy steps {policy.SCALE_STEPS['x20']:,}"
      + f" = {share:.4f}")
