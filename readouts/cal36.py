"""Proportional sampling at n = 36: does the earlier sampler draw codes in proportion to their reward?

The earlier sampler (a multilayer perceptron, width 128, trained by trajectory balance with beta = 1 on the
whole n = 36 family over Z_6 x Z_3) was graded against the exact target law, P(raw pair) ~ max(merit, 0.01),
binned by the family's merit levels. scripts/n36_calibration.py trains it and writes the files read here;
this readout reads them back, with no oracle call and no training.

  1. The sampler leg (one training, 10,000 draws) stored the target share and the sampled share of every merit
     level in data/keys/gfn_calibration_l6m3.csv. The level total variation (TV) is half the sum over levels
     of the absolute difference between the two shares.
  2. The tail leg (three retrainings, seeds 0, 1, 2, 50,000 draws each) stored, per seed, the share of draws at
     or above every merit level, the exact target's tail, a uniform proposal's tail (the exact raw-pair
     fractions) and the level TV, in data/keys/gfn_mass_tail_l6m3_s{seed}.json. The uniform sampler's level TV
     is recomputed here from its stored tail; the retrainings' level TV and their top-mass ratio (sampled share
     of the highest merit level over its target share) are read.

Data: data/keys (the four files above). Under a second on a laptop. Run from the repo root:
    python -m readouts.cal36
"""

import csv
import json
import time

from gflowqec.paths import DATA

T0 = time.time()
KEYS = DATA / "keys"
SEEDS = (0, 1, 2)
DRAWS = 10_000          # draws of the sampler leg


def level_shares(tail):
    """Undo a tail (share at or above each level) into the share at each level, levels in ascending order."""
    levels = sorted(tail, key=float)
    shares = {}
    for i, lv in enumerate(levels):
        above = tail[levels[i + 1]] if i + 1 < len(levels) else 0.0
        shares[lv] = tail[lv] - above
    return shares


def level_tv(p, q):
    return sum(abs(p[lv] - q[lv]) for lv in p) / 2


# ------------------------------------------------------------------ 1. the sampler leg, 10,000 draws
target, sampled = {}, {}
for row in csv.DictReader(open(KEYS / "gfn_calibration_l6m3.csv")):
    target[row["merit_level"]] = float(row["target_probability"])
    sampled[row["merit_level"]] = float(row["sampled_frequency"])
# every sampled share is a whole number of the 10,000 draws
assert all(abs(q * DRAWS - round(q * DRAWS)) < 1e-6 for q in sampled.values())
assert round(sum(q * DRAWS for q in sampled.values())) == DRAWS
tv_sampler = level_tv(target, sampled)

# ------------------------------------------------------------------ 2. the tail leg, three retrainings
runs = [json.load(open(KEYS / f"gfn_mass_tail_l6m3_s{s}.json")) for s in SEEDS]
target_tail = runs[0]["target_tail"]
uniform_tail = runs[0]["uniform"]["tail"]
# the three files and the sampler leg share one exact target
for run in runs:
    assert run["target_tail"] == target_tail and run["uniform"]["tail"] == uniform_tail
exact = level_shares(target_tail)
assert all(abs(exact[lv] - target[lv]) < 1e-9 for lv in target), "the two legs hold different targets"
tv_uniform = level_tv(exact, level_shares(uniform_tail))
tv_retrain = [run["sampler"]["level_tv"] for run in runs]
ratio = [run["sampler"]["top_ratio"] for run in runs]
log_z = [run["sampler"]["log_z_learned"] for run in runs]
log_z_exact = runs[0]["sampler"]["log_z_exact"]

print("App. D.2: proportional sampling at n = 36, the earlier sampler")
print(f"App. D.2: binned by the family's {len(target)} score values, the sampler's {DRAWS:,} draws lie at "
      f"{tv_sampler:.3f} from the target, against {tv_uniform:.2f} for a uniform sampler")
print(f"App. D.2: three retrainings gave {min(tv_retrain):.3f} to {max(tv_retrain):.3f}")
print(f"App. D.2: mass on the highest score value, over its target: {min(ratio):.2f} to {max(ratio):.2f} "
      f"({1 - max(ratio):.2f} to {1 - min(ratio):.2f} too little)")
print("basis: sampler leg per level (target, sampled): "
      + ", ".join(f"{lv} {target[lv]:.4f} {sampled[lv]:.4f}" for lv in sorted(target, key=float)))
print("basis: level TV per retraining " + ", ".join(f"{v:.4f}" for v in tv_retrain) + f"; uniform {tv_uniform:.4f}")
print("basis: top-mass ratio per retraining " + ", ".join(f"{v:.4f}" for v in ratio))
print("basis: learned log Z per retraining " + ", ".join(f"{v:.4f}" for v in log_z) + f" against the exact {log_z_exact:.4f}")
print(f"timed: wall {time.time() - T0:.0f} s")
