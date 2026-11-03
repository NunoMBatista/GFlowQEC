"""Proportional sampling at n = 72: does the earlier sampler draw codes in proportion to their reward?

The earlier sampler (the n = 36 multilayer perceptron, width 128, trained by trajectory balance
with beta = 1 on the whole family) was trained on the n = 72 two-block family over Z_6 x Z_6, three seeds.
The recipe is the n = 36 one, 24,000 steps at exploration 0.1 then 12,000 at 0.02, with the second stage run
on for 324,000 more steps at 0.02: 36,000 + 324,000 = 360,000 steps in all (the 'x324000' in the file names).
Every 12,000 steps the run drew 50,000 codes and stored the share of draws at or above each merit level.

What this readout does, with no oracle call and no training:
  1. Rebuilds the exact target from the answer key in data/keys, as scripts/n36_calibration.py does at
     n = 36. Each canonical form carries its raw-pair count (n_pairs in candidates_l6m6_w3.csv) and its exact
     distance (the certified, escalated and closed passes merged, the latest pass winning, as readouts.keys
     does). A raw pair has reward max(merit, 0.01), merit = k d^2 / 72 (0 when k = 0). The target mass of a
     merit level is its share of the total reward. The exact log Z is log(total reward) + 2 log 3!: every
     pair is reached by (3!)^2 orderings of its six monomials, a constant trajectory balance folds into log Z.
  2. Turns each run's stored tail shares at 360,000 and at 36,000 steps back into per-level shares of the
     50,000 draws, and recomputes from them the level total variation (TV) against the target and the
     top-level mass ratio (sampled share of the highest merit level over its target share).
  3. Prints the uniform sampler's level TV (from the key alone) and the learned log Z, which is read from the
     run files: the trained networks are not shipped.
  4. Reads the prioritised-replay runs (36,000 steps, the same recipe with a replay batch of 128 stored codes,
     half of them drawn from the top merit tenth of the store) and their plain twins. Those runs were graded by
     the sampler's exact probability of every code rather than by draws, and their files keep only the final
     numbers: the top mass is read and divided by the target's top share rebuilt here; the level TV is read.

Data: data/keys (the n = 72 key) and data/n72 (the three 360,000-step runs, the six replay runs).
About 2 s on a laptop. Run from the repo root:
    python -m readouts.cal72
"""

import ast
import csv
import hashlib
import json
import math
import time

from gflowqec.paths import DATA

T0 = time.time()
KEYS = DATA / "keys"
N72 = DATA / "n72"

# The three seeds of the long runs, and their files (one per seed, every 12,000-step grading inside).
SEEDS = (0, 1, 2)
LONG_RUN = N72 / "gfn_mass_tail_l6m6_s{seed}_x324000.json"
# The replay comparison at 36,000 steps: the plain recipe and the prioritised-replay arm, per seed.
REPLAY_RUN = N72 / "prt_m1_l6m6_{arm}_s{seed}.json"

N_QUBITS = 72
WEIGHT = 3                  # three monomials per polynomial
BETA, FLOOR = 1.0, 0.01     # reward max(merit, 0.01)^1, as at n = 36
SHORT, LONG = 36_000, 360_000


def md5_of(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def two(values):
    """'a to b' for the smallest and largest of values, two decimals."""
    return f"{min(values):.2f} to {max(values):.2f}"


# =============================================================================================
# 1. The exact target from the answer key
# =============================================================================================

# Every canonical form of the space (k = 0 included), with its raw-pair count.
n_pairs = {}
k_of = {}
for r in csv.DictReader(open(KEYS / "candidates_l6m6_w3.csv")):
    key = ast.literal_eval(r["key"])
    n_pairs[key] = int(r["n_pairs"])
    k_of[key] = int(r["k"])

# Exact distances of the k > 0 forms: three passes, the latest wins per form, every merged row exact.
final = {}
for name in ("certified_l6m6_w3.csv", "escalated_l6m6_w3.csv", "closed_l6m6_w3.csv"):
    for r in csv.DictReader(open(KEYS / name)):
        final[ast.literal_eval(r["key"])] = r
assert all(r["exact"] == "1" for r in final.values())
assert set(final) == {key for key, k in k_of.items() if k > 0}

# Reward mass and raw-pair count per merit level (levels rounded to 3 decimals, as the runs bin them).
reward_at = {}
pairs_at = {}
for key, count in n_pairs.items():
    k = k_of[key]
    merit = k * int(final[key]["d_lower"]) ** 2 / N_QUBITS if k > 0 else 0.0
    level = round(merit, 3)
    reward_at[level] = reward_at.get(level, 0.0) + count * max(merit, FLOOR) ** BETA
    pairs_at[level] = pairs_at.get(level, 0) + count
levels = sorted(reward_at)
top = levels[-1]
z = sum(reward_at.values())
total_pairs = sum(pairs_at.values())
assert total_pairs == math.comb(36, WEIGHT) ** 2      # every ordered pair of 3-subsets of the 36 cells

target = {lv: reward_at[lv] / z for lv in levels}
uniform = {lv: pairs_at[lv] / total_pairs for lv in levels}
log_z_exact = math.log(z) + 2 * math.log(math.factorial(WEIGHT))


def level_tv(shares):
    """Total variation between per-level shares and the target."""
    return sum(abs(shares[lv] - target[lv]) for lv in levels) / 2


uniform_tv = level_tv(uniform)


# =============================================================================================
# 2. The long runs: per-level draw shares back from the stored tails
# =============================================================================================

def draw_shares(cell):
    """Per-level shares of the cell's draws, from its stored tail (share at or above each level)."""
    tail = {float(lv): share for lv, share in cell["tail"].items()}
    assert sorted(tail) == levels
    shares = {}
    for i, lv in enumerate(levels):
        above = tail[levels[i + 1]] if i + 1 < len(levels) else 0.0
        shares[lv] = tail[lv] - above
    return shares


key_md5 = {"candidates": md5_of(KEYS / "candidates_l6m6_w3.csv"),
           "certified": md5_of(KEYS / "certified_l6m6_w3.csv"),
           "closed": md5_of(KEYS / "closed_l6m6_w3.csv")}

tv_long, ratio_long, ratio_short, log_z_learned = [], [], [], []
for seed in SEEDS:
    run = json.load(open(str(LONG_RUN).format(seed=seed)))
    settings = run["settings"]
    # The run read the same key files as the ones shipped here, and used the n = 36 recipe.
    assert settings["key_md5s"] == key_md5
    assert settings["hidden"] == 128 and settings["batch"] == 128
    assert settings["beta"] == BETA and settings["floor"] == FLOOR
    assert settings["draws_graded"] == 50_000
    # 360,000 steps = the two planned stages (24,000 + 12,000) + the 324,000-step extension.
    steps = sum(stage[0] for stage in settings["stages"]) + settings["extra"][0]
    assert steps == run["steps_done"] == LONG and run["complete"] and not run["diverged"]
    # The run's own exact log Z is the one rebuilt here.
    assert abs(run["log_z_exact"] - log_z_exact) < 1e-9

    long_cell = run["checkpoints"][str(LONG)]
    short_cell = run["checkpoints"][str(SHORT)]
    assert long_cell["steps"] == LONG and short_cell["steps"] == SHORT

    shares = draw_shares(long_cell)
    tv_long.append(level_tv(shares))
    ratio_long.append(shares[top] / target[top])
    ratio_short.append(draw_shares(short_cell)[top] / target[top])
    log_z_learned.append(long_cell["log_z_learned"])
    # The recomputation agrees with what the run wrote.
    assert abs(tv_long[-1] - long_cell["level_tv"]) < 1e-9
    assert abs(ratio_long[-1] - long_cell["top_ratio"]) < 1e-9

log_z_gap = [abs(v - log_z_exact) for v in log_z_learned]


# =============================================================================================
# 3. The replay runs at 36,000 steps
# =============================================================================================

replay = {}
for arm in ("plain", "prt"):
    rows = []
    for seed in SEEDS:
        run = json.load(open(str(REPLAY_RUN).format(arm=arm, seed=seed)))
        assert run["arena"] == "l6m6" and run["arm"] == arm and run["seed"] == seed and run["steps"] == SHORT
        assert abs(run["target_top_mass"] - target[top]) < 1e-12
        # The file's 'exact_log_z' is the key-level normaliser, without the 2 log 3! orderings.
        assert abs(run["exact_log_z"] + 2 * math.log(math.factorial(WEIGHT)) - log_z_exact) < 1e-9
        rows.append({"ratio": run["top_mass"] / target[top], "tv": run["level_tv"]})
    replay[arm] = rows


# =============================================================================================
# Print
# =============================================================================================

print("App. D.2: proportional sampling at n = 72, the earlier sampler, three seeds")
print(f"App. D.2: trained for {LONG:,} steps")
print(f"App. D.2: total variation from its target {min(tv_long):.3f} to {max(tv_long):.3f};"
      f" uniform sampler {uniform_tv:.2f}")
# "within": the smallest three-decimal bound at or above the largest gap
print(f"App. D.2: learned log Z within {math.ceil(max(log_z_gap) * 1000) / 1000:.3f} of the exact {log_z_exact:.4f}")
print(f"App. D.2: mass on the highest score value, over its target: {two(ratio_long)}")
# The two partial fixes, per seed: top-mass ratio before -> after (1 would close the gap).
print(f"App. D.2: ten times longer training ({SHORT:,} to {LONG:,} steps): top-mass ratio "
      + ", ".join(f"{a:.2f} -> {b:.2f}" for a, b in zip(ratio_short, ratio_long)) + " (seeds 0, 1, 2)")
print(f"App. D.2: replaying high-scoring codes more often ({SHORT:,} steps): top-mass ratio "
      + ", ".join(f"{p['ratio']:.2f} -> {q['ratio']:.2f}" for p, q in zip(replay["plain"], replay["prt"]))
      + " (seeds 0, 1, 2)")

print(f"basis: target from the key: {len(n_pairs):,} forms, {total_pairs:,} raw pairs, {len(levels)} merit levels,"
      f" top level {top} with target share {target[top]:.6f}")
print("basis: level TV at 360,000 steps per seed " + ", ".join(f"{v:.4f}" for v in tv_long)
      + f"; uniform {uniform_tv:.4f}; recomputed from the stored per-level shares of 50,000 draws, not from the networks")
print("basis: learned log Z per seed " + ", ".join(f"{v:.4f}" for v in log_z_learned)
      + " (read); gaps " + ", ".join(f"{g:.4f}" for g in log_z_gap))
print("basis: top-mass ratio at 36,000 / 360,000 steps per seed "
      + ", ".join(f"{a:.4f} / {b:.4f}" for a, b in zip(ratio_short, ratio_long)))
print("basis: replay runs graded by exact code probabilities, final numbers only: top-mass ratio plain "
      + ", ".join(f"{p['ratio']:.4f}" for p in replay["plain"]) + ", prioritised "
      + ", ".join(f"{p['ratio']:.4f}" for p in replay["prt"]) + "; level TV (read) plain "
      + ", ".join(f"{p['tv']:.4f}" for p in replay["plain"]) + ", prioritised "
      + ", ".join(f"{p['tv']:.4f}" for p in replay["prt"]))
print(f"timed: wall {time.time() - T0:.0f} s")
