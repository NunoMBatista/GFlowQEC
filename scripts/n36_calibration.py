"""The n = 36 calibration: does the trained sampler draw codes in proportion to their reward?

At n = 36 the whole space is enumerated (data/keys/landscape_l6m3_w3.csv, written by
scripts/n36_key.py), so the exact target law is known in closed form and every number the
sampler produces can be graded against it. Regenerates the n = 36 numbers
of the appendix on proportional sampling at 36 and 72 qubits (readouts/cal36.py prints them from the stored files):

  leg `sampler`: trains the trajectory-balance sampler (gflowqec/policy.py, GFN) on the
      exact merit oracle with the whole space as budget, draws 10,000 codes, and compares
      the share of draws at each of the eight merit levels with the exact target
      P(raw pair) ~ max(merit, 0.01)^1. Prints the total variation distance (the paper's
      0.089) and the learned log Z against the exact one; writes
      results/gfn_calibration_l6m3.csv (merit_level, target_probability, sampled_frequency).
  leg `tail --seed s`: the same recipe retrained from scratch per seed (the paper's three
      retrainings, seeds 0, 1, 2), checkpointed to results/gfn_calib_l6m3_s{s}.pt, then
      50,000 draws used to predict the mass of the space at or above every merit level.
      Prints the level TV (0.092 to 0.100 over the three seeds) and the top-family mass
      ratio (0.664 at seed 0: "about a third too little mass on the highest score value");
      writes results/gfn_mass_tail_l6m3_s{s}.json. A re-run reloads the checkpoint instead
      of retraining.
  leg `tail --target-only`: the exact target and two nulls from the answer key alone, no
      sampler: a uniform proposal over raw pairs (the paper's 0.72) and a point mass on the
      top family. Prints only.

Run from the repo root:
    python -m scripts.n36_calibration sampler
    python -m scripts.n36_calibration tail --target-only
    python -m scripts.n36_calibration tail --seed 0      # then --seed 1, --seed 2

Compute (laptop CPU): 36,000 training steps take about 642 s (seed 0); the sampler leg is
the same training plus 10,000 draws; --target-only takes seconds (reading the key).

Reproducibility. The checkpoints of the three tail runs ship in data/models/ for inspection;
a fresh clone retrains into results/. Training is seeded (torch.manual_seed inside GFN), but
a retrained network matches a shipped checkpoint tensor for tensor only on the same torch
build. The 0.089 run saved no checkpoint, and a re-run of its recipe is not expected to give
0.089 to the digit (one re-run read 0.092).
"""

import argparse
import csv
import json
import math
import time
from collections import Counter

import numpy as np
import torch
from scipy.stats import spearmanr

from gflowqec.codes import canonical_key, load_landscape
from gflowqec.oracle import Oracle
from gflowqec.paths import DATA, OUT
from gflowqec.policy import GFN

# The arena: two weight-3 polynomials A, B over the grid Z_6 x Z_3, n = 2 * 6 * 3 = 36 qubits.
L, M, WEIGHT = 6, 3, 3
# The target law: P(raw pair) ~ R = max(merit, FLOOR)^BETA. The floor keeps log R finite on the
# merit-0 codes (k = 0, no logical qubit), which are 73.7 % of the 665,856 raw pairs.
BETA, FLOOR = 1.0, 0.01
N_SAMPLES = 50_000      # draws graded by the tail leg
ANSWER_KEY = DATA / "keys" / f"landscape_l{L}m{M}_w{WEIGHT}.csv"   # every raw pair with exact n, k, d, merit, key


# =============================================================================================
# Leg `sampler`: level-by-level match of 10,000 draws against the exact target
# =============================================================================================

def sampler_leg():
    rows = load_landscape(ANSWER_KEY)

    # Evaluation lookups. This is the answer key, free to consult when GRADING; the sampler itself
    # only ever sees merits through the oracle. "Top" = the space's best merit level.
    top_merit = max(r.merit for r in rows)
    merit_of = {}
    for r in rows:
        merit_of[r.key] = r.merit
    top_keys = set()
    for r in rows:
        if abs(r.merit - top_merit) < 1e-9:
            top_keys.add(r.key)
    # The share of raw pairs on the top level: what a uniform sampler would hit.
    top_pairs = 0
    for r in rows:
        if r.key in top_keys:
            top_pairs += 1
    uniform_top_pct = 100 * top_pairs / len(rows)

    print(f"space l={L} m={M}: {len(rows):,} pairs, {len(merit_of):,} distinct codes, "
          f"top merit {top_merit:.2f} in {len(top_keys)} classes "
          f"({uniform_top_pct:.2f}% of raw pairs)\n")

    distribution_match(rows, merit_of)


def distribution_match(rows, merit_of):
    """Budget = the whole space, beta = 1. With nothing hidden, does the trained sampler reproduce
    the exact target distribution P(code) ~ R(code)? This validates the sampler; no search
    method is being compared."""
    print("=== distribution match (budget = whole space, beta = 1) ===")
    distinct_codes = len(merit_of)
    oracle = Oracle(L, M, budget=distinct_codes)
    gfn = GFN(L, M, WEIGHT, beta=1.0, seed=0, hidden=128)

    # Two stages: broad exploration first (10 % of steps take a uniform random cell), then a
    # low-exploration polishing phase. Width 128 and 36,000 steps were tuned on this space.
    start = time.time()
    losses = gfn.train(oracle, steps=24000, batch=128, epsilon=0.1)
    losses += gfn.train(oracle, steps=12000, batch=128, epsilon=0.02)
    print(f"36000 steps in {time.time() - start:.0f}s; "
          f"TB loss {np.mean(losses[:100]):.3f} (first 100) -> "
          f"{np.mean(losses[-100:]):.3f} (last 100)")

    # Exact target over merit levels: each raw pair contributes R = max(merit, floor)^beta.
    target = Counter()
    for r in rows:
        target[round(r.merit, 3)] += max(r.merit, gfn.floor) ** gfn.beta
    z = sum(target.values())

    # The learned log Z should equal log(total reward mass) plus the constant from the absorbed
    # uniform backward policy, log((weight!)^2): every finished (A, B) is reached by (weight!)^2
    # orderings of its monomials (see the GFN docstring).
    true_log_z = math.log(z) + 2 * math.log(math.factorial(WEIGHT))
    print(f"log Z learned {gfn.log_z.item():.2f} vs exact {true_log_z:.2f}")

    # Empirical distribution of 10,000 draws from the trained sampler, binned by exact merit level.
    samples = gfn.sample(10000)
    empirical = Counter()
    for a, b in samples:
        key = canonical_key(L, M, a, b)
        empirical[round(merit_of[key], 3)] += 1

    # Total variation distance between the target and the sampled level shares.
    print(f"{'merit level':>12} {'target':>8} {'sampled':>8}")
    tv = 0.0
    csv_rows = []
    for level in sorted(target):
        p = target[level] / z
        q = empirical.get(level, 0) / len(samples)
        tv += abs(p - q) / 2
        print(f"{level:>12} {p:>8.3f} {q:>8.3f}")
        csv_rows.append([level, p, q])
    print(f"total variation distance: {tv:.3f}  (0 = perfect match)")

    # Persist the table the calibration figure is drawn from.
    OUT.mkdir(exist_ok=True)
    out = OUT / f"gfn_calibration_l{L}m{M}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["merit_level", "target_probability", "sampled_frequency"])
        writer.writerows(csv_rows)
    print(f"wrote {out}")


# =============================================================================================
# Leg `tail`: predict the mass above every merit level from the sampler's draws alone
# =============================================================================================

def level_masses(rows, weight_fn):
    """Mass per merit level under weight_fn(row) -> unnormalised weight."""
    by_level = Counter()
    for r in rows:
        by_level[round(r.merit, 3)] += weight_fn(r)
    z = sum(by_level.values())
    return {lv: by_level[lv] / z for lv in sorted(by_level)}


def tail(mass_at_level):
    """Mass at or above each level: the threshold grid every sampler is graded on."""
    levels = sorted(mass_at_level)
    return {lv: sum(mass_at_level[u] for u in levels if u >= lv) for lv in levels}


def grade(name, sampler_tail, target_tail, top):
    """Max abs tail-mass error over the grid, plus the top-family mass ratio (sampled / exact)."""
    errors = {lv: sampler_tail[lv] - target_tail[lv] for lv in target_tail}
    max_abs = max(abs(e) for e in errors.values())
    ratio = sampler_tail[top] / target_tail[top]
    print(f"\n{name}: max |tail-mass error| = {max_abs:.4f}, "
          f"top-family mass {sampler_tail[top]:.4f} vs exact {target_tail[top]:.4f} "
          f"(ratio {ratio:.3f})")
    print(f"{'level':>8} {'target':>8} {'sampler':>8} {'error':>8}")
    for lv in sorted(target_tail):
        print(f"{lv:>8} {target_tail[lv]:>8.4f} {sampler_tail[lv]:>8.4f} {errors[lv]:>+8.4f}")
    return {"max_abs_error": max_abs, "top_ratio": ratio,
            "tail": {str(lv): sampler_tail[lv] for lv in target_tail}}


def train_or_load(oracle, seed):
    """The sampler leg's training recipe for one seed, saved after the first run and reloaded after."""
    gfn = GFN(L, M, WEIGHT, beta=BETA, reward_floor=FLOOR, seed=seed, hidden=128)
    ckpt = OUT / f"gfn_calib_l{L}m{M}_s{seed}.pt"
    if ckpt.exists():
        state = torch.load(ckpt)
        gfn.net.load_state_dict(state["net"])
        gfn.log_z.data.fill_(state["log_z"])
        print(f"loaded {ckpt}")
        return gfn
    start = time.time()
    losses = gfn.train(oracle, steps=24000, batch=128, epsilon=0.1)
    losses += gfn.train(oracle, steps=12000, batch=128, epsilon=0.02)
    print(f"36000 steps in {time.time() - start:.0f}s; TB loss "
          f"{np.mean(losses[:100]):.3f} -> {np.mean(losses[-100:]):.3f}")
    OUT.mkdir(exist_ok=True)
    torch.save({"net": gfn.net.state_dict(), "log_z": gfn.log_z.item()}, ckpt)
    print(f"saved {ckpt}")
    return gfn


def tail_leg(seed, target_only):
    rows = load_landscape(ANSWER_KEY)
    merit_of = {r.key: r.merit for r in rows}
    pairs_of = Counter(r.key for r in rows)   # raw pairs per distinct code
    top = max(round(r.merit, 3) for r in rows)
    top_keys = {k for k, v in merit_of.items() if abs(v - top) < 1e-9}
    print(f"(6,3): {len(rows):,} raw pairs, {len(merit_of):,} distinct codes, "
          f"top family {len(top_keys)} codes at merit {top}")

    # Exact target and the two nulls, all from the answer key.
    z = sum(max(r.merit, FLOOR) ** BETA for r in rows)
    target = tail(level_masses(rows, lambda r: max(r.merit, FLOOR) ** BETA))
    uniform = tail(level_masses(rows, lambda r: 1.0))
    point_mass = {lv: 1.0 for lv in target}   # always emits a top code
    out = {"target_tail": {str(lv): target[lv] for lv in target},
           "uniform": grade("uniform proposal (exact raw-pair fractions)", uniform, target, top),
           "point_mass_top": grade("point mass on the top family", point_mass, target, top)}
    if target_only:
        return

    # Train (or reload) the sampler; the budget is the whole space, as in the sampler leg.
    oracle = Oracle(L, M, budget=len(merit_of))
    gfn = train_or_load(oracle, seed)
    true_log_z = math.log(z) + 2 * math.log(math.factorial(WEIGHT))
    log_z_learned = gfn.log_z.item()
    print(f"log Z learned {log_z_learned:.2f} vs exact {true_log_z:.2f}")

    # N_SAMPLES draws, each mapped to its exact merit through the answer key. The draws are
    # seeded here, so a reloaded checkpoint gives the same draws as the run that trained it.
    torch.manual_seed(seed)
    samples = gfn.sample(N_SAMPLES)
    counts = Counter()
    top_counts = Counter()
    for a, b in samples:
        key = canonical_key(L, M, a, b)
        counts[round(merit_of[key], 3)] += 1
        if key in top_keys:
            top_counts[key] += 1
    empirical = {lv: counts.get(lv, 0) / N_SAMPLES for lv in target}
    sampler_tail = tail(empirical)
    # Level TV: undo the tails into per-level target masses and compare with the sampled shares.
    levels = sorted(target)
    target_at_level = {lv: target[lv] - (target[nxt] if nxt else 0.0)
                       for lv, nxt in zip(levels, levels[1:] + [None])}
    level_tv = sum(abs(empirical[lv] - target_at_level[lv]) for lv in levels) / 2

    out["sampler"] = grade(f"trained sampler seed {seed}, {N_SAMPLES:,} draws",
                           sampler_tail, target, top)

    # Secondary: calibration INSIDE the top family, code by code. The exact mass of one top code
    # is (its raw-pair count x top merit) / Z; level TV cannot see this, and a sampler that
    # merely finds the family fails it.
    exact_pc = [pairs_of[k] * top ** BETA / z for k in sorted(top_keys)]
    sampled_pc = [top_counts.get(k, 0) / N_SAMPLES for k in sorted(top_keys)]
    ratios = [s / e for s, e in zip(sampled_pc, exact_pc)]
    share_in_band = sum(0.5 <= r <= 2.0 for r in ratios) / len(ratios)
    rho_pc = spearmanr(sampled_pc, exact_pc).correlation
    if np.isnan(rho_pc):
        rho_pc = None   # every top code carries the same exact mass: undefined
    print(f"top family per-code: {share_in_band:.3f} of {len(ratios)} codes within [0.5, 2.0]x "
          f"(ratio range {min(ratios):.2f}-{max(ratios):.2f}); Spearman {rho_pc}; "
          f"seen {sum(c > 0 for c in sampled_pc)}/{len(ratios)}")
    print(f"level TV {level_tv:.3f}")
    out["sampler"].update({"level_tv": level_tv, "top_classes_seen": sum(c > 0 for c in sampled_pc),
                           "log_z_learned": log_z_learned, "log_z_exact": true_log_z,
                           "per_code_share_in_band": share_in_band, "per_code_spearman": rho_pc,
                           "per_code_ratios": ratios})

    path = OUT / f"gfn_mass_tail_l{L}m{M}_s{seed}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    legs = parser.add_subparsers(dest="leg", required=True)
    legs.add_parser("sampler", help="train once (seed 0), 10,000 draws, level TV against the exact target")
    tail_parser = legs.add_parser("tail", help="train or reload one seed, 50,000 draws, tail masses")
    tail_parser.add_argument("--seed", type=int, default=0)
    tail_parser.add_argument("--target-only", action="store_true",
                             help="print the exact target and the two nulls, no sampler")
    args = parser.parse_args()

    if args.leg == "sampler":
        sampler_leg()
    else:
        tail_leg(args.seed, args.target_only)


if __name__ == "__main__":
    main()
