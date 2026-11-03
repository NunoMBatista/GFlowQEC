"""The measured numbers of the text on the training data (stored evaluations), the oracle and deployment, and of
the policy appendix (the offline policy, mostly its offline-training part).

The stored forms are the training set of GFlowQEC-offline: every distinct canonical form paid by the 42 stored
runs at n = 180 (gflowqec.counts.stored_forms). This readout prints

  - how many runs, paid calls and distinct forms there are (training data), and how many are certified
    (offline training);
  - the certification thresholds, from the reference sample (oracle and references);
  - how many legal tuples each stored form has (deployment and policy appendix);
  - the score below which the reward floor binds at beta = 64 (policy appendix);
  - how many stored forms reach the threshold, and how many of those only by their bound score (offline training);
  - which runs stored those bound-only forms (offline training);
  - that no reference-sample form, or its mirror image, is a stored form (offline training);
  - the pilot comparison of the transformer with a perceptron at matched steps (offline training);
  - the two costs not charged to the deployments: the policy's training and the stored runs' CPU time (not printed in the paper).

Reads only data/n180, data/keys and data/models (through gflowqec.paths.DATA). No oracle call; about 31 s on a
laptop. Run from the repo root:
    python -m readouts.stored_forms
"""
import collections
import json
import math
import re

import torch

from gflowqec.codes import GROUP_180, RelTables, identity_index, mirror, mul_table, orbit_key
from gflowqec.counts import stored_forms, value
from gflowqec.oracle import key_top, lam_of, reference_sample
from gflowqec.paths import DATA
from gflowqec.stats import katz

N180 = DATA / "n180"
LAMS = (1.0, -1.0)      # gamma = +1 (Z-biased noise) and gamma = -1 (X-biased noise), at eta = 10


def sign_name(lam):
    return "+1" if lam > 0 else "-1"


# ----------------------------------------------------------------------------- the 42 stored runs
# The stored runs are the certified n = 180 run dumps (file names r174_*_cert*.json; counts.stored_forms reads
# the same files). Each paid 3,000 oracle calls; "rep" holds one tuple per paid call.
pool_files = sorted(N180.glob("r174_*_cert*.json"))
paid = 0
tags = {}           # canonical form -> the biases of the runs that paid for it, e.g. {"X10"} or {"Z10", "cond"}
for f in pool_files:
    d = json.load(open(f))
    assert d["complete"] and len(d["rep"]) == 3000, f"{f.name}: not a complete 3,000-call run"
    paid += len(d["rep"])
    # the run's bias from its file name: _e10Z_ is eta = 10 with noisy Z, _e10X_ noisy X; the conditional runs
    # (r174_cond_*) were trained over several biases and carry no single one
    m = re.search(r"_e(\d+)([ZX])_", f.name)
    tag = f"{m.group(2)}{m.group(1)}" if m else "cond"
    for rep in d["rep"]:
        tags.setdefault(orbit_key(GROUP_180, rep), set()).add(tag)

rows = stored_forms()       # {canonical form: row}; asserts the 116,972 forms
certified = sum(r["cert"] for r in rows.values())
# a certified form whose exact ladder stopped before it closed (censored) keeps a lower bound, not an exact value
lower_bound = sum(r["cert"] and not r["exact"] for r in rows.values())
others = sum(not r["cert"] for r in rows.values())

print(f"Sec. 5.3: stored runs {len(pool_files)}, paid calls {paid:,}, distinct canonical forms {len(rows):,}")
print(f"App. B.1: certified {certified:,}, of them at a lower bound {lower_bound}; others {others:,}")

# ----------------------------------------------------------------------------- certification thresholds
# The certifying oracle certifies a code when its bound score reaches the top score of the reference sample
# (the 666 uniform draws with exact distances, mirror-pooled): oracle.key_top.
bar = {lam: key_top(lam)[0] for lam in LAMS}
print(f"Sec. 4.3: certification thresholds {bar[1.0]:.1f} at gamma = +1 and {bar[-1.0]:.1f} at gamma = -1")


# ----------------------------------------------------------------------------- legal tuples per stored form
def leaf_count(key, mul, inv, ident):
    """Number of legal token sequences that map to this orbit: per entry, the distinct (pivot x ordering)
    representations in the policy's gauge (first term e, the other two free, non-identity, distinct), multiplied
    over the four entries because the gauge acts per entry."""
    m = 1
    for e0, side in ((0, "A"), (3, "A"), (6, "B"), (9, "B")):
        terms = [int(v) for v in key[e0:e0 + 3]]
        reps = set()
        for piv in terms:
            if side == "A":
                gauged = [int(mul[t, inv[piv]]) for t in terms]
            else:
                gauged = [int(mul[inv[piv], t]) for t in terms]
            free = [g for g in gauged if g != ident]
            assert len(free) == 2, "a canonical entry holds the identity exactly once"
            reps.add((free[0], free[1]))
            reps.add((free[1], free[0]))
        m *= len(reps)
    return m


# A sum of three distinct elements has six legal tuples (three translations that put the identity into it,
# times two orders of the other two), so 6^4 = 1,296 per form; a coset sum gives fewer.
T = RelTables(GROUP_180)
mul, inv, ident = mul_table(GROUP_180), T.inv, identity_index(GROUP_180)
leaves = collections.Counter(leaf_count(k, mul, inv, ident) for k in rows)
full = leaves.pop(1296)
fewer = ", ".join(f"{c:,}" for c in sorted(leaves))
print(f"Sec. 5.4 and App. B: legal tuples per stored form 1,296 for {full:,} forms, "
      f"fewer ({fewer}) for {sum(leaves.values()):,}")
# The augmentation reorders the two free slots of each of the four sums (the identity term stays put):
# 2! orders per sum, four sums, so 2!^4 = 16 tuples of the up to 1,296.
reorderings = math.factorial(2) ** 4
print(f"Sec. 5.4 and App. B: augmentation reorderings {reorderings} (two free slots per sum, four sums)")

# ----------------------------------------------------------------------------- the reward floor
# The reward is s^beta clamped at 1e-9 (the log of a zero reward is undefined), so the floor binds when the
# score s is below 10^(-9 / beta).
print(f"App. B: floor binds below {10 ** (-9 / 64):.3f} at beta = 64")

# ----------------------------------------------------------------------------- stored forms at the threshold
# A stored form's score is exact if it was certified, else its bound score (an upper bound): counts.value.
# Forms at the threshold by the bound alone have an unknown exact score.
at_bar, bound_only, opposite = {}, {}, {}
for lam in LAMS:
    top = [k for k, r in rows.items() if value(r, lam) >= bar[lam]]
    bound = [k for k in top if not rows[k]["cert"]]
    at_bar[lam], bound_only[lam] = len(top), len(bound)
    # runs at the opposite bias: gamma = +1 is noisy Z (lam_of(10, "Z") = +1), so its opposite runs are the X runs
    other = "X" if lam_of(10, "Z") == lam else "Z"
    opposite[lam] = sum(tags[k] == {f"{other}10"} for k in bound)


def share(lam):
    return f"{100 * bound_only[lam] / at_bar[lam]:.1f}"


print(f"App. B.1: gamma = +1: {at_bar[1.0]:,} stored forms at {bar[1.0]:.1f}, {bound_only[1.0]} by the bound alone "
      f"({share(1.0)} per cent); gamma = -1: {at_bar[-1.0]:,} and {bound_only[-1.0]} ({share(-1.0)} per cent)")
print(f"App. B.1: bound-alone forms stored by runs at the opposite bias: {opposite[1.0]} of {bound_only[1.0]} at "
      f"gamma = +1, {opposite[-1.0]} of {bound_only[-1.0]} at gamma = -1")

# ----------------------------------------------------------------------------- the reference sample
# Neither a reference-sample form nor its mirror image (the sector-swapping involution) is a stored form.
ref = reference_sample()
keys = {orbit_key(GROUP_180, r["exps"]) for r in ref}
mkeys = {orbit_key(GROUP_180, mirror(r["exps"], T)) for r in ref}
print(f"App. B.1: reference-sample forms among the stored forms {len(keys & rows.keys())} of {len(keys)}; "
      f"their mirror images {len(mkeys & rows.keys())} of {len(mkeys)}")

# ----------------------------------------------------------------------------- the pilot at matched steps
# Both networks of the pilot trained for the same number of steps on the stored forms (the stamps of the two
# transformer seeds and of the one perceptron seed); GFlowQEC-offline trained for many more.
pilot = [torch.load(DATA / "models" / name, weights_only=False, map_location="cpu")["steps"]
         for name in ("r177_transformer_w18_b256_st1699_s0.pt", "r177_transformer_w18_b256_st1699_s1.pt",
                      "r177_mlp_w18_b256_s0.pt")]
assert len(set(pilot)) == 1, f"the pilot's networks trained for different step counts: {pilot}"
record = {torch.load(DATA / "models" / f"r177_transformer_w18_b256_x20_s{s}.pt", weights_only=False,
                     map_location="cpu")["steps"] for s in (0, 1, 2)}
assert len(record) == 1, f"GFlowQEC-offline's seeds disagree: {record}"
print(f"App. B.1: pilot steps {pilot[0]:,} for both networks; GFlowQEC-offline {record.pop():,}")

# Each network drew 1,000 forms per bias; every drawn form was screened with the bound score. The lens is the
# novel-only rate: among drawn forms that are not stored forms, the share whose bound score reaches the threshold
# (stamps "proxy_top_novel" out of "n_novel"). The two transformer seeds are pooled against the one perceptron
# seed. The perceptron reads the relational counts with a condition input; the transformer is the pilot model
# trained for the matched number of steps, not GFlowQEC-offline.
# The ratio of the two rates comes with the Katz interval; "within noise" means it contains 1.
ratios, within = {}, True
for lam, s in ((1.0, "Z"), (-1.0, "X")):
    tr = [json.load(open(N180 / f"r177_screen_transformer_w18_b256_st1699_e10{s}_n1000_s{seed}.json"))
          for seed in (0, 1)]
    mlp = json.load(open(N180 / f"r177_screen_mlp_w18_b256_e10{s}_n1000_s0.json"))
    r, lo, hi = katz(sum(t["proxy_top_novel"] for t in tr), sum(t["n_novel"] for t in tr),
                     mlp["proxy_top_novel"], mlp["n_novel"])
    ratios[lam] = r
    within = within and lo <= 1.0 <= hi
    print(f"basis: gamma = {sign_name(lam)}: transformer {sum(t['proxy_top_novel'] for t in tr)} / "
          f"{sum(t['n_novel'] for t in tr):,}, perceptron {mlp['proxy_top_novel']} / {mlp['n_novel']:,}, "
          f"ratio {r:.4f}, Katz 95 per cent [{lo:.2f}, {hi:.2f}]")
print(f"App. B.1: transformer / perceptron at matched steps {ratios[1.0]:.2f} at gamma = +1, {ratios[-1.0]:.2f} at "
      f"gamma = -1; Katz 95 per cent interval contains 1 at both: {'within noise' if within else 'NOT within noise'}")

# ----------------------------------------------------------------------------- compute not charged (not printed in the paper)
# The deployments of GFlowQEC-offline are not charged for two fixed costs, printed here:
#   - the training of each GFlowQEC-offline policy: the processor seconds stamped in its checkpoint (cpuTrain, the
#     training process's CPU time over its threads), one checkpoint per seed;
#   - the 42 stored runs that paid for its training set: per paid call the bound seconds (cpuBound) plus the
#     exact-certification seconds (cpu), summed over every call of every run.
train_hours = []
for s in (0, 1, 2):
    ck = torch.load(DATA / "models" / f"r177_transformer_w18_b256_x20_s{s}.pt", weights_only=False, map_location="cpu")
    assert ck.get("device", "cpu") == "cpu", "a GPU-trained checkpoint has no CPU training time"
    train_hours.append(ck["cpuTrain"] / 3600)
bound_sec, cert_sec = 0.0, 0.0
for f in pool_files:
    d = json.load(open(f))
    bound_sec += sum(d["cpuBound"])
    cert_sec += sum(d["cpu"])
print(f"basis: training per seed {[round(h, 2) for h in train_hours]} CPU-hours; stored runs: bounds "
      f"{bound_sec / 3600:.1f} + certification {cert_sec / 3600:.1f} = {(bound_sec + cert_sec) / 3600:.1f} CPU-hours")
print(f"Not in the paper: training of GFlowQEC-offline {min(train_hours):.1f} to {max(train_hours):.1f} CPU-hours per policy")
print(f"Not in the paper: the {len(pool_files)} stored runs, {paid:,} paid calls, {(bound_sec + cert_sec) / 3600:.0f} CPU-hours")
