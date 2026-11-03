"""The further n = 180 results, the text after their per-seed table: the k check, and a pay-back reading
that the paper does not print.

The sentence on the unranked deployment against the pilot screen restates note b of the table
and is printed by readouts/res_prefix.py, not here.

Pay-back against tuned PT. The unranked deployment of GFlowQEC-offline reused the stored evaluations
and its own training; charged with their processor time (the fixed cost), how many deployments of 1,000 paid
calls would it need before its all-in seconds per novel certified top-scoring code fell to tuned PT's? That
count is R*. It is infinite when one deployment's seconds already exceed the seconds tuned PT pays for the
same number of tops: reuse divides only the fixed cost and can never repair a per-deployment deficit. The
interval resamples seeds 0-2 in pairs (deployment and tuned PT of the same seed together). The rival
is tuned PT continued, its paid calls 1,000 to 2,999, the honest rival from the second deployment on.

The k check. Does the number of logical qubits k single out the top-scoring codes? Over the distinct
codes paid by the 33 stored runs at gamma = +1 and -1, compare the top-scoring share inside each k class
with the overall share; the paper prints the largest ratio. The k distribution of the pool is read from a
seeded uniform sample of 4,000 distinct codes.

Reads only data/ through the release library; no oracle is called; about 66 s on a laptop. Run from the
repo root:
    python -m readouts.more180
"""
import collections
import json
import random
import time

import numpy as np
import torch
from ldpc import mod2

from gflowqec import counts
from gflowqec.codes import GROUP_180, build
from gflowqec.oracle import bias_score, key_top, lam_of
from gflowqec.paths import DATA

T0 = time.time()
N180 = DATA / "n180"
DELIVERY = 1000          # one deployment = one cell of 1,000 paid calls (the reuse axis)
BOOTS = 2000             # paired seed bootstrap resamples of R*
DEPLOY = "POLICY r177_screen_transformer_w18_b256_x20_e10{sign}_n2000"   # the unranked deployment, 2,000 calls
RIVAL = "PT continued t0.25-2.0 (orbits 1,000-3,000)"                     # tuned PT, paid calls 1,000-2,999
SEEDS3 = (0, 1, 2)   # processor seconds are compared on seeds 0-2 only: seeds 3-9 ran on different machines

print("App. D.1 text")


# =============================================================================================
# Pay-back R* against tuned PT
# =============================================================================================

def dump_price(rows, lam, top):
    """Processor seconds one stored run paid at this condition: the bound seconds of every paid call plus
    the certification seconds of every proxy-top call. A proxy-top call of a conditional run that was never
    certified at this condition cost nothing and charges nothing."""
    paid = 0.0
    for r in rows:
        paid += r["cpuBound"]
        proxy_top = bias_score(r["dA"], r["dB"], lam) >= top or not (r["exactA"] and r["exactB"])
        if proxy_top and r["cert"] is not None:
            paid += r["cpu"]
    return paid


def rstar(arm_cells, rival_cells, fixed):
    """R*, the smallest number of deployments at which the arm's amortised seconds per delivered novel
    certified top fall to the rival's seconds per certified top. Infinite when the per-deployment cost
    alone already exceeds the rival's.

    The rival never saw the stored forms, so all its certified tops count (tops(False)); the arm counts
    novel tops only."""
    # every number here is CPU-seconds, so the arm and its rival must have paid the same bound ladder
    assert len({c.ladder for c in list(arm_cells) + list(rival_cells)}) == 1, \
        "R* would divide seconds paid on different bound ladders"
    per_del = float(np.mean([c.sec.sum() * DELIVERY / c.n for c in arm_cells]))
    tops_del = float(np.mean([c.tops() * DELIVERY / c.n for c in arm_cells]))
    rt = sum(c.tops(False) for c in rival_cells)
    rival_per_top = sum(c.sec.sum() for c in rival_cells) / rt if rt else float("nan")
    gain = tops_del * rival_per_top - per_del
    return (fixed / gain if gain > 0 else float("inf")), per_del, tops_del, rival_per_top


def rstar_ci(arm_cells, rival_cells, fixed, rng, boots):
    """Paired bootstrap 95 % interval of R* over seeds: a resampled seed brings its arm cell and its
    rival cell together."""
    both = sorted({c.seed for c in arm_cells} & {c.seed for c in rival_cells})
    a_by = {c.seed: c for c in arm_cells}
    b_by = {c.seed: c for c in rival_cells}
    draws = []
    for _ in range(boots):
        idx = rng.integers(len(both), size=len(both))
        draws.append(rstar([a_by[both[i]] for i in idx], [b_by[both[i]] for i in idx], fixed)[0])
    # "nearest" rather than interpolation: R* is infinite for many resamples, and interpolating between
    # two infinities gives nan, not an interval end
    return (float(np.nanpercentile(draws, 2.5, method="nearest")),
            float(np.nanpercentile(draws, 97.5, method="nearest")))


def whole(x):
    """The point R* as printed: whole deployments, rounded up when finite (the interval ends are rounded)."""
    return f"{np.ceil(x):.0f}" if np.isfinite(x) else "inf"


runs, proxies = counts.load_dumps()
runs += counts.load_r200()
pool_count = counts.ledger_pool(runs)
rng = np.random.default_rng(0)       # one stream for both signs, gamma = +1 first
payback, per_seed = [], {}
for sign in ("Z", "X"):
    lam = lam_of(10, sign)
    top = key_top(lam)[0]
    arms = counts.ledger_arms(runs, proxies, sign, lam, top, pool_count)
    dep = sorted((c for c in arms[DEPLOY.format(sign=sign)] if c.seed in SEEDS3), key=lambda c: c.seed)
    pt = sorted((c for c in arms[RIVAL] if c.seed in SEEDS3), key=lambda c: c.seed)
    assert all(c.uncertified == 0 for c in dep + pt), "a proxy-top row was never certified"

    # the fixed cost: the 42 stored runs' price at this condition (the runs counts.ledger_pool counts) ...
    stored = [d for d in runs if counts.kind_of(d) == "[cert]" and not d["file"].startswith("r200_")]
    assert len(stored) == 42
    pool_paid = sum(dump_price(counts.rep_rows(d), lam, top) for d in stored)
    # ... plus the training CPU stamped in the deployed model's checkpoint (seed 0's model, as the ledger reads)
    model = json.load(open(N180 / dep[0].source))["model"]
    ck = torch.load(DATA / "models" / f"{model}.pt", weights_only=False)
    assert ck.get("device", "cpu") == "cpu", "a GPU-trained checkpoint has no CPU training time"
    fixed = pool_paid + float(ck["cpuTrain"])

    r, per_del, tops_del, rival_per_top = rstar(dep, pt, fixed)
    lo, hi = rstar_ci(dep, pt, fixed, rng, BOOTS)
    gamma = "+1" if sign == "Z" else "-1"
    payback.append(f"gamma = {gamma} {whole(r)} [{lo:.0f}, {hi:.0f}]")
    # per seed: seconds per novel certified top of the deployment against tuned PT's per certified top
    per_seed[sign] = ([float(c.sec.sum() / c.tops()) for c in dep], [float(c.sec.sum() / c.tops(False)) for c in pt])
    print(f"basis: App. D.1 {sign}: fixed {fixed:,.0f} s (stored runs {pool_paid:,.0f} s + training {ck['cpuTrain']:,.0f} s); "
          f"per deployment {per_del:,.0f} s for {tops_del:.2f} novel tops; tuned PT {rival_per_top:.1f} s per top; "
          f"R* {r} [{lo:.2f}, {hi}]; per seed {[round(x, 1) for x in per_seed[sign][0]]} against "
          f"{[round(x, 1) for x in per_seed[sign][1]]}")

print(f"Not in the paper: pay-back R* against tuned PT, in deployments of {DELIVERY:,} paid calls, point [paired-seed interval]: "
      + "; ".join(payback))
a, b = per_seed["Z"]
above = sum(x > y for x, y in zip(a, b))
print("Not in the paper: gamma = +1, CPU-s per novel certified top, deployment against tuned PT: "
      f"{' / '.join(f'{x:.1f}' for x in a)} against {' / '.join(f'{y:.1f}' for y in b)}; higher in {above} of {len(a)} seeds")


# =============================================================================================
# The k check on the stored evaluations
# =============================================================================================

# The 33 stored runs at gamma = +1 and -1, in file-name order (the conditional runs and the gamma = 0 runs
# are the other 9 of the 42).
K_FILES = sorted(
    [f"r174_lsgfn_C6xS3_e10{s}_b3000_{t}_cert_s{i}.json" for s in "XZ" for t in ("k1i7det", "k1i7detR") for i in range(3)]
    + [f"r174_mdqn_C6xS3_e10{s}_b3000_cert_s{i}.json" for s in "XZ" for i in range(3)]
    + [f"r174_pt_C6xS3_e10{s}_b3000_{t}_cert_s{i}.json" for s in "XZ" for t in ("t0.25-2.0", "t0.5-4.0") for i in range(3)]
    + [f"r174_random_C6xS3_e10Z_b3000_cert_s{i}.json" for i in range(3)])
assert len(K_FILES) == 33

kcache = {}


def k_of(rep):
    """Logical qubits k = n - rank Hx - rank Hz of the code (cached by its tuple)."""
    key = tuple(int(v) for v in rep)
    if key not in kcache:
        code = build(GROUP_180, np.array(key))
        Hx = np.asarray(code.matrix_x) % 2
        Hz = np.asarray(code.matrix_z) % 2
        kcache[key] = int(Hx.shape[1] - mod2.rank(Hx) - mod2.rank(Hz))
    return kcache[key]


pool = collections.Counter()      # k of the 4,000 sampled distinct codes
tops = collections.Counter()      # k of the certified top-scoring codes
seen = set()                      # distinct codes, first sighting only, in file order
random.seed(0)
for name in K_FILES:
    d = json.load(open(N180 / name))
    assert d["complete"] and d["lam"] in (1.0, -1.0), name
    lam = d["lam"]
    for i in range(d["calls"]):
        rep = tuple(int(v) for v in d["rep"][i])
        if rep in seen:
            continue
        seen.add(rep)
        # a certified top: certified, exact in both sectors, and scoring at least 10 at this run's bias
        certified_top = (bool(d["cert"][i]) and bool(d["exactX"][i]) and bool(d["exactZ"][i])
                         and bias_score(d["dX"][i], d["dZ"][i], lam) >= 10.0)
        if certified_top:
            tops[k_of(rep)] += 1
# the pool's own k distribution, from a uniform sample of 4,000 distinct stored codes
sample = random.sample(sorted(seen), min(4000, len(seen)))
for rep in sample:
    pool[k_of(rep)] += 1

# top-scoring share inside each k class (tops over the class's estimated size) against the overall share
overall = sum(tops.values()) / len(seen)
rate = {k: tops[k] / (pool[k] / len(sample) * len(seen)) for k in sorted(pool) if pool[k] > 0}
ratio = max(rate.values()) / overall
print(f"basis: App. D.1 {len(K_FILES)} runs, {len(seen):,} distinct codes, {sum(tops.values()):,} certified tops "
      f"(by k {dict(sorted(tops.items()))}); sample k shares "
      + ", ".join(f"k={k}: {100 * pool[k] / len(sample):.2f} %" for k in sorted(pool))
      + f"; rates " + ", ".join(f"k={k}: {100 * v:.3f} %" for k, v in rate.items())
      + f"; overall {100 * overall:.3f} %; ratio {ratio:.4f}")
print(f"App. D.1: sorting the stored evaluations by k raises their top-scoring share at most {ratio:.1f} times")
print(f"timed: {time.time() - T0:.0f} s")
