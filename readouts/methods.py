"""The methods table (comparison methods as run): the cells that are measured rather than set in code.

Tuned PT range. 180: 0.25-2.0 (better of two, all twelve bias and seed pairs);
    300: 0.01-0.15 (best of five on packed yield).
Supervised screen, 180: 666,113 parameters; 252,224 updates, chosen among four variants and twenty
    checkpoints by held-out precision on a tenth of the stored forms, then trained on all of them.
Earlier, smaller screen, 180: two hidden layers of 64 (45,889 parameters), batch 64, 315,280 updates.
Supervised screen, 300: 1,990 rounds; 96,449 parameters.

Every other cell of the table is a setting written in the code. This script reads data/ only and calls
no oracle; about 12 s on a laptop. Run from the repo root:
    python -m readouts.methods
"""

import glob
import json

import numpy as np
import torch

from gflowqec import counts, screens
from gflowqec.codes import GROUP_180, RelTables
from gflowqec.oracle import bias_score, key_top, lam_of
from gflowqec.paths import DATA

N180 = DATA / "n180"
N300 = DATA / "n300"
MODELS = DATA / "models"


# =============================================================================================
# Tuned PT range at n = 180.
#
# Two temperature ranges were run, each at both biases (sign Z is gamma = +1, sign X is gamma = -1)
# and three seeds, in two blocks: one on the exact oracle (every paid code gets exact distances) and
# one on the certifying oracle (bound ladders first, exact distances only when the bounds reach the
# bar). That is 2 blocks x 2 signs x 3 seeds = 12 pairs. In each pair the better range is the one
# with more top-scoring codes per 1,000 paid codes.
# =============================================================================================

def top_count(rows, lam, top):
    """Top-scoring codes among a run's paid rows at lam (the bar is top).

    An exact-oracle row (no bound distances dA, dB) counts when its exact score reaches the bar.
    A certifying-oracle row counts when the oracle went on to certify it, which happens exactly when
    its bound score reaches the bar or its bounds did not close (proxy-top), and its certified score
    reaches the bar too. A row that was never certified can never be proxy-top."""
    count = 0
    for r in rows:
        if "dA" not in r:                                    # an exact-oracle row
            count += bias_score(r["dX"], r["dZ"], lam) >= top
            continue
        proxy_top = bias_score(r["dA"], r["dB"], lam) >= top or not (r["exactA"] and r["exactB"])
        if r["cert"] is None:
            assert not proxy_top, "an uncertified code is proxy-top: the run never valued it here"
            continue
        if proxy_top:
            count += bias_score(r["dX"], r["dZ"], lam) >= top
    return count


RANGES_180 = ("0.25-2.0", "0.5-4.0")
wins = {t: 0 for t in RANGES_180}
for block in ("", "_cert"):                  # "" = the exact-oracle block, "_cert" = the certifying block
    for sign in ("Z", "X"):
        lam = lam_of(10, sign)
        top = key_top(lam)[0]                # the answer key's best score at this bias (10.0 at both signs)
        for seed in range(3):
            rate = {}
            for t in RANGES_180:
                d = json.load(open(N180 / f"r174_pt_C6xS3_e10{sign}_b3000_t{t}{block}_s{seed}.json"))
                assert d["complete"] and d["sign"] == sign and d["eta"] == 10.0
                rows = counts.run_rows(d)
                rate[t] = 1000 * top_count(rows, lam, top) / len(rows)
            better = max(RANGES_180, key=lambda t: rate[t])
            assert rate[RANGES_180[0]] != rate[RANGES_180[1]], "a tie: no better range in this pair"
            wins[better] += 1
            print(f"basis: PT 180 {block[1:] or 'exact'} {sign} s{seed}: top-scoring per 1,000 paid "
                  + ", ".join(f"t{t} {rate[t]:.1f}" for t in RANGES_180))
best180 = max(RANGES_180, key=lambda t: wins[t])
print(f"Table C.1 tuned PT range 180: {best180}, better of two in {wins[best180]} of {sum(wins.values())} "
      f"bias and seed pairs")


# =============================================================================================
# Tuned PT range at n = 300.
#
# Five ranges, six seeds each, 40,000 paid calls per run. Packed yield of a run: take its codes whose
# re-estimated rate ler_pq is at most tau_1% (counts.TAU1), then pack them greedily best first,
# keeping a code only if its canonical tuple differs from every kept one in more than two of its ten
# group elements. The range with the highest six-seed mean wins.
# =============================================================================================

def elites(d, tau=counts.TAU1):
    """(tuples, ler_pq) of a run's codes at or below tau, in call order."""
    keep = [(r, pq) for r, pq in zip(d["rep"], d["ler_pq"]) if pq <= tau]
    return np.array([r for r, _ in keep]), np.array([pq for _, pq in keep])


def packing(reps, pqs, radius):
    """Greedy best-first packing: hand over the best code, then the best code not within `radius`
    positions of anything already handed over. Returns how many codes are handed over."""
    kept = []
    for i in np.argsort(pqs, kind="stable"):        # best (lowest ler_pq) first
        if all((reps[i] != reps[j]).sum() > radius for j in kept):
            kept.append(i)
    return len(kept)


RANGES_300 = ("t0.01-0.15", "t0.02-0.3", "t0.05-0.67", "t0.05-0.67_c32", "t0.3-4.0")   # _c32: 32 chains
packed = {}
for t in RANGES_300:
    per_seed = []
    for seed in range(6):
        d = json.load(open(N300 / f"mitten_lersearch_pt_C10xS3_b0.35_{t}_w_s{seed}.json"))
        assert d["calls"] == d["budget"] == 40_000
        reps, pqs = elites(d)
        per_seed.append(packing(reps, pqs, 2))
    packed[t] = np.mean(per_seed)
    print(f"basis: PT 300 {t}: packed yield per seed {per_seed}, mean {packed[t]:.1f}")
best300 = max(RANGES_300, key=lambda t: packed[t])
assert best300 == "t0.01-0.15", f"the packed-yield winner is {best300}, not the range the paper uses"
print(f"Table C.1 tuned PT range 300: {best300[1:]}, best of five on packed yield")


# =============================================================================================
# The supervised screen at n = 180 (the policy's transformer turned into a regressor).
#
# Four configurations (screens.CONFIGS) were trained on nine tenths of the 116,972 stored forms and
# scored on the held-out tenth at twenty evenly spaced checkpoints. The chosen configuration and step
# (screen180match_choice.json) were then retrained on all the stored forms, once per seed.
# =============================================================================================

choice = json.load(open(N180 / "screen180match_choice.json"))
T180 = RelTables(GROUP_180)
params180 = screens.n_params(screens.build_net(choice["config"], T180.n, T180))

ckpts = [torch.load(MODELS / f"r177_mscreen_{choice['config']}_st{choice['step']}_s{seed}.pt",
                    map_location="cpu", weights_only=True) for seed in range(3)]
for c in ckpts:
    assert c["config"] == choice["config"] and c["steps"] == choice["step"] and c["params"] == params180
assert len({c["rows"] for c in ckpts}) == 1
trained_on = ckpts[0]["rows"]

selects = [json.load(open(f)) for f in sorted(glob.glob(str(N180 / "screen180match_select_*_s0.json")))]
assert len(selects) == len(screens.CONFIGS)
assert len({len(s["curve"]) for s in selects}) == 1
n_checkpoints = len(selects[0]["curve"])
assert len({(s["n_train"], s["n_held"]) for s in selects}) == 1
n_train, n_held = selects[0]["n_train"], selects[0]["n_held"]
assert n_train + n_held == trained_on == counts.STORED_FORMS
print(f"basis: held out {n_held:,} of {n_train + n_held:,} stored forms; chosen {choice['config']} at step {choice['step']:,}")
print(f"Table C.1 supervised screen 180: {params180:,} parameters; {choice['step']:,} updates, chosen among "
      f"{len(selects)} variants and {n_checkpoints} checkpoints on a held-out {n_held / (n_train + n_held):.2f} "
      f"of the stored forms, then trained on {trained_on:,}")


# =============================================================================================
# The earlier, smaller screen at n = 180: a perceptron on the bag encoding of the code (4 x 36),
# the condition vector (2) and the relational counts. Its batch and update count are stamped in the
# ten deployment dumps it ranked.
# =============================================================================================

params_mlp = screens.n_params(screens.screen_net(4 * T180.n + 2 + T180.dim))
dumps = [json.load(open(f)) for f in sorted(glob.glob(str(N180 / "r174_proxy_pool116972_MR_e10*_n1000_o*_h64_st315280_s*.json")))]
assert len(dumps) == 10 and all(d["relational"] and d["hidden"] == 64 for d in dumps)
assert len({(d["batch"], d["steps"]) for d in dumps}) == 1
print(f"Table C.1 earlier, smaller screen 180: {params_mlp:,} parameters, batch {dumps[0]['batch']}, "
      f"{dumps[0]['steps']:,} updates")


# =============================================================================================
# The supervised screen at n = 300, refitted inside each search run. After 200 uniform draws it
# pays 20 codes per round until the 40,000-call budget is spent: (40,000 - 200) / 20 rounds. The
# network is the same perceptron on the bag encoding (4 x 60) and the relational counts, no condition.
# =============================================================================================

T300 = RelTables("C10xS3")
params300 = screens.n_params(screens.screen_net(4 * T300.n + T300.dim))
runs = [json.load(open(N300 / f"mitten_lersearch_selfscreen_C10xS3_b0.35_t0.05-0.67_w_s{seed}_ss300.json"))
        for seed in range(6)]
assert all(d["ss_params"] == params300 for d in runs)
rounds = {(d["budget"] - d["ss_calib"]) // d["ss_batch"] for d in runs}
assert len(rounds) == 1
print(f"Table C.1 supervised screen 300: {rounds.pop():,} rounds, {params300:,} parameters")
