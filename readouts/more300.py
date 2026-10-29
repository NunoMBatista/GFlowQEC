"""The numbers in the text of the further n = 300 results, after the table of further runs, one block per claim.

The objective swap: GFlowQEC-online minus the objective swap, in per cent of GFlowQEC-online's mean W,
    with a 95 % paired t interval over seeds 0-5; at L = 127 under common random numbers, at L = 31 one
    stream.
The k = 60 class: the share of proposals with k = 60 logical qubits, and inside that class the share of
    re-estimates at or below y*, for GFlowQEC-online (L = 127), tuned PT and uniform sampling, one stream
    per code.
The best codes: the best of the 200 highest-scoring codes of six configurations, each re-estimated on
    4,000 independent shots: the range of the six best rates and whether their intervals overlap.
The best codes at 200 calls: per seed, the best codes of BO at 200 calls, uniform and tuned PT at their
    first 200 calls and GFlowQEC-online at 40,000 calls (common random numbers): do their four Wilson
    intervals overlap?

W, y* and the visited files are as in readouts/res300.py: W = the run's visits V times the share of 800
sampled visits re-estimated at or below y*, the per-seed median of five reference runs pooled.

Reads data/n300 only; no oracle call; about 5 s on a laptop. Run from the repo root:
    python -m readouts.more300
"""
import json
import math

import numpy as np
from scipy import stats as sps

from gflowqec.counts import REFERENCE_CELLS, SEEDS, good_count, load_visited, y_star
from gflowqec.paths import DATA
from gflowqec.stats import pq, wilson

N300 = DATA / "n300"     # the stored n = 300 runs

# Visited files: the five reference cells (for y*), the four runs of the objective-swap block, the three of
# the k = 60 block
runs = load_visited(list(REFERENCE_CELLS) + ["c127", "p127", "p31", "d127", "rnd"])
q = {s: y_star(runs, s) for s in SEEDS}
W = {c: {s: good_count(runs[c][s], q[s]) for s in SEEDS} for c in runs}

print("App. E.5 text (n = 300)")

# ---- The objective swap: GFlowQEC-online minus the swap, in per cent of GFlowQEC-online's mean W ----
# (ref, twin): c127 / p127 are depth 127 under common random numbers, d31 / p31 depth 31, one stream.
parts = []
for depth, ref, twin in (("127", "c127", "p127"), ("31", "d31", "p31")):
    dl = np.array([W[ref][s] - W[twin][s] for s in SEEDS])   # per-seed difference
    base = np.mean([W[ref][s] for s in SEEDS])               # "of its mean": GFlowQEC-online's mean W
    # 95 % t interval on the mean difference, 5 degrees of freedom
    ci = sps.t.interval(0.95, len(dl) - 1, loc=dl.mean(), scale=sps.sem(dl))
    pct, lo, hi = 100 * dl.mean() / base, 100 * ci[0] / base, 100 * ci[1] / base
    # two significant figures, trailing zero kept (-0.3965 prints -0.40); the ends carry a sign
    parts.append(f"L = {depth} {'%#.2g' % pct} [{'%#+.2g' % lo}, {'%#+.2g' % hi}]")
    print(f"basis: {ref} - {twin} mean {dl.mean():+,.0f} of {ref}'s {base:,.0f}: "
          f"{pct:+.4f} % [{lo:+.3f}, {hi:+.3f}]")
print("App. E.5: GFlowQEC-online minus the objective swap, per cent of its mean: " + "; ".join(parts))

# ---- The k = 60 class, one stream per code ----
# Over the 6 x 800 sampled visits of each run: n all visits, n60 those with k = 60, h60 those of them
# re-estimated at or below their seed's y*.
K = 60
share, hit = {}, {}
for c in ("d127", "pt", "rnd"):
    n = n60 = h60 = 0
    for s in SEEDS:
        for o in runs[c][s]["orbits"]:
            n += 1
            if int(o["k"]) == K:
                n60 += 1
                if o["ler_pq"] <= q[s]:
                    h60 += 1
    share[c] = 100 * n60 / n
    hit[c] = 100 * h60 / n60
    print(f"basis: {c} k = {K} in {n60}/{n} visits = {share[c]:.1f} %; at or below y* {h60}/{n60}")
lo, hi = min(share.values()), max(share.values())
print(f"App. E.5: k = {K} share of proposals (GFlowQEC-online, tuned PT, uniform): {lo:.0f} to {hi:.0f} per cent")
print(f"App. E.5: inside k = {K}, re-estimates at or below y*: GFlowQEC-online {hit['d127']:.1f}, "
      f"tuned PT {hit['pt']:.1f}, uniform {hit['rnd']:.1f} per cent")

# ---- The best codes: the best of each configuration's 200 highest-scoring codes ----
# Each file holds one run's top 200 codes by search-time score, each re-estimated on 4,000 independent
# shots: fails and k per code, and ler_pq = pq(fails / 4,000, k), the per-qubit rate.
SHOTS = 4000
TOP = 200
ARMS = {"d127": "lsgfn_d127",          # GFlowQEC-online, L = 127, one stream
        "u127": "lsgfn_u127",          # local search alone, L = 127
        "c127": "lsgfn_c127",          # GFlowQEC-online, L = 127, common random numbers
        "pt": "pt",                    # tuned PT, one stream
        "pt+crn": "pt_crn",            # tuned PT, common random numbers
        "ss300": "selfscreen_ss300"}   # the supervised screen trained on the run's own paid calls
rows = {}
for arm, tok in ARMS.items():
    best = None
    for s in SEEDS:
        d = json.load(open(N300 / f"g1_repay_elites_i4000_{tok}_s{s}_top200.json"))
        assert d["n_done"] == TOP and d["indep_shots"] == SHOTS, (arm, s)
        for o in d["orbits"]:
            assert abs(o["ler_pq"] - pq(o["fails"] / SHOTS, o["k"])) < 1e-12   # the per-qubit metric
        o = min(d["orbits"], key=lambda o: o["ler_pq"])
        f_lo, f_hi = wilson(o["fails"], SHOTS)              # Wilson 95 % on fails / shots ...
        seedbest = (o["ler_pq"], pq(f_lo, o["k"]), pq(f_hi, o["k"]), s)   # ... mapped to the per-qubit rate
        if best is None or seedbest[0] < best[0]:
            best = seedbest
    rows[arm] = best
    print(f"basis: {arm} best {best[0]:.6f} [{best[1]:.6f}, {best[2]:.6f}] seed {best[3]}")
assert len(rows) == 6
lo = min(b[0] for b in rows.values())
hi = max(b[0] for b in rows.values())
# the six intervals share a point when the highest lower end is at most the lowest upper end
overlap = max(b[1] for b in rows.values()) <= min(b[2] for b in rows.values())
print(f"App. E.5: configurations compared: {len(rows)}, top {TOP} codes each")
print(f"App. E.5: lowest re-estimated rates: {lo:.5f} to {hi:.5f} per logical qubit; "
      f"intervals overlap: {'yes' if overlap else 'no'}")

# ---- The best codes at 200 calls against GFlowQEC-online at 40,000 calls ----
# A column's champion is the best code of the run's first n calls by the search-time score
# s = -log(ler_pq + 1 / (k x shots)), the objective every method maximises (not argmin ler_pq: the two
# differ when k differs). Each champion was re-estimated on 4,000 independent shots; the results are
# stored in bo_rescore_cache.json under "seed:" + the code's representative.
COLUMNS = [("BO@200", "mitten_lersearch_bo_C10xS3_b0.35_t0.05-0.67_wC_s{s}_bo-onehot.json", 200),
           ("uniform@200", "mitten_lersearch_random_C10xS3_b0.35_t0.05-0.67_w_s{s}.json", 200),
           ("tunedPT@200", "mitten_lersearch_pt_C10xS3_b0.35_t0.01-0.15_w_s{s}.json", 200),
           ("c127@40k", "mitten_lersearch_lsgfn_C10xS3_b0.8_t0.05-0.67_w_k1i127detC_s{s}_c127.json", 40_000)]
cache = json.load(open(N300 / "bo_rescore_cache.json"))


def champion(path, n_calls):
    """The best paid code in a run's first n_calls, by the search-time score s."""
    d = json.load(open(path))
    assert d["shots"] == 1000 and d["p"] == 0.07 and d.get("run_seed", d["seed"]) == d["seed"]
    n = min(n_calls, d["calls"])
    assert n == n_calls, f"{path.name}: only {n:,} of {n_calls:,} calls"
    s = [-math.log(d["ler_pq"][i] + 1.0 / (d["k"][i] * d["shots"])) for i in range(n)]
    i = int(np.argmax(s))
    return d["rep"][i]


def rescored(seed, rep):
    """The stored 4,000-shot re-estimate of one champion: its per-qubit rate and Wilson interval."""
    key = f"{seed}:" + ",".join(str(int(v)) for v in rep)
    r = cache[key]                       # a missing key is a hard stop: this readout never re-estimates
    f_lo, f_hi = wilson(r["fails"], SHOTS)
    return r["ler_pq"], pq(f_lo, r["k"]), pq(f_hi, r["k"])


passed = []
for s in SEEDS:
    bands = {name: rescored(s, champion(N300 / tmpl.format(s=s), n)) for name, tmpl, n in COLUMNS}
    # the seed passes when the four intervals overlap pairwise: highest lower end <= lowest upper end
    if max(b[1] for b in bands.values()) <= min(b[2] for b in bands.values()):
        passed.append(s)
    print(f"basis: seed {s} " + "  ".join(f"{name} {b[0]:.6f} [{b[1]:.6f}, {b[2]:.6f}]"
                                          for name, b in bands.items()))
failed = [s for s in SEEDS if s not in passed]
note = "all seeds" if not failed else "all but seed " + ", ".join(str(s) for s in failed)
print(f"App. E.5: seeds where the four best codes' Wilson intervals overlap: "
      f"{', '.join(str(s) for s in passed)} ({note})")
