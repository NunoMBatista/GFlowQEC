"""Sec. 9.2 of the paper (GFlowQEC-online at n = 300): the numbers that only its text prints.

Every number comes from the stored n = 300 runs, seeds 0 to 5, 40,000 paid calls per run:

    W    good proposals of a run, Eq. 20: its visits times the share of its 800 re-estimated
         visits at or below the seed's reference rate y* (the pooled median of the five reference cells);
    D    the share of a run's visits kept by the greedy near-duplicate pass (Hamming distance 2, call order);
    W'   W x D, good proposals with near-duplicates counted once.

Paired comparisons are over the six seeds: the mean difference in per cent of the second method's mean, its
95 % paired t interval and the two-sided p of the paired t test. The table means these sentences restate
(23,230, 21,171 and the others) are printed by readouts/sum300.py, not here.

Reads data/n300 only; no oracle call; about 95 s on a laptop. Run from the repo root:
    python -m readouts.text300
"""
import numpy as np

from gflowqec import counts, stats

# Every cell this section reads (the names are the keys of counts.CELLS):
#   d127 GFlowQEC-online at depth L = 127, one stream per code; c127 the same under common random numbers
#   pt / ptc tuned PT (one stream per code / common random numbers); u127 local search alone at L = 127
#   me MAP-Elites; ppo1 PPO at entropy weight 0.1; ppo PPO at entropy weight 0.01 (stopped early)
#   rnd uniform sampling; ss300 the supervised screen trained on the run's own paid calls
#   gfn the plain sampler; sdqn soft Q-learning; bo40k one-hot Bayesian optimisation (common random numbers)
# The five reference cells (counts.REFERENCE_CELLS) are loaded too, because they fix y* at each seed.
CELLS = list(counts.REFERENCE_CELLS) + ["d127", "c127", "ptc", "u127", "me", "ppo1", "ppo",
                               "rnd", "ss300", "gfn", "sdqn", "bo40k"]
SEEDS = list(counts.SEEDS)

runs = counts.load_visited(CELLS)                 # runs[cell][seed] = the visited file of that run

# W per cell: one value per seed, each against that seed's y*
q = {s: counts.y_star(runs, s) for s in SEEDS}
W = {c: np.array([counts.good_count(runs[c][s], q[s]) for s in SEEDS]) for c in CELLS}

# D per cell (one greedy pass over the run's visits in call order), only for the cells the text needs
D_CELLS = ["d127", "u127", "pt", "me", "rnd", "ppo1", "ppo", "ss300", "gfn", "sdqn", "bo40k"]
D = {c: np.array([counts.d_visit(counts.source_path(runs[c][s])) for s in SEEDS]) for c in D_CELLS}
Wp = {c: W[c] * D[c] for c in D_CELLS}            # W', near-duplicates counted once

# Paid calls of each run, read from its search dump: 40,000 for every complete run, fewer for PPO at 0.01
calls = {c: np.array([counts.source_stamp(runs[c][s])[0] for s in SEEDS]) for c in ("d127", "me", "ppo1", "ppo")}
budget = {c: np.array([counts.source_stamp(runs[c][s])[1] for s in SEEDS]) for c in ("ppo",)}


def interval(a, b):
    """'pct per cent [low, high]' of the paired comparison a - b, plus the seeds where a is higher and p."""
    pct, lo, hi, _, wins, p = stats.paired_t(W[a], W[b])
    return f"{pct:.1f} per cent [{lo:.1f}, {hi:.1f}]", wins, p


def rate(c):
    """Good proposals per paid call of a cell, pooled over the six seeds: sum of W over sum of calls."""
    return W[c].sum() / calls[c].sum()


def seeds_ahead_per_call(c):
    """Seeds in which GFlowQEC-online's W per paid call is higher than cell c's."""
    return int(sum(W["d127"][s] / calls["d127"][s] > W[c][s] / calls[c][s] for s in SEEDS))


def seeds_ahead_dedup(a, b):
    """Seeds in which a's W' is higher than b's (the seed count of the near-duplicate lens)."""
    return int(sum(x > 0 for x in Wp[a] - Wp[b]))


print("Sec. 9.2 text, n = 300, seeds 0-5")

# GFlowQEC-online led tuned PT and local search alone on W in every seed
_, wins_pt, _ = interval("d127", "pt")
_, wins_u, _ = interval("d127", "u127")
print(f"Sec. 9.2: GFlowQEC-online (L = 127) led tuned PT in {wins_pt} of 6 seeds and local search alone in {wins_u} of 6 seeds")

# the lead over tuned PT, in each seeding mode
txt, _, p = interval("d127", "pt")
print(f"Sec. 9.2: GFlowQEC-online minus tuned PT: {txt}, p = {p:.4f}, paired t")
txt, _, _ = interval("c127", "ptc")
print(f"Sec. 9.2: common random numbers, GFlowQEC-online minus tuned PT: {txt}")

# the factors per paid call. PPO at 0.01 stopped early, so its rate is W over its own paid calls;
# for the 40,000-call rivals the factor equals the ratio of the mean W.
for c, name, fmt in (("me", "MAP-Elites", ".2f"), ("ppo1", "PPO, entropy weight 0.1", ".2f"),
                     ("ppo", "PPO, entropy weight 0.01", ".1f")):
    factor = rate("d127") / rate(c)
    print(f"Sec. 9.2: per paid call, GFlowQEC-online over {name}: {factor:{fmt}}, in {seeds_ahead_per_call(c)} of 6 seeds")

# the supervised screen
txt, wins, p = interval("d127", "ss300")
print(f"Sec. 9.2: GFlowQEC-online minus the supervised screen: {txt}, in {wins} of 6 seeds, p = {p:.2f}, paired t")

# local search alone: not separated from tuned PT, and behind GFlowQEC-online
txt, _, _ = interval("u127", "pt")
print(f"Sec. 9.2: local search alone minus tuned PT: {txt}")
txt, wins, p = interval("d127", "u127")
print(f"Sec. 9.2: GFlowQEC-online minus local search alone: {txt}, in {wins} of 6 seeds, p = {p:.3f}, paired t")

# with near-duplicates counted once, the rivals lead GFlowQEC-online
leads = [("pt", "tuned PT"), ("me", "MAP-Elites"), ("rnd", "uniform sampling"),
         ("ppo1", "PPO entropy weight 0.1"), ("ss300", "supervised screen")]
print("Sec. 9.2: near-duplicates counted once, seeds in which each led GFlowQEC-online: "
      + ", ".join(f"{name} {seeds_ahead_dedup(c, 'd127')}" for c, name in leads) + " (of 6)")

# the share kept by the near-duplicate pass. "A method that draws every tuple afresh" is every n = 300
# run in Tables IV and X without a local move: uniform sampling, PPO at both entropy
# weights, the supervised screen, the plain sampler, soft Q-learning and one-hot BO.
fresh = [("rnd", "uniform"), ("ppo1", "PPO 0.1"), ("ss300", "screen"), ("gfn", "plain sampler"),
         ("sdqn", "soft Q"), ("ppo", "PPO 0.01"), ("bo40k", "one-hot BO")]
fresh_D = [D[c].mean() for c, _ in fresh]
print(f"Sec. 9.2: share kept by the near-duplicate pass: GFlowQEC-online {D['d127'].mean():.2f}; "
      f"methods that draw every tuple afresh {min(fresh_D):.2f} to {max(fresh_D):.1f}")

# local search alone runs the same chains, so it keeps the same share
print(f"Sec. 9.2: local search alone keeps {D['u127'].mean():.2f}; GFlowQEC-online led it with near-duplicates "
      f"counted once in {seeds_ahead_dedup('d127', 'u127')} of 6 seeds")

# Supporting lines for worded claims (not diffed against the paper)
short = int(sum(calls["ppo"] < budget["ppo"]))
print(f"basis: PPO at entropy weight 0.01 stopped before {budget['ppo'][0]:,} paid calls in {short} of 6 runs "
      f"({calls['ppo'].min():,} to {calls['ppo'].max():,} paid calls)")
txt, wins, _ = interval("ss300", "pt")
print(f"basis: supervised screen minus tuned PT: {txt}, in {wins} of 6 seeds")
best = max((c for c, _ in leads), key=lambda c: Wp[c].mean())
print(f"basis: near-duplicates counted once, the largest lead over GFlowQEC-online is {dict(leads)[best]}'s: "
      f"{Wp[best].mean():,.0f} against {Wp['d127'].mean():,.0f}")
print("basis: share kept, fresh-draw runs: " + ", ".join(f"{name} {D[c].mean():.3f}" for c, name in fresh))
