"""Table X (App. D.2): the mean good-proposal count W at n = 300 for the runs that Table IV leaves out,
and the seed counts quoted in the table's caption.

W of one run at one seed (Eq. 20): the run's number of visits V times the share of 800 visits,
sampled from the run and re-estimated on 4,000 fresh shots, whose per-qubit logical error rate is at
or below y*. y* is fixed per seed as the median rate of five reference runs pooled (counts.y_star),
so every run at that seed is measured against the same bar. A table row is the mean of W over seeds 0-5.

Every run here paid 40,000 calls except PPO at entropy weight 0.01, which stopped early in every seed;
its row prints the range and mean of the paid calls it reached.

Reads data/n300 only; no oracle call; about 6 s on a laptop. Run from the repo root:
    python -m readouts.res300
"""
import numpy as np

from gflowqec.counts import REFERENCE_CELLS, SEEDS, good_count, load_visited, source_stamp, y_star

# The printed rows, in table order: (label, depth L, cell). The cell names the run's visited files
# through counts.CELLS. "--" marks a method without the local-search stream, so without a depth.
ONE_STREAM = [("The plain sampler", "--", "gfn"),
              ("Soft Q-learning", "--", "sdqn"),
              ("PPO, entropy weight 0.01, stopped early", "--", "ppo"),
              ("Local search alone", "31", "u31"),
              ("Objective swap", "31", "p31"),
              ("GFlowQEC-online", "31", "d31")]
COMMON_RANDOM = [("Objective swap", "127", "p127")]

# The caption's comparisons: (GFlowQEC-online, rival), GFlowQEC-online at the rival's depth (127 when
# the rival has none) and seeding mode. The rivals are every other run of this table and of Table IV;
# the plain sampler is not compared. The objective swap is counted apart.
NON_SWAP = [("d127", "sdqn"), ("d127", "ppo"),                  # this table, one stream
            ("d127", "u127"), ("d127", "pt"), ("d127", "me"),   # Table IV, one stream
            ("d127", "ppo1"), ("d127", "rnd"), ("d127", "ss300"),
            ("d31", "u31"),                                     # this table, depth 31
            ("c127", "ptc"), ("c127", "bo40k")]                 # Table IV, common random numbers
SWAP = [("d31", "p31"), ("c127", "p127")]                       # the objective swap at L = 31 and L = 127

# Load the visited files: the five reference cells that fix y*, every printed row and every rival.
# load_visited stops hard on a short visit leg or a search run short of its budget (PPO excepted).
cells = list(REFERENCE_CELLS)
for _, _, c in ONE_STREAM + COMMON_RANDOM:
    cells.append(c)
for a, b in NON_SWAP + SWAP:
    cells += [a, b]
cells = list(dict.fromkeys(cells))           # drop repeats, keep the order
runs = load_visited(cells)

# W per cell and seed, each seed against its own y*
W = {c: {} for c in cells}
for s in SEEDS:
    q = y_star(runs, s)
    for c in cells:
        W[c][s] = good_count(runs[c][s], q)


def mean_w(c):
    """The row value: W averaged over seeds 0-5."""
    return np.mean([W[c][s] for s in SEEDS])


# PPO at entropy weight 0.01: the paid calls each seed's run reached before it stopped
ppo_calls = [source_stamp(runs["ppo"][s])[0] for s in SEEDS]

print("Table X (n = 300): mean W over seeds 0-5, 40,000 paid calls per run")
print("One stream per code")
for label, depth, c in ONE_STREAM:
    if c == "ppo":
        label += f" ({min(ppo_calls):,} to {max(ppo_calls):,} paid calls, mean {np.mean(ppo_calls):,.0f})"
    print(f"{label} | {depth} | {mean_w(c):,.0f}")
print("Common random numbers")
for label, depth, c in COMMON_RANDOM:
    print(f"{label} | {depth} | {mean_w(c):,.0f}")

# The caption: in how many seeds GFlowQEC-online's W is the larger, per comparison
lead = {}
for a, b in NON_SWAP + SWAP:
    lead[(a, b)] = sum(W[a][s] > W[b][s] for s in SEEDS)
# "in all six seeds against every other run": the weakest of the non-swap comparisons is what counts
print(f"caption: GFlowQEC-online had the larger W in {min(lead[p] for p in NON_SWAP)} of 6 seeds "
      f"against every other run except the objective swap")
print(f"caption: against the objective swap it led in {lead[('d31', 'p31')]} of 6 seeds at L = 31 "
      f"and {lead[('c127', 'p127')]} of 6 at L = 127")

for a, b in NON_SWAP + SWAP:
    print(f"basis: {a} over {b} {lead[(a, b)]}/6")
print("basis: PPO paid calls per seed " + ", ".join(f"{n:,}" for n in ppo_calls))
