"""The n = 300 summary table: good proposals per run at n = 300, mean of seeds 0-5 and 7-10.

For each method and seed the readout reads two stored files:
  - the search run itself (data/n300/mitten_lersearch_*.json): every paid call in order;
  - its re-estimation leg (data/n300/g1_repay_visited_i4000_*.json): 800 of the run's visits drawn
    at random and re-scored on fresh noise, giving each an unbiased logical error rate "ler_pq".

Column 2 is W, the good-proposal count: the run's visit count V times the share of its 800 re-scored
visits whose rate is at or below the reference rate y*. y* is fixed per seed as the exact median of
the re-scored rates of five earlier reference runs pooled (counts.y_star over the cells counts.REFERENCE_CELLS).

Column 3 is W times D_visit, the share of the run's visits left after greedy packing that counts
near-duplicates once (two proposals whose canonical forms differ in at most two group elements).

The two row groups seed the sampled errors differently (independent streams vs common random
numbers), so the bold marks (here "*") compare methods only within a group.

Reads data/n300 only; no oracle call; about 76 s on a laptop. Run from the repo root:
    python -m readouts.sum300
"""

import numpy as np

from gflowqec import counts

# The table's rows in printed order: (label, cell). Cell names index counts.CELLS.
GROUPS = [
    ("One stream per code", [
        ("GFlowQEC-online", "d127"),
        ("Local search alone", "u127"),
        ("Tuned PT", "pt"),
        ("MAP-Elites", "me"),
        ("PPO, entropy weight 0.1", "ppo1"),
        ("Uniform sampling", "rnd"),
        ("Supervised screen, trained on the run's own paid calls", "ss300"),
    ]),
    ("Common random numbers", [
        ("GFlowQEC-online", "c127"),
        ("Tuned PT", "ptc"),
        ("Bayesian optimisation, one-hot encoding", "bo40k"),
    ]),
]
table_cells = [cell for _, rows in GROUPS for _, cell in rows]

# Load the visited files of the table's cells plus the five reference cells that fix y*
# (tuned PT "pt" is in both lists; dict.fromkeys drops the repeat and keeps the order).
# load_visited stops on any re-estimation leg short of its 800 draws or any run short of its budget.
runs = counts.load_visited(list(dict.fromkeys(list(counts.REFERENCE_CELLS) + table_cells)))

# Every table run spent its whole budget, and the budget is the same for all of them.
budgets = set()
for cell in table_cells:
    for seed in counts.SEEDS:
        calls, budget, _ = counts.source_stamp(runs[cell][seed])
        assert calls == budget, f"{cell} s{seed}: {calls} of {budget} paid calls"
        assert runs[cell][seed]["n_sub"] == 800, f"{cell} s{seed}: {runs[cell][seed]['n_sub']} re-estimates"
        budgets.add(budget)
assert len(budgets) == 1, f"the table's runs have different budgets: {budgets}"
budget = budgets.pop()

# Per seed: fix y*, then W and W with near-duplicates counted once for every cell.
w = {cell: [] for cell in table_cells}
w_once = {cell: [] for cell in table_cells}
y_stars = []
for seed in counts.SEEDS:
    q = counts.y_star(runs, seed)          # exact, never rounded (the rates are discrete)
    y_stars.append(q)
    for cell in table_cells:
        visited = runs[cell][seed]
        good = counts.good_count(visited, q)
        w[cell].append(good)
        w_once[cell].append(good * counts.d_visit(counts.source_path(visited)))

# Mean over the ten seeds, then print with the paper's rounding (whole numbers, thousands comma).
print(f"Table E.1 (n = 300): good proposals per run, mean of seeds 0-5 and 7-10, "
      f"{budget:,} paid calls per run")
print("method | good proposals per run | with near-duplicates counted once")
for group, rows in GROUPS:
    print(group)
    mean_w = {cell: np.mean(w[cell]) for _, cell in rows}
    mean_once = {cell: np.mean(w_once[cell]) for _, cell in rows}
    best_w = max(mean_w.values())
    best_once = max(mean_once.values())
    for label, cell in rows:
        mark_w = "*" if mean_w[cell] == best_w else ""
        mark_once = "*" if mean_once[cell] == best_once else ""
        print(f"{label} | {mean_w[cell]:,.0f}{mark_w} | {mean_once[cell]:,.0f}{mark_once}")
print("* best per column within a group (bold in the paper)")
print("basis: y* per seed " + ", ".join(f"{q:.6f}" for q in y_stars)
      + f"; reference cells {', '.join(counts.REFERENCE_CELLS)}")
