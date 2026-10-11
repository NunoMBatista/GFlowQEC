"""Table III (Sec. 8): the comparisons at 180 qubits, mean of three seeds.

For each method and each noise bias (gamma = +1, stored as sign "Z"; gamma = -1, sign "X") the table counts,
in each run's first 1,000 paid calls:
  - certified top-scoring codes: codes whose exact score reaches the bar (Sec. 4.3);
  - neighbourhoods: the distinct neighbourhoods those codes fall in (codes.half_class), novel or not.

The first group of rows used the stored evaluations of the 42 earlier runs (trained on them, or started
from them), so only NOVEL top-scoring codes count for it: codes whose canonical form no stored run paid.
The second group searched without stored data, so every certified top-scoring code counts.

Each printed cell is the mean over seeds 0, 1, 2 of the per-seed count, rounded to a whole number (a mean
of three integers never ends in .5, so no tie rule is needed). The per-seed counts are Table IX.

Reads only data/ (through gflowqec.counts); no oracle call, nothing written; about 37 s on a laptop. Run from
the repo root:
    python -m readouts.sum180
"""

import time

from gflowqec import counts

t0 = time.time()
SEEDS = (0, 1, 2)

# The paper's rows, in table order: (row label in Table III, row label in counts.prefix_arms).
# The first four are the methods that used the stored evaluations (novel codes only); the last four searched
# without stored data (every certified top-scoring code counts).
NOVEL_ROWS = [
    ("GFlowQEC-offline", "GFlowQEC-offline, ranked, mu 10"),          # draws 10x, pays its top 1,000
    ("Supervised screen", "Supervised screen, 10 times"),             # ranks 10x uniform candidates
    ("Soft Q-learning, trained offline", "Soft Q-learning, trained offline"),
    ("Tuned PT started from the stored forms", "Tuned PT started from the stored forms"),
]
ALL_ROWS = [
    ("Tuned PT", "Tuned PT"),
    ("MAP-Elites, five bins", "MAP-Elites, 5 bins"),
    ("Uniform sampling", "Uniform sampling"),                         # a separate run per bias (counts.prefix_rivals)
    ("BO, one-hot encoding", "BO, one-hot encoding"),
]


def cell_counts(arm, novel_only):
    """(top-scoring codes, neighbourhoods) of one run on its first 1,000 paid calls.

    Rows seeded from the stored forms (arm.warm, only for tuned PT started from them) are paid but never
    counted. A neighbourhood counts whether or not its codes are novel."""
    # every cell here holds exactly the 1,000-call prefix, all of whose proxy-top rows were certified
    assert arm.n == counts.PREFIX and arm.uncertified == 0, f"{arm.source}: {arm.n} rows, {arm.uncertified} open"
    hit = arm.is_top & arm.novel if novel_only else arm.is_top
    tops = int(hit[arm.warm:].sum())
    hoods = len(arm.classes())          # classes() already leaves the warm rows out
    return tops, hoods


# table[row] = [tops +1, tops -1, hoods +1, hoods -1], each a list of three per-seed counts
table = {}
for col, sign in enumerate(("Z", "X")):
    # every row of Table IX at this sign; inside, counts.ledger_pool asserts that the stored pool is
    # exactly 42 runs and 126,000 paid calls, the "42 earlier runs" of the header and the table note
    arms = counts.prefix_arms(sign)
    for rows, novel_only in ((NOVEL_ROWS, True), (ALL_ROWS, False)):
        for paper_row, prefix_row in rows:
            per_seed = {arm.seed: cell_counts(arm, novel_only) for arm in arms[prefix_row]}
            assert sorted(per_seed) == list(SEEDS), f"{paper_row} at {sign}: seeds {sorted(per_seed)}"
            cells = table.setdefault(paper_row, [None] * 4)
            cells[col] = [per_seed[s][0] for s in SEEDS]          # top-scoring codes
            cells[2 + col] = [per_seed[s][1] for s in SEEDS]      # neighbourhoods


def mean3(xs):
    """Mean of the three seeds, rounded to a whole number as printed."""
    return round(sum(xs) / 3)


print("Table III (n = 180, Sec. 8): first 1,000 paid calls, mean of seeds 0-2, "
      "columns tops +1 | tops -1 | neighbourhoods +1 | neighbourhoods -1")
print("-- trained on, or started from, the stored evaluations of 42 earlier runs: novel certified top-scoring codes --")
for paper_row, _ in NOVEL_ROWS:
    print(" | ".join([paper_row] + [str(mean3(c)) for c in table[paper_row]]))
print("-- searching without stored data: every certified top-scoring code counts --")
for paper_row, _ in ALL_ROWS:
    print(" | ".join([paper_row] + [str(mean3(c)) for c in table[paper_row]]))

# the bold cells: the row with the largest rounded mean in each column, over both groups
best = [max(table, key=lambda row: mean3(table[row][j])) for j in range(4)]
print(" | ".join(["bold, best per column"] + best))
print(f"timed: {time.time() - t0:.0f} s")
