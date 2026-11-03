"""The n = 180 summary table: the Inequivalent column pair, seeds 0 to 9, and the ratios of the text.

Each run is read on its first 1,000 paid calls, and a cell counts the certified top-scoring codes the paper counts
(the per-seed table): novel ones only for the methods trained on or started from the stored forms, all of them for the
others, the seeded (warm) rows of tuned PT started from the stored forms never.

Equivalence: two codes are the same object when their coloured Tanner graphs are
isomorphic (qubits relabelled, checks relabelled within their colour); the nauty certificate decides it exactly. The
mirror image of a code (X and Z checks exchanged) is the graph with the two check colours swapped. A class is
the unordered pair (certificate, mirror certificate), so equivalent codes and mirror images fall in one class.
  * methods trained on or started from the stored forms: the classes of the counted codes in which neither the
    certificate nor the mirror certificate is that of a stored form;
  * the other methods: the classes of the counted codes, with no test against the stored forms.
None of the first group is one of the 42 stored runs, so its codes are tested against every stored form.

A printed cell is the mean of the ten per-seed class counts, rounded to a whole number. The ratios are
GFlowQEC-offline's pooled count over the row's, both on 10,000 paid calls, with the Katz interval; the per-seed
ratios over tuned PT give the range over single seeds and the count of runs above two.

Reads only data/n180 through gflowqec.counts: the 42 stored runs (r174_*_cert*.json) for the stored forms, and
every cell of counts.prefix_arms (r174_, r177_screen_ and r177_certified_, r200_, r209_ files). Writes nothing.
About 300 s on a laptop, most of it the 116,972 stored certificates (computed on 4 worker processes). Needs
pynauty. Run from the repo root:
    python -m readouts.iso180
"""
import multiprocessing as mp
import time

import numpy as np

from gflowqec import codes, counts
from gflowqec.stats import katz

t0 = time.time()
PREFIX = counts.PREFIX          # 1,000 paid calls per run
SEEDS = tuple(range(10))
PAID = len(SEEDS) * PREFIX      # paid calls under a pooled count (ten seeds)
SIGNS = ("Z", "X")              # gamma = +1, gamma = -1

POLICY = "GFlowQEC-offline, ranked, mu 10"
PT = "Tuned PT"
PT_STORED = "Tuned PT started from the stored forms"

# The paper's rows, in table order: (row label in the table, row label in counts.prefix_arms).
ROWS = [
    ("GFlowQEC-offline", POLICY),
    ("Supervised screen", "Supervised screen, 10 times"),
    ("Soft Q-learning, trained offline", "Soft Q-learning, trained offline"),
    ("Tuned PT from the stored forms", PT_STORED),
    ("Tuned PT", PT),
    ("MAP-Elites, five bins", "MAP-Elites, 5 bins"),
    ("Uniform sampling", "Uniform sampling"),
    ("BO, one-hot encoding", "BO, one-hot encoding"),
]

# Acceptance tests: the counted codes per seed (the per-seed table) and the pooled class counts behind the printed cells.
EXPECT_COUNTED = {
    "Z": {
        POLICY: [308, 297, 333, 283, 302, 314, 321, 317, 337, 321],
        "Supervised screen, 10 times": [99, 69, 89, 81, 101, 101, 92, 85, 71, 78],
        "Soft Q-learning, trained offline": [256, 325, 313, 303, 325, 316, 353, 327, 370, 286],
        PT_STORED: [106, 147, 113, 124, 115, 113, 103, 114, 90, 165],
        PT: [111, 108, 96, 75, 103, 136, 95, 52, 91, 105],
        "MAP-Elites, 5 bins": [91, 92, 79, 150, 108, 124, 57, 115, 110, 108],
        "Uniform sampling": [10, 9, 14, 16, 13, 9, 6, 9, 15, 14],
        "BO, one-hot encoding": [38, 23, 41, 36, 40, 86, 60, 89, 36, 47],
    },
    "X": {
        POLICY: [294, 321, 311, 298, 342, 307, 296, 321, 338, 346],
        "Supervised screen, 10 times": [84, 85, 94, 77, 93, 80, 86, 87, 86, 68],
        "Soft Q-learning, trained offline": [338, 314, 342, 380, 399, 389, 404, 364, 390, 324],
        PT_STORED: [122, 131, 96, 129, 116, 109, 143, 122, 108, 158],
        PT: [74, 109, 102, 101, 63, 72, 109, 74, 34, 97],
        "MAP-Elites, 5 bins": [78, 95, 87, 96, 97, 92, 91, 97, 87, 102],
        "Uniform sampling": [10, 11, 7, 6, 6, 7, 12, 8, 9, 8],
        "BO, one-hot encoding": [50, 16, 82, 64, 10, 53, 103, 76, 36, 37],
    },
}
EXPECT_CLASSES = {
    "Z": {POLICY: 2214, "Supervised screen, 10 times": 623, "Soft Q-learning, trained offline": 2246, PT_STORED: 791,
          PT: 923, "MAP-Elites, 5 bins": 882, "Uniform sampling": 115, "BO, one-hot encoding": 493},
    "X": {POLICY: 2221, "Supervised screen, 10 times": 593, "Soft Q-learning, trained offline": 2592, PT_STORED: 838,
          PT: 794, "MAP-Elites, 5 bins": 744, "Uniform sampling": 84, "BO, one-hot encoding": 520},
}


def cert_of(job):
    """The nauty certificate of a code's coloured Tanner graph; job = (ten group-element indices, swap).
    swap=True exchanges the X and Z check colours: the certificate of the code's mirror image."""
    exps, swap = job
    code = codes.build(codes.GROUP_180, np.array(exps))
    Hx, Hz = np.asarray(code.matrix_x) % 2, np.asarray(code.matrix_z) % 2
    return codes.certificate(Hx, Hz, swap)


def counted(row, arm):
    """Which of the cell's first 1,000 paid calls the paper counts: certified top-scoring codes, novel ones only
    for a row above the rule, warm rows never."""
    m = arm.is_top[:PREFIX].copy()
    if row in counts.PREFIX_ABOVE:
        m &= arm.novel[:PREFIX]
    m[:arm.warm] = False
    return m


# The certificate workers are forked BEFORE any run is loaded, so each stays small.
workers = mp.get_context("fork").Pool(4)

# ------------------------------------------------------------------ the stored forms and their certificates
# One certificate per distinct stored canonical form (116,972). A canonical tuple is a translate of the stored
# code, so its Tanner graph is isomorphic and its certificate the same.
stored = set(workers.map(cert_of, [(r["traj"], False) for r in counts.stored_forms().values()], chunksize=64))

# ------------------------------------------------------------------ the cells, seeds 0 to 9
# arms[sign][row][seed] is a LedgerArm; deployment cells keep all their rows, so counts are cut to 1,000 paid calls
arms = {}
for sign in SIGNS:
    all_rows = counts.prefix_arms(sign)
    arms[sign] = {}
    for _, row in ROWS:
        by_seed = {a.seed: a for a in all_rows[row]}
        assert all(s in by_seed for s in SEEDS), f"{sign} {row}: seeds {sorted(by_seed)}"
        arms[sign][row] = {s: by_seed[s] for s in SEEDS}
    del all_rows

# certificates (plain and mirror) of every counted code
reps = sorted({tuple(arms[sign][row][s].rows[i]["rep"])
               for sign in SIGNS for _, row in ROWS for s in SEEDS
               for i in np.flatnonzero(counted(row, arms[sign][row][s]))})
jobs = [(rep, swap) for rep in reps for swap in (False, True)]
cert = dict(zip(jobs, workers.map(cert_of, jobs, chunksize=16)))
workers.close()


def classes(sign, row, seed):
    """The inequivalent classes among one cell's counted codes. Above the rule a class counts only when neither
    the code nor its mirror image is equivalent to a stored form."""
    arm = arms[sign][row][seed]
    out = set()
    for i in np.flatnonzero(counted(row, arm)):
        rep = tuple(arm.rows[i]["rep"])
        h, hm = cert[(rep, False)], cert[(rep, True)]
        if row in counts.PREFIX_ABOVE and (h in stored or hm in stored):
            continue
        out.add(tuple(sorted((h, hm))))
    return len(out)


# n_counted[sign][row] and n_classes[sign][row]: per-seed counts in seed order
n_counted, n_classes = {}, {}
for sign in SIGNS:
    n_counted[sign] = {row: [int(counted(row, arms[sign][row][s]).sum()) for s in SEEDS] for _, row in ROWS}
    n_classes[sign] = {row: [classes(sign, row, s) for s in SEEDS] for _, row in ROWS}

# ------------------------------------------------------------------ acceptance tests (fail before printing)
for sign in SIGNS:
    assert n_counted[sign] == EXPECT_COUNTED[sign], f"{sign}: counted codes {n_counted[sign]}"
    pooled = {row: sum(v) for row, v in n_classes[sign].items()}
    assert pooled == EXPECT_CLASSES[sign], f"{sign}: class counts {pooled}"

# ------------------------------------------------------------------ the Inequivalent column pair
print("Table 3 (n = 180): Inequivalent, mean of seeds 0-9, first 1,000 paid calls | gamma = +1 | gamma = -1")
for paper_row, row in ROWS:
    cells = [str(int(sum(n_classes[sign][row]) / len(SEEDS) + 0.5)) for sign in SIGNS]   # half up, as sum180
    print(" | ".join([paper_row] + cells))
for paper_row, row in ROWS:
    for sign in SIGNS:
        print(f"basis: {paper_row} {sign}: counted {n_counted[sign][row]}, classes {n_classes[sign][row]}")

# ------------------------------------------------------------------ the ratios of the text
print("Sec. 7 and App. D.1 text: GFlowQEC-offline over the row, inequivalent codes, pooled over seeds 0-9 "
      "| gamma = +1 | gamma = -1")
# App. D.1 prints the ratio over tuned PT to two decimals, Sec. 7 the other one to one
for label, row, digits in (("Over tuned PT", PT, 2), ("Over tuned PT from the stored forms", PT_STORED, 1)):
    cells = []
    for sign in SIGNS:
        r, lo, hi = katz(sum(n_classes[sign][POLICY]), PAID, sum(n_classes[sign][row]), PAID)
        cells.append(f"{r:.{digits}f} [{lo:.2f}, {hi:.2f}]")
    print(" | ".join([label] + cells))
# the per-seed ratios over tuned PT: their range (App. D.1) and the runs above two (Sec. 7: "19 of its 20 runs")
ratios = {sign: [a / b for a, b in zip(n_classes[sign][POLICY], n_classes[sign][PT])] for sign in SIGNS}
print(" | ".join(["Over tuned PT, range over single seeds"]
                 + [f"{min(ratios[sign]):.2f} to {max(ratios[sign]):.2f}" for sign in SIGNS]))
above = sum(r > 2 for sign in SIGNS for r in ratios[sign])
print(f"Over tuned PT, runs above two (both biases) | {above} of {len(SIGNS) * len(SEEDS)}")
print(f"timed: {time.time() - t0:.0f} s")
