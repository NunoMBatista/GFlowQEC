"""The n = 180 results text of the paper: every measured number in it that no table prints at the same rounding.
Lines that begin "Not in the paper:" are further reads of the same cells that the paper does not print; lines that
begin "Table D.1 ..., rounded:" restate a value of the per-seed table (readouts/res_prefix.py) at the text's rounding.

Each printed line is one claim of the text, `label | gamma = +1 | gamma = -1`, in the text's order, direction and
rounding. Sign "Z" is gamma = +1 and sign "X" is gamma = -1.

How the paper counts (experimental setup and the per-seed n = 180 table):
  * every run is read on its first 1,000 paid calls; every row holds seeds 0-9, except the pilot screen at
    100 times (seeds 1-9);
  * a pooled count is the SUM of the per-seed counts (never a union across seeds), and a ratio of two rows
    pools both over the seeds the two rows share, so it has 1,000 paid calls per shared seed under each side;
    the processor-second lines stay on seeds 0-2, because seeds 3-9 ran on different machines;
  * methods that saw the stored forms (GFlowQEC-offline, its controls, the screens, tuned PT started from the
    stored forms) count only NOVEL certified top-scoring codes, those whose canonical form no stored run paid;
    every other method counts all its certified top-scoring codes;
  * tuned PT started from the stored forms pays its seeded (warm) rows first; they are never counted;
  * a neighbourhood is the gauge class of a code's bottleneck half (codes.half_class).

Equivalence (the benchmark section): two codes are the same object when their coloured Tanner graphs are isomorphic
(qubits relabelled, checks relabelled within their colour); the nauty certificate decides it exactly. A counted
code is EQUIVALENCE-FREE when neither it nor its mirror image (X and Z exchanged) is equivalent to a stored form.
Within a cell, equivalent codes and mirror images are counted once.

Reads only data/ through the gflowqec library; writes nothing. About 282 s on a laptop, most of it the 116,972
stored certificates (computed on 4 worker processes). Run from the repo root:
    python -m readouts.text180
"""
import multiprocessing as mp

import numpy as np

from gflowqec import codes, counts
from gflowqec.oracle import bias_score, key_top, lam_of
from gflowqec.stats import katz

PREFIX = counts.PREFIX          # 1,000 paid calls per run
SEEDS3 = (0, 1, 2)              # the seeds whose processor seconds are compared (seeds 3-9 ran on different machines)
SEEDS10 = tuple(range(10))      # the seeds every row holds, the pilot screen at 100 times excepted (1-9)
PAID10 = len(SEEDS10) * PREFIX  # paid calls under a pooled count on seeds 0-9
SIGNS = ("Z", "X")              # gamma = +1, gamma = -1

# Per-seed table rows used below (paper row labels of counts.PREFIX_LEDGER and counts.prefix_rivals)
POLICY = "GFlowQEC-offline, ranked, mu 10"                  # seeds 0-9
SOFTQ = "Soft Q-learning, trained offline"
SCREEN = "Supervised screen, 10 times"                      # seeds 0-9
RANKER = "Policy as a ranker"
PT = "Tuned PT"
PT_STORED = "Tuned PT started from the stored forms"
ME5 = "MAP-Elites, 5 bins"
UNIFORM = "Uniform sampling"
DEPTH7 = "GFlowQEC-online, depth 7"
RANKED_100 = "Ranked, mu 100"                               # seeds 0-9
SCREEN_100 = "Supervised screen, 100 times (matched)"       # seeds 0-9
PILOT_100 = "Pilot screen, 100 times"                      # seeds 1-9

# The first claim compares GFlowQEC-offline with every other method of the summary table (Table 3);
# soft Q-learning trained offline is the stated exception.
TABLE3_OTHERS = (SCREEN, PT_STORED, PT, ME5, UNIFORM, "BO, one-hot encoding")

# Acceptance tests on seeds 0-2: the pooled counts behind the printed ratios (the per-seed table, summed).
EXPECT = {
    "Z": {"novel policy": 938, "novel softq": 894, "all PT": 315, "novel PT stored": 366, "all ME5": 262,
          "all uniform": 33, "novel screen": 257, "novel ranker": 188,
          "nbhd policy": 570, "nbhd PT": 73, "nbhd screen": 224, "stored top nbhd": 346,
          "ladder policy": 1328, "ladder PT": 401, "ladder screen": 349},
    "X": {"novel policy": 926, "novel softq": 994, "all PT": 285, "novel PT stored": 349, "all ME5": 260,
          "all uniform": 28, "novel screen": 263, "novel ranker": 205,
          "nbhd policy": 564, "nbhd PT": 63, "nbhd screen": 227, "stored top nbhd": 334,
          "ladder policy": 1339, "ladder PT": 365, "ladder screen": 359},
}
# ... and the per-seed equivalence-free counts (FOLD for tuned PT: equivalent codes and mirrors counted once).
EXPECT_EQ = {
    "Z": {
        POLICY: [214, 208, 233, 204, 210, 227, 225, 233, 231, 229],
        PT: [105, 100, 88, 74, 100, 132, 93, 48, 84, 99],
        SCREEN: [73, 50, 58, 62, 66, 71, 67, 68, 50, 58],
        PT_STORED: [65, 97, 77, 92, 81, 76, 71, 62, 56, 114],
        SOFTQ: [182, 227, 218, 221, 230, 219, 252, 242, 256, 199],
    },
    "X": {
        POLICY: [217, 236, 226, 213, 236, 201, 215, 225, 229, 223],
        PT: [71, 100, 95, 98, 62, 70, 102, 72, 33, 91],
        SCREEN: [64, 57, 66, 45, 61, 64, 60, 64, 66, 46],
        PT_STORED: [92, 89, 60, 90, 86, 62, 92, 86, 77, 104],
        SOFTQ: [244, 231, 249, 283, 282, 283, 284, 253, 260, 223],
    },
}
# ... and the hundredfold rows' novel counts at seeds 1, 2 (the per-seed table, as readouts/res_prefix.py prints them).
# X seed 1 of the ranked deployment is the primary 463, its own 1,000 paid calls; the top-up reads 464
# and the ratio over the screen is 0.62 either way (res_prefix prints both).
EXPECT_100 = {
    "Z": {RANKED_100: [338, 400], SCREEN_100: [642, 636], PILOT_100: [293, 266]},
    "X": {RANKED_100: [463, 488], SCREEN_100: [644, 590], PILOT_100: [312, 294]},
}
# ... and the ten-seed exact-ladder runs behind the ladder line (readouts/cost180.py checks the same counts).
EXPECT_LADDER10 = {"Z": {"policy": 4409, "PT": 1247}, "X": {"policy": 4534, "PT": 1068}}


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


def ladder_runs(arm):
    """Paid calls whose code ran the exact distance ladders: certified, with certification seconds charged."""
    return sum(r["cert"] is not None and r["cpu"] > 0 for r in arm.rows[:PREFIX])


def ci(k, f, flo, fhi):
    """A Katz ratio and its interval at the paper's digits."""
    r, lo, hi = k
    return f"{r:{f}} [{lo:{flo}}, {hi:{fhi}}]"


def yes(b):
    return "yes" if b else "no"


# The certificate workers are forked BEFORE any run is loaded, so each stays small.
workers = mp.get_context("fork").Pool(4)

# ------------------------------------------------------------------ the stored forms and their certificates
# One certificate per distinct stored canonical form (116,972). A canonical tuple is a translate of the stored
# code, so its Tanner graph is isomorphic and its certificate the same.
stored = set(workers.map(cert_of, [(r["traj"], False) for r in counts.stored_forms().values()], chunksize=64))

# ------------------------------------------------------------------ the stored certified top-scoring neighbourhoods
# Per sign, the neighbourhoods of every certified top-scoring code of the 42 stored runs (all of them; a row
# certified at the other sign keeps its exact distances, scored here at this sign's bar).
runs, _ = counts.load_dumps()
pool_dumps = [d for d in runs if counts.kind_of(d) == "[cert]"]
assert len(pool_dumps) == 42, f"{len(pool_dumps)} stored runs"
stored_nbhd = {}
for sign in SIGNS:
    lam = lam_of(10, sign)
    top = key_top(lam)[0]
    stored_nbhd[sign] = set()
    for d in pool_dumps:
        for r in counts.rep_rows(d):
            proxy_top = bias_score(r["dA"], r["dB"], lam) >= top or not (r["exactA"] and r["exactB"])
            if r["cert"] is not None and proxy_top and bias_score(r["dX"], r["dZ"], lam) >= top:
                stored_nbhd[sign].add(codes.half_class(r["rep"], sign))
del runs, pool_dumps

# ------------------------------------------------------------------------ every row of the per-seed table, per sign
# arms[sign][row][seed] is a LedgerArm (counts.prefix_arms); deployment cells keep all their rows, so every
# count below is cut to the first 1,000 paid calls.
arms = {}
for sign in SIGNS:
    arms[sign] = {row: {a.seed: a for a in cells} for row, cells in counts.prefix_arms(sign).items()}


def count(sign, row, seed):
    """The row's counted codes in one seed's first 1,000 paid calls."""
    return int(counted(row, arms[sign][row][seed]).sum())


def shared(sign, a, b):
    """The seeds rows a and b both hold, in order: a ratio of the two pools both over these."""
    return sorted(set(arms[sign][a]) & set(arms[sign][b]))


def per_seed(sign, row, seeds):
    """The row's counted codes per seed, for the given seeds in order."""
    return [count(sign, row, s) for s in seeds]


def pooled(sign, row, seeds):
    return sum(per_seed(sign, row, seeds))


def over(sign, a, b):
    """Katz ratio of row a's pooled count over row b's, both over the seeds they share."""
    seeds = shared(sign, a, b)
    paid = len(seeds) * PREFIX
    return katz(pooled(sign, a, seeds), paid, pooled(sign, b, seeds), paid)


def nbhd(sign, row, seeds):
    """Neighbourhoods among the row's certified top-scoring codes, summed over the given seeds."""
    return sum(len(arms[sign][row][s].classes(PREFIX)) for s in seeds)


def ladder(sign, row, seeds=SEEDS3):
    """Paid calls that ran the exact ladders, summed over the given seeds (default 0-2)."""
    return sum(ladder_runs(arms[sign][row][s]) for s in seeds)


def ladder_seconds(sign, row):
    """Processor seconds of the exact ladders in the row's first 1,000 paid calls, summed over seeds 0-2
    (the same sum as readouts/cost180.py's ladder_cost)."""
    total = 0.0
    for s in SEEDS3:
        for r in arms[sign][row][s].rows[:PREFIX]:
            if r["cert"] is not None and r["cpu"] > 0:
                total += r["cpu"]
    return total


def seconds(sign, row):
    """Oracle processor seconds of the row's first 1,000 paid calls, summed over seeds 0-2."""
    return sum(arms[sign][row][s].sec[:PREFIX].sum() for s in SEEDS3)


# ------------------------------------------------------------------ equivalence: certificates of the counted codes
EQUIV_ROWS = (POLICY, PT, SCREEN, PT_STORED, SOFTQ)
reps = sorted({tuple(arms[sign][row][s].rows[i]["rep"])
               for sign in SIGNS for row in EQUIV_ROWS for s in SEEDS10
               for i in np.flatnonzero(counted(row, arms[sign][row][s]))})
jobs = [(rep, swap) for rep in reps for swap in (False, True)]
cert = dict(zip(jobs, workers.map(cert_of, jobs, chunksize=16)))
workers.close()


def fold_and_free(sign, row, seed):
    """(FOLD, EF) of one cell: FOLD = the counted codes with equivalent codes and mirror images counted once;
    EF = those of them equivalent to no stored form, mirror image included. A class key is the unordered pair
    (certificate, mirror certificate). Tuned PT is one of the stored runs, so only its FOLD is used."""
    arm = arms[sign][row][seed]
    fold, free = set(), set()
    for i in np.flatnonzero(counted(row, arm)):
        rep = tuple(arm.rows[i]["rep"])
        h, hm = cert[(rep, False)], cert[(rep, True)]
        cls = tuple(sorted((h, hm)))
        fold.add(cls)
        if h not in stored and hm not in stored:
            free.add(cls)
    return len(fold), len(free)


# eq[sign][row] = per-seed counts the text uses: FOLD for tuned PT (it never saw the stored codes, so all its codes
# count), EF for every method that saw them
eq = {}
for sign in SIGNS:
    eq[sign] = {}
    for row in EQUIV_ROWS:
        fold_free = [fold_and_free(sign, row, s) for s in SEEDS10]
        eq[sign][row] = [fold if row == PT else free for fold, free in fold_free]

# ------------------------------------------------------------------ acceptance tests (fail before printing)
for sign in SIGNS:
    for row in arms[sign]:
        want = list(range(1, 10)) if row == PILOT_100 else list(SEEDS10)
        assert sorted(arms[sign][row]) == want, f"{sign} {row}: seeds {sorted(arms[sign][row])}"
    got = {"novel policy": pooled(sign, POLICY, SEEDS3), "novel softq": pooled(sign, SOFTQ, SEEDS3),
           "all PT": pooled(sign, PT, SEEDS3), "novel PT stored": pooled(sign, PT_STORED, SEEDS3),
           "all ME5": pooled(sign, ME5, SEEDS3), "all uniform": pooled(sign, UNIFORM, SEEDS3),
           "novel screen": pooled(sign, SCREEN, SEEDS3), "novel ranker": pooled(sign, RANKER, SEEDS3),
           "nbhd policy": nbhd(sign, POLICY, SEEDS3), "nbhd PT": nbhd(sign, PT, SEEDS3),
           "nbhd screen": nbhd(sign, SCREEN, SEEDS3), "stored top nbhd": len(stored_nbhd[sign]),
           "ladder policy": ladder(sign, POLICY), "ladder PT": ladder(sign, PT), "ladder screen": ladder(sign, SCREEN)}
    assert got == EXPECT[sign], f"{sign}: got {got}"
    assert eq[sign] == EXPECT_EQ[sign], f"{sign}: equivalence counts {eq[sign]}"
    got_100 = {row: per_seed(sign, row, [1, 2]) for row in EXPECT_100[sign]}
    assert got_100 == EXPECT_100[sign], f"{sign}: hundredfold counts {got_100}"
    got_l10 = {"policy": ladder(sign, POLICY, SEEDS10), "PT": ladder(sign, PT, SEEDS10)}
    assert got_l10 == EXPECT_LADDER10[sign], f"{sign}: ten-seed ladder runs {got_l10}"

# ------------------------------------------------------------------ the text, in order
out = {sign: {} for sign in SIGNS}
for sign in SIGNS:
    o = out[sign]
    pol = pooled(sign, POLICY, SEEDS10)         # GFlowQEC-offline on seeds 0-9

    # GFlowQEC-offline ahead of every other row of Table 3, seed by seed
    o["ahead"] = yes(all(count(sign, POLICY, s) > count(sign, row, s) for row in TABLE3_OTHERS for s in SEEDS10))

    # soft Q-learning trained offline over the policy (the inverse of the per-seed table's soft-Q ratio)
    sq = over(sign, SOFTQ, POLICY)
    o["softq"] = ci(sq, ".2f", ".2f", ".2f") if sign == "Z" else ci(sq, ".2f", ".3f", ".2f")
    seeds = shared(sign, SOFTQ, POLICY)
    ahead = sum(count(sign, SOFTQ, s) > count(sign, POLICY, s) for s in seeds)
    o["softq seeds"] = f"{ahead} of {len(seeds)}"

    o["pt"] = ci(over(sign, POLICY, PT), ".1f", ".1f", ".1f")
    o["pt stored"] = ci(over(sign, POLICY, PT_STORED), ".1f", ".1f", ".1f")
    # both sides pay 10,000 calls (seeds 0-9), so a ratio of counts is the ratio of rates
    o["me5"] = format(pol / pooled(sign, ME5, SEEDS10), ".1f")
    o["uniform"] = format(pol / pooled(sign, UNIFORM, SEEDS10), ".0f")
    o["depth7"] = format(np.mean(per_seed(sign, DEPTH7, SEEDS10)), ".0f")

    # equivalence on seeds 0-9 (EF for the methods that saw the stored forms, FOLD for tuned PT)
    ef = {row: sum(v) for row, v in eq[sign].items()}
    ef3 = {row: sum(v[:3]) for row, v in eq[sign].items()}     # the processor-second line stays on seeds 0-2
    o["ef share"] = format(100 * ef[POLICY] / pol, ".0f")
    k = katz(ef[POLICY], PAID10, ef[PT], PAID10)
    o["ef pt"] = ci(k, ".1f", ".2f", ".1f") if sign == "Z" else ci(k, ".1f", ".1f", ".1f")
    o["ef pt seeds"] = yes(all(a / b > 2 for a, b in zip(eq[sign][POLICY], eq[sign][PT])))
    o["ef pt stored"] = format(ef[POLICY] / ef[PT_STORED], ".1f")
    o["ef screen"] = format(ef[POLICY] / ef[SCREEN], ".1f")
    _, lo, hi = katz(ef[POLICY], PAID10, ef[SOFTQ], PAID10)
    o["ef softq"] = yes(lo <= 1 <= hi)

    # the supervised screen at 10 times, seeds 0-9: the pooled ratio, and in how many seeds the policy led
    o["screen"] = ci(over(sign, POLICY, SCREEN), ".1f", ".1f", ".1f")
    seeds = shared(sign, POLICY, SCREEN)
    led = sum(count(sign, POLICY, s) > count(sign, SCREEN, s) for s in seeds)
    o["screen seeds"] = f"{led} of {len(seeds)}"

    # the hundredfold oversample, seeds 0-9: the policy ranked at mu = 100 over the supervised screen at
    # 100 times, both counted as novel codes (note a of the per-seed table), pooled over 1,000 paid calls per seed each
    seeds = shared(sign, RANKED_100, SCREEN_100)
    r100, s100 = per_seed(sign, RANKED_100, seeds), per_seed(sign, SCREEN_100, seeds)
    o["100x counts"] = f"{', '.join(map(str, r100))} against {', '.join(map(str, s100))}"
    o["100x"] = ci(over(sign, RANKED_100, SCREEN_100), ".2f", ".2f", ".2f")
    o["100x seeds"] = f"{sum(s > r for r, s in zip(r100, s100))} of {len(seeds)}"
    # not in the paper: the screen at 100 times is also ahead of the policy at mu = 10
    o["100x over policy"] = yes(all(count(sign, SCREEN_100, s) > count(sign, POLICY, s)
                                    for s in shared(sign, SCREEN_100, POLICY)))

    o["ranker"] = format(pooled(sign, RANKER, SEEDS10) / pol, ".3f")

    # neighbourhoods, each ratio over the seeds the two rows share
    o["nbhd pt"] = format(nbhd(sign, POLICY, SEEDS10) / nbhd(sign, PT, SEEDS10), ".1f")
    seeds = shared(sign, POLICY, SCREEN)
    o["nbhd screen"] = format(nbhd(sign, POLICY, seeds) / nbhd(sign, SCREEN, seeds), ".1f")
    # per policy run: the share of its neighbourhoods that also hold a stored certified top-scoring code
    o["nbhd known"] = {s: 100 * len(arms[sign][POLICY][s].classes(PREFIX) & stored_nbhd[sign])
                       / len(arms[sign][POLICY][s].classes(PREFIX)) for s in range(10)}

    # processor seconds of the oracle and the exact ladders, seeds 0-2
    o["per second"] = (f"{1000 * ef3[PT] / seconds(sign, PT):.1f} against "
                       f"{1000 * ef3[POLICY] / seconds(sign, POLICY):.1f}")
    # the same rate per exact-ladder processor second: the denominator of the text's "per CPU second of exact ladders"
    o["per ladder second"] = (f"{1000 * ef3[PT] / ladder_seconds(sign, PT):.1f} against "
                              f"{1000 * ef3[POLICY] / ladder_seconds(sign, POLICY):.1f}")
    o["time per call"] = format(seconds(sign, POLICY) / seconds(sign, PT), ".0f")
    o["ladder pt"] = format(ladder(sign, POLICY, SEEDS10) / ladder(sign, PT, SEEDS10), ".1f")
    o["ladder screen"] = (f"{ladder(sign, POLICY, SEEDS10) / pol:.2f} against "
                          f"{ladder(sign, SCREEN, SEEDS10) / pooled(sign, SCREEN, SEEDS10):.2f}")

known3 = [out[sign]["nbhd known"][s] for sign in SIGNS for s in SEEDS3]
known10 = [out[sign]["nbhd known"][s] for sign in SIGNS for s in range(10)]

LINES = [
    ("GFlowQEC-offline ahead of every other method of Table 3 in every paired seed, soft Q-learning trained "
     "offline excepted", "ahead"),
    ("Soft Q-learning trained offline over GFlowQEC-offline", "softq"),
    ("Not in the paper: seeds in which soft Q-learning trained offline was ahead", "softq seeds"),
    ("Table D.1 ratio, rounded: over tuned PT", "pt"),
    ("Table D.1 ratio, rounded: over tuned PT started from the stored forms", "pt stored"),
    ("Table D.1 ratio, rounded: over MAP-Elites, 5 bins", "me5"),
    ("Table D.1 ratio, rounded: over uniform sampling", "uniform"),
    ("Not in the paper: GFlowQEC-online, depth 7, certified top-scoring codes per run", "depth7"),
    ("Not in the paper: per cent of novel top-scoring codes equivalent to no stored code nor its mirror", "ef share"),
    ("Not in the paper: equivalence-free, over tuned PT with equivalent codes and mirrors counted once", "ef pt"),
    ("Not in the paper: equivalence-free, above two in every seed", "ef pt seeds"),
    ("Equivalence-free, over tuned PT started from the stored forms", "ef pt stored"),
    ("Not in the paper: equivalence-free, over the supervised screen", "ef screen"),
    ("Not in the paper: equivalence-free, soft Q-learning trained offline not separated", "ef softq"),
    ("Over the supervised screen", "screen"),
    ("Over the supervised screen: seeds in which GFlowQEC-offline led", "screen seeds"),
    ("Hundredfold oversample, seeds 0-9: novel top-scoring codes per seed, GFlowQEC-offline ranked at mu = 100 "
     "against the supervised screen at 100 times", "100x counts"),
    ("Hundredfold oversample: GFlowQEC-offline ranked at mu = 100 over the supervised screen at 100 times", "100x"),
    ("Hundredfold oversample: seeds in which the supervised screen at 100 times was ahead", "100x seeds"),
    ("Not in the paper: hundredfold oversample, the supervised screen at 100 times ahead of GFlowQEC-offline "
     "in every paired seed", "100x over policy"),
    ("Policy ranking the screen's uniform candidates, over its own draws", "ranker"),
    ("Table D.1 neighbourhood ratio, rounded: over tuned PT", "nbhd pt"),
    ("Table D.1 neighbourhood ratio, rounded: over the supervised screen", "nbhd screen"),
]
LINES_AFTER = [
    ("Equivalence-free per 1,000 oracle processor seconds, seeds 0-2, tuned PT against GFlowQEC-offline", "per second"),
    ("Equivalence-free per 1,000 exact-ladder processor seconds, seeds 0-2, tuned PT against GFlowQEC-offline",
     "per ladder second"),
    ("Not in the paper: processor time per paid call, seeds 0-2, GFlowQEC-offline over tuned PT", "time per call"),
    ("Paid calls that ran the exact ladders, GFlowQEC-offline over tuned PT", "ladder pt"),
    ("Exact-ladder runs per novel certified top-scoring code counted by canonical form, "
     "GFlowQEC-offline against the supervised screen", "ladder screen"),
]

print("Sec. 7 text: n = 180, first 1,000 paid calls, seeds 0-9 (processor seconds on seeds 0-2), "
      "each ratio pooled over the seeds its two rows share")
for label, key in LINES:
    print(f"{label} | {out['Z'][key]} | {out['X'][key]}")
# one range over the policy runs of both signs: the six of seeds 0-2, then the twenty of seeds 0-9
print("Not in the paper: per cent of the policy's neighbourhoods that also hold a stored certified top-scoring "
      f"code, range over its six runs of seeds 0-2 | {min(known3):.1f} to {max(known3):.1f}")
print("Per cent of the policy's neighbourhoods that also hold a stored certified top-scoring code, range over its "
      f"twenty runs of seeds 0-9 | {min(known10):.1f} to {max(known10):.1f}")
for label, key in LINES_AFTER:
    print(f"{label} | {out['Z'][key]} | {out['X'][key]}")
