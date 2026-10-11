"""Sec. 9.1 of the paper: every measured number in its text at n = 180 that no table prints at the same rounding.

Each printed line is one claim of the text, `label | gamma = +1 | gamma = -1`, in the text's order, direction and
rounding. Sign "Z" is gamma = +1 and sign "X" is gamma = -1.

How the paper counts (Sec. 8, Table IX):
  * every run is read on its first 1,000 paid calls, seeds 0, 1, 2, and a pooled count is the SUM of the three
    seeds' counts (never a union across seeds), so every pooled rate has 3,000 paid calls under it; the
    hundredfold rows (mu = 100, 100 times) hold seeds 1 and 2 only, so their pooled rates have 2,000;
  * methods that saw the stored forms (GFlowQEC-offline, its controls, the screens, tuned PT started from the
    stored forms) count only NOVEL certified top-scoring codes, those whose canonical form no stored run paid;
    every other method counts all its certified top-scoring codes;
  * tuned PT started from the stored forms pays its seeded (warm) rows first; they are never counted;
  * a neighbourhood is the gauge class of a code's bottleneck half (codes.half_class).

Equivalence (Sec. 4.2): two codes are the same object when their coloured Tanner graphs are isomorphic
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
PAID = 3 * PREFIX               # paid calls under a pooled count (three seeds)
PAID_100 = 2 * PREFIX           # ... and under a hundredfold row's pooled count (seeds 1 and 2)
SIGNS = ("Z", "X")              # gamma = +1, gamma = -1

# Table IX rows used below (paper row labels of counts.PREFIX_LEDGER and counts.prefix_rivals)
POLICY = "GFlowQEC-offline, ranked, mu 10"
SOFTQ = "Soft Q-learning, trained offline"
SCREEN = "Supervised screen, 10 times"
RANKER = "Policy as a ranker"
PT = "Tuned PT"
PT_STORED = "Tuned PT started from the stored forms"
ME5 = "MAP-Elites, 5 bins"
UNIFORM = "Uniform sampling"
DEPTH7 = "GFlowQEC-online, depth 7"
RANKED_100 = "Ranked, mu 100"                               # seeds 1, 2
SCREEN_100 = "Supervised screen, 100 times (matched)"       # seeds 1, 2
EARLIER_100 = "Earlier, smaller screen, 100 times"          # seeds 1, 2

# The first claim compares GFlowQEC-offline with every other method; its own deployments are not other methods,
# and soft Q-learning trained offline and the supervised screen at 100 times are the two stated exceptions.
# The earlier, smaller screen at 100 times is compared on the two seeds it holds.
NOT_COMPARED = {POLICY, RANKER, RANKED_100, "Unranked deployment", SOFTQ, SCREEN_100}

# Acceptance tests: the pooled counts behind the printed ratios (Table IX per seed, summed).
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
    "Z": {POLICY: [214, 208, 233], PT: [105, 100, 88], SCREEN: [73, 50, 58], PT_STORED: [65, 97, 77],
          SOFTQ: [182, 227, 218]},
    "X": {POLICY: [217, 236, 226], PT: [71, 100, 95], SCREEN: [64, 57, 66], PT_STORED: [92, 89, 60],
          SOFTQ: [244, 231, 249]},
}
# ... and the hundredfold rows' novel counts per seed 1, 2 (Table IX, as readouts/res_prefix.py prints them).
# X seed 1 of the ranked deployment is the booked 463, its own 1,000 paid calls; the registered top-up reads 464
# and the ratio over the screen is 0.77 either way (res_prefix prints both).
EXPECT_100 = {
    "Z": {RANKED_100: [338, 400], SCREEN_100: [642, 636], EARLIER_100: [293, 266]},
    "X": {RANKED_100: [463, 488], SCREEN_100: [644, 590], EARLIER_100: [312, 294]},
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

# ------------------------------------------------------------------------ every row of Table IX, per sign
# arms[sign][row][seed] is a LedgerArm (counts.prefix_arms); deployment cells keep all their rows, so every
# count below is cut to the first 1,000 paid calls.
arms = {}
for sign in SIGNS:
    arms[sign] = {row: {a.seed: a for a in cells} for row, cells in counts.prefix_arms(sign).items()}


def count(sign, row, seed):
    """The row's counted codes in one seed's first 1,000 paid calls."""
    return int(counted(row, arms[sign][row][seed]).sum())


def per_seed(sign, row):
    """The row's counted codes per seed, in seed order: 0, 1, 2, or 1, 2 for the hundredfold rows."""
    return [count(sign, row, s) for s in sorted(arms[sign][row])]


def pooled(sign, row):
    return sum(per_seed(sign, row))


def nbhd(sign, row):
    """Neighbourhoods among the row's certified top-scoring codes, summed over seeds."""
    return sum(len(arms[sign][row][s].classes(PREFIX)) for s in (0, 1, 2))


def ladder(sign, row):
    return sum(ladder_runs(arms[sign][row][s]) for s in (0, 1, 2))


def seconds(sign, row):
    """Oracle processor seconds of the row's first 1,000 paid calls, summed over seeds."""
    return sum(arms[sign][row][s].sec[:PREFIX].sum() for s in (0, 1, 2))


# ------------------------------------------------------------------ equivalence: certificates of the counted codes
EQUIV_ROWS = (POLICY, PT, SCREEN, PT_STORED, SOFTQ)
reps = sorted({tuple(arms[sign][row][s].rows[i]["rep"])
               for sign in SIGNS for row in EQUIV_ROWS for s in (0, 1, 2)
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
        fold_free = [fold_and_free(sign, row, s) for s in (0, 1, 2)]
        eq[sign][row] = [fold if row == PT else free for fold, free in fold_free]

# ------------------------------------------------------------------ acceptance tests (fail before printing)
for sign in SIGNS:
    got = {"novel policy": pooled(sign, POLICY), "novel softq": pooled(sign, SOFTQ), "all PT": pooled(sign, PT),
           "novel PT stored": pooled(sign, PT_STORED), "all ME5": pooled(sign, ME5),
           "all uniform": pooled(sign, UNIFORM), "novel screen": pooled(sign, SCREEN),
           "novel ranker": pooled(sign, RANKER), "nbhd policy": nbhd(sign, POLICY), "nbhd PT": nbhd(sign, PT),
           "nbhd screen": nbhd(sign, SCREEN), "stored top nbhd": len(stored_nbhd[sign]),
           "ladder policy": ladder(sign, POLICY), "ladder PT": ladder(sign, PT), "ladder screen": ladder(sign, SCREEN)}
    assert got == EXPECT[sign], f"{sign}: got {got}"
    assert eq[sign] == EXPECT_EQ[sign], f"{sign}: equivalence counts {eq[sign]}"
    for row in EXPECT_100[sign]:
        assert sorted(arms[sign][row]) == [1, 2], f"{sign} {row}: seeds {sorted(arms[sign][row])}"
    got_100 = {row: per_seed(sign, row) for row in EXPECT_100[sign]}
    assert got_100 == EXPECT_100[sign], f"{sign}: hundredfold counts {got_100}"

# ------------------------------------------------------------------ the text, in order
out = {sign: {} for sign in SIGNS}
for sign in SIGNS:
    o = out[sign]
    pol = pooled(sign, POLICY)
    pol_seed = per_seed(sign, POLICY)

    # GFlowQEC-offline ahead of every other Table IX row, seed by seed over the seeds the row holds
    o["ahead"] = yes(all(count(sign, POLICY, s) > count(sign, row, s)
                         for row in arms[sign] if row not in NOT_COMPARED for s in arms[sign][row]))

    # soft Q-learning trained offline over the policy (the inverse of Table IX's soft-Q ratio)
    sq = katz(pooled(sign, SOFTQ), PAID, pol, PAID)
    o["softq"] = ci(sq, ".2f", ".2f", ".2f") if sign == "Z" else ci(sq, ".2f", ".3f", ".2f")
    o["softq seeds"] = f"{sum(q > p for q, p in zip(per_seed(sign, SOFTQ), pol_seed))} of 3"

    o["pt"] = ci(katz(pol, PAID, pooled(sign, PT), PAID), ".1f", ".1f", ".1f")
    o["pt stored"] = ci(katz(pol, PAID, pooled(sign, PT_STORED), PAID), ".1f", ".1f", ".1f")
    # both sides pay 3,000 calls, so a ratio of counts is the ratio of rates
    o["me5"] = format(pol / pooled(sign, ME5), ".1f")
    o["uniform"] = format(pol / pooled(sign, UNIFORM), ".0f")
    o["depth7"] = format(np.mean(per_seed(sign, DEPTH7)), ".0f")

    # equivalence (EF for the methods that saw the stored forms, FOLD for tuned PT)
    ef = {row: sum(v) for row, v in eq[sign].items()}
    o["ef share"] = format(100 * ef[POLICY] / pol, ".0f")
    k = katz(ef[POLICY], PAID, ef[PT], PAID)
    o["ef pt"] = ci(k, ".1f", ".2f", ".1f") if sign == "Z" else ci(k, ".1f", ".1f", ".1f")
    o["ef pt seeds"] = yes(all(a / b > 2 for a, b in zip(eq[sign][POLICY], eq[sign][PT])))
    o["ef pt stored"] = format(ef[POLICY] / ef[PT_STORED], ".1f")
    o["ef screen"] = format(ef[POLICY] / ef[SCREEN], ".1f")
    _, lo, hi = katz(ef[POLICY], PAID, ef[SOFTQ], PAID)
    o["ef softq"] = yes(lo <= 1 <= hi)

    # the supervised screen
    o["screen"] = ci(katz(pol, PAID, pooled(sign, SCREEN), PAID), ".1f", ".1f", ".1f")
    o["screen seeds"] = yes(all(p / c > 3 for p, c in zip(pol_seed, per_seed(sign, SCREEN))))

    # the hundredfold oversample, seeds 1 and 2: the policy ranked at mu = 100 over the supervised screen at
    # 100 times, both counted as novel codes (Table IX note a), pooled over 2,000 paid calls each
    r100, s100 = per_seed(sign, RANKED_100), per_seed(sign, SCREEN_100)
    o["100x counts"] = f"{', '.join(map(str, r100))} against {', '.join(map(str, s100))}"
    o["100x"] = ci(katz(sum(r100), PAID_100, sum(s100), PAID_100), ".2f", ".2f", ".2f")
    o["100x seeds"] = f"{sum(s > r for r, s in zip(r100, s100))} of {len(r100)}"
    # the second stated exception of the first claim: the screen at 100 times is ahead of the policy at mu = 10 too
    o["100x over policy"] = yes(all(count(sign, SCREEN_100, s) > count(sign, POLICY, s)
                                    for s in arms[sign][SCREEN_100]))

    o["ranker"] = format(pooled(sign, RANKER) / pol, ".3f")

    # neighbourhoods
    o["nbhd pt"] = format(nbhd(sign, POLICY) / nbhd(sign, PT), ".1f")
    o["nbhd screen"] = format(nbhd(sign, POLICY) / nbhd(sign, SCREEN), ".1f")
    # per policy run: the share of its neighbourhoods that also hold a stored certified top-scoring code
    o["nbhd known"] = [100 * len(arms[sign][POLICY][s].classes(PREFIX) & stored_nbhd[sign])
                       / len(arms[sign][POLICY][s].classes(PREFIX)) for s in (0, 1, 2)]

    # processor seconds of the oracle and the exact ladders
    o["per second"] = (f"{1000 * ef[PT] / seconds(sign, PT):.1f} against "
                       f"{1000 * ef[POLICY] / seconds(sign, POLICY):.1f}")
    o["time per call"] = format(seconds(sign, POLICY) / seconds(sign, PT), ".0f")
    o["ladder pt"] = format(ladder(sign, POLICY) / ladder(sign, PT), ".1f")
    o["ladder screen"] = (f"{ladder(sign, POLICY) / pol:.2f} against "
                          f"{ladder(sign, SCREEN) / pooled(sign, SCREEN):.2f}")

known = out["Z"]["nbhd known"] + out["X"]["nbhd known"]

LINES = [
    ("GFlowQEC-offline ahead of every other method in every paired seed, soft Q-learning trained offline and the "
     "supervised screen at 100 times excepted", "ahead"),
    ("Soft Q-learning trained offline over GFlowQEC-offline", "softq"),
    ("Seeds in which soft Q-learning trained offline was ahead", "softq seeds"),
    ("Over tuned PT", "pt"),
    ("Over tuned PT started from the stored forms", "pt stored"),
    ("Over MAP-Elites, 5 bins", "me5"),
    ("Over uniform sampling", "uniform"),
    ("GFlowQEC-online, depth 7, certified top-scoring codes per run", "depth7"),
    ("Per cent of novel top-scoring codes equivalent to no stored code nor its mirror", "ef share"),
    ("Equivalence-free, over tuned PT with equivalent codes and mirrors counted once", "ef pt"),
    ("Equivalence-free, above two in every seed", "ef pt seeds"),
    ("Equivalence-free, over tuned PT started from the stored forms", "ef pt stored"),
    ("Equivalence-free, over the supervised screen", "ef screen"),
    ("Equivalence-free, soft Q-learning trained offline not separated", "ef softq"),
    ("Over the supervised screen", "screen"),
    ("Over the supervised screen, above three in every paired seed", "screen seeds"),
    ("Hundredfold oversample, seeds 1-2: novel top-scoring codes per seed, GFlowQEC-offline ranked at mu = 100 "
     "against the supervised screen at 100 times", "100x counts"),
    ("Hundredfold oversample: GFlowQEC-offline ranked at mu = 100 over the supervised screen at 100 times", "100x"),
    ("Hundredfold oversample: seeds in which the supervised screen at 100 times was ahead", "100x seeds"),
    ("Hundredfold oversample: the supervised screen at 100 times ahead of GFlowQEC-offline in every paired seed",
     "100x over policy"),
    ("Policy ranking the screen's uniform candidates, over its own draws", "ranker"),
    ("Neighbourhoods, over tuned PT", "nbhd pt"),
    ("Neighbourhoods, over the supervised screen", "nbhd screen"),
]
LINES_AFTER = [
    ("Equivalence-free per 1,000 oracle processor seconds, tuned PT against GFlowQEC-offline", "per second"),
    ("Processor time per paid call, GFlowQEC-offline over tuned PT", "time per call"),
    ("Paid calls that ran the exact ladders, GFlowQEC-offline over tuned PT", "ladder pt"),
    ("Exact-ladder runs per novel certified top-scoring code counted by canonical form, "
     "GFlowQEC-offline against the supervised screen", "ladder screen"),
]

print("Sec. 9.1 text: n = 180, first 1,000 paid calls, seeds 0-2")
for label, key in LINES:
    print(f"{label} | {out['Z'][key]} | {out['X'][key]}")
# one range over the six policy runs, both signs
print("Per cent of the policy's neighbourhoods that also hold a stored certified top-scoring code, range over its "
      f"six runs | {min(known):.1f} to {max(known):.1f}")
for label, key in LINES_AFTER:
    print(f"{label} | {out['Z'][key]} | {out['X'][key]}")
