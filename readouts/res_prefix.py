"""The per-seed n = 180 table: certified top-scoring codes in the first 1,000 paid calls at n = 180.

Every row is one method at gamma = +1 (sign "Z") and gamma = -1 (sign "X"). The columns are:

    Seeds   how many seeds the row holds: 10 (seeds 0-9) for the four rows repeated at ten seeds
            (GFlowQEC-offline ranked at mu = 10 and 100, the supervised screen at 10 and 100 times),
            2 (seeds 1 and 2) for the earlier screen at 100 times, 3 (seeds 0-2) for every other row;
    Codes   certified top-scoring codes per seed among the method's first 1,000 paid calls;
    Ratio   GFlowQEC-offline's (ranked, mu = 10) pooled count over the row's pooled count, both summed
            over the seeds the two rows share (1,000 paid calls per shared seed), with its 95 % Katz
            interval; the interval is printed only when the ratio is below 10;
    Nbhd    the same ratio on neighbourhoods (distinct gauge classes of the bottleneck half among the
            certified tops), no interval.

So a three-seed row is read against GFlowQEC-offline's seeds 0-2, and a ten-seed row against all ten.
Ratio and Nbhd are blank for the two screens at 100 times, whose oversample the deployment at mu = 10
does not match.

Rows above the rule are methods that saw the stored forms, so a code counts only if no stored run paid its
canonical form (it is novel). Below the rule the methods never saw them and every certified top counts;
GFlowQEC-offline's numerator still counts novel codes only. Note a: the "Policy as a ranker" and
"Ranked, mu = 100" rows are read over the matched supervised screen at the same mu (10 for the ranker,
100 for the ranked deployment), not over GFlowQEC-offline. Note b: the unranked deployment over all its
2,000 paid calls against the earlier, smaller screen's 1,000.

Reads only data/ through gflowqec.counts; no oracle is called; about 38 s on a laptop. Run from the repo
root:
    python -m readouts.res_prefix
"""
import time

from gflowqec import counts
from gflowqec.stats import katz

T0 = time.time()
PREFIX = counts.PREFIX          # 1,000 paid calls per cell
SEEDS = (0, 1, 2)
SEEDS_10 = tuple(range(10))     # the four rows repeated at ten seeds
SEEDS_100 = (1, 2)              # the earlier screen at 100 times was run at seeds 1 and 2 only

# The paper's row label, then the row's name in counts.prefix_arms, in the paper's order.
# Rows above the rule (novel codes only):
ABOVE = [
    ("GFlowQEC-offline, ranked, mu = 10", "GFlowQEC-offline, ranked, mu 10"),
    ("Supervised screen, 10 times", "Supervised screen, 10 times"),
    ("Policy as a ranker", "Policy as a ranker"),
    ("Ranked, mu = 100", "Ranked, mu 100"),
    ("Supervised screen, 100 times", "Supervised screen, 100 times (matched)"),
    ("Earlier, smaller screen, 10 times", "Earlier, smaller screen, 10 times"),
    ("Earlier, smaller screen, 100 times", "Earlier, smaller screen, 100 times"),
    ("Unranked deployment", "Unranked deployment"),
    ("Soft Q-learning, trained offline", "Soft Q-learning, trained offline"),
    ("Tuned PT started from the stored forms", "Tuned PT started from the stored forms"),
]
# Rows below the rule (every certified top counts); their paper label is their name in counts.prefix_arms.
BELOW = [
    "Tuned PT",
    "MAP-Elites, 5 bins",
    "MAP-Elites, 20 bins",
    "PPO, entropy weight 0.1",
    "PPO, entropy weight 0.01",
    "Soft Q-learning, online",
    "Munchausen DQN",
    "GFlowQEC-online, Munchausen DQN as learner",
    "GFlowQEC-online, depth 7",
    "Uniform sampling",
    "BO, one-hot encoding",
]
POLICY = "GFlowQEC-offline, ranked, mu 10"        # the numerator of every other Ratio and Nbhd
RANKER = "Policy as a ranker"
RANKED_100 = "Ranked, mu 100"
MATCHED_100 = "Supervised screen, 100 times (matched)"
EARLIER_100 = "Earlier, smaller screen, 100 times"
HUNDREDFOLD = {RANKED_100, MATCHED_100, EARLIER_100}      # the rows at 100, their basis printed below
# the four rows repeated at ten seeds
TEN_SEEDS = {POLICY, "Supervised screen, 10 times", RANKED_100, MATCHED_100}
# note a: these rows are read over the matched supervised screen at the same mu, both counted as novel codes
OVER_MATCHED = {RANKER: "Supervised screen, 10 times", RANKED_100: MATCHED_100}
# the screens at 100 times print Codes only: the deployment at mu = 10 does not match their oversample
BLANK = {MATCHED_100, EARLIER_100}
# Ranked, mu = 100 at gamma = -1, seed 1: the count after the top-up that re-draws the cell's stored
# forms (its files stf_* are read by hyb180). The table prints the primary count, the cell's own 1,000 paid calls.
TOPUP_X_S1 = 464


def codes(arm, novel):
    """Certified top-scoring codes among the cell's first 1,000 paid calls, novel ones only if `novel`.

    Rows seeded from the stored forms (tuned PT started from them: arm.warm = 8) are paid but never counted.
    Deployment cells and screens keep all their rows in counts.prefix_arms, so the cut to 1,000 is here.
    """
    w = arm.warm
    top = arm.is_top[w:PREFIX]
    if novel:
        top = top & arm.novel[w:PREFIX]
    return int(top.sum())


def hoods(arm):
    """Neighbourhoods among the certified tops of the first 1,000 paid calls (warm rows excluded)."""
    return len(arm.classes(PREFIX))


def end(x):
    """One end of a Katz interval: two decimals, three when two would print 1.00 (the paper's 1.003)."""
    s = f"{x:.2f}"
    return f"{x:.3f}" if s == "1.00" else s


def ratio_cell(a, na, b, nb):
    """Pooled ratio of a out of na paid calls over b out of nb; its Katz interval only when it is below 10."""
    r, lo, hi = katz(a, na, b, nb)
    return f"{r:.2f} [{end(lo)}, {end(hi)}]" if r < 10 else f"{r:.0f}"


def hood_cell(a, b):
    """Pooled neighbourhood ratio a / b, two decimals below 10, whole numbers at or above."""
    x = a / b
    return f"{x:.2f}" if x < 10 else f"{x:.0f}"


def row_seeds(name):
    """The seeds a row of this table holds."""
    if name in TEN_SEEDS:
        return SEEDS_10
    if name == EARLIER_100:
        return SEEDS_100
    return SEEDS


# ------------------------------------------------------------------ read every row at both signs
cells = {}       # sign -> {row name: [LedgerArm per seed, in seed order]}
for sign in ("Z", "X"):
    arms = {name: sorted(arm_list, key=lambda c: c.seed) for name, arm_list in counts.prefix_arms(sign).items()}
    cells[sign] = arms
    # every row this table prints must hold its seeds (row_seeds), with every proxy-top row certified
    for _, name in ABOVE:
        assert tuple(c.seed for c in arms[name]) == row_seeds(name), (sign, name)
        assert all(c.uncertified == 0 for c in arms[name]), f"{sign} {name}: proxy-top rows never certified"
    for name in BELOW:
        assert tuple(c.seed for c in arms[name]) == SEEDS, (sign, name)


def row_line(label, name, novel):
    """One table line: Seeds, then Codes, Ratio, Nbhd at gamma = +1, then the same at gamma = -1."""
    out = [label, str(len(row_seeds(name)))]
    for sign in ("Z", "X"):
        arms = cells[sign]
        per_seed = [codes(c, novel) for c in arms[name]]
        out.append(", ".join(str(x) for x in per_seed))
        if name == POLICY or name in BLANK:
            out += ["", ""]                         # the reference row and the screens at 100 times carry no ratio
            continue
        if name in OVER_MATCHED:
            # note a: this row over the matched screen at the same mu, both counted as novel codes
            num, den, den_novel = name, OVER_MATCHED[name], True
        else:
            # GFlowQEC-offline's novel codes over this row's codes (novel or all, by the row's side of the rule)
            num, den, den_novel = POLICY, name, novel
        # pool both sides over the seeds the two rows share, 1,000 paid calls per seed
        shared = set(row_seeds(num)) & set(row_seeds(den))
        num_arms = [c for c in arms[num] if c.seed in shared]
        den_arms = [c for c in arms[den] if c.seed in shared]
        a = sum(codes(c, True) for c in num_arms)
        b = sum(codes(c, den_novel) for c in den_arms)
        n = PREFIX * len(shared)
        out.append(ratio_cell(a, n, b, n))
        out.append(hood_cell(sum(hoods(c) for c in num_arms), sum(hoods(c) for c in den_arms)))
    return " | ".join(out)


print(f"Table D.1: first {PREFIX:,} paid calls, Seeds = 10 (seeds 0-9), 3 (seeds 0-2) or 2 (seeds 1-2); "
      "each ratio pooled over the seeds its two rows share")
for label, name in ABOVE:
    print(row_line(label, name, novel=True))
print("-- below the rule every certified top-scoring code counts --")
for name in BELOW:
    print(row_line(name, name, novel=False))

# ------------------------------------------------------------------ the hundredfold rows: their basis
# Per cell: novel certified tops, the stored forms among its 1,000 paid calls (paid, never counted) and the
# neighbourhoods.
for label, name in ABOVE:
    if name not in HUNDREDFOLD:
        continue
    for sign in ("Z", "X"):
        arms = cells[sign][name]
        stored = [int((~c.novel[:PREFIX]).sum()) for c in arms]
        print(f"basis: {label} {sign} seeds {[c.seed for c in arms]}: novel tops {[codes(c, True) for c in arms]}, "
              f"stored forms paid {stored}, neighbourhoods {[hoods(c) for c in arms]}")
# Ranked, mu = 100 at gamma = -1, seed 1: the primary count, and the count after the top-up.
arms = cells["X"][RANKED_100]
s1 = [c for c in arms if c.seed == 1][0]
primary = codes(s1, True)
a = sum(codes(c, True) for c in arms)
b = sum(codes(c, True) for c in cells["X"][MATCHED_100])
n = PREFIX * len(arms)
ha = sum(hoods(c) for c in arms)
hb = sum(hoods(c) for c in cells["X"][MATCHED_100])
print(f"basis: Ranked, mu = 100 X seed 1 prints {primary}, the primary count (its {PREFIX:,} paid calls, "
      f"{int((~s1.novel[:PREFIX]).sum())} stored forms paid and never counted); the top-up that re-draws "
      f"them reads {TOPUP_X_S1} (top-up files stf_*, read by hyb180); Ratio {ratio_cell(a, n, b, n)} on {a} / {b}, "
      f"{ratio_cell(a - primary + TOPUP_X_S1, n, b, n)} on {a - primary + TOPUP_X_S1} / {b}; "
      f"Nbhd {ha / hb:.3f} on {ha} / {hb} (primary)")

# ------------------------------------------------------------------ note b
# The unranked deployment over ALL its 2,000 paid calls (novel tops) against the earlier, smaller screen's
# 1,000 (novel tops), pooled over seeds with the Katz interval; and whether it trails in every seed.
UNRANKED = "Unranked deployment"
EARLIER = "Earlier, smaller screen, 10 times"
note = []
per_seed_ratios = []
for sign in ("Z", "X"):
    dep, scr = cells[sign][UNRANKED], cells[sign][EARLIER]
    assert len({c.n for c in dep}) == 1, "the unranked deployment cells differ in length"
    calls = dep[0].n
    ua = [c.tops(True) for c in dep]           # novel tops over all rows
    sb = [c.tops(True) for c in scr]           # the screen pays 1,000 calls, so this is its whole cell
    r, lo, hi = katz(sum(ua), sum(c.n for c in dep), sum(sb), sum(c.n for c in scr))
    note.append(f"{r:.2f} [{lo:.2f}, {hi:.2f}]")
    # per seed: the deployment's rate per paid call over the screen's
    ratios = [(x / d.n) / (y / s.n) for x, y, d, s in zip(ua, sb, dep, scr)]
    per_seed_ratios += ratios
    print(f"basis: note b {sign}: {ua} of {sum(c.n for c in dep):,} against {sb} of {sum(c.n for c in scr):,}; "
          f"ratio {r:.4f} [{lo:.4f}, {hi:.4f}]; per seed {[round(x, 3) for x in ratios]}")
every = "below it in every seed" if all(x < 1 for x in per_seed_ratios) else "NOT below it in every seed"
print(f"note b: over all its {calls:,} paid calls | {note[0]} | {note[1]} | {every}")
print(f"timed: {time.time() - T0:.0f} s")
