"""Sec. 9.1, the paragraph on the supervised screen: the screen ranking the policy's own hundredfold draws.

Three arms at a hundredfold grant (100 candidates per paid call, 1,000 paid calls per cell), seeds 1 and 2, at
gamma = +1 (sign "Z") and gamma = -1 (sign "X"):

    H   the matched supervised screen ranking GFlowQEC-offline's own hundredfold draws, the stored forms skipped
        before ranking, its top 1,000 paid;
    U   the same screen ranking uniform draws at the same grant (Table IX, "Supervised screen, 100 times");
    P   GFlowQEC-offline ranked by its own log-probability at mu = 100 (Table IX, "Ranked, mu = 100").

The unit is novel certified top-scoring codes: certified tops whose canonical form no stored run paid, the
library's novelty rule (gflowqec.counts.LedgerArm). The paper prints H / U pooled over the two seeds (2,000 paid
calls per arm) with its 95 % Katz interval (Eq. C1), and that H is ahead of U in each seed at each bias. H / P is
the count behind "the policy's draws and the screen's ranking combine": H beats both the screen alone (U) and the
policy alone (P).

H and U never pay a stored form, so every paid row counts. P paid a few stored forms; as registered for this
comparison, P is read with those rows replaced by its top-up, the next-ranked novel draws (files stf_r177_*).
Table IX reads P without the top-up, so P's count can differ from the table by the replaced rows.

Reads only data/n180 and the release library; no oracle is called. Run from the repo root:
    python -m readouts.hyb180
"""
import json
import time

from gflowqec import counts
from gflowqec.codes import GROUP_180, orbit_key
from gflowqec.oracle import key_top, lam_of
from gflowqec.paths import DATA
from gflowqec.stats import katz

T0 = time.time()
N180 = DATA / "n180"
PREFIX = counts.PREFIX          # 1,000 paid calls per cell
SEEDS = (1, 2)                  # the hundredfold cells ran at seeds 1 and 2 only
POOLED = PREFIX * len(SEEDS)    # 2,000 paid calls per arm and sign
GAMMA = {"Z": "+1", "X": "-1"}

# The cells of each arm; {leg} is "screen" (every paid call with its bound distances) or "certified" (the exact
# distances of the proxy-top calls). P's top-up has the same name with the prefix stf_.
H_CELL = "hyb_r177_{leg}_mscreen_T1tfr_st252224_e10{sign}_n1000_h100_s{seed}.json"
U_CELL = "r177_{leg}_mscreen_T1tfr_st252224_e10{sign}_n1000_u100_s{seed}.json"
P_CELL = "r177_{leg}_transformer_w18_b256_x20_e10{sign}_n1000_o100_s{seed}.json"


def load_cell(name, sign, seed):
    """(screen header, rows in the run-dump schema, orbit keys) of one paid cell."""
    screen = json.load(open(N180 / name.format(leg="screen", sign=sign, seed=seed)))
    cert = json.load(open(N180 / name.format(leg="certified", sign=sign, seed=seed)))
    assert screen["sign"] == sign and screen["seed"] == seed and screen["over"] == 100, name
    rows = counts.r177_rows(screen, cert["rows"])
    keys = [orbit_key(GROUP_180, r["rep"]) for r in rows]
    return screen, rows, keys


def count_tops(label, screen, rows, keys, sign, pool_count):
    """Novel certified top-scoring codes of 1,000 paid rows (all must be novel and fully certified)."""
    lam = lam_of(10, sign)
    top = key_top(lam)[0]
    arm = counts.LedgerArm(label, screen, rows, keys, sign, lam, top, pool_count, False)
    assert arm.n == PREFIX, f"{label}: {arm.n} paid rows"
    assert arm.novel.all(), f"{label}: a paid row is a stored form"
    assert arm.uncertified == 0, f"{label}: {arm.uncertified} proxy-top rows never certified"
    return arm.tops(True)


def novel_of(keys, pool_count):
    """Per row: True when no stored run paid its canonical form."""
    return [pool_count.get(k, 0) == 0 for k in keys]


# ------------------------------------------------------------------ the stored pool (novelty reference)
runs, _ = counts.load_dumps()
runs += counts.load_r200()
pool_count = counts.ledger_pool(runs)     # orbit_key -> number of the 42 stored runs that paid it

# ------------------------------------------------------------------ count every cell
print(f"Sec. 9.1: novel certified top-scoring codes in the first {PREFIX:,} paid calls, hundredfold grant, "
      f"seeds 1-2")
tops = {}        # (arm, sign) -> [count at seed 1, count at seed 2]
p_table = {}     # sign -> P's counts without the top-up (Table IX's reading)
for sign in ("Z", "X"):
    for arm in ("H", "U", "P"):
        tops[arm, sign] = []
    p_table[sign] = []
    for seed in SEEDS:
        # H: the screen's picks among the policy's draws; the stored forms were skipped before ranking
        screen, rows, keys = load_cell(H_CELL, sign, seed)
        assert screen["source"] == "hybrid", "not an H cell"
        # the library's novelty rule must agree with the flag the sampler stored on each paid row
        assert novel_of(keys, pool_count) == [r["novel"] for r in screen["rows"]], "H: novelty flags disagree"
        tops["H", sign].append(count_tops(f"H {sign} s{seed}", screen, rows, keys, sign, pool_count))
        print(f"basis: H {sign} s{seed}: candidates {screen['cand_source']} (md5 {screen['cand_md5']}), "
              f"stored forms skipped {screen['stored_skipped']}, paid rows also paid by P {screen['overlap_policy_paid']}")

        # U: the screen's picks among uniform draws; every paid row is novel
        screen, rows, keys = load_cell(U_CELL, sign, seed)
        assert novel_of(keys, pool_count) == [r["novel"] for r in screen["rows"]], "U: novelty flags disagree"
        tops["U", sign].append(count_tops(f"U {sign} s{seed}", screen, rows, keys, sign, pool_count))

        # P: the policy's own ranking; its paid stored forms are replaced by the top-up rows
        screen, rows, keys = load_cell(P_CELL, sign, seed)
        novel = novel_of(keys, pool_count)
        assert novel == [r["novel"] for r in screen["rows"]], "P: novelty flags disagree"
        lam = lam_of(10, sign)
        table = counts.LedgerArm("P", screen, rows, keys, sign, lam, key_top(lam)[0], pool_count, False)
        p_table[sign].append(table.tops(True))                        # Table IX's reading, no top-up
        dropped = [i for i, v in enumerate(novel) if not v]
        keep = [i for i, v in enumerate(novel) if v]
        rows2 = [rows[i] for i in keep]
        keys2 = [keys[i] for i in keep]
        if dropped:
            topup, trows, tkeys = load_cell("stf_" + P_CELL, sign, seed)
            assert topup["dropped_rows"] == dropped, "P: the top-up replaces other rows"
            rows2 += trows
            keys2 += tkeys
        tops["P", sign].append(count_tops(f"P {sign} s{seed}", screen, rows2, keys2, sign, pool_count))
        print(f"basis: P {sign} s{seed}: {len(dropped)} stored form(s) replaced by the top-up; novel tops "
              f"{p_table[sign][-1]} without it (Table IX), {tops['P', sign][-1]} with it")


# ------------------------------------------------------------------ the comparisons
def ratio(num, den, sign):
    """Pooled num / den over the two seeds with the Katz interval, and the seeds where num is ahead."""
    a, b = tops[num, sign], tops[den, sign]
    r, lo, hi = katz(sum(a), POOLED, sum(b), POOLED)
    ahead = sum(x > y for x, y in zip(a, b))
    per = ", ".join(f"s{s} {x}/{y}" for s, x, y in zip(SEEDS, a, b))
    # the read of record's format, three decimals
    print(f"basis: {num} / {den} at {sign}: novel certified tops [{per}], pooled per 1,000 "
          f"{1000 * sum(a) / POOLED:.1f} / {1000 * sum(b) / POOLED:.1f}, ratio {r:.3f} Katz [{lo:.3f}, {hi:.3f}]")
    return f"{r:.2f} [{lo:.2f}, {hi:.2f}]", ahead


for arm, label in (("H", "screen ranking the policy's draws"), ("U", "screen ranking uniform draws"),
                   ("P", "policy ranked by its log-probability, with its top-up")):
    z = ", ".join(str(x) for x in tops[arm, "Z"])
    x = ", ".join(str(x) for x in tops[arm, "X"])
    print(f"{arm} ({label}) | gamma = +1: {z} | gamma = -1: {x}")
for den in ("U", "P"):
    out = []
    for sign in ("Z", "X"):
        text, ahead = ratio("H", den, sign)
        out.append(f"gamma = {GAMMA[sign]}: {text}, H ahead in {ahead} of {len(SEEDS)} seeds")
    print(f"H / {den} | " + " | ".join(out))
print(f"timed: {time.time() - T0:.0f} s")
