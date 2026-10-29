"""The relational variant of the policy (offline-training appendix): trained, deployed once and not used.

The variant is GFlowQEC-offline with one change: at every position it also reads the relational counts
(defined in the policy appendix) of the entries chosen so far. It was trained on the same stored forms and deployed the same way
(ranked deployment, mu = 10: 10,000 candidates drawn, the 1,000 with the highest policy probability paid).
This readout counts, for the variant and for GFlowQEC-offline, the novel certified top-scoring codes
among the 1,000 paid calls of each seed, at gamma = +1 (sign "Z") and gamma = -1 (sign "X").

The release does not ship the variant's code or checkpoints, only its six deployment cells: per cell the
screen file (every paid code with its bound distances) and the certified file (the exact distances of the
proxy-top rows). Their names keep the prefix "rel_r177_" of the run family that wrote them. Both methods
are counted by the same library path: counts.r177_rows turns a cell into paid rows, counts.LedgerArm
certifies and checks novelty against the stored forms (counts.ledger_pool), as the per-seed n = 180 table does.

Reads only data/n180 through gflowqec.counts; no oracle is called; about 17 s on a laptop. Run from the
repo root:
    python -m readouts.rel180
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
SEEDS = (0, 1, 2)
PREFIX = counts.PREFIX          # 1,000 paid calls per cell
OVER = 10                       # mu: candidates drawn per paid call

# The relational variant's cells (shipped for this readout only) and GFlowQEC-offline's cells
# (the first row of the per-seed n = 180 table); {kind} is "screen" or "certified".
REL_CELL = "rel_r177_{kind}_transformer_w18_b256_x20_rel_e10{sign}_n1000_o10_s{seed}.json"
POLICY_CELL = "r177_{kind}_transformer_w18_b256_x20_e10{sign}_n1000_o10_s{seed}.json"
REL_MODEL = "r177_transformer_w18_b256_x20_rel_s{seed}"     # the checkpoint name stamped in each variant cell


def read_cell(pattern, sign, seed, lam, top, pool_count):
    """One deployment cell as a LedgerArm over its 1,000 paid rows, with its screen header.

    Same path as counts.ledger_arms for the policy cells: the cell is not a stored run (from_pool False),
    so a paid code is novel when none of the 42 stored runs paid its canonical form."""
    screen_file = N180 / pattern.format(kind="screen", sign=sign, seed=seed)
    certified_file = N180 / pattern.format(kind="certified", sign=sign, seed=seed)
    screen = json.load(open(screen_file))
    certified = json.load(open(certified_file))
    assert (screen["sign"], screen["seed"], screen["over"]) == (sign, seed, OVER), screen_file.name
    assert certified["complete"], f"{certified_file.name}: certification not complete"
    head = {k: v for k, v in screen.items() if k != "rows"}
    head["file"] = screen_file.name
    rows = counts.r177_rows(screen, certified["rows"])
    keys = [orbit_key(GROUP_180, r["rep"]) for r in rows]
    arm = counts.LedgerArm(screen_file.name, head, rows, keys, sign, lam, top, pool_count, False)
    # every cell pays exactly 1,000 calls and every proxy-top row was certified
    assert arm.n == PREFIX and arm.uncertified == 0, f"{screen_file.name}: {arm.n} rows, {arm.uncertified} uncertified"
    assert len(set(keys)) == len(keys), f"{screen_file.name}: a repeated code among the paid calls"
    return arm, head


# ------------------------------------------------------------------ the stored pool novelty is read against
runs, _ = counts.load_dumps()
pool_count = counts.ledger_pool(runs)      # asserts the 42 stored runs and their 126,000 paid calls

rel, pol = {}, {}                          # sign -> [novel certified tops per seed]
for sign in ("Z", "X"):
    lam = lam_of(10, sign)
    top = key_top(lam)[0]
    rel[sign], pol[sign] = [], []
    for seed in SEEDS:
        r_arm, r_head = read_cell(REL_CELL, sign, seed, lam, top, pool_count)
        p_arm, _ = read_cell(POLICY_CELL, sign, seed, lam, top, pool_count)
        # the variant's cell must be drawn by its own checkpoint at the full grant of 10,000 candidates
        assert r_head["model"] == REL_MODEL.format(seed=seed), f"{r_arm.label}: drawn by {r_head['model']}"
        assert r_head["sampled"] == OVER * PREFIX, f"{r_arm.label}: {r_head['sampled']} candidates"
        rel[sign].append(r_arm.tops(True))
        pol[sign].append(p_arm.tops(True))
        # the counts behind the number; stored forms paid = paid codes that are not novel
        for name, arm in (("variant", r_arm), ("policy", p_arm)):
            print(f"basis: {sign} s{seed} {name}: paid {arm.n}, proxy-top {int(arm.proxy_top.sum())}, "
                  f"certified tops {arm.tops(False)}, novel {arm.tops(True)}, neighbourhoods {len(arm.classes())}, "
                  f"stored forms paid {int((~arm.novel).sum())}")
        print(f"basis: {sign} s{seed} variant drawn by {r_head['model']}, {r_head['sampled']:,} candidates "
              f"in {r_head['attempts']:,} attempts")

# the pooled ratio with its Katz interval (not printed in the paper; a cross-check)
for sign in ("Z", "X"):
    r, lo, hi = katz(sum(rel[sign]), PREFIX * len(SEEDS), sum(pol[sign]), PREFIX * len(SEEDS))
    below = sum(a < b for a, b in zip(rel[sign], pol[sign]))
    print(f"basis: {sign} pooled variant / policy {r:.3f} [{lo:.3f}, {hi:.3f}], variant below in {below} of 3 seeds")


def seeds_text(x):
    return ", ".join(str(v) for v in x)


print(f"App. B.1: relational variant, ranked deployment at mu = {OVER}, novel certified top-scoring codes "
      f"among {PREFIX:,} paid calls, seeds 0-2")
print("row | gamma = +1 | gamma = -1")
print(f"Relational variant | {seeds_text(rel['Z'])} | {seeds_text(rel['X'])}")
print(f"GFlowQEC-offline | {seeds_text(pol['Z'])} | {seeds_text(pol['X'])}")
print(f"timed: {time.time() - T0:.0f} s")
