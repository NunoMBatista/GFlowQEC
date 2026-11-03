"""The n = 180 results, the Cost paragraph: what the policy's lead costs in processor time.

Two runs are compared, in each run's first 1,000 paid calls, pooled by summing the seeds: GFlowQEC-offline
ranked at mu = 10 (the deployment of the n = 180 summary table) and tuned PT. The ladder-run line pools seeds 0-9;
every processor-second line pools seeds 0-2, because seeds 3-9 ran on different machines. Sign "Z" is
gamma = +1 and sign "X" is gamma = -1.

  - A paid call runs the exact distance ladders only when its bound score reaches the bar (a proxy-top
    call). The first line counts the paid calls that ran them, the policy's over tuned PT's.
  - The ladders' processor seconds are the certification seconds stored per paid call; the bound ladders
    every call runs are the rest of the oracle's seconds. The second line is the ladders' share of the
    oracle's seconds.
  - Seconds per top-scoring code: the ladders' seconds divided by the certified top-scoring codes counted
    as the summary table counts them (novel ones only for the policy, which used the stored evaluations; all of them
    for tuned PT). The last line is the rate of top-scoring codes per ladder second, the policy's over tuned
    PT's.

The same comparison on the equivalence-free counts needs the Tanner-graph certificates of readouts/text180.py,
which prints it on the oracle's seconds.

Reads only data/ (through gflowqec.counts); no oracle call, nothing written; about 50 s on a laptop. Run from
the repo root:
    python -m readouts.cost180
"""

import time

from gflowqec import counts

t0 = time.time()
PREFIX = counts.PREFIX          # 1,000 paid calls per run
SEEDS = (0, 1, 2)               # the processor-second lines (seeds 3-9 ran on different machines)
SEEDS10 = tuple(range(10))      # the ladder-run line: a count of paid calls, comparable across machines
SIGNS = ("Z", "X")              # gamma = +1, gamma = -1
POLICY = "GFlowQEC-offline, ranked, mu 10"
PT = "Tuned PT"

# Acceptance tests: the pooled counts behind the printed numbers (readouts/text180.py checks the same counts).
EXPECT = {
    "Z": {"ladder policy": 1328, "ladder PT": 401, "tops policy": 938, "tops PT": 315,
          "ladder10 policy": 4409, "ladder10 PT": 1247},
    "X": {"ladder policy": 1339, "ladder PT": 365, "tops policy": 926, "tops PT": 285,
          "ladder10 policy": 4534, "ladder10 PT": 1068},
}


def ladder_cost(arm):
    """(paid calls that ran the exact ladders, their certification seconds) in the first 1,000 paid calls."""
    runs, seconds = 0, 0.0
    for r in arm.rows[:PREFIX]:
        if r["cert"] is not None and r["cpu"] > 0:
            runs += 1
            seconds += r["cpu"]
    return runs, seconds


out = {}
for sign in SIGNS:
    arms = counts.prefix_arms(sign)
    got = {}
    for name, row, novel_only in (("policy", POLICY, True), ("PT", PT, False)):
        cells = {a.seed: a for a in arms[row]}
        for s in SEEDS10:
            assert cells[s].warm == 0, f"{row} seed {s} has seeded rows"
        runs, ladder_s, all_s, tops = 0, 0.0, 0.0, 0
        for s in SEEDS:
            arm = cells[s]
            r, sec = ladder_cost(arm)
            runs += r
            ladder_s += sec
            all_s += float(arm.sec[:PREFIX].sum())
            tops += arm.tops(novel_only, PREFIX)
        got[f"ladder {name}"] = runs
        got[f"tops {name}"] = tops
        got[f"ladder s {name}"] = ladder_s
        got[f"all s {name}"] = all_s
        got[f"ladder10 {name}"] = sum(ladder_cost(cells[s])[0] for s in SEEDS10)
    for key, want in EXPECT[sign].items():
        assert got[key] == want, f"{sign} {key}: {got[key]} against {want}"
    out[sign] = got

# per sign: the policy's ladder runs over tuned PT's, the ladders' share of the oracle seconds, the ladder
# seconds per top-scoring code and the policy's rate of tops per ladder second over tuned PT's
o = {}
for sign in SIGNS:
    g = out[sign]
    per_top_policy = g["ladder s policy"] / g["tops policy"]
    per_top_pt = g["ladder s PT"] / g["tops PT"]
    o[sign] = {
        "ladder": format(g["ladder10 policy"] / g["ladder10 PT"], ".1f"),
        "share": (f"{100 * g['ladder s policy'] / g['all s policy']:.0f} and "
                  f"{100 * g['ladder s PT'] / g['all s PT']:.0f}"),
        "per top": f"{per_top_policy:.1f} against {per_top_pt:.1f}",
        "rate": per_top_pt / per_top_policy,
    }

print("Sec. 7 Cost: n = 180, first 1,000 paid calls, pooled over seeds 0-2 (seeds 3-9 ran on different machines), "
      "the ladder-run line over seeds 0-9")
print("Paid calls that ran the exact ladders, GFlowQEC-offline over tuned PT, seeds 0-9 | "
      f"{o['Z']['ladder']} | {o['X']['ladder']}")
print("Not in the paper: per cent of the oracle's processor seconds spent in the exact ladders, "
      "GFlowQEC-offline and tuned PT | "
      f"{o['Z']['share']} | {o['X']['share']}")
print("Not in the paper: exact-ladder processor seconds per certified top-scoring code (novel for GFlowQEC-offline), "
      f"GFlowQEC-offline against tuned PT | {o['Z']['per top']} | {o['X']['per top']}")
print("Not in the paper: top-scoring codes per exact-ladder processor second, GFlowQEC-offline over tuned PT | "
      f"{o['Z']['rate']:.2f} | {o['X']['rate']:.2f}")
within = all(abs(o[sign]["rate"] - 1) <= 0.05 for sign in SIGNS)
print(f"Within five per cent at both biases | {'yes' if within else 'no'}")
for sign in SIGNS:
    g = out[sign]
    print(f"basis: {sign} ladder runs, seeds 0-9, {g['ladder10 policy']:,} and {g['ladder10 PT']:,}; seeds 0-2: "
          f"ladder runs {g['ladder policy']:,} and {g['ladder PT']:,}, ladder seconds "
          f"{g['ladder s policy']:,.0f} and {g['ladder s PT']:,.0f}, oracle seconds {g['all s policy']:,.0f} and "
          f"{g['all s PT']:,.0f}, top-scoring codes {g['tops policy']} and {g['tops PT']} (GFlowQEC-offline and tuned PT)")
print(f"timed: wall {time.time() - t0:.0f} s")
