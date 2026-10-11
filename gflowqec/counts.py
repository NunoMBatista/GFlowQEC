"""How the paper counts its results. Nothing here calls an oracle: every function reads stored runs in data/.

The n = 180 file names (data/n180, and the two reference-sample files of data/keys) keep the prefix of the run
family that wrote them; the prefix is a name only, read by the loaders below:
    r174_   the stored search runs (the 42 certified ones are the stored runs of Sec. 6), the legs of the
            earlier, smaller supervised screen ("leg": "proxy"), and the reference-sample read and resolve;
    r177_   the deployment cells of GFlowQEC-offline and its controls, and the matched supervised screens
            (the dumps also carry "experiment": "r177");
    r200_   later runs compared against the stored ones (tuned PT started from the stored forms, uniform
            sampling at gamma = -1, one-hot BO), never part of the stored pool;
    r209_   later rival runs (MAP-Elites, PPO, online soft Q-learning, GFlowQEC-online with Munchausen DQN as
            its learner).

n = 180 (Sec. 9.1, Tables III and IX, App. D.1)
    A run dump holds one row per paid call: the code (its orbit representative "rep"), the classical
    bound-ladder distances dA, dB with their processor seconds, and, for a row that reached the certifier,
    the exact distances dX, dZ with the certification seconds. Sign "Z" is gamma = +1 and sign "X" is
    gamma = -1; lam = oracle.lam_of(10, sign). A code is top-scoring when oracle.bias_score reaches the
    bar top = oracle.key_top(lam)[0] (10.0 at both signs).

    stored_forms   the 116,972 distinct canonical forms paid by the 42 stored runs (Sec. 6): the training
                   data of GFlowQEC-offline and the reference a novel code is checked against.
    LedgerArm      one cell (method, sign, seed) counted row by row: certified top-scoring codes, which of
                   them are novel, their neighbourhoods (codes.half_class) and the processor seconds.
    ledger_arms    every deployment cell and every stored run of one sign as LedgerArms.
    prefix_arms    every row of Table IX, the search runs cut to their first 1,000 paid calls.

n = 300 (Sec. 9.2, Tables IV and X, Fig. 7)
    A visited file holds 800 visits sampled from one search run, each re-estimated on 4,000 fresh shots
    ("ler_pq"), and the run's number of visits V ("n_local").

    y_star         per seed, the median re-estimated rate over the five reference cells (REFERENCE_CELLS).
    good_count     W, Eq. 20: V times the share of the 800 sampled visits at or below y*.
    d_visit        D_visit, the near-duplicate factor: the share of the run's visits kept by a greedy
                   packing at Hamming radius 2 in call order. W' = W x D_visit.
"""

import itertools
import json
from pathlib import Path

import numpy as np

from gflowqec.codes import GROUP_180, canonicalise, half_class, orbit_key, orbit_rep
from gflowqec.oracle import bias_score, key_top, lam_of
from gflowqec.paths import DATA

N180 = DATA / "n180"
N300 = DATA / "n300"


# =============================================================================================
# n = 180: the stored runs
# =============================================================================================

STORED_FORMS = 116_972      # distinct canonical forms paid by the 42 stored runs (Sec. 6)


def stored_forms():
    """Every distinct canonical form paid by the 42 stored runs, {orbit_key: row}.

    A row holds the canonical tuple ("traj", the training trajectory of GFlowQEC-offline), the bound
    distances dA, dB, the exact distances dX, dZ, whether the form was certified ("cert") and, for a
    certified form, whether both exact ladders closed ("exact"; None when not certified). A form paid
    by several runs keeps its first copy in file-name order, or its first certified copy.
    """
    rows = {}
    # the 42 stored runs are exactly the certified r174 run dumps of data/n180
    for f in sorted(N180.glob("r174_*_cert*.json")):
        d = json.load(open(f))
        for i, rep in enumerate(d["rep"]):
            k = orbit_key(GROUP_180, rep)
            row = {"traj": canonicalise(GROUP_180, rep), "dA": d["dA"][i], "dB": d["dB"][i], "dX": d["dX"][i],
                   "dZ": d["dZ"][i], "cert": d["cert"][i] is not None,
                   "exact": bool(d["exactX"][i] and d["exactZ"][i]) if d["cert"][i] is not None else None}
            if k not in rows or (row["cert"] and not rows[k]["cert"]):
                rows[k] = row
    # novelty and the training set are both defined against this pool, so a changed pool must fail loudly
    assert len(rows) == STORED_FORMS, f"the stored pool drifted: {len(rows)} forms, expected {STORED_FORMS}"
    return rows


def value(r, lam):
    """What the certifying oracle returned for this stored form at lam: exact if certified, else the bound."""
    if r["cert"]:
        return bias_score(r["dX"], r["dZ"], lam)
    return bias_score(r["dA"], r["dB"], lam)


def run_rows(d):
    """The rows of a run dump, one dict per paid call (a dump stores each field as a column)."""
    keys = [k for k in ("dX", "dZ", "exactX", "exactZ", "seconds", "cpu", "dA", "dB", "exactA", "exactB",
                        "secondsBound", "cpuBound", "cert", "certLam") if k in d]
    return [{k: d[k][i] for k in keys} for i in range(len(d["dX"]))]


def rep_rows(d):
    """run_rows() with each row's code ("rep") attached."""
    rows = run_rows(d)
    for r, rep in zip(rows, d["rep"]):
        r["rep"] = rep
    return rows


def kind_of(d):
    """Which oracle paid the dump: "[cert]" the certifying oracle (bound ladders for every code, exact
    distances only for codes whose bounds reach the bar), "[exact]" the exact oracle (exact for every code)."""
    return "[cert]" if d.get("oracle") == "cert" else "[exact]"


def load_dumps():
    """The r174 dumps of data/n180, each with its file name: (search runs, earlier-screen legs).

    The search runs include the 42 stored runs. The earlier-screen legs ("leg": "proxy") hold the codes
    the earlier, smaller supervised screen picked and paid."""
    runs, proxies = [], []
    for f in sorted(N180.glob("r174_*.json")):
        d = json.load(open(f))
        d["file"] = f.name
        if d.get("leg") is None and not d.get("complete"):
            continue                                   # a run that never finished its budget is not a cell
        (proxies if d.get("leg") == "proxy" else runs).append(d)
    return runs, proxies


def load_r200():
    """The later complete n = 180 runs (tuned PT started from the stored forms, uniform sampling at gamma = -1,
    one-hot BO). Their r200 prefix keeps them out of the stored runs they are compared against."""
    out = []
    for f in sorted(N180.glob("r200_*.json")):
        d = json.load(open(f))
        d["file"] = f.name
        if d.get("complete"):
            out.append(d)
    return out


def ladder_of(d):
    """Which classical bound ladder a dump's screening seconds were paid on: "full" (every rung) or "parity"
    (odd rungs skipped). Seconds paid on different ladders are never compared. A dump written before the
    field existed is on the full ladder, except the deployment cells (experiment "r177"), which skip odd rungs."""
    return d.get("ladder") or ("parity" if d.get("experiment") == "r177" else "full")


_KEYS = {}


def dump_keys(d, rows):
    """Orbit keys of a dump's rows, memoised by file name (the pool and the cells share rows)."""
    if d["file"] not in _KEYS:
        _KEYS[d["file"]] = [orbit_key(GROUP_180, r["rep"]) for r in rows]
    return _KEYS[d["file"]]


def r177_rows(screen, certified):
    """The rows of a deployment cell in the run-dump row schema.

    `screen` holds every paid code with its bound distances; `certified` holds the certification of the
    proxy-top rows, keyed by row index. The row's rep is the orbit representative of the sampled tuple,
    the key every other dump stores, so neighbourhoods are folded on the same partition for every method."""
    out = []
    for i, r in enumerate(screen["rows"]):
        c = certified.get(str(i))
        out.append({"dA": r["dA"], "dB": r["dB"], "exactA": r["exactA"], "exactB": r["exactB"],
                    "cpuBound": r["cpuBound"], "rep": list(orbit_rep(GROUP_180, r["exps"])), "cert": (i if c else None),
                    "dX": c["dX"] if c else None, "dZ": c["dZ"] if c else None,
                    "exactX": c["exactX"] if c else None, "exactZ": c["exactZ"] if c else None,
                    "cpu": c["cpu"] if c else 0.0})
    return out


class LedgerArm:
    """One cell (method, sign, seed) counted row by row.

    A row is proxy-top when its bound score reaches the bar or one of its bound ladders is censored (a
    censored bound is no bound); only proxy-top rows are certified. A certified proxy-top row whose exact
    score reaches the bar is a certified top-scoring code (is_top). A row is novel when no stored run paid
    its canonical form; a cell that is itself one of the 42 stored runs (from_pool) is read against the
    other 41. Seconds of a row: its bound-ladder seconds, plus its certification seconds if proxy-top.
    A proxy-top row that was never certified is counted in `uncertified`; the cell's certified counts
    are then incomplete and a caller must not use them.
    """

    def __init__(self, label, dump, rows, keys, sign, lam, top, pool_count, from_pool):
        self.label, self.sign, self.seed = label, sign, dump.get("seed")
        self.ladder, self.source = ladder_of(dump), dump.get("file", "?")
        self.n = len(rows)
        self.rows = rows                             # kept for the codes of counted tops and their ladder runs
        self.warm = int(dump.get("warm_paid") or 0)  # PT started from the stored forms: seeded rows, paid, never counted
        self.scr = np.zeros(self.n)                  # screening seconds (the bound ladders), every row
        self.cert_s = np.zeros(self.n)               # certification seconds, proxy-top rows only
        self.proxy_top = np.zeros(self.n, bool)
        self.is_top = np.zeros(self.n, bool)
        self.novel = np.zeros(self.n, bool)
        self.cls = [None] * self.n                   # neighbourhood (gauge class of the bottleneck half), proxy-top rows only
        self.uncertified = 0
        for i, r in enumerate(rows):
            self.scr[i] = r["cpuBound"]
            self.proxy_top[i] = bias_score(r["dA"], r["dB"], lam) >= top or not (r["exactA"] and r["exactB"])
            # how many stored runs paid this form; a stored run does not count itself
            held = pool_count.get(keys[i], 0)
            self.novel[i] = (held - 1 if from_pool else held) <= 0
            if not self.proxy_top[i]:
                continue
            self.cls[i] = half_class(r["rep"], sign)
            if r["cert"] is None:
                self.uncertified += 1
                continue
            self.cert_s[i] = r["cpu"]
            self.is_top[i] = bias_score(r["dX"], r["dZ"], lam) >= top
        self.sec = self.scr + self.cert_s            # all-in seconds, per row

    def tops(self, novel_only=True, upto=None):
        """Certified top-scoring codes among the first `upto` rows, novel ones only by default.
        Warm rows are NOT excluded here; use is_top[warm:] for a cell started from the stored forms."""
        m = self.is_top[:upto] & self.novel[:upto] if novel_only else self.is_top[:upto]
        return int(m.sum())

    def classes(self, upto=None):
        """The neighbourhoods among the certified tops of the first `upto` rows, warm rows excluded."""
        return {c for i, (c, t) in enumerate(zip(self.cls[:upto], self.is_top[:upto])) if t and i >= self.warm}


def ledger_pool(runs):
    """How many of the 42 stored runs paid each canonical form, {orbit_key: number of runs}.

    The stored runs are the complete certified r174 run dumps; the later r200 runs are never part of the
    pool they are compared against."""
    count, orbits, dumps = {}, 0, 0
    for d in runs:
        if kind_of(d) != "[cert]" or d["file"].startswith("r200_"):
            continue
        rows = rep_rows(d)
        keys = dump_keys(d, rows)
        orbits += len(rows)
        dumps += 1
        for k in set(keys):
            count[k] = count.get(k, 0) + 1
    # every novelty flag is defined against this pool (42 runs, 126,000 paid calls), so it must not move
    assert dumps == 42 and orbits == 126_000, f"the stored pool drifted: {dumps} dumps, {orbits:,} paid rows"
    return count


# The deployment cells of GFlowQEC-offline and its controls (data/n180/r177_screen_<cell>_s<seed>.json),
# with their seeds, in file-name order.
POLICY_CELLS = [
    ("mscreen_T1tfr_st252224_e10{sign}_n1000_u100", (1, 2)),              # matched supervised screen, 100 times
    ("mscreen_T1tfr_st252224_e10{sign}_n1000_u10", (0, 1, 2)),            # matched supervised screen, 10 times
    ("transformer_w18_b256_x20_e10{sign}_n1000_o100", (1, 2)),            # GFlowQEC-offline ranked, mu = 100
    ("transformer_w18_b256_x20_e10{sign}_n1000_o10", (0, 1, 2)),          # GFlowQEC-offline ranked, mu = 10
    ("transformer_w18_b256_x20_e10{sign}_n1000_u10", (0, 1, 2)),          # the policy as a ranker of uniform draws
    ("transformer_w18_b256_x20_e10{sign}_n2000", (0, 1, 2)),              # unranked deployment, 2,000 paid calls
    ("transformer_w18_b256_x20_softq_e10{sign}_n1000_o10", (0, 1, 2)),    # soft Q-learning trained offline, mu = 10
]
# The two hundredfold cells hold seeds 1 and 2 only, as Table IX prints them (Table IX rows "Ranked, mu = 100" and
# "Supervised screen, 100 times").


def ledger_arms(runs, proxies, sign, lam, top, pool_count):
    """Every n = 180 cell at this sign, {label: [one LedgerArm per seed]}, each holding all its rows.

    From the run dumps: "PT from scratch t<range>" and "PT continued t<range> (orbits 1,000-3,000)", the same
    with "POOL-WARM PT" for PT started from the stored forms, "uniform + screen", "lsgfn" (GFlowQEC-online),
    "lsgfn-rel" and "mdqn". From the earlier-screen legs: "supervised proxy-SCREEN (relational, full pool)"
    and its ", over 100)" twin. From POLICY_CELLS: "POLICY r177_screen_<cell>".
    """
    arms = {}

    def add(label, dump, rows, keys, from_pool):
        arms.setdefault(label, []).append(LedgerArm(label, dump, rows, keys, sign, lam, top, pool_count, from_pool))

    rnd = {}                                           # seed -> (dump, rows, keys) of the uniform cell
    for d in runs:
        if kind_of(d) != "[cert]":
            continue
        rows = rep_rows(d)
        keys = dump_keys(d, rows)
        if d["arm"] == "random":                       # uniform draws do not depend on the sign
            # ONE cell per seed: a uniform run re-certified at the other sign holds the same codes, so the
            # dump certified at THIS sign is preferred (its certification seconds are the ones charged)
            arms.setdefault("uniform + screen", [])    # keeps the arm's place in the label order
            if d["seed"] not in rnd or d["sign"] == sign:
                rnd[d["seed"]] = (d, rows, keys)
        elif d["eta"] != 10.0 or d["sign"] != sign:
            continue
        elif d["arm"] == "pt":
            warm = "POOL-WARM PT" if "_Wpool" in d["file"] else "PT"
            lad = f"t{d['tmin']}-{d['tmax']}"
            add(f"{warm} from scratch {lad}", d, rows, keys, True)
            if len(rows) > 1000:          # a 1,000-call run has no continuation
                add(f"{warm} continued {lad} (orbits 1,000-3,000)", d, rows[1000:], keys[1000:], True)
        elif d["arm"] in ("lsgfn", "mdqn"):
            add(d["arm"] + ("-rel" if d.get("relational") else ""), d, rows, keys, True)
    for d, rows, keys in rnd.values():
        add("uniform + screen", d, rows, keys, True)
    grants = {}                                   # screen label -> its oversampling factors (must be one)
    for d in proxies:
        if d["eta"] != 10.0 or d["sign"] != sign or kind_of(d) != "[cert]":
            continue
        head = {**d, "file": d["file"] + ":screen"}
        rel = "relational" if d.get("relational") else "bag"
        # the screen trained on all stored forms is its own arm
        src = ", full pool" if d.get("train_source") == "pool" else ""
        # a different oversampling factor (candidates drawn per paid call) is a different arm
        over = int(d.get("over", 10))
        grant = f", over {over}" if over != 10 else ""
        label = f"supervised proxy-SCREEN ({rel}{src}{grant})"
        grants.setdefault(label, set()).add(over)
        add(label, head, d["screen"], dump_keys(head, d["screen"]), False)
    for label, g in grants.items():
        assert len(g) == 1, f"arm {label!r} mixes oversampling factors {sorted(g)}"
    for cell, seeds in POLICY_CELLS:
        for seed in seeds:
            f = N180 / f"r177_screen_{cell.format(sign=sign)}_s{seed}.json"
            s = json.load(open(f))
            certified = json.load(open(N180 / f.name.replace("r177_screen", "r177_certified")))["rows"]
            head = {**{k: v for k, v in s.items() if k != "rows"}, "file": f.name}
            rows = r177_rows(s, certified)
            add("POLICY " + f.name[:-len(f"_s{s['seed']}.json")], head, rows, dump_keys(head, rows), False)
    return arms


# =============================================================================================
# n = 180: the rows of Table IX, on their first 1,000 paid calls
# =============================================================================================

PREFIX = 1000

# Rows that are deployment cells or screens: (paper row, ledger_arms label)
PREFIX_LEDGER = [
    ("GFlowQEC-offline, ranked, mu 10", "POLICY r177_screen_transformer_w18_b256_x20_e10{sign}_n1000_o10"),
    ("Supervised screen, 10 times", "POLICY r177_screen_mscreen_T1tfr_st252224_e10{sign}_n1000_u10"),
    ("Policy as a ranker", "POLICY r177_screen_transformer_w18_b256_x20_e10{sign}_n1000_u10"),
    ("Ranked, mu 100", "POLICY r177_screen_transformer_w18_b256_x20_e10{sign}_n1000_o100"),
    ("Supervised screen, 100 times (matched)", "POLICY r177_screen_mscreen_T1tfr_st252224_e10{sign}_n1000_u100"),
    ("Earlier, smaller screen, 100 times", "supervised proxy-SCREEN (relational, full pool, over 100)"),
    ("Earlier, smaller screen, 10 times", "supervised proxy-SCREEN (relational, full pool)"),
    ("Unranked deployment", "POLICY r177_screen_transformer_w18_b256_x20_e10{sign}_n2000"),
    ("Soft Q-learning, trained offline", "POLICY r177_screen_transformer_w18_b256_x20_softq_e10{sign}_n1000_o10"),
]

# Rows counted "above the rule": methods that saw the stored forms, so only their novel codes count.
# Every other row counts all its certified top-scoring codes.
PREFIX_ABOVE = {row for row, _ in PREFIX_LEDGER} | {"Tuned PT started from the stored forms"}


def prefix_rivals(sign):
    """Rows that are search runs, read file by file: (paper row, file name with {seed}, one of the 42 stored runs?)."""
    # uniform sampling was run once per sign: at gamma = +1 as a stored run, at gamma = -1 later
    uniform = (("r174_random_C6xS3_e10Z_b3000_cert_s{seed}.json", True) if sign == "Z" else
               ("r200_random_C6xS3_e10X_b3000_cert_s{seed}.json", False))
    rows = [
        ("Tuned PT started from the stored forms", "r200_pt_C6xS3_e10{sign}_b1000_t0.25-2.0_Wpool_cert_s{seed}.json", False),
        ("Tuned PT", "r174_pt_C6xS3_e10{sign}_b3000_t0.25-2.0_cert_s{seed}.json", True),
        ("MAP-Elites, 5 bins", "r209_me_C6xS3_e10{sign}_b3000_g5_cert_s{seed}.json", False),
        ("MAP-Elites, 20 bins", "r209_me_C6xS3_e10{sign}_b3000_g20_cert_s{seed}.json", False),
        ("PPO, entropy weight 0.1", "r209_ppo_C6xS3_e10{sign}_b3000_ent0.1_cert_s{seed}.json", False),
        ("PPO, entropy weight 0.01", "r209_ppo_C6xS3_e10{sign}_b3000_ent0.01_cert_s{seed}.json", False),
        ("Soft Q-learning, online", "r209_sdqn_C6xS3_e10{sign}_b3000_cert_s{seed}.json", False),
        ("Munchausen DQN", "r174_mdqn_C6xS3_e10{sign}_b3000_cert_s{seed}.json", True),
        ("GFlowQEC-online, Munchausen DQN as learner", "r209_mdqnls_C6xS3_e10{sign}_b3000_k1i7det_cert_s{seed}.json", False),
        ("Uniform sampling", *uniform),
        ("BO, one-hot encoding", "r200_bo_C6xS3_e10{sign}_b1000_onehot_cert_s{seed}.json", False),
        ("GFlowQEC-online, depth 7", "r174_lsgfn_C6xS3_e10{sign}_b3000_k1i7det_cert_s{seed}.json", True),
    ]
    return [(row, name.replace("{sign}", sign), member) for row, name, member in rows]


def prefix_arm(name, sign, seed, lam, top, pool_count, member):
    """One search run of data/n180 as a LedgerArm on its first 1,000 paid calls.

    member: the run is one of the 42 stored runs, so its novelty is read against the other 41."""
    d = json.load(open(N180 / name))
    d["file"] = name
    assert d["sign"] == sign and d["eta"] == 10.0 and d["group"] == GROUP_180 and d["seed"] == seed, name
    assert d["calls"] == len(d["rep"]), f"{name}: a row is not one paid call"
    rows = rep_rows(d)[:PREFIX]
    keys = [orbit_key(GROUP_180, r["rep"]) for r in rows]
    assert len(set(keys)) == len(keys), f"{name}: a repeated code inside the first {PREFIX} paid calls"
    arm = LedgerArm(name, d, rows, keys, sign, lam, top, pool_count, member)
    assert arm.uncertified == 0, f"{name}: {arm.uncertified} proxy-top rows never certified"
    return arm


def prefix_arms(sign):
    """Every row of Table IX at this sign, {paper row: [LedgerArm per seed]}.

    Search runs are cut to their first 1,000 paid calls. Deployment cells and screens keep all their rows
    (the unranked deployment has 2,000), so read them with upto=PREFIX.
    """
    lam = lam_of(10, sign)
    top = key_top(lam)[0]
    runs, proxies = load_dumps()
    runs += load_r200()
    pool_count = ledger_pool(runs)
    arms = ledger_arms(runs, proxies, sign, lam, top, pool_count)
    out = {row: arms[label.format(sign=sign)] for row, label in PREFIX_LEDGER}
    for row, name, member in prefix_rivals(sign):
        out[row] = [prefix_arm(name.format(seed=s), sign, s, lam, top, pool_count, member) for s in (0, 1, 2)]
    return out


# =============================================================================================
# n = 300: good proposals
# =============================================================================================

TAU1 = 0.003567     # tau_1%: 1st percentile of the re-estimated rate over the uniform draws that generate the group
CALIB = 200         # the learners' calibration draws and MAP-Elites' initial draws: uniform, paid first, never visits


def visit_indices(d):
    """The paid visits of one n = 300 search run, in call order: PT's rows after its chain starts,
    GFlowQEC-online's local-search rows (src 1), every row of uniform sampling, and every row after
    the first 200 uniform draws for the methods without a local stream."""
    n = len(d["rep"])
    if d["arm"] == "pt":
        return list(range(d.get("pt_chains", 8), n))
    if d["arm"] == "lsgfn":
        return [i for i, src in enumerate(d["src"]) if src == 1]
    if d["arm"] == "random":
        return list(range(n))
    assert d["arm"] in ("me", "sdqn", "mdqn", "ppo", "gfn", "bo", "selfscreen"), f"no visit rule for arm {d['arm']}"
    return list(range(CALIB, n))


REFERENCE_CELLS = ("d31", "f31", "g31", "u31", "pt")   # the five reference cells whose pooled median fixes y*
# PPO at entropy weight 0.01 stopped early in every seed and is read as a disclosed partial cell.
PARTIAL_CELLS = ("ppo",)
SEEDS = range(6)

# cell -> the infix of its visited files, data/n300/g1_repay_visited_i4000_<infix>_s<seed>.json
CELLS = {"d31": "det_d31",                      # GFlowQEC-online, depth 31 (reference)
         "f31": "detP_f31",                     # depth 31, accepted child re-scored on its own stream (reference)
         "g31": "lsgfn_detg_g31",               # depth 31, local moves re-emitted uniformly (reference)
         "u31": "lsgfn_detu_u31",               # local search alone, depth 31 (reference)
         "pt": "pt_b0.35_t0.01-0.15_w",         # tuned PT (reference)
         "d127": "lsgfn_k1i127det_d127",        # GFlowQEC-online, depth 127
         "c127": "lsgfn_k1i127detC_c127",       # GFlowQEC-online, depth 127, common random numbers
         "ptc": "pt_b0.35_t0.01-0.15_wC_crn",   # tuned PT, common random numbers
         "u127": "lsgfn_k1i127detu_u127",       # local search alone, depth 127
         "rnd": "random_w",                     # uniform sampling
         "me": "me_w",                          # MAP-Elites
         "sdqn": "sdqn_w",                      # soft DQN
         "mdqn": "mdqn_w",                      # Munchausen DQN
         "ppo": "ppo_w",                        # PPO, entropy weight 0.01 (partial)
         "ppo1": "ppo_w_ent01",                 # PPO, entropy weight 0.1
         "gfn": "gfn_b0.8_t0.05-0.67_w",        # the plain sampler, no local search
         "p31": "lsgfn_k1i31detS_p31",          # depth 31 with the supervised loss in place of the GFlowNet loss
         "p127": "lsgfn_k1i127detSC_p127",      # depth 127, common random numbers, supervised loss
         "bo40k": "bo_wC_bo40k-onehot",         # one-hot BO, common random numbers
         "ss300": "selfscreen_w_ss300"}         # supervised screen trained on the run's own paid calls
SUBS = list(itertools.combinations(range(10), 8))       # the 45 eight-position sub-keys of a ten-element tuple


def source_path(visited):
    """The search run a visited file re-estimated (the files store it as results/synthesis/<name>)."""
    return N300 / Path(visited["source"]).name


def source_stamp(visited):
    """(paid calls, budget, common random numbers?) of the visited file's search run. A run checkpoints
    with its full schema, so calls against budget is the only completeness test there is."""
    d = json.load(open(source_path(visited)))
    return d["calls"], d["budget"], bool(d.get("crn", False))


def load_visited(cells):
    """{cell: {seed: visited file}} for seeds 0-5. A short visit leg or a search run short of its budget
    is a hard stop, except the PPO cell of PARTIAL_CELLS. Each file gets "crn" from its search run."""
    runs = {}
    for cell in cells:
        for seed in SEEDS:
            f = N300 / f"g1_repay_visited_i4000_{CELLS[cell]}_s{seed}.json"
            d = json.load(open(f))
            assert d["n_done"] == d["n_sub"], f"{f.name}: leg {d['n_done']}/{d['n_sub']}"
            calls, budget, crn = source_stamp(d)
            assert calls == budget or cell in PARTIAL_CELLS, f"{f.name}: source has {calls:,} of {budget:,} calls"
            d["crn"] = crn
            runs.setdefault(cell, {})[d["seed"]] = d
    return runs


def y_star(runs, seed):
    """y* at this seed: the exact median of the re-estimated rates of the five reference cells pooled.
    Never rounded: the rates are discrete, so a rounded y* would move the counts."""
    assert all(c in runs for c in REFERENCE_CELLS), "the five reference cells must be loaded to fix y*"
    vals = [np.array([o["ler_pq"] for o in runs[c][seed]["orbits"]]) for c in REFERENCE_CELLS]
    return np.median(np.concatenate(vals))


def good_count(visited, q):
    """W of one run (Eq. 20): its visits V times the share of its sampled visits at or below q = y*."""
    frac = (np.array([o["ler_pq"] for o in visited["orbits"]]) <= q).mean()
    return visited["n_local"] * frac


def d_visit(source):
    """D_visit of one search run: greedy packing of its visits in call order, no values used. Two
    ten-element tuples within Hamming distance 2 agree on at least 8 positions, i.e. share one of the
    45 eight-position sub-keys, so the packing below is exact. Returns kept / visits."""
    d = json.load(open(source))
    local = visit_indices(d)
    seen, kept = set(), 0
    for i in local:
        rep = tuple(d["rep"][i])
        keys = [(sub, tuple(rep[p] for p in sub)) for sub in SUBS]
        if any(k in seen for k in keys):
            continue
        seen.update(keys)
        kept += 1
    return kept / len(local)
