"""The paid calls: every number a search arm is charged for comes out of one of these oracles.

An oracle wraps a scoring rule with a hard budget and a cache keyed by the code's canonical name,
so a revisit or a gauge copy of an already-paid code is free and a run's cost is its number of
distinct paid codes. Three scoring rules, one per scale of the paper:

  n = 36   `Oracle`: exact n, k, d and merit kd^2/n of a two-block code (Sec. 9.3).
  n = 180  `BiasOracle` / `CertifiedOracle`: the leading-order failure exponent of a CSS code under
           biased noise (Sec. 4.3),
               score(lam) = min(c_X d_X, c_Z d_Z),   c = log10(1/p) / 2,
           with p_quiet = 1e-3 on the quiet sector and p_noisy = 1e-3 * 10^|lam| on the noisy one;
           lam = log10(p_Z / p_X) is the one condition scalar (lam > 0 Z-biased, lam < 0 X-biased,
           lam = 0 unbiased, where the score is 1.5 d). d_X and d_Z are exact per-sector SAT
           distances, so the score is deterministic and decoder-free, and one paid call gives the
           score at EVERY condition.
  n = 300  `LEROracle`: -log of the per-logical-qubit logical error rate under depolarizing
           code-capacity noise, estimated with BP-OSD at 1,000 shots.

The certified n = 180 oracle. The classical distances d_A, d_B of the lifted base matrices are exact
upper bounds on the sector distances (d_X <= d_A, d_Z <= d_B), and cost about a second. A paid call
runs those two bound ladders; the value at lam is the proxy min(c_X d_A, c_Z d_B) unless it reaches
the reference sample's top score at lam, in which case the exact sector ladders run too and the value
is exact. An exact-top code has proxy >= exact = top, so it is always certified: the top set is exact
and the proxy is only a filter.

The sector-swapping symmetry. M(g, h) = (h^-1, g^-1) (`codes.mirror`) maps a generating tuple to a
generating tuple of a different orbit with d_X and d_Z swapped. The oracle never caches across it,
so every arm still pays per distinct orbit; the reference-sample statistics below pool both images.
"""

import json
import math
import time

import numpy as np
from ldpc import BpOsdDecoder

from gflowqec.certify import _kernel_basis, bound_ladder, exact_sectors, lifted_bases
from gflowqec.codes import (GROUP_180, assert_shape, build, canonical_key, generates_g, group_for,
                            orbit_key, orbit_rep, score_code)
from gflowqec.paths import DATA

T0 = time.time()       # the progress prints and the dumps' "wall" field count from import time


# =============================================================================================
# n = 36: the exact merit oracle of the two-block arena
# =============================================================================================

class Oracle:
    """The budgeted, memoized reward oracle every n = 36 search method shares.

    A method's cost is the number of distinct codes whose distance it paid for. The cache is keyed
    by the translation-gauge canonical key, so a shifted copy of an already-scored code is free, and
    `history` records every distinct evaluation in order: best-so-far curves and coverage are
    computed from it afterwards, one source of truth for every method.
    """

    def __init__(self, l, m, budget):
        self.l = l
        self.m = m
        self.budget = budget   # max distinct codes we may pay distance for
        self.cache = {}        # canonical key -> Scored
        self.history = []      # Scored in first-evaluation order, one per distinct code

    @property
    def calls(self):
        return len(self.history)

    @property
    def exhausted(self):
        return self.calls >= self.budget

    def score(self, a, b):
        key = canonical_key(self.l, self.m, a, b)

        # Already-known code (revisit or shifted copy): answering is free.
        if key in self.cache:
            return self.cache[key]

        # A genuinely new code costs one budget unit. Methods must check
        # `exhausted` before proposing; overspending is a bug, so fail loudly.
        assert not self.exhausted, "oracle budget exceeded"
        scored = score_code(self.l, self.m, a, b)   # exact n, k, d and merit
        self.cache[key] = scored
        self.history.append(scored)
        return scored

    def try_score(self, a, b):
        """score() if affordable (cached, or budget remains), else None.

        The GFN trains in batches, and a batch may propose more new codes than the
        remaining budget covers; unaffordable ones return None and are dropped from
        the training loss rather than crashing the run.
        """
        key = canonical_key(self.l, self.m, a, b)
        if key in self.cache or not self.exhausted:
            return self.score(a, b)
        return None


# =============================================================================================
# n = 180: the bias score
# =============================================================================================

P_QUIET = 1e-3
TRAIN_LAM = (-math.log10(30), -math.log10(3), 0.0, math.log10(3), math.log10(30))   # eta 30, 3, 1 both signs
HELD_OUT = (1.0, -1.0)         # eta 10 both signs: where every certified run is re-scored at the end


def lam_of(eta, sign):
    """Condition scalar from (eta, sign): eta = p_noisy / p_quiet >= 1, sign "Z" or "X" names the noisy sector."""
    return math.log10(eta) * (1.0 if sign == "Z" else -1.0)


def coeffs(lam):
    """(c_X, c_Z): the per-sector exponent coefficients at condition lam."""
    p_x = P_QUIET * 10 ** max(-lam, 0.0)
    p_z = P_QUIET * 10 ** max(lam, 0.0)
    return math.log10(1 / p_x) / 2, math.log10(1 / p_z) / 2


def bias_score(dX, dZ, lam):
    cx, cz = coeffs(lam)
    return min(cx * dX, cz * dZ)


def cond_vec(lam):
    """The two-channel condition the nets see: (Z-bias strength, X-bias strength), zero when unbiased."""
    return [max(lam, 0.0) / 1.5, max(-lam, 0.0) / 1.5]


# ------------------------------------------------------------------ the reference sample
REF_READ = DATA / "keys" / f"r174_bias_read_{GROUP_180}_c200k.json"
REF_RESOLVE = DATA / "keys" / f"r174_bias_resolve_{GROUP_180}_c5M.json"


def load(path):
    return json.load(open(path))


def reference_sample():
    """The 666 generating uniform draws of the n = 180 arena with exact per-sector distances.

    Each row carries dX, dZ and the flags exactX, exactZ. The per-sector ladders first ran at
    200,000 conflicts (REF_READ); the sectors they left censored were re-run at 5,000,000 conflicts
    (REF_RESOLVE), and a resolved sector is folded in here as exact.
    """
    rows = load(REF_READ)
    for o in load(REF_RESOLVE):
        if o["resolved"] is None:
            continue
        for r in rows:
            if r["i"] == o["i"]:
                r["d" + o["sector"]], r["exact" + o["sector"]] = o["resolved"], True
    return rows


_REF = None


def key_beta(lam, mass=0.8):
    """The reward temperature beta at condition lam, from the reference sample's scores.

    The rule is `derive_beta` at target mass 0.8, applied to the scores of the reference sample and
    of its mirror images: one deterministic beta per condition for every learning arm and seed. The
    oracle is deterministic, so the score distribution under uniform draws is a population quantity
    the reference sample already measures; a 200-draw estimate of it flips between 8 and 16 at three
    of the six conditions.
    """
    from gflowqec.policy import derive_beta     # the beta rule lives with the policies that use it
    global _REF
    if _REF is None:
        _REF = reference_sample()
    # mirror-pooled: M maps uniform draws to uniform draws with the sectors swapped, so the
    # (d_Z, d_X) pairs are a second exact sample of the same population; beta(lam) == beta(-lam)
    scores = [bias_score(r["dX"], r["dZ"], lam) for r in _REF] + [bias_score(r["dZ"], r["dX"], lam) for r in _REF]
    return derive_beta(scores, target_mass=mass)


def key_top(lam):
    """(top score, top fraction) at lam from the reference sample, MIRROR-POOLED, over codes exact in
    both sectors. The top score is the certification bar of the certified oracle and the readouts'
    definition of the top set; the fraction is the share of uniform draws that reach it."""
    global _REF
    if _REF is None:
        _REF = reference_sample()
    pairs = [(r["dX"], r["dZ"], r["exactX"] and r["exactZ"]) for r in _REF]
    pairs += [(dz, dx, ex) for dx, dz, ex in pairs]
    top = max(bias_score(dx, dz, lam) for dx, dz, ex in pairs if ex)
    frac = float(np.mean([bias_score(dx, dz, lam) >= top for dx, dz, _ in pairs]))
    return top, frac


# ------------------------------------------------------------------ the oracles
class BiasOracle:
    """The exact bias score at ONE condition, with a budget and an orbit-keyed cache.

    The interface every n = 180 and n = 300 arm talks to: score / cache / meta / calls / hits /
    budget / checkpoint / src. meta keeps (d_X, d_Z, exact flags, seconds, rep, src) per paid orbit,
    so any condition's score is recoverable from a dump without a new call."""

    def __init__(self, group, budget, lam):
        self.group, self.budget, self.lam = group, budget, lam
        self.cache, self.meta, self.calls, self.hits = {}, {}, 0, 0
        self.checkpoint = None     # set by the caller: a partial dump every 500 calls
        self.src = 0               # stream tag stamped on each new orbit: 0 = global draw, 1 = local-search move
        self.censored = 0          # paid orbits with a sector still censored after the retry

    def sectors(self, exps):
        """(d_X, d_Z) of an orbit already paid for (meta lookup, free)."""
        m = self.meta[orbit_key(self.group, exps)]
        return m["dX"], m["dZ"]

    def pay(self, exps):
        """Score the orbit if it is new (one paid call); returns its meta row, or None on budget."""
        assert generates_g(self.group, exps), "non-generating tuple reached the oracle"
        k = orbit_key(self.group, exps)
        if k in self.cache:
            self.hits += 1
            return self.meta[k]
        if self.calls >= self.budget:
            return None
        dX, exX, dZ, exZ, secs, cpu = exact_sectors(build(self.group, np.array(exps)))
        self.meta[k] = {"dX": dX, "dZ": dZ, "exactX": exX, "exactZ": exZ, "seconds": secs, "cpu": cpu,
                        "src": self.src, "rep": [int(v) for v in orbit_rep(self.group, exps)]}
        self.censored += int(not (exX and exZ))
        self.tick()
        # lam None = a caller that asks for values at explicit conditions only
        self.cache[k] = None if self.lam is None else self.value(exps, self.lam)
        return self.meta[k]

    def tick(self):
        self.calls += 1
        if self.calls % 500 == 0:
            print(f"  {self.calls}/{self.budget} calls  {time.time() - T0:.0f}s", flush=True)
            if self.checkpoint:
                self.checkpoint()

    def value(self, exps, lam):
        """The score at ANY condition of an orbit already paid for (free here: the sectors are exact)."""
        return float(bias_score(*self.sectors(exps), lam))

    def score(self, exps):
        m = self.pay(exps)
        return None if m is None else self.value(exps, self.lam)


class CertifiedOracle(BiasOracle):
    """The certified sector-bound oracle (module docstring). A paid call runs the two classical bound
    ladders (d_A, d_B); `value` at lam is the proxy score min(c_X d_A, c_Z d_B) unless it reaches the
    reference sample's top score at lam, when the exact ladders run once (certification, stamped with
    the phase and the condition that triggered it) and every later value of that orbit is exact.
    A censored bound is no bound, so such an orbit is certified outright. `calls` counts distinct
    orbits; `seconds` (exact) and `secondsBound` are kept per orbit so a readout can price a run."""

    def __init__(self, group, budget, lam):
        super().__init__(group, budget, lam)
        self.bar = {}              # lam -> the reference sample's top score at lam
        self.certified = 0
        self.phase = "run"         # "run" during the arm, "rescore" in the closing pass
        self.skip_odd = None       # the bound ladders' parity rule (certify.bound_ladder); scripts pin it per cell

    def bar_at(self, lam):
        if lam not in self.bar:
            self.bar[lam] = key_top(lam)[0]
        return self.bar[lam]

    def pay(self, exps):
        assert generates_g(self.group, exps), "non-generating tuple reached the oracle"
        k = orbit_key(self.group, exps)
        if k in self.cache:
            self.hits += 1
            return self.meta[k]
        if self.calls >= self.budget:
            return None
        A, B = lifted_bases(build(self.group, np.array(exps)), len(group_for(self.group)[1]))
        # skip_odd None = the parity ladder (verdict-identical, about 4x cheaper); False = the full
        # ladder, which every printed search, policy and matched-screen cell paid
        dA, exA, sA, cA = bound_ladder(A, skip_odd=self.skip_odd)
        dB, exB, sB, cB = bound_ladder(B, skip_odd=self.skip_odd)
        self.meta[k] = {"dA": dA, "dB": dB, "exactA": exA, "exactB": exB, "secondsBound": sA + sB,
                        "cpuBound": cA + cB,
                        "dX": None, "dZ": None, "exactX": None, "exactZ": None, "seconds": 0.0, "cpu": 0.0,
                        "cert": None, "certLam": None,
                        "src": self.src, "rep": [int(v) for v in orbit_rep(self.group, exps)]}
        self.tick()
        self.cache[k] = None if self.lam is None else self.value(exps, self.lam)
        return self.meta[k]

    def sectors(self, exps):
        m = self.meta[orbit_key(self.group, exps)]
        assert m["cert"] is not None, "sectors of an uncertified orbit"
        return m["dX"], m["dZ"]

    def value(self, exps, lam):
        m = self.meta[orbit_key(self.group, exps)]
        if m["cert"] is None:
            proxy = bias_score(m["dA"], m["dB"], lam)
            # certification happens at the BIASED conditions only: at lam 0 the proxy's top set is a
            # third of the space at 96 % precision, so certifying there would price a run at the
            # exact oracle's cost for a label the paper never uses (the readouts grade eta 10)
            if lam == 0.0 or ((m["exactA"] and m["exactB"]) and proxy < self.bar_at(lam)):
                return float(proxy)
            # proxy-top at a biased lam (or an unusable bound): pay the exact ladders once
            dX, exX, dZ, exZ, secs, cpu = exact_sectors(build(self.group, np.array(m["rep"])))
            m.update({"dX": dX, "dZ": dZ, "exactX": exX, "exactZ": exZ, "seconds": secs, "cpu": cpu,
                      "cert": self.phase, "certLam": lam})
            self.censored += int(not (exX and exZ))
            self.certified += 1
        return float(bias_score(m["dX"], m["dZ"], lam))

    def rescore(self, lams=HELD_OUT):
        """The closing pass: certify every paid orbit that is proxy-top at a held-out condition, so
        a dump re-scored there is exact wherever it matters; its seconds are stamped 'rescore'."""
        self.phase = "rescore"
        before = self.certified
        for m in list(self.meta.values()):
            for lam in lams:
                self.value(m["rep"], lam)
        print(f"  rescore pass at {lams}: {self.certified - before} more orbits certified "
              f"({self.certified} of {self.calls} in all)", flush=True)


def make_oracle(kind, group, budget, lam):
    """kind "cert" = the certified oracle; anything else = the exact one."""
    return (CertifiedOracle if kind == "cert" else BiasOracle)(group, budget, lam)


def dump_rows(oracle, header, path):
    """Write a run as JSON: the header, the oracle's counters, then one column per meta field with the
    rows in payment order (the dA / dB / cert columns appear on the certified oracle)."""
    # a checkpoint or a truncated run may have paid fewer orbits than the warm pool seeded, and the warm rows
    # are the first rows in payment order, so clamp the count a readout reads
    if header.get("warm_paid") is not None:
        header = {**header, "warm_paid": min(header["warm_paid"], oracle.calls)}
    order = list(oracle.meta.values())
    cols = {k: [m[k] for m in order] for k in (order[0] if order else ())}
    json.dump({**header, "calls": oracle.calls, "hits": oracle.hits, "distinct": len(oracle.cache),
               "censored": oracle.censored, "certified": getattr(oracle, "certified", None),
               "wall": time.time() - T0, **cols}, open(path, "w"))


# =============================================================================================
# n = 300: the code-capacity logical error rate
# =============================================================================================

# The BP-OSD decoder settings, a strong offline baseline: normalized min-sum BP with a serial schedule
# (serial converges better than flooding on quantum LDPC codes), then OSD combination sweep at order 7.
BP_OSD_KWARGS = dict(
    max_iter=100,
    bp_method="ms",          # normalized min-sum
    ms_scaling_factor=0.625,
    schedule="serial",
    osd_method="osd_cs",     # combination sweep
    osd_order=7,
)


def ler_score(name, exps, p, shots, run_seed=0, stream_exps=None, global_stream=False):
    """Estimated logical error rate of the lifted-product code `exps` under depolarizing(p), two-sector BP-OSD.

    The estimate is a pure function of (run_seed, orbit_key): both the noise stream and the scored
    matrices come from the orbit's canonical representative, so any two tuples of the same orbit get
    the same value, and arms that share a run_seed score any common orbit identically (common random
    numbers). Returns a dict with the pooled rate ("ler": either sector fails), the per-qubit rate
    ler_pq, the per-sector rates, the raw failure counts, and k.

    stream_exps: score this code on ANOTHER orbit's noise stream (the paired comparison of a local
    move against its incumbent); None = the orbit's own stream.
    global_stream: score every orbit of the run on the SAME error patterns, seeded by run_seed alone
    (common random numbers across all orbits); False = one stream per (run_seed, orbit).
    """
    # Score the orbit's canonical representative: same code up to gauge, and it makes
    # the estimate a pure function of (run_seed, orbit_key).
    exps = orbit_rep(name, exps)
    code = build(name, exps)
    assert_shape(code)    # every check row has the family's weight 9
    key_exps = exps if stream_exps is None else orbit_rep(name, stream_exps)
    assert not (global_stream and stream_exps is not None), "a global stream has no per-orbit pairing"
    if global_stream:
        seed_ints = [int(run_seed)]
    else:
        seed_ints = [int(run_seed)] + [int(v) for v in orbit_key(name, key_exps)]
    return ler_from_code(code, p, shots, seed_ints=seed_ints)


def ler_from_code(code, p, shots, seed_ints):
    """The oracle core for any CSS code object with matrix_x / matrix_z / dimension.

    Depolarizing code-capacity noise: each qubit suffers X, Y or Z with probability p/3 each, so the
    X-component (X or Y) has marginal 2p/3 and likewise the Z-component, correlated through Y.
    Standard CSS two-sector decoding: X-errors are decoded from the Z-check syndrome and vice versa,
    ignoring the Y correlation. A shot fails when either sector's residual acts nontrivially on the
    code space: the residual r (which BP-OSD leaves in the kernel of the decoded sector's checks) is a
    logical error iff it pairs oddly with some row of the OPPOSITE sector's kernel basis. Stabilizer
    rows pair evenly with everything in the kernel, so the basis needs no logical/stabilizer split.

    The cross-code ranking metric is the per-logical-qubit rate ler_pq = 1 - (1 - ler)^(1/k).
    `seed_ints` is the SeedSequence entropy list of the noise stream.
    """
    Hx = np.asarray(code.matrix_x, dtype=np.uint8) % 2
    Hz = np.asarray(code.matrix_z, dtype=np.uint8) % 2
    n = Hx.shape[1]

    # Anticommutation witnesses: X-residuals (in ker Hz) pair against ker(Hx), whose
    # non-stabilizer part is the Z-type logicals; mirrored for Z-residuals.
    wit_x = _kernel_basis(Hx)
    wit_z = _kernel_basis(Hz)

    # One noise stream per seed list.
    rng = np.random.default_rng(np.random.SeedSequence([int(v) for v in seed_ints]))
    r = rng.random((shots, n))
    x_err = (r < 2 * p / 3).astype(np.uint8)                # X or Y component
    z_err = ((r >= p / 3) & (r < p)).astype(np.uint8)       # Y or Z component

    dec_x = BpOsdDecoder(Hz, error_rate=2 * p / 3, **BP_OSD_KWARGS)
    dec_z = BpOsdDecoder(Hx, error_rate=2 * p / 3, **BP_OSD_KWARGS)
    syn_x = (x_err @ Hz.T) % 2
    syn_z = (z_err @ Hx.T) % 2

    fail_x = fail_z = fail_any = 0
    for i in range(shots):
        rx = (x_err[i] + np.asarray(dec_x.decode(syn_x[i]), dtype=np.uint8)) % 2
        rz = (z_err[i] + np.asarray(dec_z.decode(syn_z[i]), dtype=np.uint8)) % 2
        fx = bool(((wit_x @ rx) % 2).any())
        fz = bool(((wit_z @ rz) % 2).any())
        fail_x += fx
        fail_z += fz
        fail_any += fx or fz

    k = int(code.dimension)
    ler = fail_any / shots
    return {
        "ler": ler,
        "ler_pq": 1.0 - (1.0 - ler) ** (1.0 / k),   # the cross-code ranking metric
        "ler_x": fail_x / shots,
        "ler_z": fail_z / shots,
        "fails": [int(fail_x), int(fail_z), int(fail_any)],
        "shots": int(shots),
        "p": float(p),
        "k": k,
    }


class LEROracle:
    """s = -log(ler_pq + 1/(k*shots)) with a hard budget and an orbit-keyed cache.

    The 1/(k*shots) floor keeps a code with zero observed failures finite. run_seed for the noise
    stream is the ARM seed, so arms sharing a seed score any common orbit identically (common random
    numbers); different seeds are independent replications.
    """

    def __init__(self, group, budget, p=0.07, shots=1000, seed=0):
        self.group, self.budget, self.p, self.shots, self.seed = group, budget, p, shots, seed
        self.cache, self.meta, self.calls, self.hits = {}, {}, 0, 0
        self.checkpoint = None      # set by the caller: partial-result dump every 1,000 calls
        # stream tag written into meta with each NEW orbit: 0 = global (uniform draw or a fresh
        # policy sample), 1 = local (a local-search refinement), so a readout can split a run's yield
        self.src = 0
        # orbits whose value was paid on an INCUMBENT's noise stream (the paired local move), and
        # how many accepted moves were re-scored on their own stream (+1 call each)
        self.paired_keys, self.n_rescored = set(), 0
        # True = one noise stream per RUN (common random numbers across orbits);
        # False = one stream per (run, orbit)
        self.global_stream = False

    def score(self, exps, stream=None):
        # the arena guard: arms must pre-filter, and a non-generating tuple reaching the
        # oracle is a bug in the arm, not a data point
        assert generates_g(self.group, exps), "non-generating tuple reached the oracle"
        k = orbit_key(self.group, exps)
        if k in self.cache:
            self.hits += 1
            return self.cache[k]
        if self.calls >= self.budget:
            return None
        out = ler_score(self.group, exps, p=self.p, shots=self.shots, run_seed=self.seed,
                        stream_exps=stream, global_stream=self.global_stream)
        s = -np.log(out["ler_pq"] + 1.0 / (out["k"] * self.shots))   # k = |G| >= 1 in this family
        self.cache[k] = float(s)
        if stream is not None:
            self.paired_keys.add(k)
        # a 10-index representative is kept so a policy can learn from the cache
        self.meta[k] = {"ler_pq": out["ler_pq"], "k": out["k"],
                        "ler_x": out["ler_x"], "ler_z": out["ler_z"], "src": self.src,
                        "rep": [int(v) for v in orbit_rep(self.group, exps)]}
        self.calls += 1
        if self.calls % 1000 == 0:
            print(f"  {self.calls}/{self.budget} calls  {time.time() - T0:.0f}s", flush=True)
        if self.checkpoint and self.calls % 1000 == 0:
            self.checkpoint()
        return float(s)

    def rescore_own(self, exps):
        """An accepted local child becomes the incumbent, so pay ONE more call for its value on its
        OWN stream, the stream its children will be scored on. Overwrites the paired booking, so the
        orbit's value is a pure function of the orbit again. A child already scored on its own stream
        (a cache hit) is not re-paid."""
        k = orbit_key(self.group, exps)
        if k not in self.paired_keys:
            return self.cache[k]
        if self.calls >= self.budget:
            return None
        out = ler_score(self.group, exps, p=self.p, shots=self.shots, run_seed=self.seed)
        s = -np.log(out["ler_pq"] + 1.0 / (out["k"] * self.shots))
        self.cache[k] = float(s)
        self.meta[k].update({"ler_pq": out["ler_pq"], "k": out["k"],
                             "ler_x": out["ler_x"], "ler_z": out["ler_z"]})
        self.paired_keys.discard(k)
        self.n_rescored += 1
        self.calls += 1
        if self.calls % 1000 == 0:
            print(f"  {self.calls}/{self.budget} calls  {time.time() - T0:.0f}s", flush=True)
        if self.checkpoint and self.calls % 1000 == 0:
            self.checkpoint()
        return float(s)
