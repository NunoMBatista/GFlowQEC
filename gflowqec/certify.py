"""Exact distance certification of CSS codes.

Two instruments, each with its own certificate.

Part 1, the two-sided squeeze (n = 36 and 72):

  - lower bound, certificate by exhaustion: a bit-packed sweep checks EVERY
    vector of weight <= w in both sectors and finds no logical, so d > w.
    Each sector's checks are packed into uint64 words (one word up to 64
    checks), so the kernel test for millions of supports is a few
    vectorized XORs.

  - upper bound, certificate by example: qldpc's decoder-based randomized
    search (the algorithm from Bravyi et al.'s BB paper, arXiv:2308.07915)
    reports the weight of an actual logical operator it constructed, so
    d <= that weight.

When the sweep finds a logical, or the two bounds meet, d is exact. When
they do not meet, certify_distance returns the honest interval and the
caller escalates BY RAISING THE SWEEP CAP; it can never overclaim. Escalate
the sweep, not the trials: the randomized bound can sit stubbornly above the
true d on degenerate codes (observed on a factorized univariate (6,3) code,
d = 2, bound stuck at 8 across 5000 trials), while a sweep capped at the
loose upper bound always closes the squeeze exactly.

Part 2, the SAT decision ladder (the n = 72 codes the sweep left open, and
the n = 180 codes).

FORMULATION. A Z-type logical is a vector v over GF(2) with H_X v = 0 (it
commutes with every X check) that is NOT in the row space of H_Z (it is not just
a product of stabilizers). The second condition is easier to write through the
logicals: v is nontrivial exactly when it anticommutes with at least one X-type
logical operator, so at least one of the parities L_i . v must be 1.

Rather than a single minimisation, the ladder asks a sequence of decision
questions, "is there a nontrivial logical of weight <= w", for w = 1, 2, ...
The first w that answers yes IS the distance, which mirrors the sweep's
semantics exactly and means a partial run still yields an honest lower bound.

Part 3, the n = 180 ladders: the per-sector exact ladders and the cheap
classical ladders on the lifted bases that the n = 180 oracle pays first.
"""

import itertools
import time
from collections import namedtuple

import numpy as np
from ldpc import mod2
from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Cadical153
from qldpc.objects import Pauli


# ============================================================================
# Part 1: the two-sided squeeze (sweep lower bound, decoder upper bound)
# ============================================================================

# The squeeze result: d is certified to lie in [lower, upper]; exact iff equal.
Certificate = namedtuple("Certificate", "lower upper")


def certify_distance(code, max_sweep_weight=5, trials=1000):
    """Certify the distance of a CSS code: exact if the squeeze closes.

    Sweeps both sectors exhaustively up to max_sweep_weight (exact d if a
    logical is found there, since everything lighter was ruled out), then
    asks qldpc's randomized search for an upper bound. Cost of the sweep is
    ~C(n, w) at the largest weight, so pick max_sweep_weight to fit the
    budget; 5 at n = 72 is ~14M supports, a few seconds.

    The n = 72 answer key (data/keys/certified_l6m6_w3.csv) is this function
    at max_sweep_weight 5 and 1000 trials; the codes it left open were re-run
    at max_sweep_weight 6 (escalated_l6m6_w3.csv).
    """
    assert code.dimension > 0, "k = 0 codes have no distance to certify"
    Hx = np.asarray(code.matrix_x, dtype=np.uint8)
    Hz = np.asarray(code.matrix_z, dtype=np.uint8)

    swept = sweep_distance(Hx, Hz, max_sweep_weight)
    if swept is not None:
        return Certificate(swept, swept)

    lower = max_sweep_weight + 1                       # exhaustion: d > swept cap
    # cutoff stops the trials early once the bound meets the exhaustive floor
    # (it cannot go below it). Try the X sector first; skip Z if already closed.
    upper = int(code.get_distance_bound_with_decoder(
        Pauli.X, num_trials=trials, cutoff=lower))
    if upper > lower:
        upper = min(upper, int(code.get_distance_bound_with_decoder(
            Pauli.Z, num_trials=trials, cutoff=lower)))
    assert upper >= lower, "randomized bound below the exhaustive one: bug"
    return Certificate(lower, upper)


def sweep_distance(Hx, Hz, max_weight):
    """Exact distance by exhaustive sweep, or None if it exceeds max_weight.

    The first weight with a logical in either sector IS the distance, since
    every lighter vector was checked and ruled out. The kernel test runs
    vectorized over batches of supports.
    """
    for weight in range(1, max_weight + 1):
        # Z-type logicals live in ker(Hx) \ rowspace(Hz), X-type the mirror.
        if _logical_at_weight(Hx, Hz, weight) or _logical_at_weight(Hz, Hx, weight):
            return weight
    return None


def _row_reduce(H):
    """Gaussian elimination over GF(2): returns (pivot columns, reduced rows)."""
    rows = [row.copy() for row in np.asarray(H, dtype=np.uint8)]
    remaining = list(range(len(rows)))  # indices of rows not yet used as a pivot
    pivots = []
    reduced = []
    for column in range(H.shape[1]):
        # Find a remaining row with a 1 in this column.
        found = None
        for index in remaining:
            if rows[index][column]:
                found = index
                break
        if found is None:
            continue
        remaining.remove(found)
        # Clear this column from every other remaining row.
        for index in remaining:
            if rows[index][column]:
                rows[index] ^= rows[found]
        pivots.append(column)
        reduced.append(rows[found])
    return pivots, reduced


def _logical_at_weight(H_kernel, H_rowspace, weight, batch=200_000):
    """Is there a weight-`weight` vector in ker(H_kernel) not in rowspace(H_rowspace)?

    Kernel test first, vectorized: each column of H_kernel is packed into
    uint64 words (one per 64 checks), so a support is in the kernel iff the
    XOR of its packed columns is zero in every word. Only the rare kernel
    members get the (Python) stabilizer check.
    """
    n = H_kernel.shape[1]
    packed = _packed_columns(H_kernel)
    pivots, reduced_rows = _row_reduce(H_rowspace)

    supports = itertools.combinations(range(n), weight)
    while True:
        chunk = np.fromiter(
            itertools.chain.from_iterable(itertools.islice(supports, batch)),
            dtype=np.int64,
        ).reshape(-1, weight)
        if len(chunk) == 0:
            return False

        # packed[chunk] is (batch, weight, words); XOR the columns of each
        # support together, then a support is in the kernel iff every word is 0.
        syndromes = np.bitwise_xor.reduce(packed[chunk], axis=1)
        for support in chunk[~syndromes.any(axis=1)]:
            # In the kernel: reduce against the stabilizer rows; a nonzero
            # remainder means it is a genuine logical operator.
            vector = np.zeros(n, dtype=np.uint8)
            vector[support] = 1
            for pivot, row in zip(pivots, reduced_rows):
                if vector[pivot]:
                    vector ^= row
            if vector.any():
                return True


def _packed_columns(H):
    """Each column of H as uint64 words: bit i of word w is row 64*w + i.

    One word suffices up to 64 checks; more checks take more words. The sweep
    XORs whole rows of this array, so extra words cost proportionally,
    nothing else changes.
    """
    H = np.asarray(H, dtype=np.uint64)
    words = (H.shape[0] + 63) // 64
    packed = np.zeros((H.shape[1], words), dtype=np.uint64)
    for w in range(words):
        rows = H[64 * w: 64 * (w + 1)]
        bits = np.left_shift(np.uint64(1), np.arange(rows.shape[0], dtype=np.uint64))
        packed[:, w] = rows.T @ bits
    return packed


# ============================================================================
# Part 2: the SAT decision ladder
# ============================================================================

def _xor_clauses(literals, pool, value):
    """CNF for XOR(literals) == value, Tseitin-chained through fresh variables.

    A 2-input XOR gate z = x ^ y is four clauses. Chaining them left to right
    turns a k-input parity into k-1 gates, so the encoding is linear in k rather
    than exponential.
    """
    clauses = []
    if not literals:
        return clauses if value == 0 else [[]]      # empty XOR is 0; ==1 is unsat

    running = literals[0]
    for nxt in literals[1:]:
        fresh = pool.id()
        # fresh <-> running XOR nxt
        clauses += [[-fresh, running, nxt],
                    [-fresh, -running, -nxt],
                    [fresh, -running, nxt],
                    [fresh, running, -nxt]]
        running = fresh
    clauses.append([running] if value else [-running])
    return clauses


def _sector_distance_at_most(check, logicals, weight, conflict_budget=None):
    """Is there a v with check @ v = 0, some logical anticommuting, |v| <= weight?

    Returns True / False, or None when a conflict budget was given and the solver
    ran out before deciding. None means "unknown", never "no".
    """
    n = check.shape[1]
    pool = IDPool(start_from=1)
    v = [pool.id() for _ in range(n)]           # v[j] true means qubit j in support

    clauses = []
    # Commute with every check: each row is an even-parity constraint on v.
    for row in check:
        support = [v[j] for j in range(n) if row[j]]
        clauses += _xor_clauses(support, pool, 0)

    # Nontrivial: at least one logical parity is odd. One indicator per logical,
    # tied to its parity, then a clause requiring at least one indicator.
    indicators = []
    for row in logicals:
        support = [v[j] for j in range(n) if row[j]]
        if not support:
            continue
        flag = pool.id()
        # flag -> parity is 1. The reverse implication is not needed: we only
        # ever need ONE witness, and leaving it one-directional halves the CNF.
        parity_true = _xor_clauses(support, pool, 1)
        # guard every clause of the parity encoding by -flag
        clauses += [[-flag] + c for c in parity_true]
        indicators.append(flag)
    if not indicators:
        return False
    clauses.append(indicators)

    # Weight budget.
    card = CardEnc.atmost(lits=v, bound=weight, vpool=pool, encoding=EncType.seqcounter)
    clauses += card.clauses

    with Cadical153(bootstrap_with=clauses) as solver:
        if conflict_budget is None:
            return solver.solve()
        solver.conf_budget(conflict_budget)
        return solver.solve_limited()          # None if the budget ran out


def sat_distance_bounded(Hx, Hz, max_weight=None, conflict_budget=200_000,
                         min_weight=1):
    """Distance with a per-rung conflict budget: (value, exact).

    The ladder's cost is dominated by the single UNSAT proof at w = d - 1, and
    that cost grows about 10x per distance step, with a wide spread among codes
    at the same distance. Budgeting each rung keeps a bulk run from stalling on
    a handful of hard codes, and keeps the semantics honest: all rungs below
    the stopping point were proved UNSAT, so the value returned is a genuine
    lower bound and `exact` says whether the ladder closed. This is the same
    "exact when closed, honest interval when not" contract as the squeeze; it
    can never overclaim.

    MIN_WEIGHT warm-starts the ladder from an ALREADY CERTIFIED lower bound, e.g.
    the exhaustive sweep's d_lower, skipping rungs whose UNSAT verdict is already
    known. This CHAINS TWO INSTRUMENTS, so it is only sound when the supplied
    bound is itself exhaustive rather than randomized; passing a decoder-based
    upper bound here would silently overclaim.

    Data it made: the n = 72 codes the sweep left open
    (data/keys/closed_l6m6_w3.csv: 5,000,000 conflicts, warm-started at the
    sweep's d_lower), and through codes.score the minimum-over-sectors
    distance of the 1,000 n = 180 reference draws
    (data/keys/r169_headroom_C6xS3_n1000.json: 200,000 conflicts, cap 12).
    """
    Hx = np.asarray(Hx, dtype=np.uint8) % 2
    Hz = np.asarray(Hz, dtype=np.uint8) % 2
    limit = max_weight if max_weight is not None else Hx.shape[1]

    lx = _kernel_basis(Hz)
    lz = _kernel_basis(Hx)

    for weight in range(max(1, min_weight), limit + 1):
        for check, logicals in ((Hx, lx), (Hz, lz)):
            verdict = _sector_distance_at_most(check, logicals, weight, conflict_budget)
            if verdict is None:
                # Undecided at this rung. Everything below it was UNSAT, so the
                # distance is at least `weight`, but we cannot say more.
                return weight, False
            if verdict:
                return weight, True
    return limit, False


def _kernel_basis(matrix):
    """A basis of the GF(2) null space of `matrix`, as rows.

    Written out rather than reusing _row_reduce above, which returns row
    ECHELON form (it only clears a pivot column from the rows it has not yet
    consumed). The null-space read-off below needs FULL reduced row echelon,
    where every pivot column is clear in every other row.
    """
    a = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rows, cols = a.shape

    pivot_row_of_col = {}
    r = 0
    for c in range(cols):
        target = None
        for i in range(r, rows):
            if a[i, c]:
                target = i
                break
        if target is None:
            continue                       # free column
        a[[r, target]] = a[[target, r]]
        for i in range(rows):              # clear from EVERY other row
            if i != r and a[i, c]:
                a[i] ^= a[r]
        pivot_row_of_col[c] = r
        r += 1
        if r == rows:
            break

    # One basis vector per free column: set that column, then fill each pivot
    # column with the RREF entry that column contributes.
    basis = []
    for f in (c for c in range(cols) if c not in pivot_row_of_col):
        vec = np.zeros(cols, dtype=np.uint8)
        vec[f] = 1
        for c, ri in pivot_row_of_col.items():
            if a[ri, f]:
                vec[c] = 1
        basis.append(vec)
    return np.array(basis, dtype=np.uint8) if basis else np.zeros((0, cols), np.uint8)


# ============================================================================
# Part 3: the n = 180 ladders
# ============================================================================

LADDER_CAP = 12
CONFLICTS = 200_000            # every rung first
RETRY_LOW, RETRY_HIGH = 1_000_000, 5_000_000    # a censored rung <= 7 retries at 1M, >= 8 at 5M
BOUND_CONFLICTS = 1_000_000    # the classical bound ladders (no censored rung seen in 328 codes)


def sector_ladder(check, logicals, budget_retry_of):
    """(distance, exact, seconds, cpu): climb from weight 1; a rung the first budget cannot decide is
    retried once at the rung's retry budget; still undecided = censored (a lower bound). seconds =
    wall clock; cpu = this process's CPU time over the ladder (time.process_time; the SAT solver runs
    in-process). The readouts price a call with cpu, because wall seconds under many parallel
    processes are set by the scheduler, not by the sampler."""
    t, c = time.time(), time.process_time()
    for w in range(1, LADDER_CAP + 1):
        r = _sector_distance_at_most(check, logicals, w, conflict_budget=CONFLICTS)
        if r is None:
            r = _sector_distance_at_most(check, logicals, w, conflict_budget=budget_retry_of(w))
        if r is True:
            return w, True, time.time() - t, time.process_time() - c
        if r is None:
            return w, False, time.time() - t, time.process_time() - c
    return LADDER_CAP + 1, False, time.time() - t, time.process_time() - c


def _all_ones_in_rowspace(H):
    """True when the all-ones vector lies in the GF(2) row space of H. Every kernel vector is then
    orthogonal to all-ones, i.e. has EVEN weight, so no odd rung of the bound ladder can hold a
    witness. The lifted bases have odd column weight, so the sum of all their rows is all-ones."""
    ones = np.ones((1, H.shape[1]), dtype=np.uint8)
    return int(mod2.rank(H)) == int(mod2.rank(np.vstack([H, ones])))


def bound_ladder(H, cap=LADDER_CAP, skip_odd=None):
    """(classical minimum weight of ker H, exact, seconds, cpu): the sector ladder with the identity
    as the logicals, so "some logical anticommutes" reads "v is nonzero". One budget, no retry.

    skip_odd: None = skip the odd rungs iff all-ones lies in the row space of H (the PARITY ladder:
    verdict-identical, about 4x cheaper at n 180); False = every rung (the FULL ladder). Every
    printed search, policy and matched-screen cell at n = 180 paid the full ladder; the paid rows of
    the earlier supervised screens went through the parity ladder. Verdicts are identical either
    way; only the stamped seconds differ."""
    t, c = time.time(), time.process_time()
    if skip_odd is None:
        skip_odd = _all_ones_in_rowspace(H)
    eye = np.eye(H.shape[1], dtype=np.uint8)
    for w in range(1, cap + 1):
        if skip_odd and w % 2:
            continue
        r = _sector_distance_at_most(H, eye, w, conflict_budget=BOUND_CONFLICTS)
        if r is True:
            return w, True, time.time() - t, time.process_time() - c
        if r is None:
            return w, False, time.time() - t, time.process_time() - c
    return cap + 1, False, time.time() - t, time.process_time() - c


def lifted_bases(code, n):
    """A = [L(a0) | L(a1)] from matrix_x and B = [L(b0) | L(b1)] from matrix_z, n x 2n each with
    n = |G| the block size (blocks 0 and 2 of matrix_x, blocks 0 and 1 of matrix_z). L(a) is the
    n x n matrix of multiplication by the group-algebra element a. Their bound_ladder values
    (dA, dB) are the cheap classical score the n = 180 oracle pays before any exact ladder."""
    Hx = np.asarray(code.matrix_x) % 2
    Hz = np.asarray(code.matrix_z) % 2
    return np.hstack([Hx[:n, :n], Hx[:n, 2 * n:3 * n]]), np.hstack([Hz[:n, :n], Hz[:n, n:2 * n]])


def exact_sectors(code):
    """(d_X, exactX, d_Z, exactZ, seconds, cpu): the two exact per-sector ladders, each from weight 1
    at 200,000 conflicts with one retry (1M at rungs <= 7, 5M above). This made the d_X, d_Z of every
    certified row in the n = 180 search cells (data/n180/)."""
    Hx = np.asarray(code.matrix_x) % 2
    Hz = np.asarray(code.matrix_z) % 2
    retry = lambda w: RETRY_LOW if w <= 7 else RETRY_HIGH
    dX, exX, sX, cX = sector_ladder(Hx, _kernel_basis(Hz), retry)
    dZ, exZ, sZ, cZ = sector_ladder(Hz, _kernel_basis(Hx), retry)
    return dX, exX, dZ, exZ, sX + sZ, cX + cZ


def ladder_from(check, logicals, start, budget, cap=12):
    """(distance, exact). Climbs from `start`; a rung the budget cannot decide is returned censored.

    Warm-started at a code's known minimum distance d (from sat_distance_bounded), with one budget
    and no retry. This made the per-sector d_X, d_Z of the n = 180 reference sample:
    data/keys/r174_bias_read_C6xS3_c200k.json (budget 200,000, start d), then
    data/keys/r174_bias_resolve_C6xS3_c5M.json (budget 5,000,000 on the censored sectors, start at
    the censored rung)."""
    for w in range(start, cap + 1):
        r = _sector_distance_at_most(check, logicals, w, conflict_budget=budget)
        if r is True:
            return w, True
        if r is None:
            return w, False           # d_sector >= w, undecided at w
    return cap + 1, False
