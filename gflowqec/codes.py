"""The codes of the paper: how a construction choice becomes a quantum code, and how codes are named.

Part A, two-block codes over Z_l x Z_m (the n = 36 and n = 72 answer keys).
A construction choice is (l, m, A, B): grid orders l and m, and the two defining polynomials
A, B given as collections of (i, j) exponent pairs, each pair meaning the monomial x^i y^j.
Every monomial x^i y^j is allowed. That makes the arena a strict superset of Bravyi et al.'s
bivariate bicycle (BB) codes (Nature 2024, Eq. 1), whose terms are pure powers of x or of y.
Under Eberhardt and Steffan's definition (arXiv:2407.03973, Sec. 4.1: any c, d over
F_2[Z_l x Z_m]) every code of the arena is a BB code. CSS validity is automatic for the
family, so every choice yields a valid code.

Part B, the lifted-product family at n = 180 and n = 300 (the construction of
arXiv:2607.28795, Definition 4):

    A = [a0  a1] = [g1 + g2 + g3,  e + g4 + g5]
    B = [b0  b1] = [h1 + h2 + h3,  e + h4 + h5]

1x2 base matrices over the group algebra F_2[G] of a non-abelian group G, combined as the
lifted product LP(A, B). Block length n = 5|G|, dimension k = |G| for the published code,
check weight 9. G = Z_6 x S_3 ("C6xS3", n = 180) or Z_10 x S_3 ("C10xS3", n = 300, the group
of the published [[300,60,14]]). A code is a ten-element tuple of group-element indices,
g1..g5 then h1..h5; the index order is fixed by `group_for` and every stored tuple uses it.

Part C, exact equivalence: two codes are the same object when their coloured Tanner graphs
are isomorphic, decided by nauty's canonical labelling.
"""

import csv
import hashlib
import itertools
from ast import literal_eval
from collections import namedtuple

import numpy as np
from qldpc.abstract import AbelianGroup, Group, GroupRing, RingArray, RingMember, SymmetricGroup
from qldpc.codes import CSSCode, LPCode
from qldpc.objects import Pauli

from gflowqec.certify import _logical_at_weight, sat_distance_bounded, sweep_distance


# =============================================================================================
# Part A: two-block codes over Z_l x Z_m
# =============================================================================================

# One scored construction: parameters [[n, k, d]], merit kd^2/n, and the canonical
# key identifying the code up to monomial shifts. k = 0 codes get d = 0, merit 0.
Scored = namedtuple("Scored", "n k d merit key")


def _shift_matrix(order, power):
    """The cyclic-shift permutation matrix: the identity with its columns rolled
    `power` steps. This is exactly "multiplication by x^power" in a cyclic group
    of size `order` (each 1 sits `power` spots right of the diagonal, wrapping)."""
    return np.roll(np.eye(order, dtype=int), power % order, axis=1)


def _poly_matrix(l, m, monomials):
    """Turn one polynomial into its lm x lm binary matrix.

    Each monomial x^i y^j becomes kron(shift_l^i, shift_m^j): the Kronecker product
    realizes "shift i steps in the x direction AND j steps in the y direction" on
    the l x m grid. The polynomial is the sum of its monomials, taken mod 2.
    """
    monomials = list(monomials)

    # Bring every exponent pair into the grid: x^i wraps around at l, y^j at m.
    reduced = set()
    for i, j in monomials:
        reduced.add((i % l, j % m))

    # Over GF(2), x^i y^j + x^i y^j = 0. A repeated monomial would silently vanish
    # and we would score a different polynomial than the caller intended.
    assert len(reduced) == len(monomials), "repeated monomial (cancels over GF(2))"

    total = np.zeros((l * m, l * m), dtype=int)
    for i, j in reduced:
        total += np.kron(_shift_matrix(l, i), _shift_matrix(m, j))
    return total % 2


def build_code(l, m, a, b):
    """Build the two-block code for polynomials a, b: collections of (i, j) exponent pairs.

    Glues the two circulant blocks into the CSS pair H_X = (A | B), H_Z = (B^T | A^T).
    The blocks commute (they are polynomials in the same shifts), which is exactly
    what makes H_X @ H_Z^T = A B + B A = 2 A B = 0 mod 2: CSS-valid by construction.
    (qldpc's own BBCode class is not used because it rejects polynomials that use only
    one of the two variables, and those legitimately appear in a full sweep of the space.)
    """
    A = _poly_matrix(l, m, a)
    B = _poly_matrix(l, m, b)
    Hx = np.hstack([A, B])
    Hz = np.hstack([B.T, A.T])
    return CSSCode(Hx, Hz)


def score_code(l, m, a, b):
    """Score one construction: exact n, k, d and merit kd^2/n.

    This is the reward oracle of the n = 36 arena. d is the NP-hard term; at this scale we
    pay for it exactly, and only when k > 0 (a code with no logical qubits has no distance).

    Two exact methods, chosen by k (measured at n = 36): qldpc's get_distance_exact
    enumerates a space that grows ~12x per +4 logical qubits (6 ms at k = 4, 74 ms at
    k = 8, >150 s at k = 24), while high-k codes have *small* distances, which a
    bottom-up weight sweep certifies in milliseconds. So: qldpc for small k, sweep for
    large k, and the sweep falls back to qldpc in the (unseen) case of a high-k code
    with distance beyond the sweep cap.
    """
    code = build_code(l, m, a, b)
    n = len(code)         # physical qubits, always 2*l*m
    k = code.dimension    # logical qubits, computed from GF(2) ranks (cheap, exact)
    key = canonical_key(l, m, a, b)

    # No logical qubits means nothing to protect: no distance, zero merit.
    if k == 0:
        return Scored(n, 0, 0, 0.0, key)

    if k >= 10:
        Hx = np.asarray(code.matrix_x, dtype=np.uint8)
        Hz = np.asarray(code.matrix_z, dtype=np.uint8)
        d = sweep_distance(Hx, Hz, 6)
    else:
        d = None
    if d is None:
        # qldpc returns a float (nan is possible only at k = 0), so pin it to int.
        d = int(code.get_distance_exact())
    return Scored(n, k, d, k * d * d / n, key)


def sector_distances(code, max_weight=6):
    """Exact per-sector distances (d_X, d_Z); the code distance is their minimum.

    d_X is the weight of the lightest X-type logical operator (in ker H_Z but not
    in rowspace H_X); d_Z is the Z-type mirror. Same method split as score_code:
    bottom-up sweep for high-k codes (their distances are small), qldpc's exact
    method otherwise.

    Every code of Part A satisfies d_X = d_Z: over an abelian group, H_Z equals H_X
    up to a qubit permutation (swap the two blocks, invert exponents), which maps
    either sector's logicals onto the other's weight for weight. We still compute
    both sectors independently: the equality is a structural claim worth checking
    per code, not assuming.
    """
    k = code.dimension
    if k == 0:
        return 0, 0

    Hx = np.asarray(code.matrix_x, dtype=np.uint8)
    Hz = np.asarray(code.matrix_z, dtype=np.uint8)
    d_x = _sector_sweep(Hz, Hx, max_weight) if k >= 10 else None
    d_z = _sector_sweep(Hx, Hz, max_weight) if k >= 10 else None
    if d_x is None:
        d_x = int(code.get_distance_exact(Pauli.X))
    if d_z is None:
        d_z = int(code.get_distance_exact(Pauli.Z))
    return d_x, d_z


def _sector_sweep(H_kernel, H_rowspace, max_weight):
    """One sector of the sweep: the weight of the lightest vector in ker(H_kernel)
    that is not in rowspace(H_rowspace), or None if it exceeds max_weight."""
    for weight in range(1, max_weight + 1):
        if _logical_at_weight(H_kernel, H_rowspace, weight):
            return weight
    return None


def canonical_key(l, m, a, b):
    """Canonical name for the code up to monomial shifts of A and B independently.

    Multiplying a polynomial by x^a y^b translates all its exponents; the code does
    not change (a stabilizer-generator change plus a qubit relabeling absorbs it).
    So the canonical form translates each polynomial to put one of its own monomials
    at the origin, and keeps the lexicographically smallest result. Deeper equivalences
    (X/Z swap, x <-> y when l = m, ring automorphisms) are not folded in: this is the
    translation-gauge key, and "distinct codes" in the answer keys counts these keys.
    """
    return (l, m, _canon_poly(l, m, a), _canon_poly(l, m, b))


def _canon_poly(l, m, monomials):
    """The canonical (translated, sorted) form of one polynomial's exponent set."""

    # Bring the exponents into the grid first, so translations below wrap correctly.
    reduced = []
    for i, j in monomials:
        reduced.append((i % l, j % m))

    # Dividing the polynomial by its own monomial x^a y^b translates every exponent
    # by (-a, -b), putting that monomial at the origin (0, 0). Each choice of a, b
    # gives an equally valid name for the same code; try them all.
    best = None
    for a, b in reduced:
        shifted = []
        for i, j in reduced:
            shifted.append(((i - a) % l, (j - b) % m))
        shifted = tuple(sorted(shifted))

        # Keep the lexicographically smallest name: that one is the canonical form.
        if best is None or shifted < best:
            best = shifted
    return best


# The answer key: one row per raw (a, b) pair, its scorecard, and the canonical key
# that identifies which distinct code this pair is a copy of.
Row = namedtuple("Row", "a b n k d merit key")


def all_polynomials(l, m, weight):
    """Every distinct weight-`weight` polynomial over the l x m grid.

    A polynomial is a set of `weight` distinct monomials x^i y^j, so this is just
    "all ways to choose `weight` cells from the grid". Returned as sorted tuples of
    (i, j) pairs, e.g. weight 3 on a 3x3 grid gives C(9,3) = 84 polynomials.
    """
    grid = []
    for i in range(l):
        for j in range(m):
            grid.append((i, j))
    return list(itertools.combinations(grid, weight))


def enumerate_space(l, m, weight):
    """Score every (A, B) pair of weight-`weight` polynomials. Returns a list of Row.

    The table has one row per raw pair (that is the space search methods walk), but
    the expensive scoring (distance!) runs once per *distinct code*: pairs with the
    same canonical key are shifted copies, so they share one score_code call.
    """
    polynomials = all_polynomials(l, m, weight)

    scored = {}  # canonical key -> Scored, so duplicates never pay for distance twice
    rows = []
    for a in polynomials:
        for b in polynomials:
            key = canonical_key(l, m, a, b)
            if key not in scored:
                scored[key] = score_code(l, m, a, b)
            s = scored[key]
            rows.append(Row(a, b, s.n, s.k, s.d, s.merit, key))
    return rows


def save_landscape(rows, path):
    """Write the answer key as CSV. Tuples (a, b, key) are stored as their repr."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(Row._fields)
        for r in rows:
            writer.writerow([repr(r.a), repr(r.b), r.n, r.k, r.d, r.merit, repr(r.key)])


def load_landscape(path):
    """Read the answer key back; exact inverse of save_landscape."""
    rows = []
    with open(path, newline="") as f:
        for record in csv.DictReader(f):
            rows.append(Row(
                literal_eval(record["a"]),      # repr of a tuple -> tuple
                literal_eval(record["b"]),
                int(record["n"]),
                int(record["k"]),
                int(record["d"]),
                float(record["merit"]),
                literal_eval(record["key"]),
            ))
    return rows


# =============================================================================================
# Part B: the lifted-product family over Z_6 x S_3 (n = 180) and Z_10 x S_3 (n = 300)
# =============================================================================================

GROUPS = {
    "C6xS3": lambda: AbelianGroup(6) * SymmetricGroup(3),      # order 36, n = 180
    "C10xS3": lambda: AbelianGroup(10) * SymmetricGroup(3),    # order 60, n = 300, the published group
}

_CACHE = {}


def group_for(name):
    """(GroupRing, ordered elements, identity), cached.

    The element order (sorted by repr) defines every index in every stored tuple.

    The group is rebuilt from its generators with NO custom lift, because qldpc ships
    permutation groups with one and then refuses them for non-commutative anti-representations;
    without a custom lift it falls back to the regular representation, which is the
    representation the lifted product is defined with.
    """
    if name not in _CACHE:
        src = GROUPS[name]()
        g = Group(*src.generators)
        els = sorted(g.generate(), key=repr)
        _CACHE[name] = (GroupRing(g), els, els[0] ** 0)
    return _CACHE[name]


def build(name, exps):
    """The lifted-product code LP(A, B) for ten group-element indices: g1..g5 then h1..h5."""
    ring, els, e = group_for(name)
    g = [els[int(i)] for i in exps[:5]]
    h = [els[int(i)] for i in exps[5:]]
    A = RingArray.build([[RingMember(ring, g[0], g[1], g[2]),
                          RingMember(ring, e, g[3], g[4])]], ring)
    B = RingArray.build([[RingMember(ring, h[0], h[1], h[2]),
                          RingMember(ring, e, h[3], h[4])]], ring)
    return LPCode(A, B)


# Which positions form each unordered entry of the two base matrices:
#   A = [ g1+g2+g3 , e+g4+g5 ]      B = [ h1+h2+h3 , e+h4+h5 ]
#         0  1  2      3  4              5  6  7      8  9
# Entries 1 and 3 already contain the identity, so the identity may not be reused there.
ENTRIES = [(0, 1, 2), (3, 4), (5, 6, 7), (8, 9)]
HAS_IDENTITY = (1, 3)


def identity_index(name):
    ring, els, e = group_for(name)
    return next(i for i, x in enumerate(els) if x == e)


def legal_positions(name):
    """(entries, forbidden-index-per-entry): the arena's real constraint set.

    Nothing requires a g to differ from an h (they live in different base matrices), nor
    a0's terms to differ from a1's, which are different entries. Verified directly:
    g1 == h1 gives n=300 k=60 with every check weight 9, and g1 == g4 gives n=300 k=62
    with every check weight 9. Both are legal codes of the family.

    The identity may not appear in entries 1 or 3, because a1 = e + g4 + g5 with g4 = e
    collapses to g5 over GF(2). Verified: the weight profile goes from {9: 240} to
    {7: 180, 9: 60}, a weight-7 object of a different family.

    So every code in the arena has the published shape at check weight 9; `assert_shape`
    is the guard, applied per draw.
    """
    ident = identity_index(name)
    return ENTRIES, {i: (ident if i in HAS_IDENTITY else None) for i in range(len(ENTRIES))}


def random_exps(rng, name):
    """A uniform draw from the arena: distinct WITHIN each entry, identity excluded
    from the two entries that already contain it, no constraint across entries."""
    n = len(group_for(name)[1])
    entries, forbidden = legal_positions(name)
    exps = [0] * sum(len(positions) for positions in entries)
    for ei, positions in enumerate(entries):
        pool = np.arange(n)
        if forbidden[ei] is not None:
            pool = pool[pool != forbidden[ei]]
        for p, v in zip(positions, rng.choice(pool, len(positions), replace=False)):
            exps[p] = int(v)
    return np.array(exps)


def assert_shape(code, weight=9):
    """Guard against impostor codes: every check row must have the published weight."""
    ws = set(np.asarray(code.matrix_x).sum(axis=1).tolist())
    ws |= set(np.asarray(code.matrix_z).sum(axis=1).tolist())
    assert ws == {weight}, f"check weights {sorted(ws)}, expected all {weight}"


def score(name, exps, max_weight=9, conflict_budget=60_000, min_weight=1):
    """(n, k, d, exact, weight) where d is EXACT if `exact`, else a genuine LOWER BOUND.

    `sat_distance_bounded` proves every rung below the stopping point UNSAT, so an
    unclosed ladder returns a lower bound: d = 9 with exact=False means "d >= 9", NOT
    "d = 9". Reading it as a point estimate would invert a comparison against a published
    distance. Only CLOSED ladders certify a code as worse than a reference; unclosed ones
    are censored at the cap and must be reported as such, because censoring inverts the
    ranking precisely at the top.
    """
    code = build(name, exps)
    n, k = len(code), code.dimension
    Hx = np.asarray(code.matrix_x) % 2
    Hz = np.asarray(code.matrix_z) % 2
    weight = int(Hx.sum(axis=1).max())
    # min_weight warm-starts the ladder at an ALREADY PROVEN lower bound; the default 1 is
    # the from-scratch ladder.
    d, exact = sat_distance_bounded(Hx, Hz, max_weight=max_weight,
                                    conflict_budget=conflict_budget, min_weight=min_weight)
    return n, k, d, exact, weight


# ---------------------------------------------------------------------------
# GAUGE STRUCTURE (spectra of Hx Hx^T compared, 6/6 per transform).
# The code is EXACTLY invariant under: right-multiplying all terms of an A-entry by any g
# (per entry, independently), LEFT-multiplying all terms of a B-entry by any g, and
# conjugating all of A's terms (or all of B's) by any g. A/B are chiral because they enter
# the lifted product on opposite sides. The ~6e17 tuple space has exactly 131,504,342
# (~1.3e8) orbits under these moves plus term reordering, renaming by an automorphism of
# Z10 x S3 and the two entry swaps a1 <-> a2, b1 <-> b2 (Burnside count, computed by
# readouts/background.py). Every tuple in an orbit builds the same code, so this count
# bounds the number of inequivalent codes from above and "at most about 3 x 10^8" holds.
# Consequence: raw element identities are pure gauge and carry no signal on their own.
# ---------------------------------------------------------------------------


def canonicalise(name, exps):
    """Map any tuple to the canonical gauge g1 = h1 = e.

    a0 is right-multiplied by g1^{-1}, b0 is LEFT-multiplied by h1^{-1}; a1 and b1 are
    untouched. In canonical coordinates the remaining eight elements are relative to g1/h1
    and therefore carry signal that raw identities provably cannot.
    """
    ring, els, e = group_for(name)
    idx = {x: i for i, x in enumerate(els)}
    g = [els[int(i)] for i in exps[:5]]
    h = [els[int(i)] for i in exps[5:]]
    ident = idx[e]
    return [ident, idx[g[1] * ~g[0]], idx[g[2] * ~g[0]], int(exps[3]), int(exps[4]),
            ident, idx[~h[0] * h[1]], idx[~h[0] * h[2]], int(exps[8]), int(exps[9])]


def _entry_canonical(idx, terms, side):
    """Canonical form of ONE entry under its (gauge x ordering) orbit, as sorted indices.

    Pivoting on term i (right-multiplying an A-entry by term_i^{-1}, LEFT-multiplying a
    B-entry) always yields {e, t_j t_i^{-1}, ...} INDEPENDENT of any prior gauge already
    applied, so the lexicographic minimum over pivots is a true canonical form: every
    member of the |G| x orderings orbit maps to the same tuple. Entries are independent
    because the gauge acts per entry.
    """
    best = None
    for piv in terms:
        if side == "A":
            rep = sorted(idx[t * ~piv] for t in terms)
        else:
            rep = sorted(idx[~piv * t] for t in terms)
        if best is None or rep < best:
            best = rep
    return best


def orbit_key(name, exps):
    """Canonical key for the FULL (column gauge x within-entry ordering) orbit of a tuple.

    Stronger than `canonicalise`, which pivots only on position 0 and so still depends on
    the ordering. Residual gauge NOT collapsed here: per-matrix conjugation (use
    `conj_table` for that). Used as the oracle cache key and to count distinct codes, so
    every tuple building the same code-up-to-conjugation counts as one object.
    """
    ring, els, e = group_for(name)
    idx = {x: i for i, x in enumerate(els)}
    g = [els[int(i)] for i in exps[:5]]
    h = [els[int(i)] for i in exps[5:]]
    return tuple(_entry_canonical(idx, [g[0], g[1], g[2]], "A")
                 + _entry_canonical(idx, [e, g[3], g[4]], "A")
                 + _entry_canonical(idx, [h[0], h[1], h[2]], "B")
                 + _entry_canonical(idx, [e, h[3], h[4]], "B"))


_MUL = {}


def mul_table(name):
    """mul_table[i, j] = index of els[i] * els[j], computed once per group."""
    if name not in _MUL:
        ring, els, e = group_for(name)
        idx = {x: i for i, x in enumerate(els)}
        n = len(els)
        t = np.zeros((n, n), dtype=np.int64)
        for i, a in enumerate(els):
            for j, b in enumerate(els):
                t[i, j] = idx[a * b]
        _MUL[name] = t
    return _MUL[name]


def generates_g(name, exps):
    """True iff BOTH sides' elements generate the full group.

    Arena rule: if <A-side> or <B-side> is a proper subgroup H < G, the code is not a
    genuine member of the family. When both sides live in H the lifted product decomposes
    into [G:H] disjoint copies of the smaller H-code: same n, same k, check weight still 9,
    but the distance and LER of the SMALL code. (With one side in H and the other
    generating G the Tanner graph stays connected, but the rule still excludes it.) The
    check is free (no oracle), and every arm applies it identically.

    Tested on the ORBIT REPRESENTATIVE, not the raw tuple: generation of the raw
    elements is gauge-variant (a gauge shift moves a0's support onto a coset while
    preserving the code), while the representative's entries each contain the identity,
    so its elements are exactly the relative generators the decomposition depends on.
    """
    exps = orbit_rep(name, exps)
    mul = mul_table(name)
    n = mul.shape[0]
    for gens in (set(int(v) for v in exps[:5]), set(int(v) for v in exps[5:])):
        seen = set(gens)
        frontier = list(seen)
        while frontier:
            nxt = []
            for a in frontier:
                for b in list(seen):
                    for c in (int(mul[a, b]), int(mul[b, a])):
                        if c not in seen:
                            seen.add(c)
                            nxt.append(c)
            frontier = nxt
            if len(seen) == n:
                break
        if len(seen) < n:
            return False
    return True


def orbit_rep(name, exps):
    """The canonical arena tuple of exps' orbit: the ten indices that rebuild the orbit's
    representative through `build`.

    Each canonical entry contains the identity exactly once (the pivot construction
    guarantees it, and within-entry distinctness makes it unique). For a0/b0 the identity
    is a legal free element and all three terms pass through; for a1/b1 the identity IS
    the implicit leading e, so it is dropped and the remaining two fill the free
    positions. Fixed point: orbit_key(orbit_rep(x)) == orbit_key(x).
    """
    key = orbit_key(name, exps)
    ident = identity_index(name)
    a1 = [v for v in key[3:6] if v != ident]
    b1 = [v for v in key[9:12] if v != ident]
    return np.array(list(key[0:3]) + a1 + list(key[6:9]) + b1)


_CONJ = {}


def conj_table(name):
    """conj_table[g, x] = index of els[g] * els[x] * els[g]^{-1}. The residual gauge in
    canonical coordinates (per-matrix conjugation); used as free
    reward-preserving replay augmentation."""
    if name not in _CONJ:
        ring, els, e = group_for(name)
        idx = {x: i for i, x in enumerate(els)}
        n = len(els)
        t = np.zeros((n, n), dtype=np.int64)
        for gi, g in enumerate(els):
            for xi, x in enumerate(els):
                t[gi, xi] = idx[g * x * ~g]
        _CONJ[name] = t
    return _CONJ[name]


class RelTables:
    """Multiplication, inverse and conjugacy-class tables for the relational (gauge-invariant)
    state encoding. Within an A entry the ratios t_i t_j^-1 are invariant under the column
    gauge (right scaling); within a B entry t_i^-1 t_j under the row gauge (left scaling);
    their conjugacy classes, and the joint (class_A, class_B) histogram, are invariant under
    the residual per-matrix conjugation as well."""

    def __init__(self, name):
        ring, els, e = group_for(name)
        idx = {x: i for i, x in enumerate(els)}
        self.n = len(els)
        self.e = idx[e]
        self.mul = mul_table(name)
        self.inv = np.array([idx[~x] for x in els])
        conj = conj_table(name)
        rep = np.array([conj[:, x].min() for x in range(self.n)])
        classes = sorted(set(rep.tolist()))
        self.cls = np.array([classes.index(r) for r in rep])
        self.n_cls = len(classes)
        self.dim = 4 * self.n + 2 * self.n_cls + self.n_cls ** 2


# entry -> (its positions, the side it is gauged on, whether e is an implicit term)
_REL_ENTRIES = [((0, 1, 2), "A", False), ((3, 4), "A", True),
                ((5, 6, 7), "B", False), ((8, 9), "B", True)]


def rel_features(chosen, T):
    """Relational features of a (possibly PARTIAL) ten-tuple prefix `chosen`: per-entry ratio
    histograms over G, class histograms per matrix, and the joint A x B class histogram.
    Only the terms already chosen (plus the implicit e) enter, so intermediate GFN states
    carry the relations among what has been placed so far."""
    n = T.n
    out = np.zeros(T.dim, dtype=np.float32)
    ratios = {"A": [], "B": []}
    for k, (ent, side, has_e) in enumerate(_REL_ENTRIES):
        terms = [chosen[p] for p in ent if p < len(chosen)]
        if has_e:
            terms.append(T.e)
        if side == "A":
            rs = [T.mul[t, T.inv[u]] for t in terms for u in terms if t != u]
        else:
            rs = [T.mul[T.inv[t], u] for t in terms for u in terms if t != u]
        for r in rs:
            out[k * n + r] += 1.0
        ratios[side] += rs
    base = 4 * n
    for r in ratios["A"]:
        out[base + T.cls[r]] += 1.0
    for r in ratios["B"]:
        out[base + T.n_cls + T.cls[r]] += 1.0
    base += 2 * T.n_cls
    for r in ratios["A"]:
        for u in ratios["B"]:
            out[base + T.cls[r] * T.n_cls + T.cls[u]] += 1.0
    return out


def draw_gen(rng, group):
    """A uniform draw from the GENERATING arena: redraw until both sides generate G.
    The check is free; 65.9% of raw draws pass (measured over 20,000 draws)."""
    while True:
        e = random_exps(rng, group)
        if generates_g(group, e):
            return e


def mirror(exps, T):
    """(h^-1, g^-1): the sector-swapping involution. Canonical form is preserved (e^-1 = e)."""
    return [int(T.inv[v]) for v in exps[5:]] + [int(T.inv[v]) for v in exps[:5]]


# The neighbourhood counts of the paper are read at n = 180 only.
GROUP_180 = "C6xS3"


def canonical_half(half):
    """The bottleneck half up to the residual per-matrix conjugation: the lexicographic
    minimum over the conjugating elements of the entry-sorted image. A gauge copy of one
    code then counts as ONE neighbourhood."""
    conj = conj_table(GROUP_180)    # cached by conj_table
    best = None
    for g in range(len(conj)):
        img = [int(conj[g, v]) for v in half]
        cand = tuple(sorted(img[:3]) + sorted(img[3:]))
        if best is None or cand < best:
            best = cand
    return best


_HALF = {}


def half_class(rep, sign):
    """The gauge-canonical class of the orbit's BOTTLENECK half (Z-biased: the second half of
    the rep, X-biased: the first): the neighbourhood of a code at n = 180. Memoised on the
    raw half; canonical_half is the expensive part and halves repeat heavily across runs."""
    h = tuple(rep[5:]) if sign == "Z" else tuple(rep[:5])
    if h not in _HALF:
        _HALF[h] = canonical_half(h)
    return _HALF[h]


# =============================================================================================
# Part C: exact equivalence of two codes through their coloured Tanner graphs
# =============================================================================================

def tanner(Hx, Hz, swap=False):
    """The colored Tanner graph: vertices 0..nq-1 qubits, then the X checks, then the Z checks. The ordered color
    classes are [qubits, X checks, Z checks], or [qubits, Z checks, X checks] when swap (the mirror test)."""
    # pynauty (GPL-3.0-or-later) is imported only here and in certificate(), so the rest of the library
    # runs without it; it is needed only for the Tanner-class counts (see THIRD-PARTY.md)
    import pynauty
    nq, mx, mz = Hx.shape[1], Hx.shape[0], Hz.shape[0]
    adj = {v: [] for v in range(nq + mx + mz)}
    for r, c in zip(*np.nonzero(Hx)):
        adj[nq + int(r)].append(int(c))
    for r, c in zip(*np.nonzero(Hz)):
        adj[nq + mx + int(r)].append(int(c))
    qs, xs, zs = set(range(nq)), set(range(nq, nq + mx)), set(range(nq + mx, nq + mx + mz))
    return pynauty.Graph(nq + mx + mz, directed=False, adjacency_dict=adj,
                         vertex_coloring=[qs, zs, xs] if swap else [qs, xs, zs])


def certificate(Hx, Hz, swap=False):
    """sha256 of the nauty certificate of the code's Tanner graph. Two codes are the same object
    (qubits relabelled, checks relabelled within their colour) iff their certificates are equal;
    swap=True compares against the code with its X and Z sectors exchanged."""
    import pynauty   # optional dependency, see tanner()
    return hashlib.sha256(pynauty.certificate(tanner(Hx, Hz, swap))).hexdigest()
