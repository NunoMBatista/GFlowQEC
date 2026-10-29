"""The size of the n = 300 space (in the introduction and in the appendix on the n = 300 oracle and settings).

"the tuple makes ten choices among the sixty elements of the group, giving at most about 3 x 10^8
codes up to equivalence". The receipt is an exact count, by Burnside's lemma, of the orbits of the
family's tuples under relabellings that are known to give the same code (qubits and checks
relabelled, X and Z colours kept):

  (a) reordering the terms inside an entry;
  (b) translating an entry (right-multiplying every term of an A entry by one element, LEFT-
      multiplying every term of a B entry); for the entry 1 + g4 + g5 only translations that keep
      the identity in it stay in the family, and they give the same classes as the first entry;
  (c) renaming the group elements by an automorphism of Z10 x S3;
  (d) conjugating all of A's terms (or all of B's) by one element;
  plus the two entry swaps a1 <-> a2 and b1 <-> b2 (a permutation of the blocks of the lifted
  product's matrices, so the same code).

Every tuple in one orbit builds the same code, so the number of orbits is an upper bound on the
number of inequivalent codes. The paper's list (a) to (d) alone gives 517,872,884 orbits; adding the
two entry swaps gives 131,504,342 = 1.3 x 10^8, inside the printed "at most about 3 x 10^8". No lower
bound is computed here. A code and its mirror image count as two codes, as in the benchmark section.

Method: per entry, the orbit of a 3-element set under (a) and (b) is its translation class (right
classes for A, left classes for B: 571 each). A pair of renamings acts on the four entries' classes,
and Burnside's lemma averages its fixed points over the group. No data is read; about 6 s on a laptop.
Run from the repo root:
    python -m readouts.background
"""

import itertools
import time
from fractions import Fraction

from gflowqec.codes import group_for, mul_table

T0 = time.time()
print("Sec. 1 and App. E.2 text")

name = "C10xS3"
mul = mul_table(name)
n = mul.shape[0]                                     # |Z_10 x S_3| = 60
ring, els, e_el = group_for(name)
e = next(i for i, x in enumerate(els) if x == e_el)
inv = [next(j for j in range(n) if mul[i, j] == e) for i in range(n)]


# ---- Aut(G) by brute force over the images of a small generating set ----
def generated(gens):
    """The set of elements reached from the identity by multiplying by the generators."""
    seen = {e}
    frontier = [e]
    while frontier:
        nxt = []
        for a in frontier:
            for s in gens:
                c = int(mul[a, s])
                if c not in seen:
                    seen.add(c)
                    nxt.append(c)
        frontier = nxt
    return seen


# The smallest generating set, by trying sets of one, two, three elements.
gens = None
for r in (1, 2, 3):
    for cand in itertools.combinations(range(n), r):
        if len(generated(cand)) == n:
            gens = cand
            break
    if gens:
        break


def order(x):
    k, y = 1, x
    while y != e:
        y = int(mul[y, x])
        k += 1
    return k


orders = [order(x) for x in range(n)]

# Every element as parent * generator (a breadth-first tree), so an image of the generators
# extends to a candidate map of the whole group.
parent = {e: None}
frontier = [e]
while frontier:
    nxt = []
    for a in frontier:
        for si, s in enumerate(gens):
            c = int(mul[a, s])
            if c not in parent:
                parent[c] = (a, si)
                nxt.append(c)
    frontier = nxt

# An automorphism sends each generator to an element of the same order; keep the candidate
# maps that are bijective and respect the multiplication table.
auts = []
cands = [[y for y in range(n) if orders[y] == orders[s]] for s in gens]
for imgs in itertools.product(*cands):
    phi = [None] * n
    phi[e] = e
    for x in parent:
        if parent[x] is None:
            continue
        a, si = parent[x]
        phi[x] = int(mul[phi[a], imgs[si]])
    if len(set(phi)) != n:
        continue
    ok = all(phi[int(mul[x, y])] == int(mul[phi[x], phi[y]]) for x in range(n) for y in range(n))
    if ok:
        auts.append(tuple(phi))
# Inner automorphisms: conjugation x -> g x g^-1.
inn = sorted({tuple(int(mul[mul[g, x], inv[g]]) for x in range(n)) for g in range(n)})

# ---- translation classes of 3-element sets: A side right, B side left ----
subsets = list(itertools.combinations(range(n), 3))


def canon_right(S):
    return min(tuple(sorted(int(mul[s, g]) for s in S)) for g in range(n))


def canon_left(S):
    return min(tuple(sorted(int(mul[g, s]) for s in S)) for g in range(n))


R = sorted({canon_right(S) for S in subsets})
L = sorted({canon_left(S) for S in subsets})
Ridx = {c: i for i, c in enumerate(R)}
Lidx = {c: i for i, c in enumerate(L)}


def act(phi, classes, idx, canon):
    """The permutation a renaming phi induces on the translation classes."""
    return [idx[canon(tuple(phi[s] for s in c))] for c in classes]


def fixed(perm):
    return sum(1 for i, j in enumerate(perm) if i == j)


def compose(p, q):  # p after q
    return [p[q[i]] for i in range(len(q))]


# Fixed classes of each renaming, and of its square: an entry swap combined with phi fixes the
# pair (c1, c2) when c1 = phi(c2) and c2 = phi(c1), that is when c1 is fixed by phi applied twice.
fR, fR2, fL, fL2 = {}, {}, {}, {}
for phi in auts:
    pr = act(phi, R, Ridx, canon_right)
    pl = act(phi, L, Lidx, canon_left)
    fR[phi], fR2[phi] = fixed(pr), fixed(compose(pr, pr))
    fL[phi], fL2[phi] = fixed(pl), fixed(compose(pl, pl))

inn_set = set(inn)


def comp_phi(p, q):
    return tuple(p[q[i]] for i in range(n))


def inverse_phi(p):
    r = [0] * n
    for i, v in enumerate(p):
        r[v] = i
    return tuple(r)


# (d) alone: an independent inner automorphism on each side.
pairs_inn = [(a, b) for a in inn for b in inn]
# (c) and (d): one automorphism on both sides, up to an inner one per side, i.e. the pairs
# (phiA, phiB) with phiA^-1 phiB inner.
pairs_aut = [(a, b) for a in auts for b in auts if comp_phi(inverse_phi(a), b) in inn_set]


def burnside(pairs, swaps):
    """Number of orbits = the average number of fixed points over the group."""
    tot = Fraction(0)
    count = 0
    for a, b in pairs:
        for sa in (swaps if swaps else (0,)):
            for sb in (swaps if swaps else (0,)):
                FA = fR[a] ** 2 if sa == 0 else fR2[a]
                FB = fL[b] ** 2 if sb == 0 else fL2[b]
                tot += FA * FB
                count += 1
    return tot / count


o_ab = len(R) ** 2 * len(L) ** 2
o_abd = burnside(pairs_inn, None)
o_abcd = burnside(pairs_aut, None)
o_all = burnside(pairs_aut, (0, 1))
assert o_all.denominator == 1 and o_all <= 3e8     # the printed "at most about 3 x 10^8" holds

exponent = len(str(int(o_all))) - 1                  # 131,504,342 = 1.32 x 10^8
print(f"Z10 x S3: at most about 3 x 10^8 codes up to equivalence"
      f" (exact orbit count, ops (a)-(d) and the two entry swaps: {int(o_all):,} = {int(o_all) / 10 ** exponent:.2f}e{exponent})")
print(f"basis: |G| = {n}, |Aut| = {len(auts)}, |Inn| = {len(inn)}; {len(subsets):,} 3-element sets,"
      f" {len(R)} right and {len(L)} left translation classes")
print(f"basis: orbits under (a)+(b) {o_ab:,}; + (d) {int(o_abd):,}; + (c) {int(o_abcd):,};"
      f" + entry swaps {int(o_all):,}")
print(f"timed: wall {time.time() - T0:.0f} s")
