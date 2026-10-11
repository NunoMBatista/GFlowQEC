"""App. A: the measured counts of its text, in App. A.2 (lifted-product family) and App. A.4 (search
spaces, oracle and distances).

A.2  Over Z_6 x S_3, lambda(1 + g4 + g5) is invertible for only 123 of the 595 pairs of distinct
     elements other than the identity (and likewise rho, the count for b_2).
A.4  A and B have full row rank for 383 of the 666 codes of the reference sample, and
     d_Z <= d_A and d_X <= d_B held on all 666 of them.

Reads data/keys (the reference sample) and data/n180/ref666_bounds.csv (the classical bounds d_A, d_B
of the 666 codes, made by scripts/ref666_bounds.py); no oracle call; about 7 s on a laptop. Run from the
repo root:
    python -m readouts.appa_codes
"""

import csv
import itertools
import time

import numpy as np
from ldpc import mod2

from gflowqec import certify, codes, oracle
from gflowqec.paths import DATA

T0 = time.time()
print("App. A.2 and App. A.4 text")

# =============================================================================================
# A.2. Z_6 x S_3 built by hand, independently of qldpc: an element is (i, p) with i in Z_6 and p a
# permutation of {0, 1, 2}. lam(g) is the 36 x 36 matrix of left multiplication v -> g v, rho(g) that
# of right multiplication v -> v g. An element a of the group algebra F2[G] is invertible exactly
# when its multiplication matrix has full rank over GF(2).
# =============================================================================================
perms = list(itertools.permutations(range(3)))


def pmul(p, q):  # (p q)(x) = p(q(x))
    return tuple(p[q[x]] for x in range(3))


G = [(i, p) for i in range(6) for p in perms]
idx = {g: t for t, g in enumerate(G)}


def mul(g, h):
    return ((g[0] + h[0]) % 6, pmul(g[1], h[1]))


e = (0, (0, 1, 2))
N = len(G)


def lam(g):
    M = np.zeros((N, N), dtype=np.uint8)
    for v in G:
        M[idx[mul(g, v)], idx[v]] = 1
    return M


def rho(g):
    M = np.zeros((N, N), dtype=np.uint8)
    for v in G:
        M[idx[mul(v, g)], idx[v]] = 1
    return M


def rank2(M):
    """Rank over GF(2) by Gauss-Jordan elimination."""
    M = M.copy() % 2
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
    return r


# every unordered pair {g4, g5} of distinct elements other than the identity: C(35, 2) = 595
nonid = [g for g in G if g != e]
cnt = 0
cntr = 0
tot = 0
for g4, g5 in itertools.combinations(nonid, 2):
    tot += 1
    L = (lam(e) + lam(g4) + lam(g5)) % 2
    R = (rho(e) + rho(g4) + rho(g5)) % 2
    cnt += rank2(L) == N
    cntr += rank2(R) == N
print(f"Z6 x S3: 1 + g4 + g5 invertible for {cnt} of the {tot} pairs (lambda), {cntr} (rho)")

# =============================================================================================
# A.4. For each code of the reference sample, the two classical matrices A = [L(a0) | L(a1)] and
# B = [L(b0) | L(b1)] (36 x 72 each), cut from the code's check matrices. Full row rank = rank 36.
# =============================================================================================
ref = oracle.reference_sample()
full = 0
fa = fb = 0
for r in ref:
    A, B = certify.lifted_bases(codes.build("C6xS3", np.array(r["exps"])), 36)
    ra, rb = mod2.rank(A) == 36, mod2.rank(B) == 36
    fa += ra
    fb += rb
    full += ra and rb
print(f"reference sample: A and B of full row rank for {full} of the {len(ref)} codes")

print(f"basis: A of full row rank {fa}, B of full row rank {fb}, both {full}")

# =============================================================================================
# A.4, the second claim. d_A and d_B are the minimum weights of ker A and ker B (the SAT bound
# ladder, scripts/ref666_bounds.py). The sample stores the per-sector distances under the keys dX
# and dZ; key dX is the paper's d_Z and key dZ the paper's d_X, so d_Z pairs with d_A and d_X with
# d_B. The distances are taken from the sample itself, the bounds from the file, joined by index.
# =============================================================================================
bounds = {int(b["index"]): b for b in csv.DictReader(open(DATA / "n180" / "ref666_bounds.csv"))}
held = both_equal = eq_a = eq_b = inexact = key_exact = 0
for r in ref:
    b = bounds[r["i"]]
    dZ, dX = r["dX"], r["dZ"]                     # the paper's names
    dA, dB = int(b["d_A"]), int(b["d_B"])
    held += dZ <= dA and dX <= dB
    eq_a += dZ == dA
    eq_b += dX == dB
    both_equal += dZ == dA and dX == dB
    inexact += not (b["exactA"] == "1" and b["exactB"] == "1")
    key_exact += r["exactX"] and r["exactZ"]
print(f"reference sample: d_Z <= d_A and d_X <= d_B held on {held} of the {len(ref)} codes")

print(f"basis: bounds file rows {len(bounds)}; d_Z = d_A on {eq_a}, d_X = d_B on {eq_b}, both {both_equal}")
print(f"basis: bound ladders not closed {inexact}; codes with both key sectors exact {key_exact}")
print(f"timed: wall {time.time() - T0:.0f} s")
