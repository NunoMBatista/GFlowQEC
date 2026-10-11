"""Every canonical form of a two-block space over Z_L x Z_M, with k, and no distance.

Regenerates data/keys/candidates_l6m3_w3.csv (2,304 forms) and data/keys/candidates_l6m6_w3.csv
(40,401 forms): the "complete key" columns of Table II (Sec. 4.3), and at (6, 6) the list of
forms the n = 72 key certifies (scripts/n72_key.py).

A form is a code up to independent translation of A and B (codes.canonical_key, the
translation-gauge key). That symmetry factorises per polynomial, so the raw pairs never need
walking: the distinct codes are all pairs of canonical polynomials, and a pair's multiplicity
(how many raw (A, B) pairs land on it) is orbit_size(A) * orbit_size(B). Each form costs one
GF(2) rank for k; no distance is computed.

Output: results/candidates_l{L}m{M}_w3.csv, one row per form (key, k, n_pairs).

Run from the repo root:
    python -m scripts.key_forms 6 3
    python -m scripts.key_forms 6 6

Compute (laptop, one core): (6, 3) about 4 s; (6, 6) about 130 s.
Deterministic; md5 sums candidates_l6m3_w3.csv 3126dce4e6d36c863b188c9a821aa4fa and
candidates_l6m6_w3.csv cbe64e5b8f66378c801f646fac9de1f3.
"""

import argparse
import csv
import time
from collections import Counter

from gflowqec.codes import _canon_poly, all_polynomials, build_code
from gflowqec.paths import OUT

WEIGHT = 3   # three monomials per polynomial


def main(L, M):
    start = time.time()
    out = OUT / f"candidates_l{L}m{M}_w{WEIGHT}.csv"

    # Orbit table: canonical polynomial -> how many raw polynomials shift onto it.
    polynomials = all_polynomials(L, M, WEIGHT)
    orbit = Counter()
    for p in polynomials:
        orbit[_canon_poly(L, M, p)] += 1
    canon_polys = sorted(orbit)
    print(f"polynomials: {len(polynomials)} raw -> {len(canon_polys)} canonical")

    # Distinct codes = pairs of canonical polynomials; k costs one rank each.
    rows = []
    total_pairs = 0
    junk_pairs = 0
    k_counts = Counter()
    for ca in canon_polys:
        for cb in canon_polys:
            k = build_code(L, M, ca, cb).dimension
            mult = orbit[ca] * orbit[cb]
            rows.append(((L, M, ca, cb), k, mult))
            total_pairs += mult
            k_counts[k] += 1
            if k == 0:
                junk_pairs += mult
    elapsed = time.time() - start

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "k", "n_pairs"])
        for key, k, mult in rows:
            writer.writerow([repr(key), k, mult])

    junk_codes = k_counts[0]
    print(f"raw pairs:      {total_pairs}")
    print(f"distinct codes: {len(rows)}")
    print(f"k = 0 junk:     {junk_codes}/{len(rows)} codes "
          f"({100 * junk_pairs / total_pairs:.0f}% of raw pairs)")
    print("k spectrum (distinct codes): "
          + "  ".join(f"k={k}: {c}" for k, c in sorted(k_counts.items())))
    print(f"elapsed {elapsed:.1f}s; wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("L", type=int, help="order of the first cyclic factor (x)")
    parser.add_argument("M", type=int, help="order of the second cyclic factor (y)")
    args = parser.parse_args()
    main(args.L, args.M)
