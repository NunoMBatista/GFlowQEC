"""The n = 36 answer key: every two-block code over Z_6 x Z_3, scored exactly.

Regenerates two files of data/keys/ (the n = 36 row of the answer-key table; the landscape is
also the exact target of the n = 36 sampler check):

  leg `landscape`: results/landscape_l6m3_w3.csv, one row per raw (A, B) pair of weight-3
      polynomials (816 x 816 = 665,856 rows) with exact n, k, d, merit k d^2 / n and the
      canonical key. Each distinct code (canonical key) pays for its distance once.
  leg `datasheet`: results/datasheet_l6m3_w3.csv, one row per distinct code with k > 0
      (704 rows): exact per-sector distances d_X, d_Z and membership of the exact Pareto
      front over (rate, d). Reads the landscape from data/keys/ (the landscape leg writes
      the byte-identical file).

The arena: A and B are weight-3 polynomials whose terms are any monomial x^i y^j, a strict
superset of Bravyi et al.'s BB codes (pure powers of x or y); see gflowqec/codes.py.

Run from the repo root:
    python -m scripts.n36_key landscape
    python -m scripts.n36_key datasheet

Compute (laptop, one core): landscape about 19 s; datasheet about 26 s.
Both outputs are deterministic and carry no timing column; their md5 sums are
landscape 684a3d75468867931582344b24864016 and datasheet 0160e6244b75d22dc531b390628471fe.
"""

import argparse
import ast
import csv
import time
from collections import Counter

from gflowqec.codes import build_code, enumerate_space, save_landscape, sector_distances
from gflowqec.paths import DATA, OUT

# The grid Z_6 x Z_3 (n = 2 * 6 * 3 = 36 qubits) and the polynomial weight.
L, M = 6, 3
WEIGHT = 3
LANDSCAPE_OUT = OUT / f"landscape_l{L}m{M}_w{WEIGHT}.csv"
ANSWER_KEY = DATA / "keys" / f"landscape_l{L}m{M}_w{WEIGHT}.csv"   # the datasheet's input
DATASHEET_OUT = OUT / f"datasheet_l{L}m{M}_w{WEIGHT}.csv"


def landscape_leg():
    """Enumerate the space, score every code, save the answer key and print its summary:
    how many distinct codes exist, how much of the space is junk (k = 0), the high-merit tail."""
    start = time.time()
    rows = enumerate_space(L, M, WEIGHT)
    elapsed = time.time() - start

    save_landscape(rows, LANDSCAPE_OUT)

    # --- How big is the space, and how much of it is the same code in disguise? ---
    distinct = {}  # canonical key -> one representative Row per distinct code
    for r in rows:
        distinct[r.key] = r
    print(f"pairs scored:   {len(rows)}  in {elapsed:.1f}s "
          f"({1000 * elapsed / len(rows):.2f} ms/pair incl. shift-duplicates)")
    print(f"distinct codes: {len(distinct)} (after collapsing monomial shifts)")

    # --- How sparse is the reward? (fraction of the space with no logical qubits) ---
    junk_pairs = 0
    for r in rows:
        if r.k == 0:
            junk_pairs += 1
    print(f"k = 0 junk:     {junk_pairs}/{len(rows)} pairs "
          f"({100 * junk_pairs / len(rows):.0f}% of the raw space)")

    # --- The high-merit tail: the codes a search method is supposed to find. ---
    # Count how many raw pairs map to each distinct code: that multiplicity sets the
    # chance that uniform random sampling hits it.
    multiplicity = Counter()
    for r in rows:
        multiplicity[r.key] += 1

    print("\ntop distinct codes by merit kd^2/n:")
    ranked = sorted(distinct.values(), key=lambda r: r.merit, reverse=True)
    for r in ranked[:10]:
        copies = multiplicity[r.key]
        chance = 100 * copies / len(rows)
        print(f"  [[{r.n},{r.k},{r.d}]]  merit {r.merit:.2f}  "
              f"{copies} raw pairs ({chance:.2f}% of space)  e.g. A={r.a} B={r.b}")

    print(f"\nanswer key saved to {LANDSCAPE_OUT}")


def load_answer_key():
    """One entry per distinct code from the landscape CSV: key -> merit."""
    distinct = {}
    with open(ANSWER_KEY) as f:
        for row in csv.DictReader(f):
            key = ast.literal_eval(row["key"])   # (l, m, a_canonical, b_canonical)
            if key not in distinct:
                distinct[key] = float(row["merit"])
    return distinct


def pareto_front(points):
    """Indices of (rate, d) points no other point beats on both axes at once.
    Ties are not domination: only strictly-better-somewhere counts."""
    front = []
    for i, (rate_i, d_i) in enumerate(points):
        dominated = False
        for rate_j, d_j in points:
            if rate_j >= rate_i and d_j >= d_i and (rate_j > rate_i or d_j > d_i):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def datasheet_leg():
    """Exact per-sector distances of every distinct non-junk code, then two summaries.

    The ZX-duality census: two-block codes over an abelian group satisfy d_X = d_Z (H_Z is
    H_X up to a qubit permutation), so the mismatch count must be zero. The exact Pareto
    front over (rate, d): the codes no other code beats on both axes at once; the merit
    k d^2 / n is one scalarisation of this front.
    """
    start = time.time()
    answer_key = load_answer_key()
    junk = sum(1 for merit in answer_key.values() if merit == 0.0)

    rows = []
    mismatches = 0
    for key, merit in sorted(answer_key.items()):
        if merit == 0.0:
            continue                    # k = 0: no logical qubits, no distances
        l, m, a, b = key
        code = build_code(l, m, a, b)
        n, k = len(code), code.dimension
        d_x, d_z = sector_distances(code)
        d = min(d_x, d_z)
        if d_x != d_z:
            mismatches += 1
        # Free cross-validation: the recomputed merit must match the answer key.
        assert abs(k * d * d / n - merit) < 1e-9, key
        rows.append({"key": repr(key), "n": n, "k": k, "rate": k / n,
                     "d_x": d_x, "d_z": d_z, "d": d, "merit": merit})

    # The on_front column: 1 for the codes on the exact (rate, d) Pareto front.
    front = set(pareto_front([(row["rate"], row["d"]) for row in rows]))
    for i, row in enumerate(rows):
        row["on_front"] = int(i in front)

    DATASHEET_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASHEET_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} distinct non-junk codes ({junk} junk skipped)")
    print(f"ZX-duality census: {mismatches} d_X != d_Z mismatches (expected 0)")
    profile = Counter((rows[i]["k"], rows[i]["d"]) for i in front)
    print("exact Pareto front over (rate, d), as [[n, k, d]] x count:")
    for (k, d), count in sorted(profile.items()):
        print(f"  [[{2 * L * M}, {k}, {d}]] x {count}")
    print(f"elapsed {time.time() - start:.1f}s; wrote {DATASHEET_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("leg", choices=["landscape", "datasheet"])
    args = parser.parse_args()
    if args.leg == "landscape":
        landscape_leg()
    else:
        datasheet_leg()
