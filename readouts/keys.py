"""Table II and the measured numbers of the text of Sec. 4.3 (oracle, scores and references).

Prints, for each size, the measured cells of the table: what a proposal is graded against, how many
codes with k > 0 there are (as canonical forms and as inequivalent codes) and the best or published
codes. Then the numbers the Sec. 4.3 text quotes about the n = 72 key and the n = 180 reference sample.

Two codes are equivalent (Sec. 4.2) when relabelling the qubits and the checks turns one into the
other, X checks staying X checks. That holds exactly when their coloured Tanner graphs are isomorphic,
which is when their nauty certificates (codes.certificate) are equal. So the number of inequivalent
codes in a set is the number of distinct certificates in it.

Reads data/keys and the 42 stored n = 180 runs; no oracle call. It builds every counted code once
to compute its certificate (about 12,000 codes), about 12 s on a laptop. Run from the repo root:
    python -m readouts.keys
"""

import ast
import csv
import json
import time

import numpy as np

from gflowqec import codes, oracle
from gflowqec.paths import DATA

T0 = time.time()
KEYS = DATA / "keys"
N180 = DATA / "n180"


def read_csv(name):
    return list(csv.DictReader(open(KEYS / name)))


def cert_two_block(key):
    """Certificate of a two-block code from its key (l, m, a, b)."""
    l, m, a, b = key
    code = codes.build_code(l, m, a, b)
    return codes.certificate(np.asarray(code.matrix_x) % 2, np.asarray(code.matrix_z) % 2)


def cert_lifted(exps):
    """Certificate of an n = 180 lifted-product code from its ten group-element indices."""
    code = codes.build("C6xS3", np.array(exps))
    return codes.certificate(np.asarray(code.matrix_x) % 2, np.asarray(code.matrix_z) % 2)


def n_codes(c):
    """'one code' or '<c> codes', as the table prints a class count."""
    return "one code" if c == 1 else f"{c} codes"


print("Table II and Sec. 4 text")

# =============================================================================================
# n = 36: the complete key over Z_6 x Z_3.
# candidates_l6m3_w3.csv holds every canonical form of the space (k = 0 included);
# datasheet_l6m3_w3.csv holds the forms with k > 0, with exact n, k, d and merit k d^2 / n.
# =============================================================================================
cand36 = {ast.literal_eval(r["key"]): int(r["k"]) for r in read_csv("candidates_l6m3_w3.csv")}
ds36 = {ast.literal_eval(r["key"]): r for r in read_csv("datasheet_l6m3_w3.csv")}
assert set(ds36) == {key for key, k in cand36.items() if k > 0}     # the datasheet is exactly the k > 0 forms

cert36 = {key: cert_two_block(key) for key in cand36}
classes_all36 = len(set(cert36.values()))
classes_k36 = len({cert36[key] for key in ds36})

# the forms of highest merit: one (n, k, d) triple, how many forms, how many inequivalent codes
top_merit36 = max(float(r["merit"]) for r in ds36.values())
tops36 = [key for key, r in ds36.items() if abs(float(r["merit"]) - top_merit36) < 1e-9]
triples36 = {(int(ds36[key]["n"]), int(ds36[key]["k"]), int(ds36[key]["d"])) for key in tops36}
assert len(triples36) == 1
(n, k, d), = triples36
print(f"n = 36, two-block, Z_6 x Z_3 | complete key: {len(cand36):,} forms, {classes_all36} inequivalent codes"
      f" | {len(ds36):,} / {classes_k36}"
      f" | [[{n},{k},{d}]], merit {top_merit36:.1f}: {len(tops36)} forms, {n_codes(len({cert36[t] for t in tops36}))}")

# =============================================================================================
# n = 72: the complete key over Z_6 x Z_6, merged from three passes.
# certified_l6m6_w3.csv is the first pass; escalated_l6m6_w3.csv re-ran its open rows with a wider
# sweep; closed_l6m6_w3.csv closed the rest with the SAT ladder. The latest pass wins per form, and
# every merged row must be exact: that is what "complete key" means. d is then d_lower (= d_upper).
# =============================================================================================
cand72 = read_csv("candidates_l6m6_w3.csv")
final72 = {}
for name in ("certified_l6m6_w3.csv", "escalated_l6m6_w3.csv", "closed_l6m6_w3.csv"):
    for r in read_csv(name):
        final72[ast.literal_eval(r["key"])] = r
assert all(r["exact"] == "1" for r in final72.values())
assert len(final72) == sum(1 for r in cand72 if int(r["k"]) > 0)

kd72 = {key: (int(r["k"]), int(r["d_lower"])) for key, r in final72.items()}
cert72 = {key: cert_two_block(key) for key in final72}


def best72(value):
    """The forms at the largest value of value(k, d): their single [[72,k,d]], the value, count, classes."""
    top = max(value(k, d) for k, d in kd72.values())
    keys = [key for key, (k, d) in kd72.items() if abs(value(k, d) - top) < 1e-9]
    triples = {kd72[key] for key in keys}
    assert len(triples) == 1
    (k, d), = triples
    return k, d, top, keys


k, d, top_merit72, tops72 = best72(lambda k, d: k * d * d / 72)
k8, d8, _, maxd72 = best72(lambda k, d: d)
print(f"n = 72, two-block, Z_6 x Z_6 | complete key: {len(cand72):,} forms"
      f" | {len(final72):,} / {len(set(cert72.values()))}"
      f" | [[72,{k},{d}]], merit {top_merit72:.1f}: {len(tops72)} forms, {n_codes(len({cert72[t] for t in tops72}))}"
      f"; [[72,{k8},{d8}]], largest d: {len(maxd72):,} forms, {n_codes(len({cert72[t] for t in maxd72}))}")

# =============================================================================================
# n = 180: the reference sample, the 666 of 1,000 uniform draws over Z_6 x S_3 that satisfy the
# second restriction (both sides generate the group), with exact d_X and d_Z.
# =============================================================================================
draws = json.load(open(KEYS / "r169_headroom_C6xS3_n1000.json"))["codes"]
generating = [r for r in draws if codes.generates_g("C6xS3", np.array(r["exps"]))]
ref = oracle.reference_sample()
assert len(generating) == len(ref)
assert all(r["exactX"] and r["exactZ"] for r in ref)
assert {r["i"] for r in generating} == {r["i"] for r in ref}

# k is stored with the draws, not in the reference rows: join by draw index "i"
k_of = {r["i"]: r["k"] for r in draws}
ref_k = [r for r in ref if k_of[r["i"]] > 0]
forms180 = len({codes.orbit_key("C6xS3", r["exps"]) for r in ref_k})
classes180 = len({cert_lifted(r["exps"]) for r in ref_k})

# The 42 stored n = 180 search runs (the certified oracle, every arm and seed). A row is certified
# when both sectors closed exactly; the largest min(d_X, d_Z) among them is the table's artifact.
run_files = sorted(N180.glob("r174_*_cert_s[0-9].json"))
assert len(run_files) == 42
runs = [json.load(open(f)) for f in run_files]
assert all(d["complete"] for d in runs)

min_d = {}           # min(dX, dZ) -> set of (dX, dZ) pairs seen at it
rows_at = {}         # min(dX, dZ) -> number of certified rows at it
for d in runs:
    for dx, dz, ex, ez in zip(d["dX"], d["dZ"], d["exactX"], d["exactZ"]):
        if dx is not None and ex and ez:
            min_d.setdefault(min(dx, dz), set()).add((dx, dz))
            rows_at[min(dx, dz)] = rows_at.get(min(dx, dz), 0) + 1
best = max(min_d)
assert min_d[best] == {(best, best)}     # the code's (dX, dZ) are the paper's (d_Z, d_X); equal here
print(f"n = 180, lifted product, Z_6 x S_3 | reference sample: {len(ref)} uniform draws, exact distances"
      f" | {forms180} / {classes180} | certified, d_X = d_Z = {best}, from search runs")

# =============================================================================================
# Sec. 4 text
# =============================================================================================

# Sec. 4.3, merit. Bravyi et al.'s [[72,12,6]] (their Table 3): A = x^3 + y + y^2, B = y^3 + x + x^2, written as
# exponent pairs (i, j) = x^i y^j. Its canonical form must be one of the top-merit forms, which are one code.
bravyi = codes.canonical_key(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
assert bravyi in tops72 and len({cert72[t] for t in tops72}) == 1
print(f"Sec. 4.3: [[72,{kd72[bravyi][0]},{kd72[bravyi][1]}]] is equivalent to the code of Bravyi et al.")

print(f"Sec. 4.3: the {len(generating)} of {len(draws):,} uniform draws that satisfy the second restriction, with exact distances")

# Sec. 4.3, failure exponent. The threshold at each bias is the reference sample's top score (oracle.key_top), pooled
# with the mirror images, so it is the same at +1 and -1 by construction. lam passed to bias_score and
# key_top is the paper's gamma. The counts of codes reaching it are per sign, not mirror-pooled.
top_plus, top_minus, top_zero = oracle.key_top(1.0)[0], oracle.key_top(-1.0)[0], oracle.key_top(0.0)[0]
assert top_plus == top_minus
print(f"Sec. 4.3: highest score at the run's bias: {top_plus:.1f} at gamma = +-1")


def reaching(lam):
    top = oracle.key_top(lam)[0]
    return sum(1 for r in ref if oracle.bias_score(r["dX"], r["dZ"], lam) >= top - 1e-9)


print(f"Sec. 4.3: at gamma = 0 the top value is reached by {reaching(0.0)} of the {len(ref)} codes")
print(f"Sec. 4.3: at gamma = +1, {reaching(1.0)} of {len(ref)} reach the threshold")
print(f"Sec. 4.3: at gamma = -1, {reaching(-1.0)} of {len(ref)} reach the threshold")

# Sec. 4.3, failure exponent. Among certified rows of the 42 runs (cert stamped), the share whose exact ladder stopped at
# a lower bound in either sector. A code paid in several runs counts once per run.
n_cert = n_cens = 0
for d in runs:
    for i, c in enumerate(d["cert"]):
        if c is None:
            continue
        n_cert += 1
        n_cens += not (d["exactX"][i] and d["exactZ"][i])
print(f"Sec. 4.3: an exact ladder stops early for {100 * n_cens / n_cert:.1f} per cent of certified codes")

print(f"basis: {100 * n_cens / n_cert:.1f} per cent = {n_cens} of {n_cert:,} certified rows (repeats across runs counted)"
      f" of the {len(runs)} stored n = 180 runs")
print(f"basis: largest certified min(d_X, d_Z) in the {len(runs)} stored runs {best}, all {rows_at[best]} such rows"
      f" at ({best}, {best}); reference sample largest {max(min(r['dX'], r['dZ']) for r in ref)}")
print(f"timed: wall {time.time() - T0:.0f} s")
