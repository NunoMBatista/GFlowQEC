# GFlowQEC

Code and data for the pre-print *GFlowQEC: Generative Flow Networks that Learn to Sample Quantum LDPC Codes*.

The `readouts/` print the paper's measured numbers, its tables and its main figure at n = 300 (Fig. 7) from the stored runs in `data/`. They call no oracle and train nothing. The `scripts/` regenerate `data/` from scratch, at the compute costs listed below. The library they share is `gflowqec/`.

## Get the data

The data lives in two places. This repository holds the code, the tests and the small data files: `data/keys` (all but one file), `data/n72`, `data/timing` and `data/MANIFEST.md5`. The large files live in the Hugging Face dataset [`NunoBatista/gflowqec-data`](https://huggingface.co/datasets/NunoBatista/gflowqec-data): `data/n180`, `data/n300`, `data/models` and `data/keys/landscape_l6m3_w3.csv`. They take about 826 MB. The dataset uses the same paths, so the download lands in the repo root and restores `data/` exactly. Run it once before the quick start.

```
pip install huggingface_hub     # for this step only, not in requirements.txt
python -m scripts.fetch_data    # downloads data/, then checks every file against data/MANIFEST.md5
```

`python -m scripts.fetch_data --check-only` checks the manifest without downloading. The data is licensed under CC BY 4.0 (`LICENSE-DATA`). The code release is archived on Zenodo. Its DOI will be added here at release.

## Quick start

```
pip install -r requirements.txt
python -m readouts.sum180          # Table III, from the repo root
python -m pytest tests -q          # every readout against tests/expected/, and data/ against its manifest
```

Run everything from the repo root as a module (`python -m ...`). The pins in `requirements.txt` are the laptop environment the paper's numbers were read in (Python 3.12.3, the CPU build of torch). pytest is not pinned. Without it, `python -m tests.test_readouts` and `python -m tests.test_data` run the same checks.

A readout's output is compared line by line with `tests/expected/<name>.txt`. Lines starting with `timed:` or `basis:` are diagnostics and are not compared. Readouts write nothing, except `fig_main300`, which writes `results/fig-main300.pdf`.

## Readouts

Paper numbering is that of the compiled pre-print. Wall times are one run each on the author's laptop. Run in sequence, they add up to about 14 minutes.

| Paper item | Command | Wall time |
|---|---|---|
| Sec. 3 and Sec. 1: the size of the n = 300 space | `python -m readouts.background` | 6 s |
| Table II (answer keys) and the Sec. 4 text | `python -m readouts.keys` | 12 s |
| Sec. 6 text and App. B.1 (stored forms, offline policy) | `python -m readouts.stored_forms` | 31 s |
| Table III (comparisons at n = 180) | `python -m readouts.sum180` | 37 s |
| Table IV (comparisons at n = 300) | `python -m readouts.sum300` | 76 s |
| Sec. 8 text (y* and the screen's training length) | `python -m readouts.sec8` | 4 s |
| Sec. 9.1 text (n = 180) | `python -m readouts.text180` | 282 s |
| Sec. 9.1 text (the screen ranking the policy's draws) | `python -m readouts.hyb180` | 18 s |
| Sec. 9.2 text (n = 300) | `python -m readouts.text300` | 95 s |
| Fig. 7 (n = 300), written to `results/` | `python -m readouts.fig_main300` | 82 s |
| Sec. 9.3 text (proportional sampling at n = 72) | `python -m readouts.cal72` | 2 s |
| App. A text (counts on the lifted-product family) | `python -m readouts.appa_codes` | 7 s |
| Table V (cost of one paid call) | `python -m readouts.oracle_costs` | 1 s |
| Table VI (networks) | `python -m readouts.nets` | 4 s |
| App. B.1 text (the relational variant, not used) | `python -m readouts.rel180` | 17 s |
| Table VII (comparison methods as run) | `python -m readouts.methods` | 12 s |
| Table VIII (statistical procedures) and the text above it | `python -m readouts.stats_appendix` | 34 s |
| App. C text (budgets, y*) | `python -m readouts.appe_protocol` | 3 s |
| Table IX (first 1,000 paid calls at n = 180) | `python -m readouts.res_prefix` | 38 s |
| App. D.1 text after Table IX | `python -m readouts.appg180` | 66 s |
| Table X (the other n = 300 runs) | `python -m readouts.res300` | 6 s |
| App. D.2 text (n = 300) | `python -m readouts.appg300` | 5 s |

Sec. 9.3 tests proportional sampling at n = 36 and n = 72. At n = 72, `readouts.cal72` rebuilds the exact target from the answer key and grades the stored results of the runs in `data/n72/`. The trained n = 72 networks are not shipped. The n = 36 numbers have no readout. They are printed by `scripts/n36_calibration.py`, which makes them: `tail --target-only` reads the answer key in seconds, while the sampler legs retrain (see below). To grade the shipped samplers instead of retraining, copy `data/models/gfn_calib_l6m3_s{seed}.pt` into `results/` first, and the `tail` leg reloads them.

## Regenerating the data

Every script writes to `results/`, never to `data/`, under the name of the shipped file it regenerates. Each docstring gives the exact commands of the paper's runs, the random streams and the cost. The costs below are copied from the docstrings.

| Script | Regenerates | Compute cost |
|---|---|---|
| `key_forms` | `keys/candidates_l6m3_w3.csv`, `candidates_l6m6_w3.csv` (every canonical form, with k) | laptop, one core: about 4 s at (6, 3), about 130 s at (6, 6) |
| `n36_key` | `keys/landscape_l6m3_w3.csv`, `datasheet_l6m3_w3.csv` (the n = 36 key) | laptop, one core: about 19 s and 26 s |
| `n72_key` | `keys/certified_`, `escalated_`, `closed_l6m6_w3.csv` (the n = 72 key) | 31.30 CPU-hours: base 6.28 (cluster job), escalation 24.62 (16-task cluster array), SAT close 0.40 (359 s wall on 4 laptop cores) |
| `n36_calibration` | `keys/gfn_calibration_l6m3.csv`, `gfn_mass_tail_l6m3_s*.json`, `models/gfn_calib_l6m3_s*.pt` | laptop: about 642 s per seed of training; `--target-only` seconds |
| `n180_reference` | `keys/r169_*`, `keys/r174_bias_*` (the n = 180 reference sample) | about 4.0 CPU-hours on one server CPU |
| `ref666_bounds` | `n180/ref666_bounds.csv` | laptop, one core: about 3 minutes |
| `n180_search` | `n180/r174_*`, `r200_*`, `r209_*` search runs and the learners' networks | about 3 to 44 CPU-hours per run; 320.4 CPU-hours for the 42 stored runs |
| `n180_offline` | GFlowQEC-offline and offline soft Q: `models/r177_transformer_*`, `n180/r177_*` | training 34.4 to 34.6 CPU-hours per seed (soft Q: 3,728 to 3,766 s on one GPU); paying one table row 26 to 145 CPU-hours |
| `n180_screens` | the two supervised screens: `models/r177_mscreen_*`, `n180/screen180match_*`, `n180/r174_proxy_*` | matched screen: 2,792 to 2,952 s of GPU per configuration, 2,366 to 2,443 s per seed; earlier screen: see docstring |
| `n300_search` | `n300/mitten_lersearch_*` (one run, 40,000 paid calls) | about 14 to 28 CPU-hours per run in the oracle; the grid of 20 cells x 6 seeds about 2,000 to 3,500 CPU-hours |
| `n300_rescore` | `n300/g1_repay_visited_*`, `g1_repay_elites_*` (re-estimates on fresh shots) | about 0.4 CPU-hours per visited run, about 46 over the 120 runs; an elites run about a quarter of that |
| `time_oracle` | `timing/oracle_times.json` (Table V at n = 36 and n = 300) | about two minutes |

Run a script as `python -m scripts.<name> ...`. A few files ship as data only, with no script here that writes them: 15 of the 42 stored n = 180 runs (named at the end of the commands in `scripts/n180_search.py`), the 40 files of uniform draws at n = 300 (`n300/mitten_ler_data_*`), `n300/bo_rescore_cache.json`, and the pilot perceptron `models/r177_mlp_w18_b256_s0.pt`. The files read by `readouts.hyb180`, `readouts.rel180` and `readouts.cal72` are data only too: `n180/hyb_r177_*` and `n180/stf_r177_*`, `n180/rel_r177_*`, and all of `n72/`. The relational variant's code and networks are not shipped.

## The codes

The paper uses two families, both specified by a few elements of a finite group (`gflowqec/codes.py`).

**Two-block codes, n = 36 and n = 72** (the exact answer keys). The group is the abelian Z_l x Z_m, with (l, m) = (6, 3) and (6, 6). A and B are polynomials of three terms each, and every monomial x^i y^j is allowed as a term. The code is H_X = (A | B), H_Z = (B^T | A^T), so every check has weight 6 and every choice is a valid CSS code. In this repository "bivariate bicycle (BB) code" means Bravyi et al.'s definition (Nature 2024, Eq. 1), whose terms are pure powers of x or of y. The two-block arena is a strict superset of those codes. Under Eberhardt and Steffan's broader definition (arXiv:2407.03973, Sec. 4.1: any pair over F_2[Z_l x Z_m]), every code of the arena is a BB code.

**Lifted-product codes, n = 180 and n = 300** (the searches). The groups are non-abelian: G = Z_6 x S_3 (order 36, n = 180) and Z_10 x S_3 (order 60, n = 300, the group of the published [[300,60,14]]). The base matrices are A = [g1 + g2 + g3, e + g4 + g5] and B = [h1 + h2 + h3, e + h4 + h5], combined as LP(A, B), the construction of arXiv:2607.28795, Definition 4. Then n = 5|G| and every check has weight 9. A code is the tuple (g1, ..., g5, h1, ..., h5), and only tuples whose elements generate all of G are legal.

Two codes are counted as the same code when their coloured Tanner graphs are isomorphic, decided exactly by nauty's canonical labelling.

What one paid call scores (`gflowqec/oracle.py`): at n = 36, the exact n, k, d and the merit kd^2/n; at n = 180, the leading-order failure exponent under biased noise, min(c_X d_X, c_Z d_Z), from exact per-sector distances; at n = 300, the per-logical-qubit logical error rate under depolarizing code-capacity noise (p = 0.07), estimated with BP-OSD at 1,000 shots. Every n = 300 count in the paper is read from re-estimates on 4,000 independent shots (`scripts/n300_rescore.py`), never from a run's own scores.

## Distances and certification

No distance is reported as a value unless it is exactly certified. When the certifier cannot close a distance, it returns an honest interval or a lower bound, flagged as such, and never a value it has not proved (`gflowqec/certify.py`).

At n = 36 and n = 72 the certifier squeezes two bounds. An exhaustive, bit-packed sweep of every vector up to weight w proves d > w, or finds a logical and gives d exactly. qldpc's decoder-based randomized search (the algorithm of Bravyi et al., arXiv:2308.07915) constructs an actual logical operator, so d is at most its weight. When the two meet, d is exact. When they do not, the escalation raises the sweep cap, never the decoder trials, and a SAT decision ladder closed the n = 72 forms the sweep left open: all 9,280 n = 72 distances are exact.

At n = 180 the per-sector SAT ladders ask "is there a logical of weight at most w?" for w = 1, 2, and so on. Every "no" is a proof, so a ladder that runs out of budget leaves a lower bound flagged `exact = False`. The certifying oracle pays cheap classical bounds first and runs the exact ladders on every code that could reach the top score, so the set of top-scoring codes is exact. At n = 300 the score is an estimated error rate, not a distance.

The upper-bound search draws unseeded randomness inside qldpc. The upper bounds of interval rows are therefore not bit-reproducible, and a re-run of the n = 72 base pass may leave a slightly different set of forms open. An exact distance is a proof and does not depend on it.

## What this code does not claim

- No discovery superiority over Bayesian optimisation, reinforcement learning or LLM-driven evolution.
- Proportional sampling is measured only at n = 36 and n = 72, for an earlier, simpler sampler, where the whole space is enumerated and the exact target is known. That sampler under-samples the best codes: at n = 72 it gives the highest score value about half of its target mass. The policies of the n = 180 and n = 300 searches were not tested this way.
- The n = 180 and n = 300 results are search results: certified top-scoring codes at n = 180, good proposals graded on re-estimated error rates at n = 300. They are not calibration results.
- At n = 180 GFlowQEC-offline beat the supervised screen at ten candidates per paid call but not at a hundred, where the screen did best ranking the policy's own draws.
- Soft Q-learning trained on the same stored scores did about as well as GFlowQEC-offline, so the claims concern the trained proposer, not its objective.
- At n = 300 the gain is in how many good codes a budget yields, not in the best code found.

## Reproducibility

Before release, the ten scripts other than `time_oracle` and `ref666_bounds` were checked against the research code they came from, at the same seed: the n = 36 key and the form lists match the shipped files by md5, and the others passed determinism checks. Four things to know when re-running:

- **Across machines, results are near-exact, not bitwise.** Between two machines, only 1 of 200 trajectory-balance losses was bit-equal (largest difference 0.016), 940 of 1,000 offline log-probabilities were bit-equal, and screen predictions differed by at most 1.2e-5, enough to swap near-ties in a ranking (26 adjacent swaps against the archived one). A retrained network matches a shipped checkpoint tensor for tensor only on the same torch build. Re-estimates at n = 300 are a pure function of the run seed and the code, and reproduce exactly on the same ldpc build.
- **The certify leg of `scripts/n180_offline.py` joins its screen file by row index.** Never resume a shipped certified file against a screen file regenerated on another machine.
- **qldpc's distance upper bound is unseeded** (see above).
- **Where a leg reads its predecessor.** The `escalate` and `close` legs of `scripts/n72_key.py` read the archived pass before them from `data/keys`, not from `results/`. The later legs of `scripts/n180_reference.py` and `scripts/n180_screens.py` read the earlier legs from `results/`; copy the shipped files there to start from them.

Two known exceptions. The 0.089 total variation of Sec. 9.3 came from a run that saved no checkpoint. Three re-runs of its recipe, the shipped samplers, read 0.092 to 0.100, as the paper reports. `data/keys/gfn_mass_tail_l6m3_s0.json` reproduces by value but not by md5 (a missing value is written as NaN in one and null in the other).

## Data layout

`data/` holds 831 MB in six folders. `data/MANIFEST.md5` lists the md5 of every other file under `data/` (696 entries), and `tests/test_data.py` checks both directions. `data/n180`, `data/n300`, `data/models` and `data/keys/landscape_l6m3_w3.csv` come from the Hugging Face dataset (see Get the data).

```
data/keys/     15 files   answer keys: n = 36 (every raw pair; the 704 forms with k > 0), n = 72 (9,280 forms),
                          the canonical-form lists, the n = 180 reference sample (666 of 1,000 uniform draws),
                          and the n = 36 calibration outputs
data/n72/       9 files   the n = 72 runs of Sec. 9.3: three of 360,000 steps, six of 36,000 with and without prioritised replay
data/n180/    242 files   search runs and the earlier screen's runs (r174_*, r200_*, r209_*), GFlowQEC-offline's and the matched screen's
                          samples, screen and certify files (r177_*), the screen ranking the policy's draws (hyb_r177_*),
                          the policy's top-up at mu = 100 (stf_r177_*), the relational variant's runs (rel_r177_*),
                          the screen selection files, ref666_bounds.csv
data/n300/    348 files   151 search runs (mitten_lersearch_*), their re-estimates (120 g1_repay_visited_*,
                          36 g1_repay_elites_*), 40 files of uniform draws, bo_rescore_cache.json
data/models/   81 files   trained networks: the n = 36 samplers, the n = 180 policies, screens and learning rivals,
                          the n = 300 learning arms
data/timing/    1 file    oracle_times.json, the per-call times of Table V at n = 36 and n = 300
```

Library paths come from `gflowqec/paths.py`: inputs from `data/`, outputs to `results/`.

## License and citation

Code: MIT (`LICENSE`). Data under `data/`: CC BY 4.0 (`LICENSE-DATA`). To cite, use `CITATION.cff`. Third-party packages and their licenses are listed in `THIRD-PARTY.md`; pynauty, used only for the Tanner-graph equivalence counts, is GPL-3.0-or-later and optional: the library imports it inside the two functions that need it.
