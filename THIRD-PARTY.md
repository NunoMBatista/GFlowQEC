# Third-party software and credits

GFlowQEC's own code is MIT (see `LICENSE`) and its data is CC BY 4.0 (see `LICENSE-DATA`). This repository ships no third-party code; the packages below are installed separately from `requirements.txt`. Each license is as stated in the installed package's metadata (the `License` or `License-Expression` field, the license classifier, or the header of the license file in its `dist-info`), read on 2026-10-11. A license that could not be read from the installed files is marked UNCONFIRMED.

## Dependencies

| Package (pip name) | Version | Import name | License as stated in installed metadata | Used for |
|---|---|---|---|---|
| numpy | 2.4.6 | `numpy` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 (`License-Expression`; BSD-3-Clause is NumPy's own, the rest cover bundled components) | arrays everywhere |
| scipy | 1.18.0 | `scipy` | BSD License (classifier); `LICENSE.txt` is the 3-clause BSD text, "Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers." | statistics, the GP's linear algebra and optimiser |
| torch | 2.12.1 (CPU wheel `2.12.1+cpu` on the laptop) | `torch` | BSD-3-Clause (`License`; a `NOTICE` file lists bundled third-party software) | policies and rival networks |
| matplotlib | 3.11.0 | `matplotlib` | Matplotlib License, "License agreement for matplotlib versions 1.3.0 and later" (classifier: Python Software Foundation License) | the figure readout |
| ldpc | 2.4.1 | `ldpc` | MIT License, Copyright (c) 2024 Joschka Roffe | GF(2) linear algebra (`ldpc.mod2`), BP+OSD decoding |
| qldpc | 0.3.0 | `qldpc` | Apache-2.0 (`License-Expression`; no license file in the installed `dist-info`) | code construction (groups, lifted products), the decoder-based distance upper bound |
| galois | 0.4.11 | not imported directly | MIT, Copyright (c) 2020-2025 Matt Hostetter | finite-field arrays underneath qldpc |
| python-sat | 1.9.dev13 | `pysat` | MIT, Copyright (c) 2018 Alexey Ignatiev, Joao Marques-Silva, Antonio Morgado. The licenses of the SAT solvers it bundles (the code uses CaDiCaL 1.5.3 through `Cadical153`) are UNCONFIRMED: no solver license file is installed. | the SAT distance ladder (cardinality encoding and solver) |
| pynauty | 2.8.8.1 | `pynauty` | GNU General Public License v3 (`License`); the classifier and `COPYING` say version 3 or any later version (GPL-3.0-or-later), Copyright (c) 2015-2022 Peter Dobsan. The bundled nauty library is "covered by its own licensing terms" per pynauty's README; nauty's license is UNCONFIRMED: its `COPYRIGHT` file is not installed. | Tanner-graph canonical forms for the class counts (`gflowqec/codes.py`) |

**pynauty is the one copyleft dependency, and it is optional.** This repository does not ship it, and `gflowqec/codes.py` imports it only inside `tanner()` and `certificate()`, the two functions behind the Tanner-class counts. Everything else in the library imports and runs without pynauty installed. The code stays under MIT.

## Algorithms from the literature

The distance certifier, `gflowqec/certify.py`, credits one published algorithm. Its upper bound is "qldpc's decoder-based randomized search (the algorithm from Bravyi et al.'s BB paper, arXiv:2308.07915)", called as `get_distance_bound_with_decoder`; the implementation is qldpc's, not this repository's. The certifier's other two instruments (the bit-packed exhaustive lower-bound sweep and the SAT decision ladder) cite no reference in the code.

Other modules credit the following sources in their docstrings and comments, quoted as written there.

| Where | What the code says |
|---|---|
| `gflowqec/codes.py:6-9` | the two-block arena is "a strict superset of Bravyi et al.'s bivariate bicycle (BB) codes (Nature 2024, Eq. 1)"; "Under Eberhardt and Steffan's definition (arXiv:2407.03973, Sec. 4.1 ...) every code of the arena is a BB code" |
| `gflowqec/codes.py:12-13` | the lifted-product family at n = 180 and n = 300 is "the construction of arXiv:2607.28795, Definition 4" |
| `gflowqec/rivals.py:12, 249, 254` | the soft Q-learning and Munchausen DQN rival, "which targets the same distribution as a GFlowNet (Tiapkin et al. 2024)", with the two settings "following Tiapkin et al.'s baselines" |
| `gflowqec/policy.py:535` | "GFlowQEC-online: a GFlowNet inside a local search (after Kim et al., ICLR 2024)" |
| `gflowqec/rivals.py:766-767` | the Bayesian-optimisation baseline names "arXiv:2601.18562" as "the incumbent" and states it is "NOT a reimplementation of that paper's graph-network surrogate" |
| `gflowqec/stats.py` | the Katz log interval for a ratio of rates and the Wilson score interval, named without a reference |
