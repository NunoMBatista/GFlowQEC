# GFlowQEC

Code and data for *GFlowQEC: Generative Flow Networks that Learn to Sample Quantum LDPC Codes* by Nuno Batista, Gabriel Falcão and Penousal Machado (2026). GFlowQEC trains generative flow networks to propose quantum LDPC codes (two-block and lifted-product codes over finite groups) and certifies their minimum distances exactly. The offline variant learns from stored codes without calling the oracle and is then deployed frozen; the online variant pays the oracle as it trains, with local search and a replay buffer.

![GFlowQEC-offline](figures/pipeline-offline.png)

![GFlowQEC-online](figures/pipeline-online.png)

The tag v1.0 is the exact code and data behind the paper. Later commits may add to it.

## Install

Python 3.12. From the repository root:

```
pip install -r requirements.txt
```

Everything runs from the root as a module (`python -m ...`). There is no package to install.

`readouts.fig_main300` draws its labels with LaTeX (matplotlib `text.usetex`), so it needs `latex` on the path, for example from `apt install texlive-latex-extra cm-super dvipng` on Debian or Ubuntu. Nothing else needs LaTeX, and without it pytest skips that one test.

## Get the data

The data (about 1.4 GB, with ten seeds of the main n = 180 runs) is in the Hugging Face dataset [`NunoBatista/gflowqec-data`](https://huggingface.co/datasets/NunoBatista/gflowqec-data).

```
pip install huggingface_hub
python -m scripts.fetch_data
```

This fills `data/` and checks every file against `data/MANIFEST.md5`. `python -m scripts.fetch_data --check-only` repeats the check without downloading.

Stored file names keep a short prefix naming the run family that wrote them (for example `r174_` or `r177_`). The prefix is a name only. The docstring of `gflowqec/counts.py` says what each n = 180 family holds.

## Quick start

Sample codes from a trained GFlowQEC-offline policy at n = 180 and certify one distance exactly. This takes a few seconds on a laptop CPU once the data is in place.

```python
import torch
from gflowqec.certify import exact_sectors
from gflowqec.codes import build, generates_g, group_for, identity_index
from gflowqec.oracle import cond_vec, lam_of
from gflowqec.paths import DATA
from gflowqec.policy import TupleTransformer

# The n = 180 arena: lifted-product codes over G = Z_6 x S_3 (36 elements, n = 5|G|)
G = "C6xS3"
elements = group_for(G)[1]

# A trained GFlowQEC-offline policy (seed 0)
policy = TupleTransformer(len(elements))
policy.load_state_dict(torch.load(DATA / "models" / "r177_transformer_w18_b256_x20_s0.pt")["state"])

# Sample 16 codes for Z-biased noise (p_Z = 10 p_X); keep the first 5 whose elements generate G
torch.manual_seed(0)
cond = torch.tensor([cond_vec(lam_of(10, "Z"))] * 16)
tuples, _ = policy.sample(16, cond, identity_index(G))
picks = [t for t in tuples.tolist() if generates_g(G, t)][:5]

# A code is ten indices into `elements` (24 is the identity e; print(elements[i]) shows element i):
# A = [g1 + g2 + g3, e + g4 + g5], B = [h1 + h2 + h3, e + h4 + h5]
for t in picks:
    code = build(G, t)
    print(f"[[{len(code)}, {code.dimension}]]  g = {t[:5]}  h = {t[5:]}")
# [[180, 36]]  g = [24, 13, 3, 34, 10]  h = [24, 5, 10, 11, 8]   (then four more codes)

# Certify the first code's distance exactly: one SAT ladder per sector, d = min(d_X, d_Z)
dX, exact_X, dZ, exact_Z = exact_sectors(build(G, picks[0]))[:4]
print(f"d_X = {dX}, d_Z = {dZ}, exact: {exact_X and exact_Z}")   # d_X = 6, d_Z = 6, exact: True
```

## Reproduce the paper

```
python -m readouts.sum180      # prints Table 3
python -m pytest tests -q      # checks every readout against its expected output
```

Each readout prints one table, figure or section of the paper from the stored runs in `data/`. Readouts train nothing and call no oracle. Each docstring names the paper item it prints.

Table, figure and section numbers follow arXiv v1.

| Paper item | Readout |
|---|---|
| Table 1 (prior code searches) | none, literature only |
| Fig. 1 (Tanner graph of the [[4,2,2]] code), Fig. 2 (GFlowNet toy) | none, definitions only |
| Table 2 (answer keys and reference sample) and Sec. 4.3 text | `keys`, `stored_forms` (the certification threshold) |
| Fig. 3 (GFlowQEC-offline pipeline) | none, schematic (`figures/pipeline-offline.png`) |
| Sec. 5.3 text (stored runs, paid calls, stored forms, ten models) | `stored_forms`, `nets` (the ten models) |
| Sec. 5.4 text (legal tuples per canonical form) | `stored_forms` |
| Sec. 6 text (comparison methods) | `methods` (the screen's choice on a held-out tenth); its other numbers are settings fixed in code |
| Table 3 (results at 180 qubits): Certified top-scoring codes, Neighbourhoods and Seeds columns | `sum180` |
| Table 3: Inequivalent columns | `iso180` |
| Sec. 7 text (n = 180 results) | `text180`, `res_prefix` (per-seed counts), `hyb180` (screen ranking the policy's draws), `iso180` (inequivalent ratios), `cost180` (Cost paragraph) |
| Sec. 1, Sec. 9 and abstract (headline numbers) | `text180`, `background` (size of the n = 300 space) |
| Sec. 8 (discussion) | `sum180`, `res_prefix`, `cal36`, `cal72`, `sum300` |
| App. A.2 and App. A.4 text | `family_counts` |
| Table A.1 (oracle cost per paid call) | `oracle_costs` (its n = 36 and n = 300 cells come from `scripts.time_oracle`) |
| App. A.1, App. A.3 and App. B.2 | none, definitions only |
| App. B text (legal tuples, reward floor) | `stored_forms` |
| Table B.1 (networks) | `nets`; the screen's training length also in `settings` |
| App. B.1 text (stored forms, pilot, relational variant) | `stored_forms`, `rel180` |
| App. C text (budgets, y* range) | `protocol` |
| Table C.1 (comparison methods as run) | `methods` |
| Table C.2 and App. C.2 text (statistical procedures) | `stats_appendix` |
| Table D.1 (per-seed results at n = 180) and its notes | `res_prefix` |
| App. D.1 text | `res_prefix` (note b, five and a half to six times), `text180` (ladder runs), `more180` (sorting by k) |
| App. D.2 (proportional sampling at 36 and 72 qubits) | `cal36` (n = 36), `cal72` (n = 72) |
| Fig. E.1 (GFlowQEC-online pipeline) | none, schematic (`figures/pipeline-online.png`) |
| App. E.2 text (size of the space, beta = 64) | `background`, `nets` |
| App. E.3 text (y*) | `settings` |
| Table E.1 (comparisons at 300 qubits) | `sum300` |
| Fig. E.2 (comparisons at 300 qubits) | `fig_main300` |
| App. E.4 text (n = 300 results) | `text300`, `fig_main300` (the depth 31 sentence) |
| Table E.2 (further runs at n = 300) | `res300` |
| App. E.5 text (objective swap, k = 60, best codes) | `more300` |

## Regenerate the data

The scripts in `scripts/` rerun the experiments from scratch and write to `results/`, never to `data/`. Some take hundreds of CPU-hours; each docstring gives the commands, seeds and cost. Reruns agree closely across machines but not bit for bit.

## Layout

```
gflowqec/   library: codes, oracles, distance certifier, policies, comparison methods
readouts/   one script per paper table, figure or section
scripts/    data download and experiment reruns
data/       answer keys, stored runs and trained networks
tests/      checks for every readout and the data manifest
figures/    the two pipeline diagrams
```

## License and citation

Code is MIT (`LICENSE`), data is CC BY 4.0 (`LICENSE-DATA`), and third-party licenses are listed in `THIRD-PARTY.md`. If you use this work, please cite the paper (`CITATION.cff`).
