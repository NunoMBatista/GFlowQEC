# GFlowQEC

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/) [![PyTorch 2.12](https://img.shields.io/badge/PyTorch-2.12-blue)](https://pytorch.org/) [![Code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE) [![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)](LICENSE-DATA)

Code and data for the pre-print *GFlowQEC: Generative Flow Networks that Learn to Sample Quantum LDPC Codes* (Nuno Batista, Gabriel Falcão, Penousal Machado, 2026).

GFlowQEC trains generative flow networks to propose quantum LDPC codes (two-block and lifted-product codes over finite groups) and certifies their minimum distances exactly.

**GFlowQEC-offline** learns from stored codes without calling the oracle, then is deployed frozen:

![GFlowQEC-offline](figures/pipeline-offline.png)

**GFlowQEC-online** pays the oracle as it trains, with local search and a replay buffer:

![GFlowQEC-online](figures/pipeline-online.png)

This repository will grow as the work continues. Release v1.0 is the exact code and data behind the pre-print, and that tag is what the paper cites.

## Install

```
pip install -r requirements.txt
```

Python 3.12. Run everything from the repo root as a module (`python -m ...`).

## Get the data

The large data files (about 830 MB) live in the Hugging Face dataset [`NunoBatista/gflowqec-data`](https://huggingface.co/datasets/NunoBatista/gflowqec-data). Download them once:

```
pip install huggingface_hub
python -m scripts.fetch_data
```

This fills `data/` and checks every file against `data/MANIFEST.md5`.

## Quick start

Sample five codes from a trained GFlowQEC-offline policy at n = 180 and certify one code's distance exactly, in a few seconds on a laptop CPU (from the repository root, after `pip install -r requirements.txt` and `python -m scripts.fetch_data`):

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
python -m readouts.sum180      # prints Table III
python -m pytest tests -q      # checks every readout against its expected output
```

Each readout prints one table, figure or section of the paper from the stored runs in `data/`. It trains nothing and calls no oracle, and its docstring names the paper item it prints.

| Paper item | Readout |
|---|---|
| Table II, Sec. 4 | `keys` |
| Table III | `sum180` |
| Table IV | `sum300` |
| Table V | `oracle_costs` |
| Table VI | `nets` |
| Table VII | `methods` |
| Table VIII | `stats_appendix` |
| Table IX | `res_prefix` |
| Table X | `res300` |
| Fig. 7 | `fig_main300` |
| Secs. 1 and 3 | `background` |
| Sec. 6 and App. B.1 | `stored_forms` |
| Sec. 8 | `sec8` |
| Sec. 9.1 | `text180`, `hyb180` |
| Sec. 9.2 | `text300` |
| Sec. 9.3 | `cal72` (n = 72), `scripts.n36_calibration` (n = 36) |
| App. A | `appa_codes` |
| App. B.1 | `rel180` |
| App. C | `appe_protocol` |
| App. D.1 | `appg180` |
| App. D.2 | `appg300` |

## Regenerate the data

The scripts in `scripts/` rerun the experiments from scratch and write to `results/`, never to `data/`. Some take hundreds of CPU-hours. Each docstring gives the exact commands, seeds and cost. Results agree closely across machines but not bit for bit.

## Layout

```
gflowqec/   the library: codes, oracles, distance certifier, policies, comparison methods
readouts/   one script per paper table, figure or section
scripts/    rerun the experiments
data/       answer keys, stored runs and trained networks
tests/      check every readout and the data manifest
figures/    the two pipeline diagrams above
```

## License and citation

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA`). Third-party licenses: `THIRD-PARTY.md`. To cite, use `CITATION.cff`.
