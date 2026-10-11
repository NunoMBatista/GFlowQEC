# GFlowQEC

Code and data for the pre-print *GFlowQEC: Generative Flow Networks that Learn to Sample Quantum LDPC Codes* (Nuno Batista, Gabriel Falcão, Penousal Machado, 2026).

GFlowQEC trains generative flow networks to propose quantum LDPC codes (two-block and lifted-product codes over finite groups) and certifies their minimum distances exactly.

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
```

## License and citation

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA`). Third-party licenses: `THIRD-PARTY.md`. To cite, use `CITATION.cff`.
