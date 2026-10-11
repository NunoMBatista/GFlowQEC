"""Filesystem locations, derived from the repo root so they resolve from any working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # gflowqec/ is one level under the repo root
DATA = ROOT / "data"                            # shipped inputs: answer keys, run dumps, checkpoints
OUT = ROOT / "results"                          # git-ignored; everything a script writes lands here
