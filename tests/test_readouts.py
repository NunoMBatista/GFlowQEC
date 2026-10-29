"""Every readout reprints the paper's numbers: run each one and compare with tests/expected/<name>.txt.

One test per readout, discovered from the expected files. A readout runs as
    python -m readouts.<name>
in a subprocess, from the release root, with PYTHONPATH set to the release root only. Its stdout is
compared line by line with the expected file. Lines starting with 'timed:' or 'basis:' are diagnostics
(wall times, the counts behind a number) and are dropped from both sides before comparing.

Run with pytest (python -m pytest tests -q), or without it:
    python -m tests.test_readouts
which runs every readout and prints PASS / FAIL and the wall time per readout.
"""

import difflib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = ROOT / "tests" / "expected"

# The readout names, one per expected file (tests/expected/keys.txt -> readouts.keys).
NAMES = sorted(path.stem for path in EXPECTED.glob("*.txt"))

# Diagnostic lines that are never compared.
SKIP_PREFIXES = ("timed:", "basis:")


def compared_lines(text):
    """The lines that are compared: diagnostics dropped, trailing spaces ignored."""
    lines = []
    for line in text.splitlines():
        if line.startswith(SKIP_PREFIXES):
            continue
        lines.append(line.rstrip())
    return lines


def run_readout(name):
    """Run one readout from the release root; return (stdout, stderr, return code, wall seconds)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)  # the release root only, so no other copy of gflowqec is imported
    start = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", f"readouts.{name}"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    return proc.stdout, proc.stderr, proc.returncode, time.time() - start


def check_readout(name):
    """Run the readout and compare it with its expected file.

    Returns (ok, message, wall seconds). The message is the unified diff (or the error) on failure.
    """
    stdout, stderr, code, wall = run_readout(name)
    if code != 0:
        # A crash: show the end of stderr, where the traceback is.
        return False, f"readouts.{name} exited with code {code}\n{stderr[-3000:]}", wall

    expected = compared_lines((EXPECTED / f"{name}.txt").read_text())
    got = compared_lines(stdout)
    if got == expected:
        return True, "", wall

    diff = difflib.unified_diff(expected, got, fromfile=f"expected/{name}.txt",
                                tofile=f"readouts.{name} stdout", lineterm="")
    return False, "\n".join(diff), wall


# pytest entry point: one test per readout. pytest is imported only here so that the plain-python
# entry point below works without it.
try:
    import pytest

    @pytest.mark.parametrize("name", NAMES)
    def test_readout(name):
        # fig_main300 draws its labels with LaTeX; without it the figure cannot be written
        if name == "fig_main300" and shutil.which("latex") is None:
            pytest.skip("fig_main300 needs LaTeX (latex not found)")
        ok, message, _ = check_readout(name)
        assert ok, message
except ImportError:
    pass


# Plain-python entry point: run all readouts (or the ones named on the command line).
if __name__ == "__main__":
    names = sys.argv[1:] or NAMES
    failed = 0
    total = time.time()
    for name in names:
        ok, message, wall = check_readout(name)
        print(f"{'PASS' if ok else 'FAIL'} {name} ({wall:.1f} s)", flush=True)
        if not ok:
            failed += 1
            print(message, flush=True)
    print(f"{len(names) - failed} of {len(names)} readouts pass, {time.time() - total:.1f} s in total")
    sys.exit(1 if failed else 0)
