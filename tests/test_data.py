"""The data files are the ones the paper was read from: check them against data/MANIFEST.md5.

Each manifest line is '<md5>  <path>', the path relative to the release root (the md5sum format).
Two checks: every listed file exists with that md5, and every file under data/ is listed (the
manifest itself excepted).

Run with pytest, or without it: python -m tests.test_data
"""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MANIFEST = DATA / "MANIFEST.md5"


def read_manifest():
    """{path relative to the release root: md5} from the manifest."""
    entries = {}
    for line in MANIFEST.read_text().splitlines():
        md5, path = line.split(maxsplit=1)
        entries[path] = md5
    return entries


def md5_of(path):
    """md5 of a file, read in 1 MB chunks so large model files do not fill memory."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_manifest_md5():
    """Every manifest entry exists and has the listed md5."""
    bad = []
    for path, md5 in read_manifest().items():
        file = ROOT / path
        if not file.is_file():
            bad.append(f"missing: {path}")
        elif md5_of(file) != md5:
            bad.append(f"md5 differs: {path}")
    assert not bad, "\n".join(bad)


def test_no_unlisted_files():
    """Every file under data/ is in the manifest (the manifest itself excepted)."""
    listed = set(read_manifest())
    unlisted = []
    for file in sorted(DATA.rglob("*")):
        if file.is_file() and file != MANIFEST:
            path = file.relative_to(ROOT).as_posix()
            if path not in listed:
                unlisted.append(path)
    assert not unlisted, "not in the manifest:\n" + "\n".join(unlisted)


if __name__ == "__main__":
    failed = 0
    for test in (test_manifest_md5, test_no_unlisted_files):
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as error:
            failed += 1
            print(f"FAIL {test.__name__}\n{error}")
    sys.exit(1 if failed else 0)
