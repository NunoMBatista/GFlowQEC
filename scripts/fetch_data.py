"""Download the large data files from Hugging Face, then check data/ against data/MANIFEST.md5.

The GitHub repo holds the code, the tests, the small data files and data/MANIFEST.md5. The large
files (data/n180, data/n300, data/models and data/keys/landscape_l6m3_w3.csv, about 826 MB) live in
a Hugging Face dataset under the same relative paths, so downloading it into the repo root restores
data/ exactly.

    pip install huggingface_hub                 # needed for the download only
    python -m scripts.fetch_data                # download, then check every manifest line
    python -m scripts.fetch_data --check-only   # no download, check only

Exit code 0 when every manifest line is OK, 1 otherwise.
"""

import argparse
import hashlib
import sys

from gflowqec.paths import DATA, ROOT

# The Hugging Face dataset id. It must match the real dataset id on huggingface.co.
HF_REPO = "NunoBatista/gflowqec-data"

MANIFEST = DATA / "MANIFEST.md5"


def download():
    """Download the dataset's data/ folder into the repo root."""
    # Imported here so that --check-only runs without huggingface_hub installed.
    from huggingface_hub import snapshot_download

    # Only data/ is fetched. The dataset root also holds its own README.md (the dataset card) and
    # LICENSE-DATA, which would otherwise overwrite this repo's files of the same name.
    # The pattern is matched with fnmatch, where * also matches '/', so it covers every subfolder.
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=ROOT, allow_patterns=["data/*"])


def md5_of(path):
    """md5 of a file, read in 1 MB chunks so large files do not fill memory."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check():
    """Check every manifest line ('<md5>  <path>', path relative to the repo root). True if all OK."""
    lines = MANIFEST.read_text().splitlines()
    missing = []
    differs = []
    for line in lines:
        md5, path = line.split(maxsplit=1)
        file = ROOT / path
        if not file.is_file():
            missing.append(path)
        elif md5_of(file) != md5:
            differs.append(path)

    for path in missing:
        print(f"missing: {path}")
    for path in differs:
        print(f"md5 differs: {path}")
    ok = len(lines) - len(missing) - len(differs)
    print(f"{ok} of {len(lines)} manifest lines OK ({len(missing)} missing, {len(differs)} with a different md5)")
    return ok == len(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-only", action="store_true", help="skip the download, only check the manifest")
    args = parser.parse_args()

    if not args.check_only:
        print(f"downloading {HF_REPO} (data/ only) into {ROOT}")
        download()
    sys.exit(0 if check() else 1)
