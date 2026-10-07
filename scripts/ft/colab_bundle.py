"""Build the zip that colab/train_7b_lora.ipynb trains from (run on the Mac).

python scripts/ft/colab_bundle.py --out /path/colab-bundle-7b.zip

Contents (exactly): ft_v3/train.jsonl, ft_v3/valid.jsonl, ft_v3/manifest.json, SHA256SUMS.
If manifest.json records sha256 values for train/valid they are verified first.
Nothing else (models, raw data) is ever added. Prints file count and zip sha256.
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/ft_v3"
FILES = ("train.jsonl", "valid.jsonl", "manifest.json")
FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def recorded_hashes(manifest):
    """Find sha256 values the manifest records for train/valid (any common layout)."""
    found = {}

    def walk(node, path):
        if isinstance(node, dict):
            for key, val in node.items():
                walk(val, path + [str(key)])
        elif isinstance(node, str) and len(node) == 64 and all(c in "0123456789abcdef" for c in node):
            text = "/".join(path).lower()
            if any(w in text for w in ("frozen", "corpus", "reuse", "input")):
                return
            for split in ("train", "valid"):
                if split in text and ("sha" in text or "hash" in text):
                    found[split] = node

    walk(manifest, [])
    return found


def build(out, data_dir=DATA_DIR):
    out = Path(out)
    if out.exists() or out.is_symlink():
        sys.exit("refuse: output exists")
    blobs = {}
    for name in FILES:
        p = Path(data_dir) / name
        if p.is_symlink() or not p.is_file():
            sys.exit(f"refuse: missing or not a regular file: {name}")
        blobs[name] = p.read_bytes()
    expected = recorded_hashes(json.loads(blobs["manifest.json"]))
    for split, want in expected.items():
        got = sha256_bytes(blobs[f"{split}.jsonl"])
        if got != want:
            sys.exit(f"refuse: {split}.jsonl sha256 {got} != manifest {want}")
    print(f"manifest hashes verified: {sorted(expected) or 'none recorded'}")
    sums = "".join(f"{sha256_bytes(blobs[n])}  ft_v3/{n}\n" for n in FILES).encode()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in FILES:
            zf.writestr(zipfile.ZipInfo(f"ft_v3/{name}", FIXED_TIME), blobs[name], zipfile.ZIP_DEFLATED)
        zf.writestr(zipfile.ZipInfo("SHA256SUMS", FIXED_TIME), sums, zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    print("\n".join(names))
    print(f"files: {len(names)}")
    print(f"zip sha256: {sha256_bytes(out.read_bytes())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
