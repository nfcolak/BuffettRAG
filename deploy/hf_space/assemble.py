"""Assemble the Hugging Face Space build directory.

Usage:
    python deploy/hf_space/assemble.py --out <dir>

Copies exactly the serving files (code, chunks_v3_paragraph, faiss_v3 index, built frontend)
and refuses when the frontend build or the index is missing. Never copies .env, models,
evaluation data, openspec or any heldout or sealed file; the output is scanned for them.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

CHUNKS = "data/processed/chunks_v3_paragraph.jsonl"
CHUNKS_MANIFEST = "data/processed/chunks_v3_paragraph_manifest.json"
INDEX_DIR = "data/indices/faiss_v3"
FRONTEND_DIST = "frontend/dist"
FORBIDDEN_TOKENS = ("heldout", "sealed", "openspec")
SKIP_NAMES = {"__pycache__", ".DS_Store"}


def forbidden_files(root: Path) -> List[str]:
    """Paths under ``root`` that must never ship (env files, gguf, models, evaluation, heldout, sealed, openspec)."""
    bad = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        parts = [part.lower() for part in rel.parts]
        name = parts[-1]
        if (
            name.startswith(".env")
            or name.endswith(".gguf")
            or parts[0] == "models"
            or parts[:2] == ["data", "evaluation"]
            or any(token in part for part in parts for token in FORBIDDEN_TOKENS)
        ):
            bad.append(rel.as_posix())
    return bad


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(
        src, dst, ignore=lambda _d, names: [n for n in names if n in SKIP_NAMES or n.endswith(".pyc")]
    )


def assemble(out: Path, repo: Path = REPO_ROOT, here: Path = HERE) -> List[Path]:
    """Build the bundle in ``out`` (absent or empty); return the copied files."""
    dist = repo / FRONTEND_DIST
    if not (dist / "index.html").is_file():
        raise SystemExit(f"refusing: {dist}/index.html missing (run `npm run build` in frontend/)")
    index = repo / INDEX_DIR
    if not (index / "index.faiss").is_file():
        raise SystemExit(f"refusing: FAISS index missing at {index}")
    for required in (repo / CHUNKS, repo / "config.py", repo / "requirements-space.txt", here / "Dockerfile", here / "README.md"):
        if not required.is_file():
            raise SystemExit(f"refusing: required file missing: {required}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing: output dir {out} is not empty")
    out.mkdir(parents=True, exist_ok=True)

    _copy_tree(repo / "src", out / "src")
    shutil.copy2(repo / "config.py", out / "config.py")
    shutil.copy2(repo / "requirements-space.txt", out / "requirements-space.txt")
    shutil.copy2(here / "Dockerfile", out / "Dockerfile")
    shutil.copy2(here / "README.md", out / "README.md")
    (out / "data" / "processed").mkdir(parents=True)
    shutil.copy2(repo / CHUNKS, out / CHUNKS)
    if (repo / CHUNKS_MANIFEST).is_file():
        shutil.copy2(repo / CHUNKS_MANIFEST, out / CHUNKS_MANIFEST)
    (out / "data" / "indices").mkdir(parents=True)
    _copy_tree(index, out / INDEX_DIR)
    (out / "frontend").mkdir()
    _copy_tree(dist, out / FRONTEND_DIST)

    bad = forbidden_files(out)
    if bad:
        shutil.rmtree(out)
        raise SystemExit("refusing: forbidden files in bundle: " + ", ".join(bad))
    return sorted(p for p in out.rglob("*") if p.is_file())


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="build directory (created, must be empty)")
    args = parser.parse_args(argv)
    out = args.out.expanduser().resolve()
    files = assemble(out)
    total = sum(p.stat().st_size for p in files)
    for path in files:
        print(path.relative_to(out).as_posix())
    print(f"files: {len(files)}  total: {total / 1e6:.1f} MB  forbidden: {len(forbidden_files(out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
