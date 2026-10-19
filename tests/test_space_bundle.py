"""deploy/hf_space/assemble.py against a fake repo tree."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ASSEMBLE = Path(__file__).resolve().parent.parent / "deploy" / "hf_space" / "assemble.py"
spec = importlib.util.spec_from_file_location("hf_space_assemble", ASSEMBLE)
assemble_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assemble_mod)


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture()
def fake_repo(tmp_path):
    repo = tmp_path / "repo"
    here = repo / "deploy" / "hf_space"
    _write(repo / "config.py")
    _write(repo / "requirements-space.txt")
    _write(here / "Dockerfile")
    _write(here / "README.md")
    _write(repo / "src" / "__init__.py")
    _write(repo / "src" / "evaluation" / "metric.py")
    _write(repo / "src" / "__pycache__" / "a.pyc")
    _write(repo / "data" / "processed" / "chunks_v3_paragraph.jsonl")
    _write(repo / "data" / "processed" / "chunks_v3_paragraph_manifest.json")
    _write(repo / "data" / "indices" / "faiss_v3" / "index.faiss")
    _write(repo / "data" / "indices" / "faiss_v3" / "meta.json")
    _write(repo / "frontend" / "dist" / "index.html")
    _write(repo / "frontend" / "dist" / "assets" / "app.js")
    # files that must stay out of the bundle
    _write(repo / ".env", "SECRET=1")
    _write(repo / "models" / "m.gguf")
    _write(repo / "data" / "evaluation" / "gold.jsonl")
    _write(repo / "openspec" / "changes" / "c.md")
    _write(repo / "data" / "processed" / "chunks_v2.jsonl")
    return repo, here


def test_assemble_copies_required_and_skips_forbidden(fake_repo, tmp_path) -> None:
    repo, here = fake_repo
    out = tmp_path / "bundle"
    files = assemble_mod.assemble(out, repo=repo, here=here)
    rel = {p.relative_to(out).as_posix() for p in files}
    for required in (
        "Dockerfile", "README.md", "config.py", "requirements-space.txt", "src/__init__.py",
        "src/evaluation/metric.py", "data/processed/chunks_v3_paragraph.jsonl",
        "data/processed/chunks_v3_paragraph_manifest.json", "data/indices/faiss_v3/index.faiss",
        "data/indices/faiss_v3/meta.json", "frontend/dist/index.html", "frontend/dist/assets/app.js",
    ):
        assert required in rel
    assert not any(r.startswith(("models", "openspec", "data/evaluation")) for r in rel)
    assert ".env" not in rel and "data/processed/chunks_v2.jsonl" not in rel
    assert not any("__pycache__" in r or r.endswith(".gguf") for r in rel)
    assert assemble_mod.forbidden_files(out) == []


def test_refuses_when_dist_or_index_missing(fake_repo, tmp_path) -> None:
    repo, here = fake_repo
    (repo / "frontend" / "dist" / "index.html").unlink()
    with pytest.raises(SystemExit, match="index.html"):
        assemble_mod.assemble(tmp_path / "a", repo=repo, here=here)
    _write(repo / "frontend" / "dist" / "index.html")
    (repo / "data" / "indices" / "faiss_v3" / "index.faiss").unlink()
    with pytest.raises(SystemExit, match="FAISS index"):
        assemble_mod.assemble(tmp_path / "b", repo=repo, here=here)
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_refuses_non_empty_output(fake_repo, tmp_path) -> None:
    repo, here = fake_repo
    out = tmp_path / "bundle"
    _write(out / "stale.txt")
    with pytest.raises(SystemExit, match="not empty"):
        assemble_mod.assemble(out, repo=repo, here=here)


def test_forbidden_scan_flags_bad_names(tmp_path) -> None:
    for name in (".env", ".env.local", "m.gguf", "heldout_v7.jsonl", "sealed/x.json", "openspec/y.md", "models/z.bin"):
        _write(tmp_path / name)
    _write(tmp_path / "data" / "evaluation" / "g.jsonl")
    _write(tmp_path / "ok.py")
    bad = set(assemble_mod.forbidden_files(tmp_path))
    assert {".env", ".env.local", "m.gguf", "heldout_v7.jsonl", "sealed/x.json", "openspec/y.md", "models/z.bin", "data/evaluation/g.jsonl"} <= bad
    assert "ok.py" not in bad
