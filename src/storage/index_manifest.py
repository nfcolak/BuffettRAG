"""Build and verify corpus-bound persistent index manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.storage.types import StoredDoc

MANIFEST_FILE = "index_manifest.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def document_ids_sha256(docs: Sequence[StoredDoc]) -> str:
    payload = "\n".join(doc.id for doc in docs).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_index_manifest(
    *, corpus: Path, docs: Sequence[StoredDoc], backend: str,
    model_name: str, dimension: int, artifacts_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "backend": backend,
        "corpus": str(corpus.resolve()),
        "corpus_sha256": sha256_file(corpus),
        "document_ids_sha256": document_ids_sha256(docs),
        "document_count": len(docs),
        "embedding": {"model": model_name, "dimension": int(dimension)},
    }
    if artifacts_sha256 is not None:
        manifest["artifacts_sha256"] = dict(artifacts_sha256)
    return manifest


def write_index_manifest(
    persist_dir: Path, *, corpus: Path, docs: Sequence[StoredDoc], backend: str,
    model_name: str, dimension: int, artifact_names: Sequence[str],
) -> dict[str, Any]:
    persist_dir = Path(persist_dir)
    artifact_hashes = {
        name: sha256_file(persist_dir / name)
        for name in artifact_names
    }
    manifest = build_index_manifest(
        corpus=corpus, docs=docs, backend=backend, model_name=model_name,
        dimension=dimension, artifacts_sha256=artifact_hashes,
    )
    (persist_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def validate_index_manifest(
    manifest: Mapping[str, Any], *, corpus: Path, docs: Sequence[StoredDoc],
    backend: str, model_name: str, dimension: int,
    persist_dir: Path | None = None,
) -> None:
    if manifest.get("backend") != backend:
        raise ValueError("index backend does not match requested backend")
    if manifest.get("corpus_sha256") != sha256_file(corpus):
        raise ValueError("index corpus hash does not match requested corpus")
    if manifest.get("document_ids_sha256") != document_ids_sha256(docs):
        raise ValueError("index document ID hash does not match requested corpus")
    if int(manifest.get("document_count", -1)) != len(docs):
        raise ValueError("index document count does not match requested corpus")
    embedding = manifest.get("embedding") or {}
    if embedding.get("model") != model_name or int(embedding.get("dimension", -1)) != dimension:
        raise ValueError("index embedding configuration does not match requested model")
    if persist_dir is None:
        return
    declared = manifest.get("artifacts_sha256")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("index manifest has no artifact hashes")
    persist_dir = Path(persist_dir)
    actual_names = {
        path.name for path in persist_dir.iterdir()
        if path.is_file() and path.name != MANIFEST_FILE
    }
    if actual_names != set(declared):
        raise ValueError("index artifact set does not match manifest")
    for name, expected_hash in declared.items():
        if sha256_file(persist_dir / name) != expected_hash:
            raise ValueError(f"index artifact hash mismatch: {name}")


def load_and_validate_index_manifest(
    persist_dir: Path, *, corpus: Path, docs: Sequence[StoredDoc], backend: str,
    model_name: str, dimension: int,
) -> dict[str, Any]:
    manifest_path = Path(persist_dir) / MANIFEST_FILE
    if not manifest_path.exists():
        raise ValueError(f"missing index manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_index_manifest(
        manifest, corpus=corpus, docs=docs, backend=backend,
        model_name=model_name, dimension=dimension, persist_dir=Path(persist_dir),
    )
    return manifest


# ---------------------------------------------------------------------------
# Backend-agnostic identity (faiss artifacts, or sidecar JSON for chroma/pgvector)
# ---------------------------------------------------------------------------
MANIFEST_SIDECAR_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "indices" / "manifests"
_REBUILD = "rebuild the index for the active corpus"


def _store_label(store: Any) -> str:
    label = (
        getattr(store, "table", None)
        or getattr(store, "collection_name", None)
        or getattr(getattr(store, "_collection", None), "name", None)
        or "default"
    )
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(label))


def sidecar_manifest_path(store: Any, backend: str) -> Path:
    return MANIFEST_SIDECAR_DIR / f"{backend}_{_store_label(store)}.json"


def write_index_identity(
    store: Any, *, backend: str, corpus: Path, docs: Sequence[StoredDoc],
    model_name: str, dimension: int,
) -> dict[str, Any]:
    """Persist the identity of a freshly built index."""
    if backend == "faiss" and hasattr(store, "persist_dir"):
        return write_index_manifest(
            Path(store.persist_dir), corpus=corpus, docs=docs, backend=backend,
            model_name=model_name, dimension=dimension,
            artifact_names=(store.INDEX_FILE, store.META_JSON_FILE),
        )
    manifest = build_index_manifest(
        corpus=corpus, docs=docs, backend=backend, model_name=model_name, dimension=dimension,
    )
    manifest["row_count"] = len(store)
    path = sidecar_manifest_path(store, backend)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def ensure_index_identity(
    store: Any, *, backend: str, corpus: Path, docs: Sequence[StoredDoc],
    model_name: str, dimension: int,
) -> None:
    """Raise RuntimeError unless a non-empty store matches the active corpus/model."""
    rows = len(store)
    if rows == 0:
        return
    if rows != len(docs):
        raise RuntimeError(
            f"Vector store has {rows} rows but active corpus has {len(docs)}; {_REBUILD}"
        )
    try:
        if backend == "faiss" and hasattr(store, "persist_dir"):
            load_and_validate_index_manifest(
                Path(store.persist_dir), corpus=corpus, docs=docs, backend=backend,
                model_name=model_name, dimension=dimension,
            )
            return
        path = sidecar_manifest_path(store, backend)
        if not path.exists():
            raise ValueError(f"missing index manifest: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_index_manifest(
            manifest, corpus=corpus, docs=docs, backend=backend,
            model_name=model_name, dimension=dimension,
        )
        if int(manifest.get("row_count", -1)) != rows:
            raise ValueError("index row count does not match manifest")
    except (ValueError, OSError) as exc:
        raise RuntimeError(f"Index identity check failed ({exc}); {_REBUILD}") from exc
