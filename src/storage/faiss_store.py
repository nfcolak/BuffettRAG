"""Persistent FAISS vector store and document metadata serialization."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config import FAISS_DIR
from src.storage.filters import _meta_matches, validate_where_filter
from src.storage.types import SearchHit, StoredDoc


class FaissStore:
    INDEX_FILE = "index.faiss"
    META_JSON_FILE = "meta.json"
    LEGACY_META_PICKLE_FILE = "meta.pkl"

    def __init__(self, persist_dir: Path = FAISS_DIR, dim: Optional[int] = None) -> None:
        try:
            import faiss
        except ImportError as e:
            raise ImportError("faiss is required. `pip install faiss-cpu`.") from e

        self._faiss = faiss
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        idx_path = self.persist_dir / self.INDEX_FILE
        meta_json_path = self.persist_dir / self.META_JSON_FILE
        legacy_meta_path = self.persist_dir / self.LEGACY_META_PICKLE_FILE

        if idx_path.exists() and meta_json_path.exists():
            self._index = faiss.read_index(str(idx_path))
            self._docs = _load_faiss_docs_json(meta_json_path)
        elif idx_path.exists() and legacy_meta_path.exists():
            if os.getenv("ALLOW_UNSAFE_FAISS_PICKLE") != "1":
                raise RuntimeError(
                    "Refusing to load legacy FAISS pickle metadata. Rebuild the FAISS "
                    "index to create meta.json, or set ALLOW_UNSAFE_FAISS_PICKLE=1 "
                    "only for a trusted local migration."
                )
            self._index = faiss.read_index(str(idx_path))
            with open(legacy_meta_path, "rb") as f:
                self._docs = pickle.load(f)
            self._persist()
        else:
            if dim is None:
                raise ValueError("dim must be provided when creating a fresh FAISS index")
            self._index = faiss.IndexFlatIP(dim)
            self._docs = []

    def __len__(self) -> int:
        return self._index.ntotal

    def add(
        self,
        docs: Sequence[StoredDoc],
        embeddings: Sequence[Sequence[float]],
        batch_size: int = 1024,
    ) -> None:
        assert len(docs) == len(embeddings)
        arr = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms

        self._index.add(arr)
        self._docs.extend(docs)
        self._persist()

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        validate_where_filter(where)

        q = np.asarray([query_embedding], dtype=np.float32)
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1.0
        q = q / q_norm

        fetch_k = self._index.ntotal if where else top_k
        fetch_k = min(fetch_k, max(1, self._index.ntotal))
        scores, idxs = self._index.search(q, fetch_k)

        hits: List[SearchHit] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue

            doc = self._docs[idx]
            if where and not _meta_matches(doc.metadata, where):
                continue

            hits.append(
                SearchHit(
                    id=doc.id,
                    text=doc.text,
                    metadata=doc.metadata,
                    score=float(score),
                )
            )

            if len(hits) >= top_k:
                break

        return hits

    def _persist(self) -> None:
        self._faiss.write_index(self._index, str(self.persist_dir / self.INDEX_FILE))
        _save_faiss_docs_json(self.persist_dir / self.META_JSON_FILE, self._docs)


def _load_faiss_docs_json(path: Path) -> List[StoredDoc]:
    with open(path, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)

    docs: List[StoredDoc] = []
    for obj in raw_docs:
        docs.append(
            StoredDoc(
                id=str(obj["id"]),
                text=str(obj["text"]),
                metadata=dict(obj.get("metadata") or {}),
            )
        )
    return docs


def _save_faiss_docs_json(path: Path, docs: Sequence[StoredDoc]) -> None:
    payload = [
        {"id": doc.id, "text": doc.text, "metadata": doc.metadata}
        for doc in docs
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
