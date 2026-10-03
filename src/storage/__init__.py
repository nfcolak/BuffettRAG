"""Unified vector store interface for Chroma, FAISS, and pgvector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from src.storage.chroma_store import ChromaStore
from src.storage.faiss_store import FaissStore
from src.storage.filters import validate_where_filter
from src.storage.pgvector_store import PgVectorStore
from src.storage.types import SearchHit, StoredDoc

__all__ = [
    "get_vector_store",
    "load_chunks_as_docs",
    "StoredDoc",
    "SearchHit",
    "validate_where_filter",
]


def get_vector_store(backend: str = "pgvector", dim: Optional[int] = None):
    if backend == "pgvector":
        return PgVectorStore(dim=dim)
    if backend == "chroma":
        return ChromaStore()
    if backend == "faiss":
        return FaissStore(dim=dim)

    raise ValueError(f"Unknown backend: {backend}")


def load_chunks_as_docs(chunks_file: Path) -> List[StoredDoc]:
    docs: List[StoredDoc] = []

    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.pop("text")
            doc_id = obj.get("id")
            docs.append(StoredDoc(id=doc_id, text=text, metadata=obj))

    return docs
