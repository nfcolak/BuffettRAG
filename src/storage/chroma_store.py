"""Persistent Chroma vector store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from config import CHROMA_COLLECTION, CHROMA_DIR
from src.storage.filters import _chroma_where, _sanitize_metadata, validate_where_filter
from src.storage.types import SearchHit, StoredDoc


class ChromaStore:
    def __init__(
        self,
        persist_dir: Path = CHROMA_DIR,
        collection_name: str = CHROMA_COLLECTION,
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as e:
            raise ImportError("chromadb is required. `pip install chromadb`.") from e

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def __len__(self) -> int:
        return self._collection.count()

    def add(
        self,
        docs: Sequence[StoredDoc],
        embeddings: Sequence[Sequence[float]],
        batch_size: int = 256,
    ) -> None:
        assert len(docs) == len(embeddings), "docs and embeddings length mismatch"
        for start in tqdm(range(0, len(docs), batch_size), desc="Chroma upsert"):
            batch = docs[start : start + batch_size]
            self._collection.upsert(
                ids=[d.id for d in batch],
                documents=[d.text for d in batch],
                metadatas=[_sanitize_metadata(d.metadata) for d in batch],
                embeddings=[
                    list(map(float, e))
                    for e in embeddings[start : start + batch_size]
                ],
            )

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        validate_where_filter(where)
        result = self._collection.query(
            query_embeddings=[list(map(float, query_embedding))],
            n_results=top_k,
            where=_chroma_where(where),
            include=["documents", "metadatas", "distances"],
        )

        hits: List[SearchHit] = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]

        for doc_id, text, meta, dist in zip(ids, docs, metas, dists):
            similarity = 1.0 - float(dist)
            hits.append(
                SearchHit(
                    id=doc_id,
                    text=text,
                    metadata=meta or {},
                    score=similarity,
                )
            )

        return hits
