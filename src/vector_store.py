"""Unified vector store interface for Chroma, FAISS, and pgvector."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    FAISS_DIR,
)

ALLOWED_METADATA_FIELDS = {
    "year",
    "decade",
    "source_file",
    "chunk_index",
    "topics",
}


@dataclass
class StoredDoc:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


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
            where=where or None,
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


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif isinstance(value, list):
            out[key] = ",".join(str(x) for x in value)
        else:
            out[key] = str(value)
    return out


class FaissStore:
    INDEX_FILE = "index.faiss"
    META_FILE = "meta.pkl"

    def __init__(self, persist_dir: Path = FAISS_DIR, dim: Optional[int] = None) -> None:
        try:
            import faiss
        except ImportError as e:
            raise ImportError("faiss is required. `pip install faiss-cpu`.") from e

        self._faiss = faiss
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        idx_path = self.persist_dir / self.INDEX_FILE
        meta_path = self.persist_dir / self.META_FILE

        if idx_path.exists() and meta_path.exists():
            self._index = faiss.read_index(str(idx_path))
            with open(meta_path, "rb") as f:
                self._docs: List[StoredDoc] = pickle.load(f)
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
        with open(self.persist_dir / self.META_FILE, "wb") as f:
            pickle.dump(self._docs, f)


def validate_where_filter(where: Optional[Dict[str, Any]]) -> None:
    if not where:
        return

    for field, cond in where.items():
        if field not in ALLOWED_METADATA_FIELDS:
            raise ValueError(f"Unsupported metadata filter field: {field}")

        if isinstance(cond, dict):
            for op, value in cond.items():
                if op not in {"$eq", "$gte", "$lte", "$gt", "$lt", "$in"}:
                    raise ValueError(f"Unsupported metadata filter operator: {op}")
                if op == "$in" and not isinstance(value, (list, tuple, set)):
                    raise ValueError("$in metadata filter value must be a list, tuple, or set")


def _meta_matches(meta: Dict[str, Any], where: Dict[str, Any]) -> bool:
    validate_where_filter(where)

    for key, cond in where.items():
        value = meta.get(key)

        if isinstance(cond, dict):
            for op, target in cond.items():
                if op == "$eq" and value != target:
                    return False
                if op == "$gte" and not (value is not None and value >= target):
                    return False
                if op == "$lte" and not (value is not None and value <= target):
                    return False
                if op == "$gt" and not (value is not None and value > target):
                    return False
                if op == "$lt" and not (value is not None and value < target):
                    return False
                if op == "$in" and value not in target:
                    return False
        else:
            if value != cond:
                return False

    return True


class PgVectorStore:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        table: Optional[str] = None,
        dim: Optional[int] = None,
        ivfflat_lists: Optional[int] = None,
    ) -> None:
        try:
            import psycopg2
            from psycopg2.extras import execute_values
            from pgvector.psycopg2 import register_vector
        except ImportError as e:
            raise ImportError(
                "psycopg2-binary and pgvector are required for the pgvector backend."
            ) from e

        self._psycopg2 = psycopg2
        self._execute_values = execute_values
        self._register_vector = register_vector

        from config import (
            PG_DATABASE,
            PG_HOST,
            PG_IVFFLAT_LISTS,
            PG_PASSWORD,
            PG_PORT,
            PG_TABLE,
            PG_USER,
        )

        self.host = host or PG_HOST
        self.port = port or PG_PORT
        self.user = user or PG_USER
        self.password = password or PG_PASSWORD
        self.database = database or PG_DATABASE
        self.table = table or PG_TABLE
        self.ivfflat_lists = ivfflat_lists or PG_IVFFLAT_LISTS
        self.dim = dim

        self._conn = self._connect()
        self._ensure_extension()

        if dim is not None:
            self._ensure_schema(dim)

    def _connect(self):
        conn = self._psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.database,
        )
        conn.autocommit = True
        self._register_vector(conn)
        return conn

    def _ensure_extension(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    def _ensure_schema(self, dim: int) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id           TEXT PRIMARY KEY,
                    text         TEXT NOT NULL,
                    year         INT,
                    decade       INT,
                    source_file  TEXT,
                    chunk_index  INT,
                    topics       TEXT,
                    embedding    vector({dim})
                );
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_year_idx "
                f"ON {self.table} (year);"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_decade_idx "
                f"ON {self.table} (decade);"
            )

    def _ensure_ann_index(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx
                ON {self.table} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {self.ivfflat_lists});
                """
            )
            cur.execute(f"ANALYZE {self.table};")

    def __len__(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table};")
            return int(cur.fetchone()[0])

    def add(
        self,
        docs: Sequence[StoredDoc],
        embeddings: Sequence[Sequence[float]],
        batch_size: int = 256,
    ) -> None:
        assert len(docs) == len(embeddings), "docs and embeddings length mismatch"
        if not docs:
            return

        dim = len(embeddings[0])
        if self.dim is None:
            self.dim = dim
            self._ensure_schema(dim)
        elif self.dim != dim:
            raise ValueError(
                f"Embedding dim mismatch: store was created with dim={self.dim} "
                f"but received vectors of dim={dim}"
            )

        rows = []
        for doc, emb in zip(docs, embeddings):
            meta = doc.metadata
            rows.append(
                (
                    doc.id,
                    doc.text,
                    meta.get("year"),
                    meta.get("decade"),
                    meta.get("source_file"),
                    meta.get("chunk_index"),
                    meta.get("topics") if isinstance(meta.get("topics"), str) else None,
                    list(map(float, emb)),
                )
            )

        sql = (
            f"INSERT INTO {self.table} "
            f"(id, text, year, decade, source_file, chunk_index, topics, embedding) "
            f"VALUES %s "
            f"ON CONFLICT (id) DO UPDATE SET "
            f"text = EXCLUDED.text, "
            f"year = EXCLUDED.year, "
            f"decade = EXCLUDED.decade, "
            f"source_file = EXCLUDED.source_file, "
            f"chunk_index = EXCLUDED.chunk_index, "
            f"topics = EXCLUDED.topics, "
            f"embedding = EXCLUDED.embedding;"
        )

        with self._conn.cursor() as cur:
            for start in tqdm(range(0, len(rows), batch_size), desc="pgvector upsert"):
                self._execute_values(cur, sql, rows[start : start + batch_size])

        self._ensure_ann_index()

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        where_sql, params = _pg_build_where_clause(where)
        sql = (
            f"SELECT id, text, year, decade, source_file, chunk_index, topics, "
            f"       1 - (embedding <=> %s::vector) AS similarity "
            f"FROM {self.table} "
            f"{where_sql} "
            f"ORDER BY embedding <=> %s::vector "
            f"LIMIT %s;"
        )

        q = list(map(float, query_embedding))
        with self._conn.cursor() as cur:
            cur.execute(sql, [q, *params, q, top_k])
            rows = cur.fetchall()

        hits: List[SearchHit] = []
        for row in rows:
            doc_id, text, year, decade, source_file, chunk_index, topics, sim = row
            hits.append(
                SearchHit(
                    id=doc_id,
                    text=text,
                    metadata={
                        "year": year,
                        "decade": decade,
                        "source_file": source_file,
                        "chunk_index": chunk_index,
                        "topics": topics or "",
                    },
                    score=float(sim),
                )
            )

        return hits

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


def _pg_build_where_clause(where: Optional[Dict[str, Any]]) -> tuple:
    if not where:
        return "", []

    validate_where_filter(where)

    clauses = []
    params = []

    for field, cond in where.items():
        if isinstance(cond, dict):
            for op, value in cond.items():
                if op == "$eq":
                    clauses.append(f"{field} = %s")
                    params.append(value)
                elif op == "$gte":
                    clauses.append(f"{field} >= %s")
                    params.append(value)
                elif op == "$lte":
                    clauses.append(f"{field} <= %s")
                    params.append(value)
                elif op == "$gt":
                    clauses.append(f"{field} > %s")
                    params.append(value)
                elif op == "$lt":
                    clauses.append(f"{field} < %s")
                    params.append(value)
                elif op == "$in":
                    if not value:
                        clauses.append("FALSE")
                    else:
                        placeholders = ",".join(["%s"] * len(value))
                        clauses.append(f"{field} IN ({placeholders})")
                        params.extend(value)
        else:
            clauses.append(f"{field} = %s")
            params.append(cond)

    return "WHERE " + " AND ".join(clauses), params


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