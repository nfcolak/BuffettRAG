"""PostgreSQL pgvector store and SQL metadata-filter helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from src.storage.filters import validate_where_filter
from src.storage.types import SearchHit, StoredDoc


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
            from psycopg2 import sql
            from psycopg2.extras import execute_values
            from pgvector.psycopg2 import register_vector
        except ImportError as e:
            raise ImportError(
                "psycopg2-binary and pgvector are required for the pgvector backend."
            ) from e

        self._psycopg2 = psycopg2
        self._sql = sql
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
        self.table = _validate_pg_identifier(table or PG_TABLE, "PG_TABLE")
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
        if dim <= 0:
            raise ValueError("Embedding dimension must be positive")

        with self._conn.cursor() as cur:
            cur.execute(
                self._sql.SQL(
                    """
                CREATE TABLE IF NOT EXISTS {} (
                    id           TEXT PRIMARY KEY,
                    text         TEXT NOT NULL,
                    year         INT,
                    decade       INT,
                    source_file  TEXT,
                    chunk_index  INT,
                    topics       TEXT,
                    embedding    vector({})
                );
                """
                ).format(self._sql.Identifier(self.table), self._sql.SQL(str(int(dim))))
            )
            cur.execute(
                self._sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (year);").format(
                    self._sql.Identifier(_pg_index_name(self.table, "year_idx")),
                    self._sql.Identifier(self.table),
                )
            )
            cur.execute(
                self._sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (decade);").format(
                    self._sql.Identifier(_pg_index_name(self.table, "decade_idx")),
                    self._sql.Identifier(self.table),
                )
            )

    def _ensure_ann_index(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                self._sql.SQL(
                    """
                CREATE INDEX IF NOT EXISTS {}
                ON {} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {});
                """
                ).format(
                    self._sql.Identifier(_pg_index_name(self.table, "embedding_idx")),
                    self._sql.Identifier(self.table),
                    self._sql.SQL(str(int(self.ivfflat_lists))),
                )
            )
            cur.execute(
                self._sql.SQL("ANALYZE {};").format(self._sql.Identifier(self.table))
            )

    def __len__(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                self._sql.SQL("SELECT COUNT(*) FROM {};").format(
                    self._sql.Identifier(self.table)
                )
            )
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

        sql = self._sql.SQL(
            """
            INSERT INTO {}
            (id, text, year, decade, source_file, chunk_index, topics, embedding)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
            text = EXCLUDED.text,
            year = EXCLUDED.year,
            decade = EXCLUDED.decade,
            source_file = EXCLUDED.source_file,
            chunk_index = EXCLUDED.chunk_index,
            topics = EXCLUDED.topics,
            embedding = EXCLUDED.embedding;
            """
        ).format(self._sql.Identifier(self.table))

        with self._conn.cursor() as cur:
            sql_text = sql.as_string(cur)
            for start in tqdm(range(0, len(rows), batch_size), desc="pgvector upsert"):
                self._execute_values(cur, sql_text, rows[start : start + batch_size])

        self._ensure_ann_index()

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        where_sql, params = _pg_build_where_clause(where)
        sql = self._sql.SQL(
            """
            SELECT id, text, year, decade, source_file, chunk_index, topics,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM {}
            {}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """
        ).format(self._sql.Identifier(self.table), self._sql.SQL(where_sql))

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


_PG_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


def _validate_pg_identifier(value: str, label: str) -> str:
    if not _PG_IDENTIFIER_RE.match(value):
        raise ValueError(
            f"{label} must be a simple PostgreSQL identifier: letters, numbers, "
            "and underscores only; it must not start with a number."
        )
    return value


def _pg_index_name(table: str, suffix: str) -> str:
    return f"{table}_{suffix}"[:63]
