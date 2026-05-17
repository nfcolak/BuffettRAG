"""Retriever with four strategies.

Strategies (selectable per-call via the `strategy` argument or per-instance
via the constructor):

    'naive'    -- vector search, no metadata filter, no rerank.
                  Baseline from the task list.

    'metadata' -- vector search + metadata filtering (year / decade / topic).
                  Year filters are auto-detected from the query.

    'vector'   -- pure vector search with optional metadata filter and
                  optional reranking.

    'hybrid'   -- vector + BM25, fused with Reciprocal Rank Fusion.
                  Optional metadata filter and optional reranking.

The design pulls all candidate sets through a uniform `SearchHit` shape so the
downstream reranker and generator don't care which path produced them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from config import DEFAULT_TOP_K, RETRIEVAL_FETCH_K, RRF_K
from src.retrieval.bm25 import BM25Retriever
from src.vector_store import SearchHit, StoredDoc

if TYPE_CHECKING:
    from src.embeddings import BGEEmbedder
    from src.retrieval.reranker import CrossEncoderReranker


@dataclass
class RetrievalResult:
    query: str
    strategy: str
    hits: List[SearchHit]
    used_filter: Optional[Dict[str, Any]] = None
    reranked: bool = False


_YEAR_RE = re.compile(r"\b(19[7-9]\d|20[0-2]\d)\b")
_DECADE_WORD_RE = re.compile(
    r"\b(1970s|1980s|1990s|2000s|2010s|2020s|seventies|eighties|nineties)\b",
    re.IGNORECASE,
)
_DECADE_WORD_TO_RANGE = {
    "1970s": (1977, 1979), "seventies": (1977, 1979),
    "1980s": (1980, 1989), "eighties": (1980, 1989),
    "1990s": (1990, 1999), "nineties": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2024),
}
_DECADE_RANGE_RE = re.compile(
    r"\b(1970s|1980s|1990s|2000s|2010s|2020s|seventies|eighties|nineties)"
    r"\s+(?:to|through|–|—|-)\s+(?:the\s+)?"
    r"(1970s|1980s|1990s|2000s|2010s|2020s|seventies|eighties|nineties)\b",
    re.IGNORECASE,
)


def detect_year_filter(query: str) -> Optional[Dict[str, Any]]:
    q = query.strip()

    m = re.search(r"(?:from|between)\s+(\d{4})\s+(?:to|and)\s+(\d{4})", q, re.IGNORECASE)
    if m:
        a, b = sorted([int(m.group(1)), int(m.group(2))])
        return {"year": {"$gte": a, "$lte": b}}

    m = _DECADE_RANGE_RE.search(q)
    if m:
        a, _ = _DECADE_WORD_TO_RANGE[m.group(1).lower()]
        _, b = _DECADE_WORD_TO_RANGE[m.group(2).lower()]
        return {"year": {"$gte": a, "$lte": b}}

    m = _DECADE_WORD_RE.search(q)
    if m:
        a, b = _DECADE_WORD_TO_RANGE[m.group(1).lower()]
        return {"year": {"$gte": a, "$lte": b}}

    years = _YEAR_RE.findall(q)
    if len(years) == 1:
        return {"year": int(years[0])}
    if len(years) >= 2:
        ys = sorted(set(int(y) for y in years))
        return {"year": {"$gte": ys[0], "$lte": ys[-1]}}

    return None


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]],
    k: int = RRF_K,
    top_k: int = DEFAULT_TOP_K,
) -> List[SearchHit]:
    fused: Dict[str, float] = {}
    by_id: Dict[str, SearchHit] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            fused[hit.id] = fused.get(hit.id, 0.0) + 1.0 / (k + rank + 1)
            by_id.setdefault(hit.id, hit)

    ranked_ids = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[:top_k]

    return [
        SearchHit(
            id=by_id[i].id,
            text=by_id[i].text,
            metadata=by_id[i].metadata,
            score=fused[i],
        )
        for i in ranked_ids
    ]


class Retriever:
    def __init__(
        self,
        vector_store,
        embedder: "BGEEmbedder",
        docs: Sequence[StoredDoc],
        reranker: Optional["CrossEncoderReranker"] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25 = BM25Retriever(docs)
        self.reranker = reranker

    def _vector_search(
        self,
        query: str,
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        q_emb = self.embedder.embed_query(query)
        return self.vector_store.search(q_emb, top_k=top_k, where=where)

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        return self.bm25.search(query, top_k=top_k, where=where)

    def search(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = DEFAULT_TOP_K,
        fetch_k: int = RETRIEVAL_FETCH_K,
        rerank: bool = False,
        where: Optional[Dict[str, Any]] = None,
        auto_year_filter: bool = True,
    ) -> RetrievalResult:
        applied_filter: Optional[Dict[str, Any]] = dict(where) if where else None

        if strategy in ("metadata", "hybrid", "vector") and auto_year_filter:
            auto = detect_year_filter(query)
            if auto:
                applied_filter = {**(applied_filter or {}), **auto}

        if strategy == "naive":
            candidates = self._vector_search(query, top_k=fetch_k)
        elif strategy == "vector":
            candidates = self._vector_search(query, top_k=fetch_k, where=applied_filter)
        elif strategy == "metadata":
            candidates = self._vector_search(query, top_k=fetch_k, where=applied_filter)
        elif strategy == "hybrid":
            vec = self._vector_search(query, top_k=fetch_k, where=applied_filter)
            bm = self._bm25_search(query, top_k=fetch_k, where=applied_filter)
            candidates = reciprocal_rank_fusion([vec, bm], top_k=fetch_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        reranked = False
        if rerank and self.reranker is not None and candidates:
            candidates = self.reranker.rerank(query, candidates, top_k=top_k)
            reranked = True
        else:
            candidates = candidates[:top_k]

        return RetrievalResult(
            query=query,
            strategy=strategy,
            hits=candidates,
            used_filter=applied_filter,
            reranked=reranked,
        )

    def cross_decade_compare(
        self,
        query: str,
        decades: Sequence[int] = (1980, 1990, 2000, 2010, 2020),
        per_decade_k: int = 3,
        rerank: bool = False,
    ) -> Dict[int, List[SearchHit]]:
        out: Dict[int, List[SearchHit]] = {}

        for d in decades:
            res = self.search(
                query,
                strategy="hybrid",
                top_k=per_decade_k,
                where={"decade": d},
                rerank=rerank,
                auto_year_filter=False,
            )
            out[d] = res.hits

        return out