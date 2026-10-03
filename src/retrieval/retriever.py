"""Retriever with four strategies plus multi-subquery decomposition.

Strategies (selectable per-call via the `strategy` argument or per-instance
via the constructor):

    'naive'        -- vector search, no metadata filter, no rerank.
                      Baseline from the task list.

    'metadata'     -- vector search + metadata filtering (year / decade / topic).
                      Year filters are auto-detected from the query.

    'vector'       -- pure vector search with optional metadata filter and
                      optional reranking.

    'hybrid'       -- vector + BM25, fused with Reciprocal Rank Fusion.
                      Optional metadata filter and optional reranking.
                      Automatically uses multi-subquery decomposition when a
                      temporal comparison pattern is detected (e.g. "changed
                      from the 1990s to the 2020s").

    'hybrid_multi' -- hybrid search run separately per detected time period,
                      results merged with RRF. Set automatically; can also be
                      forced by passing strategy='hybrid_multi'.

The design pulls all candidate sets through a uniform `SearchHit` shape so the
downstream reranker and generator don't care which path produced them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from config import DEFAULT_TOP_K, RERANK_CANDIDATES, RETRIEVAL_FETCH_K, RRF_K
from src.retrieval.bm25 import BM25Retriever, _meta_matches
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
    "1970s": (1970, 1979), "seventies": (1970, 1979),
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


_TEMPORAL_COMPARISON_RE = re.compile(
    r"\b(changed?|evolved?|shifted?|developed?|differed?|grown?|"
    r"compared?|contrasted?|over time|over the years|across (?:the )?decades?|"
    r"throughout|historically|then (?:vs|versus)|now (?:vs|versus))\b",
    re.IGNORECASE,
)


def detect_temporal_comparison(query: str) -> Optional[List[Dict[str, Any]]]:
    """Detect two or more explicit time periods in question order.

    Years and decades are kept distinct; aliases of the same decade collapse.
    The caller isolates retrieval and answer generation for every period.
    """
    # Explicit periods, not comparison verbs, determine the decomposition.
    # Keep all distinct periods in question order, including mixed years/decades;
    # do not silently replace distant explicit years with intervening ranges.
    period_re = re.compile(
        r"\b(?:((?:19|20)\d0)s|((?:19|20)\d{2})|"
        r"(seventies|eighties|nineties))\b", re.IGNORECASE
    )
    periods: List[Dict[str, Any]] = []
    for match in period_re.finditer(query):
        decade, year, word = match.groups()
        if decade:
            start = int(decade)
            filt = {"year": {"$gte": start, "$lte": start + 9}}
        elif word:
            start, end = _DECADE_WORD_TO_RANGE[word.lower()]
            filt = {"year": {"$gte": start, "$lte": end}}
        else:
            filt = {"year": int(year)}
        if filt not in periods:
            periods.append(filt)
    return periods if len(periods) >= 2 else None


def reserve_period_hits(
    hits: Sequence[SearchHit], periods: Sequence[Dict[str, Any]], top_k: int,
) -> List[SearchHit]:
    """Reserve each nonempty period's best hit before the global relevance cut."""
    reserved: List[SearchHit] = []
    seen = set()
    for period in periods:
        best = next((hit for hit in hits if _meta_matches(hit.metadata, period)), None)
        if best is not None and best.id not in seen:
            reserved.append(best)
            seen.add(best.id)
    limit = max(top_k, len(reserved))
    for hit in hits:
        if len(reserved) >= limit:
            break
        if hit.id not in seen:
            reserved.append(hit)
            seen.add(hit.id)
    return reserved


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


_DEDUP_TOKEN_RE = re.compile(r"[a-z0-9']+")


def deduplicate_hits(hits: Sequence[SearchHit]) -> List[SearchHit]:
    """Drop hits whose text near-duplicates an earlier (higher-ranked) hit.

    Overlapping chunk windows and repeated boilerplate produce passages that
    are identical or mostly contained in one another; keeping both wastes
    context slots and makes the LLM repeat itself.
    """
    kept: List[SearchHit] = []
    kept_tokens: List[set] = []
    for hit in hits:
        tokens = set(_DEDUP_TOKEN_RE.findall(hit.text.lower()))
        is_dup = False
        for previous, seen in zip(kept, kept_tokens):
            if hit.metadata.get("year") != previous.metadata.get("year"):
                continue
            # Small changes to quantities/polarity are material evidence, even
            # when almost every other token is identical. Preserve them.
            guard = r"\b(?:\d+(?:[.,]\d+)*|not|never|no|without)\b"
            if re.findall(guard, hit.text.lower()) != re.findall(guard, previous.text.lower()):
                continue
            overlap = len(tokens & seen)
            union = len(tokens | seen) or 1
            smaller = min(len(tokens), len(seen)) or 1
            if overlap / union >= 0.85 or overlap / smaller >= 0.92:
                is_dup = True
                break
        if not is_dup:
            kept.append(hit)
            kept_tokens.append(tokens)
    return kept


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

    def _hybrid_for_filter(
        self,
        query: str,
        fetch_k: int,
        where: Optional[Dict[str, Any]],
    ) -> List[SearchHit]:
        """Single hybrid (vector + BM25 → RRF) pass with a specific filter."""
        vec = self._vector_search(query, top_k=fetch_k, where=where)
        bm = self._bm25_search(query, top_k=fetch_k, where=where)
        return reciprocal_rank_fusion([vec, bm], top_k=fetch_k)

    def _multi_subquery_search(
        self,
        query: str,
        subquery_filters: List[Dict[str, Any]],
        top_k: int,
        fetch_k: int,
        rerank: bool,
        retrieval_query: Optional[str] = None,
        strategy: str = "hybrid",
    ) -> "RetrievalResult":
        """Search each explicit period independently, then reserve its best hit."""
        sub_rankings: List[List[SearchHit]] = []
        for filt in subquery_filters:
            search = self._bm25_search if strategy == "bm25" else self._hybrid_for_filter
            hits = search(query, fetch_k, filt)
            if retrieval_query and retrieval_query != query:
                expanded = search(retrieval_query, fetch_k, filt)
                hits = reciprocal_rank_fusion([hits, expanded], top_k=fetch_k)
            sub_rankings.append(hits)

        # Keep every period's candidate pool until after reranking. Global
        # truncation can otherwise remove an entire side of a comparison.
        candidates = reciprocal_rank_fusion(
            sub_rankings, top_k=sum(len(hits) for hits in sub_rankings)
        )
        reranked = False
        if rerank and self.reranker is not None and candidates:
            per_period = max(RERANK_CANDIDATES // max(len(subquery_filters), 1), 4)
            pool: List[SearchHit] = []
            pooled = set()
            for filt in subquery_filters:
                matching = [h for h in candidates if _meta_matches(h.metadata, filt)]
                for h in matching[:per_period]:
                    if h.id not in pooled:
                        pooled.add(h.id)
                        pool.append(h)
            pool = pool or candidates[:RERANK_CANDIDATES]
            candidates = self.reranker.rerank(query, pool, top_k=len(pool))
            reranked = True

        # Reserve the best supported candidate per nonempty period; fill the
        # remainder by relevance. Dedupe within periods, never across years.
        buckets = [deduplicate_hits([h for h in candidates if _meta_matches(h.metadata, filt)])
                   for filt in subquery_filters]
        eligible = {h.id for bucket in buckets for h in bucket}
        candidates = reserve_period_hits(
            [h for h in candidates if h.id in eligible], subquery_filters, top_k
        )

        filter_summary = {"multi_subquery": subquery_filters}
        return RetrievalResult(
            query=query,
            strategy="bm25_multi" if strategy == "bm25" else "hybrid_multi",
            hits=candidates,
            used_filter=filter_summary,
            reranked=reranked,
        )

    def search(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = DEFAULT_TOP_K,
        fetch_k: int = RETRIEVAL_FETCH_K,
        rerank: bool = False,
        where: Optional[Dict[str, Any]] = None,
        auto_year_filter: bool = True,
        retrieval_query: Optional[str] = None,
    ) -> "RetrievalResult":
        applied_filter: Optional[Dict[str, Any]] = dict(where) if where else None

        # The lexical path uses the same per-period reservation as hybrid.
        if strategy in ("hybrid", "hybrid_multi", "bm25") and auto_year_filter and not where:
            subquery_filters = detect_temporal_comparison(query)
            if subquery_filters:
                return self._multi_subquery_search(
                    query, subquery_filters, top_k, fetch_k, rerank, retrieval_query, strategy
                )

        auto_applied = False
        if strategy in ("metadata", "hybrid", "vector", "bm25") and auto_year_filter and not where:
            auto = detect_year_filter(query)
            if auto:
                applied_filter = {**(applied_filter or {}), **auto}
                auto_applied = True

        if strategy == "naive":
            candidates = self._vector_search(query, top_k=fetch_k)
        elif strategy == "bm25":
            candidates = self._bm25_search(query, top_k=fetch_k, where=applied_filter)
            if retrieval_query and retrieval_query != query:
                expanded = self._bm25_search(retrieval_query, top_k=fetch_k, where=applied_filter)
                candidates = reciprocal_rank_fusion([candidates, expanded], top_k=fetch_k)
        elif strategy == "vector":
            candidates = self._vector_search(query, top_k=fetch_k, where=applied_filter)
        elif strategy == "metadata":
            candidates = self._vector_search(query, top_k=fetch_k, where=applied_filter)
        elif strategy in ("hybrid", "hybrid_multi"):
            candidates = self._hybrid_for_filter(query, fetch_k, applied_filter)
            if retrieval_query and retrieval_query != query:
                expanded = self._hybrid_for_filter(retrieval_query, fetch_k, applied_filter)
                # Expansion proposes candidates; it never replaces original
                # intent, invents a year filter, or becomes the rerank question.
                candidates = reciprocal_rank_fusion([candidates, expanded], top_k=fetch_k)
            if auto_applied:
                # The auto-detected year filter is a guess; treat it as a boost
                # rather than a hard constraint so a wrong guess cannot zero
                # out recall. Filtered ranking is weighted double in the merge.
                unfiltered = self._hybrid_for_filter(query, fetch_k, None)
                candidates = reciprocal_rank_fusion(
                    [candidates, candidates, unfiltered], top_k=fetch_k
                )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        reranked = False
        if rerank and self.reranker is not None and candidates:
            to_rerank = candidates[:RERANK_CANDIDATES]
            rerank_pool = self.reranker.rerank(
                query, to_rerank, top_k=min(len(to_rerank), max(top_k * 2, top_k + 4))
            )
            candidates = deduplicate_hits(rerank_pool)[:top_k]
            reranked = True
        else:
            candidates = deduplicate_hits(candidates)[:top_k]

        return RetrievalResult(
            query=query,
            strategy=strategy,
            hits=candidates,
            used_filter=applied_filter,
            reranked=reranked,
        )
