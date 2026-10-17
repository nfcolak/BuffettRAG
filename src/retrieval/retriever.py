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

from config import DEFAULT_TOP_K, RETRIEVAL_FETCH_K, retrieval_setting
from src.retrieval.bm25 import BM25Retriever, _meta_matches
from src.storage import SearchHit, StoredDoc

if TYPE_CHECKING:
    from src.storage.embeddings import BGEEmbedder
    from src.retrieval.reranker import CrossEncoderReranker


@dataclass
class RetrievalResult:
    query: str
    strategy: str
    hits: List[SearchHit]
    used_filter: Optional[Dict[str, Any]] = None
    reranked: bool = False


_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")  # RETRIEVAL_YEAR_FILTER_MIN (1970) picks the earliest year used
# Cue directly before a year; "from 1980 through 1990" keeps its end year (a range, not a reference).
_YEAR_REF_BEFORE_RE = re.compile(
    r"(?<!\bfrom\s(?:19|20)\d{2}\s)(?<!\bbetween\s(?:19|20)\d{2}\s)"
    r"\b(?:since|end\s+of|as\s+of|through|same\b[^.?!,;]{0,40}?\bas(?:\s+in)?)\s+(?:the\s+)?(?:year\s+)?$",
    re.IGNORECASE,
)
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


CORPUS_YEAR_MIN, CORPUS_YEAR_MAX = 1977, 2024
_PERIOD_RE = re.compile(
    r"\b(?:((?:19|20)\d0)s|((?:19|20)\d{2})|(seventies|eighties|nineties))\b", re.IGNORECASE
)
_COMPARISON_CUE_RE = re.compile(
    r"\b(?:compar\w*|contrast\w*|versus|vs|between|chang\w*|differ\w*|both years|each year)\b"
    r"|\bfrom\b[^.?!]*?\bto\b",
    re.IGNORECASE,
)
# "the 2002 and 2003 letters", "1986 annual report".
_LETTER_YEARS_RE = re.compile(
    r"\b((?:19|20)\d{2}(?:\s*(?:,|and|or|/)\s*(?:and\s+)?(?:19|20)\d{2})*)"
    r"\s+(?:annual\s+|shareholder\s+)?(?:letters?|reports?)\b",
    re.IGNORECASE,
)
_LETTER_OF_YEAR_RE = re.compile(r"\bletters?\s+(?:of|for|from)\s+(?:19|20)\d{2}\b", re.IGNORECASE)


def explicit_periods(text: str, keep_out_of_corpus: bool = False) -> List[Dict[str, Any]]:
    """Distinct in-corpus periods named in ``text``, in order, with no cue rule.

    Years and decades wholly outside 1977-2024 are not letter periods, unless
    ``keep_out_of_corpus`` keeps single years (COMPARE_ROUTING_V2: an unanswerable half).
    """
    periods: List[Dict[str, Any]] = []
    for match in _PERIOD_RE.finditer(text):
        decade, year, word = match.groups()
        if decade:
            start, end = int(decade), int(decade) + 9
            filt: Dict[str, Any] = {"year": {"$gte": start, "$lte": end}}
        elif word:
            start, end = _DECADE_WORD_TO_RANGE[word.lower()]
            filt = {"year": {"$gte": start, "$lte": end}}
        else:
            start = end = int(year)
            filt = {"year": start}
        if end < CORPUS_YEAR_MIN or start > CORPUS_YEAR_MAX:
            if not (keep_out_of_corpus and year and 1900 <= start <= 2099):
                continue
        if filt not in periods:
            periods.append(filt)
    return periods


def history_pins_letter(history: Optional[Sequence[Dict[str, Any]]]) -> bool:
    """True when an earlier turn already names a specific letter year."""
    for turn in history or []:
        text = str(turn.get("content", ""))
        if _LETTER_YEARS_RE.search(text) or _LETTER_OF_YEAR_RE.search(text):
            return True
    return False


def detect_temporal_comparison(
    query: str, history: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Detect two or more explicit time periods in question order.

    Years and decades are kept distinct; aliases of the same decade collapse.
    The caller isolates retrieval and answer generation for every period.
    Years outside the corpus (1977-2024) are ignored, a comparison cue or two
    "YYYY letter" mentions is required, and a follow-up whose history already
    pins a letter is never split: its years are event years, not letters.
    """
    if history_pins_letter(history):
        return None
    if compare_routing_v2():
        return _detect_temporal_comparison_v2(query)
    periods = explicit_periods(query)
    if len(periods) < 2:
        return None
    letter_years = {
        y for m in _LETTER_YEARS_RE.finditer(query) for y in re.findall(r"(?:19|20)\d{2}", m.group(1))
    }
    if not _COMPARISON_CUE_RE.search(query) and len(letter_years) < 2:
        return None
    return periods


def compare_routing_v2() -> bool:
    """COMPARE_ROUTING_V2 at call time (default off): the pv8 routing rules."""
    return bool(retrieval_setting("COMPARE_ROUTING_V2"))


# Years that are rows of a table or a reference point, not letters: "each of 2001, 2002 and 2003", "since 2005".
_YEAR_LIST_RE = re.compile(
    r"\b(?:(?:19|20)\d{2}\s*,\s*){2,}(?:and\s+)?(?:19|20)\d{2}\b"
    r"|\b(?:19|20)\d{2}\s*,\s*(?:19|20)\d{2}\s*,?\s+(?:and|&)\s+(?:19|20)\d{2}\b"
    r"|\beach of\s+(?:19|20)\d{2}\b[^?]*",
    re.IGNORECASE,
)
_SINCE_YEAR_RE = re.compile(r"\b(?:since|as of|starting in|starting from)\s+(?:19|20)\d{2}\b(?!\s+(?:annual\s+)?letters?)", re.IGNORECASE)
# "in 1990 and in 1996", "in 1996 and a year later in 1997" (two event years in one fact question are no cue).
_EXTRA_CUE_RE = re.compile(
    r"\b(?:\w+\s+)?years?\s+later\b|\bin\s+(?:19|20)\d{2}\s*(?:,|\band\b|\bor\b)\s*(?:again\s+|then\s+)?in\s+(?:19|20)\d{2}\b",
    re.IGNORECASE,
)


def _detect_temporal_comparison_v2(query: str) -> Optional[List[Dict[str, Any]]]:
    """Routing rules behind COMPARE_ROUTING_V2.

    Table-row year lists and "since Y" reference years are not letter periods; a year outside the
    corpus stays as a period (the answer path reports it as not covered); "in A and in B" and
    "a year later" count as comparison cues. At least one period must be an in-corpus letter.
    """
    masked = _SINCE_YEAR_RE.sub(" ", _YEAR_LIST_RE.sub(" ", query))
    periods = explicit_periods(masked, keep_out_of_corpus=True)
    if len(periods) < 2 or not explicit_periods(masked):
        return None
    letter_years = {
        y for m in _LETTER_YEARS_RE.finditer(masked) for y in re.findall(r"(?:19|20)\d{2}", m.group(1))
    }
    if not (_COMPARISON_CUE_RE.search(masked) or len(letter_years) >= 2 or _EXTRA_CUE_RE.search(masked)):
        return None
    return periods


def reserve_period_hits(
    hits: Sequence[SearchHit], periods: Sequence[Dict[str, Any]], top_k: int,
    per_period: Optional[int] = None,
) -> List[SearchHit]:
    """Reserve each nonempty period's best hit(s) before the global relevance cut.

    ``per_period`` defaults to the RETRIEVAL_PERIOD_RESERVE setting (1).
    """
    per_period = max(1, per_period if per_period is not None else retrieval_setting("RETRIEVAL_PERIOD_RESERVE"))
    reserved: List[SearchHit] = []
    seen = set()
    for period in periods:
        taken = 0
        for hit in hits:
            if taken >= per_period:
                break
            if hit.id not in seen and _meta_matches(hit.metadata, period):
                reserved.append(hit)
                seen.add(hit.id)
                taken += 1
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

    first_year = retrieval_setting("RETRIEVAL_YEAR_FILTER_MIN")
    if retrieval_setting("RETRIEVAL_YEAR_REF_SKIP"):
        # "since 1990", "end of 2001": a reference point, not the asked letter.
        years = [m.group() for m in _YEAR_RE.finditer(q)
                 if int(m.group()) >= first_year and not _YEAR_REF_BEFORE_RE.search(q[:m.start()])]
    else:
        years = [y for y in _YEAR_RE.findall(q) if int(y) >= first_year]
    if len(years) == 1:
        return {"year": int(years[0])}
    if len(years) >= 2:
        ys = sorted(set(int(y) for y in years))
        return {"year": {"$gte": ys[0], "$lte": ys[-1]}}

    return None


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]],
    k: Optional[int] = None,
    top_k: int = DEFAULT_TOP_K,
    weights: Optional[Sequence[float]] = None,
) -> List[SearchHit]:
    """Weighted reciprocal-rank fusion; ``k`` defaults to the RRF_K setting (60)."""
    if k is None:
        k = retrieval_setting("RRF_K")
    fused: Dict[str, float] = {}
    by_id: Dict[str, SearchHit] = {}

    for index, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[index]
        for rank, hit in enumerate(ranking):
            fused[hit.id] = fused.get(hit.id, 0.0) + weight / (k + rank + 1)
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


def mmr_select(hits: Sequence[SearchHit], top_k: int, lam: float) -> List[SearchHit]:
    """Maximal marginal relevance over reranked hits: lam * relevance - (1 - lam) * max token-Jaccard to picked.

    Relevance is the reranker's min-max score (kept on ``hit.score``); the best hit is always first.
    """
    pool = list(hits)
    if not pool:
        return []
    tokens = {hit.id: set(_DEDUP_TOKEN_RE.findall(hit.text.lower())) for hit in pool}
    picked: List[SearchHit] = [pool.pop(0)]
    while pool and len(picked) < top_k:
        def gain(hit: SearchHit) -> float:
            sim = max(len(tokens[hit.id] & tokens[p.id]) / (len(tokens[hit.id] | tokens[p.id]) or 1) for p in picked)
            return lam * hit.score - (1.0 - lam) * sim
        best = max(pool, key=gain)
        pool.remove(best)
        picked.append(best)
    return picked


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
        weights = (retrieval_setting("HYBRID_VECTOR_WEIGHT"), retrieval_setting("HYBRID_BM25_WEIGHT"))
        return reciprocal_rank_fusion([vec, bm], top_k=fetch_k, weights=None if weights == (1.0, 1.0) else weights)

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
            depth = retrieval_setting("RERANK_CANDIDATES")
            n_periods = max(len(subquery_filters), 1)
            per_period = max(depth // n_periods, 4)
            pool_size = retrieval_setting("RETRIEVAL_PERIOD_POOL")
            if pool_size > 0:  # wider per-period pool; the total stays <= 60 (rerank cost)
                per_period = min(pool_size, max(60 // n_periods, 4))
            pool: List[SearchHit] = []
            period_pools: List[List[SearchHit]] = []
            pooled = set()
            for filt in subquery_filters:
                matching = [h for h in candidates if _meta_matches(h.metadata, filt)]
                period_pools.append([])
                for h in matching[:per_period]:
                    if h.id not in pooled:
                        pooled.add(h.id)
                        pool.append(h)
                        period_pools[-1].append(h)
            if pool and retrieval_setting("RETRIEVAL_PERIOD_FOCUS_RERANK"):
                # Each period is judged against its own sub-question; reranker scores of different
                # queries are not comparable, so the lists are interleaved by per-period rank.
                from src.generation.compare import period_focused_query  # compare imports this module

                ranked = [self.reranker.rerank(period_focused_query(query, filt, subquery_filters), part,
                                               top_k=len(part)) if part else []
                          for filt, part in zip(subquery_filters, period_pools)]
                candidates = [lst[rank] for rank in range(max(len(lst) for lst in ranked))
                              for lst in ranked if rank < len(lst)]
            else:
                pool = pool or candidates[:depth]
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

    def _append_neighbor_anchors(self, hits: List[SearchHit]) -> List[SearchHit]:
        """Append the previous/next chunk of the top-N hits as passages of their own.

        RETRIEVAL_NEIGHBOR_ANCHORS = N (default 0 = off). Same letter only; ids already in the list are skipped.
        """
        count = retrieval_setting("RETRIEVAL_NEIGHBOR_ANCHORS")
        if count <= 0 or not hits:
            return hits
        if getattr(self, "_docs_by_id", None) is None:
            self._docs_by_id = {doc.id: doc for doc in self.bm25.docs}
        seen = {hit.id for hit in hits}
        extra: List[SearchHit] = []
        for hit in hits[:count]:
            doc = self._docs_by_id.get(hit.id)
            if doc is None:
                continue
            for field in ("previous_chunk_id", "next_chunk_id"):
                neighbour = self._docs_by_id.get(str(doc.metadata.get(field) or ""))
                if (neighbour is None or neighbour.id in seen
                        or neighbour.metadata.get("year") != doc.metadata.get("year")
                        or neighbour.metadata.get("source_file") != doc.metadata.get("source_file")):
                    continue
                seen.add(neighbour.id)
                extra.append(SearchHit(neighbour.id, neighbour.text, neighbour.metadata, hit.score))
        return [*hits, *extra]

    def _multipart_search(
        self, query: str, strategy: str, top_k: int, fetch_k: int, rerank: bool, auto_year_filter: bool,
        retrieval_query: Optional[str], history: Optional[Sequence[Dict[str, Any]]],
    ) -> Optional["RetrievalResult"]:
        """Split a multi-part question; each part keeps RETRIEVAL_PART_SLOTS reserved slots.

        Returns None when the question does not split into 2+ parts. The whole-question ranking
        fills the remaining slots; part hits not in it are appended in part order.
        """
        from src.generation.decompose import split_question

        parts = split_question(query)
        if len(parts) < 2:
            return None
        common = dict(strategy=strategy, top_k=top_k, fetch_k=fetch_k, rerank=rerank,
                      auto_year_filter=auto_year_filter, history=history, _split_parts=False)
        base = self.search(query, retrieval_query=retrieval_query, **common)
        slots = max(1, retrieval_setting("RETRIEVAL_PART_SLOTS"))
        reserved: List[SearchHit] = []
        reserved_ids = set()
        for part in parts:
            taken = 0
            for hit in self.search(part, **common).hits:
                if taken >= slots:
                    break
                if hit.id not in reserved_ids:
                    reserved.append(hit)
                    reserved_ids.add(hit.id)
                    taken += 1
        reserved = reserved[:top_k]
        reserved_ids = {hit.id for hit in reserved}
        chosen = list(reserved)
        chosen_ids = set(reserved_ids)
        for hit in base.hits:
            if len(chosen) >= top_k:
                break
            if hit.id not in chosen_ids:
                chosen.append(hit)
                chosen_ids.add(hit.id)
        rank = {hit.id: i for i, hit in enumerate(base.hits)}
        chosen.sort(key=lambda hit: rank.get(hit.id, len(rank)))  # stable: part-only hits keep part order last
        return RetrievalResult(query=query, strategy=strategy, hits=chosen, used_filter=base.used_filter,
                               reranked=base.reranked)

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
        history: Optional[Sequence[Dict[str, Any]]] = None,
        _split_parts: bool = True,
    ) -> "RetrievalResult":
        applied_filter: Optional[Dict[str, Any]] = dict(where) if where else None
        if rerank and self.reranker is not None and retrieval_setting("RETRIEVAL_RERANK_FETCH"):
            fetch_k = max(fetch_k, retrieval_setting("RERANK_CANDIDATES"))
        if (retrieval_setting("RETRIEVAL_QUERY_STRIP") and not retrieval_query
                and strategy in ("hybrid", "hybrid_multi", "bm25")):
            from src.retrieval.query_expansion import strip_question_scaffolding
            stripped = strip_question_scaffolding(query)
            retrieval_query = stripped if stripped and stripped != query else None

        # The lexical path uses the same per-period reservation as hybrid.
        if strategy in ("hybrid", "hybrid_multi", "bm25") and auto_year_filter and not where:
            subquery_filters = detect_temporal_comparison(query, history)
            if subquery_filters:
                return self._multi_subquery_search(
                    query, subquery_filters, top_k, fetch_k, rerank, retrieval_query, strategy
                )

        if (_split_parts and not where and strategy in ("hybrid", "hybrid_multi", "bm25")
                and retrieval_setting("RETRIEVAL_MULTIPART")):
            merged = self._multipart_search(query, strategy, top_k, fetch_k, rerank, auto_year_filter,
                                            retrieval_query, history)
            if merged is not None:
                return merged

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
                weight = max(1, retrieval_setting("RETRIEVAL_YEAR_FILTER_WEIGHT"))
                candidates = reciprocal_rank_fusion(
                    [*([candidates] * weight), unfiltered], top_k=fetch_k
                )
                keep = retrieval_setting("RETRIEVAL_YEAR_FILTER_KEEP")
                if keep > 0:
                    # A wrong year guess can push the unfiltered top hits out of the rerank window: pin
                    # the top-K of them at the end of it, the lowest merged candidates move down.
                    window = retrieval_setting("RERANK_CANDIDATES")
                    in_window = {h.id for h in candidates[:window]}
                    pinned = [h for h in unfiltered[:keep] if h.id not in in_window]
                    if pinned:
                        pinned_ids = {h.id for h in pinned}
                        rest = [h for h in candidates if h.id not in pinned_ids]
                        cut = max(window - len(pinned), 0)
                        candidates = [*rest[:cut], *pinned, *rest[cut:]]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        reranked = False
        if rerank and self.reranker is not None and candidates:
            to_rerank = candidates[:retrieval_setting("RERANK_CANDIDATES")]
            rerank_pool = self.reranker.rerank(
                query, to_rerank, top_k=min(len(to_rerank), max(top_k * 2, top_k + 4))
            )
            deduped = deduplicate_hits(rerank_pool) if retrieval_setting("RETRIEVAL_DEDUP") else list(rerank_pool)
            lam = retrieval_setting("RETRIEVAL_MMR_LAMBDA")
            candidates = mmr_select(deduped, top_k, lam) if lam < 1.0 else deduped[:top_k]
            reranked = True
        else:
            candidates = deduplicate_hits(candidates)[:top_k]
        candidates = self._append_neighbor_anchors(candidates)

        return RetrievalResult(
            query=query,
            strategy=strategy,
            hits=candidates,
            used_filter=applied_filter,
            reranked=reranked,
        )
