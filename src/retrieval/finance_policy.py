"""Finance retrieval policy (finance-query-pipeline, group 4).

`finance_search(plan, retriever, store, k)` retrieves the passages that answer a planned finance question. Callers
gate it behind `FINANCE_PIPELINE=1`; this module does not read the flag. With the flag on it replaces
`Retriever.search`, so the retriever's comparison split, multipart split and auto year filter are BYPASSED: this
module calls the retriever's primitives (`_hybrid_for_filter`, `_bm25_search`, the reranker; a retriever without them
is searched through its public `search`) and owns year handling,
diversity and period reservation itself (period and comparison plans reuse `reserve_period_hits`).

Steps:
  1. Over-fetch an unfiltered hybrid ranking and, when the plan has a year range, the same ranking under
     `{"year": {"$gte": a, "$lte": b}}` (the existing filter form; no new filter key).
  2. Year range = boost: both rankings go through weighted `reciprocal_rank_fusion` (`FINANCE_YEAR_BOOST_WEIGHT`),
     so a retrospective passage outside the range (a 1994 letter on the 1970s oil crisis) stays reachable.
     Hard filter only when `plan.year_hard` ("in the YYYY letter"). Events use the event's letter_years as the
     boost range when the plan has none. Topic terms (oil crisis: oil, energy, OPEC) add a lexical ranking.
     The top `FINANCE_YEAR_KEEP` (3) unfiltered hits that pass the financial filter are protected: they stay among
     the served hits even when the boost ranks in-range passages above them, so the range alone never removes the
     best out-of-range passage (diversity caps still apply to them).
  3. Passages tagged with the plan's events, sectors, companies or themes are boosted (an extra fused ranking);
     event-tagged passages go to the front of the rerank pool.
  4. Keep financial passages only (tables only for numeric plans). A passage with no annotation counts as
     financial, so an empty or partial store gives today's candidates.
  5. Rerank, de-duplicate, then select k: trend, period and corpus_wide plans serve at most
     `FINANCE_MAX_PER_LETTER` passages per letter and at least `FINANCE_MIN_LETTERS` letters when available;
     period and comparison plans reserve the best hit of each named period first.

The debug dict reports the annotation coverage of the candidate pool and `year_filtered_out`, the ids of the
unfiltered ranking outside the year range that the range removed (hard) or ranked lower (boost), so the
evaluation can count gold passages lost to the year filter.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import RETRIEVAL_FETCH_K, finance_setting, retrieval_setting
from src.knowledge import KnowledgeStore
from src.query.plan import COMPARISON, CORPUS_WIDE, NUMERIC, PERIOD, TREND, QueryPlan
from src.retrieval.bm25 import _meta_matches
from src.retrieval.retriever import (
    deduplicate_hits,
    explicit_periods,
    reciprocal_rank_fusion,
    reserve_period_hits,
)
from src.storage import SearchHit

DIVERSE_KINDS = (TREND, PERIOD, CORPUS_WIDE)
RESERVE_KINDS = (PERIOD, COMPARISON)


def _year_filter(rng: Tuple[int, int]) -> Dict[str, Any]:
    a, b = sorted(rng)
    return {"year": {"$gte": a, "$lte": b}}


def _boost_range(plan: QueryPlan, store: KnowledgeStore) -> Optional[Tuple[int, int]]:
    """The plan's year range, else the span of its events' letter_years."""
    if plan.year_range:
        return tuple(plan.year_range)  # type: ignore[return-value]
    if plan.events:
        known = store.events()
        years = [y for e in plan.events for y in (known.get(e) or {}).get("letter_years", [])]
        if years:
            return min(years), max(years)
    return None


def _letter(hit: SearchHit) -> Any:
    return hit.metadata.get("year")


def _tag_hits(plan: QueryPlan, store: KnowledgeStore, pool: Sequence[SearchHit]) -> Tuple[List[str], List[str]]:
    """(ids tagged with a plan event, ids tagged with a plan event/sector/company/theme)."""
    wanted = {"events": set(plan.events), "sectors": set(plan.sectors),
              "companies": set(plan.companies), "themes": set(plan.themes)}
    events_tagged: List[str] = []
    scored: List[Tuple[int, int, str]] = []
    for rank, hit in enumerate(pool):
        ann = store.annotation(hit.id)
        if not ann:
            continue
        overlap = {f: len(values.intersection(ann.get(f) or ())) for f, values in wanted.items() if values}
        if overlap.get("events"):
            events_tagged.append(hit.id)
        if any(overlap.values()):
            scored.append((-sum(overlap.values()), rank, hit.id))
    return events_tagged, [pid for _, _, pid in sorted(scored)]


def _diverse_ok(hits: Sequence[SearchHit], letters_before: int) -> bool:
    counts: Dict[Any, int] = {}
    for hit in hits:
        counts[_letter(hit)] = counts.get(_letter(hit), 0) + 1
    return (max(counts.values(), default=0) <= max(1, finance_setting("FINANCE_MAX_PER_LETTER"))
            and len(counts) >= min(letters_before, finance_setting("FINANCE_MIN_LETTERS")))


def _protect(hits: List[SearchHit], pool: Sequence[SearchHit], protected: Sequence[str], k: int,
             diverse: bool) -> List[SearchHit]:
    """Put each protected pool hit that was not selected back, replacing the lowest-ranked unprotected hit."""
    order = {h.id: i for i, h in enumerate(pool)}
    keep = set(protected)
    for hit in pool:
        if hit.id not in keep or hit.id in {h.id for h in hits}:
            continue
        if len(hits) < k and (not diverse or _diverse_ok([*hits, hit], len({_letter(h) for h in hits}))):
            hits = [*hits, hit]
            continue
        for victim in sorted((h for h in hits if h.id not in keep), key=lambda h: -order[h.id]):
            trial = [h for h in hits if h is not victim] + [hit]
            if not diverse or _diverse_ok(trial, len({_letter(h) for h in hits})):
                hits = trial
                break
    return sorted(hits, key=lambda h: order[h.id])


def _select(ranked: Sequence[SearchHit], plan: QueryPlan, k: int, protected: Sequence[str] = ()) -> List[SearchHit]:
    """Pick k hits: period reservation first, then the per-letter cap and the minimum number of letters."""
    pool = list(ranked)
    if plan.kind in RESERVE_KINDS:
        periods = explicit_periods(plan.query)
        if not periods and plan.year_range:
            periods = [_year_filter(plan.year_range)]
        if periods:
            pool = reserve_period_hits(pool, periods, k)
    if plan.kind not in DIVERSE_KINDS:
        return _protect(pool[:k], pool, protected, k, diverse=False)

    cap = max(1, finance_setting("FINANCE_MAX_PER_LETTER"))
    min_letters = finance_setting("FINANCE_MIN_LETTERS")
    picked: List[SearchHit] = []
    counts: Dict[Any, int] = {}

    def take(hit: SearchHit) -> None:
        picked.append(hit)
        counts[_letter(hit)] = counts.get(_letter(hit), 0) + 1

    for hit in pool:  # first pass: the best hit of each new letter until min_letters letters are served
        if len(picked) >= k or len(counts) >= min_letters:
            break
        if _letter(hit) not in counts:
            take(hit)
    chosen = {h.id for h in picked}
    for hit in pool:  # second pass: fill by rank under the cap
        if len(picked) >= k:
            break
        if hit.id not in chosen and counts.get(_letter(hit), 0) < cap:
            take(hit)
            chosen.add(hit.id)
    order = {h.id: i for i, h in enumerate(pool)}
    return _protect(sorted(picked, key=lambda h: order[h.id]), pool, protected, k, diverse=True)


def _hybrid_ranking(retriever: Any, query: str, fetch_k: int, where: Optional[Dict[str, Any]]) -> List[SearchHit]:
    """The retriever's hybrid ranking primitive; a retriever without it falls back to its public search
    (hybrid, no rerank, no auto year filter), so the finance filtering below still applies."""
    primitive = getattr(retriever, "_hybrid_for_filter", None)
    if primitive is not None:
        return list(primitive(query, fetch_k, where))
    result = retriever.search(query=query, strategy="hybrid", top_k=fetch_k, fetch_k=fetch_k, rerank=False,
                              where=where, auto_year_filter=False)
    return list(result.hits)


def finance_search(
    plan: QueryPlan,
    retriever: Any,
    store: Optional[KnowledgeStore] = None,
    k: Optional[int] = None,
) -> Tuple[List[SearchHit], Dict[str, Any]]:
    """Retrieve for a planned finance question; returns (hits, debug). `k` defaults to FINANCE_K (12)."""
    store = store if store is not None else KnowledgeStore.empty()
    k = int(k) if k else finance_setting("FINANCE_K")
    query = plan.financial_query or plan.query
    fetch_k = max(RETRIEVAL_FETCH_K, k * finance_setting("FINANCE_OVERFETCH"))

    unfiltered = _hybrid_ranking(retriever, query, fetch_k, None)
    rng = _boost_range(plan, store)
    year_filtered_out: List[str] = []
    rankings: List[Sequence[SearchHit]] = []
    weights: List[float] = []
    if rng:
        where = _year_filter(rng)
        filtered = _hybrid_ranking(retriever, query, fetch_k, where)
        if plan.terms and hasattr(retriever, "_bm25_search"):
            terms = list(retriever._bm25_search(" ".join(plan.terms), fetch_k, where))
        else:
            terms = []
        if plan.year_hard:
            rankings, weights = [filtered], [1.0]
            year_filtered_out = [h.id for h in unfiltered if not _meta_matches(h.metadata, where)]
        else:
            rankings, weights = [filtered, unfiltered], [float(max(1, finance_setting("FINANCE_YEAR_BOOST_WEIGHT"))), 1.0]
        if terms:
            rankings.append(terms)
            weights.append(1.0)
    else:
        rankings, weights = [unfiltered], [1.0]

    total = sum(len(r) for r in rankings)
    fused = reciprocal_rank_fusion(rankings, top_k=max(total, 1), weights=weights) if total else []
    if rng and not plan.year_hard:
        new_rank = {h.id: i for i, h in enumerate(fused)}
        year_filtered_out = [h.id for i, h in enumerate(unfiltered)
                             if not _meta_matches(h.metadata, _year_filter(rng))
                             and new_rank.get(h.id, len(fused) + 1) > i]

    events_tagged, tagged = _tag_hits(plan, store, fused)
    if tagged:  # entity boost: the tagged passages as one more fused ranking
        by_id = {h.id: h for h in fused}
        weight = float(max(0, finance_setting("FINANCE_ENTITY_BOOST_WEIGHT")))
        if weight:
            fused = reciprocal_rank_fusion([fused, [by_id[p] for p in tagged]], top_k=len(fused),
                                           weights=[1.0, weight])
    if events_tagged:  # events: tagged passages first, stable
        first = set(events_tagged)
        fused = [h for h in fused if h.id in first] + [h for h in fused if h.id not in first]

    pool_ids = [h.id for h in fused]
    numeric = plan.kind == NUMERIC
    kept = [h for h in fused if store.is_financial(h.id) and (numeric or not store.is_table(h.id))]
    kept_ids = {h.id for h in kept}
    dropped = [h.id for h in fused if h.id not in kept_ids]

    reranked = False
    reranker = getattr(retriever, "reranker", None)
    if reranker is not None and kept:
        window = kept[:retrieval_setting("RERANK_CANDIDATES")]
        kept = list(reranker.rerank(query, window, top_k=len(window)))
        reranked = True
    if retrieval_setting("RETRIEVAL_DEDUP"):
        kept = deduplicate_hits(kept)

    protected: List[str] = []
    if rng and not plan.year_hard:
        in_pool = {h.id for h in kept}
        protected = [h.id for h in unfiltered if h.id in in_pool][:max(0, finance_setting("FINANCE_YEAR_KEEP"))]
    hits = _select(kept, plan, k, protected)
    debug = {
        "coverage": store.coverage(pool_ids),
        "annotated": sum(1 for pid in pool_ids if store.annotation(pid) is not None),
        "pool_size": len(pool_ids),
        "year_range": list(rng) if rng else None,
        "year_hard": bool(plan.year_hard),
        "year_filtered_out": year_filtered_out,
        "event_tagged": len(events_tagged),
        "entity_tagged": len(tagged),
        "dropped_not_financial": dropped,
        "reranked": reranked,
        "k": k,
        "fetch_k": fetch_k,
        "letters": len({_letter(h) for h in hits}),
    }
    return hits, debug
