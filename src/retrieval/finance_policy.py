"""Finance retrieval policy (finance-query-pipeline, group 4).

`finance_search(plan, retriever, store, k)` retrieves the passages that answer a planned finance question. Callers
gate it behind `FINANCE_PIPELINE=1`; this module does not read the flag. With the flag on it replaces
`Retriever.search`, so the retriever's comparison split, multipart split and auto year filter are BYPASSED: this
module calls the retriever's primitives (`_hybrid_for_filter`, `_bm25_search`, the reranker; a retriever without them
is searched through its public `search`) and owns year handling,
diversity and period reservation itself (period and comparison plans reuse `reserve_period_hits`).

Steps:
  1. Over-fetch an unfiltered hybrid ranking and, when the plan has boost blocks (letter ranges), the same ranking
     under `{"year": {"$gte": a, "$lte": b}}` for each block (the existing filter form; no new filter key).
  2. Blocks = boost: all rankings go through weighted `reciprocal_rank_fusion` (`FINANCE_YEAR_BOOST_WEIGHT`), so a
     retrospective passage outside the range (a 1994 letter on the 1970s oil crisis) stays reachable. Hard filter
     only when `plan.year_hard` ("in the YYYY letter"). A wide explicit range is one block. A narrow question (every
     explicit year mention spans at most `NARROW_LETTERS` letters, or no explicit years but a known event) has one
     block per mention, a single year Y covering the letters Y and Y+1, plus each event's core letters
     (letter_years); its in-block passages DOMINATE: after the rerank they are served first. Topic terms (oil
     crisis: oil, energy, OPEC) add a lexical ranking. The top `FINANCE_YEAR_KEEP` (3) unfiltered hits that pass the
     financial filter are protected: they stay among the served hits even when the boost ranks in-block passages
     above them, so the range alone never removes the best out-of-range passage (diversity caps still apply).
  3. Passages tagged with the plan's events, sectors, companies or themes, drawn from a deeper ranking
     (`FINANCE_TAG_DEPTH`, inside the blocks of a narrow question), are boosted (an extra fused ranking), and the
     best `FINANCE_TAG_POOL` of them always reach the reranker, so a tagged passage the shallow ranking misses (the
     2008 letter's crisis passages) still competes.
  4. Keep financial passages only (tables only for numeric plans). A passage with no annotation counts as
     financial, so an empty or partial store gives today's candidates.
  5. Rerank, de-duplicate, then select k (default `llm_context_passages()`, what the reader context takes). Only a
     question that spans several letters (an explicit range of `WIDE_LETTERS` or more letters, or a trend /
     corpus_wide plan without explicit years) is served diverse: at most `FINANCE_MAX_PER_LETTER` passages per
     letter (raised to ceil(k / letters of the range) for a short range, so the slots are not left empty) and at
     least `FINANCE_MIN_LETTERS` letters when available; a single year or letter is never capped.
     Period, comparison and trend plans reserve the best hit of each named period first. `pinned` hits (the source
     passages of matching figure rows) are served first, before the k - len(pinned) selected ones.

The served budget k is adaptive by plan kind (`served_budget`): `FINANCE_K_TREND`, `FINANCE_K_CORPUS`,
`FINANCE_K_PERIOD` (period and comparison plans) and `FINANCE_K_SPECIFIC` (specific and numeric plans) when set, else
FINANCE_K / `llm_context_passages()`. The caller caps the prompt by the reader window (`reader_window_chars`). The
reranker takes `rerank_depth()` (`FINANCE_RERANK_CANDIDATES`) candidates.

A candidate source never fails a request: one that raises is skipped and recorded in the debug dict as
`source_errors` (`{source, error_type}`), and one whose data is missing (`source_unavailable`, the summary index) is
skipped with a note in `source_skipped`; the request is served from the remaining candidates.

The debug dict reports the annotation coverage of the candidate pool and `year_filtered_out`, the ids of the
unfiltered ranking outside the year range that the range removed (hard) or ranked lower (boost), so the
evaluation can count gold passages lost to the year filter.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from config import (RETRIEVAL_DEFAULTS, RETRIEVAL_FETCH_K, finance_setting, llm_context_passages, llm_n_ctx,
                    retrieval_setting)
from src.knowledge import KnowledgeStore
from src.query.plan import (COMPARISON, CORPUS_WIDE, FIRST_LETTER_YEAR, LAST_LETTER_YEAR, NUMERIC, PERIOD, SPECIFIC,
                            TREND, QueryPlan)
from src.retrieval.bm25 import _meta_matches
from src.retrieval.retriever import (
    deduplicate_hits,
    explicit_periods,
    reciprocal_rank_fusion,
    reserve_period_hits,
)
from src.storage import SearchHit

log = logging.getLogger(__name__)

DIVERSE_KINDS = (TREND, CORPUS_WIDE)       # served diverse when the question names no years
RESERVE_KINDS = (PERIOD, COMPARISON, TREND)
NARROW_LETTERS = 2  # an explicit year mention of at most this many letters is narrow: its letters dominate
WIDE_LETTERS = 3    # a question spanning at least this many letters is served diverse

Block = Tuple[int, int]

# Candidate-source registry (finance-recall-75): (setting name, callable) pairs; the callable is
# `(plan, store, retriever, query, k) -> List[SearchHit]`. A source runs only when its setting is non-zero and its hits
# join the candidate pool before the cross-encoder rerank. The modules below list their own sources; none is registered yet.
Source = Callable[..., List[SearchHit]]
EXTRA_SOURCES: List[Tuple[str, Source]] = []


def registered_sources() -> List[Tuple[str, Source]]:
    from src.retrieval import knowledge_source, summary_source
    return [*summary_source.SOURCES, *knowledge_source.SOURCES, *EXTRA_SOURCES]


def candidate_sources() -> List[Tuple[str, Source]]:
    """The registered sources whose setting is on."""
    return [(name, fn) for name, fn in registered_sources() if finance_setting(name)]


def source_unavailable(name: str, source: Source) -> Optional[str]:
    """A note when the registered source `source` (setting `name`) cannot run because its data is missing, else None.

    It is skipped with the note, not counted as an error. Only the summary source has data of its own: its index is
    git-ignored, so a fresh clone serves without it. A source that merely shares the setting name (a test double) is
    not checked.
    """
    from src.retrieval import summary_source  # lazy, like registered_sources(): faiss loads only when needed
    if dict(summary_source.SOURCES).get(name) is source:
        from src.storage.faiss_store import FaissStore
        from src.storage.index_manifest import MANIFEST_FILE
        directory = summary_source.index_dir()
        missing = [f for f in (MANIFEST_FILE, FaissStore.INDEX_FILE, FaissStore.META_JSON_FILE)
                   if not (directory / f).is_file()]
        if missing:
            return f"summary index missing: {', '.join(missing)} not in {directory}"
    return None


def kind_budget(plan: QueryPlan) -> int:
    """The adaptive served budget of the plan's kind (FINANCE_K_TREND / _CORPUS / _PERIOD / _SPECIFIC), 0 = not set."""
    if plan.kind == TREND:
        return max(0, finance_setting("FINANCE_K_TREND"))
    if plan.kind == CORPUS_WIDE:
        return max(0, finance_setting("FINANCE_K_CORPUS"))
    if plan.kind in (PERIOD, COMPARISON):
        return max(0, finance_setting("FINANCE_K_PERIOD"))
    if plan.kind in (SPECIFIC, NUMERIC):
        return max(0, finance_setting("FINANCE_K_SPECIFIC"))
    return 0


def served_budget(plan: QueryPlan) -> int:
    """Passages the plan is served: its kind's setting, else FINANCE_K, else `llm_context_passages()`."""
    return kind_budget(plan) or finance_setting("FINANCE_K") or llm_context_passages()


def rerank_depth() -> int:
    """Candidates the finance path sends to the cross-encoder: FINANCE_RERANK_CANDIDATES. While that setting is
    neither set in the environment nor changed from today's value, the global RERANK_CANDIDATES applies as before."""
    depth = finance_setting("FINANCE_RERANK_CANDIDATES")
    if not os.environ.get("FINANCE_RERANK_CANDIDATES", "").strip() and depth == RETRIEVAL_DEFAULTS["RERANK_CANDIDATES"]:
        depth = retrieval_setting("RERANK_CANDIDATES")
    return max(1, depth)


def reader_window_chars(max_new_tokens: int) -> int:
    """Characters the reader prompt may take: the window (`llm_n_ctx()`) minus the answer budget, at
    LLM_CHARS_PER_TOKEN (4) characters per token."""
    return int(max(0, llm_n_ctx() - max_new_tokens) * retrieval_setting("LLM_CHARS_PER_TOKEN"))


def _year_filter(rng: Tuple[int, int]) -> Dict[str, Any]:
    a, b = sorted(rng)
    return {"year": {"$gte": a, "$lte": b}}


def _width(span: Block) -> int:
    return abs(span[1] - span[0]) + 1


def _spans(plan: QueryPlan) -> List[Block]:
    """The question's explicit year mentions; a plan built without them counts its year range as one."""
    if plan.year_spans:
        return [tuple(sorted(s)) for s in plan.year_spans]  # type: ignore[misc]
    if plan.year_range and plan.year_source != "event":
        return [tuple(sorted(plan.year_range))]  # type: ignore[list-item]
    return []


def _merge(blocks: Sequence[Block]) -> List[Block]:
    out: List[List[int]] = []
    for a, b in sorted(blocks):
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _boost(plan: QueryPlan, store: KnowledgeStore) -> Tuple[List[Block], bool]:
    """(blocks, narrow): the letter ranges the ranking boosts, and whether their passages dominate it."""
    known = store.events()
    events = [(min(y), max(y)) for y in ((known.get(e) or {}).get("letter_years") for e in plan.events) if y]
    spans = _spans(plan)
    if plan.year_hard and plan.year_range:
        return [tuple(sorted(plan.year_range))], True  # type: ignore[list-item]
    if spans and all(_width(s) <= NARROW_LETTERS for s in spans):
        return _merge([(a, b + 1) if a == b else (a, b) for a, b in spans] + events), True  # "in Y": letters Y, Y+1
    if spans:
        return [(min(a for a, _ in spans), max(b for _, b in spans))], False
    if events:
        return _merge(events), True  # the event's core letters
    if plan.year_range:  # alias years without a store event (oil crisis): a plain boost
        return [tuple(sorted(plan.year_range))], False  # type: ignore[list-item]
    return [], False


def _diverse(plan: QueryPlan) -> bool:
    """Only a question that spans several letters; never a single year or a single letter."""
    spans = _spans(plan)
    if spans:
        return not plan.year_hard and any(_width(s) >= WIDE_LETTERS for s in spans)
    return plan.kind in DIVERSE_KINDS


def _in_blocks(hit: SearchHit, blocks: Sequence[Block]) -> bool:
    return any(_meta_matches(hit.metadata, _year_filter(b)) for b in blocks)


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


def _letter_cap(plan: QueryPlan, k: int) -> int:
    """FINANCE_MAX_PER_LETTER, raised so k passages fit into the letters of the question's range (a 5-letter range
    with k = 16 takes 4 per letter); a question without explicit years keeps the setting."""
    cap = max(1, finance_setting("FINANCE_MAX_PER_LETTER"))
    spans = _spans(plan)
    if not spans:
        return cap
    lo = max(FIRST_LETTER_YEAR, min(a for a, _ in spans))
    hi = min(LAST_LETTER_YEAR, max(b for _, b in spans))
    letters = max(1, hi - lo + 1)
    return max(cap, -(-k // letters))


def _diverse_ok(hits: Sequence[SearchHit], letters_before: int, cap: Optional[int] = None) -> bool:
    counts: Dict[Any, int] = {}
    for hit in hits:
        counts[_letter(hit)] = counts.get(_letter(hit), 0) + 1
    cap = max(1, finance_setting("FINANCE_MAX_PER_LETTER")) if cap is None else cap
    return (max(counts.values(), default=0) <= cap
            and len(counts) >= min(letters_before, finance_setting("FINANCE_MIN_LETTERS")))


def _protect(hits: List[SearchHit], pool: Sequence[SearchHit], protected: Sequence[str], k: int,
             diverse: bool, cap: Optional[int] = None) -> List[SearchHit]:
    """Put each protected pool hit that was not selected back, replacing the lowest-ranked unprotected hit."""
    order = {h.id: i for i, h in enumerate(pool)}
    keep = set(protected)
    for hit in pool:
        if hit.id not in keep or hit.id in {h.id for h in hits}:
            continue
        if len(hits) < k and (not diverse or _diverse_ok([*hits, hit], len({_letter(h) for h in hits}), cap)):
            hits = [*hits, hit]
            continue
        for victim in sorted((h for h in hits if h.id not in keep), key=lambda h: -order[h.id]):
            trial = [h for h in hits if h is not victim] + [hit]
            if not diverse or _diverse_ok(trial, len({_letter(h) for h in hits}), cap):
                hits = trial
                break
    return sorted(hits, key=lambda h: order[h.id])


def _select(ranked: Sequence[SearchHit], plan: QueryPlan, k: int, protected: Sequence[str] = (),
            diverse: Optional[bool] = None) -> List[SearchHit]:
    """Pick k hits: period reservation first, then (diverse questions) the per-letter cap and the minimum letters."""
    pool = list(ranked)
    if k <= 0:
        return []
    if plan.kind in RESERVE_KINDS:
        periods = explicit_periods(plan.query)
        if not periods and plan.year_range and plan.kind != TREND:
            periods = [_year_filter(plan.year_range)]
        if periods:  # reserved hits first; the rest stays available to the per-letter cap below
            reserved = reserve_period_hits(pool, periods, k)
            taken = {h.id for h in reserved}
            pool = reserved + [h for h in pool if h.id not in taken]
    if not (_diverse(plan) if diverse is None else diverse):
        return _protect(pool[:k], pool, protected, k, diverse=False)

    cap = _letter_cap(plan, k)
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
    return _protect(sorted(picked, key=lambda h: order[h.id]), pool, protected, k, diverse=True, cap=cap)


def _hybrid_ranking(retriever: Any, query: str, fetch_k: int, where: Optional[Dict[str, Any]]) -> List[SearchHit]:
    """The retriever's hybrid ranking primitive; a retriever without it falls back to its public search
    (hybrid, no rerank, no auto year filter), so the finance filtering below still applies."""
    primitive = getattr(retriever, "_hybrid_for_filter", None)
    if primitive is not None:
        return list(primitive(query, fetch_k, where))
    result = retriever.search(query=query, strategy="hybrid", top_k=fetch_k, fetch_k=fetch_k, rerank=False,
                              where=where, auto_year_filter=False)
    return list(result.hits)


def _has_entities(plan: QueryPlan) -> bool:
    return bool(plan.events or plan.sectors or plan.companies or plan.themes)


def finance_search(
    plan: QueryPlan,
    retriever: Any,
    store: Optional[KnowledgeStore] = None,
    k: Optional[int] = None,
    pinned: Sequence[SearchHit] = (),
) -> Tuple[List[SearchHit], Dict[str, Any]]:
    """Retrieve for a planned finance question; returns (hits, debug).

    `k` defaults to `served_budget(plan)`: the plan kind's FINANCE_K_* setting, else FINANCE_K, else
    `llm_context_passages()`. `pinned` hits are served first (the source passages of matching figure rows), whatever
    their scope or table flag.
    """
    store = store if store is not None else KnowledgeStore.empty()
    k = int(k) if k else served_budget(plan)
    query = plan.financial_query or plan.query
    fetch_k = max(RETRIEVAL_FETCH_K, k * finance_setting("FINANCE_OVERFETCH"))

    unfiltered = _hybrid_ranking(retriever, query, fetch_k, None)
    blocks, narrow = _boost(plan, store)
    year_filtered_out: List[str] = []
    rankings: List[Sequence[SearchHit]] = []
    weights: List[float] = []
    if blocks:
        boost = float(max(1, finance_setting("FINANCE_YEAR_BOOST_WEIGHT")))
        for block in blocks:
            where = _year_filter(block)
            rankings.append(_hybrid_ranking(retriever, query, fetch_k, where))
            weights.append(1.0 if plan.year_hard else boost)
            if plan.terms and hasattr(retriever, "_bm25_search"):
                terms = list(retriever._bm25_search(" ".join(plan.terms), fetch_k, where))
                if terms:
                    rankings.append(terms)
                    weights.append(1.0)
        if plan.year_hard:
            year_filtered_out = [h.id for h in unfiltered if not _in_blocks(h, blocks)]
        else:
            rankings.append(unfiltered)
            weights.append(1.0)
    else:
        rankings, weights = [unfiltered], [1.0]

    total = sum(len(r) for r in rankings)
    fused = reciprocal_rank_fusion(rankings, top_k=max(total, 1), weights=weights) if total else []
    if blocks and not plan.year_hard:
        new_rank = {h.id: i for i, h in enumerate(fused)}
        year_filtered_out = [h.id for i, h in enumerate(unfiltered)
                             if not _in_blocks(h, blocks) and new_rank.get(h.id, len(fused) + 1) > i]

    # entity tags: candidates from a deeper ranking (inside the blocks of a narrow question) join the pool
    combined = list(fused)
    if _has_entities(plan) and len(store) and fused:
        seen = {h.id for h in combined}
        depth = max(fetch_k, finance_setting("FINANCE_TAG_DEPTH"))
        for where in ([_year_filter(b) for b in blocks] if (blocks and narrow) else [None]):
            for hit in _hybrid_ranking(retriever, query, depth, where):
                if hit.id not in seen:
                    combined.append(hit)
                    seen.add(hit.id)
    events_tagged, tagged = _tag_hits(plan, store, combined)
    if tagged:  # entity boost: the tagged passages as one more fused ranking
        by_id = {h.id: h for h in combined}
        weight = float(max(0, finance_setting("FINANCE_ENTITY_BOOST_WEIGHT")))
        if weight:
            fused = reciprocal_rank_fusion([fused, [by_id[p] for p in tagged]],
                                           top_k=len({h.id for h in fused} | set(tagged)), weights=[1.0, weight])

    if finance_setting("FINANCE_SUMMARY_FUSE_WEIGHT"):  # promote pooled ids whose summary matches (0 = unchanged)
        try:
            from src.retrieval.summary_source import fuse_summary
            fused = fuse_summary(fused, plan, store, retriever, query)
        except Exception:  # noqa: BLE001 - a fusion failure never fails the request
            pass
    source_hits: List[SearchHit] = []  # extra candidate sources: they join the pool before the rerank
    source_counts: Dict[str, int] = {}
    source_errors: List[Dict[str, str]] = []   # a source that raised: skipped, the request is served without it
    source_skipped: List[Dict[str, str]] = []  # a source whose data is missing: skipped with a note
    seen_ids = {h.id for h in fused}
    for name, source in candidate_sources():
        try:
            note = source_unavailable(name, source)
            if note:
                source_skipped.append({"source": name, "note": note})
                continue
            got = [h for h in source(plan, store, retriever, query, k) if h.id not in seen_ids]
        except Exception as exc:  # noqa: BLE001 - no candidate source may fail the request
            log.warning("finance candidate source %s skipped: %s: %s", name, type(exc).__name__, exc)
            source_errors.append({"source": name, "error_type": type(exc).__name__})
            continue
        source_counts[name] = len(got)
        for h in got:
            seen_ids.add(h.id)
            source_hits.append(h)
    if source_hits:
        fused = [*fused, *source_hits]
        combined = [*combined, *source_hits]

    pool_ids = [h.id for h in fused]
    numeric = plan.kind == NUMERIC
    kept = [h for h in fused if store.is_financial(h.id) and (numeric or not store.is_table(h.id))]
    kept_ids = {h.id for h in kept}
    dropped = [h.id for h in fused if h.id not in kept_ids]

    reranked = False
    window_ids = [h.id for h in kept]  # no reranker: every kept candidate is in play
    reranker = getattr(retriever, "reranker", None)
    injected: List[SearchHit] = []
    if reranker is not None and kept:
        window = kept[:rerank_depth()]
        in_window = {h.id for h in window}
        by_id = {h.id: h for h in combined}
        injected = [by_id[p] for p in tagged if p not in in_window and store.is_financial(p)
                    and (numeric or not store.is_table(p))][:max(0, finance_setting("FINANCE_TAG_POOL"))]
        in_window |= {h.id for h in injected}
        by_source = [h for h in source_hits if h.id not in in_window and store.is_financial(h.id)
                     and (numeric or not store.is_table(h.id))]
        injected = injected + by_source
        window = window + injected
        window_ids = [h.id for h in window]
        kept = list(reranker.rerank(query, window, top_k=len(window)))
        reranked = True
    if retrieval_setting("RETRIEVAL_DEDUP"):
        kept = deduplicate_hits(kept)
    if blocks and narrow and not plan.year_hard:  # a narrow question: its in-block passages are served first
        kept = [h for h in kept if _in_blocks(h, blocks)] + [h for h in kept if not _in_blocks(h, blocks)]

    stage_ranked = [h.id for h in kept]  # after rerank, dedup and the narrow-block reorder, before select
    protected: List[str] = []
    if blocks and not plan.year_hard:
        in_pool = {h.id for h in kept}
        protected = [h.id for h in unfiltered if h.id in in_pool][:max(0, finance_setting("FINANCE_YEAR_KEEP"))]
    pins = list({h.id: h for h in pinned}.values())[:k]
    pin_ids = {h.id for h in pins}
    diverse = _diverse(plan)
    hits = pins + _select([h for h in kept if h.id not in pin_ids], plan, k - len(pins),
                          [p for p in protected if p not in pin_ids], diverse)
    lo_hi = (min(a for a, _ in blocks), max(b for _, b in blocks)) if blocks else None
    debug = {
        "coverage": store.coverage(pool_ids),
        "annotated": sum(1 for pid in pool_ids if store.annotation(pid) is not None),
        "pool_size": len(pool_ids),
        "year_range": list(lo_hi) if lo_hi else None,
        "year_blocks": [list(b) for b in blocks],
        "narrow": bool(blocks and narrow),
        "diverse": diverse,
        "year_hard": bool(plan.year_hard),
        "year_filtered_out": year_filtered_out,
        "event_tagged": len(events_tagged),
        "entity_tagged": len(tagged),
        "tag_injected": len(injected),
        "pinned": [h.id for h in pins],
        "dropped_not_financial": dropped,
        "source_candidates": source_counts,
        "source_errors": source_errors,
        "source_skipped": source_skipped,
        "stage_pool": pool_ids,                 # every candidate before the financial filter (loss analysis)
        "stage_window": window_ids,             # the ids the reranker saw
        "stage_ranked": stage_ranked,           # reranked order, before the budget and per-letter cap
        "reranked": reranked,
        "k": k,
        "fetch_k": fetch_k,
        "letters": len({_letter(h) for h in hits}),
    }
    return hits, debug
