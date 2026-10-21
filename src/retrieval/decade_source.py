"""Finance candidate source: decade-block searches for corpus-wide and trend plans (finance-recall-75-r4, group 2).

`SOURCES` lists (setting name, callable) pairs; `knowledge_source.SOURCES` appends them last, so
`finance_policy.candidate_sources()` runs a callable `(plan, store, retriever, query, k) -> List[SearchHit]` only when
its `finance_setting(name)` is non-zero, and the hits join the candidate pool before the cross-encoder rerank.

A question about the whole collection or a trend ("how did Buffett's view of X change over the decades") needs passages
from every era, but one unfiltered ranking is dominated by the few letters that use the question's words most, so the
gold passages of the other decades never reach the pool. FINANCE_DECADE_BLOCKS=M runs one hybrid search per block
(BLOCKS: 1977-1989, 1990-1999, 2000-2009, 2010-2024) with the question's financial core
(`multiquery_source.financial_core`), under the year filter the finance policy uses (`finance_policy._year_filter`),
and adds from each block its best M financial passages that the pool does not hold yet, at most PER_LETTER per letter
(raised when the block has fewer than M letters), the best of every block first.

It fires only for CORPUS_WIDE and TREND plans that have no year range or whose year range spans more than WIDE_RANGE
letters; a plan with such a range searches only the blocks inside it (cut to the range). `pooled` is what
`finance_search` and every source registered before this one hold, recomputed exactly (`multiquery_source.pooled` plus
the multi-query sources that are on), so the M places of a block are spent on new candidates. No LLM runs at query time.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple

from config import finance_setting
from src.knowledge import KnowledgeStore
from src.query.plan import CORPUS_WIDE, FIRST_LETTER_YEAR, LAST_LETTER_YEAR, TREND, QueryPlan
from src.retrieval import finance_policy, multiquery_source
from src.retrieval.finance_policy import _hybrid_ranking, _width, _year_filter
from src.storage import SearchHit

BLOCKS: Tuple[Tuple[int, int], ...] = ((FIRST_LETTER_YEAR, 1989), (1990, 1999), (2000, 2009), (2010, LAST_LETTER_YEAR))
KINDS = (CORPUS_WIDE, TREND)
WIDE_RANGE = 15          # a plan with a year range fires only when the range spans more than this many letters
DECADE_DEPTH = 60        # hits fetched per block before the pool's ids are skipped
PER_LETTER = 1           # passages per letter and block, at most (raised when the block has fewer than M letters)

_RUNNING = threading.local()  # set while `pooled` runs the registered sources: this source must not call itself


def _ks():
    from src.retrieval import knowledge_source  # lazy: knowledge_source imports this module for its SOURCES
    return knowledge_source


def applies(plan: QueryPlan) -> bool:
    """A corpus-wide or trend plan with no year range, or a year range of more than WIDE_RANGE letters."""
    return plan.kind in KINDS and (not plan.year_range or _width(plan.year_range) > WIDE_RANGE)


def blocks_of(plan: QueryPlan) -> List[Tuple[int, int]]:
    """The BLOCKS to search: all of them, or (with a year range) those inside it, each cut to the range."""
    lo, hi = sorted(plan.year_range) if plan.year_range else (FIRST_LETTER_YEAR, LAST_LETTER_YEAR)
    cut = [(max(a, lo), min(b, hi)) for a, b in BLOCKS]
    return [(a, b) for a, b in cut if a <= b]


def pooled(plan: QueryPlan, store: KnowledgeStore, retriever: Any, query: str, k: int) -> Set[str]:
    """The ids the pool holds before this source runs. `multiquery_source.pooled` covers `finance_search` and the
    sources before the multi-query ones; the multi-query sources (when on) run before this one too."""
    _RUNNING.on = True  # multiquery_source.pooled runs every registered source up to the multi-query ones, this one too
    try:
        out = set(multiquery_source.pooled(plan, store, retriever, query, k))
    finally:
        _RUNNING.on = False
    after = False
    for name, source in finance_policy.candidate_sources():
        if source is decade_candidates:
            break
        after = after or source in multiquery_source.FAMILY  # the multi-query and plan-query sources
        if not after or finance_policy.source_unavailable(name, source):
            continue
        try:
            out.update(h.id for h in source(plan, store, retriever, query, k))
        except Exception:  # noqa: BLE001 - finance_search skips a failing source too
            continue
    return out


def _spread(ids: Sequence[str], m: int, letter: Callable[[str], int]) -> List[str]:
    """The first m ids in order, at most PER_LETTER of each letter (raised so m ids fit into the letters there are)."""
    per_letter = max(PER_LETTER, -(-m // max(1, len({letter(pid) for pid in ids}))))
    counts: Dict[int, int] = {}
    out: List[str] = []
    for pid in ids:
        if len(out) >= m:
            break
        y = letter(pid)
        if counts.get(y, 0) < per_letter:
            counts[y] = counts.get(y, 0) + 1
            out.append(pid)
    return out


def decade_candidates(plan: QueryPlan, store: KnowledgeStore, retriever: Any, query: str, k: int) -> List[SearchHit]:
    """The best FINANCE_DECADE_BLOCKS new passages of every decade block, for a corpus-wide or trend plan."""
    m = max(0, finance_setting("FINANCE_DECADE_BLOCKS"))
    if m <= 0 or not applies(plan) or getattr(_RUNNING, "on", False):
        return []
    ks = _ks()
    docs = ks._docs(retriever)
    spans = blocks_of(plan)
    if not docs or not spans:
        return []
    core = multiquery_source.financial_core(plan)
    with ThreadPoolExecutor(max_workers=min(multiquery_source.MQ_WORKERS, len(spans))) as pool:
        searches = [pool.submit(_hybrid_ranking, retriever, core, DECADE_DEPTH, _year_filter(span)) for span in spans]
        skip = pooled(plan, store, retriever, query, k)  # here, in this thread: `_RUNNING` is thread-local
        rankings = [search.result() for search in searches]
    letter = lambda pid: ks.letter_of(pid, docs)  # noqa: E731
    rows = []
    tables = finance_policy.tables_allowed(plan)  # a corpus-wide / trend plan is never numeric: figure questions only
    for ranking in rankings:
        ids = [h.id for h in ranking if h.id not in skip and h.id in docs and store.is_financial(h.id)
               and (tables or not store.is_table(h.id))]
        rows.append(_spread(list(dict.fromkeys(ids)), m, letter))
    picked = [pid for row in zip_longest(*rows) for pid in row if pid is not None]  # the best of every block first
    return ks._hits(list(dict.fromkeys(picked)), docs)


SOURCES: List[Tuple[str, Callable]] = [("FINANCE_DECADE_BLOCKS", decade_candidates)]
