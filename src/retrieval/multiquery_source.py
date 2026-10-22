"""Finance candidate source: multi-query searches for view questions (finance-recall-75-r3, group 2).

`SOURCES` lists (setting name, callable) pairs; `knowledge_source.SOURCES` appends them, so
`finance_policy.candidate_sources()` runs a callable `(plan, store, retriever, query, k) -> List[SearchHit]` only when
its `finance_setting(name)` is non-zero, and the hits join the candidate pool before the cross-encoder rerank.

A view question ("what does Buffett think of ...", VIEW_CUE, not a numeric plan) often names its topic in words the
passages do not use, so its gold passages never reach the pool. The source runs extra hybrid searches with the topic
written out, fuses them by RRF and adds the best passages the pool does not hold yet:
  - FINANCE_MULTI_QUERY=N: up to N searches, one per topic of the question: the plan's themes, then the themes
    `knowledge_source.infer_themes` infers, then (MQ_SECTORS) the plan's sectors. Each search is the question's
    financial core (`financial_core`: the finance query without the opinion frame, VIEW_CUE words and Buffett's name)
    plus the top words of the summaries of the topic's tagged passages (THEME_QUERY "keywords", `topic_keywords`, the
    default), its taxonomy description ("description", the first sentence as theme inference embeds it) or both.
  - FINANCE_DIGEST_QUERY=1: when a digest matches the plan's topic (`knowledge_source.digest_topics`, the inferred
    themes only when no named topic has one), one more search with the first DIGEST_LINES sentences of the digest's
    overview as the query.
Every search follows the year logic of `finance_search`: a hard year searches only its range; a boost plan adds one
ranking per block (FINANCE_YEAR_BOOST_WEIGHT) to the unfiltered one. With MQ_SUMMARY each query also searches the
annotation-summary index (`summary_source.summary_ranking`, same year logic). The searches run in parallel threads and
every ranking goes through one weighted RRF. Of the fused ranking the source keeps financial passages (tables only for
numeric plans) that the pool does not hold yet (`pooled`: what `finance_search` and the sources before this one hold,
recomputed exactly, so the cap is not spent on pooled ids), in-range passages first, and returns at most MQ_CAP of
them, at most `knowledge_source.PER_LETTER` per letter (raised when few letters are available).

With both settings on, the multi-query source runs the digest search with its own (one fusion, one cap) and the digest
source returns nothing. No LLM runs at query time.

FINANCE_PLAN_QUERY=N (finance-answer-quality, group 4) rewrites the search from the plan for the questions the
multi-query source leaves alone (`plan_queries`): up to N topic queries (as above) for a question that is not a view
question (or when FINANCE_MULTI_QUERY is off), and for every plan event one query with the event's aliases and its real
years (`event_query`); a plan with topic terms but no store event (an alias such as "oil crisis") gets one query with
those terms. `plan_query_source` searches them like the multi-query source (same year logic, one fusion) and adds at most
PQ_CAP new passages; it skips what the pool and the multi-query sources before it already hold.
"""
from __future__ import annotations

import math
import re
import weakref
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from config import RETRIEVAL_FETCH_K, finance_setting
from src.knowledge import KnowledgeStore
from src.query.plan import NUMERIC, QueryPlan
from src.retrieval import finance_policy, summary_source
from src.retrieval.bm25 import _SEARCH_STOPWORDS, _search_tokens
from src.retrieval.finance_policy import _boost, _has_entities, _hybrid_ranking, _year_filter
from src.retrieval.retriever import reciprocal_rank_fusion
from src.storage import SearchHit

MQ_CAP = 16              # passages the source adds per question, at most
MQ_DEPTH = 60            # hits fetched per search (dense and BM25) before the fusion
MQ_WORKERS = 4           # threads the searches run on
MQ_SECTORS = True        # the plan's sectors are topics too, after the themes
MQ_SUMMARY = False       # each query also searches the annotation-summary index (summary_source.summary_ranking)
THEME_QUERY = "keywords"     # what a topic adds to the core: "description" | "keywords" | "both"
# MQ_SUMMARY and THEME_QUERY were chosen on dev (finance-recall-75-r3, task 2.2): keywords beat the description and
# "both", and the summary rankings only crowded the cap; heldout_v6 did not move under any of them.
KEYWORDS = 8             # summary words per topic ("keywords")
KEYWORD_MIN_DOCS = 3     # ... each in at least this many tagged summaries
DIGEST_LINES = 2         # sentences of the digest overview that make the digest query
PQ_CAP = 16              # passages the plan-query source adds per question, at most (FINANCE_PLAN_QUERY)
EVENT_ALIASES = 6        # aliases of an event in its rewritten search
PQ_TOPICS = True         # FINANCE_PLAN_QUERY: topic queries for the questions the multi-query source leaves alone
PQ_EVENTS = True         # FINANCE_PLAN_QUERY: event (and topic-term) queries

_FRAME = re.compile(r"\b(?:Mr\.?\s+|Warren\s+(?:E\.\s+)?)?Buffett(?:'s|’s)?(?!\w)", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_KEYWORD_STOP = _SEARCH_STOPWORDS | frozenset(
    """not only one two year yearly per other its while than make made has gave gives give took take much many
    well very even still may might must should each every both since until under same own way ways time
    times""".split())
_KEYWORDS: "weakref.WeakKeyDictionary[Any, Dict[Tuple[str, str], Tuple[str, ...]]]" = weakref.WeakKeyDictionary()


def _ks():
    from src.retrieval import knowledge_source  # lazy: knowledge_source imports this module for its SOURCES
    return knowledge_source


def is_view(plan: QueryPlan) -> bool:
    """A view or opinion question: VIEW_CUE in the finance query, and not a numeric plan."""
    return plan.kind != NUMERIC and bool(_ks().VIEW_CUE.search(plan.financial_query or plan.query))


def financial_core(plan: QueryPlan) -> str:
    """The finance query without the opinion frame: VIEW_CUE words and Buffett's name removed (the query itself when
    nothing is left)."""
    text = plan.financial_query or plan.query
    core = _ks().VIEW_CUE.sub(" ", _FRAME.sub(" ", text))
    core = re.sub(r"\s+([,.;?])", r"\1", re.sub(r"\s+", " ", core)).strip(" ,;")
    return core if _search_tokens(core) else text


def topics(plan: QueryPlan, store: KnowledgeStore, retriever: Any, query: str) -> List[Tuple[str, str]]:
    """(field, key) of the question's topics in search order: the plan's themes, the inferred themes, then (MQ_SECTORS)
    the plan's sectors; only those the taxonomy describes."""
    taxonomy = {"themes": store.taxonomy.get("themes") or {}, "sectors": store.taxonomy.get("sectors") or {}}
    inferred = _ks().infer_themes(query, store, getattr(retriever, "embedder", None))
    found = [("themes", t) for t in (*plan.themes, *inferred)]
    if MQ_SECTORS:
        found += [("sectors", s) for s in plan.sectors]
    return [pair for pair in dict.fromkeys(found) if pair[1] in taxonomy[pair[0]]]


def topic_description(store: KnowledgeStore, field: str, key: str) -> str:
    """"key name: first sentence of its taxonomy description" (the text theme inference embeds)."""
    text = str((store.taxonomy.get(field) or {}).get(key, "")).strip()
    return f"{key.replace('_', ' ')}: {_SENTENCE.split(text, 1)[0]}"


def topic_keywords(store: KnowledgeStore, field: str, key: str) -> Tuple[str, ...]:
    """The KEYWORDS words that set the summaries of the passages tagged with the topic apart from all summaries
    (share of tagged summaries with the word x log inverse share of all summaries; in at least KEYWORD_MIN_DOCS tagged
    ones). Words are counted by their BM25 search token and written in their most frequent surface form. Memoised per
    store."""
    memo = _ks()._memo(_KEYWORDS, store, dict)
    if (field, key) in memo:
        return memo[(field, key)]
    df_all: Counter = Counter()
    df_topic: Counter = Counter()
    surface: Dict[str, Counter] = {}
    n_all = n_topic = 0
    for ann in store.annotations():
        words: Dict[str, str] = {}
        for raw in re.findall(r"[a-z][a-z'-]+", str(ann.get("summary") or "").lower()):
            token = _search_tokens(raw)
            if len(raw) > 2 and len(token) == 1 and token[0] not in _KEYWORD_STOP:
                words.setdefault(token[0], raw)
        if not words:
            continue
        n_all += 1
        df_all.update(words.keys())
        if key in (ann.get(field) or ()):
            n_topic += 1
            df_topic.update(words.keys())
            for token, raw in words.items():
                surface.setdefault(token, Counter())[raw] += 1
    scored = [((df_topic[w] / n_topic) * math.log(n_all / df_all[w]), w) for w in df_topic
              if df_topic[w] >= KEYWORD_MIN_DOCS]
    top = sorted(scored, key=lambda s: (-s[0], s[1]))[:KEYWORDS]
    memo[(field, key)] = tuple(surface[w].most_common(1)[0][0] for _, w in top)
    return memo[(field, key)]


def topic_query(core: str, store: KnowledgeStore, field: str, key: str) -> str:
    """One multi-query search: the financial core plus the topic's description, keywords or both (THEME_QUERY)."""
    parts = [core]
    if THEME_QUERY in ("description", "both"):
        parts.append(topic_description(store, field, key))
    if THEME_QUERY in ("keywords", "both"):
        parts.append(" ".join(topic_keywords(store, field, key)))
    return " ".join(p for p in parts if p)


def digest_query(plan: QueryPlan, store: KnowledgeStore, inferred: Sequence[str] = ()) -> Optional[str]:
    """The first DIGEST_LINES sentences of the overview of the plan's first digest topic; None without one."""
    for topic in _ks().digest_topics(plan, store, inferred):
        overview = str((store.digest(topic) or {}).get("overview") or "").strip()
        if overview:
            return " ".join(_SENTENCE.split(overview)[:DIGEST_LINES])
    return None


def queries(plan: QueryPlan, store: KnowledgeStore, retriever: Any, query: str, n: int, digest: bool) -> List[str]:
    """The extra search queries of a view question: up to n topic queries, then the digest query (when `digest`)."""
    if not is_view(plan):
        return []
    found = topics(plan, store, retriever, query) if (n > 0 or digest) else []
    core = financial_core(plan)
    out = [topic_query(core, store, field, key) for field, key in found[:max(0, n)]]
    if digest:
        text = digest_query(plan, store, [key for field, key in found if field == "themes"])
        if text:
            out.append(text)
    return list(dict.fromkeys(q for q in out if q.strip()))


def event_query(core: str, store: KnowledgeStore, event: str) -> str:
    """One rewritten search for a plan event: the financial core, the event's first EVENT_ALIASES aliases and its real
    years (`event_years`, the years it happened, not the letters that discuss it)."""
    spec = store.events().get(event) or {}
    years = [str(y) for y in spec.get("event_years") or ()]
    return " ".join([core, *list(spec.get("aliases") or ())[:EVENT_ALIASES], *years]).strip()


def plan_queries(plan: QueryPlan, store: KnowledgeStore, retriever: Any, query: str, n: int) -> List[str]:
    """FINANCE_PLAN_QUERY=n: up to n topic queries (`topic_query`) when the multi-query source does not search the
    question's topics (not a view question, or FINANCE_MULTI_QUERY off), one `event_query` per plan event, and for a
    plan with topic terms but no store event one query with those terms."""
    if n <= 0:
        return []
    core = financial_core(plan)
    out: List[str] = []
    if PQ_TOPICS and not (is_view(plan) and finance_setting("FINANCE_MULTI_QUERY") > 0):
        out += [topic_query(core, store, field, key) for field, key in topics(plan, store, retriever, query)[:n]]
    if PQ_EVENTS:
        out += [event_query(core, store, event) for event in plan.events]
        if plan.terms and not plan.events:
            out.append(" ".join([core, *plan.terms]))
    return list(dict.fromkeys(q for q in out if q.strip() and q != core))


def _searches(plan: QueryPlan, store: KnowledgeStore, texts: Sequence[str]
              ) -> List[Tuple[str, Optional[Dict[str, Any]], float]]:
    """(query, filter, RRF weight) of every ranking: per query one per year block and the unfiltered one, as in
    finance_search (a hard year searches only its range)."""
    blocks, _ = _boost(plan, store)
    boost = 1.0 if plan.year_hard else float(max(1, finance_setting("FINANCE_YEAR_BOOST_WEIGHT")))
    out: List[Tuple[str, Optional[Dict[str, Any]], float]] = []
    for text in texts:
        out += [(text, _year_filter(block), boost) for block in blocks]
        if not (blocks and plan.year_hard):
            out.append((text, None, 1.0))
    return out


def _summary_ranking(plan: QueryPlan, store: KnowledgeStore, retriever: Any, text: str) -> List[SearchHit]:
    """The summary-index ranking of `text` (its own year logic); [] when the index is missing or does not load."""
    try:
        return summary_source.summary_ranking(plan, store, retriever, text)
    except ValueError:
        return []


def _parallel(calls: Sequence[Callable[[], Any]]) -> List[Any]:
    """The results of `calls`, in order, run on up to MQ_WORKERS threads."""
    if len(calls) <= 1:
        return [call() for call in calls]
    with ThreadPoolExecutor(max_workers=min(MQ_WORKERS, len(calls))) as pool:
        return list(pool.map(lambda call: call(), calls))


def _ranking_calls(plan: QueryPlan, store: KnowledgeStore, retriever: Any, texts: Sequence[str]
                   ) -> List[Tuple[Callable[[], List[SearchHit]], float]]:
    """(search, RRF weight) of every ranking of `texts`: `_searches`, and with MQ_SUMMARY each text's summary-index
    ranking."""
    calls: List[Tuple[Callable[[], List[SearchHit]], float]] = [
        ((lambda s=s: _hybrid_ranking(retriever, s[0], MQ_DEPTH, s[1])), s[2]) for s in _searches(plan, store, texts)]
    if MQ_SUMMARY and calls:
        calls += [((lambda t=t: _summary_ranking(plan, store, retriever, t)), 1.0) for t in texts]
    return calls


def _fuse(rankings: Sequence[Sequence[SearchHit]], weights: Sequence[float]) -> List[SearchHit]:
    total = sum(len(r) for r in rankings)
    return reciprocal_rank_fusion(rankings, top_k=max(total, 1), weights=list(weights)) if total else []


def fused_ranking(plan: QueryPlan, store: KnowledgeStore, retriever: Any, texts: Sequence[str]) -> List[SearchHit]:
    """Every ranking of `texts` (`_ranking_calls`) run in parallel and fused by weighted RRF, best first."""
    calls = _ranking_calls(plan, store, retriever, texts)
    return _fuse(_parallel([call for call, _ in calls]), [w for _, w in calls])


def pooled(plan: QueryPlan, store: KnowledgeStore, retriever: Any, query: str, k: int) -> Set[str]:
    """The ids `finance_search` holds before this source runs: its shallow rankings (the unfiltered one unless the year
    is hard, one per year block, the plan's topic terms per block), the entity-tagged passages of its deeper ranking
    (inside the blocks of a narrow question) and the hits of the candidate sources registered before the first source
    of this module (`FAMILY`) that are on. The cap is spent on new candidates only."""
    fetch_k = max(RETRIEVAL_FETCH_K, k * finance_setting("FINANCE_OVERFETCH"))
    blocks, narrow = _boost(plan, store)
    rankings = [] if (blocks and plan.year_hard) else [_hybrid_ranking(retriever, query, fetch_k, None)]
    for block in blocks:
        where = _year_filter(block)
        rankings.append(_hybrid_ranking(retriever, query, fetch_k, where))
        if plan.terms and hasattr(retriever, "_bm25_search"):
            rankings.append(list(retriever._bm25_search(" ".join(plan.terms), fetch_k, where)))
    out = {h.id for ranking in rankings for h in ranking}
    if _has_entities(plan) and len(store) and out:
        wanted = {"events": set(plan.events), "sectors": set(plan.sectors), "companies": set(plan.companies),
                  "themes": set(plan.themes)}
        depth = max(fetch_k, finance_setting("FINANCE_TAG_DEPTH"))
        for where in ([_year_filter(b) for b in blocks] if (blocks and narrow) else [None]):
            for hit in _hybrid_ranking(retriever, query, depth, where):
                ann = store.annotation(hit.id) or {}
                if any(values.intersection(ann.get(f) or ()) for f, values in wanted.items()):
                    out.add(hit.id)
    for name, source in finance_policy.candidate_sources():
        if source in FAMILY:
            break
        if finance_policy.source_unavailable(name, source):
            continue
        try:
            out.update(h.id for h in source(plan, store, retriever, query, k))
        except Exception:  # noqa: BLE001 - finance_search skips a failing source too
            continue
    return out


def _pooled_before(plan: QueryPlan, store: KnowledgeStore, retriever: Any, query: str, k: int, me: Callable
                   ) -> Set[str]:
    """`pooled` plus the hits of the sources of this module registered before `me` that are on."""
    out = pooled(plan, store, retriever, query, k)
    for _name, source in finance_policy.candidate_sources():
        if source is me:
            break
        if source not in FAMILY:
            continue
        try:
            out.update(h.id for h in source(plan, store, retriever, query, k))
        except Exception:  # noqa: BLE001 - finance_search skips a failing source too
            continue
    return out


def _source(plan: QueryPlan, store: KnowledgeStore, retriever: Any, query: str, k: int, n: int,
            digest: bool) -> List[SearchHit]:
    return _new_hits(plan, store, retriever, query, k, queries(plan, store, retriever, query, n, digest), MQ_CAP,
                     lambda: pooled(plan, store, retriever, query, k))


def _new_hits(plan: QueryPlan, store: KnowledgeStore, retriever: Any, query: str, k: int, texts: Sequence[str],
              cap: int, skipped: Callable[[], Set[str]]) -> List[SearchHit]:
    """At most `cap` passages of the fused rankings of `texts` that `skipped()` does not hold (in range first)."""
    ks = _ks()
    docs = ks._docs(retriever)
    if not texts or not docs:
        return []
    calls = _ranking_calls(plan, store, retriever, texts)
    skip, *rankings = _parallel([skipped, *(call for call, _ in calls)])
    blocks, _ = _boost(plan, store)
    numeric = finance_policy.tables_allowed(plan)
    inside: List[str] = []
    outside: List[str] = []
    for hit in _fuse(rankings, [w for _, w in calls]):
        pid = hit.id
        if pid in skip or pid not in docs or not store.is_financial(pid) or (store.is_table(pid) and not numeric):
            continue
        year = ks.letter_of(pid, docs)
        (inside if not blocks or any(a <= year <= b for a, b in blocks) else outside).append(pid)
    letter = lambda pid: ks.letter_of(pid, docs)  # noqa: E731
    picked = ks._cap_letters(inside, cap, letter)
    if not plan.year_hard:
        picked += ks._cap_letters(outside, cap - len(picked), letter)
    return ks._hits(picked, docs)


def multi_query_source(plan: QueryPlan, store: KnowledgeStore, retriever: Any, query: str, k: int
                       ) -> List[SearchHit]:
    """Topic searches of a view question (FINANCE_MULTI_QUERY), with the digest search when FINANCE_DIGEST_QUERY is
    on."""
    return _source(plan, store, retriever, query, k, max(0, finance_setting("FINANCE_MULTI_QUERY")),
                   bool(finance_setting("FINANCE_DIGEST_QUERY")))


def digest_query_source(plan: QueryPlan, store: KnowledgeStore, retriever: Any, query: str, k: int
                        ) -> List[SearchHit]:
    """The digest search of a view question (FINANCE_DIGEST_QUERY); nothing while FINANCE_MULTI_QUERY is on (the
    multi-query source runs it)."""
    if finance_setting("FINANCE_MULTI_QUERY") > 0:
        return []
    return _source(plan, store, retriever, query, k, 0, True)


def plan_query_source(plan: QueryPlan, store: KnowledgeStore, retriever: Any, query: str, k: int
                      ) -> List[SearchHit]:
    """The searches rewritten from the plan (FINANCE_PLAN_QUERY), at most PQ_CAP new passages."""
    texts = plan_queries(plan, store, retriever, query, max(0, finance_setting("FINANCE_PLAN_QUERY")))
    return _new_hits(plan, store, retriever, query, k, texts, PQ_CAP,
                     lambda: _pooled_before(plan, store, retriever, query, k, plan_query_source))


SOURCES: List[Tuple[str, Callable]] = [
    ("FINANCE_MULTI_QUERY", multi_query_source),
    ("FINANCE_DIGEST_QUERY", digest_query_source),
]
PLAN_SOURCES: List[Tuple[str, Callable]] = [("FINANCE_PLAN_QUERY", plan_query_source)]  # registered right after SOURCES
FAMILY = tuple(source for _, source in (*SOURCES, *PLAN_SOURCES))  # `pooled` stops at the first of them
