"""Finance candidate sources: knowledge (finance-recall-75, group 3).

`SOURCES` lists (setting name, callable) pairs. `finance_policy.candidate_sources()` runs a callable
`(plan, store, retriever, query, k) -> List[SearchHit]` only when its `finance_setting(name)` is non-zero; the hits join
the candidate pool before the cross-encoder rerank.

  - `tagged_source` (FINANCE_TAGGED_SOURCE): passages tagged with the plan's sectors, companies, events or themes, or
    with a theme inferred from the question (`infer_themes`), by hybrid rank; at most TAGGED_CAP, at most PER_LETTER
    per letter.
  - `digest_source` (FINANCE_DIGEST_SOURCE): trend and corpus-wide plans whose topic has a digest (the plan's events,
    companies, sectors and themes; the inferred themes only when none of those has one): the timeline `source_id`s,
    spread across the letters (decades) of the timeline, at most DIGEST_CAP.
  - `stance_source` (FINANCE_STANCE_SOURCE): superlative questions and view questions (VIEW_CUE) that name a sector or
    company: the passages with a stance on it (the asked polarity for a superlative), by hybrid rank; at most
    STANCE_CAP, at most PER_LETTER per letter.

Every source keeps what `finance_search` keeps (financial passages, tables only for numeric plans), only the range of a
"YYYY letter" question, and puts the passages inside the plan's year range (its boost blocks) first. It skips the
passages `finance_search` already pools: the top of its unfiltered ranking (fetch_k) and, for a plan with entities, the
entity-tagged passages of its deeper ranking (FINANCE_TAG_DEPTH); so the caps are spent on new candidates. The hybrid
rank is the rank in the hybrid ranking of the whole corpus, computed once per question for the three sources.

Theme inference matches the question's bge query embedding against the taxonomy theme descriptions (the theme name
and the first sentence of its description, embedded as passages). Some descriptions are close to every question about
Berkshire (berkshire_performance, buybacks), so a theme counts only when its cosine is at least THEME_MIN_SIM and at
least THEME_MIN_MARGIN above its cosine with THEME_ANCHOR, a question about nothing in particular; the best THEME_TOP
such themes are kept. The constants were set on the heldout_v4 / v5 questions, not on dev or heldout_v6.
FINANCE_THEME_CENTROIDS (default 0, finance-recall-75-r2) also represents each theme by the centroid of the summary
embeddings of its tagged passages and matches the question against both (THEME_CENTROID_MIN_SIM and
THEME_CENTROID_MIN_Z, also set on heldout_v4 / v5), so a paraphrase far from the description's words reaches it.

`stance_passages`, `spread_letters` and `letter_of` are shared with the superlative route
(`finance_answer.superlative_answer`).
"""
from __future__ import annotations

import logging
import re
import weakref
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config import RETRIEVAL_FETCH_K, finance_setting
from src.knowledge import KnowledgeStore
from src.query.plan import CORPUS_WIDE, NUMERIC, TREND, QueryPlan
from src.retrieval import multiquery_source, summary_source
from src.retrieval.finance_policy import _boost, _has_entities, _hybrid_ranking
from src.storage import SearchHit

log = logging.getLogger(__name__)

TAGGED_CAP = 24          # tagged candidates per question, at most
DIGEST_CAP = 16          # digest timeline passages per question, at most (one served budget)
STANCE_CAP = 16          # stance passages per question, at most
PER_LETTER = 2           # tagged / stance candidates per letter, at most (raised when few letters are available)
THEME_TOP = 2            # inferred themes per question, at most
THEME_MIN_SIM = 0.50     # cosine of the question (bge query embedding) and a theme description, at least
THEME_MIN_MARGIN = 0.15  # ... and at least this much above the theme's cosine with THEME_ANCHOR
THEME_ANCHOR = "What did Warren Buffett write about Berkshire Hathaway in his letters to shareholders?"
THEME_CENTROID_MIN_SIM = 0.58    # FINANCE_THEME_CENTROIDS: cosine of the question and a theme's summary centroid, at least
THEME_CENTROID_MIN_Z = 2.625     # ... and its margin over THEME_ANCHOR at least this many standard deviations above the
                                 # mean margin of the question's themes (the theme stands out for this question)
DIGEST_KINDS = (TREND, CORPUS_WIDE)
# Opinion words: the question asks what Buffett thinks, says or feels about something.
VIEW_CUE = re.compile(
    r"\b(?:think|thinks|thought|views?|viewed|opinions?|feel|feels|felt|believe[sd]?|regard(?:s|ed)?|"
    r"consider(?:s|ed)?|say|says|said|describe[sd]?|attitude|stance|prais\w*|critici[sz]\w*|warn\w*|"
    r"lik(?:e|es|ed)|admir\w*|favou?r\w*|skeptic\w*|optimis\w*|pessimis\w*|worr\w*|concern\w*|judg\w*|assess\w*|"
    r"recommend\w*|lessons?|mistakes?)\b", re.I)

_RANK_MEMO = 8  # whole-corpus rankings kept per retriever
_DOCS: "weakref.WeakKeyDictionary[Any, Dict[str, Any]]" = weakref.WeakKeyDictionary()
_RANKS: "weakref.WeakKeyDictionary[Any, OrderedDict]" = weakref.WeakKeyDictionary()
_THEMES: "weakref.WeakKeyDictionary[Any, Dict[Tuple[str, ...], Tuple[np.ndarray, np.ndarray]]]" = \
    weakref.WeakKeyDictionary()
_CENTROIDS: "weakref.WeakKeyDictionary[Any, Dict[Optional[str], Tuple[Any, ...]]]" = weakref.WeakKeyDictionary()
_CENTROID_WARNED: set = set()


# --- shared helpers ---
def letter_of(pid: str, docs_by_id: Optional[Dict[str, Any]] = None) -> int:
    """The passage's letter year (its metadata, else the id prefix); -1 when unknown."""
    doc = (docs_by_id or {}).get(pid)
    year = (getattr(doc, "metadata", None) or {}).get("year") if doc is not None else None
    if year is None and pid[:4].isdigit():
        year = pid[:4]
    try:
        return int(year)
    except (TypeError, ValueError):
        return -1


def _even(items: Sequence[Any], n: int) -> List[Any]:
    """n items evenly spread over the sequence (all of them when there are at most n)."""
    if n <= 0:
        return []
    if len(items) <= n:
        return list(items)
    step = (len(items) - 1) / max(1, n - 1)
    return [items[round(i * step)] for i in range(n)]


def spread_letters(ids: Iterable[str], n: int, letter: Callable[[str], int]) -> List[str]:
    """At most n ids taken in rounds: the first round takes the first id of every letter, the next round the second,
    and so on; a round with more letters than places left takes letters evenly spread over time."""
    groups: Dict[int, List[str]] = {}
    for pid in dict.fromkeys(ids):
        groups.setdefault(letter(pid), []).append(pid)
    letters = sorted(groups)
    out: List[str] = []
    depth = 0
    while len(out) < n:
        row = [y for y in letters if len(groups[y]) > depth]
        if not row:
            break
        out += [groups[y][depth] for y in _even(row, n - len(out))]
        depth += 1
    return out


def _cap_letters(ids: Sequence[str], n: int, letter: Callable[[str], int]) -> List[str]:
    """The first n ids in order, at most PER_LETTER of each letter, raised so n ids fit into the letters there are
    (a single-letter question is not capped)."""
    per_letter = max(PER_LETTER, -(-n // max(1, len({letter(pid) for pid in ids}))))
    counts: Dict[int, int] = {}
    out: List[str] = []
    for pid in ids:
        if len(out) >= n:
            break
        y = letter(pid)
        if counts.get(y, 0) < per_letter:
            counts[y] = counts.get(y, 0) + 1
            out.append(pid)
    return out


def stance_passages(store: KnowledgeStore, targets: Iterable[str],
                    polarity: Optional[str] = None) -> Dict[str, List[str]]:
    """target -> ids of the passages with a stance on it (of `polarity` when given), in annotation order."""
    out: Dict[str, List[str]] = {t: [] for t in targets}
    for ann in store.annotations():
        for stance in ann.get("stance") or ():
            target = str(stance.get("target", ""))
            if target in out and (polarity is None or stance.get("polarity") == polarity) \
                    and ann["id"] not in out[target]:
                out[target].append(ann["id"])
    return out


def _memo(table: weakref.WeakKeyDictionary, owner: Any, make: Callable[[], Any]) -> Any:
    """The value memoised for `owner` (a retriever or an embedder); unmemoised when it takes no weak reference."""
    try:
        if owner not in table:
            table[owner] = make()
        return table[owner]
    except TypeError:
        return make()


def _docs(retriever: Any) -> Dict[str, Any]:
    """id -> stored passage of the retriever's corpus (its `docs`, or its BM25 index's docs); empty without them."""
    def make() -> Dict[str, Any]:
        docs = getattr(retriever, "docs", None) or getattr(getattr(retriever, "bm25", None), "docs", None) or ()
        return {d.id: d for d in docs}
    return _memo(_DOCS, retriever, make)


def _ranks(retriever: Any, query: str, n: int) -> Dict[str, int]:
    """Rank of every passage in the hybrid ranking of the whole corpus (passages it leaves out are absent)."""
    memo = _memo(_RANKS, retriever, OrderedDict)
    key = (query, n)
    if key not in memo:
        memo[key] = {h.id: i for i, h in enumerate(_hybrid_ranking(retriever, query, n, None))}
        while len(memo) > _RANK_MEMO:
            memo.popitem(last=False)
    memo.move_to_end(key)
    return memo[key]


def _hits(ids: Sequence[str], docs: Dict[str, Any]) -> List[SearchHit]:
    return [SearchHit(id=pid, text=docs[pid].text, metadata=dict(docs[pid].metadata or {}), score=0.0) for pid in ids]


def _candidates(plan: QueryPlan, store: KnowledgeStore, retriever: Any, query: str, k: int, ids: Iterable[str],
                order: Optional[Dict[str, int]] = None) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """(ids inside the plan's year range, ids outside it, docs): the ids finance_search would keep and does not pool
    already, each list by hybrid rank (or by `order`). A "YYYY letter" question has no outside list."""
    docs = _docs(retriever)
    if not docs:
        return [], [], docs
    ranks = _ranks(retriever, query, len(docs))
    shallow = max(RETRIEVAL_FETCH_K, k * finance_setting("FINANCE_OVERFETCH"))  # finance_search's fetch_k
    depth = max(shallow, finance_setting("FINANCE_TAG_DEPTH")) if _has_entities(plan) and len(store) else shallow
    plan_tags = {"sectors": set(plan.sectors), "companies": set(plan.companies), "events": set(plan.events),
                 "themes": set(plan.themes)}
    blocks, _ = _boost(plan, store)
    numeric = plan.kind == NUMERIC
    inside: List[str] = []
    outside: List[str] = []
    for pid in dict.fromkeys(ids):
        if pid not in docs or not store.is_financial(pid) or (store.is_table(pid) and not numeric):
            continue
        rank = ranks.get(pid, len(docs))
        ann = store.annotation(pid) or {}
        if rank < shallow or (rank < depth and any(v.intersection(ann.get(f) or ()) for f, v in plan_tags.items())):
            continue  # in the pool already: the unfiltered ranking, or the entity-tagged deeper ranking
        year = letter_of(pid, docs)
        (inside if not blocks or any(a <= year <= b for a, b in blocks) else outside).append(pid)
    key = (lambda p: order[p]) if order is not None else (lambda p: ranks.get(p, len(docs)))
    return sorted(inside, key=key), ([] if plan.year_hard else sorted(outside, key=key)), docs


# --- theme inference ---
def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def theme_texts(store: KnowledgeStore) -> Tuple[List[str], List[str]]:
    """(theme keys, "theme name: first sentence of its description")."""
    themes = store.taxonomy.get("themes") or {}
    keys = list(themes)
    first = [re.split(r"(?<=[.])\s", str(themes[key]).strip(), 1)[0] for key in keys]  # py3.11: no backslash in f-strings
    return keys, [key.replace("_", " ") + ": " + text for key, text in zip(keys, first)]


def theme_centroids(store: KnowledgeStore, embedder: Any, keys: Sequence[str]
                    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(unit centroids, has one): per theme of `keys`, the mean of the summary embeddings (the summary index,
    `summary_source.summary_vectors`) of the passages tagged with it; a theme no summarised passage carries has none
    (a zero row). None when the summary index is missing or does not match the embedder (the descriptions still work)."""
    model = getattr(embedder, "model_name", None)
    try:
        found = summary_source.summary_vectors(model)
    except ValueError as exc:
        if model not in _CENTROID_WARNED:
            _CENTROID_WARNED.add(model)
            log.warning("theme centroids off, the summary index does not load: %s", exc)
        return None
    if found is None:
        return None
    ids, matrix = found
    memo = _memo(_CENTROIDS, store, dict)
    entry = memo.get(model)
    if entry is None or entry[0] is not matrix or entry[1] != tuple(keys):  # once per index, store and taxonomy
        column = {key: i for i, key in enumerate(keys)}
        members = np.zeros((len(keys), len(ids)), dtype=np.float32)
        for j, pid in enumerate(ids):
            for theme in (store.annotation(pid) or {}).get("themes") or ():
                if theme in column:
                    members[column[theme], j] = 1.0
        entry = (matrix, tuple(keys), _unit(members @ matrix), members.sum(axis=1) > 0)
        memo[model] = entry
    return entry[2], entry[3]


def infer_themes(query: str, store: KnowledgeStore, embedder: Any) -> Tuple[str, ...]:
    """The best THEME_TOP taxonomy themes the question matches. () without an embedder, a query or themes.

    A theme matches by its description: cosine >= THEME_MIN_SIM and >= THEME_MIN_MARGIN above the description's cosine
    with THEME_ANCHOR. With FINANCE_THEME_CENTROIDS on it also matches by its centroid (`theme_centroids`), which
    carries how the letters talk about the theme, so a general paraphrase that shares no word with the description
    still reaches it. Every centroid sits close to any Buffett question (the margins over THEME_ANCHOR are small and
    alike), so the centroid test is relative: cosine >= THEME_CENTROID_MIN_SIM and a margin THEME_CENTROID_MIN_Z
    standard deviations above the mean margin of the question's centroids. Themes are ranked by the larger of their
    two scores, each over its threshold (description margin / THEME_MIN_MARGIN, centroid z / THEME_CENTROID_MIN_Z)."""
    keys, texts = theme_texts(store)
    if embedder is None or not query or not keys:
        return ()
    memo = _memo(_THEMES, embedder, dict)
    if tuple(texts) not in memo:  # the descriptions and the anchor are embedded once per embedder
        vectors = _unit(np.asarray(embedder.embed_documents(texts), dtype=np.float32))
        anchor = _unit(np.asarray([embedder.embed_query(THEME_ANCHOR)], dtype=np.float32))[0]
        memo[tuple(texts)] = (vectors, vectors @ anchor)
    vectors, base = memo[tuple(texts)]
    question = _unit(np.asarray([embedder.embed_query(query)], dtype=np.float32))[0]
    sims = vectors @ question
    match = (sims >= THEME_MIN_SIM) & (sims - base >= THEME_MIN_MARGIN)
    score = sims - base
    centroids = theme_centroids(store, embedder, keys) if finance_setting("FINANCE_THEME_CENTROIDS") else None
    if centroids is not None and centroids[1].sum() > 1:
        rows, has = centroids
        anchor = _unit(np.asarray([embedder.embed_query(THEME_ANCHOR)], dtype=np.float32))[0]
        c_sims = rows @ question
        margin = c_sims - rows @ anchor
        spread = float(margin[has].std()) or 1.0
        z = np.where(has, (margin - margin[has].mean()) / spread, -np.inf)
        match |= has & (c_sims >= THEME_CENTROID_MIN_SIM) & (z >= THEME_CENTROID_MIN_Z)
        score = np.maximum(score / THEME_MIN_MARGIN, z / THEME_CENTROID_MIN_Z)
    order = np.argsort(-score, kind="stable")
    return tuple(keys[i] for i in order if match[i])[:THEME_TOP]


def _inferred(plan: QueryPlan, store: KnowledgeStore, retriever: Any, query: str) -> Tuple[str, ...]:
    return tuple(t for t in infer_themes(query, store, getattr(retriever, "embedder", None)) if t not in plan.themes)


# --- sources ---
def tagged_source(plan: QueryPlan, store: KnowledgeStore, retriever: Any, query: str, k: int) -> List[SearchHit]:
    """Passages tagged with the plan's entities or an inferred theme (FINANCE_TAGGED_SOURCE)."""
    wanted = {"sectors": set(plan.sectors), "companies": set(plan.companies), "events": set(plan.events),
              "themes": set(plan.themes) | set(_inferred(plan, store, retriever, query))}
    wanted = {field: values for field, values in wanted.items() if values}
    if not wanted:
        return []
    tagged = (ann["id"] for ann in store.annotations()
              if any(values.intersection(ann.get(field) or ()) for field, values in wanted.items()))
    inside, outside, docs = _candidates(plan, store, retriever, query, k, tagged)
    letter = lambda pid: letter_of(pid, docs)  # noqa: E731
    picked = _cap_letters(inside, TAGGED_CAP, letter)
    return _hits(picked + _cap_letters(outside, TAGGED_CAP - len(picked), letter), docs)


def digest_topics(plan: QueryPlan, store: KnowledgeStore, inferred: Sequence[str] = ()) -> List[str]:
    """The plan's events, companies, sectors and themes that have a digest; else the inferred themes that have one."""
    named = [t for t in (*plan.events, *plan.companies, *plan.sectors, *plan.themes) if store.digest(t)]
    return list(dict.fromkeys(named or [t for t in inferred if store.digest(t)]))


def digest_source(plan: QueryPlan, store: KnowledgeStore, retriever: Any, query: str, k: int) -> List[SearchHit]:
    """The timeline source passages of the plan's digests, spread across letters (FINANCE_DIGEST_SOURCE)."""
    if plan.kind not in DIGEST_KINDS:
        return []
    topics = digest_topics(plan, store) or digest_topics(plan, store, _inferred(plan, store, retriever, query))
    entries = sorted((e for topic in topics for e in (store.digest(topic) or {}).get("timeline") or []
                      if e.get("source_id")), key=lambda e: int(e.get("year") or 0))
    order: Dict[str, int] = {}
    for entry in entries:
        order.setdefault(entry["source_id"], len(order))
    inside, outside, docs = _candidates(plan, store, retriever, query, k, order, order)
    letter = lambda pid: letter_of(pid, docs)  # noqa: E731
    picked = spread_letters(inside, DIGEST_CAP, letter)
    return _hits(picked + spread_letters(outside, DIGEST_CAP - len(picked), letter), docs)


def stance_source(plan: QueryPlan, store: KnowledgeStore, retriever: Any, query: str, k: int) -> List[SearchHit]:
    """Passages with a stance on the sector or company of a superlative or view question (FINANCE_STANCE_SOURCE)."""
    targets = [*plan.sectors, *plan.companies]
    if not targets or not (plan.superlative or VIEW_CUE.search(plan.financial_query or plan.query)):
        return []
    by_target = stance_passages(store, targets, plan.superlative.partition(":")[2] or None)
    ids = (pid for target in targets for pid in by_target[target])
    inside, outside, docs = _candidates(plan, store, retriever, query, k, ids)
    letter = lambda pid: letter_of(pid, docs)  # noqa: E731
    picked = _cap_letters(inside, STANCE_CAP, letter)
    return _hits(picked + _cap_letters(outside, STANCE_CAP - len(picked), letter), docs)


SOURCES: List[Tuple[str, Callable]] = [
    ("FINANCE_TAGGED_SOURCE", tagged_source),
    ("FINANCE_DIGEST_SOURCE", digest_source),
    ("FINANCE_STANCE_SOURCE", stance_source),
    *multiquery_source.SOURCES,  # FINANCE_MULTI_QUERY, FINANCE_DIGEST_QUERY (finance-recall-75-r3)
]
