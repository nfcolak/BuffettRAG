"""Finance answer routes (finance-query-pipeline, group 5).

Callers gate everything here behind `FINANCE_PIPELINE=1`; this module does not read the flag.

Final answers, returned as they are (no evidence gate, no claim validator, no `expand_answer`):
  - `scope_answer()`: the fixed scope reply for personal plans, no hits;
  - `numeric_answer(plan, store, docs_by_id)`: rows of `store.numbers` that match the question, each with its value,
    year and the `as_written` text, citing the source passage that is served as a hit; None when nothing matches
    (the caller falls back to the reader);
  - `digest_answer(plan, store, docs_by_id, reader_hits)`: the overview of the matching `store.digest` topic, then dated
    verbatim quotes. The hits are the timeline `source_id` passages ordered by year and numbered in that order, then
    reader passages for gaps (letters the timeline does not cover). Every [n] in the answer points at a returned hit.

Reader route helpers (the answer still goes through `ask_flow._finalize_answer`):
  - `sort_by_year(hits)` and `trend_prompt(query, hits, history)`: trend answers keep the system prompt v1 (FT-r1 was
    trained on it), serve the passages sorted by year (the [n] markers follow the served list) and append one sentence
    to the user block that asks for a year-by-year answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.generation.prompt import build_cited_prompt, parse_citations
from src.knowledge import KnowledgeStore
from src.query.plan import QueryPlan
from src.storage import SearchHit

SCOPE_REPLY = "I only answer questions about Buffett's financial views and Berkshire's financial results."
TREND_SENTENCE = "Answer year by year, in the order of the passages, and say which year each point comes from."
MAX_NUMERIC_ROWS = 6
DIGEST_GAP_PASSAGES = 4  # reader passages appended after the digest's quoted passages, at most

_ANSWER_SUFFIX = "\n\nAnswer:"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_METRIC_STOP = {"in", "of", "the", "a", "an", "and", "to", "usd", "pct", "per"}
_UNIT_FORMATS = {
    "usd_millions": lambda v: f"${_num(v)} million",
    "usd_billions": lambda v: f"${_num(v)} billion",
    "usd": lambda v: f"${_num(v)}",
    "percent": lambda v: f"{_num(v)}%",
}


@dataclass
class FinanceAnswer:
    """A finished answer: the text, the hits its [n] markers point at, and the parsed citations."""
    answer: str
    hits: List[SearchHit] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    route: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


def _num(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value:,}" if isinstance(value, (int, float)) else str(value)


def _tokens(text: str) -> List[str]:
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in _TOKEN_RE.findall(str(text).lower())]


def _year(hit: SearchHit) -> int:
    value = hit.metadata.get("year")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 9999


def _hit_from_doc(doc: Any, fallback_year: Optional[int] = None) -> SearchHit:
    metadata = dict(getattr(doc, "metadata", None) or {})
    if fallback_year is not None and not metadata.get("year"):
        metadata["year"] = fallback_year
    return SearchHit(id=doc.id, text=doc.text, metadata=metadata, score=1.0)


def sort_by_year(hits: Sequence[SearchHit]) -> List[SearchHit]:
    """Served order for trend answers: oldest letter first, ties keep their rank."""
    return sorted(hits, key=_year)


def trend_prompt(query: str, hits: Sequence[SearchHit], history: Optional[Sequence[Dict[str, str]]] = None) -> str:
    """The v1-compatible cited prompt plus one sentence in the user block asking for a year-by-year answer."""
    prompt = build_cited_prompt(query=query, hits=hits, history=history)
    if not prompt.endswith(_ANSWER_SUFFIX):  # defensive: never rewrite a prompt whose tail we do not know
        return prompt
    return f"{prompt[:-len(_ANSWER_SUFFIX)]} {TREND_SENTENCE}{_ANSWER_SUFFIX}"


# --- scope ---
def scope_answer() -> FinanceAnswer:
    return FinanceAnswer(answer=SCOPE_REPLY, route="scope")


# --- numeric ---
def _value_text(row: Mapping[str, Any]) -> str:
    fmt = _UNIT_FORMATS.get(str(row.get("unit", "")))
    return fmt(row["value"]) if fmt and row.get("value") is not None else str(row.get("as_written", ""))


def _match_numbers(plan: QueryPlan, store: KnowledgeStore) -> List[Mapping[str, Any]]:
    text = plan.financial_query or plan.query
    query_tokens = set(_tokens(text))
    low = text.lower()
    rows = store.numbers()

    def named(entity: str) -> bool:
        return bool(entity) and re.search(r"(?<![\w])" + re.escape(entity) + r"(?:'s)?(?![\w])", low) is not None

    asked = {str(r.get("entity", "")).strip().lower() for r in rows if named(str(r.get("entity", "")).strip().lower())}
    scored: List[Tuple[float, Mapping[str, Any]]] = []
    for row in rows:
        metric_tokens = [t for t in _tokens(str(row["metric"]).replace("_", " ")) if t not in _METRIC_STOP]
        if not metric_tokens:
            continue
        share = sum(t in query_tokens for t in metric_tokens) / len(metric_tokens)
        entity = str(row.get("entity", "")).strip().lower()
        entity_hit = named(entity)
        if share < 0.5 or (share < 1 and not entity_hit):
            continue
        if asked and not entity_hit:  # the question names other companies from the table: not this row
            continue
        year = row.get("year")
        if plan.years and year not in plan.years:
            continue
        if not plan.years and plan.year_range and not plan.year_range[0] <= int(year or 0) <= plan.year_range[1]:
            continue
        scored.append((share * 2 + entity_hit + (1 if plan.years else 0), row))
    if not scored:
        return []
    best = max(scored, key=lambda item: item[0])[1]
    group = [row for _, row in scored if row["metric"] == best["metric"] and row.get("entity") == best.get("entity")]
    return sorted(group, key=lambda row: row.get("year") or 0)[:MAX_NUMERIC_ROWS]


def numeric_answer(plan: QueryPlan, store: KnowledgeStore, docs_by_id: Mapping[str, Any]) -> Optional[FinanceAnswer]:
    """Answer from the numbers table, or None when no row matches or no source passage can be served."""
    rows = [row for row in _match_numbers(plan, store) if row.get("source_id") in docs_by_id]
    if not rows:
        return None
    hits: List[SearchHit] = []
    index: Dict[str, int] = {}
    lines: List[str] = []
    for row in rows:
        sid = row["source_id"]
        if sid not in index:
            hits.append(_hit_from_doc(docs_by_id[sid], row.get("year")))
            index[sid] = len(hits)
        label = str(row["metric"]).replace("_", " ")
        entity = f"{row['entity']}, " if row.get("entity") else ""
        lines.append(f"- {row.get('year')} ({entity}{label}): {_value_text(row)}; "
                     f"as written: \"{row.get('as_written', '')}\" [{index[sid]}]")
    answer = "\n".join(lines)
    return FinanceAnswer(answer=answer, hits=hits, citations=parse_citations(answer, hits), route="numeric",
                         detail={"rows": len(rows)})


# --- digest ---
def digest_topic(plan: QueryPlan, store: KnowledgeStore) -> Optional[Tuple[str, Dict[str, Any]]]:
    """The first digest among the plan's events, sectors and themes (in that order)."""
    for topic in (*plan.events, *plan.sectors, *plan.themes):
        digest = store.digest(topic)
        if digest:
            return topic, digest
    return None


def digest_answer(
    plan: QueryPlan,
    store: KnowledgeStore,
    docs_by_id: Mapping[str, Any],
    reader_hits: Sequence[SearchHit] = (),
    gap_passages: int = DIGEST_GAP_PASSAGES,
) -> Optional[FinanceAnswer]:
    """Overview plus dated quotes citing their source passages; None when the plan has no usable digest."""
    found = digest_topic(plan, store)
    if found is None:
        return None
    topic, digest = found
    timeline = [e for e in digest.get("timeline") or [] if e.get("source_id") in docs_by_id and e.get("quote")]
    if not timeline:
        return None
    timeline = sorted(timeline, key=lambda e: int(e.get("year") or 0))
    hits: List[SearchHit] = []
    index: Dict[str, int] = {}
    quotes: List[str] = []
    for entry in timeline:
        sid = entry["source_id"]
        if sid not in index:
            hits.append(_hit_from_doc(docs_by_id[sid], entry.get("year")))
            index[sid] = len(hits)
        quotes.append(f"- {entry.get('year')}: \"{str(entry['quote']).strip()}\" [{index[sid]}]")
    answer = "\n\n".join(part for part in (str(digest.get("overview", "")).strip(), "\n".join(quotes)) if part)
    cited = len(hits)
    covered = {_year(h) for h in hits}
    gaps = [h for h in reader_hits if h.id not in index and _year(h) not in covered][:max(0, gap_passages)]
    hits.extend(gaps)
    return FinanceAnswer(answer=answer, hits=hits, citations=parse_citations(answer, hits[:cited]), route="digest",
                         detail={"topic": topic, "cited": cited, "gaps": len(gaps)})
