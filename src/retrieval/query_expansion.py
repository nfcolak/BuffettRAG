"""LLM-based query expansion for retrieval.

Meta-questions and topical shorthand often miss the corpus vocabulary
("Middle East" never appears in the letters, but ISCAR/Israel/OPEC do).
Before retrieval we ask the configured LLM (EXPANSION_MODE defaults to "off":
small embedded models expand unreliably) for a handful of extra search
keywords and append them to the query used for BM25 + embedding search.
The original question is still what the answer model sees.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from config import retrieval_setting
from src.retrieval.bm25 import _SEARCH_STOPWORDS, tokenize


_REFERENTIAL_RE = re.compile(r"\b(?:it|that|this|they|them|then|those|these|what about)\b", re.I)
_HISTORY_STOPWORDS = _SEARCH_STOPWORDS | {"let's", "discuss", "described", "reported"}


def build_followup_retrieval_query(query: str, history: Optional[Sequence[Dict[str, str]]] = None) -> str:
    """Resolve short/referential retrieval intent from the last user turn only.

    History contributes search terms, never evidence or generated facts. Years
    are retained so the retriever can apply the same filters as a full question.
    An explicit period in the follow-up supersedes periods in the earlier turn.
    """
    content = [word for word in tokenize(query) if word not in _HISTORY_STOPWORDS]
    always = bool(retrieval_setting("RETRIEVAL_FOLLOWUP_ALWAYS"))
    if not history or (not always and len(content) >= 6 and not _REFERENTIAL_RE.search(query)):
        return query
    # RETRIEVAL_FOLLOWUP_TURNS user turns (default 1: the last one), newest first.
    turns = max(1, retrieval_setting("RETRIEVAL_FOLLOWUP_TURNS"))
    previous = " ".join([str(turn.get("content", "")) for turn in reversed(history)
                         if turn.get("role") == "user"][:turns])
    words = [word for word in tokenize(previous) if word not in _HISTORY_STOPWORDS]
    if re.search(r"\b(?:19|20)\d{2}s?\b", query):
        words = [word for word in words if not re.fullmatch(r"(?:19|20)\d{2}s?", word)]
    return " ".join([query, *dict.fromkeys(words)]).strip()

_SCAFFOLD_WORDS = _SEARCH_STOPWORDS | frozenset(
    """whom whose where explain describe tell mention mentioned discuss discussed give gives given according
    he please me us tell does doing done being may might should shall must much many one two own his her him
    first second third part parts both each another whether regarding concerning refer refers""".split())


def strip_question_scaffolding(query: str) -> str:
    """The query's content words without question scaffolding (wh-words, auxiliaries, "tell me about").

    Numbers, years, names and every other content word keep their order; the result is a candidate query for
    retrieval only (the full question is still what the reranker and the answer model see).
    """
    kept = [word for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.%$,-]*", query)
            if word.lower().strip(".,'’-") not in _SCAFFOLD_WORDS or re.search(r"\d", word)]
    return " ".join(kept).strip()


_EXPANSION_PROMPT = """\
You expand search queries over Warren Buffett's Berkshire Hathaway shareholder \
letters (1977-2024). Given the question below, list 3 to 8 extra search keywords \
— synonyms, related people, companies, events or financial terms likely to appear \
in the letters. If the question refers back to the earlier conversation (e.g. \
"what about GEICO?"), use that context to pick keywords for what is actually \
being asked. Reply with comma-separated keywords only, no explanations.
{history_block}
Question: {query}

Keywords:"""

_MAX_KEYWORDS = 8
_MAX_KEYWORD_CHARS = 40
_BAD_KEYWORD_RE = re.compile(r"passage|shareholder letters do not|keyword|question", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19[7-9]\d|20[0-2]\d)\b")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z'’-]*)(?:\s+[A-Z][A-Za-z'’-]*)*\b")


@dataclass(frozen=True)
class StructuredExpansion:
    terms: List[str]
    entities: List[str]
    years: List[int]
    retrieval_query: str


def parse_expansion_keywords(raw: str) -> List[str]:
    """Extract a clean keyword list from an LLM expansion response."""
    first_line = raw.strip().splitlines()[0] if raw.strip() else ""
    keywords: List[str] = []
    for part in re.split(r"[,;]", first_line):
        term = part.strip().strip(".:-–—\"'`[]()")
        if not term or len(term) > _MAX_KEYWORD_CHARS:
            continue
        if len(term.split()) > 4 or _BAD_KEYWORD_RE.search(term):
            continue
        if term.lower() not in (k.lower() for k in keywords):
            keywords.append(term)
        if len(keywords) >= _MAX_KEYWORDS:
            break
    return keywords


def expand_query(query: str, llm, history_text: str = "") -> Optional[str]:
    """Return ``query + keywords`` for retrieval, or None when unavailable.

    Never raises: any provider failure just means retrieval runs on the
    original query. A provider that cannot generate (``unavailable``) is skipped.
    """
    if llm is None or getattr(llm, "provider_name", "") == "unavailable":
        return None
    history_block = f"\nRecent conversation:\n{history_text}\n" if history_text else ""
    try:
        raw = llm.generate(
            _EXPANSION_PROMPT.format(query=query, history_block=history_block),
            max_new_tokens=60,
        )
    except Exception:
        return None
    keywords = parse_expansion_keywords(raw or "")
    if not keywords:
        return None
    return f"{query} {' '.join(keywords)}"


def _query_entities(query: str) -> List[str]:
    ignored = {"What", "How", "Why", "When", "Where", "Who", "Did", "Does", "Buffett", "Berkshire"}
    return [entity for entity in _ENTITY_RE.findall(query) if entity not in ignored]


def expand_query_structured(query: str, llm, history_text: str = "") -> Optional[StructuredExpansion]:
    """Request bounded JSON expansion while preserving query entities and years.

    The original question remains the first component of ``retrieval_query``;
    generated fields can only add candidates and cannot rewrite temporal/entity intent.
    """
    if llm is None or getattr(llm, "provider_name", "") == "unavailable":
        return None
    prompt = (
        "Return JSON only: {\"terms\":[...],\"entities\":[...],\"years\":[...]} for "
        "a Berkshire shareholder-letter retrieval query. At most 8 short terms. "
        f"Question: {query}\n{history_text}"
    )
    try:
        raw = llm.generate(prompt, max_new_tokens=100)
        parsed = json.loads(raw)
    except (Exception,):
        return None
    if not isinstance(parsed, dict):
        return None
    terms = parse_expansion_keywords(",".join(str(x) for x in parsed.get("terms", []) if isinstance(x, str)))
    original_entities = _query_entities(query)
    generated_entities = [str(x).strip() for x in parsed.get("entities", []) if isinstance(x, str)]
    entities = list(dict.fromkeys(original_entities + [x for x in generated_entities if x and len(x) <= _MAX_KEYWORD_CHARS]))
    original_years = [int(y) for y in _YEAR_RE.findall(query)]
    generated_years = [int(y) for y in parsed.get("years", []) if isinstance(y, int) and 1977 <= y <= 2024]
    years = list(dict.fromkeys(original_years + generated_years))
    additions = list(dict.fromkeys(terms + entities + [str(year) for year in years]))[:_MAX_KEYWORDS + 8]
    return StructuredExpansion(terms=terms, entities=entities, years=years,
                               retrieval_query=" ".join([query, *additions]).strip())
