"""LLM-based query expansion for retrieval.

Meta-questions and topical shorthand often miss the corpus vocabulary
("Middle East" never appears in the letters, but ISCAR/Israel/OPEC do).
Before retrieval we ask the configured LLM for a handful of extra search
keywords and append them to the query used for BM25 + embedding search.
The original question is still what the answer model sees.
"""

from __future__ import annotations

import re
from typing import List, Optional

_EXPANSION_PROMPT = """\
You expand search queries over Warren Buffett's Berkshire Hathaway shareholder \
letters (1977-2024). Given the question below, list 3 to 8 extra search keywords \
— synonyms, related people, companies, events or financial terms likely to appear \
in the letters. Reply with comma-separated keywords only, no explanations.

Question: {query}

Keywords:"""

_MAX_KEYWORDS = 8
_MAX_KEYWORD_CHARS = 40
_BAD_KEYWORD_RE = re.compile(r"passage|shareholder letters do not|keyword|question", re.IGNORECASE)


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


def expand_query(query: str, llm) -> Optional[str]:
    """Return ``query + keywords`` for retrieval, or None when unavailable.

    Never raises: any provider failure just means retrieval runs on the
    original query. The embedded extractive provider cannot expand queries,
    so it is skipped.
    """
    if llm is None or getattr(llm, "provider_name", "") == "local":
        return None
    try:
        raw = llm.generate(_EXPANSION_PROMPT.format(query=query), max_new_tokens=60)
    except Exception:
        return None
    keywords = parse_expansion_keywords(raw or "")
    if not keywords:
        return None
    return f"{query} {' '.join(keywords)}"
