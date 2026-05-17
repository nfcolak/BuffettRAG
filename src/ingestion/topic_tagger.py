"""Lightweight rule-based topic tagger for chunks.

Why not run a topic model? Two reasons:
    1. Reproducible -- a fresh clone of the repo gives the same tags.
    2. Cheap        -- runs in milliseconds on the full 7K-chunk corpus.

The taxonomy below was built from a manual pass over the Buffett letters and
covers the recurring themes the eval queries exercise (inflation, acquisitions,
insurance, technology, etc.). Multi-label: a chunk can carry several tags.
"""

from __future__ import annotations

import re
from typing import Dict, List

# Keyword lists are lowercased and matched as whole words / phrases.
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "insurance": [
        "insurance", "underwriting", "geico", "reinsurance", "float",
        "national indemnity", "ajit jain", "premium", "loss reserve",
    ],
    "acquisitions": [
        "acquisition", "acquired", "buy a business", "purchase price",
        "tuck-in", "bolt-on", "elephant", "merger",
    ],
    "investment_philosophy": [
        "intrinsic value", "margin of safety", "circle of competence",
        "moat", "economic moat", "owner earnings", "long-term",
        "value investing", "mr. market",
    ],
    "inflation": [
        "inflation", "purchasing power", "cpi", "deflation",
        "rising prices", "inflationary",
    ],
    "macroeconomics": [
        "interest rate", "federal reserve", "fed", "recession",
        "gdp", "economy", "monetary policy",
    ],
    "stock_market": [
        "stock market", "s&p 500", "dow jones", "index fund",
        "market crash", "bull market", "bear market",
    ],
    "technology": [
        "technology", "tech stock", "internet", "software",
        "apple", "ibm", "computer",
    ],
    "financial_crisis": [
        "2008", "financial crisis", "subprime", "lehman",
        "great recession", "credit crisis", "bailout",
    ],
    "derivatives": [
        "derivative", "credit default swap", "weapons of mass destruction",
        "cds", "options contract",
    ],
    "succession": [
        "succession", "successor", "ceo", "next generation",
        "greg abel", "ajit jain", "charlie",
    ],
    "shareholders": [
        "shareholder", "owner", "annual meeting", "share repurchase",
        "buyback", "dividend",
    ],
    "accounting": [
        "accounting", "gaap", "earnings per share", "book value",
        "balance sheet", "goodwill", "depreciation",
    ],
    "berkshire_subsidiary": [
        "see's candy", "see's candies", "burlington northern", "bnsf",
        "blue chip stamps", "dairy queen", "nebraska furniture",
        "borsheims", "marmon", "lubrizol",
    ],
    "risk_management": [
        "risk", "catastrophic", "black swan", "tail risk",
        "leverage", "debt",
    ],
}

# Pre-compile word-boundary patterns once.
_PATTERNS = {
    topic: re.compile(
        r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b",
        re.IGNORECASE,
    )
    for topic, keywords in TOPIC_KEYWORDS.items()
}


def tag_topics(text: str, max_tags: int = 4) -> List[str]:
    """Return the topics whose keywords appear in `text`.

    Tags are ordered by hit count (most-mentioned first) so that callers can
    truncate to the top-N if needed. Capped at `max_tags` to keep metadata
    small in the vector store.
    """
    if not text:
        return []
    scores: List[tuple[str, int]] = []
    for topic, pat in _PATTERNS.items():
        hits = len(pat.findall(text))
        if hits:
            scores.append((topic, hits))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scores[:max_tags]]
