"""Evidence sufficiency gate run before any answer generation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

_STOP = {"what", "did", "does", "the", "and", "about", "with", "from", "that", "this", "buffett", "berkshire", "say", "write", "his", "her", "their", "how", "why", "are", "was", "were", "in", "on", "of", "to", "a", "an"}

@dataclass(frozen=True)
class EvidenceAssessment:
    sufficient: bool
    reason: str
    best_overlap: float


def _terms(text: str) -> set[str]:
    return {token for token in re.findall(r"[^\W_]+", text.lower(), re.UNICODE) if len(token) > 2 and token not in _STOP}


def assess_evidence(query: str, hits: Sequence[object], *, extra_queries: Sequence[str] = (), min_overlap: float = 0.2) -> EvidenceAssessment:
    terms = _terms(query)
    for extra in extra_queries:
        terms |= _terms(extra)
    if not hits:
        return EvidenceAssessment(False, "no_retrieved_passages", 0.0)
    if not terms:
        return EvidenceAssessment(False, "query_has_no_checkable_terms", 0.0)
    overlap = max((len(terms & _terms(hit.text)) / len(terms) for hit in hits), default=0.0)
    return EvidenceAssessment(overlap >= min_overlap, "sufficient" if overlap >= min_overlap else "no_lexical_evidence", overlap)
