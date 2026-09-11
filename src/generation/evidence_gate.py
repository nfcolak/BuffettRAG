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
    return {token for token in re.findall(r"[a-z0-9']+", text.lower()) if len(token) > 2 and token not in _STOP}


def assess_evidence(query: str, hits: Sequence[object], *, min_overlap: float = 0.2) -> EvidenceAssessment:
    terms = _terms(query)
    if not hits:
        return EvidenceAssessment(False, "no_retrieved_passages", 0.0)
    if not terms:
        return EvidenceAssessment(False, "query_has_no_checkable_terms", 0.0)
    overlap = max((len(terms & _terms(hit.text)) / len(terms) for hit in hits), default=0.0)
    return EvidenceAssessment(overlap >= min_overlap, "sufficient" if overlap >= min_overlap else "no_lexical_evidence", overlap)
