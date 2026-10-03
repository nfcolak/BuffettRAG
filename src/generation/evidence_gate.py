"""Evidence sufficiency gate run before any answer generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.generation.providers.local_provider import (
    _MIN_QUESTION_OVERLAP, _best_question_overlap, _relevance_words, _split_sentences,
)


@dataclass(frozen=True)
class EvidenceAssessment:
    sufficient: bool
    reason: str
    best_overlap: float


def _terms(text: str) -> set[str]:
    return _relevance_words(text)


def assess_evidence(query: str, hits: Sequence[object], *, extra_queries: Sequence[str] = (),
                    min_overlap: float = _MIN_QUESTION_OVERLAP) -> EvidenceAssessment:
    terms = _terms(query)
    for extra in extra_queries:
        terms |= _terms(extra)
    if not hits:
        return EvidenceAssessment(False, "no_retrieved_passages", 0.0)
    if not terms:
        return EvidenceAssessment(False, "query_has_no_checkable_terms", 0.0)
    # Topic, action and quantity may span adjacent source sentences. Share
    # the bounded-window relevance check with the extractive engine.
    overlap = max((_best_question_overlap(terms, _split_sentences(hit.text))
                   for hit in hits), default=0.0)
    sufficient = overlap >= min_overlap
    return EvidenceAssessment(sufficient, "sufficient" if sufficient else "no_lexical_evidence", overlap)
