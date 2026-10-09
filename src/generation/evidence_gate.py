"""Evidence sufficiency gate run before any answer generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.generation.text_relevance import (
    MIN_QUESTION_OVERLAP, best_question_overlap, relevance_words, split_sentences,
)


@dataclass(frozen=True)
class EvidenceAssessment:
    sufficient: bool
    reason: str
    best_overlap: float


def _terms(text: str) -> set[str]:
    return relevance_words(text)


def assess_evidence(query: str, hits: Sequence[object], *, extra_queries: Sequence[str] = (),
                    min_overlap: float = MIN_QUESTION_OVERLAP) -> EvidenceAssessment:
    terms = _terms(query)
    for extra in extra_queries:
        terms |= _terms(extra)
    if not hits:
        return EvidenceAssessment(False, "no_retrieved_passages", 0.0)
    if not terms:
        return EvidenceAssessment(False, "query_has_no_checkable_terms", 0.0)
    # Topic, action and quantity may span adjacent source sentences. Use
    # the bounded-window relevance check (see text_relevance).
    overlap = max((best_question_overlap(terms, split_sentences(hit.text))
                   for hit in hits), default=0.0)
    sufficient = overlap >= min_overlap
    return EvidenceAssessment(sufficient, "sufficient" if sufficient else "no_lexical_evidence", overlap)
