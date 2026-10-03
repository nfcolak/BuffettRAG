"""Evidence sufficiency gate run before any answer generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.generation.providers.local_provider import (
    _MIN_QUESTION_OVERLAP, _content_words, _split_sentences,
)


@dataclass(frozen=True)
class EvidenceAssessment:
    sufficient: bool
    reason: str
    best_overlap: float


def _terms(text: str) -> set[str]:
    return set(_content_words(text))


def assess_evidence(query: str, hits: Sequence[object], *, extra_queries: Sequence[str] = (),
                    min_overlap: float = _MIN_QUESTION_OVERLAP) -> EvidenceAssessment:
    terms = _terms(query)
    for extra in extra_queries:
        terms |= _terms(extra)
    if not hits:
        return EvidenceAssessment(False, "no_retrieved_passages", 0.0)
    if not terms:
        return EvidenceAssessment(False, "query_has_no_checkable_terms", 0.0)
    # Incidental words scattered across a long passage are not an answer.
    # Apply the same sentence-level relevance floor to local and llama paths.
    overlap = max((len(terms & _terms(sentence)) / len(terms)
                   for hit in hits for sentence in _split_sentences(hit.text)), default=0.0)
    sufficient = overlap >= min_overlap
    return EvidenceAssessment(sufficient, "sufficient" if sufficient else "no_lexical_evidence", overlap)
