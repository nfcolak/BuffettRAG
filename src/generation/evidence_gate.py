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
                    min_overlap: float = MIN_QUESTION_OVERLAP, query_alone: bool = False) -> EvidenceAssessment:
    """Best lexical overlap of the question's terms with any retrieved sentence window.

    `extra_queries` (earlier user turns, an expanded query) add terms. Their union with the question can only lower
    the share when the history is about another topic, so with `query_alone` the question's own terms are scored too
    and the better score counts: history can help a short follow-up but never hurt a full question.
    """
    own = _terms(query)
    terms = set(own)
    for extra in extra_queries:
        terms |= _terms(extra)
    if not hits:
        return EvidenceAssessment(False, "no_retrieved_passages", 0.0)
    if not terms:
        return EvidenceAssessment(False, "query_has_no_checkable_terms", 0.0)
    term_sets = [terms] + ([own] if query_alone and own and own != terms else [])
    # Topic, action and quantity may span adjacent source sentences. Use
    # the bounded-window relevance check (see text_relevance).
    overlap = max((best_question_overlap(ts, split_sentences(hit.text))
                   for ts in term_sets for hit in hits), default=0.0)
    sufficient = overlap >= min_overlap
    return EvidenceAssessment(sufficient, "sufficient" if sufficient else "no_lexical_evidence", overlap)
