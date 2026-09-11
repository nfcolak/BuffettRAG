"""Conservative post-generation claim/citation validation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

from src.evaluation.citation_faithfulness import lexical_support_score, split_sentences as sentence_units

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])\$?\d[\d,]*(?:\.\d+)?%?(?:bn|m|b)?", re.I)
_NEGATIONS = frozenset({"no", "not", "never", "neither", "nor", "without"})
_PREDICATE_NEGATION_RE = re.compile(
    r"\b(?:is|are|was|were|did|does|do|has|have|had|will|would|can|could)\s+not\b"
    r"|\b(?:isn't|aren't|wasn't|weren't|didn't|doesn't|don't|hasn't|haven't|hadn't|won't|wouldn't|can't|couldn't)\b"
    r"|\bnever\b|\bno\s+[A-Za-z]",
    re.I,
)
_STOPWORDS = frozenset(
    "a an and are as at be been but by did do does for from had has have in into is it its of on or that the their then these they this to was were with would".split()
)

@dataclass
class ClaimValidationResult:
    safe_answer: str
    blocked_claims: List[str]
    validations: List[dict]
    nli_available: bool


def split_claims(text: str) -> List[str]:
    clean = _CITATION_RE.sub("", text).strip().rstrip(".!?")
    return [part.strip(" ,.;") for part in re.split(r"\s+(?:and|but)\s+|;", clean, flags=re.I) if part.strip(" ,.;")]


def _deterministic_agreement(claim: str, passage: str) -> tuple[bool, float]:
    """Conservative approval: wording, polarity, quantities and entities must agree."""
    claim_tokens = [token.lower() for token in _TOKEN_RE.findall(claim)]
    claim_content = {token for token in claim_tokens if token not in _STOPWORDS}
    claim_numbers = {value.lower().replace(",", "") for value in _NUMBER_RE.findall(claim)}
    claim_entities = {
        token.lower() for token in _TOKEN_RE.findall(claim)
        if (token[:1].isupper() or token.isupper()) and token.lower() not in _STOPWORDS
    }
    best_score = 0.0
    normalized_passage = re.sub(r"\s+", " ", passage).strip()
    for sentence in sentence_units(normalized_passage):
        evidence_tokens = [token.lower() for token in _TOKEN_RE.findall(sentence)]
        evidence_set = set(evidence_tokens)
        evidence_content = {token for token in evidence_tokens if token not in _STOPWORDS}
        coverage = len(claim_content & evidence_content) / len(claim_content) if claim_content else 0.0
        best_score = max(best_score, lexical_support_score(claim, sentence))
        evidence_numbers = {value.lower().replace(",", "") for value in _NUMBER_RE.findall(sentence)}
        same_polarity = bool(_PREDICATE_NEGATION_RE.search(claim)) == bool(
            _PREDICATE_NEGATION_RE.search(sentence)
        )
        if (
            coverage == 1.0
            and same_polarity
            and claim_numbers.issubset(evidence_numbers)
            and claim_entities.issubset(evidence_set)
        ):
            return True, max(best_score, coverage)
    return False, best_score


def validate_and_filter_answer(answer: str, hits: Sequence[Any], *, nli_scorer: Callable[[str, str], float] | None = None,
                               threshold: float = 0.2) -> ClaimValidationResult:
    """Remove claims without cited textual/NLI support; never invent replacements."""
    kept_sentences, blocked, validations = [], [], []
    # Keep terminal citations attached to line/bullet sentence units.
    for raw_sentence in sentence_units(answer):
        if not raw_sentence:
            continue
        marker_values = list(dict.fromkeys(_CITATION_RE.findall(raw_sentence)))
        marker_numbers = [int(n.strip()) - 1 for marker in marker_values
                          for n in marker.split(",") if n.strip().isdigit()]
        marker_text = ", ".join(marker_values)
        valid = [i for i in marker_numbers if 0 <= i < len(hits)]
        for claim in split_claims(raw_sentence):
            scores = []
            agreements = []
            for index in valid:
                if nli_scorer:
                    score = float(nli_scorer(hits[index].text, claim))
                    agreement = score >= max(0.5, threshold)
                else:
                    agreement, score = _deterministic_agreement(claim, hits[index].text)
                scores.append(score)
                agreements.append(agreement)
            support = max(scores, default=0.0)
            supported = bool(valid) and any(agreements)
            validations.append({"claim": claim, "cited_indexes": valid, "support": support,
                                "method": "nli" if nli_scorer else "deterministic_agreement", "supported": supported})
            if supported:
                citation = f" [{marker_text}]" if marker_text else ""
                kept_sentences.append(f"{claim}.{citation}")
            else:
                blocked.append(claim)
    return ClaimValidationResult(" ".join(kept_sentences).strip(), blocked, validations, nli_scorer is not None)
