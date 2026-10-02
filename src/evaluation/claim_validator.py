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

_IGNORE = frozenset(
    "buffett warren berkshire hathaway said says wrote writes argued noted explained according "
    "letter letters shareholders he his him".split()
)
_SPLIT_RE = re.compile(r"([.!?](?:[ \t]*\[\d+(?:\s*,\s*\d+)*\])*)[ \t]+(?=[A-Z0-9\"'])")
_MIN_COVERAGE = 0.6


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
    """Approve when content coverage >= 0.6 and polarity, numbers and entities agree."""
    raw_tokens = _TOKEN_RE.findall(claim)
    claim_content = {t.lower() for t in raw_tokens if t.lower() not in _STOPWORDS and t.lower() not in _IGNORE}
    claim_numbers = {value.lower().replace(",", "") for value in _NUMBER_RE.findall(claim)}
    claim_entities = {
        t.lower() for t in raw_tokens[1:]
        if (t[:1].isupper() or t.isupper()) and t.lower() not in _STOPWORDS and t.lower() not in _IGNORE
        and not t[:1].isdigit()
    }
    # Very short claims have too little content for partial coverage.
    needed = 1.0 if len(claim_content) <= 3 else _MIN_COVERAGE
    best_score = 0.0
    normalized_passage = re.sub(r"\s+", " ", passage).strip()
    for sentence in sentence_units(normalized_passage):
        evidence_tokens = [token.lower() for token in _TOKEN_RE.findall(sentence)]
        evidence_set = set(evidence_tokens)
        coverage = len(claim_content & evidence_set) / len(claim_content) if claim_content else 0.0
        best_score = max(best_score, lexical_support_score(claim, sentence), coverage)
        evidence_numbers = {value.lower().replace(",", "") for value in _NUMBER_RE.findall(sentence)}
        same_polarity = bool(_PREDICATE_NEGATION_RE.search(claim)) == bool(
            _PREDICATE_NEGATION_RE.search(sentence)
        )
        if (
            coverage >= needed
            and same_polarity
            and claim_numbers.issubset(evidence_numbers)
            and claim_entities.issubset(evidence_set)
        ):
            return True, max(best_score, coverage)
    return False, best_score


def _split_original(line: str) -> List[str]:
    """Split a line into sentences keeping the original text (bullets, bold, markers)."""
    return [part for part in _SPLIT_RE.sub(r"\1\n", line).split("\n") if part.strip()]


def validate_and_filter_answer(answer: str, hits: Sequence[Any], *, nli_scorer: Callable[[str, str], float] | None = None,
                               threshold: float = 0.2) -> ClaimValidationResult:
    """Drop sentences without cited textual/NLI support; kept sentences stay verbatim."""
    out_lines: List[str] = []
    blocked, validations = [], []
    for line in answer.split("\n"):
        if not line.strip():
            out_lines.append("")
            continue
        kept = []
        for sentence in _split_original(line):
            marker_values = list(dict.fromkeys(_CITATION_RE.findall(sentence)))
            marker_numbers = [int(n.strip()) - 1 for marker in marker_values
                              for n in marker.split(",") if n.strip().isdigit()]
            valid = [i for i in marker_numbers if 0 <= i < len(hits)]
            sentence_ok = bool(valid)
            failed = []
            body = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", sentence)
            for claim in split_claims(body):
                scores, agreements = [], []
                for index in valid:
                    if nli_scorer:
                        score = float(nli_scorer(hits[index].text, claim))
                        agreement = score >= max(0.5, threshold)
                    else:
                        agreement, score = _deterministic_agreement(claim, hits[index].text)
                    scores.append(score)
                    agreements.append(agreement)
                supported = bool(valid) and any(agreements)
                validations.append({"claim": claim, "cited_indexes": valid, "support": max(scores, default=0.0),
                                    "method": "nli" if nli_scorer else "deterministic_agreement",
                                    "supported": supported})
                if not supported:
                    sentence_ok = False
                    failed.append(claim)
            if sentence_ok:
                kept.append(sentence.strip())
            else:
                blocked.extend(failed or [body.strip()])
        if kept:
            out_lines.append(" ".join(kept))
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()
    return ClaimValidationResult(text, blocked, validations, nli_scorer is not None)
