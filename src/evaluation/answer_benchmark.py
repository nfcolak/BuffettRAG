"""Deterministic scoring for human-curated answer-quality benchmark cases."""
from __future__ import annotations

import re
from typing import Any, Dict, Sequence

from src.evaluation.citation_faithfulness import split_sentences

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])\$?\d[\d,]*(?:\.\d+)?%?(?:bn|m|b)?", re.I)
_NEGATIONS = frozenset({"no", "not", "never", "neither", "nor", "without"})
_PREDICATE_NEGATION_RE = re.compile(
    r"\b(?:is|are|was|were|did|does|do|has|have|had|will|would|can|could)\s+not\b|\bnever\b|\bno\s+[A-Za-z]",
    re.I,
)


def validate_benchmark_case(case: Dict[str, Any]) -> None:
    required = {"qid", "query", "gold_passage_ids", "gold_claims", "accept", "reject"}
    missing = required - set(case)
    if missing or not isinstance(case["gold_passage_ids"], list) or not case["gold_passage_ids"]:
        raise ValueError(f"Invalid benchmark case: missing/invalid {sorted(missing)}")
    if not case["gold_claims"] or not all(c.get("claim") and c.get("required_terms") for c in case["gold_claims"]):
        raise ValueError("Every benchmark case needs a claim and required_terms")
    if not 0.0 <= float(case["accept"].get("min_claim_coverage", -1)) <= 1.0:
        raise ValueError("accept.min_claim_coverage must be in [0, 1]")


def _cited_hit_indexes(answer: str) -> set[int]:
    indexes: set[int] = set()
    for raw in _CITATION_RE.findall(answer):
        indexes.update(int(n.strip()) - 1 for n in raw.split(",") if n.strip().isdigit())
    return indexes


def _sentence_cited_ids(sentence: str, hits: Sequence[Any]) -> set[str]:
    indexes = _cited_hit_indexes(sentence)
    return {hits[index].id for index in indexes if 0 <= index < len(hits)}


def _polarity_and_numbers_agree(reference: str, sentence: str) -> bool:
    if bool(_PREDICATE_NEGATION_RE.search(reference)) != bool(_PREDICATE_NEGATION_RE.search(sentence)):
        return False
    expected_numbers = {value.lower().replace(",", "") for value in _NUMBER_RE.findall(reference)}
    actual_numbers = {value.lower().replace(",", "") for value in _NUMBER_RE.findall(sentence)}
    return expected_numbers.issubset(actual_numbers)


def evaluate_answer(answer: str, hits: Sequence[Any], case: Dict[str, Any]) -> Dict[str, Any]:
    """Apply explicit accept/reject criteria; this is not an LLM judge."""
    validate_benchmark_case(case)
    answer_lc = answer.lower()
    sentences = split_sentences(answer)
    gold_ids = set(case["gold_passage_ids"])
    claims = []
    require_citation = bool(case["accept"].get("require_valid_citation", False))
    for claim in case["gold_claims"]:
        claim_gold_ids = set(claim.get("gold_passage_ids") or gold_ids)
        if not claim_gold_ids.issubset(gold_ids):
            raise ValueError("claim gold_passage_ids must be a subset of case gold_passage_ids")
        terms = [term.lower() for term in claim["required_terms"]]
        matching = []
        for sentence in sentences:
            sentence_lc = sentence.lower()
            cited_gold = bool(_sentence_cited_ids(sentence, hits) & claim_gold_ids)
            if (
                all(term in sentence_lc for term in terms)
                and _polarity_and_numbers_agree(claim["claim"], sentence)
                and (not require_citation or cited_gold)
            ):
                matching.append(sentence)
        claims.append({"claim": claim["claim"], "met": bool(matching),
                       "supporting_sentences": matching})
    coverage = sum(row["met"] for row in claims) / len(claims)
    forbidden = [term for term in case["reject"].get("forbidden_terms", []) if term.lower() in answer_lc]
    valid_citation = all(row["met"] for row in claims) if require_citation else True
    accepted = (
        coverage >= float(case["accept"]["min_claim_coverage"])
        and (not case["accept"].get("require_valid_citation", False) or valid_citation)
        and not forbidden
    )
    cited_ids = set().union(*(_sentence_cited_ids(sentence, hits) for sentence in sentences)) if sentences else set()
    return {"qid": case["qid"], "accepted": accepted, "claim_coverage": coverage,
            "claims": claims, "cited_passage_ids": sorted(cited_ids),
            "has_gold_citation": valid_citation, "forbidden_terms_found": forbidden}
