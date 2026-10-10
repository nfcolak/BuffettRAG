"""Deterministic scoring for human-curated answer-quality benchmark cases."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from src.evaluation.citation_faithfulness import split_sentences

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])\$?\d[\d,]*(?:\.\d+)?%?(?:bn|m|b)?", re.I)
_NEGATIONS = frozenset({"no", "not", "never", "neither", "nor", "without"})
# Predicate negation. A bare "no" is NOT here: idioms ("no matter", "no doubt", "of no importance") are not
# negations, so "no" counts only directly before a claim required term (see _has_negation).
_PREDICATE_NEGATION_RE = re.compile(
    r"\b(?:is|are|was|were|did|does|do|has|have|had|will|would|can|could|should|shall|must|may|might)\s+not\b"
    r"|\bcannot\b|\bnever\b|\bno\s+longer\b",
    re.I,
)
# Contractions become their spelled-out form before the negation check.
_CONTRACTION_RES = (
    (re.compile(r"\bwon[\u2019']t\b", re.I), "will not"),
    (re.compile(r"\bcan[\u2019']t\b", re.I), "can not"),
    (re.compile(r"\bshan[\u2019']t\b", re.I), "shall not"),
    (re.compile(r"(?<=[A-Za-z])n[\u2019']t\b", re.I), " not"),
)
# Clause boundaries: , ; : ( ) and but/although/while. A comma between digits ("$21,904,000") is not one.
_CLAUSE_SPLIT_RE = re.compile(r"(?<!\d),|,(?!\d)|[;:()]|\b(?:but|although|while)\b", re.I)


def _normalise_contractions(text: str) -> str:
    for pattern, replacement in _CONTRACTION_RES:
        text = pattern.sub(replacement, text)
    return text


# A predicate negation scopes a required term when the term starts within this many words after it. A term
# two words after it is scoped only when the words between (its verb) belong to the claim: "had not increased
# to $704 million" negates "increased to $704 million", "have never operated on Sunday" does not negate
# "closed on Sunday".
_NEGATION_SCOPE_WORDS = 3
_SCOPE_STOPWORDS = frozenset(
    "a an the of to in on at by for from with and or as is are was were be been being has have had it its this "
    "that these those our we us their his her they not never also already still even ever yet just only".split())


def _stem_set(text: str) -> set:
    return {w[:5] for w in re.findall(r"[a-z0-9$%]+", text.lower()) if len(w) >= 4 and w not in _SCOPE_STOPWORDS}


def _negation_scopes_term(tail: str, terms: Sequence[str], claim: str) -> bool:
    """`tail` = the lowered clause text after a predicate negation; `terms` are lowered."""
    words = list(re.finditer(r"\S+", tail))[:_NEGATION_SCOPE_WORDS]
    if not words:
        return False
    reach = words[-1].end()
    claim_stems = _stem_set(claim)
    for term in terms:
        index = tail.find(term)
        while index != -1 and index < reach:
            between = [w.group(0) for w in words if w.start() < index]
            if len(between) <= 1:
                return True
            content = [w.strip(".,;:()\"'\u2019") for w in between]
            content = [w for w in content if w and w not in _SCOPE_STOPWORDS]
            if not content or any(w[:5] in claim_stems for w in content if len(w) >= 4):
                return True
            index = tail.find(term, index + 1)
    return False


def _has_negation(text: str, terms: Sequence[str] = (), scoped: bool = False, claim: str = "") -> bool:
    """Does the text negate? With `scoped`, a predicate negation counts only when it scopes a required term
    of the claim (`claim` is the reference wording, used to tell a negated claim verb from another verb)."""
    text = _normalise_contractions(text)
    lowered = text.lower()
    lowered_terms = [t.lower() for t in terms if t]
    for match in _PREDICATE_NEGATION_RE.finditer(text):
        if not scoped or not lowered_terms:
            return True
        if any(match.start() <= lowered.find(t, match.start()) < match.end() for t in lowered_terms):
            return True  # the term is the negation itself
        if _negation_scopes_term(lowered[match.end():], lowered_terms, claim):
            return True
    for term in lowered_terms:
        if term.startswith("no ") and term in lowered:  # the required term itself is the "no X" phrase
            return True
        if re.search(r"\bno\s+" + re.escape(term), lowered):
            return True
    return False


def _best_clauses(sentence: str, terms: Sequence[str]) -> List[str]:
    """The clause(s) of the sentence containing the most required terms (all of them on a tie)."""
    clauses = [c for c in _CLAUSE_SPLIT_RE.split(sentence) if c and c.strip()]
    if not clauses:
        return [sentence]
    lowered_terms = [t.lower() for t in terms if t]
    counts = [sum(t in c.lower() for t in lowered_terms) for c in clauses]
    if max(counts) == 0:  # terms straddle clause boundaries: judge the whole sentence
        return [sentence]
    return [c for c, n in zip(clauses, counts) if n == max(counts)]


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


def _polarity_and_numbers_agree(reference: str, sentence: str, terms: Sequence[str] = (),
                                extra_numbers: Sequence[str] = ()) -> bool:
    """Polarity is compared inside the sentence clause holding the most claim required terms.

    `extra_numbers` (e.g. the year of a stripped, hit-supported label) count for the number check only.
    """
    # On a tie between clauses every tied clause must agree (a tie never relaxes the check).
    # A negated claim needs a negation anywhere in the clause; a non-negated claim is contradicted only by a
    # negation that scopes one of its required terms (see _has_negation).
    clauses = _best_clauses(_normalise_contractions(sentence), terms)
    reference_negative = _has_negation(reference, terms)
    if len(clauses) > 1:
        # Tied clauses hold different terms ("is needed; has no close substitute; is not regulated"): each is
        # compared with the polarity of the reference clause(s) holding the same terms, not with one global flag.
        reference_clauses = [c for c in _CLAUSE_SPLIT_RE.split(_normalise_contractions(reference)) if c and c.strip()]
        lowered_terms = [t.lower() for t in terms if t]

        def expected(clause: str) -> bool:
            held = [t for t in lowered_terms if t in clause.lower()]
            matching = [rc for rc in reference_clauses if any(t in rc.lower() for t in held)]
            return any(_has_negation(rc, terms) for rc in matching) if matching else reference_negative

        if any(_has_negation(clause, terms, not expected(clause), reference) != expected(clause) for clause in clauses):
            return False
    elif any(_has_negation(clause, terms, not reference_negative, reference) != reference_negative for clause in clauses):
        return False
    expected_numbers = {value.lower().replace(",", "") for value in _NUMBER_RE.findall(reference)}
    actual_numbers = {value.lower().replace(",", "") for value in _NUMBER_RE.findall(sentence)}
    actual_numbers.update(str(value).lower().replace(",", "") for value in extra_numbers)
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
                and _polarity_and_numbers_agree(claim["claim"], sentence, terms)
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


def is_unanswerable_case(case: Dict[str, Any]) -> bool:
    """Explicit abstention fixtures have no invented gold passage or claim."""
    return case.get("answerable") is False or case.get("question_type") == "unanswerable"


def validate_live_benchmark_case(case: Dict[str, Any]) -> None:
    """Validate the existing schema plus its additive exact-refusal extension."""
    from src.generation.prompt import REFUSAL_LINE

    if is_unanswerable_case(case):
        required = {"qid", "query", "gold_passage_ids", "gold_claims", "accept", "reject", "expected_answer"}
        if required - set(case) or not case.get("qid") or not case.get("query"):
            raise ValueError("Invalid unanswerable benchmark case")
        if case["gold_passage_ids"] != [] or case["gold_claims"] != [] or case.get("relevant_ids", []) != []:
            raise ValueError("Unanswerable cases must not invent gold evidence")
        if case["expected_answer"] != REFUSAL_LINE or not case["accept"].get("require_exact_refusal"):
            raise ValueError("Unanswerable cases require the exact refusal line")
        if case["accept"].get("require_valid_citation"):
            raise ValueError("Refusals must not require citations")
        return
    validate_benchmark_case(case)
    gold_ids = set(case["gold_passage_ids"])
    for claim in case["gold_claims"]:
        accepted_ids = claim.get("gold_passage_ids")
        if not isinstance(accepted_ids, list) or not accepted_ids or not set(accepted_ids).issubset(gold_ids):
            raise ValueError("Each live-benchmark claim needs its own accepted gold_passage_ids")
    if not case["accept"].get("require_valid_citation"):
        raise ValueError("Scored answers require claim-level gold citations")


def validate_fixture_ids(cases: Sequence[Dict[str, Any]], docs: Sequence[Any]) -> Dict[str, Any]:
    """Check the entire fixture before limiting cases or initializing any models."""
    corpus_ids = {doc.id for doc in docs}
    if len(corpus_ids) != len(docs):
        raise ValueError("Corpus contains duplicate passage IDs")
    fixture_ids = set()
    qids = set()
    for case in cases:
        validate_live_benchmark_case(case)
        if case["qid"] in qids:
            raise ValueError("Duplicate benchmark qid")
        qids.add(case["qid"])
        fixture_ids.update(case["gold_passage_ids"])
        fixture_ids.update(case.get("relevant_ids", []))
        for claim in case["gold_claims"]:
            fixture_ids.update(claim["gold_passage_ids"])
    missing = sorted(fixture_ids - corpus_ids)
    if missing:
        raise ValueError(f"Missing fixture passage IDs: {', '.join(missing)}")
    return {"checked_fixture_ids": len(fixture_ids), "missing_fixture_ids": [], "missing_fixture_id_count": 0}


def evaluate_live_answer(answer: str, hits: Sequence[Any], case: Dict[str, Any]) -> Dict[str, Any]:
    """Reuse development scoring, reporting accepted evidence for every claim."""
    from src.generation.prompt import REFUSAL_LINE

    validate_live_benchmark_case(case)
    if is_unanswerable_case(case):
        correct = answer == REFUSAL_LINE and not _CITATION_RE.search(answer)
        return {"qid": case["qid"], "accepted": correct, "claim_coverage": None,
                "claims": [], "cited_passage_ids": [], "has_gold_citation": False,
                "forbidden_terms_found": [], "refusal_correct": correct}
    score = evaluate_answer(answer, hits, case)
    for row, claim in zip(score["claims"], case["gold_claims"]):
        accepted_ids = set(claim["gold_passage_ids"])
        supporting_ids = set()
        for sentence in row["supporting_sentences"]:
            supporting_ids.update(_sentence_cited_ids(sentence, hits) & accepted_ids)
        row.update({"required_terms": claim["required_terms"],
                    "accepted_passage_ids": sorted(accepted_ids),
                    "supporting_passage_ids": sorted(supporting_ids)})
    score["refusal_correct"] = None
    return score
