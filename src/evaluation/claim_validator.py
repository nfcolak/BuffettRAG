"""Conservative post-generation claim/citation validation.

Lexical agreement is a deterministic guard, not semantic entailment. Typed
quantities and polarity are hard checks even when an NLI scorer is supplied.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, List, Sequence

from src.evaluation.citation_faithfulness import lexical_support_score

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_NUMBER_RE = re.compile(
    r"(?<![\w.])(?P<currency>[$£€]?)\s*(?P<value>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s*(?P<scale>billion|million|thousand|bn|[mb])\b)?"
    r"\s*(?P<percent>%|percent\b|per\s+cent\b)?", re.I,
)
_SCALES = {"thousand": 1000, "million": 1000000, "m": 1000000,
           "billion": 1000000000, "bn": 1000000000, "b": 1000000000}
_PREDICATE_NEGATION_RE = re.compile(
    r"\b(?:is|are|was|were|did|does|do|has|have|had|will|would|can|could)\s+not\b"
    r"|\b(?:isn't|aren't|wasn't|weren't|didn't|doesn't|don't|hasn't|haven't|hadn't|won't|wouldn't|can't|couldn't)\b"
    r"|\bnever\b|\bno\s+[A-Za-z]", re.I,
)
_STOPWORDS = frozenset(
    "a an and are as at be been but by did do does for from had has have in into is it its of on or that the their then these they this to was were with would our we me i you your".split()
)
_IGNORE = frozenset(
    "buffett warren berkshire hathaway said says wrote writes argued noted explained according "
    "called predicted letter letters shareholders he his him".split()
)
_SPLIT_RE = re.compile(r"([.!?](?:[ \t]*\[\d+(?:\s*,\s*\d+)*\])*)[ \t]+(?=(?:[o•▪◦][ \t]+)?[A-Z0-9\"'“‘(])")
_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs"})
_MIN_COVERAGE = 0.6


@dataclass
class ClaimValidationResult:
    safe_answer: str
    blocked_claims: List[str]
    validations: List[dict]
    nli_available: bool


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(str.maketrans({
        "’": "'", "‘": "'", "“": '"', "”": '"', "−": "-", "–": "-", "—": "-",
    }))
    text = _CITATION_RE.sub("", text)
    # Possessive typography must not manufacture an entity/token named 's'.
    text = re.sub(r"\b([A-Za-z]+)'s\b", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _quantities(text: str) -> set[tuple[Decimal, str]]:
    quantities = set()
    for match in _NUMBER_RE.finditer(_normalize(text)):
        value = Decimal(match.group("value").replace(",", ""))
        scale = (match.group("scale") or "").lower()
        value *= _SCALES.get(scale, 1)
        unit = "percent" if match.group("percent") else match.group("currency") or "number"
        quantities.add((value, unit))
    return quantities


def _is_boundary(text: str, match: re.Match) -> bool:
    if match.group(1).startswith(".") and "[" not in match.group(1):
        prefix = text[:match.start()]
        if re.match(r"\d", text[match.end():]) or re.search(
            r"(?:^|[.!?]\s)(?:In|During|For|By|Since|From|Between)\s+\d[\d, -]*$", prefix, re.I
        ):
            return False
        word = re.search(r"([A-Za-z]+)$", prefix)
        if word and (word.group(1).lower() in _ABBREVIATIONS or
                     (len(word.group(1)) == 1 and word.group(1).isupper())):
            return False
    return True


def _split_original(line: str) -> List[str]:
    """Citation-aware boundaries, preserving abbreviations and original text."""
    parts, start = [], 0
    for match in _SPLIT_RE.finditer(line):
        if _is_boundary(line, match):
            end = match.start() + len(match.group(1))
            parts.append(line[start:end])
            start = match.end()
    parts.append(line[start:])
    return [part for part in parts if part.strip()]


def evidence_sentences(text: str) -> List[str]:
    """Split source prose without breaking titles, initials, or year lists."""
    parts = _split_original(re.sub(r"\s+", " ", text).strip())
    joined: List[str] = []
    for part in parts:
        part = part.strip()
        if joined and (re.match(r"^\d[\d,.%$]*\b", part) or
                       re.fullmatch(r"(?:In|During|For|By|Since|From|Between)\s+\d[\d, -]*[.!?]", joined[-1], re.I)):
            joined[-1] += " " + part
        else:
            joined.append(part)
    # PDF bullet glyphs ("o ", "•") that opened a sentence are layout, not text.
    return [re.sub(r"^[o•▪◦]\s+(?=[A-Z0-9\"'“‘(])", "", part) for part in joined]


def _evidence_units(passage: str) -> List[str]:
    """Evidence sentences plus their clauses (split exactly like claims are).

    Claims are checked clause by clause, so the evidence must be checkable at the
    same granularity: a claim clause is compared with the clause that states it,
    not with a neighbouring clause's negation or numbers. Every unit still has to
    satisfy the full coverage/quantity/entity/polarity rules on its own.
    """
    units: List[str] = []
    for sentence in evidence_sentences(passage):
        units.append(sentence)
        clauses = split_claims(sentence)
        if len(clauses) > 1:
            units.extend(clauses)
    return units


def split_claims(text: str) -> List[str]:
    clean = _CITATION_RE.sub("", text).strip().rstrip(".!?")
    parts = [part.strip(" ,.;") for part in re.split(r"\s+(?:and|but)\s+|;", clean, flags=re.I)
             if part.strip(" ,.;")]
    # A conjunction between subjects is not a standalone attribution claim.
    if len(parts) > 1 and len(_TOKEN_RE.findall(parts[0])) <= 2:
        return [clean.strip(" ,.;")]
    return parts


def _hard_agreement(claim: str, sentence: str) -> bool:
    return (_quantities(claim).issubset(_quantities(sentence)) and
            bool(_PREDICATE_NEGATION_RE.search(_normalize(claim))) ==
            bool(_PREDICATE_NEGATION_RE.search(_normalize(sentence))))


def _nli_hard_agreement(claim: str, passage: str) -> bool:
    # An unrelated affirmative sentence must not license a negated claim's
    # opposite. Apply hard checks to the best lexical evidence, not any sentence.
    content = {t.lower() for t in _TOKEN_RE.findall(_normalize(claim))
               if t.lower() not in _STOPWORDS and t.lower() not in _IGNORE}
    candidates = [(len(content & {t.lower() for t in _TOKEN_RE.findall(_normalize(s))}), s)
                  for s in evidence_sentences(passage)]
    best = max((score for score, _ in candidates), default=0)
    return best > 0 and any(_hard_agreement(claim, s) for score, s in candidates if score == best)


def _deterministic_agreement(claim: str, passage: str) -> tuple[bool, float]:
    """Require content coverage, typed quantities, named entities and polarity."""
    normalized_claim = _normalize(claim)
    raw_tokens = _TOKEN_RE.findall(normalized_claim)
    claim_content = {t.lower() for t in raw_tokens if t.lower() not in _STOPWORDS and t.lower() not in _IGNORE}
    claim_entities = {
        t.lower() for t in raw_tokens[1:]
        if t[:1].isupper() and t.lower() not in _STOPWORDS and t.lower() not in _IGNORE
        and not t[:1].isdigit()
    }
    needed = 1.0 if len(claim_content) <= 3 else _MIN_COVERAGE
    best_score = 0.0
    for sentence in _evidence_units(_normalize(passage)):
        evidence_set = {token.lower() for token in _TOKEN_RE.findall(sentence)}
        coverage = len(claim_content & evidence_set) / len(claim_content) if claim_content else 0.0
        best_score = max(best_score, lexical_support_score(normalized_claim, sentence), coverage)
        if coverage >= needed and _hard_agreement(claim, sentence) and claim_entities.issubset(evidence_set):
            return True, max(best_score, coverage)
    return False, best_score


def validate_and_filter_answer(answer: str, hits: Sequence[Any], *, nli_scorer: Callable[[str, str], float] | None = None,
                               threshold: float = 0.2) -> ClaimValidationResult:
    """Drop unsupported sentences individually; supported sentences stay verbatim."""
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
            claims = split_claims(body)
            # Do not approve a numeric relation by splicing quantities from
            # different source sentences/periods. Check the unsplit relation too.
            if len(claims) > 1 and len(_quantities(body)) > 1:
                claims.append(_CITATION_RE.sub("", body).strip().rstrip(".!?"))
            for claim in claims:
                scores, agreements = [], []
                for index in valid:
                    passage = hits[index].text
                    if nli_scorer:
                        score = float(nli_scorer(passage, claim))
                        agreement = (score >= max(0.5, threshold) and
                                     _nli_hard_agreement(claim, passage))
                    else:
                        agreement, score = _deterministic_agreement(claim, passage)
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
