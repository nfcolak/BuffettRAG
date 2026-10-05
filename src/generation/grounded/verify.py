"""Deterministic verifier for one group's composed text.

Each answer sentence must (1) carry only well-formed markers [k] that all name
an existing group evidence sentence, (2) pass the legacy
validate_and_filter_answer against ONLY the group's evidence sentences, and
(3) satisfy the R1 numeric guard, in one of two modes (GROUNDED_NUMERIC_GUARD):

verbatim - a sentence that carries a quantity (years included) must equal one
  of its cited source sentences after whitespace / citation / quote-typography /
  trailing-punctuation / case normalisation. Numeric paraphrases are rejected.
bound (default) - a sentence with quantities passes if ONE cited source
  sentence (a) contains every typed quantity (value, unit, sign), (b) keeps the
  source order of the shared non-year quantities, (c) attaches exactly the same
  qualifiers (pre-tax, after-tax, per share, estimated, expected, projected,
  forecast, budgeted, approximately) to each quantity, and (d) shares, for each
  non-year quantity, a content word within 4 tokens of it in the answer with
  the 6 tokens around it in the source.

Dropped sentences are counted by reason (counts only, no text) in
VerifyOutcome.dropped. NLI is not used.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Sequence

from src.evaluation.claim_validator import (
    _NUMBER_RE,
    _SCALES,
    _STOPWORDS,
    _normalize,
    _quantities,
    evidence_sentences,
    validate_and_filter_answer,
)

from .types import GroupEvidence, VerifiedSentence

__all__ = ["DeterministicVerifier", "VerifyOutcome"]

_BRACKET_RE = re.compile(r"\[([^\[\]\n]*)\]")
_MARKER_BODY_RE = re.compile(r"\d+(?:\s*,\s*\d+)*")
_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "−": "-", "–": "-", "—": "-"})


_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
_NUMERIC_GUARDS = ("verbatim", "bound")
_ANSWER_WINDOW = 4
_SOURCE_WINDOW = 6
_QUALIFIER_WINDOW = 4
_WEAK_WORDS = frozenset(
    "year years only also about than more most very while when which who what where there here "
    "such some any all each both other our".split()
)
_QUALIFIER_STEMS = (
    ("expected", re.compile(r"expect(?:s|ed|ing|ation|ations)?")),
    ("estimated", re.compile(r"estimat(?:e|es|ed|ing)")),
    ("projected", re.compile(r"project(?:s|ed|ing|ion|ions)?")),
    ("forecast", re.compile(r"forecast(?:s|ed|ing)?")),
    ("budgeted", re.compile(r"budget(?:s|ed|ing)?")),
    ("approximately", re.compile(r"approximate(?:ly)?")),
)
_MERGED = {("pre", "tax"): "pretax", ("after", "tax"): "aftertax", ("per", "share"): "pershare"}
_QUALIFIER_TOKENS = {"pretax": "pre-tax", "aftertax": "after-tax", "pershare": "per share"}


@dataclass(frozen=True)
class _Quantity:
    key: tuple          # (value, unit, sign)
    year: bool


def _qualifier(word: str) -> str | None:
    if word in _QUALIFIER_TOKENS:
        return _QUALIFIER_TOKENS[word]
    for name, pattern in _QUALIFIER_STEMS:
        if pattern.fullmatch(word):
            return name
    return None


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _tokens(text: str) -> list[str | _Quantity]:
    """Words (lower-case) and typed quantities in text order.

    Quantities reuse claim_validator's extraction (same regex, same value/unit
    arithmetic) plus a sign; the pairs "pre tax", "after tax", "per share" merge
    into single qualifier tokens.
    """
    norm = _normalize(text)
    out: list[str | _Quantity] = []

    def words(chunk: str) -> None:
        for word in _WORD_RE.findall(chunk):
            word = word.lower().replace("-", "").replace("'", "")
            if out and isinstance(out[-1], str) and (out[-1], word) in _MERGED:
                out[-1] = _MERGED[(out[-1], word)]
            else:
                out.append(word)

    pos = 0
    for match in _NUMBER_RE.finditer(norm):
        words(norm[pos:match.start()])
        pos = match.end()
        raw = match.group("value")
        scale = (match.group("scale") or "").lower()
        value = Decimal(raw.replace(",", "")) * _SCALES.get(scale, 1)
        unit = "percent" if match.group("percent") else match.group("currency") or "number"
        before = norm[:match.start()].rstrip(" ")
        negative = bool(before) and (
            (before[-1] == "-" and not (len(before) > 1 and (before[-2].isalnum() or before[-2] in "$%")))
            or (before[-1] == "(" and norm[match.end():].lstrip().startswith(")"))
        )
        year = (unit == "number" and not scale and raw.isdigit() and len(raw) == 4
                and 1800 <= int(raw) <= 2100)
        out.append(_Quantity((value, unit, "-" if negative else "+"), year))
    words(norm[pos:])
    return out


def _qualifiers_at(tokens: list, index: int) -> frozenset[str]:
    """Qualifiers within the window around one quantity, not crossing another non-year quantity."""
    found: set[str] = set()
    for step in (-1, 1):
        pos = index + step
        for _ in range(_QUALIFIER_WINDOW):
            if not 0 <= pos < len(tokens):
                break
            token = tokens[pos]
            if isinstance(token, _Quantity):
                if not token.year:
                    break
            else:
                name = _qualifier(token)
                if name:
                    found.add(name)
            pos += step
    return frozenset(found)


def _content_words(tokens: list, index: int, radius: int) -> set[str]:
    lo, hi = max(0, index - radius), min(len(tokens), index + radius + 1)
    return {
        _stem(token) for pos, token in enumerate(tokens[lo:hi], lo)
        if pos != index and isinstance(token, str) and len(token) > 2
        and token not in _STOPWORDS and token not in _WEAK_WORDS and _qualifier(token) is None
    }


def _is_subsequence(wanted: list, pool: list) -> bool:
    it = iter(pool)
    return all(any(item == other for other in it) for item in wanted)


def _bound(answer_body: str, source: str) -> bool:
    ans, src = _tokens(answer_body), _tokens(source)
    ans_q = [i for i, t in enumerate(ans) if isinstance(t, _Quantity)]
    src_q = [i for i, t in enumerate(src) if isinstance(t, _Quantity)]
    src_keys = {src[i].key for i in src_q}
    if any(ans[i].key not in src_keys for i in ans_q):                      # (a)
        return False
    if not _is_subsequence([ans[i].key for i in ans_q if not ans[i].year],
                           [src[i].key for i in src_q if not src[i].year]):  # (b)
        return False
    for i in ans_q:
        matches = [j for j in src_q if src[j].key == ans[i].key]
        want_qualifiers = _qualifiers_at(ans, i)
        if not any(_qualifiers_at(src, j) == want_qualifiers for j in matches):  # (c)
            return False
        if ans[i].year:
            continue
        near = _content_words(ans, i, _ANSWER_WINDOW)                        # (d)
        if near and not any(near & _content_words(src, j, _SOURCE_WINDOW) for j in matches):
            return False
    return True


def _default_guard() -> str:
    mode = os.environ.get("GROUNDED_NUMERIC_GUARD", "bound")
    if mode not in _NUMERIC_GUARDS:
        raise ValueError("GROUNDED_NUMERIC_GUARD must be verbatim or bound")
    return mode


@dataclass
class VerifyOutcome:
    sentences: list[VerifiedSentence] = field(default_factory=list)
    dropped: dict[str, int] = field(default_factory=dict)

    def count(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1


def _comparable(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = _CITATION_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text.rstrip(" .!?;:\"')")


def _markers(sentence: str, size: int) -> tuple[list[int] | None, str | None]:
    """Return (zero-based cited indexes, None) or (None, drop reason)."""
    indexes: list[int] = []
    for inner in _BRACKET_RE.findall(sentence):
        if not _MARKER_BODY_RE.fullmatch(inner.strip()):
            return None, "malformed_marker"
        for number in (int(part) for part in inner.split(",")):
            if not 1 <= number <= size:
                return None, "unknown_marker"
            if number - 1 not in indexes:
                indexes.append(number - 1)
    if _BRACKET_RE.sub("", sentence).count("[") or _BRACKET_RE.sub("", sentence).count("]"):
        return None, "malformed_marker"
    if not indexes:
        return None, "unmarked"
    return sorted(indexes), None


class DeterministicVerifier:
    def __init__(self, numeric_guard: str | None = None) -> None:
        mode = _default_guard() if numeric_guard is None else numeric_guard
        if mode not in _NUMERIC_GUARDS:
            raise ValueError("numeric_guard must be verbatim or bound")
        self.numeric_guard = mode

    def verify(self, group_text: str, units: Sequence[GroupEvidence]) -> list[VerifiedSentence]:
        return self.verify_detailed(group_text, units).sentences

    def verify_detailed(self, group_text: str, units: Sequence[GroupEvidence]) -> VerifyOutcome:
        outcome = VerifyOutcome()
        evidence = list(units)
        if [e.local_index for e in evidence] != list(range(len(evidence))):
            raise ValueError("units must be ordered by contiguous zero-based local_index")
        for line in group_text.split("\n"):
            line = _BULLET_RE.sub("", line)
            if not line.strip():
                continue
            for sentence in evidence_sentences(line):
                sentence = sentence.strip()
                if not sentence:
                    continue
                verified = self._verify_sentence(sentence, evidence, outcome)
                if verified is not None:
                    outcome.sentences.append(verified)
        return outcome

    def _verify_sentence(self, sentence: str, evidence: list[GroupEvidence],
                         outcome: VerifyOutcome) -> VerifiedSentence | None:
        cited, reason = _markers(sentence, len(evidence))
        if cited is None:
            outcome.count(reason or "malformed_marker")
            return None
        legacy = validate_and_filter_answer(sentence, evidence)
        if not legacy.safe_answer or _comparable(legacy.safe_answer) != _comparable(sentence):
            outcome.count("unsupported")
            return None
        body = _CITATION_RE.sub("", sentence)
        if _quantities(body):
            wanted = _comparable(body)
            if self.numeric_guard == "bound":
                passed = any(_comparable(evidence[i].text) == wanted or _bound(body, evidence[i].text)
                             for i in cited)
            else:
                passed = any(_comparable(evidence[i].text) == wanted for i in cited)
            if not passed:
                outcome.count("numeric_guard")
                return None
        return VerifiedSentence(
            text=sentence,
            local_indexes=list(cited),
            unit_ids=[evidence[i].unit.eid for i in cited],
            hit_indexes=[evidence[i].hit_index for i in cited],
        )
