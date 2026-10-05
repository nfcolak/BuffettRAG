"""Deterministic verifier for one group's composed text.

Each answer sentence must (1) carry only well-formed markers [k] that all name
an existing group evidence sentence, (2) pass the legacy
validate_and_filter_answer against ONLY the group's evidence sentences, and
(3) satisfy the R1 numeric guard: a sentence that carries a quantity (years
included) must equal one of its cited source sentences verbatim after
whitespace / citation / quote-typography / trailing-punctuation / case
normalisation. Signs and qualifiers (pre-tax, per share, estimated, ...) are
therefore kept; numeric paraphrases are rejected. Dropped sentences are counted
by reason (counts only, no text) in VerifyOutcome.dropped. NLI is not used.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Sequence

from src.evaluation.claim_validator import (
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
            if not any(_comparable(evidence[i].text) == wanted for i in cited):
                outcome.count("numeric_guard")
                return None
        return VerifiedSentence(
            text=sentence,
            local_indexes=list(cited),
            unit_ids=[evidence[i].unit.eid for i in cited],
            hit_indexes=[evidence[i].hit_index for i in cited],
        )
