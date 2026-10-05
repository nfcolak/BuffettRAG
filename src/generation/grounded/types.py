"""Request-local values and group-local evidence for the grounded engine.

All hit indexes and local indexes are zero-based; citation markers are index + 1.
Slot.filled_by and EvidencePlan.groups contain EvidenceUnit.eid strings, never
list positions. Groups map a slot/group name to an ordered list of unit IDs;
"all" is the single non-period group. Slot kinds are "general" and "period";
period endpoints are inclusive LETTER years. Quantity sets use the legacy
validator's (Decimal value, unit string) shape, but contain only eligible anchor
quantities (not years, pages, ordinals or list markers).

GroupEvidence is the R2 adapter: one complete whitespace-normalised source
sentence per entry, in local-index order. Its .text is unescaped verification
text, .prompt_text escapes source bracket syntax only for the prompt, and .unit
and .hit_index preserve the local -> unit -> original client hit mapping. Keep
the original offset span separately; never verify the escaped prompt copy.
VerifiedSentence indexes describe surviving local citations and their provenance.

DecisionResult is the pure decision return value. Reasons are ordered, unique
codes. A GroupResult.note is an engine-written coverage note, not a factual
sentence. GroundedAnswer.raw_outputs maps group names to complete ComposeResults.
Timings are milliseconds: load_scorer/load_composer/load_nli separately from
score/plan/compose/verify/total; further load_* entries may be added by resources.
REFUSAL_LINE is a lazy re-export of the existing prompt constant, NOT a new
literal: plain contract imports must not trigger prompt/storage/config imports.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

__all__ = [
    "DecisionKind", "ReasonCode", "SlotKind", "FallbackReason", "REFUSAL_LINE",
    "SourceSpan", "EvidenceUnit", "Slot", "EvidencePlan", "GroupEvidence",
    "ComposeRequest", "ComposeResult", "VerifiedSentence", "GroupResult",
    "StageFailure", "GroundedAnswer", "DecisionResult",
]

DecisionKind = Literal["answer", "partial", "refuse"]
ReasonCode = Literal[
    "no_context", "low_relevance", "missing_quantity", "missing_period",
    "too_many_periods", "capacity",
]
SlotKind = Literal["general", "period"]
FallbackReason = Literal[
    "empty", "marker_only", "verifier_rejected", "model_refused", "backend_error",
]


def __getattr__(name: str) -> Any:
    if name == "REFUSAL_LINE":
        from src.generation.prompt import REFUSAL_LINE
        return REFUSAL_LINE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class SourceSpan:
    hit_index: int
    start: int
    end: int
    text: str


@dataclass
class EvidenceUnit:
    eid: str
    hit_index: int
    passage_id: str
    letter_year: int | None
    anchor: SourceSpan
    window: SourceSpan
    score: float
    quantities: set[tuple[Decimal, str]] = field(default_factory=set)
    negated: bool = False
    slots: list[str] = field(default_factory=list)


@dataclass
class Slot:
    name: str
    kind: SlotKind = "general"
    year_from: int | None = None
    year_to: int | None = None
    needs_number: bool = False
    filled_by: list[str] = field(default_factory=list)


@dataclass
class EvidencePlan:
    original_query: str
    scoring_query: str
    slots: list[Slot] = field(default_factory=list)
    units: list[EvidenceUnit] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)
    decision: DecisionKind = "refuse"
    reasons: list[ReasonCode] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupEvidence:
    local_index: int
    text: str
    unit: EvidenceUnit
    source_span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if type(self.local_index) is not int or self.local_index < 0:
            raise ValueError("local_index must be a non-negative integer")
        if not self.text.strip() or self.text != " ".join(self.text.split()):
            raise ValueError("text must be a complete whitespace-normalised source sentence")
        if self.source_span is not None and self.source_span.hit_index != self.unit.hit_index:
            raise ValueError("source_span and unit must refer to the same client hit")

    @property
    def hit_index(self) -> int:
        return self.unit.hit_index

    @property
    def id(self) -> str:
        return self.unit.passage_id

    @property
    def metadata(self) -> dict[str, Any]:
        return {"year": self.unit.letter_year}

    @property
    def prompt_text(self) -> str:
        return re.sub(r"\[([^\[\]\n]+)\]", r"(\1)", self.text)


@dataclass
class ComposeRequest:
    system: str
    question: str
    evidence: list[GroupEvidence]
    history_summary: str | None = None
    max_tokens: int = 200
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 200:
            raise ValueError("max_tokens must be an integer in [1, 200]")
        if (isinstance(self.temperature, bool) or
                not isinstance(self.temperature, (int, float)) or
                not math.isfinite(self.temperature) or not 0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be finite and in [0, 2]")
        if [entry.local_index for entry in self.evidence] != list(range(len(self.evidence))):
            raise ValueError("evidence must be ordered by contiguous zero-based local_index")


@dataclass
class ComposeResult:
    text: str
    raw: str
    backend: str
    model_id: str
    finish_reason: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    error_type: str | None = None


@dataclass
class VerifiedSentence:
    text: str
    local_indexes: list[int] = field(default_factory=list)
    unit_ids: list[str] = field(default_factory=list)
    hit_indexes: list[int] = field(default_factory=list)


@dataclass
class GroupResult:
    group: str
    text: str
    sentences: list[VerifiedSentence] = field(default_factory=list)
    fallback_reason: FallbackReason | None = None
    note: str | None = None


@dataclass
class StageFailure:
    stage: str
    error_type: str


@dataclass
class GroundedAnswer:
    answer: str
    citations: list[dict[str, Any]]
    plan: EvidencePlan
    groups: list[GroupResult] = field(default_factory=list)
    composer: str = ""
    model_id: str = ""
    raw_outputs: dict[str, ComposeResult] = field(default_factory=dict)
    failures: list[StageFailure] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=lambda: {
        "load_scorer": 0.0, "load_composer": 0.0, "load_nli": 0.0,
        "score": 0.0, "plan": 0.0, "compose": 0.0, "verify": 0.0, "total": 0.0,
    })


@dataclass
class DecisionResult:
    decision: DecisionKind
    reasons: list[ReasonCode] = field(default_factory=list)
