"""Pure final-unit decision table; never mutate planner coverage or settings.

Three positional arguments implement section 2.1[F]. The optional keyword
context_hit_count is needed to distinguish zero client hits (no_context) from
nonzero client hits with no admitted windows (low_relevance). Engine callers
must pass len(context_hits); without it, presence is inferred from units.

Coverage is recomputed from final units at T_SLOT, using inclusive letter-year
bounds and eligible anchor quantities, never trusting stale filled_by. A stale
filled_by ID absent from final units identifies budget loss (capacity). A
missing period takes precedence over a missing quantity within the same slot;
reasons across slots preserve their order without duplicates. Too many period
slots is an unconditional refusal after the no-context check. No slots means
no slot filled, not vacuous "all filled": the planner should create a general
slot. Relevant units below T_SLOT yield low_relevance for unfilled general
slots. No context, low global relevance and too many periods are terminal.
"""

from __future__ import annotations

import math
from typing import Sequence

from .settings import GroundedSettings
from .types import DecisionResult, EvidenceUnit, ReasonCode, Slot

__all__ = ["decide"]


def _period_match(slot: Slot, unit: EvidenceUnit) -> bool:
    if slot.kind != "period":
        return True
    if unit.letter_year is None or slot.year_from is None:
        return False
    end = slot.year_to if slot.year_to is not None else slot.year_from
    return slot.year_from <= unit.letter_year <= end


def decide(
    slots: Sequence[Slot],
    units: Sequence[EvidenceUnit],
    settings: GroundedSettings,
    *,
    context_hit_count: int | None = None,
) -> DecisionResult:
    if context_hit_count is not None and (type(context_hit_count) is not int or context_hit_count < 0):
        raise ValueError("context_hit_count must be a non-negative integer")
    for unit in units:
        if (isinstance(unit.score, bool) or not isinstance(unit.score, (int, float)) or
                not math.isfinite(unit.score) or not 0.0 <= unit.score <= 1.0):
            raise ValueError("unit scores must be finite probabilities in [0, 1]")
    if context_hit_count == 0 or (context_hit_count is None and not units):
        return DecisionResult("refuse", ["no_context"])
    if sum(slot.kind == "period" for slot in slots) > settings.MAX_PERIODS:
        return DecisionResult("refuse", ["too_many_periods"])
    if max((unit.score for unit in units), default=-1.0) < settings.T_RELEVANT:
        return DecisionResult("refuse", ["low_relevance"])

    final_ids = {unit.eid for unit in units}
    eligible = [unit for unit in units if unit.score >= settings.T_SLOT]
    reasons: list[ReasonCode] = []
    filled = 0
    for slot in slots:
        matching = [unit for unit in eligible if _period_match(slot, unit)]
        if any(not slot.needs_number or bool(unit.quantities) for unit in matching):
            filled += 1
            continue
        if any(eid not in final_ids for eid in slot.filled_by):
            reason: ReasonCode = "capacity"
        elif slot.kind == "period" and not matching:
            reason = "missing_period"
        elif slot.needs_number and matching:
            reason = "missing_quantity"
        else:
            reason = "low_relevance"
        if reason not in reasons:
            reasons.append(reason)
    if not filled:
        return DecisionResult("refuse", reasons or ["low_relevance"])
    if filled < len(slots):
        return DecisionResult("partial", reasons)
    return DecisionResult("answer", [])
