"""Evidence planning for the grounded engine (plan 2.1 steps A, B, C, E, F).

Public surface: build_plan, group_evidence, source_spans.

[A] Source spans keep character offsets into hit.text; sentence boundaries are
those of claim_validator.evidence_sentences, computed on a whitespace-collapsed
copy with an index map back to the original text, so span.text is a true
substring. Header paragraphs (short, unpunctuated, own paragraph) are never
anchors and split sentence runs, but stay inside a window that spans them.
A unit's window is anchor +/- 1 sentence of the same hit.
[B] Slots come from original_query only: single years, inclusive ranges written
with a dash or "through", and decades. Several explicit years are separate
period slots ("from 1985 to 1995" is two slots, like the retriever's own
temporal-comparison split). history never creates slots or needs_number; it
only resolves scoring_query through build_followup_retrieval_query.
[C] Candidate windows: MAX_WINDOWS budget, SLOT_RESERVE best-lexical windows per
period slot first, then global lexical fill. With period slots only windows
whose hit letter year lies in some period slot are candidates.
[E] Selection: per slot the best satisfying window, then fill by score (floor
T_SLOT) to MAX_UNITS; max MAX_PER_HIT per hit; dedupe inside the same hit only.
[F] decision.decide on the final units.

Local indexes in group_evidence are zero-based (frozen GroupEvidence /
ComposeRequest contract); the citation marker is local_index + 1.
Unit ids are "h<hit_index>:<anchor.start>".
"""

from __future__ import annotations

import math
import re
from decimal import Decimal
from functools import lru_cache
from typing import Any, Mapping, Sequence

from src.evaluation.claim_validator import (
    _NUMBER_RE, _PREDICATE_NEGATION_RE, _SCALES, _SPLIT_RE, _is_boundary, _normalize,
    _quantities,
)
from src.retrieval.bm25 import _search_tokens, tokenize
from src.retrieval.query_expansion import build_followup_retrieval_query

from .decision import decide
from .protocols import SentenceScorer
from .settings import GroundedSettings
from .types import EvidencePlan, EvidenceUnit, GroupEvidence, Slot, SourceSpan

__all__ = ["build_plan", "group_evidence", "source_spans"]

_Y = r"(?:19[7-9]\d|20[0-2]\d)"
_PERIOD_RE = re.compile(
    rf"(?<![\w$,.])(?:"
    rf"(?P<a>{_Y})\s*(?:-|–|—|\bthrough\b|\bthru\b)\s*(?P<b>{_Y})"
    rf"|(?P<dec>(?:19[7-9]|20[0-2])0)s"
    rf"|(?P<word>seventies|eighties|nineties)"
    rf"|(?P<y>{_Y}))(?!\w|,\d|\.\d)",
    re.I,
)
_DECADE_WORDS = {"seventies": 1970, "eighties": 1980, "nineties": 1990}
_NEEDS_NUMBER_RE = re.compile(
    r"\bhow\s+(?:much|many)\b|\bwhat\s+percent|\bratios?\b|\brates?\b|\bmargins?\b"
    r"|\bprices?\b|\bcosts?\b|\bearnings\b|\bvalue\s+of\b|\bgive\s+the\b|[$%]",
    re.I,
)
_YEAR_TOKEN_RE = re.compile(r"(?:19|20)\d\ds?")
_MONTH_BEFORE_RE = re.compile(
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s*$", re.I,
)
_LABEL_BEFORE_RE = re.compile(
    r"\b(?:pages?|pp?\.?|notes?|items?|sections?|tables?|figures?|exhibits?|chapters?|no\.?)\s*$",
    re.I,
)
_SALUTATION_RE = re.compile(
    r"(?:To|Dear)\s+(?:the\s+)?(?:Stockholders|Shareholders|Shareowners|Owners|Partners)\b[^:]{0,120}:\s*"
)
_MIN_ANCHOR_WORDS = 3
_MAX_HEADER_WORDS = 12


# --- hit access -------------------------------------------------------------

def _get(hit: Any, name: str, default: Any = None) -> Any:
    if isinstance(hit, Mapping):
        return hit.get(name, default)
    return getattr(hit, name, default)


def _hit_text(hit: Any) -> str:
    text = _get(hit, "text", "")
    return text if isinstance(text, str) else ""


def _letter_year(hit: Any) -> int | None:
    meta = _get(hit, "metadata") or {}
    year = meta.get("year") if isinstance(meta, Mapping) else None
    if isinstance(year, bool):
        return None
    if isinstance(year, int):
        return year
    if isinstance(year, str) and re.fullmatch(r"\d{4}", year.strip()):
        return int(year)
    return None


# --- [A] source spans -------------------------------------------------------

def _is_header(paragraph: str) -> bool:
    body = paragraph.strip()
    if not body or "\n" in body:
        return False
    text = " ".join(body.split())
    words = text.split()
    if len(words) > _MAX_HEADER_WORDS or text[-1] in ".!?;,:" or not re.search(r"[A-Za-z]", text):
        return False
    if not (text[0].isupper() or text[0].isdigit()):
        return False
    numeric = sum(bool(re.fullmatch(r"[$£€]?[\d.,%()-]+", word)) for word in words)
    return numeric * 2 < len(words)


def _sentence_ranges(text: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """evidence_sentences boundaries over text[lo:hi], as original-text ranges."""
    chars: list[str] = []
    index: list[int] = []
    space_at = -1
    for pos in range(lo, hi):
        char = text[pos]
        if char.isspace():
            if chars and space_at < 0:
                space_at = pos
            continue
        if space_at >= 0:
            chars.append(" ")
            index.append(space_at)
            space_at = -1
        chars.append(char)
        index.append(pos)
    norm = "".join(chars)
    parts: list[list[int]] = []
    start = 0
    for match in _SPLIT_RE.finditer(norm):
        if _is_boundary(norm, match):
            end = match.start() + len(match.group(1))
            parts.append([start, end])
            start = match.end()
    parts.append([start, len(norm)])
    joined: list[list[int]] = []
    for part in parts:
        piece = norm[part[0]:part[1]]
        if not piece.strip():
            continue
        if joined and (re.match(r"^\d[\d,.%$]*\b", piece) or re.fullmatch(
                r"(?:In|During|For|By|Since|From|Between)\s+\d[\d, -]*[.!?]",
                norm[joined[-1][0]:joined[-1][1]], re.I)):
            joined[-1][1] = part[1]
        else:
            joined.append(part)
    return [(index[a], index[b - 1] + 1) for a, b in joined]


@lru_cache(maxsize=256)
def _pieces(text: str) -> tuple[tuple[int, int, bool], ...]:
    """(start, end, is_header) in text order; sentences never cross a header."""
    paragraphs = []
    for match in re.finditer(r"(?:[^\n]|\n(?![ \t]*\n))+", text):
        body = match.group(0)
        lead = len(body) - len(body.lstrip())
        paragraphs.append((match.start() + lead, match.start() + len(body.rstrip()), body))
    out: list[tuple[int, int, bool]] = []
    run_lo: int | None = None
    run_hi = 0

    def flush() -> None:
        nonlocal run_lo
        if run_lo is not None:
            out.extend((a, b, False) for a, b in _sentence_ranges(text, run_lo, run_hi))
            run_lo = None

    for lo, hi, body in paragraphs:
        if lo >= hi:
            continue
        if _is_header(body):
            flush()
            out.append((lo, hi, True))
        else:
            if run_lo is None:
                run_lo = lo
            run_hi = hi
    flush()
    return tuple(out)


def source_spans(text: str, hit_index: int = 0, *, include_headers: bool = False) -> list[SourceSpan]:
    """Sentence spans with offsets; span.text == text[start:end] exactly."""
    return [
        SourceSpan(hit_index, start, end, text[start:end])
        for start, end, header in _pieces(text) if include_headers or not header
    ]


# --- [B] slots --------------------------------------------------------------

def _parse_periods(query: str) -> list[Slot]:
    slots: list[Slot] = []
    seen: set[tuple[int, int]] = set()
    for match in _PERIOD_RE.finditer(query):
        if match.group("a"):
            low, high = sorted((int(match.group("a")), int(match.group("b"))))
            name = f"{low}-{high}" if low != high else str(low)
        elif match.group("dec") or match.group("word"):
            low = int(match.group("dec")) if match.group("dec") else _DECADE_WORDS[match.group("word").lower()]
            high, name = low + 9, f"{low}s"
        else:
            low = high = int(match.group("y"))
            name = str(low)
        if (low, high) not in seen:
            seen.add((low, high))
            slots.append(Slot(name, "period", low, high))
    return slots


def _slot_match(slot: Slot, year: int | None) -> bool:
    if slot.kind != "period":
        return True
    if year is None or slot.year_from is None:
        return False
    return slot.year_from <= year <= (slot.year_to if slot.year_to is not None else slot.year_from)


def _eligible_quantities(text: str) -> set[tuple[Decimal, str]]:
    """Anchor quantities excluding years, pages, ordinals, list markers, dates, labels."""
    norm = _normalize(text)
    found: set[tuple[Decimal, str]] = set()
    for match in _NUMBER_RE.finditer(norm):
        raw = match.group("value").rstrip(",")
        scale = (match.group("scale") or "").lower()
        currency, percent = match.group("currency"), bool(match.group("percent"))
        before, after = norm[:match.start("value")], norm[match.end("value"):]
        if not (currency or scale or percent):
            if re.fullmatch(r"(?:19|20)\d\d", raw) or re.match(r"(?:st|nd|rd|th)\b", after, re.I):
                continue
            if before.rstrip().endswith("(") and after.startswith(")"):
                continue
            if not before.strip() and after[:1] in (".", ")"):
                continue
            if _MONTH_BEFORE_RE.search(before) or _LABEL_BEFORE_RE.search(before):
                continue
        value = Decimal(raw.replace(",", "")) * _SCALES.get(scale, 1)
        found.add((value, "percent" if percent else currency or "number"))
    return found


# --- [C]-[E] candidates -----------------------------------------------------

class _Cand:
    __slots__ = ("hit_index", "start", "anchor", "window", "norm", "year", "passage_id",
                 "anchor_tokens", "window_tokens", "quantities", "all_quantities", "negated",
                 "lex", "score")

    def __init__(self, hit_index, anchor, window, year, passage_id):
        self.hit_index, self.start = hit_index, anchor.start
        self.anchor, self.window, self.year, self.passage_id = anchor, window, year, passage_id
        self.norm = " ".join(window.text.split())
        self.anchor_tokens = set(_search_tokens(anchor.text))
        self.window_tokens = set(_search_tokens(window.text))
        self.quantities = _eligible_quantities(anchor.text)
        self.all_quantities = _quantities(anchor.text)
        self.negated = bool(_PREDICATE_NEGATION_RE.search(_normalize(anchor.text)))
        self.lex = 0.0
        self.score = 0.0

    @property
    def eid(self) -> str:
        return f"h{self.hit_index}:{self.start}"

    def rank(self) -> tuple[float, int, int]:
        return (-self.score, self.hit_index, self.start)


def _candidates(context_hits: Sequence[Any]) -> list[_Cand]:
    out: list[_Cand] = []
    for hit_index, hit in enumerate(context_hits):
        text = _hit_text(hit)
        sentences = source_spans(text, hit_index)
        year, passage_id = _letter_year(hit), str(_get(hit, "id", "") or "")
        for i, anchor in enumerate(sentences):
            if len(re.findall(r"[A-Za-z]+", anchor.text)) < _MIN_ANCHOR_WORDS:
                continue
            first, last = sentences[max(0, i - 1)], sentences[min(len(sentences) - 1, i + 1)]
            window = SourceSpan(hit_index, first.start, last.end, text[first.start:last.end])
            out.append(_Cand(hit_index, anchor, window, year, passage_id))
    return out


def _lexical(cands: list[_Cand], scoring_query: str) -> None:
    query = sorted({t for t in _search_tokens(scoring_query) if not _YEAR_TOKEN_RE.fullmatch(t)})
    n = len(cands)
    df = {t: sum(t in c.window_tokens for c in cands) for t in query}
    idf = {t: math.log(1.0 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in query}
    for cand in cands:
        cand.lex = sum(
            idf[t] * (1.0 if t in cand.anchor_tokens else 0.5)
            for t in query if t in cand.window_tokens
        )


def _lex_rank(cand: _Cand) -> tuple[float, int, int]:
    return (-cand.lex, cand.hit_index, cand.start)


def _duplicate(a: _Cand, b: _Cand, jaccard: float) -> bool:
    if a.hit_index != b.hit_index:
        return False
    ta, tb = set(tokenize(a.anchor.text)), set(tokenize(b.anchor.text))
    union = ta | tb
    similar = (len(ta & tb) / len(union) if union else 1.0) >= jaccard
    return similar and a.all_quantities == b.all_quantities and a.negated == b.negated


def _satisfies(slot: Slot, cand: _Cand) -> bool:
    return _slot_match(slot, cand.year) and (not slot.needs_number or bool(cand.quantities))


# --- public: build_plan -----------------------------------------------------

def build_plan(
    original_query: str,
    context_hits: Sequence[Any],
    *,
    history: Sequence[Mapping[str, str]] = (),
    scorer: SentenceScorer,
    settings: GroundedSettings,
) -> EvidencePlan:
    history_list = [dict(turn) for turn in history]
    scoring_query = build_followup_retrieval_query(original_query, history_list or None)
    needs_number = bool(_NEEDS_NUMBER_RE.search(original_query))
    period_slots = _parse_periods(original_query)
    slots = period_slots or [Slot("all", "general")]
    for slot in slots:
        slot.needs_number = needs_number
    trace: dict[str, Any] = {
        "provenance": "units cite the context hits (client list), not original corpus paragraphs",
        "needs_number": needs_number, "followup": scoring_query != original_query,
        "slots": [slot.name for slot in slots], "hits": len(context_hits),
        "thresholds": {"T_RELEVANT": settings.T_RELEVANT, "T_SLOT": settings.T_SLOT},
        "windows_budget": settings.MAX_WINDOWS, "scoring_batches": 0,
    }
    plan = EvidencePlan(original_query, scoring_query, slots, trace=trace)

    if len(period_slots) > settings.MAX_PERIODS or not context_hits:
        trace.update(candidates_total=0, windows_scored=0, candidates=[], selected=[])
        return _decide(plan, [], context_hits, settings)

    cands = _candidates(context_hits)
    pool = [c for c in cands if any(_slot_match(s, c.year) for s in period_slots)] if period_slots else cands
    _lexical(pool, scoring_query)

    chosen: dict[str, _Cand] = {}
    reserved: dict[str, int] = {}
    for slot in period_slots:
        best = sorted((c for c in pool if _slot_match(slot, c.year)), key=_lex_rank)[:settings.SLOT_RESERVE]
        reserved[slot.name] = len(best)
        for cand in best:
            chosen.setdefault(cand.eid, cand)
    for cand in sorted(pool, key=_lex_rank):
        if len(chosen) >= settings.MAX_WINDOWS:
            break
        chosen.setdefault(cand.eid, cand)
    windows = sorted(chosen.values(), key=lambda c: (c.hit_index, c.start))
    trace.update(candidates_total=len(cands), pool=len(pool), windows_scored=len(windows),
                 windows_reserved=reserved)

    if windows:
        raw = scorer.score(scoring_query, [c.norm for c in windows])
        if len(raw) != len(windows):
            raise ValueError("scorer returned a different number of scores than windows")
        trace["scoring_batches"] = 1
        for cand, value in zip(windows, raw):
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("scorer returned a non-finite score")
            cand.score = min(1.0, max(0.0, value))
    trace["candidates"] = [
        {"eid": c.eid, "hit_index": c.hit_index, "start": c.start, "lex": round(c.lex, 6),
         "score": c.score} for c in windows
    ]

    selected = _select(slots, windows, settings, trace)
    return _decide(plan, selected, context_hits, settings)


def _select(slots: list[Slot], windows: list[_Cand], settings: GroundedSettings,
            trace: dict[str, Any]) -> list[_Cand]:
    ranked = sorted(windows, key=_Cand.rank)
    selected: list[_Cand] = []
    per_hit: dict[int, int] = {}
    dropped = {"per_hit": 0, "dedupe": 0, "below_t_slot": 0}

    def admit(cand: _Cand) -> bool:
        if per_hit.get(cand.hit_index, 0) >= settings.MAX_PER_HIT:
            dropped["per_hit"] += 1
            return False
        if any(_duplicate(cand, other, settings.DEDUPE_JACCARD) for other in selected):
            dropped["dedupe"] += 1
            return False
        selected.append(cand)
        per_hit[cand.hit_index] = per_hit.get(cand.hit_index, 0) + 1
        return True

    picks: dict[str, str | None] = {}
    for slot in slots:
        picks[slot.name] = None
        for cand in ranked:
            if not _satisfies(slot, cand):
                continue
            if cand in selected or admit(cand):
                picks[slot.name] = cand.eid
                break
    has_periods = any(s.kind == "period" for s in slots)
    for cand in ranked:
        if len(selected) >= settings.MAX_UNITS:
            break
        if cand in selected:
            continue
        if has_periods and not any(_slot_match(s, cand.year) for s in slots):
            continue
        if cand.score < settings.T_SLOT:
            dropped["below_t_slot"] += 1
            continue
        admit(cand)
    trace.update(slot_picks=picks, dropped=dropped)
    return selected


def _decide(plan: EvidencePlan, selected: list[_Cand], context_hits: Sequence[Any],
            settings: GroundedSettings) -> EvidencePlan:
    units = [
        EvidenceUnit(c.eid, c.hit_index, c.passage_id, c.year, c.anchor, c.window, c.score,
                     set(c.quantities), c.negated, [])
        for c in sorted(selected, key=_Cand.rank)
    ]
    for slot in plan.slots:
        slot.filled_by = []
    for unit, cand in zip(units, sorted(selected, key=_Cand.rank)):
        for slot in plan.slots:
            if _satisfies(slot, cand):
                unit.slots.append(slot.name)
                if cand.score >= settings.T_SLOT:
                    slot.filled_by.append(unit.eid)
    order = lambda u: (u.hit_index, u.anchor.start)  # noqa: E731
    plan.units = units
    if any(slot.kind == "period" for slot in plan.slots):
        plan.groups = {
            slot.name: [u.eid for u in sorted(units, key=order)
                        if _slot_match(slot, u.letter_year)]
            for slot in plan.slots
        }
    else:
        plan.groups = {"all": [u.eid for u in sorted(units, key=order)]}
    result = decide(plan.slots, units, settings, context_hit_count=len(context_hits))
    plan.decision, plan.reasons = result.decision, list(result.reasons)
    plan.trace["selected"] = [u.eid for u in units]
    plan.trace["unit_scores"] = {u.eid: u.score for u in units}
    plan.trace["decision"] = {"decision": plan.decision, "reasons": list(plan.reasons)}
    return plan


# --- public: group_evidence -------------------------------------------------

def group_evidence(plan: EvidencePlan, group: str, context_hits: Sequence[Any]) -> list[GroupEvidence]:
    """R2 rendering: one GroupEvidence per complete source sentence of the group's windows."""
    if group not in plan.groups:
        raise KeyError(f"unknown group {group!r}")
    by_eid = {unit.eid: unit for unit in plan.units}
    out: list[GroupEvidence] = []
    seen: set[tuple[int, int]] = set()
    for eid in plan.groups[group]:
        unit = by_eid[eid]
        text = _hit_text(context_hits[unit.hit_index])
        window = unit.window
        if text[window.start:window.end] != window.text:
            raise ValueError(f"context hit {unit.hit_index} no longer matches unit {eid}")
        for start, end, header in _pieces(text):
            if header or start < window.start or end > window.end or (unit.hit_index, start) in seen:
                continue
            seen.add((unit.hit_index, start))
            # a letter salutation ("To the Stockholders of ...:") is not part of the sentence;
            # the rendered span starts after it, so span.text == text[start:end] still holds
            salutation = _SALUTATION_RE.match(text, start, end)
            shown = salutation.end() if salutation else start
            if shown >= end:
                continue
            out.append(GroupEvidence(
                len(out), " ".join(text[shown:end].split()), unit,
                SourceSpan(unit.hit_index, shown, end, text[shown:end]),
            ))
    return out
