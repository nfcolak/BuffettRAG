"""Period-isolated grounded generation shared by HTTP, SSE and the pipeline."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.evaluation.claim_validator import ClaimValidationResult, validate_and_filter_answer
from src.generation.evidence_gate import assess_evidence
from src.generation.prompt import (
    REFUSAL_LINE, build_cited_prompt, format_answer_markdown, parse_citations, strip_chat_artifacts,
)
from src.retrieval.bm25 import _meta_matches
from src.retrieval.context import expand_hits_with_neighbors, fit_context_to_llm
from src.generation.text_relevance import content_words
from src.retrieval.retriever import detect_temporal_comparison, explicit_periods
from src.storage import SearchHit

_CITATIONS = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Each period's list keeps its anchors plus neighbours up to this many passages (token budget still applies).
PERIOD_PASSAGES = 12
SOFT_GATE_FLOOR = 0.15
_CLAUSE_SPLIT_RE = re.compile(r"[,;:?]|\s+(?:and|with|versus|vs\.?)\s+", re.IGNORECASE)
# Closing quote + citation marker ends a sentence even without . ! ?
_QUOTE_CITE_END_RE = re.compile(r"([\"\u201d\u2019']\s*(?:\[\d+(?:\s*,\s*\d+)*\])+)[ \t]+(?=[A-Z0-9\"\u201c(])")


def comparison_periods(query: str, history=None) -> List[Dict[str, Any]]:
    return detect_temporal_comparison(query, history) or []


def period_focused_query(query: str, period: Dict[str, Any], periods: Sequence[Dict[str, Any]]) -> str:
    """The question minus the sub-ask and year tokens that belong to the OTHER periods.

    The query is cut into clauses; a clause that names only other periods is
    dropped. A clause with no content words of its own ("... 2008 and 2020")
    stays attached to the previous clause, so a shared stem is never lost.
    """
    pieces, last = [], 0
    for match in _CLAUSE_SPLIT_RE.finditer(query):
        pieces.append(query[last:match.start()])
        last = match.end()
    pieces.append(query[last:])
    clauses: List[str] = []
    for piece in pieces:
        if clauses and not content_words(re.sub(r"\b(?:19|20)\d{2}s?\b", " ", piece)):
            clauses[-1] += " " + piece
        elif piece.strip():
            clauses.append(piece)
    kept = []
    for clause in clauses:
        named = explicit_periods(clause)
        if named and period not in named:
            continue
        kept.append(clause)
    others = [p for p in periods if p != period]
    text = " ".join(kept)
    for other in others:
        if isinstance(other["year"], int):
            text = re.sub(rf"\b{other['year']}\b", " ", text)
    return re.sub(r"\s+", " ", text).strip() or query


def _soft_gate() -> bool:
    return os.environ.get("EVIDENCE_GATE_SOFT", "") == "1"


def _join_part(part: str) -> str:
    """One paragraph per period, but never merge two sentences into one line.

    Line breaks the model wrote are kept (the scorer splits answers per line);
    a sentence closed by a quote and a citation marker is broken onto its own line.
    """
    lines = []
    for line in part.splitlines():
        line = _QUOTE_CITE_END_RE.sub(r"\1\n", re.sub(r"[ \t]+", " ", line).strip())
        lines.extend(piece for piece in line.split("\n") if piece.strip())
    return "\n".join(lines)


def period_label(period: Dict[str, Any]) -> str:
    year = period["year"]
    if not isinstance(year, dict):
        return str(year)
    start, end = year["$gte"], year["$lte"]
    return f"{start}s" if start % 10 == 0 and end == start + 9 else f"{start}–{end}"


def prepare_answer_context(
    llm, hits, docs_by_id, query, *, history=None, followup=False,
    neighbors: int, max_chars: int, max_new_tokens: int, n_ctx: int,
    max_passages: int, passage_max_chars: int,
) -> List[SearchHit]:
    """Fit each isolated comparison prompt, not a never-generated global prompt.

    Neighbors are independently citable for comparisons and resolved follow-ups.
    The combined client list is the union of the exact per-period prompt lists;
    each model call still obeys the configured passage/window limits.
    """
    periods = comparison_periods(query, history)
    context = expand_hits_with_neighbors(
        hits, docs_by_id, neighbors=neighbors, max_chars=max_chars,
        separate_neighbors=bool(periods) or followup,
    )
    if getattr(llm, "provider_name", "") != "llama":
        return context
    if not periods:
        return fit_context_to_llm(
            context, hits, query, history=history, max_new_tokens=max_new_tokens, n_ctx=n_ctx,
            max_passages=max_passages, passage_max_chars=passage_max_chars,
        )
    fitted, seen = [], set()
    anchor_ids = {hit.id for hit in hits}
    cap = max(max_passages, PERIOD_PASSAGES)
    for period in periods:
        subset = [hit for hit in context if _meta_matches(hit.metadata, period)]
        if not subset:
            continue
        # Every retrieved anchor before any neighbour, so the cap cuts neighbours first.
        subset = ([hit for hit in subset if hit.id in anchor_ids]
                  + [hit for hit in subset if hit.id not in anchor_ids])
        focus = f"{query} (focus: {period_label(period)})"
        subset = fit_context_to_llm(
            subset, hits, focus, history=history, max_new_tokens=min(max_new_tokens, 300), n_ctx=n_ctx,
            max_passages=cap, passage_max_chars=passage_max_chars, periods=[period],
        )
        for hit in subset:
            if hit.id not in seen:
                fitted.append(hit)
                seen.add(hit.id)
    return fitted


@dataclass
class ComparisonAnswer:
    answer: str
    citations: List[dict]
    validation: ClaimValidationResult


def generate_comparison_answer(
    llm, query: str, context: Sequence[SearchHit], *,
    history: Optional[Sequence[Dict[str, str]]] = None, max_new_tokens: int,
    extra_queries: Sequence[str] = (),
) -> ComparisonAnswer:
    """Generate and validate each part against ONLY that period's passages.

    No fallback engine, synthetic evidence or refusal retries. Renumber only
    after validation, using the combined passage list returned to the client.
    Unsupported periods get an explicit coverage note, not a global refusal.
    """
    paragraphs, blocked, validations = [], [], []
    nli_available = False
    all_periods = comparison_periods(query, history)
    for period in all_periods:
        label = period_label(period)
        indexed = [(index + 1, hit) for index, hit in enumerate(context)
                   if _meta_matches(hit.metadata, period)]
        subset = [hit for _, hit in indexed]
        part = ""
        # Period-focused terms can only add evidence: when the shared stem sits in a clause that
        # names the other period, focusing drops it (hv31 2016: 0.33 -> 0.00), so keep the better.
        focused = period_focused_query(query, period, all_periods)
        gates = [assess_evidence(q, subset, extra_queries=extra_queries) for q in dict.fromkeys([focused, query])]
        overlap = max(g.best_overlap for g in gates)
        # Soft gate: weak lexical overlap still reaches the model, which (or the validator) may refuse.
        if any(g.sufficient for g in gates) or (_soft_gate() and overlap >= SOFT_GATE_FLOOR):
            prompt = build_cited_prompt(f"{query} (focus: {label})", subset, history=history)
            raw = llm.generate(prompt, max_new_tokens=min(max_new_tokens, 300))
            clean = format_answer_markdown(strip_chat_artifacts(raw))
            if clean != REFUSAL_LINE:
                validation = validate_and_filter_answer(clean, subset)
                part = validation.safe_answer
                blocked.extend(validation.blocked_claims)
                for item in validation.validations:
                    item = dict(item)
                    item["cited_indexes"] = [indexed[i][0] - 1 for i in item["cited_indexes"]]
                    validations.append(item)
                nli_available |= validation.nli_available
        if part:
            mapping = {local: global_index for local, (global_index, _) in enumerate(indexed, 1)}
            def renumber(match):
                numbers = [mapping[int(value.strip())] for value in match.group(1).split(",")
                           if int(value.strip()) in mapping]
                return "[" + ",".join(map(str, numbers)) + "]" if numbers else ""
            part = _CITATIONS.sub(renumber, part)
            # Keep one paragraph per period without altering supported wording.
            paragraphs.append(f"In {label}: {_join_part(part)}")
        else:
            paragraphs.append(f"In {label}: The retrieved passages for this period do not cover this question.")
    answer = "\n\n".join(paragraphs)
    return ComparisonAnswer(answer, parse_citations(answer, context),
                            ClaimValidationResult(answer, blocked, validations, nli_available))
