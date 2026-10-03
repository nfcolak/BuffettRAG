"""Period-isolated grounded generation shared by HTTP, SSE and the pipeline."""
from __future__ import annotations

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
from src.retrieval.retriever import detect_temporal_comparison
from src.vector_store import SearchHit

_CITATIONS = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def comparison_periods(query: str) -> List[Dict[str, Any]]:
    return detect_temporal_comparison(query) or []


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
    periods = comparison_periods(query)
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
    for period in periods:
        subset = [hit for hit in context if _meta_matches(hit.metadata, period)]
        if not subset:
            continue
        focus = f"{query} (focus: {period_label(period)})"
        subset = fit_context_to_llm(
            subset, hits, focus, history=history, max_new_tokens=min(max_new_tokens, 300), n_ctx=n_ctx,
            max_passages=max_passages, passage_max_chars=passage_max_chars, periods=[period],
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
    for period in comparison_periods(query):
        label = period_label(period)
        indexed = [(index + 1, hit) for index, hit in enumerate(context)
                   if _meta_matches(hit.metadata, period)]
        subset = [hit for _, hit in indexed]
        part = ""
        if assess_evidence(query, subset, extra_queries=extra_queries).sufficient:
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
            paragraphs.append(f"In {label}: {re.sub(r'\s+', ' ', part).strip()}")
        else:
            paragraphs.append(f"In {label}: The retrieved passages for this period do not cover this question.")
    answer = "\n\n".join(paragraphs)
    return ComparisonAnswer(answer, parse_citations(answer, context),
                            ClaimValidationResult(answer, blocked, validations, nli_available))
