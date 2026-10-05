"""Question-answer preparation, generation and shared backend state."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    ANSWER_CONTEXT_MAX_CHARS,
    ANSWER_CONTEXT_NEIGHBORS,
    CHUNKS_FILE,
    CHUNKS_V2_FILE,
    CHUNKS_V3_FILE,
    EXPANSION_MODE,
    EXPOSE_DEBUG_STATUS,
    LLM_CONTEXT_PASSAGES,
    LLM_N_CTX,
    LLM_PASSAGE_MAX_CHARS,
)
from src.generation.prompt import (
    REFUSAL_LINE,
    build_cited_prompt,
    format_answer_markdown,
    format_history_block,
    parse_citations,
    strip_chat_artifacts,
)
from src.generation.compare import comparison_periods, generate_comparison_answer, prepare_answer_context
from src.generation.evidence_gate import assess_evidence
from src.evaluation.claim_validator import validate_and_filter_answer
from src.generation.providers import create_llm_provider
from src.retrieval.query_expansion import build_followup_retrieval_query, expand_query, expand_query_structured
from src.retrieval.retriever import Retriever
from src.services.schemas import AskRequest, HitOut, SearchRequest
from src.storage import SearchHit


_state: Dict[str, Any] = {}


def _resolve_chunks_path() -> Path:
    for candidate in (CHUNKS_V3_FILE, CHUNKS_V2_FILE, CHUNKS_FILE):
        if candidate.exists():
            return candidate
    return CHUNKS_V3_FILE


def _hits_to_out(hits: List[SearchHit]) -> List[HitOut]:
    return [
        HitOut(
            id=h.id,
            score=h.score,
            year=h.metadata.get("year"),
            source_file=h.metadata.get("source_file"),
            topics=h.metadata.get("topics", ""),
            text=h.text,
        )
        for h in hits
    ]


def _llm_error_message(exc: Exception) -> str:
    return "[LLM unavailable: the embedded model failed to generate an answer]"


def _do_search(req: SearchRequest, retrieval_query: Optional[str] = None):
    retriever: Retriever = _state["retriever"]
    history = [turn.model_dump() for turn in req.history] if isinstance(req, AskRequest) else []
    resolved = build_followup_retrieval_query(req.query, history)
    result = retriever.search(
        query=resolved,
        strategy=req.strategy,
        top_k=req.top_k,
        fetch_k=req.fetch_k,
        rerank=req.rerank,
        where=req.where,
        auto_year_filter=req.auto_year_filter,
        retrieval_query=retrieval_query,
    )
    return result.hits, result.used_filter, result.reranked


def _server_llm():
    if "llm" not in _state:
        _state["llm"] = create_llm_provider()
        attach_grounded_resources(_state["llm"], _state.get("retriever"))
    return _state["llm"]


def is_grounded(llm) -> bool:
    return getattr(llm, "provider_name", "") == "grounded"


def attach_grounded_resources(llm, retriever) -> None:
    """Let the grounded scorer reuse the retriever's reranker (no-op for other providers)."""
    reranker = getattr(retriever, "reranker", None)
    if is_grounded(llm) and reranker is not None and hasattr(llm, "attach_resources"):
        llm.attach_resources(reranker=reranker)


def _generate_grounded(llm, query: str, context_hits, max_new_tokens: int, history=None):
    """The one grounded generation path, shared by /ask, /ask/stream and _generate_answer.

    No evidence gate and no comparison special case: the engine owns the decision,
    including the exact refusal for an empty context.
    """
    try:
        result = llm.answer_grounded(
            query, context_hits, history=list(history or []), max_new_tokens=max_new_tokens,
        )
    except Exception as exc:
        if EXPOSE_DEBUG_STATUS:
            print(f"[backend] grounded provider unavailable: {exc}", flush=True)
        return _llm_error_message(exc), []
    return result.answer, result.citations


def _extra_queries(req: AskRequest, expanded: Optional[str]) -> List[str]:
    """History user turns and the expanded query count toward the evidence terms."""
    extras = [turn.content for turn in req.history if turn.role == "user"]
    if expanded:
        extras.append(expanded)
    return extras


def _prepare_ask_steps(req: AskRequest):
    """Generator: yields stage names, returns the prepared tuple (via StopIteration.value)."""
    llm = _server_llm()
    history_dicts = [turn.model_dump() for turn in req.history]
    history_text = format_history_block(history_dicts)

    def _expand() -> Optional[str]:
        structured = expand_query_structured(req.query, llm, history_text=history_text)
        result = structured.retrieval_query if structured else expand_query(
            req.query, llm, history_text=history_text
        )
        if result and EXPOSE_DEBUG_STATUS:
            print(f"[backend] expanded query: {result!r}", flush=True)
        return result

    mode = EXPANSION_MODE if req.expand_query else "off"
    expanded = None
    if mode == "always":
        yield "expanding"
        expanded = _expand()
    yield "retrieving"
    hits, used_filter, reranked = _do_search(req, retrieval_query=expanded)
    if mode == "auto" and assess_evidence(req.query, hits).best_overlap < 0.5:
        yield "expanding"
        expanded = _expand()
        if expanded:
            yield "retrieving"
            hits, used_filter, reranked = _do_search(req, retrieval_query=expanded)

    context_hits: List[Any] = []
    prompt = ""
    if hits:
        context_hits = prepare_answer_context(
            llm, hits, _state.get("docs_by_id", {}), req.query, history=history_dicts,
            followup=build_followup_retrieval_query(req.query, history_dicts) != req.query,
            neighbors=ANSWER_CONTEXT_NEIGHBORS, max_chars=ANSWER_CONTEXT_MAX_CHARS,
            max_new_tokens=req.max_new_tokens, n_ctx=LLM_N_CTX,
            max_passages=LLM_CONTEXT_PASSAGES, passage_max_chars=LLM_PASSAGE_MAX_CHARS,
        )
        prompt = build_cited_prompt(query=req.query, hits=context_hits, history=history_dicts)

    return llm, hits, used_filter, reranked, context_hits, prompt, expanded


def _prepare_ask(req: AskRequest):
    """Shared /ask preparation: expansion, retrieval, context and prompt."""
    steps = _prepare_ask_steps(req)
    while True:
        try:
            next(steps)
        except StopIteration as stop:
            return stop.value


def _finalize_answer(llm, prompt: str, context_hits, raw_answer: str, max_new_tokens: int):
    """Format and resolve citation references; this is not entailment validation.

    A refusal is not a provider error. Repeating the same prompt until the
    model answers selects against justified abstention without new evidence.
    """
    answer = format_answer_markdown(strip_chat_artifacts(raw_answer))
    if answer == REFUSAL_LINE:
        return answer, []
    validation = validate_and_filter_answer(answer, context_hits)
    answer = validation.safe_answer or REFUSAL_LINE
    return answer, parse_citations(answer, context_hits)


def _generate_answer(llm, prompt: str, context_hits, max_new_tokens: int, query: Optional[str] = None,
                     extra_queries=(), history=None):
    if is_grounded(llm):
        return _generate_grounded(llm, query or "", context_hits, max_new_tokens, history)
    is_comparison = bool(query and comparison_periods(query))
    if query and not is_comparison and not assess_evidence(query, context_hits, extra_queries=extra_queries).sufficient:
        return REFUSAL_LINE, []
    try:
        if is_comparison and query is not None:
            comparison = generate_comparison_answer(
                llm, query, context_hits, history=history, max_new_tokens=max_new_tokens,
                extra_queries=extra_queries,
            )
            return comparison.answer, comparison.citations
        raw_answer = llm.generate(prompt, max_new_tokens=max_new_tokens)
    except Exception as exc:
        if EXPOSE_DEBUG_STATUS:
            print(f"[backend] LLM provider unavailable: {exc}", flush=True)
        return _llm_error_message(exc), []
    return _finalize_answer(llm, prompt, context_hits, raw_answer, max_new_tokens)
