"""Question-answer preparation, generation and shared backend state."""
from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    ANSWER_CONTEXT_MAX_CHARS,
    CHUNKS_V3_FILE,
    EXPANSION_MODE,
    EXPOSE_DEBUG_STATUS,
    llm_context_passages,
    llm_n_ctx,
    retrieval_setting,
)
from src.generation.prompt import (
    REFUSAL_LINE,
    build_cited_prompt,
    format_answer_markdown,
    format_history_block,
    parse_citations,
    strip_chat_artifacts,
)
from src.generation import finance_answer
from src.generation.answer_expand import expand_answer
from src.generation.compare import comparison_periods, generate_comparison_answer, prepare_answer_context
from src.generation.decompose import answer_by_parts, split_question
from src.generation.evidence_gate import assess_evidence
from src.evaluation.claim_validator import validate_and_filter_answer
from src.generation.providers import create_llm_provider
from src.knowledge import KnowledgeStore
from src.query.plan import CORPUS_WIDE, MIXED, NUMERIC, PERSONAL, TREND, QueryPlan, plan_query
from src.retrieval.finance_policy import finance_search
from src.retrieval.query_expansion import build_followup_retrieval_query, expand_query, expand_query_structured
from src.retrieval.retriever import Retriever
from src.services.schemas import AskRequest, HitOut, SearchRequest
from src.storage import SearchHit


_state: Dict[str, Any] = {}
_CITATION_MARKER_RE = re.compile(r"\[\d+\]")
_ANSWER_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
_MIN_ANSWER_WORDS = 3


def _resolve_chunks_path() -> Path:
    """The v3 corpus only; backend startup fails loudly when it is missing."""
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
    top_k = req.top_k
    if isinstance(req, AskRequest):
        # The prompt takes up to LLM_CONTEXT_PASSAGES passages; retrieve at least that many (plus the
        # neighbour anchors appended after the ranked hits), never fewer than the request asked for.
        top_k = max(top_k, min(20, llm_context_passages() - 2 * retrieval_setting("RETRIEVAL_NEIGHBOR_ANCHORS")))
    result = retriever.search(
        query=resolved,
        strategy=req.strategy,
        top_k=top_k,
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
    return _state["llm"]


def _extra_queries(req: AskRequest, expanded: Optional[str]) -> List[str]:
    """History user turns and the expanded query count toward the evidence terms."""
    extras = [turn.content for turn in req.history if turn.role == "user"]
    if expanded:
        extras.append(expanded)
    return extras


def order_passages(hits):
    """Served passage order (LLM_PASSAGE_ORDER, read at call time): natural | reverse | shuffle:<seed>.

    Applied to the served list itself, so [n] markers, validation and citations all follow the new order.
    """
    order = (os.environ.get("LLM_PASSAGE_ORDER") or "natural").strip().lower()
    if order == "natural":
        return hits
    if order == "reverse":
        return list(reversed(hits))
    if order.startswith("shuffle:") and order[8:].lstrip("-").isdigit():
        shuffled = list(hits)
        random.Random(int(order[8:])).shuffle(shuffled)
        return shuffled
    raise ValueError(f"LLM_PASSAGE_ORDER={order!r}; expected natural, reverse or shuffle:<seed>")


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
            neighbors=retrieval_setting("ANSWER_CONTEXT_NEIGHBORS"), max_chars=ANSWER_CONTEXT_MAX_CHARS,
            max_new_tokens=req.max_new_tokens, n_ctx=llm_n_ctx(),
            max_passages=llm_context_passages(), passage_max_chars=retrieval_setting("LLM_PASSAGE_MAX_CHARS"),
        )
        context_hits = order_passages(context_hits)
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


def is_citation_only(answer: str) -> bool:
    """True when fewer than three words remain after removing [n] markers."""
    return len(_ANSWER_WORD_RE.findall(_CITATION_MARKER_RE.sub(" ", answer or ""))) < _MIN_ANSWER_WORDS


def _finalize_answer(llm, prompt: str, context_hits, raw_answer: str, max_new_tokens: int):
    """Format and resolve citation references; this is not entailment validation.

    A refusal is not a provider error. Repeating the same prompt until the
    model answers selects against justified abstention without new evidence.
    """
    answer = format_answer_markdown(strip_chat_artifacts(raw_answer))
    if is_citation_only(answer):
        # A bare "[5]" is a failed generation, not an answer: refuse, with no citations.
        if EXPOSE_DEBUG_STATUS:
            print(f"[backend] citation-only answer {answer!r}, refusing", flush=True)
        return REFUSAL_LINE, []
    if answer == REFUSAL_LINE:
        return answer, []
    validation = validate_and_filter_answer(answer, context_hits)
    answer = validation.safe_answer or REFUSAL_LINE
    if answer != REFUSAL_LINE:
        answer = expand_answer(answer, context_hits)
    return answer, parse_citations(answer, context_hits)


def _refusal_retry_enabled() -> bool:
    import config
    return os.getenv("ANSWER_REFUSAL_RETRY", str(config.ANSWER_REFUSAL_RETRY)) == "1"


def _generate_answer(llm, prompt: str, context_hits, max_new_tokens: int, query: Optional[str] = None,
                     extra_queries=(), history=None):
    is_comparison = bool(query and comparison_periods(query))
    soft_gate = False
    if query and not is_comparison:
        evidence = assess_evidence(query, context_hits, extra_queries=extra_queries)
        soft_gate = (
            not evidence.sufficient
            and os.getenv("EVIDENCE_GATE_SOFT") != "0"
            and 0.15 <= evidence.best_overlap < 0.30
        )
        if not evidence.sufficient and not soft_gate:
            return REFUSAL_LINE, []
        if EXPOSE_DEBUG_STATUS:
            print(f"[backend] evidence overlap: {evidence.best_overlap:.3f}", flush=True)
    if query and not is_comparison and os.getenv("ANSWER_DECOMPOSE", "0") == "1":
        parts = split_question(query)
        if len(parts) >= 2:
            try:
                return answer_by_parts(llm, query, context_hits, history, max_new_tokens)
            except Exception as exc:
                if EXPOSE_DEBUG_STATUS:
                    print(f"[backend] LLM provider unavailable: {exc}", flush=True)
                return _llm_error_message(exc), []
    try:
        if is_comparison and query is not None:
            comparison = generate_comparison_answer(
                llm, query, context_hits, history=history, max_new_tokens=max_new_tokens,
                extra_queries=extra_queries,
            )
            expanded = expand_answer(comparison.answer, context_hits)
            if expanded == comparison.answer:
                return comparison.answer, comparison.citations
            return expanded, parse_citations(expanded, context_hits)
        raw_answer = llm.generate(prompt, max_new_tokens=max_new_tokens)
    except Exception as exc:
        if EXPOSE_DEBUG_STATUS:
            print(f"[backend] LLM provider unavailable: {exc}", flush=True)
        return _llm_error_message(exc), []
    answer, citations = _finalize_answer(llm, prompt, context_hits, raw_answer, max_new_tokens)
    if answer == REFUSAL_LINE and soft_gate and len(context_hits) > 3 and _refusal_retry_enabled():
        # The soft gate let weak evidence through and the model refused: ask once with the top 3 passages.
        top = list(context_hits[:3])
        retry_prompt = build_cited_prompt(query=query, hits=top, history=history)
        try:
            retry_raw = llm.generate(retry_prompt, max_new_tokens=max_new_tokens)
        except Exception as exc:
            if EXPOSE_DEBUG_STATUS:
                print(f"[backend] LLM provider unavailable: {exc}", flush=True)
            return answer, citations
        retried = _finalize_answer(llm, retry_prompt, top, retry_raw, max_new_tokens)
        if retried[0] != REFUSAL_LINE:
            return retried
    return answer, citations


# -----------------------------------------------------------------------------
# Finance pipeline (FINANCE_PIPELINE=1, finance-query-pipeline): one route decision for /ask and /ask/stream.
# The caller checks `config.finance_pipeline_enabled()`; with the flag off nothing below runs.
# -----------------------------------------------------------------------------
SCOPE_VOTE_POOL = 10  # passages of the scope vote's hybrid retrieval (no rerank, no scope filter)


@dataclass
class FinanceRoute:
    """Where a planned question went: a finished answer (scope / numeric / digest) or the reader inputs."""
    route: str                      # scope | numeric | digest | reader
    plan: QueryPlan
    query: str                      # the question the reader and the evidence gate see (personal clauses dropped)
    answer: Optional[str] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    hits: List[SearchHit] = field(default_factory=list)            # served, [n] markers point at these
    retrieved_hits: List[SearchHit] = field(default_factory=list)
    used_filter: Optional[Dict[str, Any]] = None
    reranked: bool = False
    llm: Any = None
    prompt: str = ""
    debug: Dict[str, Any] = field(default_factory=dict)

    @property
    def final(self) -> bool:
        """True for scope, numeric and digest answers: returned as they are, no gate, validator or expansion."""
        return self.route != "reader"

    def prepared(self):
        """The reader inputs in the shape `_prepare_ask` returns."""
        return self.llm, self.retrieved_hits, self.used_filter, self.reranked, self.hits, self.prompt, None


def _knowledge_store() -> KnowledgeStore:
    if "knowledge" not in _state:
        _state["knowledge"] = KnowledgeStore.load()  # KNOWLEDGE_DIR or data/knowledge/; missing = empty store
    return _state["knowledge"]


def _scope_vote(retriever, store: KnowledgeStore, timing: Dict[str, float]):
    """Share of the top passages (hybrid, no rerank, no scope filter) that are non-financial or carry personal facts."""
    if not len(store):
        return None  # nothing is annotated: every passage counts as financial, the vote would always be 0

    def vote(text: str) -> float:
        started = time.perf_counter()
        pool = retriever._hybrid_for_filter(text, SCOPE_VOTE_POOL, None)[:SCOPE_VOTE_POOL]
        timing["scope_vote_s"] = round(time.perf_counter() - started, 3)
        if not pool:
            return 0.0
        return sum(1 for h in pool if not store.is_financial(h.id) or store.has_personal(h.id)) / len(pool)

    return vote


def finance_route_steps(req: AskRequest):
    """Generator like `_prepare_ask_steps`: yields stage names, returns a `FinanceRoute` (StopIteration.value).

    Plans the question (deterministic, `src/query/plan.py`) and routes it: scope reply, numeric answer, digest
    answer, or `finance_search` plus the reader. The query-expansion LLM call, the retriever's comparison and
    multipart splits and its auto year filter are bypassed. The reader route caps the served passages at
    `llm_context_passages()` for every provider (the remote provider's context is not trimmed by
    `prepare_answer_context`).
    """
    retriever: Retriever = _state["retriever"]
    store = _knowledge_store()
    docs_by_id = _state.get("docs_by_id", {})
    history = [turn.model_dump() for turn in req.history]
    resolved = build_followup_retrieval_query(req.query, history)
    timing: Dict[str, float] = {}
    plan = plan_query(resolved, store, scope_vote=_scope_vote(retriever, store, timing))
    debug: Dict[str, Any] = {"kind": plan.kind, "scope": plan.scope, "scope_source": plan.scope_source, **timing}
    query = plan.financial_query if plan.scope == MIXED and plan.financial_query else req.query

    if plan.scope == PERSONAL:
        done = finance_answer.scope_answer()
        return FinanceRoute("scope", plan, query, done.answer, done.citations, done.hits, [], debug=debug)

    if plan.kind == NUMERIC:
        done = finance_answer.numeric_answer(plan, store, docs_by_id)
        if done is not None:  # served from the numbers table: no retrieval
            debug.update(done.detail)
            return FinanceRoute("numeric", plan, query, done.answer, done.citations, done.hits, [], debug=debug)
    yield "retrieving"
    hits, search_debug = finance_search(plan, retriever, store)
    debug.update(search_debug)
    used_filter = ({"year": {"$gte": min(plan.year_range), "$lte": max(plan.year_range)}}
                   if plan.year_hard and plan.year_range else None)
    reranked = bool(search_debug.get("reranked"))
    cap = llm_context_passages()

    if plan.kind == CORPUS_WIDE:
        done = finance_answer.digest_answer(plan, store, docs_by_id, hits)
        if done is not None:
            done.hits = done.hits[:max(cap, done.detail["cited"])]  # the cited quotes always stay
            debug.update(done.detail)
            return FinanceRoute("digest", plan, query, done.answer, done.citations, done.hits, hits, used_filter,
                                reranked, debug=debug)

    llm = _server_llm()
    context_hits: List[Any] = []
    prompt = ""
    if hits:
        context_hits = prepare_answer_context(
            llm, hits, docs_by_id, query, history=history,
            followup=build_followup_retrieval_query(req.query, history) != req.query,
            neighbors=retrieval_setting("ANSWER_CONTEXT_NEIGHBORS"), max_chars=ANSWER_CONTEXT_MAX_CHARS,
            max_new_tokens=req.max_new_tokens, n_ctx=llm_n_ctx(),
            max_passages=cap, passage_max_chars=retrieval_setting("LLM_PASSAGE_MAX_CHARS"),
        )
        context_hits = list(context_hits)[:cap]
        if plan.kind == TREND:
            context_hits = finance_answer.sort_by_year(context_hits)
        context_hits = order_passages(context_hits)
        if plan.kind == TREND:
            prompt = finance_answer.trend_prompt(query, context_hits, history)
        else:
            prompt = build_cited_prompt(query=query, hits=context_hits, history=history)
    return FinanceRoute("reader", plan, query, hits=context_hits, retrieved_hits=hits, used_filter=used_filter,
                        reranked=reranked, llm=llm, prompt=prompt, debug=debug)


def finance_route(req: AskRequest) -> FinanceRoute:
    """Plan and route one question without streaming stage names."""
    steps = finance_route_steps(req)
    while True:
        try:
            next(steps)
        except StopIteration as stop:
            return stop.value
