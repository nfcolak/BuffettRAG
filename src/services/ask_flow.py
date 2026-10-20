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
from src.query.plan import COMPARISON, CORPUS_WIDE, MIXED, NUMERIC, PERSONAL, TREND, QueryPlan, plan_query
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


def _finalize_answer(llm, prompt: str, context_hits, raw_answer: str, max_new_tokens: int,
                     trace: Optional[Dict[str, Any]] = None):
    """Format and resolve citation references; this is not entailment validation.

    `trace` (FINANCE_DEBUG only) collects the raw output and the validator's kept/dropped sentences.

    A finance reader prompt (`finance_answer.is_reader_prompt`) gets the reader settings: number-safe list formatting,
    the validator's reader options (`finance_answer.READER_VALIDATION`) and neighbour expansion
    (`finance_answer.READER_EXPANSION`). Every other prompt is finalized exactly as before.

    A refusal is not a provider error. Repeating the same prompt until the
    model answers selects against justified abstention without new evidence.
    """
    reader = finance_answer.is_reader_prompt(prompt)
    answer = (finance_answer.format_reader_answer if reader else format_answer_markdown)(strip_chat_artifacts(raw_answer))
    if trace is not None:
        trace["raw_answer"] = raw_answer
    if is_citation_only(answer):
        if trace is not None:
            trace["validator"] = {"skipped": "citation_only", "kept": [], "dropped": []}
        # A bare "[5]" is a failed generation, not an answer: refuse, with no citations.
        if EXPOSE_DEBUG_STATUS:
            print(f"[backend] citation-only answer {answer!r}, refusing", flush=True)
        return REFUSAL_LINE, []
    if answer == REFUSAL_LINE:
        if trace is not None:
            trace["validator"] = {"skipped": "model_refused", "kept": [], "dropped": []}
        return answer, []
    validation = validate_and_filter_answer(answer, context_hits, **(finance_answer.READER_VALIDATION if reader else {}))
    if trace is not None:
        trace["validator"] = {
            "kept": [s["sentence"] for s in validation.sentences if s["kept"]],
            "dropped": [{"sentence": s["sentence"], "reason": s["reason"], "failed_claims": s.get("failed_claims", [])}
                        for s in validation.sentences if not s["kept"]],
        }
        repaired = [{"sentence": s["original"], "repaired": s["sentence"]}
                    for s in validation.sentences if s["reason"] == "citation_repaired"]
        if repaired:  # reader route only, and only when a marker was repaired
            trace["validator"]["repaired"] = repaired
    answer = validation.safe_answer or REFUSAL_LINE
    if answer != REFUSAL_LINE and reader:
        answer = expand_answer(answer, finance_answer.expansion_hits(context_hits, _knowledge_store()),
                               **finance_answer.READER_EXPANSION)
    elif answer != REFUSAL_LINE:
        answer = expand_answer(answer, context_hits)
    return answer, parse_citations(answer, context_hits)


def _refusal_retry_enabled() -> bool:
    import config
    return os.getenv("ANSWER_REFUSAL_RETRY", str(config.ANSWER_REFUSAL_RETRY)) == "1"


def _generate_answer(llm, prompt: str, context_hits, max_new_tokens: int, query: Optional[str] = None,
                     extra_queries=(), history=None, per_period: Optional[bool] = None,
                     trace: Optional[Dict[str, Any]] = None):
    """`per_period` None: answer per period when the query names comparison periods (the legacy rule); the finance
    route passes its own decision (True only for comparison plans), so a trend gets one reader call.
    `trace` (FINANCE_DEBUG only) is filled in place: raw_answer, validator, refusal_retry, refusal_by."""
    if trace is not None:
        trace["refusal_retry"] = False
    if per_period is None:
        is_comparison = bool(query and comparison_periods(query))
    else:
        is_comparison = bool(query and per_period)
    soft_gate = False
    if query and not is_comparison:
        evidence = assess_evidence(query, context_hits, extra_queries=extra_queries)
        soft_gate = (
            not evidence.sufficient
            and os.getenv("EVIDENCE_GATE_SOFT") != "0"
            and 0.15 <= evidence.best_overlap < 0.30
        )
        if not evidence.sufficient and not soft_gate:
            if trace is not None:
                trace["refusal_by"] = "evidence_gate"
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
            if trace is not None:
                trace["raw_answer"] = comparison.answer  # per-period answers, already assembled and validated
                trace["raw_answer_note"] = "per_period"
            expanded = expand_answer(comparison.answer, context_hits)
            if expanded == comparison.answer:
                return comparison.answer, comparison.citations
            return expanded, parse_citations(expanded, context_hits)
        raw_answer = llm.generate(prompt, max_new_tokens=max_new_tokens)
    except Exception as exc:
        if EXPOSE_DEBUG_STATUS:
            print(f"[backend] LLM provider unavailable: {exc}", flush=True)
        return _llm_error_message(exc), []
    answer, citations = _finalize_answer(llm, prompt, context_hits, raw_answer, max_new_tokens, trace)
    if answer == REFUSAL_LINE and trace is not None:
        trace["refusal_by"] = "model_or_validator"
    if answer == REFUSAL_LINE and soft_gate and len(context_hits) > 3 and _refusal_retry_enabled() \
            and not finance_answer.is_reader_prompt(prompt):
        # The soft gate let weak evidence through and the model refused: ask once with the top 3 passages.
        # Not on the finance reader route: its prompt already asks for the part the passages cover, and a second
        # answer from 3 of the served passages would replace the reader's own (staging_v3: 2 retries, 0 answers).
        top = list(context_hits[:3])
        retry_prompt = build_cited_prompt(query=query, hits=top, history=history)
        try:
            retry_raw = llm.generate(retry_prompt, max_new_tokens=max_new_tokens)
        except Exception as exc:
            if EXPOSE_DEBUG_STATUS:
                print(f"[backend] LLM provider unavailable: {exc}", flush=True)
            return answer, citations
        retry_trace: Optional[Dict[str, Any]] = {} if trace is not None else None
        retried = _finalize_answer(llm, retry_prompt, top, retry_raw, max_new_tokens, retry_trace)
        if trace is not None:
            trace["refusal_retry"] = True
            trace["first_raw_answer"] = trace.get("raw_answer")
            trace["retry_prompt_chars"] = len(retry_prompt)
            trace.update(retry_trace)
        if retried[0] != REFUSAL_LINE:
            if trace is not None:
                trace.pop("refusal_by", None)
            return retried
    return answer, citations


# -----------------------------------------------------------------------------
# Finance pipeline (FINANCE_PIPELINE=1, finance-query-pipeline): one route decision for /ask and /ask/stream.
# The caller checks `config.finance_pipeline_enabled()`; with the flag off nothing below runs.
# -----------------------------------------------------------------------------
SCOPE_VOTE_POOL = 10  # passages of the scope vote's hybrid retrieval (no rerank, no scope filter)


@dataclass
class FinanceRoute:
    """Where a planned question went: a finished answer (scope / superlative / numeric / digest) or the reader inputs."""
    route: str                      # scope | superlative | numeric | digest | reader
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
    per_period: bool = False        # reader: one call per comparison period (comparison plans only), else one call

    @property
    def final(self) -> bool:
        """True for scope, superlative, numeric and digest answers: returned as they are, no gate, validator or
        expansion."""
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

    Plans the question (deterministic, `src/query/plan.py`) and routes it: scope reply, superlative answer
    (corpus-wide "which sector did he praise most"), numeric answer (plain figure lookups only), digest answer, or
    `finance_search` plus the reader. The query-expansion LLM call, the retriever's comparison and multipart splits
    and its auto year filter are bypassed. The reader route caps the served passages at `llm_context_passages()` for
    every provider (the remote provider's context is not trimmed by `prepare_answer_context`). A question with
    numeric intent that is not a plain lookup reaches the reader with its matching figure rows: their source passages
    are served first and the rows are added to the prompt. Only comparison plans are answered per period; every other
    reader question (a trend included) is one reader call over the served passages, with the finance reader prompt
    (`finance_answer.reader_prompt`) and, in `_finalize_answer`, the reader settings.
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
    cap = llm_context_passages()

    if plan.superlative:
        done = finance_answer.superlative_answer(plan, store, docs_by_id, max_hits=cap)
        if done is not None:  # ranked from the stance annotations: no retrieval
            debug.update(done.detail)
            return FinanceRoute("superlative", plan, query, done.answer, done.citations, done.hits, [], debug=debug)

    rows: List[Any] = []
    if plan.kind == NUMERIC or plan.numeric:
        rows = finance_answer.figure_rows(plan, store, docs_by_id)
        if rows and plan.kind == NUMERIC and finance_answer.plain_figure_lookup(plan) \
                and finance_answer.metric_named(plan, store):
            done = finance_answer.numeric_answer(plan, store, docs_by_id)
            if done is not None:  # a plain figure lookup, served from the numbers table: no retrieval
                debug.update(done.detail)
                return FinanceRoute("numeric", plan, query, done.answer, done.citations, done.hits, [], debug=debug)
    pinned = finance_answer.figure_source_hits(rows, docs_by_id)[:max(1, cap // 3)]
    rows = [row for row in rows if row["source_id"] in {h.id for h in pinned}]
    yield "retrieving"
    hits, search_debug = finance_search(plan, retriever, store, pinned=pinned)
    debug.update(search_debug)
    debug["figure_rows"] = len(rows)
    used_filter = ({"year": {"$gte": min(plan.year_range), "$lte": max(plan.year_range)}}
                   if plan.year_hard and plan.year_range else None)
    reranked = bool(search_debug.get("reranked"))

    if plan.kind == CORPUS_WIDE:
        done = finance_answer.digest_answer(plan, store, docs_by_id, hits)
        if done is not None:
            done.hits = done.hits[:max(cap, done.detail["cited"])]  # the cited quotes always stay
            debug.update(done.detail)
            return FinanceRoute("digest", plan, query, done.answer, done.citations, done.hits, hits, used_filter,
                                reranked, debug=debug)

    llm = _server_llm()
    per_period = plan.kind == COMPARISON and bool(comparison_periods(query, history))
    context_hits: List[Any] = []
    prompt = ""
    if hits:
        context_hits = prepare_answer_context(
            llm, hits, docs_by_id, query, history=history,
            followup=build_followup_retrieval_query(req.query, history) != req.query,
            neighbors=retrieval_setting("ANSWER_CONTEXT_NEIGHBORS"), max_chars=ANSWER_CONTEXT_MAX_CHARS,
            max_new_tokens=req.max_new_tokens, n_ctx=llm_n_ctx(),
            max_passages=cap, passage_max_chars=retrieval_setting("LLM_PASSAGE_MAX_CHARS"),
            periods=None if per_period else (),
        )
        context_hits = list(context_hits)[:cap]
        if plan.kind == TREND:
            context_hits = finance_answer.sort_by_year(context_hits)
        context_hits = order_passages(context_hits)
        prompt = finance_answer.reader_prompt(query, context_hits, history, trend=plan.kind == TREND)
        if rows:
            prompt = finance_answer.figures_prompt(prompt, rows, context_hits)
    return FinanceRoute("reader", plan, query, hits=context_hits, retrieved_hits=hits, used_filter=used_filter,
                        reranked=reranked, llm=llm, prompt=prompt, debug=debug, per_period=per_period)


def finance_answer_mode(route: FinanceRoute) -> str:
    """How the backend answers a route: the final route name, or reader / reader:periods (one call per period)."""
    if route.final:
        return route.route
    return "reader:periods" if route.per_period else "reader"


def finance_route(req: AskRequest) -> FinanceRoute:
    """Plan and route one question without streaming stage names."""
    steps = finance_route_steps(req)
    while True:
        try:
            next(steps)
        except StopIteration as stop:
            return stop.value
