"""Backend FastAPI service.

Deployment: runs on the Nuvolos **Backend app**.
Responsibilities:
    - Loads chunks once at startup
    - Connects to pgvector (Database app) for vector storage
    - Hosts the configured embedding model and cross-encoder reranker
      (see EMBEDDING_MODEL_PRIMARY and RERANKER_MODEL in config.py)
    - Runs retrieval (vector / metadata / hybrid) and reranking
    - For answer generation, sends grounded prompts to the configured LLM provider.

Endpoints:
    GET  /health
    GET  /stats
    POST /search       -- retrieval only
    POST /ask          -- retrieval + LLM generation

Run:
    uvicorn src.services.backend_app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    API_KEYS,
    ANSWER_CONTEXT_MAX_CHARS,
    ANSWER_CONTEXT_NEIGHBORS,
    CHUNKS_FILE,
    CHUNKS_V2_FILE,
    CHUNKS_V3_FILE,
    CORS_ORIGINS,
    DEFAULT_TOP_K,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_PRIMARY,
    EXPANSION_MODE,
    EXPOSE_DEBUG_STATUS,
    FAISS_DIR,
    LLM_CONTEXT_PASSAGES,
    LLM_N_CTX,
    LLM_PASSAGE_MAX_CHARS,
    MAX_REQUEST_BODY_BYTES,
    PUBLIC_DEMO_MODE,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    RETRIEVAL_FETCH_K,
    TRUST_PROXY_HEADERS,
    VECTOR_BACKEND,
)
from src.embeddings import BGEEmbedder
from src.generation.prompt import (
    REFUSAL_LINE,
    build_cited_prompt,
    format_answer_markdown,
    format_history_block,
    parse_citations,
    strip_chat_artifacts,
)
from src.generation.evidence_gate import assess_evidence
from src.evaluation.claim_validator import validate_and_filter_answer
from src.generation.providers import create_llm_provider
from src.index_manifest import ensure_index_identity, write_index_identity
from src.retrieval.context import build_doc_lookup, expand_hits_with_neighbors, fit_context_to_llm
from src.retrieval.query_expansion import expand_query, expand_query_structured
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import Retriever
from src.services.security import FixedWindowRateLimiter, client_key, is_authorized
from src.vector_store import FaissStore, SearchHit, get_vector_store, load_chunks_as_docs


# -----------------------------------------------------------------------------
# Pydantic schemas
# -----------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    strategy: Literal["hybrid", "vector", "metadata", "naive"] = "hybrid"
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    fetch_k: int = Field(default=RETRIEVAL_FETCH_K, ge=1, le=100)
    rerank: bool = True
    where: Optional[Dict[str, Any]] = None
    auto_year_filter: bool = True

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must contain non-whitespace characters")
        return value

    @field_validator("where")
    @classmethod
    def validate_where(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        allowed_fields = {"year", "decade", "source_file", "chunk_index", "topics"}
        allowed_ops = {"$eq", "$gte", "$lte", "$gt", "$lt", "$in"}
        if len(value) > len(allowed_fields) or set(value) - allowed_fields:
            raise ValueError("unsupported metadata filter field")
        for field, condition in value.items():
            if isinstance(condition, dict):
                if not condition or set(condition) - allowed_ops:
                    raise ValueError("unsupported metadata filter operator")
                for operator, target in condition.items():
                    if operator == "$in":
                        if not isinstance(target, list) or not target or len(target) > 50:
                            raise ValueError("metadata $in requires 1 to 50 scalar values")
                        targets = target
                    else:
                        targets = [target]
                    if any(isinstance(item, (dict, list)) or item is None for item in targets):
                        raise ValueError("metadata filters require scalar values")
            elif isinstance(condition, (dict, list)) or condition is None:
                raise ValueError("metadata filters require scalar values")
            if field in {"year", "decade", "chunk_index"}:
                raw_values = condition.values() if isinstance(condition, dict) else [condition]
                for raw in raw_values:
                    values = raw if isinstance(raw, list) else [raw]
                    if any(not isinstance(item, int) or isinstance(item, bool) for item in values):
                        raise ValueError(f"{field} filters require integers")
        return value


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class AskRequest(SearchRequest):
    max_new_tokens: int = Field(default=900, ge=1, le=2000)
    expand_query: bool = True
    history: List[HistoryTurn] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_ask(self):
        if sum(len(turn.content) for turn in self.history) > 8000:
            raise ValueError("conversation history is too large")
        return self


class HitOut(BaseModel):
    id: str
    score: float
    year: Optional[int] = None
    source_file: Optional[str] = None
    topics: str = ""
    text: str


class SearchResponse(BaseModel):
    query: str
    strategy: str
    reranked: bool
    used_filter: Optional[Dict[str, Any]] = None
    hits: List[HitOut]


class AskResponse(SearchResponse):
    retrieved_hits: List[HitOut] = Field(default_factory=list)
    answer: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# App + global state
# -----------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    await startup()
    yield


app = FastAPI(title="BuffettRAG Backend", version="1.0.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ORIGINS),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

_state: Dict[str, Any] = {}
_rate_limiter = FixedWindowRateLimiter(
    max_requests=RATE_LIMIT_REQUESTS,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
)


def validate_deployment_security(
    *, public_demo: bool, api_keys, cors_origins, debug: bool,
    trust_proxy_headers: bool = False,
) -> None:
    if not public_demo:
        return
    if not tuple(key for key in api_keys if key):
        raise RuntimeError("PUBLIC_DEMO_MODE requires at least one API_KEYS entry")
    if debug:
        raise RuntimeError("PUBLIC_DEMO_MODE does not permit EXPOSE_DEBUG_STATUS")
    if trust_proxy_headers:
        raise RuntimeError("PUBLIC_DEMO_MODE does not permit trusted proxy headers; use edge rate limiting")
    for origin in cors_origins:
        if origin in {"*", "null"} or not origin.startswith("https://"):
            raise RuntimeError("PUBLIC_DEMO_MODE permits only exact HTTPS CORS origins")


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    public_endpoint = path in {"/health", "/ready"}
    if not public_endpoint and (
        (PUBLIC_DEMO_MODE and not API_KEYS) or not is_authorized(request, API_KEYS)
    ):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        content_length = request.headers.get("content-length")
        try:
            declared_size = int(content_length) if content_length is not None else None
        except ValueError:
            return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
        if declared_size is not None and declared_size > MAX_REQUEST_BODY_BYTES:
            return JSONResponse({"detail": "Request body too large"}, status_code=413)
        body = await request.body()
        if len(body) > MAX_REQUEST_BODY_BYTES:
            return JSONResponse({"detail": "Request body too large"}, status_code=413)

        key = f"{client_key(request, TRUST_PROXY_HEADERS)}:{path}"
        if not _rate_limiter.allow(key):
            return JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
            )

    return await call_next(request)


def _resolve_chunks_path() -> Path:
    for candidate in (CHUNKS_V3_FILE, CHUNKS_V2_FILE, CHUNKS_FILE):
        if candidate.exists():
            return candidate
    return CHUNKS_V3_FILE


async def startup() -> None:
    validate_deployment_security(
        public_demo=PUBLIC_DEMO_MODE,
        api_keys=API_KEYS,
        cors_origins=CORS_ORIGINS,
        debug=EXPOSE_DEBUG_STATUS,
        trust_proxy_headers=TRUST_PROXY_HEADERS,
    )
    chunks_path = _resolve_chunks_path()
    if not chunks_path.exists():
        raise RuntimeError(
            "No chunks file found. Run `python -m src.ingestion.pipeline_v2` first."
        )

    embedder = BGEEmbedder(model_name=EMBEDDING_MODEL_PRIMARY, device=EMBEDDING_DEVICE)
    docs = load_chunks_as_docs(chunks_path)

    vs = get_vector_store(backend=VECTOR_BACKEND, dim=embedder.dimension)
    identity = dict(
        backend=VECTOR_BACKEND, corpus=chunks_path, docs=docs,
        model_name=embedder.model_name, dimension=embedder.dimension,
    )
    if len(vs) == 0:
        embeddings = embedder.embed_documents([d.text for d in docs])
        vs.add(docs, embeddings)
        write_index_identity(vs, **identity)
    else:
        ensure_index_identity(vs, **identity)

    reranker = CrossEncoderReranker()
    retriever = Retriever(vector_store=vs, embedder=embedder, docs=docs, reranker=reranker)

    _state["vector_store"] = vs
    _state["retriever"] = retriever
    _state["docs"] = docs
    _state["docs_by_id"] = build_doc_lookup(docs)
    _state["chunks_path"] = str(chunks_path)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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
    result = retriever.search(
        query=req.query,
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
    return _state["llm"]


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "alive"}


@app.get("/ready")
def readiness():
    vs = _state.get("vector_store")
    docs = _state.get("docs")
    retriever = _state.get("retriever")
    indexed_count = len(vs) if vs is not None else 0
    document_count = len(docs) if docs is not None else 0
    ready = retriever is not None and indexed_count > 0 and indexed_count == document_count
    payload: Dict[str, Any] = {
        "status": "ready" if ready else "not_ready",
        "indexed_count": indexed_count,
        "document_count": document_count,
    }
    if EXPOSE_DEBUG_STATUS:
        payload["chunks_path"] = _state.get("chunks_path")
    return JSONResponse(payload, status_code=200 if ready else 503)


@app.get("/stats")
def stats() -> Dict[str, Any]:
    vs = _state.get("vector_store")
    docs = _state.get("docs", [])
    years = sorted({d.metadata.get("year") for d in docs if d.metadata.get("year")})
    return {
        "total_chunks": len(docs),
        "indexed_count": len(vs) if vs else 0,
        "year_range": [years[0], years[-1]] if years else [],
        "embedding_model": EMBEDDING_MODEL_PRIMARY if EXPOSE_DEBUG_STATUS else "configured",
        "chunks_path": _state.get("chunks_path") if EXPOSE_DEBUG_STATUS else None,
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if not _state:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        hits, used_filter, reranked = _do_search(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid retrieval configuration") from exc
    return SearchResponse(
        query=req.query,
        strategy=req.strategy,
        reranked=reranked,
        used_filter=used_filter,
        hits=_hits_to_out(hits),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not _state:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        llm, hits, used_filter, reranked, context_hits, prompt, expanded = _prepare_ask(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid retrieval configuration") from exc

    answer: Optional[str] = None
    citations: List[Dict[str, Any]] = []
    if hits:
        answer, citations = _generate_answer(
            llm, prompt, context_hits, req.max_new_tokens, req.query, _extra_queries(req, expanded)
        )

    return AskResponse(
        query=req.query,
        strategy=req.strategy,
        reranked=reranked,
        used_filter=used_filter,
        hits=_hits_to_out(context_hits),
        retrieved_hits=_hits_to_out(hits),
        answer=answer,
        citations=citations,
    )


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
        context_hits = expand_hits_with_neighbors(
            hits,
            _state.get("docs_by_id", {}),
            neighbors=ANSWER_CONTEXT_NEIGHBORS,
            max_chars=ANSWER_CONTEXT_MAX_CHARS,
        )
        if getattr(llm, "provider_name", "") == "llama":
            context_hits = fit_context_to_llm(
                context_hits, hits, req.query, history=history_dicts,
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
                     extra_queries=()):
    if query and not assess_evidence(query, context_hits, extra_queries=extra_queries).sufficient:
        return REFUSAL_LINE, []
    try:
        raw_answer = llm.generate(prompt, max_new_tokens=max_new_tokens)
    except Exception as exc:
        if EXPOSE_DEBUG_STATUS:
            print(f"[backend] LLM provider unavailable: {exc}", flush=True)
        return _llm_error_message(exc), []
    return _finalize_answer(llm, prompt, context_hits, raw_answer, max_new_tokens)


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Streaming variant of /ask: SSE with status, meta and done events."""
    if not _state:
        raise HTTPException(status_code=503, detail="Service not ready")

    def event_source():
        steps = _prepare_ask_steps(req)
        try:
            while True:
                try:
                    yield _sse("status", {"stage": next(steps)})
                except StopIteration as stop:
                    prepared = stop.value
                    break
        except ValueError:
            yield _sse("error", {"detail": "Invalid retrieval configuration"})
            return
        llm, hits, used_filter, reranked, context_hits, prompt, expanded = prepared
        extras = _extra_queries(req, expanded)
        yield _sse(
            "meta",
            {
                "query": req.query,
                "strategy": req.strategy,
                "reranked": reranked,
                "used_filter": used_filter,
                "hits": [h.model_dump() for h in _hits_to_out(context_hits)],
                "retrieved_hits": [h.model_dump() for h in _hits_to_out(hits)],
            },
        )
        if not hits:
            yield _sse("done", {"answer": None, "citations": []})
            return
        if not assess_evidence(req.query, context_hits, extra_queries=extras).sufficient:
            yield _sse("done", {"answer": REFUSAL_LINE, "citations": []})
            return

        yield _sse("status", {"stage": "generating"})
        raw_answer = ""
        try:
            if hasattr(llm, "generate_stream"):
                # Raw tokens cannot be claim-validated incrementally. Buffer them
                # and expose only the post-validation answer in the done event.
                for delta in llm.generate_stream(prompt, max_new_tokens=req.max_new_tokens):
                    raw_answer += delta
            else:
                raw_answer = llm.generate(prompt, max_new_tokens=req.max_new_tokens)
        except Exception as exc:
            if EXPOSE_DEBUG_STATUS:
                print(f"[backend] stream failed, falling back: {exc}", flush=True)
            # The non-streaming path carries the full retry/fallback logic.
            answer, citations = _generate_answer(llm, prompt, context_hits, req.max_new_tokens, req.query, extras)
            yield _sse("done", {"answer": answer, "citations": citations})
            return

        yield _sse("status", {"stage": "validating"})
        answer, citations = _finalize_answer(llm, prompt, context_hits, raw_answer, req.max_new_tokens)
        yield _sse("done", {"answer": answer, "citations": citations})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
