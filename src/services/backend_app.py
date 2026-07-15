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
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    API_KEYS,
    ANSWER_CONTEXT_MAX_CHARS,
    ANSWER_CONTEXT_NEIGHBORS,
    CHUNKS_FILE,
    CHUNKS_V2_FILE,
    CORS_ORIGINS,
    DEFAULT_TOP_K,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_PRIMARY,
    EXPOSE_DEBUG_STATUS,
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
from src.generation.providers import create_llm_provider
from src.retrieval.context import build_doc_lookup, expand_hits_with_neighbors
from src.retrieval.query_expansion import expand_query
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import Retriever
from src.services.security import FixedWindowRateLimiter, client_key, is_authorized
from src.vector_store import SearchHit, get_vector_store, load_chunks_as_docs


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


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class AskRequest(SearchRequest):
    max_new_tokens: int = Field(default=900, ge=1, le=2000)
    llm_provider: Optional[Literal["openai", "openrouter", "anthropic", "local"]] = None
    llm_api_key: Optional[str] = Field(default=None, max_length=4096)
    llm_model: Optional[str] = Field(default=None, max_length=200)
    expand_query: bool = True
    history: List[HistoryTurn] = Field(default_factory=list, max_length=12)


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
    answer: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# App + global state
# -----------------------------------------------------------------------------

app = FastAPI(title="BuffettRAG Backend", version="1.0.0")
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


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    if path != "/health" and not is_authorized(request, API_KEYS):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        key = f"{client_key(request, TRUST_PROXY_HEADERS)}:{path}"
        if not _rate_limiter.allow(key):
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)

    return await call_next(request)


@app.on_event("startup")
async def startup() -> None:
    chunks_path = CHUNKS_V2_FILE if CHUNKS_V2_FILE.exists() else CHUNKS_FILE
    if not chunks_path.exists():
        raise RuntimeError(
            "No chunks file found. Run `python -m src.ingestion.pipeline_v2` first."
        )

    embedder = BGEEmbedder(model_name=EMBEDDING_MODEL_PRIMARY, device=EMBEDDING_DEVICE)
    docs = load_chunks_as_docs(chunks_path)

    vs = get_vector_store(backend=VECTOR_BACKEND, dim=embedder.dimension)
    if len(vs) == 0:
        embeddings = embedder.embed_documents([d.text for d in docs])
        vs.add(docs, embeddings)
    elif len(vs) != len(docs):
        # A partially populated store (interrupted indexing, stale chunks file)
        # would otherwise serve incomplete results silently.
        print(
            f"[backend] warning: vector store holds {len(vs)} rows but the chunks "
            f"file has {len(docs)}; re-run indexing if these should match",
            flush=True,
        )

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
    raw = str(exc).lower()
    if "insufficient_quota" in raw or "exceeded your current quota" in raw:
        return "[LLM unavailable: OpenAI API quota or billing limit reached]"
    if "invalid_api_key" in raw or "incorrect api key" in raw:
        return "[LLM unavailable: provider API key is invalid]"
    if "authentication_error" in raw or "no auth credentials" in raw or "unauthorized" in raw:
        return "[LLM unavailable: provider API key is missing or invalid]"
    if "api_key is required" in raw:
        return "[LLM unavailable: provider API key is missing]"
    if "model_not_found" in raw or "does not exist" in raw:
        return "[LLM unavailable: selected provider model is not available]"
    if any(m in raw for m in ("429", "rate-limit", "rate limit", "resourceexhausted", "resource exhausted", "limit reached", "502", "503", "overloaded")):
        return "[LLM unavailable: the free model is busy or rate-limited right now — please retry in a moment]"
    return "[LLM unavailable]"


def _do_search(req: SearchRequest):
    retriever: Retriever = _state["retriever"]
    result = retriever.search(
        query=req.query,
        strategy=req.strategy,
        top_k=req.top_k,
        fetch_k=req.fetch_k,
        rerank=req.rerank,
        where=req.where,
        auto_year_filter=req.auto_year_filter,
    )
    return result.hits, result.used_filter, result.reranked


def _llm_for_request(req: AskRequest):
    if req.llm_provider or req.llm_api_key or req.llm_model:
        return create_llm_provider(
            provider=req.llm_provider,
            api_key=req.llm_api_key or None,
            model=req.llm_model or None,
        )
    if "llm" not in _state:
        _state["llm"] = create_llm_provider()
    return _state["llm"]


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    vs = _state.get("vector_store")
    payload: Dict[str, Any] = {
        "status": "ok",
        "indexed_count": len(vs) if vs else 0,
    }
    if EXPOSE_DEBUG_STATUS:
        payload["chunks_path"] = _state.get("chunks_path")
    return payload


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
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
        llm, hits, used_filter, reranked, context_hits, prompt = _prepare_ask(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    answer: Optional[str] = None
    citations: List[Dict[str, Any]] = []
    if hits:
        answer, citations = _generate_answer(llm, prompt, context_hits, req.max_new_tokens)

    return AskResponse(
        query=req.query,
        strategy=req.strategy,
        reranked=reranked,
        used_filter=used_filter,
        hits=_hits_to_out(hits),
        answer=answer,
        citations=citations,
    )


def _prepare_ask(req: AskRequest):
    """Shared /ask preparation: expansion, retrieval, context and prompt."""
    llm = _llm_for_request(req)
    history_dicts = [turn.model_dump() for turn in req.history]

    search_req = req
    if req.expand_query:
        expanded = expand_query(
            req.query, llm, history_text=format_history_block(history_dicts)
        )
        if expanded:
            search_req = req.model_copy(update={"query": expanded})
            if EXPOSE_DEBUG_STATUS:
                print(f"[backend] expanded query: {expanded!r}", flush=True)

    hits, used_filter, reranked = _do_search(search_req)

    context_hits: List[Any] = []
    prompt = ""
    if hits:
        context_hits = expand_hits_with_neighbors(
            hits,
            _state.get("docs_by_id", {}),
            neighbors=ANSWER_CONTEXT_NEIGHBORS,
            max_chars=ANSWER_CONTEXT_MAX_CHARS,
        )
        prompt = build_cited_prompt(query=req.query, hits=context_hits, history=history_dicts)

    return llm, hits, used_filter, reranked, context_hits, prompt


def _finalize_answer(llm, prompt: str, context_hits, raw_answer: str, max_new_tokens: int):
    """Format a raw completion, apply the refusal-retry safety net, parse citations."""
    answer = format_answer_markdown(strip_chat_artifacts(raw_answer))
    # Free-tier models occasionally refuse borderline questions the
    # passages can answer; one retry is a cheap safety net.
    if answer.strip().startswith(REFUSAL_LINE):
        try:
            retry_raw = llm.generate(prompt, max_new_tokens=max_new_tokens)
            retry_answer = format_answer_markdown(strip_chat_artifacts(retry_raw))
            if retry_answer.strip() and not retry_answer.strip().startswith(REFUSAL_LINE):
                answer = retry_answer
        except Exception:
            pass
    return answer, parse_citations(answer, context_hits)


def _generate_answer(llm, prompt: str, context_hits, max_new_tokens: int):
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
    """Streaming variant of /ask: SSE with meta, delta and done events."""
    if not _state:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        llm, hits, used_filter, reranked, context_hits, prompt = _prepare_ask(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def event_source():
        yield _sse(
            "meta",
            {
                "query": req.query,
                "strategy": req.strategy,
                "reranked": reranked,
                "used_filter": used_filter,
                "hits": [h.model_dump() for h in _hits_to_out(hits)],
            },
        )
        if not hits:
            yield _sse("done", {"answer": None, "citations": []})
            return

        raw_answer = ""
        try:
            if hasattr(llm, "generate_stream"):
                for delta in llm.generate_stream(prompt, max_new_tokens=req.max_new_tokens):
                    raw_answer += delta
                    yield _sse("delta", {"text": delta})
            else:
                raw_answer = llm.generate(prompt, max_new_tokens=req.max_new_tokens)
                yield _sse("delta", {"text": raw_answer})
        except Exception as exc:
            if EXPOSE_DEBUG_STATUS:
                print(f"[backend] stream failed, falling back: {exc}", flush=True)
            # The non-streaming path carries the full retry/fallback logic.
            answer, citations = _generate_answer(llm, prompt, context_hits, req.max_new_tokens)
            yield _sse("done", {"answer": answer, "citations": citations})
            return

        answer, citations = _finalize_answer(llm, prompt, context_hits, raw_answer, req.max_new_tokens)
        yield _sse("done", {"answer": answer, "citations": citations})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
