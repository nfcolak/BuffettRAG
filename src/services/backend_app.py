"""Backend FastAPI service.

Deployment: runs on the Nuvolos **Backend app**.
Responsibilities:
    - Loads chunks once at startup
    - Connects to pgvector (Database app) for vector storage
    - Hosts the embedding model (bge-small) and reranker (bge-reranker-base)
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

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
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
    VECTOR_BACKEND,
)
from src.embeddings import BGEEmbedder
from src.generation.prompt import build_cited_prompt, format_answer_markdown, parse_citations, strip_chat_artifacts
from src.generation.providers import create_llm_provider
from src.retrieval.context import build_doc_lookup, expand_hits_with_neighbors
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


class AskRequest(SearchRequest):
    max_new_tokens: int = Field(default=600, ge=1, le=1200)
    llm_provider: Optional[Literal["openai", "openrouter", "anthropic"]] = None
    llm_api_key: Optional[str] = Field(default=None, max_length=4096)
    llm_model: Optional[str] = Field(default=None, max_length=200)


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
        key = f"{client_key(request)}:{path}"
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
        hits, used_filter, reranked = _do_search(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    answer: Optional[str] = None
    citations: List[Dict[str, Any]] = []

    if hits:
        context_hits = expand_hits_with_neighbors(
            hits,
            _state.get("docs_by_id", {}),
            neighbors=ANSWER_CONTEXT_NEIGHBORS,
            max_chars=ANSWER_CONTEXT_MAX_CHARS,
        )
        prompt = build_cited_prompt(query=req.query, hits=context_hits)
        try:
            raw_answer = _llm_for_request(req).generate(prompt, max_new_tokens=req.max_new_tokens)
            answer = format_answer_markdown(strip_chat_artifacts(raw_answer))
            citations = parse_citations(answer, context_hits)
        except Exception as exc:
            if EXPOSE_DEBUG_STATUS:
                print(f"[backend] LLM provider unavailable: {exc}", flush=True)
            answer = _llm_error_message(exc)

    return AskResponse(
        query=req.query,
        strategy=req.strategy,
        reranked=reranked,
        used_filter=used_filter,
        hits=_hits_to_out(hits),
        answer=answer,
        citations=citations,
    )
