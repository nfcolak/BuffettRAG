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
from typing import Any, Dict, List, Optional

import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    API_KEYS,
    CORS_ORIGINS,
    DEFAULT_LLM_PROVIDER,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_PRIMARY,
    EXPOSE_DEBUG_STATUS,
    MAX_REQUEST_BODY_BYTES,
    PUBLIC_DEMO_MODE,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    TRUST_PROXY_HEADERS,
    VECTOR_BACKEND,
)
from src.storage.embeddings import BGEEmbedder
from src.generation.prompt import REFUSAL_LINE
from src.generation.compare import comparison_periods
from src.generation.evidence_gate import assess_evidence
from src.storage.index_manifest import ensure_index_identity, write_index_identity
from src.retrieval.context import build_doc_lookup
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import Retriever
from src.services import ask_flow
from src.services.schemas import AskRequest, AskResponse, SearchRequest, SearchResponse
from src.services.security import FixedWindowRateLimiter, client_key, is_authorized
from src.storage import get_vector_store, load_chunks_as_docs


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


async def startup() -> None:
    validate_deployment_security(
        public_demo=PUBLIC_DEMO_MODE,
        api_keys=API_KEYS,
        cors_origins=CORS_ORIGINS,
        debug=EXPOSE_DEBUG_STATUS,
        trust_proxy_headers=TRUST_PROXY_HEADERS,
    )
    chunks_path = ask_flow._resolve_chunks_path()
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

    ask_flow._state["vector_store"] = vs
    ask_flow._state["retriever"] = retriever
    ask_flow._state["docs"] = docs
    ask_flow._state["docs_by_id"] = build_doc_lookup(docs)
    ask_flow._state["chunks_path"] = str(chunks_path)
    if DEFAULT_LLM_PROVIDER == "grounded":
        # Create the provider now so it shares the retriever's reranker (one scorer).
        ask_flow.attach_grounded_resources(ask_flow._server_llm(), retriever)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "alive"}


@app.get("/ready")
def readiness():
    vs = ask_flow._state.get("vector_store")
    docs = ask_flow._state.get("docs")
    retriever = ask_flow._state.get("retriever")
    indexed_count = len(vs) if vs is not None else 0
    document_count = len(docs) if docs is not None else 0
    ready = retriever is not None and indexed_count > 0 and indexed_count == document_count
    payload: Dict[str, Any] = {
        "status": "ready" if ready else "not_ready",
        "indexed_count": indexed_count,
        "document_count": document_count,
    }
    if EXPOSE_DEBUG_STATUS:
        payload["chunks_path"] = ask_flow._state.get("chunks_path")
    return JSONResponse(payload, status_code=200 if ready else 503)


@app.get("/stats")
def stats() -> Dict[str, Any]:
    vs = ask_flow._state.get("vector_store")
    docs = ask_flow._state.get("docs", [])
    years = sorted({d.metadata.get("year") for d in docs if d.metadata.get("year")})
    return {
        "total_chunks": len(docs),
        "indexed_count": len(vs) if vs else 0,
        "year_range": [years[0], years[-1]] if years else [],
        "embedding_model": EMBEDDING_MODEL_PRIMARY if EXPOSE_DEBUG_STATUS else "configured",
        "chunks_path": ask_flow._state.get("chunks_path") if EXPOSE_DEBUG_STATUS else None,
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if not ask_flow._state:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        hits, used_filter, reranked = ask_flow._do_search(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid retrieval configuration") from exc
    return SearchResponse(
        query=req.query,
        strategy=req.strategy,
        reranked=reranked,
        used_filter=used_filter,
        hits=ask_flow._hits_to_out(hits),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not ask_flow._state:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        llm, hits, used_filter, reranked, context_hits, prompt, expanded = ask_flow._prepare_ask(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid retrieval configuration") from exc

    answer: Optional[str] = None
    citations: List[Dict[str, Any]] = []
    if ask_flow.is_grounded(llm):
        # The engine owns refusal/partial/no-hit/comparison decisions; no gate here.
        answer, citations = ask_flow._generate_grounded(
            llm, req.query, context_hits, req.max_new_tokens,
            [turn.model_dump() for turn in req.history],
        )
    elif hits or comparison_periods(req.query):
        answer, citations = ask_flow._generate_answer(
            llm, prompt, context_hits, req.max_new_tokens, req.query, ask_flow._extra_queries(req, expanded),
            [turn.model_dump() for turn in req.history],
        )

    return AskResponse(
        query=req.query,
        strategy=req.strategy,
        reranked=reranked,
        used_filter=used_filter,
        hits=ask_flow._hits_to_out(context_hits),
        retrieved_hits=ask_flow._hits_to_out(hits),
        answer=answer,
        citations=citations,
    )


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Streaming variant of /ask: SSE with status, meta and done events."""
    if not ask_flow._state:
        raise HTTPException(status_code=503, detail="Service not ready")

    def event_source():
        steps = ask_flow._prepare_ask_steps(req)
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
        extras = ask_flow._extra_queries(req, expanded)
        yield _sse(
            "meta",
            {
                "query": req.query,
                "strategy": req.strategy,
                "reranked": reranked,
                "used_filter": used_filter,
                "hits": [h.model_dump() for h in ask_flow._hits_to_out(context_hits)],
                "retrieved_hits": [h.model_dump() for h in ask_flow._hits_to_out(hits)],
            },
        )
        if ask_flow.is_grounded(llm):
            yield _sse("status", {"stage": "generating"})
            answer, citations = ask_flow._generate_grounded(
                llm, req.query, context_hits, req.max_new_tokens,
                [turn.model_dump() for turn in req.history],
            )
            yield _sse("status", {"stage": "validating"})
            yield _sse("done", {"answer": answer, "citations": citations})
            return
        if comparison_periods(req.query):
            yield _sse("status", {"stage": "generating"})
            answer, citations = ask_flow._generate_answer(
                llm, prompt, context_hits, req.max_new_tokens, req.query, extras,
                [turn.model_dump() for turn in req.history],
            )
            yield _sse("done", {"answer": answer, "citations": citations})
            return
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
            answer, citations = ask_flow._generate_answer(llm, prompt, context_hits, req.max_new_tokens, req.query, extras)
            yield _sse("done", {"answer": answer, "citations": citations})
            return

        yield _sse("status", {"stage": "validating"})
        answer, citations = ask_flow._finalize_answer(llm, prompt, context_hits, raw_answer, req.max_new_tokens)
        yield _sse("done", {"answer": answer, "citations": citations})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
