"""Backend FastAPI service.

Deployment: runs on the Nuvolos **Backend app**.
Responsibilities:
    - Loads chunks once at startup
    - Connects to pgvector (Database app) for vector storage
    - Hosts the embedding model (bge-small) and reranker (bge-reranker-base)
    - Runs retrieval (vector / metadata / hybrid) and reranking
    - For answer generation, forwards prompts to the LLM service running on
      the Trainer app (which has the GPU + Qwen)

Endpoints:
    GET  /health
    GET  /stats
    POST /search       -- retrieval only
    POST /ask          -- retrieval + LLM (forwards to Trainer)

Run:
    uvicorn services.backend_app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHUNKS_FILE,
    CHUNKS_V2_FILE,
    DEFAULT_TOP_K,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_PRIMARY,
    LLM_SERVICE_URL,
    RETRIEVAL_FETCH_K,
    VECTOR_BACKEND,
)
from src.embeddings import BGEEmbedder
from src.generation.prompt import build_cited_prompt, parse_citations, strip_chat_artifacts
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import Retriever
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
    prompt_style: str = "qwen"
    max_new_tokens: int = Field(default=600, ge=1, le=1200)


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
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: Dict[str, Any] = {}


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


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    vs = _state.get("vector_store")
    return {
        "status": "ok",
        "indexed_count": len(vs) if vs else 0,
        "chunks_path": _state.get("chunks_path"),
    }


@app.get("/stats")
def stats() -> Dict[str, Any]:
    vs = _state.get("vector_store")
    docs = _state.get("docs", [])
    years = sorted({d.metadata.get("year") for d in docs if d.metadata.get("year")})
    return {
        "total_chunks": len(docs),
        "indexed_count": len(vs) if vs else 0,
        "year_range": [years[0], years[-1]] if years else [],
        "embedding_model": EMBEDDING_MODEL_PRIMARY,
        "chunks_path": _state.get("chunks_path"),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if not _state:
        raise HTTPException(status_code=503, detail="Service not ready")
    hits, used_filter, reranked = _do_search(req)
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

    hits, used_filter, reranked = _do_search(req)

    answer: Optional[str] = None
    citations: List[Dict[str, Any]] = []

    if LLM_SERVICE_URL and hits:
        prompt = build_cited_prompt(query=req.query, hits=hits, style=req.prompt_style)
        try:
            resp = requests.post(
                f"{LLM_SERVICE_URL}/generate",
                json={"prompt": prompt, "max_new_tokens": req.max_new_tokens},
                timeout=120,
            )
            resp.raise_for_status()
            raw_answer = resp.json().get("text", "")
            answer = strip_chat_artifacts(raw_answer)
            citations = parse_citations(answer, hits)
        except requests.RequestException as exc:
            answer = f"[LLM unavailable: {exc}]"

    return AskResponse(
        query=req.query,
        strategy=req.strategy,
        reranked=reranked,
        used_filter=used_filter,
        hits=_hits_to_out(hits),
        answer=answer,
        citations=citations,
    )
