"""Central configuration for the BuffettRAG project.

Kept backward compatible with the original ingestion pipeline (paths,
CHUNK_SIZE, CHUNK_OVERLAP, year range). New sections cover embeddings,
vector store, retrieval, generation, and evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


def _csv_env(name: str, default: str = "") -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "processed"
CHUNKS_FILE = OUTPUT_DIR / "chunks.jsonl"
CHUNKS_V2_FILE = OUTPUT_DIR / "chunks_v2.jsonl"  # output of improved pipeline
METADATA_FILE = OUTPUT_DIR / "metadata.json"
METADATA_V2_FILE = OUTPUT_DIR / "metadata_v2.json"

INDEX_DIR = DATA_DIR / "indices"
CHROMA_DIR = INDEX_DIR / "chroma"
FAISS_DIR = INDEX_DIR / "faiss"

EVAL_DIR = DATA_DIR / "evaluation"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Ingestion / chunking
# -----------------------------------------------------------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Improved pipeline uses larger semantic chunks because bge-* handles them well
# and Buffett's letters are paragraph-heavy.
CHUNK_SIZE_V2 = 800
CHUNK_OVERLAP_V2 = 120
CHUNK_MIN_CHARS_V2 = 160

START_YEAR = 1977
END_YEAR = 2024
TOTAL_LETTERS = END_YEAR - START_YEAR + 1

TEXT_FILES_PATTERN = "buffet_*.txt"
PDF_FILES_PATTERN = "buffet_*.pdf"

# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------
# Primary -- ~440MB, 768-dim, MIT
EMBEDDING_MODEL_PRIMARY = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_DIM_PRIMARY = 768

# Alias kept for get_embedder('base'/'large') compatibility; same model as primary.
EMBEDDING_MODEL_LARGE = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM_LARGE = 768

# bge-* prepends a query instruction for retrieval
BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

EMBEDDING_BATCH_SIZE = 64
def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", _default_device())

# -----------------------------------------------------------------------------
# Vector store
# -----------------------------------------------------------------------------
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "pgvector")  # 'pgvector' | 'chroma' | 'faiss'
CHROMA_COLLECTION = "buffett_letters"

# pgvector / Postgres connection.
# Nuvolos exposes the Database app via fixed hostnames inside the same
# instance. These env vars are typically set by the Nuvolos UI; we fall back
# to localhost defaults so the code is also runnable outside the platform.
PG_HOST = os.getenv("PGHOST", "localhost")
PG_PORT = int(os.getenv("PGPORT", "5432"))
PG_USER = os.getenv("PGUSER", "postgres")
PG_PASSWORD = os.getenv("PGPASSWORD", "postgres")
PG_DATABASE = os.getenv("PGDATABASE", "postgres")
PG_TABLE = os.getenv("PG_TABLE", "buffett_chunks")
# IVFFlat index list count -- rule of thumb: rows/1000, capped to a few hundred
PG_IVFFLAT_LISTS = int(os.getenv("PG_IVFFLAT_LISTS", "100"))

# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------
DEFAULT_TOP_K = 8           # final passages returned to the LLM
RETRIEVAL_FETCH_K = 30      # candidates fetched before reranking / fusion
HYBRID_ALPHA = 0.6          # weight for vector score; (1 - alpha) for BM25
RRF_K = 60                  # reciprocal rank fusion constant
ANSWER_CONTEXT_NEIGHBORS = int(os.getenv("ANSWER_CONTEXT_NEIGHBORS", "1"))
# Per-passage context budget. Modern cloud models handle 100k+ token windows;
# 9000 chars (~2250 tokens) per passage keeps neighbor expansion from truncating.
ANSWER_CONTEXT_MAX_CHARS = int(os.getenv("ANSWER_CONTEXT_MAX_CHARS", "9000"))

# Reranker
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K = 8

# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------
LLM_MAX_NEW_TOKENS = 600
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openrouter").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
# Free-tier models are often rate-limited upstream; retry once on this model.
OPENROUTER_FALLBACK_MODEL = os.getenv("OPENROUTER_FALLBACK_MODEL", "openai/gpt-oss-120b:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "BuffettRAG")

# -----------------------------------------------------------------------------
# Service URLs (3-tier deployment on Nuvolos)
# -----------------------------------------------------------------------------
# Backend FastAPI URL -- Frontend uses this to reach the RAG service.
# On Nuvolos, the Backend app gets a fixed hostname inside the instance.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
# -----------------------------------------------------------------------------
# API security
# -----------------------------------------------------------------------------
# Leave API_KEYS empty for local-only development. In shared or public
# deployments, set comma-separated keys and send one as X-API-Key.
API_KEYS = _csv_env("API_KEYS")

CORS_ORIGINS = _csv_env(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:5173,http://127.0.0.1:5173",
)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
# Only trust X-Forwarded-For for rate-limit client identity when the service
# actually runs behind a reverse proxy that sets it; otherwise the header is
# client-controlled and lets callers reset their own rate-limit bucket.
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "0") == "1"
EXPOSE_DEBUG_STATUS = os.getenv("EXPOSE_DEBUG_STATUS", "0") == "1"

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
EVAL_TOP_K_VALUES = [1, 3, 5, 10]
