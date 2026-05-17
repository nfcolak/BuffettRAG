"""Central configuration for the BuffettRAG project.

Kept backward compatible with the original ingestion pipeline (paths,
CHUNK_SIZE, CHUNK_OVERLAP, year range). New sections cover embeddings,
vector store, retrieval, generation, and evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
OUTPUT_DIR = BASE_DIR / "processed_data"
CHUNKS_FILE = OUTPUT_DIR / "chunks.jsonl"
CHUNKS_V2_FILE = OUTPUT_DIR / "chunks_v2.jsonl"  # output of improved pipeline
METADATA_FILE = OUTPUT_DIR / "metadata.json"
METADATA_V2_FILE = OUTPUT_DIR / "metadata_v2.json"

INDEX_DIR = BASE_DIR / "indices"
CHROMA_DIR = INDEX_DIR / "chroma"
FAISS_DIR = INDEX_DIR / "faiss"

EVAL_DIR = BASE_DIR / "eval_results"

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

START_YEAR = 1977
END_YEAR = 2024
TOTAL_LETTERS = END_YEAR - START_YEAR + 1

TEXT_FILES_PATTERN = "buffet_*.txt"
PDF_FILES_PATTERN = "buffet_*.pdf"

# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------
# Primary -- ~130MB, fast, MIT
EMBEDDING_MODEL_PRIMARY = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM_PRIMARY = 384

# Higher-quality alternative
EMBEDDING_MODEL_LARGE = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM_LARGE = 768

# bge-* prepends a query instruction for retrieval
BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

EMBEDDING_BATCH_SIZE = 64
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # 'cuda' on Nuvolos

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

# Reranker
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_TOP_K = 8

# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------
LLM_MODEL_PRIMARY = "Qwen/Qwen2.5-3B-Instruct"
LLM_MODEL_ALT = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_MAX_NEW_TOKENS = 400
LLM_TEMPERATURE = 0.2
LLM_TOP_P = 0.9

# -----------------------------------------------------------------------------
# Service URLs (3-tier deployment on Nuvolos)
# -----------------------------------------------------------------------------
# Backend FastAPI URL -- Frontend uses this to reach the RAG service.
# On Nuvolos, the Backend app gets a fixed hostname inside the instance.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
# Trainer LLM service URL -- Backend forwards generation requests here.
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8001")

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
EVAL_TOP_K_VALUES = [1, 3, 5, 10]
