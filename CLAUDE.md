# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BuffettRAG** is a Retrieval-Augmented Generation (RAG) system for querying Warren Buffett's shareholder letters (1977–2024). It's deployed on Nuvolos as a 3-tier stack:
- **Frontend**: Streamlit UI for interactive querying
- **Backend**: FastAPI service handling retrieval, reranking, and citation parsing
- **Database**: PostgreSQL with pgvector for vector search
- **LLM Service**: Separate GPU-based service for generation (Qwen2.5-7B)

The system ingests 48 shareholder letters, chunks them semantically, embeds with BGE models, and uses hybrid retrieval (vector + BM25) with optional reranking.

## Architecture & Key Concepts

### Data Flow

```
Raw Letters (txt/pdf) 
  → Pipeline (ingestion/pipeline_v2.py) 
  → Chunks + Topics (JSONL) 
  → Embeddings (BGE small, 384-dim)
  → Vector Store (pgvector / Chroma / FAISS)
  → Retriever (hybrid/vector/metadata/naive strategies)
  → Reranker (BGE cross-encoder, optional)
  → LLM (Qwen2.5-7B, optional)
  → Citations + Answer
```

### Core Components

**Ingestion (`src/ingestion/`)**
- `pipeline_v2.py`: Main pipeline (improved over original). Extracts PDFs with PyMuPDF (fallback: pdfplumber), cleans headers/footers/hyphens, chunks with LangChain's RecursiveCharacterTextSplitter preserving paragraph/sentence boundaries, auto-tags topics.
- `pdf_extractor.py`: PDF text extraction with text cleaning.
- `chunker.py`: Semantic chunking using LangChain (respects paragraph/sentence/word boundaries).
- `topic_tagger.py`: Auto-labels chunks with topics (inflation, technology, derivatives, etc.).

**Vector Store (`src/vector_store.py`)**
- Unified interface supporting three backends: **pgvector** (production), **Chroma**, **FAISS**.
- `StoredDoc`: id, text, metadata (year, decade, source_file, topics).
- `SearchHit`: Same + similarity score.
- Handles embedding batching and persistence.

**Embeddings (`src/embeddings.py`)**
- `BGEEmbedder`: Wraps sentence-transformers BAAI/bge-* models.
- Applies BGE query instruction ("Represent this sentence for searching...") to queries only (not documents).
- Supports bge-small-en-v1.5 (384-dim, primary) and bge-base-en-v1.5 (768-dim, alt).

**Retrieval (`src/retrieval/`)**
- `retriever.py`: Four strategies:
  - **naive**: Vector search only (baseline).
  - **metadata**: Vector + year/decade/topic filtering (auto-detects years in queries with regex).
  - **vector**: Pure vector + optional rerank.
  - **hybrid**: Vector + BM25 fused with Reciprocal Rank Fusion (RRF). Recommended strategy.
- `bm25.py`: Sparse retrieval using rank-bm25.
- `reranker.py`: CrossEncoderReranker (BAAI/bge-reranker-base) reranks candidates by relevance.

**Generation (`src/generation/`)**
- `prompt.py`: Citation-aware prompt templates (supports Qwen ChatML, Mistral, Llama 3 formats). Model instructed to cite as `[n]` where n is passage index.
- `llm.py`: `LocalLLM` loader for Qwen/Mistral on CUDA or CPU (fp32 fallback for testing).

**Pipeline (`src/pipeline.py`)**
- `BuffettRAGPipeline.build()`: Factory that composes embedder → vector store → retriever → optional reranker → optional LLM.
- `.ask()`: End-to-end query handling. Returns answer, citations, passages, metadata.
- `.compare_decades()`: Cross-decade analysis (query results per decade).

**Evaluation (`src/evaluation/`)**
- `retrieval_metrics.py`: Recall@k, NDCG, MRR.
- `citation_faithfulness.py`: Validates citations point to correct passages.
- `run_eval.py`: Full eval pipeline against gold set.
- `gold_set.py`: Human-curated reference answers for benchmark queries.

### Configuration

**`config.py`**: Single source of truth for all settings (kept backward-compatible with original ingestion).

Key sections:
- **Paths**: raw_data, processed_data, indices, eval_results.
- **Ingestion**: CHUNK_SIZE=500 (v1), CHUNK_SIZE_V2=800 (v2 preferred), overlaps.
- **Embeddings**: EMBEDDING_MODEL_PRIMARY (bge-small), EMBEDDING_DEVICE ('cpu' or 'cuda').
- **Vector Store**: VECTOR_BACKEND ('pgvector', 'chroma', 'faiss'), pgvector connection env vars (PGHOST, PGUSER, etc.).
- **Retrieval**: DEFAULT_TOP_K=8, RETRIEVAL_FETCH_K=30 (candidates before rerank), HYBRID_ALPHA=0.6 (vector vs BM25 weight), RRF_K=60.
- **Generation**: LLM_MODEL_PRIMARY (Qwen2.5-7B), LLM_MAX_NEW_TOKENS=400, temperature=0.2.
- **Services**: BACKEND_URL, LLM_SERVICE_URL.

### Services (3-Tier Deployment)

**Backend (`services/backend_app.py`)**
- FastAPI service on Nuvolos Backend app.
- Loads chunks once at startup, embeds them (CPU, ~10–15 min first time).
- Connects to pgvector for storage/retrieval.
- Hosts embedder (bge-small) and reranker (bge-reranker-base).
- Endpoints: `/health`, `/stats`, `/search` (retrieval only), `/ask` (retrieval + LLM via HTTP to Trainer).
- Forwards generation requests to LLM service at `LLM_SERVICE_URL`.

**LLM Service (`services/llm_service.py`)**
- FastAPI on Nuvolos Trainer T4 (GPU).
- Loads Qwen2.5-7B-Instruct on CUDA (14 GB, 5–10 min download first time).
- Single `/generate` endpoint accepting prompt + max_new_tokens.
- Isolated network namespace; Backend reaches it via public port URL.

**Frontend (`app.py`)**
- Streamlit UI on Nuvolos Frontend app.
- Thin client: calls Backend over HTTP (no model loading).
- Session state for chat history, per-query configuration (strategy, top_k, rerank, year/decade filters).
- Renders retrieved passages with metadata (year, source, topics, relevance score).
- Displays citations extracted from LLM answer.

## Common Development Tasks

### Setup & Installation

```bash
# One-time: install core dependencies
pip install -r requirements.txt

# Or install service-specific dependencies
pip install -r services/requirements_backend.txt    # Backend only
pip install -r services/requirements_llm.txt        # LLM service only
pip install -r services/requirements_frontend.txt   # Frontend only
```

### Build & Index

```bash
# Ingest shareholder letters, chunk, tag topics, output chunks_v2.jsonl + metadata_v2.json
python -m src.ingestion.pipeline_v2

# Custom chunk settings
python -m src.ingestion.pipeline_v2 --chunk-size 600 --overlap 100

# Embed chunks and populate vector store (first time only; done at backend startup otherwise)
python scripts/build_index.py
```

### Run Tests

```bash
# Smoke tests (no LLM, no DB needed)
python -m tests.test_e2e

# With pytest (if installed)
pytest tests/test_e2e.py -q
```

### Local Development (Single Machine)

```bash
# Terminal 1: Build index
python scripts/build_index.py

# Terminal 2: Backend (port 8000)
export VECTOR_BACKEND=chroma  # or pgvector if DB running
uvicorn services.backend_app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 3: LLM service (port 8001, if you have a GPU)
uvicorn services.llm_service:app --host 0.0.0.0 --port 8001 --reload

# Terminal 4: Frontend (port 8501)
export BACKEND_URL=http://localhost:8000
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Then open http://localhost:8501.

### Validate Chunks

```bash
# View summary statistics
python validate_chunks.py

# Show 5 random chunk samples
python validate_chunks.py --sample 5

# Show all chunks from 1977
python validate_chunks.py --year 1977
```

### End-to-End Query (CLI)

```bash
# Build pipeline, then query programmatically
python -c "
from src.pipeline import BuffettRAGPipeline
pipeline = BuffettRAGPipeline.build()
result = pipeline.ask('How did Buffett react to the 2008 financial crisis?')
print(result['answer'])
for c in result['citations']:
    print(c)
"
```

### Evaluation

```bash
# Run full retrieval evaluation against gold set
python -m src.evaluation.run_eval

# Benchmark specific strategies
python -m src.evaluation.run_eval --strategies naive vector hybrid --top_k 5 10
```

## Key Design Decisions

### Why Four Retrieval Strategies?

Different use cases need different trade-offs:
- **naive**: Baseline; fast but ignores metadata.
- **metadata**: Auto-detects years in queries (regex); filters before reranking.
- **vector**: Pure semantic similarity; best for nuanced questions.
- **hybrid**: Combines semantic (BGE) + lexical (BM25) via RRF; recommended for balanced recall + precision.

### Why Separate LLM Service?

Backend and Trainer apps have separate network namespaces on Nuvolos. To allow Backend (CPU) to use Trainer (GPU) LLM, we expose the Trainer as a public port and Backend POSTs prompts via HTTP.

### Why BGE Models?

- Small (384-dim) is efficient for deployment on CPU.
- Built for retrieval (query-document asymmetry).
- Supports matryoshka scaling (reuse embeddings at lower dims).
- Pairs naturally with BGE reranker.

### Why RecursiveCharacterTextSplitter?

Original naive chunking on whitespace-collapsed text destroyed paragraph structure and cut mid-sentence. LangChain's recursive splitter respects paragraph → sentence → word boundaries, preserving semantic unit boundaries that embeddings expect.

## Important Patterns & Conventions

### Chunk Metadata Structure

```json
{
  "id": "2008_42",
  "text": "...",
  "year": 2008,
  "decade": 2000,
  "source_file": "buffet_2008.pdf",
  "topics": "derivatives,risk,financial-crisis",
  "chunk_index": 42,
  "total_chunks": 123,
  "chunk_id": "43/123"
}
```

### Where Clauses (Metadata Filtering)

Chroma/pgvector support `where` dicts:
```python
where = {"year": 2008}                           # Single year
where = {"year": {"$gte": 2000, "$lte": 2010}}  # Range
where = {"decade": 2000}                         # Decade
where = {"topics": "inflation"}                  # Topic match (partial)
```

### Citation Parsing

Model outputs `[1]` or `[1,3]` in answer. Parser extracts valid passage indices, validates against hit count, and builds citation metadata:
```python
{
  "marker": "[1]",
  "passage_indices": [0],
  "years": [2008],
  "sources": ["buffet_2008.pdf"]
}
```

## Nuvolos Deployment

See **NUVOLOS_DEPLOYMENT.md** for step-by-step setup:
1. Start Database app (pgvector preinstalled).
2. Editor app: Run `python -m src.ingestion.pipeline_v2`.
3. Backend app: `uvicorn services.backend_app:app` (embeds chunks on first startup).
4. Trainer T4 app: `uvicorn services.llm_service:app` (loads Qwen on startup).
5. Frontend app: `streamlit run app.py` (calls Backend over HTTP).

All apps share workspace storage at `/space_mounts/pars` for large files.

## File Structure (Brief)

```
/files
├── app.py                    # Streamlit frontend (thin client)
├── config.py                 # Centralized configuration
├── requirements.txt          # Full-stack dependencies
├── PIPELINE.md              # Ingestion pipeline overview
├── NUVOLOS_DEPLOYMENT.md    # Deployment guide
├── raw_data/                # Shareholder letters (txt/pdf)
├── processed_data/          # chunks.jsonl, metadata.json
├── indices/                 # Persistent vector stores (chroma, faiss)
├── eval_results/            # Evaluation outputs
├── src/
│   ├── pipeline.py          # End-to-end RAG pipeline
│   ├── vector_store.py      # Unified vector DB interface
│   ├── embeddings.py        # BGE embedder
│   ├── ingestion/           # PDF extraction, chunking, topic tagging
│   ├── retrieval/           # Retriever (4 strategies), BM25, reranker
│   ├── generation/          # Prompt templates, LLM loader, citation parsing
│   └── evaluation/          # Metrics, faithfulness, gold set
├── services/
│   ├── backend_app.py       # FastAPI backend service
│   ├── llm_service.py       # LLM service on GPU
│   └── requirements_*.txt
├── tests/
│   └── test_e2e.py          # Smoke tests
└── scripts/                 # Utility scripts (build_index.py)
```

