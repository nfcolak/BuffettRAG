# BuffettRAG

BuffettRAG answers questions about Warren Buffett's Berkshire Hathaway shareholder letters (1977 to 2024) with sentence-level citations back to the source passages. All 48 letters are split into 5,833 chunks of roughly 800 characters. A FastAPI backend runs hybrid retrieval and cross-encoder reranking, an LLM writes the answer from the retrieved passages alone, and a React frontend renders the answer next to the passages it cites.

## How a question is answered

The backend first expands the query. The configured LLM proposes up to eight extra search keywords (companies, people, events, financial terms) so that questions phrased outside the corpus vocabulary still land, for example "Middle East" maps to ISCAR and Israel. Expansion failures are swallowed and retrieval falls back to the original query.

Retrieval is hybrid. The expanded query runs through BM25 and through vector search over bge-base-en-v1.5 embeddings, and the two rankings are merged with reciprocal rank fusion. Thirty candidates go into a bge-reranker-v2-m3 cross-encoder, which reorders them by relevance to the actual question. Near-duplicate passages are dropped by token-overlap comparison, since overlapping chunk windows would otherwise fill the context with repeats. The top passages are then widened with their neighboring chunks (the chunk file stores previous and next chunk ids) so the LLM sees full paragraphs while retrieval stays precise over compact chunks.

Generation is grounded by contract. The system prompt requires the model to test each passage against the question, answer only from passages that pass, cite every sentence as [n], and output a fixed refusal line when nothing is relevant. Citations are parsed and validated server-side; markers pointing outside the passage list are dropped. Both a blocking endpoint and a server-sent-events streaming endpoint are available.

## Retrieval design choices

Year handling treats detected years as a hint rather than a constraint. When a query mentions "in 2008" or "the 1990s", the backend runs the search twice, with and without the year filter, and fuses both rankings with the filtered one weighted double. A wrongly guessed year can therefore lower ranking quality without zeroing out recall.

Questions that compare two periods ("How did his view on technology change from the 1990s to the 2020s?") are detected by pattern and decomposed into one search per period. Each sub-search keeps the full original query so the embedding stays on topic while only the year filter changes, and the per-period rankings are fused into one result set that spans both eras.

Three vector backends share one interface. pgvector is the production store, Chroma and FAISS remain available for local work and comparison runs, selected with `VECTOR_BACKEND` (see configuration). Metadata filters are validated against a field whitelist and built as parameterized SQL, and the table name is checked against a strict identifier pattern.

## Evaluation

The evaluation pipeline scores retrieval strategies against a 50-query gold set with year-labeled relevance judgments. On that set, hybrid retrieval with reranking reaches MRR 0.739, recall@1 0.60 and recall@10 0.98. A separate citation-faithfulness check over generated answers measures citation coverage at 0.91 with zero citations pointing at nonexistent passages. Raw reports live in `data/evaluation/`.

## Quick start

```bash
pip install -r requirements.txt

# build chunks from the letters in data/raw/
python -m src.ingestion.pipeline_v2

# start the backend (indexes chunks into the vector store on first run)
uvicorn src.services.backend_app:app --host 0.0.0.0 --port 8000

# start the frontend
cd frontend
npm install
VITE_BACKEND_URL=http://localhost:8000 npm run dev
```

The first backend start downloads the embedding model (~440MB) and the reranker (~2.3GB). On CPU the reranker adds noticeable latency per query; a GPU removes most of it.

The frontend has a settings dialog for choosing the LLM provider per browser. A provider API key entered there is sent only with `/ask` requests, and "remember on this device" keeps it in that browser's local storage.

## Configuration

Everything is set through environment variables, read in `config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `DEFAULT_LLM_PROVIDER` | `openrouter` | `openai`, `anthropic`, `openrouter`, or `local` (offline extractive engine) |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | empty, `gpt-4.1-mini` | OpenAI provider |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | empty, `claude-haiku-4-5-20251001` | Anthropic provider |
| `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` | empty, a free-tier model | OpenRouter provider; retries once on a fallback model when rate-limited |
| `VECTOR_BACKEND` | `pgvector` | `pgvector`, `chroma`, or `faiss` |
| `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`, `PG_TABLE` | localhost defaults | Postgres connection for pgvector |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model id |
| `EMBEDDING_DEVICE` | auto | `cuda` when available, otherwise `cpu` |

For shared or public deployments there are separate hardening knobs.

| Variable | Default | Purpose |
|---|---|---|
| `API_KEYS` | empty | Comma-separated backend keys; requests must send one as `X-API-Key`. Empty disables auth for local development |
| `CORS_ORIGINS` | localhost ports | Allowed browser origins |
| `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS` | 60, 60 | Fixed-window rate limit per client and path |
| `TRUST_PROXY_HEADERS` | `0` | Set to `1` only behind a reverse proxy, so rate limiting keys on `X-Forwarded-For` |
| `EXPOSE_DEBUG_STATUS` | `0` | Include internal paths and model names in `/health` and `/stats` |

## API

- `GET /health` reports index count, no auth required
- `GET /stats` reports corpus statistics
- `POST /search` runs retrieval only and returns scored passages
- `POST /ask` runs retrieval plus generation and returns the answer with parsed citations
- `POST /ask/stream` streams the answer as server-sent events (meta, delta, done)

`/ask` accepts an optional `llm_provider`, `llm_api_key` and `llm_model`, so a caller can bring their own key per request instead of configuring one on the server.

## Project layout

```text
.
├── config.py               # central paths and runtime settings
├── data/
│   ├── raw/                # source shareholder letters
│   ├── processed/          # chunk files and metadata
│   ├── indices/            # local vector-store persistence
│   └── evaluation/         # evaluation reports
├── frontend/               # React/Vite frontend
├── scripts/                # CLI utilities (build_index, ask, validation)
├── src/
│   ├── ingestion/          # PDF/text extraction, chunking, topic tagging
│   ├── retrieval/          # BM25, vector search, RRF, reranking, dedup
│   ├── generation/         # prompt contract and LLM provider wrappers
│   ├── evaluation/         # gold set and metrics pipeline
│   └── services/           # FastAPI backend and security helpers
└── tests/                  # unit and end-to-end smoke tests
```

Tests run without a database or an LLM, so they finish in seconds:

```bash
python -m unittest discover tests
```

## Limitations

Free-tier OpenRouter models are rate-limited upstream and sometimes refuse or stall; the backend retries once on a fallback model and surfaces remaining failures as marked error messages rather than answers. The corpus is English only, and answers are only as current as the 2024 letter. The letters themselves are copyright Berkshire Hathaway and are included here for research use; the originals are published at berkshirehathaway.com.
