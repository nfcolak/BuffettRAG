# BuffettRAG

BuffettRAG answers questions about Warren Buffett's Berkshire Hathaway shareholder letters (1977 to 2024) with sentence-level citations back to the source passages. All 48 letters are indexed as 5,831 paragraph-aware records with source hashes and neighbour provenance. A FastAPI backend runs hybrid retrieval and cross-encoder reranking, an embedded local LLM (llama.cpp, Qwen2.5-7B-Instruct fine-tuned with LoRA, GGUF) writes the answer from the retrieved passages alone, and a React frontend renders the answer next to the passages it cites.

## How a question is answered

The backend can expand the query (`EXPANSION_MODE`: `auto` expands only when first-pass evidence is weak, `always`, or `off`). The embedded LLM proposes up to eight extra search keywords (companies, people, events, financial terms) so that questions phrased outside the corpus vocabulary still land, for example "Middle East" maps to ISCAR and Israel. Expansion failures are swallowed and retrieval falls back to the original query.

Retrieval is hybrid. The expanded query runs through BM25 and through vector search over bge-base-en-v1.5 embeddings, and the two rankings are merged with reciprocal rank fusion. The original query and expanded variant contribute candidate rankings; temporal interpretation and cross-encoder reranking always use the original question. Normal hybrid retrieval sends up to `RERANK_CANDIDATES` (default 15) fused candidates to bge-reranker-v2-m3; temporal comparison can retain up to thirty per period before reranking. Near-duplicate passages are dropped by token-overlap comparison, since overlapping chunk windows would otherwise fill the context with repeats. The top passages are then widened with their neighboring chunks (the chunk file stores previous and next chunk ids) so the LLM sees full paragraphs while retrieval stays precise over compact chunks.

Generation is grounded by contract. The system prompt requires the model to test each passage against the question, answer only from passages that pass, cite every sentence as [n], and output a fixed refusal line when nothing is relevant. Citation references are resolved server-side, with invalid numbers reported explicitly. This validates reference existence, not whether a claim is true or entailed. Answer responses expose the exact expanded passage text used in the prompt; original retrieval chunks remain available separately. The claim validator keeps a sentence only when its cited passage covers most of the sentence's content words with matching numbers and negation; this is deterministic lexical checking, not semantic entailment. Both a blocking endpoint and a server-sent-events streaming endpoint are available.

## Retrieval design choices

Year handling treats detected years as a hint rather than a constraint. When a query mentions "in 2008" or "the 1990s", the backend runs the search twice, with and without the year filter, and fuses both rankings with the filtered one weighted double. A wrongly guessed year can therefore lower ranking quality without zeroing out recall.

Questions that compare two periods ("How did his view on technology change from the 1990s to the 2020s?") are detected by pattern and decomposed into one search per period. Each sub-search keeps the full original query so the embedding stays on topic while only the year filter changes, and the final selection reserves a top candidate from each nonempty period when at least two passage slots are available. This preserves coverage, but does not itself establish relevance or a change of opinion.

Three vector backends share one interface, and each persists an index identity manifest (corpus, model, dimension) that is checked at startup. pgvector is the production store, Chroma and FAISS remain available for local work and comparison runs, selected with `VECTOR_BACKEND` (see configuration). Metadata filters are validated against a field whitelist and built as parameterized SQL, and the table name is checked against a strict identifier pattern.

## Evaluation

The evaluation pipeline scores retrieval strategies against a 50-query gold set with year-labeled relevance judgments. On that set, hybrid retrieval with reranking reaches MRR 0.739, recall@1 0.60 and recall@10 0.98. The historical answer run attempted 50 questions: 39 were scored and 11 failed at the provider. Its citation coverage was 0.912 and lexical support proxy 0.254; neither is a measured faithfulness rate, and the old artifact did not save passage text. Raw reports live in `data/evaluation/`.

## Held-out answer benchmark

The frozen 24-question set in `data/evaluation/heldout_v1/` tests cited answers, required claims and refusals through the backend answer flow. These runs use BM25 retrieval and temperature 0. Scoring uses deterministic lexical claim checks, not an LLM judge; it does not establish semantic entailment.

The final results are recorded in [`comparison.md`](data/evaluation/heldout_v1/final/comparison.md):

| System | Accepted | Required claims met | Correct refusals | Unexpected refusals | Provider failures | Mean latency ms |
| --- | --- | --- | --- | --- | --- | --- |
| Past result: removed extractive engine | 18/24 | 23/31 | 2/2 | 0 | 0 | 43.578 |
| Base Qwen2.5-1.5B | 3/24 | 2/31 | 2/2 | 18 | 0 | 1171.486 |
| Fine-tuned round 1 | 5/24 | 4/31 | 2/2 | 7 | 0 | 652.433 |
| Fine-tuned round 2 | 10/24 | 10/31 | 2/2 | 3 | 0 | 1139.950 |

The extractive engine (first row) was removed from the code and is no longer a runnable option; its row is a past result, and its fixes were diagnosed partly on this held-out set, so its 18/24 is optimistic. The LLM rows were not tuned on it. Latencies are measurements from these runs, not deployment guarantees.

## Quick start

Python 3.10+ is recommended.

```bash
python3 -m venv .venv && . .venv/bin/activate
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

The frontend defaults to the backend at `http://localhost:8000`; it has no provider, model or key settings.

The first backend start downloads the embedding model (~440MB) and the reranker (~2.3GB). On CPU the reranker adds noticeable latency per query; a GPU removes most of it.

## Embedded answer model

Answers come from a small model that runs inside the backend process through llama.cpp (`llama-cpp-python`): Qwen2.5-7B-Instruct fine-tuned with LoRA on Kaggle (FT-r1), 4-bit GGUF (`q4_k_m`, about 4.7 GB). Without the file the backend still starts and search keeps working, but `/ask` returns an "LLM unavailable" message. **No external API keys are used anywhere**: no OpenAI, Anthropic or OpenRouter calls, no per-request provider or key fields, and the frontend stores no keys.

The GGUF is produced by the Kaggle/Colab LoRA notebook under `colab/`, is not downloadable, and is git-ignored; place it at `models/ft/7b_r1/buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf`.

Providers: `llama` (default), `grounded`, `mlx`. If `llama` is selected but the model file is missing or `llama_cpp` cannot be imported, the factory logs one warning and the backend runs without an answer engine: search works, `/ask` returns the "LLM unavailable" message.

Latency: with Metal (`LLM_GPU_LAYERS=-1`) an answer takes a few seconds. On a CPU-only server (`LLM_GPU_LAYERS=0`) expect roughly 10 to 40 seconds per answer depending on cores. To keep the prompt inside what a 1.5B model handles, the prompt uses only the first `LLM_CONTEXT_PASSAGES` expanded passages (5 by default, 8 for 7B GGUF files), each cut around its anchor chunk to `LLM_PASSAGE_MAX_CHARS`, and trailing passages are dropped until the prompt fits `LLM_N_CTX` minus the answer budget (4 chars per token estimate). Citation numbers always match the passages shown.

Query expansion is off by default (`EXPANSION_MODE=off`): a 1.5B model proposes unreliable keywords. Set `auto` or `always` only if you accept that.

## Fine-tuning (LoRA)

The teacher, Qwen2.5-7B running in MLX 4-bit, generated cited answers from the corpus. Leakage checks exclude held-out evidence and its neighbors, reject questions that overlap the frozen set, and check that training and validation passages remain separate. Qwen2.5-1.5B was then fine-tuned with LoRA through `mlx-lm`.

Round 1 used 193 single-sentence training examples and 13 validation examples, stored in `data/ft/`. Round 2 used answers of 2–4 cited sentences, including comparison, multi-part and follow-up questions; `data/ft_v2/` contains 560 training examples and 70 validation examples. These split counts come from each directory's `manifest.json`. Training records for round 2 are in `data/ft_v2/round2_metrics/`; the held-out table above uses the final comparison, not the earlier metrics snapshot.

The resumable pipeline generates examples, splits and checks leakage, trains LoRA, exports a GGUF model and evaluates it:

```bash
bash scripts/ft/run_round.sh <round> <target_examples> <max_gen_minutes>
```

The current runner uses `data/ft_v2/` and stores stage markers and model artifacts under `models/ft/round<round>/`. It needs the local teacher and student weights, MLX/`mlx-lm`, and the llama.cpp conversion and quantization tools; it is not part of the backend quick start.

The fine-tuned GGUF files are not published: they are git-ignored under `models/ft/`. To serve a locally generated round-2 model:

```bash
LLM_MODEL_PATH=models/ft/7b_r1/buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf uvicorn src.services.backend_app:app --host 0.0.0.0 --port 8000
```

## Configuration

Everything is set through environment variables, read in `config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `DEFAULT_LLM_PROVIDER` | `llama` | `llama` (embedded GGUF model), `grounded` or `mlx` |
| `LLM_MODEL_PATH` | `models/ft/7b_r1/buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf` | GGUF file, relative to the repo root |
| `LLM_N_CTX` | `8192` | Context window |
| `LLM_N_THREADS` | `0` | CPU threads (0 = auto) |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature (seed is fixed) |
| `LLM_GPU_LAYERS` | `-1` | `-1` all layers on Metal/GPU, `0` CPU only |
| `LLM_CONTEXT_PASSAGES` | `5` (8 for 7B GGUF files) | Passages placed in the prompt for the llama provider |
| `LLM_PASSAGE_MAX_CHARS` | `1800` | Per-passage character cap for the llama provider |
| `VECTOR_BACKEND` | `pgvector` | `pgvector`, `chroma`, or `faiss` |
| `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`, `PG_TABLE` | localhost defaults | Postgres connection for pgvector |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model id |
| `EMBEDDING_DEVICE` | auto | `cuda` when available, otherwise `cpu` |
| `RERANK_CANDIDATES` | `15` | Max candidates sent to the cross-encoder |
| `EXPANSION_MODE` | `off` | `auto`, `always`, or `off` for the LLM query-expansion call |

For shared or public deployments there are separate hardening knobs.

| Variable | Default | Purpose |
|---|---|---|
| `API_KEYS` | empty | Comma-separated backend keys; requests must send one as `X-API-Key`. Empty disables auth for local development |
| `CORS_ORIGINS` | localhost ports | Allowed browser origins |
| `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS` | 60, 60 | Fixed-window rate limit per client and path |
| `TRUST_PROXY_HEADERS` | `0` | Set to `1` only behind a reverse proxy, so rate limiting keys on `X-Forwarded-For` |
| `EXPOSE_DEBUG_STATUS` | `0` | Include internal paths and model names in `/ready` and `/stats` |
| `PUBLIC_DEMO_MODE` | `0` | Public-demo hardening (see `config.py`) |
| `MAX_REQUEST_BODY_BYTES` | see `config.py` | Reject request bodies larger than this |

## API

- `GET /health` returns liveness only (`{"status": "alive"}`), no auth required
- `GET /ready` reports `indexed_count` and `document_count` (503 when the index is not ready), no auth required
- `GET /stats` reports corpus statistics
- `POST /search` runs retrieval only and returns scored passages
- `POST /ask` runs retrieval plus generation and returns the answer with parsed citations
- `POST /ask/stream` streams the answer as server-sent events (meta, status, done); the answer is validated before it is sent, so there is no token streaming

`/ask` takes no provider, key or model fields; it always uses the server's embedded provider.

## Code layout

- `src/ingestion/`: PDF/text extraction, paragraph-aware chunking and topic tagging.
- `src/retrieval/`: BM25, vector retrieval, rank fusion, reranking and context expansion.
- `src/storage/`: Vector stores, embeddings and index identity manifests.
- `src/generation/`: Prompt contract, evidence gate, comparison and answer providers.
- `src/evaluation/`: Retrieval metrics, answer benchmarks and lexical claim validation.
- `src/services/`: `backend_app` routes, `ask_flow`, request/response schemas and security.
- `scripts/index/`: Index building, corpus audits and chunk validation.
- `scripts/eval/`: Ablations and answer/live benchmarks.
- `scripts/ft/`: Teacher-data generation, leakage checks, LoRA training and GGUF export/evaluation.
- `frontend/`: React/Vite answer and cited-passage interface.
- `tests/`: Unit, regression and end-to-end smoke tests.

## Project layout

```text
.
├── config.py               # central paths and runtime settings
├── data/
│   ├── raw/                # source shareholder letters
│   ├── processed/          # chunk files and metadata
│   ├── indices/            # local vector-store persistence
│   ├── evaluation/         # retrieval and answer reports, frozen held-out set
│   ├── ft/                 # round-1 data and manifest
│   └── ft_v2/              # round-2 data, manifest and training metrics
├── frontend/               # React/Vite frontend
├── scripts/
│   ├── index/              # index building and corpus validation
│   ├── eval/               # quality checks, ablations and benchmarks
│   ├── ft/                 # fine-tuning pipeline
│   └── ask.py              # question CLI
├── src/
│   ├── ingestion/          # PDF/text extraction, chunking, topic tagging
│   ├── retrieval/          # BM25, vector search, RRF, reranking, dedup
│   ├── storage/            # vector stores, embeddings, index manifests
│   ├── generation/         # prompt, evidence gate, comparison, providers
│   ├── evaluation/         # gold set and metrics pipeline
│   └── services/           # backend routes, answer flow, schemas, security
└── tests/                  # unit and end-to-end smoke tests
```

Tests run without a database (the llama smoke test runs only if the GGUF file exists) (install `pytest` for the regression suite). Corpus audit:

```bash
PYTHON_DOTENV_DISABLED=1 python -m pytest tests -q
PYTHON_DOTENV_DISABLED=1 python scripts/index/audit_corpus.py
```

## Limitations

The embedded model is smaller than large hosted models: it can miss nuance and refuse more often, and claim validation drops unsupported sentences. The corpus is English only, and answers are only as current as the 2024 letter. The letters themselves are copyright Berkshire Hathaway and are included here for research use; the originals are published at berkshirehathaway.com.
