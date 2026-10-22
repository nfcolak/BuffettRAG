# BuffettRAG

BuffettRAG answers questions about Warren Buffett's Berkshire Hathaway shareholder letters (1977 to 2024) with sentence-level citations back to the source passages. All 48 letters are indexed as 5,831 paragraph-aware records with source hashes and neighbour provenance. A FastAPI backend runs hybrid retrieval and cross-encoder reranking, an embedded local LLM (llama.cpp, Qwen2.5-7B-Instruct fine-tuned with LoRA, GGUF) writes the answer from the retrieved passages alone, and a React frontend renders the answer next to the passages it cites.

## How a question is answered

The backend can expand the query (`EXPANSION_MODE`: `auto` expands only when first-pass evidence is weak, `always`, or `off`). The embedded LLM proposes up to eight extra search keywords (companies, people, events, financial terms) so that questions phrased outside the corpus vocabulary still land, for example "Middle East" maps to ISCAR and Israel. Expansion failures are swallowed and retrieval falls back to the original query.

Retrieval is hybrid. The expanded query runs through BM25 and through vector search over bge-base-en-v1.5 embeddings, and the two rankings are merged with reciprocal rank fusion. The original query and expanded variant contribute candidate rankings; temporal interpretation and cross-encoder reranking always use the original question. Normal hybrid retrieval sends up to `RERANK_CANDIDATES` (default 30) fused candidates to bge-reranker-v2-m3; temporal comparison can retain up to thirty per period before reranking. Near-duplicate passages are dropped by token-overlap comparison, since overlapping chunk windows would otherwise fill the context with repeats. The top passages are then widened with their neighboring chunks (the chunk file stores previous and next chunk ids) so the LLM sees full paragraphs while retrieval stays precise over compact chunks.

Generation is grounded by contract. The system prompt requires the model to test each passage against the question, answer only from passages that pass, cite every sentence as [n], and output a fixed refusal line when nothing is relevant. Citation references are resolved server-side, with invalid numbers reported explicitly. This validates reference existence, not whether a claim is true or entailed. Answer responses expose the exact expanded passage text used in the prompt; original retrieval chunks remain available separately. The claim validator keeps a sentence only when its cited passage covers most of the sentence's content words with matching numbers and negation; this is deterministic lexical checking, not semantic entailment. Both a blocking endpoint and a server-sent-events streaming endpoint are available.

## Retrieval design choices

Year handling treats detected years as a hint rather than a constraint. When a query mentions "in 2008" or "the 1990s", the backend runs the search twice, with and without the year filter, and fuses both rankings with the filtered one weighted double. A wrongly guessed year can therefore lower ranking quality without zeroing out recall.

Questions that compare two periods ("How did his view on technology change from the 1990s to the 2020s?") are detected by pattern and decomposed into one search per period. Each sub-search keeps the full original query so the embedding stays on topic while only the year filter changes, and the final selection reserves a top candidate from each nonempty period when at least two passage slots are available. This preserves coverage, but does not itself establish relevance or a change of opinion.

FAISS is the single vector store; it persists an index identity manifest (corpus, model, dimension) that is checked at startup. Metadata filters are validated against a field whitelist.

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

## Finance product (2026-10)

### Goal

The app answers questions about Warren Buffett's financial views and Berkshire Hathaway's financial data (crises and macro events, sectors and companies, views over time, figures, whole-collection questions). Personal content from the letters (family, friends, anecdotes) is out of scope: the app replies with a fixed scope sentence instead of answering. The finance path is switched on with `FINANCE_PIPELINE=1`; with it off, the app behaves as described in the sections above.

### Pipeline

1. **Finance plan** (`src/query/plan.py`). Each question is turned into a plan by rules, without a model: scope (financial, personal or mixed), kind (specific, period, trend, comparison, numeric or corpus-wide), the years or ranges it asks for, and the sectors, companies, events and themes it names.
2. **Knowledge layer** (`data/knowledge/`, built offline). All 5,831 passages carry annotations (financial or not, sectors, themes, stance, personal-content flag). A numbers table holds 3,670 figures with their letter and year. 106 topic digests give a timeline per topic. Events and a taxonomy complete it. A MANIFEST lists the file hashes.
3. **Finance search** (`src/retrieval/finance_policy.py`). BM25 and vector search are fused as before, with extra candidate sources: the annotation-summary index, passages tagged with the plan's targets, digest timelines, decade blocks and extra theme searches for opinion questions. Year hints boost rather than filter, and questions that span years keep a minimum number of different letters.
4. **Window packing** (`src/generation/window_pack.py`, `token_count.py`). The reader's context is filled by token count, not characters, and drops neighbouring text before it drops served passages.
5. **FT-r1 reader.** The same embedded Qwen2.5-7B-Instruct LoRA model described below (FT-r1) writes the cited answer.
6. **Personal-sentence filter** (`src/generation/finance_answer.py`). A sentence that carries a personal cue and cites only personal or non-financial passages is removed from the answer.

### Measured results

Cases come from a private evaluation set (finance_v1, 80 cases: 40 dev and 40 frozen test) and a sealed second set (finance_v2, 40 cases). The dev half was used to tune retrieval; the test halves were each run once per system. "Served gold" is the share of gold passages that reached the reader (anchor passages only; "merged" also counts neighbours merged into them). "Terms" is the share of required terms found in the answer. The scorer is lexical, not an LLM judge. Raw score files are kept outside the repository.

finance_v1 frozen test half (40 cases, run once). The new path is the 2026-10-21 retrieval with the FT-r1 reader; "live legacy" is the app before the finance path:

| Measure | Live legacy | New path | Target |
| --- | --- | --- | --- |
| Served gold | 51% | 75% (merged 78%) | at least 75%: met |
| Terms in the answer | 7% | 15% | at least 25%: not met |
| Claim coverage | 4% | 12% | none set |
| Unanswerable questions refused | 5/5 | 5/5 | keep: met |
| Personal questions with scope reply | 0/3 | 3/3 | 3/3: met |
| Mixed questions leaking personal content | 0/2 | 0/2 | 0/2: met |
| Seconds per answer | 7.8 | 11.6 | none set |

finance_v2 sealed test (40 cases, each system run once). Staging is the 2026-10-21 retrieval from cloud-serve `d2f506e` with default `FINANCE_*` settings; live is `fa47f6e`:

| Measure | Live | Staging |
| --- | --- | --- |
| Served gold | 75% | 76% (merged 87%) |
| Terms in the answer | 13% | 14% |
| Claim coverage | 10% | 10% |
| Unanswerable questions refused | 3/3 | 3/3 |
| Personal questions with scope reply | 1/3 | 1/3 |
| Mixed questions leaking personal content | 1/2 | 1/2 |
| Errors | 0 | 0 |

A promotion rule was written before the run: staging replaces live only if no measure is lower and mixed leaks are 0. Staging met every condition except the leak count, which live fails too, so **live stays as it is**. The two systems are within two points of each other on every measure. Gold passages that were served but not cited stay high (78% staging, 80% live): the reader uses few of the passages it is given.

finance_v1 dev half (40 cases, tuning set), final staging run `staging_v8`: served gold 82% (live legacy 40%), terms 16% (live legacy 7%), claim coverage 12% (live legacy 3%), unanswerable refused 4/4, scope reply 3/3, mixed leaks 0/2. These numbers are optimistic because retrieval was tuned on this half.

### FT-r4 (Gemma 4 12B finance fine-tune): open, not trained

The reader (FT-r1) is the bottleneck: answers are short and most served gold is not cited. FT-r4 was planned as a Gemma 4 12B fine-tune on finance data. The training data (623 rows: 557 train, 66 validation, from 726 teacher jobs) and the Colab notebook (`colab/train_gemma4_12b_r4.ipynb`, built by `colab/build_train_gemma4_r4.py`) are ready, and a smoke run trained and exported a GGUF. The full run was **not trained**: the Colab compute units ran out (idle GPU sessions and one stalled run used them), and a full run needs about 140 steps, roughly 3.3 hours on a G4 GPU. The project was closed before more compute was bought. The model, its evaluation and the serving switch are therefore absent; the live app still uses FT-r1.

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

The GGUF is produced by the LoRA notebook `colab/train_7b_lora_r1.ipynb` (commit 7c522f5, data/ft_v3), is not downloadable, and is git-ignored; place it at `models/ft/7b_r1/buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf`.

Providers: `llama` (default) and `remote`. If `llama` is selected but the model file is missing or `llama_cpp` cannot be imported, the factory logs one warning and the backend runs without an answer engine: search works, `/ask` returns the "LLM unavailable" message.

Latency: with Metal (`LLM_GPU_LAYERS=-1`) an answer takes a few seconds. On a CPU-only server (`LLM_GPU_LAYERS=0`) expect roughly 10 to 40 seconds per answer depending on cores. To keep the prompt inside what a 1.5B model handles, the prompt uses only the first `LLM_CONTEXT_PASSAGES` expanded passages (5 by default, 16 for 7B GGUF files), each cut around its anchor chunk to `LLM_PASSAGE_MAX_CHARS`, and trailing passages are dropped until the prompt fits `LLM_N_CTX` minus the answer budget (4 chars per token estimate). Citation numbers always match the passages shown.

Query expansion is off by default (`EXPANSION_MODE=off`): a 1.5B model proposes unreliable keywords. Set `auto` or `always` only if you accept that.

## Fine-tuning (LoRA)

The teacher, Qwen2.5-7B running in MLX 4-bit, generated cited answers from the corpus. Leakage checks exclude held-out evidence and its neighbors, reject questions that overlap the frozen set, and check that training and validation passages remain separate. Qwen2.5-1.5B was then fine-tuned with LoRA through `mlx-lm`.

Round 1 used 193 single-sentence training examples and 13 validation examples, stored in `data/ft/`. Round 2 used answers of 2–4 cited sentences, including comparison, multi-part and follow-up questions; `data/ft_v2/` contains 560 training examples and 70 validation examples. These split counts come from each directory's `manifest.json`. Training records for round 2 are in `data/ft_v2/round2_metrics/`; the held-out table above uses the final comparison, not the earlier metrics snapshot.

Training data is generated locally with the Python generators in `scripts/ft/` (`round3.py`, `round4.py`, `round5.py`, `build_ft_v6.py`), which need the local teacher weights and are not part of the backend quick start. LoRA training and the GGUF export run in the Colab/Kaggle notebooks under `colab/`.

The fine-tuned GGUF files are not published: they are git-ignored under `models/ft/`. To serve a locally generated round-2 model:

```bash
LLM_MODEL_PATH=models/ft/7b_r1/buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf uvicorn src.services.backend_app:app --host 0.0.0.0 --port 8000
```

## Configuration

Everything is set through environment variables, read in `config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `DEFAULT_LLM_PROVIDER` | `llama` | `llama` (embedded GGUF model) or `remote` |
| `LLM_MODEL_PATH` | `models/ft/7b_r1/buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf` | GGUF file, relative to the repo root |
| `LLM_N_CTX` | `16384` | Context window |
| `LLM_N_THREADS` | `0` | CPU threads (0 = auto) |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature (seed is fixed) |
| `LLM_GPU_LAYERS` | `-1` | `-1` all layers on Metal/GPU, `0` CPU only |
| `LLM_CONTEXT_PASSAGES` | `5` (16 for 7B GGUF files) | Passages placed in the prompt for the llama provider |
| `LLM_PASSAGE_MAX_CHARS` | `1800` | Per-passage character cap for the llama provider |
| `VECTOR_BACKEND` | `faiss` | Vector store (`faiss`) |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model id |
| `EMBEDDING_DEVICE` | auto | `cuda` when available, otherwise `cpu` |
| `RERANK_CANDIDATES` | `30` | Max candidates sent to the cross-encoder |
| `EXPANSION_MODE` | `off` | `auto`, `always`, or `off` for the LLM query-expansion call |
| `EVIDENCE_GATE_SOFT` | on | Default on; `0` restores the strict word-overlap gate. Measured on heldout_v6 (95 blind questions): 72 vs 70 strict claims, correct refusals 16/16 both |

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
