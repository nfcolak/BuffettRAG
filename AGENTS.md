# AGENTS.md

Guidance for Codex and other coding agents working in this repository.

## Project Overview

**BuffettRAG** is a Streamlit + FastAPI retrieval-augmented generation app for asking questions across Warren Buffett's Berkshire Hathaway shareholder letters from 1977 through 2024.

The deployed shape is a 3-tier Nuvolos stack:

- **Frontend:** `frontend/`, a React/Vite client.
- **Legacy frontend:** `app.py`, a Streamlit fallback client.
- **Backend:** `services/backend_app.py`, a FastAPI retrieval service.
- **LLM service:** `services/llm_service.py`, a GPU FastAPI generation service.
- **Storage/retrieval:** PostgreSQL + pgvector in production, with Chroma/FAISS support for local workflows.

The frontend does not load embedding, reranker, or LLM models. The React app calls the backend over HTTP using `VITE_BACKEND_URL`.

## Frontend Structure

The primary frontend is now React/Vite:

- `frontend/src/main.jsx`: React app, query orchestration, chat state, backend calls, and rendered components.
- `frontend/src/styles.css`: React frontend visual system.
- `frontend/public/assets/`: assets served by Vite.
- `frontend/package.json`: frontend scripts and dependencies.

The legacy Streamlit frontend remains available:

- `app.py`: page setup, topbar, left control rail, three-column dashboard layout, and query orchestration.
- `src/frontend/settings.py`: frontend paths, constants, example queries, and environment-derived `BACKEND_URL`.
- `src/frontend/assets.py`: cached CSS/image/year metadata loading and decorative background markup.
- `src/frontend/state.py`: message creation, session-state initialization, selected answer handling, and local history persistence.
- `src/frontend/api.py`: backend payload creation and `/ask` or `/search` HTTP calls.
- `src/frontend/rendering.py`: chat message, source panel, source cards, metadata pills, and autoscroll rendering.
- `assets/style.css`: all custom visual styling, including Streamlit selector overrides.
- `assets/header_logo.png`: main logo rendered at the top of the app.
- `assets/chatbot_avatar.png`: assistant avatar used in chat messages.
- `assets/user_avatar.png`: user avatar used in chat messages.
- `.streamlit/config.toml`: base Streamlit theme values.

Prefer new UI work in `frontend/`. Only modify the Streamlit frontend when maintaining the fallback path.

## Frontend Flow

`app.py` now reads as a high-level Streamlit shell:

1. **Constants and cached I/O**
   - Stored in `src/frontend/settings.py` and `src/frontend/assets.py`.
   - `assets.load_available_years()` reads `processed_data/metadata_v2.json`, then `metadata.json`, then falls back to `1977..2024`.

2. **Page setup**
   - `st.set_page_config(...)` sets the title, icon, and wide layout.
   - `render_page_chrome()` injects CSS, animated background markup, logo, and caption.
   - Header logo and scrolling caption are rendered before the dashboard layout.

3. **Left rail controls**
   - `render_left_rail()` returns a `QueryOptions` dataclass.
   - Retrieval strategy: `hybrid`, `vector`, `metadata`, `naive`.
   - `top_k`, rerank toggle, LLM answer toggle.
   - Optional year/decade filters, converted into backend `where` filters.
   - Clear chat delegates to `state.clear_chat()`.

4. **Message and source helpers**
   - Message creation and history compatibility live in `state.py`.
   - `render_message()` uses `st.chat_message(...)` and adds a per-answer source button.
   - `render_source_panel()` shows the currently selected assistant answer's passages.
   - `render_sources()` renders backend hits as cards with year, source file, score, topics, preview, and full passage details.

5. **Backend calls**
   - `api.ask_backend()` selects `/ask` when LLM generation is enabled and `/search` when retrieval-only mode is selected.
   - `api.backend_post()` wraps `requests.post(...)` and returns `None` on network/API failure.
   - Failed backend calls show `Backend did not respond. Is the server running?`

6. **Session state and persistence**
   - Messages are loaded from `.chat_history.json`.
   - The app persists up to `_MAX_PERSISTED_MESSAGES = 200`.
   - `selected_message_id` controls which answer's sources are visible.
   - `pending_query` lets example-question buttons submit a query on the next rerun.
   - `_max_visible` progressively reveals older messages.

7. **Main layout**
   - Three dashboard columns: examples, chat, sources.
   - Current width ratio is `[0.20, 0.50, 0.30]`.
   - The chat column is the primary working surface.
   - The source column is a persistent inspector for citations and retrieved passages.

8. **Query handling**
   - User input can come from `st.chat_input` or an example button.
   - The user message is appended before calling the backend.
   - The assistant message stores answer text, hits, citations, and retrieval metadata.
   - The latest assistant message becomes the selected source message.

## Frontend State Contracts

User messages:

```python
{
    "id": "user_<uuid>",
    "role": "user",
    "content": "...",
    "created_at": "HH:MM",
}
```

Assistant messages:

```python
{
    "id": "assistant_<uuid>",
    "role": "assistant",
    "content": "...",
    "sources": [...],
    "meta": {
        "strategy": "...",
        "reranked": True,
        "used_filter": {...},
    },
    "citations": [...],
    "created_at": "HH:MM",
}
```

Backend hit objects are expected to include:

```python
{
    "text": "...",
    "year": 2008,
    "source_file": "buffet_2008.pdf",
    "topics": "derivatives,risk,financial-crisis",
    "score": 0.73,
}
```

Keep these contracts backward-compatible when possible. If changing them, update `_normalize_messages()`, `_render_sources()`, and the backend response model together.

## Frontend Styling Rules

The frontend heavily customizes Streamlit with CSS selectors in `assets/style.css`.

Important styling anchors:

- `.rocket-field`, `.rocket`: animated background layer.
- `.brand-logo`, `.app-brand-caption`, `.terminal-marquee`: top branding.
- `.st-key-dashboard_content`: outer dashboard layout.
- `.st-key-chat_shell`: fixed-height chat panel.
- `.st-key-chat_messages`: scrollable message area.
- `.st-key-source_panel`: source inspector panel.
- `.st-key-source_scroll`: scrollable source list.
- `.source-card`, `.source-year`, `.source-file`, `.source-score`, `.source-topic`: source result cards.
- `.message-status`, `.message-status-pill`: metadata row under messages.

When editing layout, check both desktop and mobile widths. The dashboard height is controlled by `--workspace-height`, and mobile overrides live at the bottom of `assets/style.css`. Small CSS changes can affect scroll behavior, chat input placement, or source panel overflow.

Prefer adding stable Streamlit `key=` values to containers and styling the resulting `.st-key-*` classes instead of relying only on deep `data-testid` selectors.

Avoid moving user-facing copy into CSS pseudo-elements unless it is purely decorative. User-facing content should remain in `app.py`.

## Frontend UX Guidelines

- Preserve the thin-client model: frontend should call backend APIs, not import or build the RAG pipeline directly.
- Keep the chat, examples, and sources visible as a research workspace rather than turning the app into a marketing page.
- Source visibility is a core feature. Any answer-rendering change should keep a clear path to inspect retrieved passages.
- Keep failures calm and actionable. Backend/network failures should not crash the app.
- Do not expose health/status noise in the main UI unless specifically requested.
- Be careful with `unsafe_allow_html=True`; only use it for controlled markup generated by this app, and escape dynamic user/backend text with `html.escape(...)`.
- Keep persisted chat history local. `.chat_history.json` is ignored and should not be committed.

## Backend API Used By Frontend

`POST {BACKEND_URL}/ask`

Payload:

```json
{
  "query": "How did Buffett react to the 2008 financial crisis?",
  "strategy": "hybrid",
  "top_k": 8,
  "rerank": true,
  "where": {"year": 2008},
  "auto_year_filter": true
}
```

Expected response includes:

- `answer`: generated answer text.
- `hits`: retrieved passages.
- `citations`: parsed citation metadata.
- `strategy`: retrieval strategy used.
- `reranked`: whether reranking was applied.
- `used_filter`: backend-side metadata filter.

`POST {BACKEND_URL}/search` uses the same payload and returns retrieval results without generated answer text.

## Local Development

Install frontend-only dependencies:

```bash
pip install -r services/requirements_frontend.txt
```

Run the frontend:

```bash
export BACKEND_URL=http://localhost:8000
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Run the backend locally:

```bash
export VECTOR_BACKEND=chroma
uvicorn services.backend_app:app --host 0.0.0.0 --port 8000 --reload
```

For full local RAG setup, build the index first:

```bash
python scripts/build_index.py
```

Smoke tests:

```bash
python -m tests.test_e2e
```

With pytest:

```bash
pytest tests/test_e2e.py -q
```

## Project Commands

Ingest and chunk letters:

```bash
python -m src.ingestion.pipeline_v2
```

Build/populate vector index:

```bash
python scripts/build_index.py
```

Validate chunks:

```bash
python validate_chunks.py
python validate_chunks.py --sample 5
python validate_chunks.py --year 1977
```

Run retrieval evaluation:

```bash
python -m src.evaluation.run_eval
```

## Core Backend Components

- `services/backend_app.py`: FastAPI service for retrieval and optional LLM forwarding.
- `services/llm_service.py`: GPU LLM service with `/generate`.
- `src/pipeline.py`: end-to-end local RAG pipeline composition.
- `src/vector_store.py`: pgvector, Chroma, and FAISS abstraction.
- `src/embeddings.py`: BGE embedder wrapper.
- `src/retrieval/retriever.py`: retrieval strategies and RRF fusion.
- `src/retrieval/bm25.py`: sparse retrieval.
- `src/retrieval/reranker.py`: cross-encoder reranking.
- `src/generation/prompt.py`: citation-aware prompt formatting.
- `src/generation/llm.py`: local model loading.
- `src/ingestion/pipeline_v2.py`: preferred ingestion pipeline.

## Configuration

`config.py` is the central configuration file. Important values include:

- `VECTOR_BACKEND`: `pgvector`, `chroma`, or `faiss`.
- `EMBEDDING_MODEL_PRIMARY`: default BGE embedding model.
- `EMBEDDING_DEVICE`: `cpu` or `cuda`.
- `DEFAULT_TOP_K`, `RETRIEVAL_FETCH_K`, `HYBRID_ALPHA`, `RRF_K`.
- `LLM_MODEL_PRIMARY`, `LLM_MAX_NEW_TOKENS`, generation temperature.
- `BACKEND_URL`, `LLM_SERVICE_URL`.

Frontend runtime URL is read directly from the `BACKEND_URL` environment variable in `app.py`.

## Data And Metadata

Raw letters live in `raw_data/`.

Processed chunk metadata lives in `processed_data/`, especially:

- `processed_data/chunks_v2.jsonl`
- `processed_data/metadata_v2.json`
- `processed_data/metadata.json`

Chunk metadata shape:

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

Metadata filters passed from the frontend use backend-compatible `where` dictionaries:

```python
{"year": 2008}
{"decade": 2000}
{"year": {"$gte": 2000, "$lte": 2010}}
{"topics": "inflation"}
```

## Repository Hygiene

- Do not commit local chat history, shell files, crash dumps, vector indexes, or virtual environments.
- `.gitignore` already excludes `.chat_history.json`, `core.*`, `.nuvolos/`, `.claude/`, `.vscode/`, caches, venvs, and generated local indexes.
- Keep large generated model/index artifacts out of Git unless the user explicitly requests otherwise.
- Be cautious with raw data files, but the current shareholder letters are part of the project dataset.

## Nuvolos Notes

Nuvolos deployment is documented in `NUVOLOS_DEPLOYMENT.md`.

Expected app split:

1. Database app with pgvector.
2. Editor app for ingestion.
3. Backend app running `uvicorn services.backend_app:app`.
4. Trainer T4 app running `uvicorn services.llm_service:app`.
5. Frontend app running `streamlit run app.py`.

All apps share workspace storage at `/space_mounts/pars` for large files.
