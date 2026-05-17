# Nuvolos Deployment Guide

This document explains how to deploy BuffettRAG across the four Nuvolos apps
provided in the workspace:

```
┌──────────┐  HTTP  ┌──────────┐  SQL   ┌──────────┐
│ Frontend │ ─────► │ Backend  │ ─────► │ Database │
│ Streamlit│        │ FastAPI  │        │ pgvector │
└──────────┘        └────┬─────┘        └──────────┘
                         │ HTTP
                         ▼
                    ┌──────────┐
                    │ Trainer  │
                    │ Qwen LLM │
                    └──────────┘
```

| App | Purpose | Resources |
|---|---|---|
| Database | Postgres + pgvector — chunk storage + ANN search | 4 CU |
| Backend | FastAPI service (embedding, retrieval, reranking, citation parsing) | 4 CU, CPU |
| Trainer T4 | LLM service hosting Qwen2.5-7B | Tesla T4 (16 GB VRAM) |
| Frontend | Streamlit UI — talks to Backend over HTTP | 4 CU |
| Editor | Code editing, ad-hoc scripts (not part of runtime path) | 1 CU |

Frontend, Backend and Database share a network namespace and can reach each
other by fixed hostnames shown in the Nuvolos UI under
**Applications → … → CONFIGURE**. Trainer is isolated, so we expose it as a
public port and the Backend reaches it via that URL.

---

## 1. Database app (pgvector)

Start the **Database** app from the dashboard.

In the Nuvolos UI **Configure** panel, note down:
- Hostname (e.g. `database-01.nuvolos.cloud` or an internal alias)
- Port (typically `5432`)
- User / password / dbname (Nuvolos provides these)

The pgvector extension is preinstalled on the Database app. The Backend
creates the schema and the IVFFlat index automatically on first startup, so
nothing else to do here.

---

## 2. Editor app — one-time data prep

Start the **Editor** app and clone or upload the repo to the workspace.

```bash
cd ~/BuffettRAG
pip install -r services/requirements_backend.txt
python -m src.ingestion.pipeline_v2     # produces processed_data/chunks_v2.jsonl
```

The chunks file ends up on the shared workspace storage so all apps can read
it. (The Editor has 1 CU but ingestion is CPU-light and finishes in
~10 seconds for 47 letters.)

---

## 3. Backend app

Start the **Backend** app. In its terminal:

```bash
cd ~/BuffettRAG
pip install -r services/requirements_backend.txt

# Database connection -- replace with values from the Database app's Configure panel.
export PGHOST=<database-hostname>
export PGPORT=5432
export PGUSER=<user>
export PGPASSWORD=<password>
export PGDATABASE=<dbname>

# URL of the LLM service running on the Trainer (set after step 4).
export LLM_SERVICE_URL=http://<trainer-public-host>:8001

# Start the API.
uvicorn services.backend_app:app --host 0.0.0.0 --port 8000
```

On first startup the Backend will:
1. Read `processed_data/chunks_v2.jsonl`
2. Embed all 5,911 chunks with bge-small-en-v1.5 (CPU, ~10–15 minutes)
3. Insert them into pgvector and create the IVFFlat index
4. Load the bge-reranker-base model in memory

Subsequent restarts are fast because the chunks are already in pgvector.

Verify:
```bash
curl http://localhost:8000/health
# {"ok": true, "n_docs": 5911, ...}
```

---

## 4. Trainer T4 app — LLM service

Start the **Trainer T4** app. In its terminal:

```bash
cd ~/BuffettRAG
pip install -r services/requirements_llm.txt

# Run the LLM service.
uvicorn services.llm_service:app --host 0.0.0.0 --port 8001
```

On startup it loads Qwen2.5-7B-Instruct (~14 GB of weights; first run
downloads from HuggingFace — allow ~5–10 minutes).

Then in the Nuvolos UI, expose port 8001 of the Trainer as a public port
and copy the resulting URL. Update the Backend's `LLM_SERVICE_URL` env var
to that URL and restart the Backend.

Verify:
```bash
curl http://<trainer-public-host>:8001/health
# {"ok": true, "model": "Qwen/Qwen2.5-7B-Instruct"}
```

---

## 5. Frontend app

Start the **Frontend** app. In its terminal:

```bash
cd ~/BuffettRAG
pip install -r services/requirements_frontend.txt

# Backend URL -- internal hostname provided by Nuvolos.
export BACKEND_URL=http://<backend-hostname>:8000

streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Expose port 8501 publicly. The UI is now reachable.

---

## Smoke test (end-to-end)

From any terminal that can reach the Backend (e.g. the Editor):

```bash
curl -X POST http://<backend-host>:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What did Buffett say about derivatives?",
    "strategy": "hybrid",
    "top_k": 5,
    "rerank": true
  }'
```

You should see a JSON response with an answer, citations, and the
retrieved passages.

---

## Local development (no Nuvolos)

Everything also runs locally on a single machine -- defaults point at
`localhost`. Either:

- Use ChromaDB instead of pgvector (`export VECTOR_BACKEND=chroma`) and run
  `streamlit run app.py` after `python scripts/build_index.py`, OR
- Run all three services locally (`backend_app` on 8000, `llm_service` on
  8001 if you have a GPU, `streamlit run app.py` on 8501).
