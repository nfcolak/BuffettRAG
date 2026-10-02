# BuffettRAG React Frontend

React/Vite frontend for the BuffettRAG FastAPI backend.

## Run locally

```bash
npm install
npm run dev
```

The app defaults to `http://localhost:8000` for the backend.
The LLM provider is configured on the backend. For local bring-your-own-key,
set `ALLOW_LLM_REQUEST_OVERRIDES=1` on the backend and build with
`VITE_ALLOW_LLM_OVERRIDES=1` to enable the provider settings dialog.

To point at another backend:

```bash
VITE_BACKEND_URL=http://localhost:8000 npm run dev
```

For lightweight local backend testing without Postgres:

```bash
VECTOR_BACKEND=chroma \
  python3 -m uvicorn src.services.backend_app:app --host 127.0.0.1 --port 8000
```

## Backend endpoints used

- `GET /ready`
- `GET /health`
- `POST /search`
- `POST /ask`
- `POST /ask/stream`

