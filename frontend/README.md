# BuffettRAG React Frontend

React/Vite frontend for the BuffettRAG FastAPI backend.

## Run locally

```bash
npm install
npm run dev
```

The app defaults to `http://localhost:8000` for the backend.

To point at another backend:

```bash
VITE_BACKEND_URL=http://localhost:8000 npm run dev
```

For lightweight local backend testing without Postgres:

```bash
VECTOR_BACKEND=chroma LLM_SERVICE_URL= \
  python3 -m uvicorn services.backend_app:app --host 127.0.0.1 --port 8000
```

## Backend endpoints used

- `GET /health`
- `POST /search`
- `POST /ask`

The Streamlit app at the repository root is now a legacy/fallback frontend.
