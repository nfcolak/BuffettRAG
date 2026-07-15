# BuffettRAG React Frontend

React/Vite frontend for the BuffettRAG FastAPI backend.

## Run locally

```bash
npm install
npm run dev
```

The app defaults to `http://localhost:8000` for the backend.
Use the Settings button in the top bar to choose OpenAI, Anthropic, or
OpenRouter and enter the provider API key for answer generation.

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

- `GET /health`
- `POST /search`
- `POST /ask`

