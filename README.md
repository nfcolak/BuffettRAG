# BuffettRAG

BuffettRAG is a retrieval-augmented research workspace for Warren Buffett shareholder letters.
It includes a Python retrieval backend with multi-provider answer generation (OpenAI, Anthropic, OpenRouter, or an offline extractive fallback) and a React/Vite frontend.

## Project Layout

```text
.
├── config.py               # Central paths and runtime settings
├── data/
│   ├── raw/                # Source shareholder letters
│   ├── processed/          # Chunk files and metadata
│   ├── indices/            # Vector-store persistence
│   └── evaluation/         # Evaluation reports
├── docs/
│   └── reference/          # Historical project notes
├── frontend/               # React/Vite frontend
├── scripts/                # CLI and maintenance scripts
├── src/
│   ├── evaluation/         # Evaluation pipeline
│   ├── generation/         # Prompting and LLM provider wrappers
│   ├── ingestion/          # Parsing, chunking, and tagging
│   ├── retrieval/          # BM25, vector retrieval, reranking
│   └── services/           # FastAPI backend service
└── tests/                  # Automated tests
```

## Quick Start

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Build or refresh processed chunks:

```bash
python -m src.ingestion.pipeline_v2
```

Run the backend:

```bash
uvicorn src.services.backend_app:app --host 0.0.0.0 --port 8000 --reload
```

Run the React frontend:

```bash
cd frontend
npm install
VITE_BACKEND_URL=http://localhost:8000 npm run dev
```

The frontend Settings button lets you choose OpenAI or OpenRouter per browser.
If you enter a provider API key there, it is sent only with `/ask` requests.
Selecting "remember on this device" stores the key in that browser's local storage.

## Security Configuration

For shared or public deployments, set API keys and restrict browser origins:

```bash
export API_KEYS="replace-with-backend-key"
export BACKEND_API_KEY="replace-with-backend-key"
export CORS_ORIGINS="https://your-frontend.example"
export DEFAULT_LLM_PROVIDER="openai"
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_MODEL="gpt-4.1-mini"
```

To use Anthropic instead:

```bash
export DEFAULT_LLM_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export ANTHROPIC_MODEL="claude-haiku-4-5-20251001"
```

To use OpenRouter instead:

```bash
export DEFAULT_LLM_PROVIDER="openrouter"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export OPENROUTER_MODEL="openrouter/free"
```

Optional hardening knobs:

```bash
export RATE_LIMIT_REQUESTS=60
export RATE_LIMIT_WINDOW_SECONDS=60
export EXPOSE_DEBUG_STATUS=0
# Set only when the backend runs behind a reverse proxy that sets
# X-Forwarded-For; otherwise clients could spoof their rate-limit identity.
export TRUST_PROXY_HEADERS=1
```

Local development can leave keys empty, but production should not.

## Notes

- Active data paths are configured in `config.py`.
- Historical docs and older project notes live in `docs/reference/`.
- Old frontend prototype files live in `archive/frontend-prototype/`.
