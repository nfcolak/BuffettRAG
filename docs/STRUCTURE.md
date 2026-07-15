# Project Structure

This project now follows a source-first layout with generated data and historical artifacts separated from active code.

## Active Code

- `src/`: Python application code.
- `src/services/`: FastAPI backend entrypoint and service utilities.
- `src/ingestion/`: Data extraction, cleaning, chunking, and topic tagging.
- `src/retrieval/`: Search, filtering, hybrid retrieval, and reranking.
- `src/generation/`: Prompt construction and pluggable LLM provider wrappers.
- `frontend/`: React/Vite application.
- `scripts/`: Command-line utilities and legacy ingestion helpers.

## Runtime Assets and Data

- `data/raw/`: Original shareholder letters.
- `data/processed/`: Generated chunks and metadata.
- `data/indices/`: Vector-store files.
- `data/evaluation/`: Evaluation outputs.

## Documentation and Archive

- `docs/reference/`: Older notes and project references.
- `archive/frontend-prototype/`: Superseded frontend prototype files.

## Root Files

The root is intentionally kept small: `README.md`, `config.py`, dependency files, and project-level tool configuration.
