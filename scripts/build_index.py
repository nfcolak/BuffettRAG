#!/usr/bin/env python3
"""Build the vector index from a chunks JSONL.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --chunks processed_data/chunks_v2.jsonl --backend faiss
    python scripts/build_index.py --embedder base --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHUNKS_FILE,
    CHUNKS_V2_FILE,
    EMBEDDING_DEVICE,
    VECTOR_BACKEND,
)
from src.embeddings import get_embedder
from src.vector_store import get_vector_store, load_chunks_as_docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=None,
                    help="Chunks JSONL (defaults to v2 if it exists, else v1).")
    ap.add_argument("--backend", choices=["chroma", "faiss"], default=VECTOR_BACKEND)
    ap.add_argument("--embedder", choices=["small", "base"], default="small")
    ap.add_argument("--device", default=EMBEDDING_DEVICE)
    args = ap.parse_args()

    chunks_file = args.chunks
    if chunks_file is None:
        chunks_file = CHUNKS_V2_FILE if CHUNKS_V2_FILE.exists() else CHUNKS_FILE
    if not chunks_file.exists():
        ap.error(f"Chunks file does not exist: {chunks_file}")

    print(f"Reading chunks from {chunks_file}")
    docs = load_chunks_as_docs(chunks_file)
    print(f"  loaded {len(docs)} chunks")

    embedder = get_embedder(args.embedder, device=args.device)
    print(f"Embedding with {embedder.model_name} (dim={embedder.dimension}) on {args.device}")
    embeddings = embedder.embed_documents([d.text for d in docs])

    store = get_vector_store(backend=args.backend, dim=embedder.dimension)
    store.add(docs, embeddings)
    print(f"Index built. Backend={args.backend}, total vectors={len(store)}")


if __name__ == "__main__":
    main()
