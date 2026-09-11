#!/usr/bin/env python3
"""Build the vector index from a chunks JSONL.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --chunks data/processed/chunks_v2.jsonl --backend faiss
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
    CHUNKS_V3_FILE,
    EMBEDDING_DEVICE,
    VECTOR_BACKEND,
)
from src.embeddings import get_embedder
from src.index_manifest import build_index_manifest, write_index_manifest
from src.vector_store import FaissStore, get_vector_store, load_chunks_as_docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=None,
                    help="Chunks JSONL (defaults to V3, then legacy corpus fallback).")
    ap.add_argument("--backend", choices=["chroma", "faiss"], default=VECTOR_BACKEND)
    ap.add_argument("--embedder", choices=["small", "base"], default="small")
    ap.add_argument("--device", default=EMBEDDING_DEVICE)
    ap.add_argument("--persist-dir", type=Path, default=None,
                    help="Dedicated FAISS output directory; required to keep corpus versions isolated.")
    args = ap.parse_args()

    if args.persist_dir is not None and args.backend != "faiss":
        ap.error("--persist-dir is currently supported only for the FAISS backend")

    chunks_file = args.chunks
    if chunks_file is None:
        chunks_file = next(
            (path for path in (CHUNKS_V3_FILE, CHUNKS_V2_FILE, CHUNKS_FILE) if path.exists()),
            CHUNKS_V3_FILE,
        )
    if not chunks_file.exists():
        ap.error(f"Chunks file does not exist: {chunks_file}")

    print(f"Reading chunks from {chunks_file}")
    docs = load_chunks_as_docs(chunks_file)
    print(f"  loaded {len(docs)} chunks")

    embedder = get_embedder(args.embedder, device=args.device)
    print(f"Embedding with {embedder.model_name} (dim={embedder.dimension}) on {args.device}")
    embeddings = embedder.embed_documents([d.text for d in docs])

    if args.backend == "faiss" and args.persist_dir is not None:
        store = FaissStore(persist_dir=args.persist_dir, dim=embedder.dimension)
    else:
        store = get_vector_store(backend=args.backend, dim=embedder.dimension)
    if len(store) != 0:
        ap.error("Refusing to append to a non-empty index; choose a new --persist-dir")
    store.add(docs, embeddings)

    if isinstance(store, FaissStore):
        write_index_manifest(
            Path(store.persist_dir),
            corpus=chunks_file,
            docs=docs,
            backend=args.backend,
            model_name=embedder.model_name,
            dimension=embedder.dimension,
            artifact_names=(store.INDEX_FILE, store.META_JSON_FILE),
        )
    print(f"Index built. Backend={args.backend}, total vectors={len(store)}")


if __name__ == "__main__":
    main()
