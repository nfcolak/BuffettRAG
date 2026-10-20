#!/usr/bin/env python3
"""Build the annotation-summary index used by the FINANCE_SUMMARY_SOURCE candidate source (finance-recall-75).

For every financial passage of data/knowledge/annotations.jsonl that has a summary it embeds the one-sentence summary
with the main index's model (bge-base) and writes a FAISS index plus the summaries (id = passage id, metadata = letter
year) to data/knowledge/summary_index/, with index_manifest.json. The BM25 side is not stored: like the main index, it
is rebuilt from the stored summaries at load time (src/retrieval/summary_source.py); the manifest records its size.

Usage:
    python scripts/index/build_summary_index.py
    python scripts/index/build_summary_index.py --out data/knowledge/summary_index --overwrite --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHUNKS_V3_FILE, EMBEDDING_DEVICE  # noqa: E402
from src.knowledge import default_dir  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.summary_source import INDEX_DIRNAME, summary_docs  # noqa: E402
from src.storage.faiss_store import FaissStore  # noqa: E402
from src.storage.index_manifest import MANIFEST_FILE, write_index_manifest  # noqa: E402


def read_annotations(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def chunk_years(path: Path) -> Dict[str, int]:
    """Passage id -> letter year of the main corpus (one JSONL row per passage)."""
    years: Dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("year") is not None:
                years[str(row["id"])] = int(row["year"])
    return years


def build(annotations: Path, chunks: Optional[Path], out_dir: Path, embedder: Any, overwrite: bool = False
          ) -> Dict[str, Any]:
    """Embed the financial summaries and write the index and its manifest; returns the manifest."""
    rows = read_annotations(annotations)
    years = chunk_years(chunks) if chunks is not None else None
    docs = summary_docs(rows, years)
    if not docs:
        raise SystemExit(f"{annotations} has no financial passage with a summary")
    if years is not None:
        missing = [d.id for d in docs if d.id not in years]
        if missing:
            raise SystemExit(f"{len(missing)} annotated passages are not in the chunk corpus {chunks}: {missing[:5]}")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise SystemExit(f"{out_dir} is not empty; pass --overwrite to rebuild it")
        for path in out_dir.iterdir():
            if path.is_file():
                path.unlink()
    embeddings = embedder.embed_documents([d.text for d in docs])  # documents carry no query instruction
    store = FaissStore(persist_dir=out_dir, dim=embedder.dimension)
    store.add(docs, embeddings)
    manifest = write_index_manifest(out_dir, corpus=annotations, docs=docs, backend="faiss",
                                    model_name=embedder.model_name, dimension=embedder.dimension,
                                    artifact_names=(store.INDEX_FILE, store.META_JSON_FILE))
    bm25 = BM25Retriever(docs)
    lengths = [len(tokens) for tokens in bm25._tokens]
    manifest.update({
        "kind": "annotation_summary_index",
        "scope": "financial",
        "bm25": {"source": "meta.json", "tokenizer": "src.retrieval.bm25._search_tokens", "documents": len(docs),
                 "vocabulary": len({t for tokens in bm25._tokens for t in tokens}),
                 "mean_tokens": round(sum(lengths) / len(lengths), 2)},
    })
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotations", type=Path, default=None, help="annotations.jsonl (default: the knowledge dir's)")
    ap.add_argument("--chunks", type=Path, default=CHUNKS_V3_FILE,
                    help="chunk corpus the passage ids and letter years come from (default: the V3 corpus)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <knowledge dir>/summary_index)")
    ap.add_argument("--embedder", choices=["small", "base"], default="base")
    ap.add_argument("--device", default=EMBEDDING_DEVICE)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing, non-empty output directory")
    args = ap.parse_args(argv)

    annotations = args.annotations or default_dir() / "annotations.jsonl"
    out_dir = args.out or default_dir() / INDEX_DIRNAME
    if not annotations.is_file():
        ap.error(f"annotations file does not exist: {annotations}")
    if args.chunks is not None and not args.chunks.is_file():
        ap.error(f"chunks file does not exist: {args.chunks}")

    from src.storage.embeddings import get_embedder
    embedder = get_embedder(args.embedder, device=args.device)
    print(f"Embedding the summaries of {annotations} with {embedder.model_name} (dim={embedder.dimension}) "
          f"on {args.device}")
    manifest = build(annotations, args.chunks, out_dir, embedder, overwrite=args.overwrite)
    print(f"Summary index built: {manifest['document_count']} summaries in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
