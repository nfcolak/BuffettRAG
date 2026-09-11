"""Rebuild paragraph-aware corpus artifacts without overwriting v2.

Produces a new JSONL corpus, a deterministic v2->v3 provenance mapping, and an
index manifest. It intentionally does not contact vector databases or download
models; vector indexing is a separately explicit deployment step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ingestion.legacy_utils import get_all_letters_sorted
from src.ingestion.paragraph_index import build_paragraph_records
from src.vector_store import load_chunks_as_docs


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _map_old_to_new(old_docs, new_records):
    by_year = {}
    for row in new_records:
        by_year.setdefault(row["year"], []).append(row)
    mapping = []
    for old in old_docs:
        candidates = by_year.get(old.metadata.get("year"), [])
        old_tokens = _tokens(old.text)
        scored = []
        for row in candidates:
            new_tokens = _tokens(row["text"])
            score = len(old_tokens & new_tokens) / max(1, len(old_tokens | new_tokens))
            scored.append((score, row["id"]))
        scored.sort(key=lambda item: (-item[0], item[1]))
        mapping.append({"old_id": old.id, "new_ids": [hid for _, hid in scored[:3]],
                        "best_token_jaccard": round(scored[0][0], 6) if scored else 0.0})
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/chunks_v3_paragraph.jsonl")
    parser.add_argument("--mapping", type=Path, default=ROOT / "data/processed/chunk_id_map_v2_to_v3.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/chunks_v3_paragraph_manifest.json")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--min-chunk-chars", type=int, default=160)
    args = parser.parse_args()

    records = []
    for year, path, _source_type in get_all_letters_sorted():
        records.extend(build_paragraph_records(path, year=year, chunk_size=args.chunk_size,
                       overlap=args.overlap, min_chunk_chars=args.min_chunk_chars))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    old_docs = load_chunks_as_docs(ROOT / "data/processed/chunks_v2.jsonl")
    mapping = _map_old_to_new(old_docs, records)
    with args.mapping.open("w", encoding="utf-8") as handle:
        for row in mapping:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {"schema_version": 1, "corpus": "paragraph_v3", "source_files": len(get_all_letters_sorted()),
                "chunks": len(records), "chunk_config": {"size": args.chunk_size, "overlap": args.overlap,
                "min_chunk_chars": args.min_chunk_chars}, "corpus_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "mapping_sha256": hashlib.sha256(args.mapping.read_bytes()).hexdigest(),
                "vector_index": {"status": "not_built", "reason": "requires explicit local vector backend/model run"}}
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
