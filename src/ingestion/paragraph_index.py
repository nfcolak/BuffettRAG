"""Deterministic paragraph-aware records with source provenance."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List

from src.ingestion.chunker import split_text
from src.ingestion.pdf_extractor import extract_text_from_pdf, load_text_file
from src.ingestion.topic_tagger import tag_topics


def _load_source(path: Path) -> tuple[str, str]:
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path), "pymupdf_or_pdfplumber"
    return load_text_file(path), "text"


def build_paragraph_records(source: Path, *, year: int, chunk_size: int = 800, overlap: int = 120,
                            min_chunk_chars: int = 160) -> List[Dict]:
    text, extraction_method = _load_source(source)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks = split_text("\n\n".join(paragraphs), chunk_size, overlap, min_chunk_chars)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    records: List[Dict] = []
    for idx, chunk in enumerate(chunks):
        paragraph_count = max(1, len([p for p in re.split(r"\n\s*\n+", chunk) if p.strip()]))
        records.append({"id": f"{year}_p{idx:04d}", "text": chunk, "year": year,
                        "decade": (year // 10) * 10, "source_file": source.name,
                        "chunk_index": idx, "topics": ",".join(tag_topics(chunk)),
                        "provenance": {"source_sha256": source_sha256, "extraction_method": extraction_method,
                                       "paragraph_count": paragraph_count, "chunker": "paragraph_v3"}})
    for idx, row in enumerate(records):
        row["total_chunks"] = len(records)
        row["previous_chunk_id"] = records[idx - 1]["id"] if idx else None
        row["next_chunk_id"] = records[idx + 1]["id"] if idx + 1 < len(records) else None
    return records
