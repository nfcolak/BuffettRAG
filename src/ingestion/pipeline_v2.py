"""Improved ingestion pipeline.

Differences vs. the original `ingestion_pipeline.py`:
    - PDF extraction uses PyMuPDF (with pdfplumber fallback) and cleans
      headers/footers + de-hyphenation
    - Chunking uses LangChain's RecursiveCharacterTextSplitter so paragraph
      and sentence boundaries are respected
    - Each chunk is tagged with topic labels
    - Output format is unchanged JSONL so it stays compatible with anything
      downstream (Chroma builder, eval scripts, etc.)

Run:
    python -m src.ingestion.pipeline_v2
    python -m src.ingestion.pipeline_v2 --chunk-size 600 --overlap 80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

# Allow running both as a module and as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHUNK_SIZE_V2,
    CHUNK_OVERLAP_V2,
    CHUNKS_V2_FILE,
    METADATA_V2_FILE,
    RAW_DATA_DIR,
    START_YEAR,
    END_YEAR,
)
from src.ingestion.chunker import split_text
from src.ingestion.pdf_extractor import extract_text_from_pdf, load_text_file
from src.ingestion.topic_tagger import tag_topics
from utils import extract_year_from_filename, get_all_letters_sorted


def build_chunk_record(
    chunk_text: str,
    year: int,
    source_file: str,
    chunk_index: int,
    total_chunks: int,
) -> Dict:
    decade = (year // 10) * 10
    topics = tag_topics(chunk_text)
    return {
        "id": f"{year}_{chunk_index}",
        "text": chunk_text,
        "year": year,
        "decade": decade,
        "source_file": source_file,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "chunk_id": f"{chunk_index + 1}/{total_chunks}",
        # Comma-joined string is friendlier to Chroma/FAISS metadata than a list
        # (Chroma rejects list-valued metadata in some versions).
        "topics": ",".join(topics),
        "n_topics": len(topics),
    }


def process_letter(
    file_path: Path,
    year: int,
    file_type: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict]:
    if file_type == "pdf":
        text = extract_text_from_pdf(file_path)
    else:
        text = load_text_file(file_path)

    if not text:
        print(f"Warning: no text extracted from {file_path.name}")
        return []

    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        build_chunk_record(
            chunk_text=c,
            year=year,
            source_file=file_path.name,
            chunk_index=i,
            total_chunks=len(chunks),
        )
        for i, c in enumerate(chunks)
    ]


def run(chunk_size: int, chunk_overlap: int, output_path: Path, meta_path: Path) -> Dict:
    print("=" * 60)
    print("BUFFETT RAG -- IMPROVED INGESTION (v2)")
    print("=" * 60)
    print(f"Years      : {START_YEAR}-{END_YEAR}")
    print(f"Chunk size : {chunk_size} chars  | overlap: {chunk_overlap}")
    print(f"Output     : {output_path}")
    print()

    letters = get_all_letters_sorted()
    if not letters:
        print("No letter files found.")
        sys.exit(1)

    all_chunks: List[Dict] = []
    year_stats: Dict[int, Dict] = {}

    for year, file_path, file_type in tqdm(letters, desc="Letters"):
        try:
            chunks = process_letter(file_path, year, file_type, chunk_size, chunk_overlap)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue
        all_chunks.extend(chunks)
        year_stats[year] = {
            "file": file_path.name,
            "type": file_type,
            "chunks": len(chunks),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    metadata = {
        "total_letters": len(letters),
        "total_chunks": len(all_chunks),
        "year_range": [START_YEAR, END_YEAR],
        "avg_chunks_per_letter": len(all_chunks) / max(len(letters), 1),
        "chunk_config": {"size": chunk_size, "overlap": chunk_overlap},
        "year_statistics": year_stats,
        "version": "v2",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print(f"Wrote {len(all_chunks)} chunks to {output_path}")
    print(f"Wrote metadata    to {meta_path}")
    return metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BuffettRAG improved ingestion (v2)")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_V2)
    p.add_argument("--overlap", type=int, default=CHUNK_OVERLAP_V2)
    p.add_argument("--output", type=Path, default=CHUNKS_V2_FILE)
    p.add_argument("--meta", type=Path, default=METADATA_V2_FILE)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.chunk_size, args.overlap, args.output, args.meta)
