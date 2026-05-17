#!/usr/bin/env python3
"""
RAG Ingestion Pipeline for Warren Buffett Shareholder Letters

This script processes all shareholder letters (1977-2024) from both text and PDF formats,
chunks them for RAG, and saves them as JSONL with metadata.
"""

import sys
from pathlib import Path
from tqdm import tqdm
from config import (
    RAW_DATA_DIR, OUTPUT_DIR, CHUNKS_FILE, METADATA_FILE,
    CHUNK_SIZE, CHUNK_OVERLAP, START_YEAR, END_YEAR, TOTAL_LETTERS
)
from utils import (
    extract_text_from_pdf, load_text_file, chunk_text,
    create_chunk_object, get_all_letters_sorted,
    save_chunks_jsonl, save_metadata, print_summary
)


def process_letter(file_path: Path, year: int, file_type: str) -> list:
    """Process a single letter file and return chunks."""
    # Extract text based on file type
    if file_type == "pdf":
        text = extract_text_from_pdf(file_path)
    else:
        text = load_text_file(file_path)
    
    if not text:
        print(f"Warning: No text extracted from {file_path.name}")
        return []
    
    # Split into chunks
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    
    # Create chunk objects with metadata
    chunk_objects = [
        create_chunk_object(
            chunk_text=chunk,
            year=year,
            source_file=file_path.name,
            chunk_index=i,
            total_chunks=len(chunks)
        )
        for i, chunk in enumerate(chunks)
    ]
    
    return chunk_objects


def run_pipeline():
    """Main pipeline execution."""
    print(f"\n{'='*60}")
    print("STARTING RAG INGESTION PIPELINE")
    print(f"{'='*60}")
    print(f"Processing {TOTAL_LETTERS} shareholder letters ({START_YEAR}-{END_YEAR})")
    print(f"Chunk size: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Collect all letters
    letters = get_all_letters_sorted()
    
    if not letters:
        print("Error: No letter files found in raw_data directory!")
        sys.exit(1)
    
    all_chunks = []
    year_stats = {}
    total_text_size = 0
    
    # Process each letter
    progress_bar = tqdm(letters, desc="Processing letters", unit="letter")
    for year, file_path, file_type in progress_bar:
        try:
            chunks = process_letter(file_path, year, file_type)
            all_chunks.extend(chunks)
            
            # Track statistics
            if year not in year_stats:
                year_stats[year] = {
                    "file": file_path.name,
                    "type": file_type,
                    "chunks": len(chunks)
                }
            
            # Estimate text size (rough)
            total_text_size += file_path.stat().st_size
            
            progress_bar.set_postfix({
                "chunks": len(chunks),
                "total_chunks": len(all_chunks)
            })
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue
    
    # Save chunks
    print(f"\nSaving {len(all_chunks)} chunks to JSONL format...")
    save_chunks_jsonl(all_chunks, CHUNKS_FILE)
    
    # Create and save metadata
    metadata = {
        "total_letters": len(letters),
        "total_chunks": len(all_chunks),
        "year_range": [START_YEAR, END_YEAR],
        "avg_chunks_per_letter": len(all_chunks) / len(letters) if letters else 0,
        "total_text_size_mb": total_text_size / (1024 * 1024),
        "chunk_config": {
            "size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP
        },
        "chunks_file": str(CHUNKS_FILE),
        "metadata_file": str(METADATA_FILE),
        "year_statistics": year_stats
    }
    
    save_metadata(metadata, METADATA_FILE)
    
    # Print summary
    print_summary(metadata)
    
    print("✅ Pipeline completed successfully!")
    
    return metadata


if __name__ == "__main__":
    try:
        metadata = run_pipeline()
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
