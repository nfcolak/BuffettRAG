#!/usr/bin/env python3
"""
Validation and exploration script for processed chunks.

Usage:
    python scripts/validate_chunks.py                    # Show summary
    python scripts/validate_chunks.py --sample 5         # Show 5 random chunks
    python scripts/validate_chunks.py --year 1977        # Show chunks from specific year
"""

import json
import argparse
import random
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHUNKS_FILE, CHUNKS_V2_FILE, METADATA_FILE, METADATA_V2_FILE


def resolve_chunks_file() -> Path:
    """Use the current v2 chunks file when available, otherwise fall back to v1."""
    return CHUNKS_V2_FILE if CHUNKS_V2_FILE.exists() else CHUNKS_FILE


def resolve_metadata_file() -> Path:
    """Use metadata that matches the active chunks file."""
    return METADATA_V2_FILE if CHUNKS_V2_FILE.exists() else METADATA_FILE


def load_metadata() -> dict:
    """Load metadata JSON."""
    metadata_file = resolve_metadata_file()
    if not metadata_file.exists():
        print("Error: metadata.json not found. Run scripts/ingestion_pipeline.py first.")
        return None
    
    with open(metadata_file, 'r') as f:
        return json.load(f)


def load_chunks_by_year(target_year: int = None) -> list:
    """Load chunks, optionally filtered by year."""
    chunks = []
    chunks_file = resolve_chunks_file()
    if not chunks_file.exists():
        print("Error: no chunks file found. Run scripts/ingestion_pipeline.py first.")
        return None
    
    with open(chunks_file, 'r') as f:
        for line in f:
            chunk = json.loads(line)
            if target_year is None or chunk['year'] == target_year:
                chunks.append(chunk)
    
    return chunks


def print_summary():
    """Print processing summary."""
    metadata = load_metadata()
    if not metadata:
        return
    
    print("\n" + "="*70)
    print("PROCESSED DATA SUMMARY")
    print("="*70)
    print(f"Total letters: {metadata['total_letters']}")
    print(f"Total chunks: {metadata['total_chunks']}")
    print(f"Year range: {metadata['year_range'][0]}-{metadata['year_range'][1]}")
    print(f"Average chunks per letter: {metadata['avg_chunks_per_letter']:.1f}")
    if "total_text_size_mb" in metadata:
        print(f"Total text size: {metadata['total_text_size_mb']:.2f} MB")
    print("\nChunks per year:")
    print("-" * 70)
    
    year_chunks = defaultdict(int)
    chunks = load_chunks_by_year()
    for chunk in chunks:
        year_chunks[chunk['year']] += 1
    
    for year in sorted(year_chunks.keys()):
        count = year_chunks[year]
        bar_length = int(count / 5)  # Scale for display
        bar = "█" * bar_length
        print(f"  {year}: {count:3d} chunks  {bar}")
    
    print("="*70 + "\n")


def print_sample_chunks(num_samples: int = 5):
    """Print random sample chunks."""
    chunks = load_chunks_by_year()
    if not chunks:
        return
    
    samples = random.sample(chunks, min(num_samples, len(chunks)))
    
    print("\n" + "="*70)
    print(f"RANDOM SAMPLE ({num_samples} chunks)")
    print("="*70 + "\n")
    
    for i, chunk in enumerate(samples, 1):
        print(f"Sample {i}:")
        print(f"  Year: {chunk['year']}")
        print(f"  Source: {chunk['source_file']}")
        print(f"  Position: Chunk {chunk['chunk_id']}")
        print(f"  ID: {chunk['id']}")
        print(f"  Text preview: {chunk['text'][:200]}...")
        print()


def print_year_chunks(year: int):
    """Print all chunks from a specific year."""
    chunks = load_chunks_by_year(target_year=year)
    if not chunks:
        print(f"No chunks found for year {year}")
        return
    
    print("\n" + "="*70)
    print(f"CHUNKS FROM {year} ({len(chunks)} total)")
    print("="*70 + "\n")
    
    for chunk in chunks[:5]:  # Show first 5
        print(f"Chunk {chunk['chunk_id']} ({chunk['id']}):")
        print(f"  Text: {chunk['text'][:300]}...")
        print()
    
    if len(chunks) > 5:
        print(f"... and {len(chunks) - 5} more chunks from {year}")


def main():
    parser = argparse.ArgumentParser(description="Validate and explore processed chunks")
    parser.add_argument("--sample", type=int, help="Show N random sample chunks")
    parser.add_argument("--year", type=int, help="Show chunks from specific year")
    
    args = parser.parse_args()
    
    print_summary()
    
    if args.sample:
        print_sample_chunks(args.sample)
    elif args.year:
        print_year_chunks(args.year)


if __name__ == "__main__":
    main()
