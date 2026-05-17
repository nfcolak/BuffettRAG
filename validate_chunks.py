#!/usr/bin/env python3
"""
Validation and exploration script for processed chunks.

Usage:
    python validate_chunks.py                    # Show summary
    python validate_chunks.py --sample 5         # Show 5 random chunks
    python validate_chunks.py --year 1977        # Show chunks from specific year
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from config import OUTPUT_DIR, CHUNKS_FILE, METADATA_FILE


def load_metadata() -> dict:
    """Load metadata JSON."""
    if not METADATA_FILE.exists():
        print("Error: metadata.json not found. Run ingestion_pipeline.py first.")
        return None
    
    with open(METADATA_FILE, 'r') as f:
        return json.load(f)


def load_chunks_by_year(target_year: int = None) -> list:
    """Load chunks, optionally filtered by year."""
    chunks = []
    if not CHUNKS_FILE.exists():
        print("Error: chunks.jsonl not found. Run ingestion_pipeline.py first.")
        return None
    
    with open(CHUNKS_FILE, 'r') as f:
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
