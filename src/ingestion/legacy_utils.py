"""Utility functions for the RAG ingestion pipeline."""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber
from config import RAW_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def extract_year_from_filename(filename: str) -> int:
    """Extract year from filename (e.g., 'buffet_1977.txt' -> 1977)."""
    match = re.search(r'(\d{4})', filename)
    return int(match.group(1)) if match else None


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file using pdfplumber."""
    text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
                except Exception as e:
                    print(f"Warning: Failed to extract page {page_num + 1} from {pdf_path.name}: {e}")
        return "\n".join(text)
    except Exception as e:
        print(f"Error reading PDF {pdf_path.name}: {e}")
        return ""


def load_text_file(text_path: Path) -> str:
    """Load text from a .txt file."""
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading text file {text_path.name}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    if not text:
        return chunks
    
    # Clean up text
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) <= chunk_size:
        return [text]
    
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def create_chunk_object(
    chunk_text: str,
    year: int,
    source_file: str,
    chunk_index: int,
    total_chunks: int
) -> Dict[str, Any]:
    """Create a structured chunk object with metadata."""
    decade = (year // 10) * 10
    return {
        "id": f"{year}_{chunk_index}",
        "text": chunk_text,
        "year": year,
        "decade": decade,
        "source_file": source_file,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "chunk_id": f"{chunk_index + 1}/{total_chunks}"
    }


def get_all_letters_sorted() -> List[tuple]:
    """Get all letter files sorted by year."""
    letters = []
    
    # Get text files
    text_files = sorted(RAW_DATA_DIR.glob("buffet_*.txt"))
    for txt_file in text_files:
        year = extract_year_from_filename(txt_file.name)
        if year:
            letters.append((year, txt_file, "text"))
    
    # Get PDF files
    pdf_files = sorted(RAW_DATA_DIR.glob("buffet_*.pdf"))
    for pdf_file in pdf_files:
        year = extract_year_from_filename(pdf_file.name)
        if year:
            letters.append((year, pdf_file, "pdf"))
    
    # Sort by year
    letters.sort(key=lambda x: x[0])
    return letters


def save_chunks_jsonl(chunks: List[Dict[str, Any]], output_path: Path):
    """Save chunks in JSONL format (one JSON object per line)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')


def save_metadata(metadata: Dict[str, Any], output_path: Path):
    """Save metadata as JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def print_summary(metadata: Dict[str, Any]):
    """Print processing summary."""
    print("\n" + "="*60)
    print("RAG INGESTION PIPELINE - SUMMARY")
    print("="*60)
    print(f"Total letters processed: {metadata['total_letters']}")
    print(f"Total chunks created: {metadata['total_chunks']}")
    print(f"Year range: {metadata['year_range'][0]} - {metadata['year_range'][1]}")
    print(f"Average chunks per letter: {metadata['avg_chunks_per_letter']:.1f}")
    print(f"Total text size: {metadata['total_text_size_mb']:.2f} MB")
    print(f"\nOutput files:")
    print(f"  - Chunks: {metadata['chunks_file']}")
    print(f"  - Metadata: {metadata['metadata_file']}")
    print("="*60 + "\n")
