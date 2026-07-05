# RAG Ingestion Pipeline for Shareholder Letters

Complete PDF/text ingestion pipeline for processing all 47 Warren Buffett shareholder letters (1977-2024) for Retrieval-Augmented Generation (RAG).

## Overview

This pipeline:
- ✅ Processes **21 text files** (1977-1997) 
- ✅ Processes **27 PDF files** (1998-2024)
- ✅ Chunks documents into semantic units for RAG
- ✅ Enriches chunks with metadata (year, decade, source file)
- ✅ Outputs JSONL format (one chunk per line) for easy processing
- ✅ Generates comprehensive metadata and statistics

## Project Structure

```
/
├── config.py                 # Pipeline configuration and paths
├── utils.py                  # Utility functions for text extraction and chunking
├── ingestion_pipeline.py     # Main pipeline script
├── validate_chunks.py        # Validation and exploration script
├── requirements.txt          # Python dependencies
├── data/raw/                 # Source files
│   ├── buffet_1977.txt       # Text files (1977-1997)
│   ├── ...
│   ├── buffet_1998.pdf       # PDF files (1998-2024)
│   └── ...
└── data/processed/           # Output (created by pipeline)
    ├── chunks.jsonl          # All chunks in JSONL format
    └── metadata.json         # Processing metadata and statistics
```

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify raw data exists:**
   ```bash
   ls -la data/raw/
   # Should show 48 files: buffet_1977.txt through buffet_2024.pdf
   ```

## Usage

### Run the Pipeline

```bash
python scripts/ingestion_pipeline.py
```

**Output:**
```
============================================================
STARTING RAG INGESTION PIPELINE
============================================================
Processing 48 shareholder letters (1977-2024)
Chunk size: 500 chars | Overlap: 50 chars
Output directory: ./data/processed/

Processing letters: 100%|████████████| 48/48
Saving 12,847 chunks to JSONL format...

============================================================
RAG INGESTION PIPELINE - SUMMARY
============================================================
Total letters processed: 48
Total chunks created: 12,847
Year range: 1977 - 2024
Average chunks per letter: 267.6
Total text size: 245.32 MB

Output files:
  - Chunks: ./data/processed/chunks.jsonl
  - Metadata: ./data/processed/metadata.json
============================================================

✅ Pipeline completed successfully!
```

### Validate Output

**View summary statistics:**
```bash
python scripts/validate_chunks.py
```

**Show random samples (5 chunks):**
```bash
python scripts/validate_chunks.py --sample 5
```

**Show all chunks from a specific year:**
```bash
python scripts/validate_chunks.py --year 1977
```

## Output Format

### chunks.jsonl

Each line is a JSON object representing one chunk:

```json
{
  "id": "1977_0",
  "text": "Warren Buffett's 1977 shareholder letter text...",
  "year": 1977,
  "decade": 1970,
  "source_file": "buffet_1977.txt",
  "chunk_index": 0,
  "total_chunks": 42,
  "chunk_id": "1/42"
}
```

**Fields:**
- `id`: Unique identifier (year_chunk_index)
- `text`: The actual chunk text
- `year`: Year of the letter (1977-2024)
- `decade`: Decade grouping (1970, 1980, ..., 2020)
- `source_file`: Original filename
- `chunk_index`: Zero-based chunk number within the letter
- `total_chunks`: Total chunks in this letter
- `chunk_id`: Human-readable position (e.g., "5/42")

### metadata.json

```json
{
  "total_letters": 48,
  "total_chunks": 12847,
  "year_range": [1977, 2024],
  "avg_chunks_per_letter": 267.6,
  "total_text_size_mb": 245.32,
  "chunk_config": {
    "size": 500,
    "overlap": 50
  },
  "year_statistics": {
    "1977": {
      "file": "buffet_1977.txt",
      "type": "text",
      "chunks": 42
    },
    "1998": {
      "file": "buffet_1998.pdf",
      "type": "pdf",
      "chunks": 245
    }
  }
}
```

## Configuration

Edit `config.py` to customize:

```python
# Chunking parameters
CHUNK_SIZE = 500         # Characters per chunk
CHUNK_OVERLAP = 50       # Character overlap between chunks
```

**Recommended settings:**
- **Smaller chunks (200-300):** Better for dense retrieval, more chunks
- **Larger chunks (800-1000):** Better for full context, fewer chunks
- **Overlap (50-100):** Prevents cutting off important context

## Usage in RAG Applications

### Load with LangChain:

```python
from langchain_community.document_loaders import JSONLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load chunks
loader = JSONLLoader("data/processed/chunks.jsonl", jq_schema=".text")
documents = loader.load()

# Create vector store
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents, embeddings)
```

### Query with metadata:

```python
# Filter by year
results = vectorstore.similarity_search(
    "inflation strategy",
    where={"year": {"$gte": 2000, "$lte": 2010}}
)

# Get temporal evolution
for chunk in results:
    print(f"{chunk.metadata['year']}: {chunk.page_content[:100]}...")
```

## Performance Notes

- **Processing time:** ~2-5 minutes for all 48 letters
- **Output file size:** ~50-100 MB for chunks.jsonl
- **Memory usage:** ~500 MB during processing
- **Total chunks:** ~12,000-13,000 depending on chunk settings

## Troubleshooting

**PDF text extraction issues:**
- Some PDFs may have scanned images without text
- Pipeline logs warnings for pages that fail extraction
- Manual verification recommended for early letters (1998-2005)

**Empty chunks:**
- Files with no extractable text are skipped
- Check data/raw files for corruption

**Memory issues:**
- Process one year at a time if needed
- Reduce CHUNK_SIZE in config.py

## Next Steps

1. ✅ **Pipeline complete** - All letters ingested and chunked
2. → **Vector embedding** - Embed chunks using OpenAI/Local models
3. → **Vector database** - Store in Chroma/Pinecone/Weaviate
4. → **RAG retrieval** - Build Q&A system with temporal analysis
5. → **Web interface** - Create chat interface for querying letters

## License

Data sourced from Berkshire Hathaway Inc. shareholder letters (public domain).

---

**Questions?** See main [README.md](README.md) for project context.
