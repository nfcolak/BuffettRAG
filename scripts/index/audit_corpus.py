"""Read-only full-corpus extraction/integrity audit; never replaces the index."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import CHUNKS_V3_FILE
from src.ingestion.pdf_extractor import extract_text_from_pdf, load_text_file
from src.storage import load_chunks_as_docs


def audit(corpus: Path = CHUNKS_V3_FILE):
    corpus = Path(corpus).resolve()
    docs = load_chunks_as_docs(corpus)
    by_id = {d.id: d for d in docs}
    bad_links = []
    for doc in docs:
        for field in ('previous_chunk_id', 'next_chunk_id'):
            target_id = doc.metadata.get(field)
            if target_id:
                target = by_id.get(target_id)
                if target is None or any(target.metadata.get(k) != doc.metadata.get(k)
                                         for k in ('year', 'source_file')):
                    bad_links.append([doc.id, field, target_id])
    files = []
    for path in sorted((ROOT / 'data/raw').glob('buffet_*')):
        if path.suffix not in ('.txt', '.pdf'):
            continue
        text = extract_text_from_pdf(path) if path.suffix == '.pdf' else load_text_file(path)
        chunks = [d for d in docs if d.metadata.get('source_file') == path.name]
        files.append({'file': path.name, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'extracted_chars': len(text), 'paragraph_breaks': text.count('\n\n'),
                      'line_breaks': text.count('\n'), 'stored_chunks': len(chunks),
                      'chunks_with_paragraph_breaks': sum('\n\n' in d.text for d in chunks),
                      'chunks_without_sentence_end': sum(not d.text.rstrip().endswith(('.', '!', '?', '"', '”')) for d in chunks)})
    return {'corpus': str(corpus.relative_to(ROOT)),
            'corpus_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
            'n_sources': len(files), 'n_chunks': len(docs),
            'duplicate_ids': len(docs) - len(by_id), 'bad_neighbor_links': bad_links,
            'empty_source_files': [f['file'] for f in files if not f['extracted_chars']],
            'note': 'Line/paragraph and sentence-end counts are diagnostics, not quality labels. Tables/headings need layout-aware inspection.',
            'files': files}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=CHUNKS_V3_FILE)
    args = parser.parse_args()
    print(json.dumps(audit(args.corpus), indent=2))
