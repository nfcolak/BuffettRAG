"""Semantic chunking using LangChain's RecursiveCharacterTextSplitter.

The original `utils.chunk_text` did naive character slicing on whitespace-
collapsed text, which destroys paragraph structure and frequently cuts
mid-sentence. This module preserves paragraph -> sentence -> word boundaries
in that order, which matches how bge embeddings expect input.
"""

from __future__ import annotations

from typing import List

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover -- fallback for older langchain
    from langchain.text_splitter import RecursiveCharacterTextSplitter


# Order matters: try paragraph break first, then sentence-ending punctuation,
# then commas / semicolons, then word boundary, then character.
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def build_splitter(
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=_DEFAULT_SEPARATORS,
        keep_separator=True,
        add_start_index=False,
    )


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    min_chunk_chars: int = 160,
) -> List[str]:
    if not text or not text.strip():
        return []
    splitter = build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]
    return merge_short_chunks(
        chunks,
        min_chunk_chars=min_chunk_chars,
        target_chars=chunk_size + chunk_overlap,
    )


def merge_short_chunks(
    chunks: List[str],
    min_chunk_chars: int = 160,
    target_chars: int = 920,
) -> List[str]:
    """Attach tiny headings/table fragments to nearby context.

    Recursive splitting can produce semantically weak crumbs such as a section
    title, table label, or signature line. Those are poor embedding units on
    their own, so we merge them into the nearest neighbor when it stays near the
    intended chunk budget.
    """
    merged: List[str] = []
    pending = ""

    for raw_chunk in chunks:
        chunk = raw_chunk.strip()
        if not chunk:
            continue

        if pending:
            chunk = f"{pending}\n\n{chunk}"
            pending = ""

        if len(chunk) < min_chunk_chars and not merged:
            pending = chunk
            continue

        if (
            len(chunk) < min_chunk_chars
            and merged
            and len(merged[-1]) + len(chunk) + 2 <= target_chars
        ):
            merged[-1] = f"{merged[-1]}\n\n{chunk}"
            continue

        merged.append(chunk)

    if pending:
        if merged and len(merged[-1]) + len(pending) + 2 <= target_chars:
            merged[-1] = f"{merged[-1]}\n\n{pending}"
        else:
            merged.append(pending)

    return merged
