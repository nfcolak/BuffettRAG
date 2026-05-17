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
) -> List[str]:
    if not text or not text.strip():
        return []
    splitter = build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)
