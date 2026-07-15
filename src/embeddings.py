"""Embedding model wrapper.

Wraps `sentence-transformers` with the BGE-specific query instruction.
The `embed_documents` and `embed_query` API matches what LangChain expects,
so this class can be passed directly to `langchain_chroma.Chroma` if needed.
"""

from __future__ import annotations

from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is required. `pip install sentence-transformers`."
    ) from e

from config import (
    BGE_QUERY_INSTRUCTION,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_LARGE,
    EMBEDDING_MODEL_PRIMARY,
)


class BGEEmbedder:
    """Embedder for BAAI/bge-* models.

    Parameters
    ----------
    model_name : str
        HF model ID. Defaults to EMBEDDING_MODEL_PRIMARY (bge-base-en-v1.5).
    device : str
        'cpu' or 'cuda'.
    normalize : bool
        BGE models are trained with cosine similarity; embeddings should be
        L2-normalized so dot product == cosine.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_PRIMARY,
        device: str = EMBEDDING_DEVICE,
        normalize: bool = True,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 256,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32).tolist()

    def embed_query(self, query: str) -> List[float]:
        # BGE retrieval models expect the instruction prefix on the QUERY only,
        # never on the documents. This is the official guidance from BAAI.
        instructed = BGE_QUERY_INSTRUCTION + query
        emb = self._model.encode(
            [instructed],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )[0]
        return emb.astype(np.float32).tolist()


def get_embedder(size: str = "small", device: str = EMBEDDING_DEVICE) -> BGEEmbedder:
    """Factory kept for CLI compatibility: every size currently resolves to
    the configured bge-base model ('small' -> EMBEDDING_MODEL_PRIMARY,
    'base' / 'large' -> EMBEDDING_MODEL_LARGE, both bge-base-en-v1.5)."""
    if size == "small":
        return BGEEmbedder(EMBEDDING_MODEL_PRIMARY, device=device)
    if size in ("base", "large"):
        return BGEEmbedder(EMBEDDING_MODEL_LARGE, device=device)
    raise ValueError(f"Unknown embedder size: {size}")
