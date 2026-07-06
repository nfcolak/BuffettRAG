"""Cross-encoder reranker.

We use BAAI/bge-reranker-base because it pairs naturally with the bge
retrieval embeddings and is small enough (~280MB) to run on CPU at
batch sizes 32-64. On GPU it's effectively free.

Standard usage in the pipeline:
    candidates = retriever.search(query, top_k=30)
    top = reranker.rerank(query, candidates, top_k=8)
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from config import EMBEDDING_DEVICE, RERANKER_MODEL
from src.vector_store import SearchHit


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        device: str = EMBEDDING_DEVICE,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover
            raise ImportError("sentence-transformers is required.") from e

        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchHit],
        top_k: int = 8,
    ) -> List[SearchHit]:
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        raw = np.asarray(
            self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False),
            dtype=float,
        )
        # Cross-encoder outputs are uncalibrated (often clustered near 0):
        # min-max normalize within the candidate set so downstream consumers
        # (UI score bars, thresholds) see a meaningful 0..1 spread.
        spread = float(raw.max() - raw.min())
        normalized = (raw - raw.min()) / spread if spread > 0 else np.ones_like(raw)
        order = np.argsort(-raw)
        reranked: List[SearchHit] = []
        for idx in order[:top_k]:
            c = candidates[int(idx)]
            reranked.append(
                SearchHit(
                    id=c.id,
                    text=c.text,
                    metadata=c.metadata,
                    score=float(normalized[int(idx)]),
                )
            )
        return reranked
