"""Sentence scorer over the shared cross-encoder (sigmoid of raw logits)."""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from typing import Any, Sequence

__all__ = ["RerankerSentenceScorer"]

_CACHE_SIZE = 2048


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class RerankerSentenceScorer:
    def __init__(self, reranker: Any, *, corpus_sha: str = "", model_id: str = "", lock: Any = None) -> None:
        self._reranker = reranker
        self._corpus_sha = corpus_sha
        self._model_id = model_id
        self._lock = lock
        self._cache: OrderedDict[tuple, float] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _key(self, query: str, text: str) -> tuple:
        return (self._corpus_sha, query, hashlib.sha256(text.encode("utf-8")).hexdigest(), self._model_id)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        texts = list(texts)
        if not texts:
            return []
        keys = [self._key(query, t) for t in texts]
        out: list[float | None] = [None] * len(texts)
        missing: list[int] = []
        with self._cache_lock:
            for i, k in enumerate(keys):
                if k in self._cache:
                    self._cache.move_to_end(k)
                    out[i] = self._cache[k]
                else:
                    missing.append(i)
        if missing:
            batch = [texts[i] for i in missing]
            if self._lock is not None:
                with self._lock:
                    logits = self._reranker.score_pairs(query, batch)
            else:
                logits = self._reranker.score_pairs(query, batch)
            logits = list(logits)
            if len(logits) != len(batch):
                raise ValueError(f"scorer returned {len(logits)} values for {len(batch)} texts")
            probs = []
            for v in logits:
                f = float(v)
                if not math.isfinite(f):
                    raise ValueError("scorer returned a non-finite logit")
                probs.append(_sigmoid(f))
            with self._cache_lock:
                for i, p in zip(missing, probs):
                    out[i] = p
                    self._cache[keys[i]] = p
                    self._cache.move_to_end(keys[i])
                while len(self._cache) > _CACHE_SIZE:
                    self._cache.popitem(last=False)
        return [float(p) for p in out]  # type: ignore[arg-type]
