"""Retrieval strategies: vector, metadata-aware, hybrid, reranking.

The submodules (`retriever`, `reranker`) pull in `sentence-transformers` and
`numpy`. Importing them lazily here so that consumers that only need BM25 or
utility functions (e.g. unit tests, scripts that just want `detect_year_filter`)
don't have to install the full ML stack.
"""

from importlib import import_module
from typing import Any

__all__ = ["Retriever", "RetrievalResult", "CrossEncoderReranker"]


def __getattr__(name: str) -> Any:
    if name in ("Retriever", "RetrievalResult"):
        mod = import_module(".retriever", __name__)
        return getattr(mod, name)
    if name == "CrossEncoderReranker":
        mod = import_module(".reranker", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
