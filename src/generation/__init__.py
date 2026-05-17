"""LLM loading and cited answer generation.

Lazy imports so that callers using only `prompt.py` (e.g. tests of citation
parsing) don't pull in `transformers` and `torch`.
"""

from importlib import import_module
from typing import Any

__all__ = ["LocalLLM", "build_cited_prompt", "parse_citations", "strip_chat_artifacts"]


def __getattr__(name: str) -> Any:
    if name == "LocalLLM":
        return getattr(import_module(".llm", __name__), name)
    if name in ("build_cited_prompt", "parse_citations", "strip_chat_artifacts"):
        return getattr(import_module(".prompt", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
