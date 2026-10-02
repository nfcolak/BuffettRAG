"""LLM loading and cited answer generation."""

from importlib import import_module
from typing import Any

__all__ = [
    "LLMProvider",
    "LlamaCppProvider",
    "build_cited_prompt",
    "create_llm_provider",
    "parse_citations",
    "strip_chat_artifacts",
]


def __getattr__(name: str) -> Any:
    if name in ("LLMProvider", "LlamaCppProvider", "create_llm_provider"):
        return getattr(import_module(".providers", __name__), name)
    if name in ("build_cited_prompt", "parse_citations", "strip_chat_artifacts"):
        return getattr(import_module(".prompt", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
