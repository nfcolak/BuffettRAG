"""Shared interface for answer-generation providers."""

from __future__ import annotations

from typing import Optional, Protocol


class LLMProvider(Protocol):
    """Minimal interface every generation provider must implement."""

    provider_name: str
    model: str

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate a grounded answer from a fully-built prompt."""
        ...
