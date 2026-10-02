"""Shared interface for answer-generation providers."""

from __future__ import annotations

from typing import Iterator, Optional, Protocol


class LLMProvider(Protocol):
    """Minimal interface every generation provider must implement."""

    provider_name: str
    model: str

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate a grounded answer from a fully-built prompt."""
        ...

    def generate_stream(self, prompt: str, max_new_tokens: Optional[int] = None) -> Iterator[str]:
        """Optional: yield answer text deltas."""
        ...
