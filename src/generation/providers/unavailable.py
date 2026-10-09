"""Placeholder provider used when no answer engine can be loaded."""

from __future__ import annotations

from typing import Optional


class UnavailableProvider:
    """Cannot generate: ``generate`` always raises with the reason.

    Lets the backend start (search keeps working) while ``/ask`` reports the
    existing "LLM unavailable" message.
    """

    provider_name = "unavailable"
    model = "unavailable"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        raise RuntimeError(f"No answer engine available: {self.reason}")
