"""Anthropic Claude generation provider."""

from __future__ import annotations

from typing import Optional

from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL


class AnthropicProvider:
    provider_name = "anthropic"

    def __init__(
        self,
        api_key: str = ANTHROPIC_API_KEY,
        model: str = ANTHROPIC_MODEL,
    ) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic generation")

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("anthropic is required. Install with `pip install anthropic`.") from exc

        self.model = model
        self._client = Anthropic(api_key=api_key)

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_new_tokens or 600,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = []
        for block in message.content:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "\n".join(part for part in parts if part).strip()
