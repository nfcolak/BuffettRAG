"""OpenAI generation provider."""

from __future__ import annotations

from typing import Optional

from config import OPENAI_API_KEY, OPENAI_MODEL


class OpenAIProvider:
    provider_name = "openai"

    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        model: str = OPENAI_MODEL,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI generation")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai is required. Install with `pip install openai`.") from exc

        self.model = model
        self._client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        response = self._client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=max_new_tokens,
        )
        return response.output_text.strip()
