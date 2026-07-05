"""OpenRouter generation provider."""

from __future__ import annotations

from typing import Optional

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_SITE_URL,
)


class OpenRouterProvider:
    provider_name = "openrouter"

    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        model: str = OPENROUTER_MODEL,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter generation")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai is required. Install with `pip install openai`.") from exc

        headers = {}
        if OPENROUTER_SITE_URL:
            headers["HTTP-Referer"] = OPENROUTER_SITE_URL
        if OPENROUTER_APP_NAME:
            headers["X-OpenRouter-Title"] = OPENROUTER_APP_NAME

        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers=headers or None,
        )

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
        )
        content = response.choices[0].message.content
        return (content or "").strip()
