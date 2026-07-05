"""OpenRouter generation provider."""

from __future__ import annotations

from typing import Optional

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_FALLBACK_MODEL,
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
        try:
            return self._complete(self.model, prompt, max_new_tokens)
        except Exception as exc:
            # Free-tier models get rate-limited or exhausted upstream; retry
            # once on the fallback model before giving up.
            fallback = OPENROUTER_FALLBACK_MODEL
            if fallback and fallback != self.model and _is_retryable(exc):
                return self._complete(fallback, prompt, max_new_tokens)
            raise

    def _complete(self, model: str, prompt: str, max_new_tokens: Optional[int]) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            # Reasoning models narrate their thinking into the answer otherwise
            # (exclude:true only hides the reasoning field, not the narration);
            # non-reasoning models ignore this OpenRouter extension.
            extra_body={"reasoning": {"enabled": False}},
        )
        choices = getattr(response, "choices", None)
        if not choices:
            # OpenRouter reports upstream failures as {choices: null, error: {...}}
            # which the OpenAI SDK parses without raising.
            detail = getattr(response, "error", None)
            if detail is None:
                detail = (getattr(response, "model_extra", None) or {}).get("error")
            raise RuntimeError(f"OpenRouter returned no choices: {detail}")
        content = choices[0].message.content
        return (content or "").strip()


_RETRYABLE_MARKERS = (
    "429",
    "rate-limit",
    "rate limit",
    "resourceexhausted",
    "resource exhausted",
    "limit reached",
    "502",
    "503",
    "overloaded",
    "no choices",
    "temporarily",
)


def _is_retryable(exc: Exception) -> bool:
    raw = str(exc).lower()
    return any(marker in raw for marker in _RETRYABLE_MARKERS)
