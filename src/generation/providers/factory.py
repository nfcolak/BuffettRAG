"""Factory for selecting an LLM provider."""

from __future__ import annotations

from typing import Optional

from config import DEFAULT_LLM_PROVIDER
from src.generation.providers.anthropic_provider import AnthropicProvider
from src.generation.providers.base import LLMProvider
from src.generation.providers.local_provider import LocalProvider
from src.generation.providers.openai_provider import OpenAIProvider
from src.generation.providers.openrouter_provider import OpenRouterProvider


def create_llm_provider(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMProvider:
    """Create a generation provider without exposing provider internals upstream."""
    selected = (provider or DEFAULT_LLM_PROVIDER).strip().lower()

    if selected == "openai":
        kwargs = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if model is not None:
            kwargs["model"] = model
        return OpenAIProvider(**kwargs)

    if selected == "openrouter":
        kwargs = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if model is not None:
            kwargs["model"] = model
        return OpenRouterProvider(**kwargs)

    if selected == "anthropic":
        kwargs = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if model is not None:
            kwargs["model"] = model
        return AnthropicProvider(**kwargs)

    if selected == "local":
        # The embedded engine needs no API key; ignore one if supplied.
        kwargs = {}
        if model is not None:
            kwargs["model"] = model
        return LocalProvider(**kwargs)

    raise ValueError(f"Unsupported LLM provider: {selected}")
