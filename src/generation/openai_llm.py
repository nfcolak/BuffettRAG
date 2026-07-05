"""Backward-compatible OpenAI provider import."""

from src.generation.providers.openai_provider import OpenAIProvider


class OpenAILLM(OpenAIProvider):
    """Compatibility alias for older imports.

    New code should use `src.generation.providers.create_llm_provider`.
    """
