"""Factory for selecting an LLM provider."""

from __future__ import annotations

import importlib.util
import logging
from typing import Optional

from config import DEFAULT_LLM_PROVIDER, LLM_MODEL_PATH
from src.generation.providers.base import LLMProvider
from src.generation.providers.llama_provider import LlamaCppProvider
from src.generation.providers.local_provider import LocalProvider

logger = logging.getLogger(__name__)


def create_llm_provider(provider: Optional[str] = None) -> LLMProvider:
    """Create the embedded provider: ``llama`` (default), ``local`` (extractive), ``mlx`` or ``grounded``."""
    selected = (provider or DEFAULT_LLM_PROVIDER).strip().lower()

    if selected == "llama":
        if not LLM_MODEL_PATH.is_file():
            logger.warning("llama model file missing (%s); falling back to 'local' provider", LLM_MODEL_PATH)
            return LocalProvider()
        if importlib.util.find_spec("llama_cpp") is None:
            logger.warning("llama_cpp is not importable; falling back to 'local' provider")
            return LocalProvider()
        return LlamaCppProvider()

    if selected == "local":
        return LocalProvider()

    # Factory-local imports: grounded/mlx stay out of the import graph of old providers.
    if selected == "grounded":
        from src.generation.providers.grounded_provider import GroundedProvider

        return GroundedProvider()

    if selected == "mlx":
        from src.generation.providers.mlx_provider import MlxProvider

        return MlxProvider()

    raise ValueError(f"Unsupported LLM provider: {selected}")
