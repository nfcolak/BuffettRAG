"""Factory for selecting an LLM provider."""

from __future__ import annotations

import importlib.util
import logging
from typing import Optional

import config
from config import DEFAULT_LLM_PROVIDER
from src.generation.providers.base import LLMProvider
from src.generation.providers.llama_provider import LlamaCppProvider
from src.generation.providers.unavailable import UnavailableProvider

logger = logging.getLogger(__name__)


def create_llm_provider(provider: Optional[str] = None) -> LLMProvider:
    """Create the answer provider: ``llama`` (default).

    When the llama model file or ``llama_cpp`` is unavailable an
    ``UnavailableProvider`` is returned: the backend still starts and search works,
    while ``generate()`` raises. The extractive engine was removed; ``local`` and
    ``extractive`` raise ``ValueError``.
    """
    selected = (provider or DEFAULT_LLM_PROVIDER).strip().lower()

    if selected == "llama":
        path = config.LLM_MODEL_PATH
        if not path.is_file():
            reason = f"llama model file missing ({path})"
            logger.warning("%s; answers are unavailable", reason)
            return UnavailableProvider(reason)
        if importlib.util.find_spec("llama_cpp") is None:
            reason = "llama_cpp is not importable"
            logger.warning("%s; answers are unavailable", reason)
            return UnavailableProvider(reason)
        return LlamaCppProvider(model_path=path)

    if selected in ("local", "extractive"):
        raise ValueError("The extractive engine was removed; use 'llama'.")
    raise ValueError(f"Unsupported LLM provider: {selected}")
