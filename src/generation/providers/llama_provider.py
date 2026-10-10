"""Embedded llama.cpp (GGUF) generation provider. No network, no API keys."""

from __future__ import annotations

import atexit
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from config import (
    LLM_GPU_LAYERS,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_PATH,
    LLM_N_CTX,
    LLM_N_THREADS,
    LLM_TEMPERATURE,
)
from src.generation.prompt import SYSTEM_PROMPT

_SEED = 42
_USER_MARKERS = ("BEGIN RECENT CONVERSATION", "BEGIN UNTRUSTED PASSAGES")

# Loaded once per (path, n_ctx, n_threads, gpu_layers) and shared.
_MODEL_CACHE: Dict[Tuple[str, int, int, int], Any] = {}
_LOAD_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()  # a llama.cpp context is not thread-safe


def _close_models() -> None:
    """Free Metal resources before interpreter shutdown (avoids a ggml assert on exit)."""
    for model in list(_MODEL_CACHE.values()):
        try:
            model.close()
        except Exception:
            pass
    _MODEL_CACHE.clear()


atexit.register(_close_models)


def _load_model(model_path: Path, n_ctx: int, n_threads: int, n_gpu_layers: int):
    key = (str(model_path), n_ctx, n_threads, n_gpu_layers)
    with _LOAD_LOCK:
        if key not in _MODEL_CACHE:
            from llama_cpp import Llama

            kwargs: Dict[str, Any] = dict(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                seed=_SEED,
                verbose=False,
            )
            if n_threads > 0:
                kwargs["n_threads"] = n_threads
            _MODEL_CACHE[key] = Llama(**kwargs)
        return _MODEL_CACHE[key]


def split_prompt(prompt: str) -> Tuple[str, str]:
    """Split the grounded prompt from build_cited_prompt into (system, user)."""
    # Keep the trailing "Answer:" cue and the system prompt's trailing newline:
    # both match the fine-tuning rows exactly (training/serving parity).
    text = prompt.strip()
    if text.startswith(SYSTEM_PROMPT.strip()):
        user = text[len(SYSTEM_PROMPT.strip()):].strip()
        return SYSTEM_PROMPT, user
    positions = [text.find(m) for m in _USER_MARKERS if text.find(m) >= 0]
    if not positions:
        return "", text
    cut = min(positions)
    return text[:cut].strip(), text[cut:].strip()


class LlamaCppProvider:
    provider_name = "llama"

    def __init__(
        self,
        model_path: Path | str = LLM_MODEL_PATH,
        n_ctx: int = LLM_N_CTX,
        n_threads: int = LLM_N_THREADS,
        n_gpu_layers: int = LLM_GPU_LAYERS,
        temperature: float = LLM_TEMPERATURE,
    ) -> None:
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.model = self.model_path.name

    def _llm(self):
        return _load_model(self.model_path, self.n_ctx, self.n_threads, self.n_gpu_layers)

    def _messages(self, prompt: str):
        system, user = split_prompt(prompt)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return messages

    def _kwargs(self, max_new_tokens: Optional[int]) -> Dict[str, Any]:
        return dict(
            max_tokens=max_new_tokens or LLM_MAX_NEW_TOKENS,
            temperature=self.temperature,
            seed=_SEED,
        )

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        llm = self._llm()
        with _INFER_LOCK:
            out = llm.create_chat_completion(messages=self._messages(prompt), **self._kwargs(max_new_tokens))
        return (out["choices"][0]["message"].get("content") or "").strip()

    def generate_stream(self, prompt: str, max_new_tokens: Optional[int] = None) -> Iterator[str]:
        llm = self._llm()
        with _INFER_LOCK:
            for chunk in llm.create_chat_completion(
                messages=self._messages(prompt), stream=True, **self._kwargs(max_new_tokens)
            ):
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta
