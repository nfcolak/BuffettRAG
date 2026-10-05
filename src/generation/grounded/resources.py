"""Process-level grounded resource registry; imports no torch/mlx/llama_cpp."""

from __future__ import annotations

import importlib
import os
import platform
import threading
from typing import Any

from src.retrieval.reranker import RERANK_LOCK

from .scorer import RerankerSentenceScorer
from .settings import GroundedSettings

__all__ = ["GroundedResourcesImpl", "get_resources", "rerank_lock", "reset_resources"]

_REGISTRY_LOCK = threading.Lock()
_REGISTRY: dict[GroundedSettings, "GroundedResourcesImpl"] = {}


def rerank_lock() -> Any:
    """The one process lock shared by rerank() and score_pairs()."""
    return RERANK_LOCK


def _module_available(name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


class GroundedResourcesImpl:
    def __init__(self, settings: GroundedSettings) -> None:
        self.settings = settings
        self.locks: dict[str, Any] = {
            "load": threading.RLock(),
            "scorer": RERANK_LOCK,
            "composer": threading.RLock(),
        }
        self._reranker: Any = None
        self._owns_reranker = False
        self._device = None
        self._composers: dict[tuple[str, str], Any] = {}
        self.scorer = RerankerSentenceScorer(_LazyReranker(self), model_id="reranker", lock=None)

    # -- reranker ---------------------------------------------------------
    def attach_resources(self, *, reranker: Any) -> None:
        with self.locks["load"]:
            if self._reranker is None:
                self._reranker = reranker
                self._owns_reranker = False
                self._device = str(getattr(reranker, "device", None) or "attached")

    def _resolve_device(self) -> str:
        dev = self.settings.SCORER_DEVICE
        if dev != "auto":
            return dev
        import torch

        mps = getattr(torch.backends, "mps", None)
        return "mps" if mps is not None and mps.is_available() else "cpu"

    def _ensure_reranker(self) -> Any:
        with self.locks["load"]:
            if self._reranker is None:
                from src.retrieval.reranker import CrossEncoderReranker

                device = self._resolve_device()
                self._reranker = CrossEncoderReranker(device=device)
                self._owns_reranker = True
                self._device = device
            return self._reranker

    def load(self) -> None:
        self._ensure_reranker()

    # -- composers --------------------------------------------------------
    def _resolve_kind(self, kind: str) -> str:
        if kind != "auto":
            return kind
        if platform.machine() == "arm64" and platform.system() == "Darwin" and _module_available("mlx_lm"):
            return "mlx"
        if _module_available("llama_cpp"):
            return "llama"
        return "template"

    def get_composer(self, kind: str) -> Any:
        if kind not in ("auto", "mlx", "llama", "template"):
            raise ValueError(f"unknown composer kind: {kind}")
        kind = self._resolve_kind(kind)
        if kind == "template":
            path = ""
        else:
            path = str(self.settings.MLX_MODEL if kind == "mlx" else self.settings.LLAMA_MODEL)
            path = os.path.abspath(path)
        key = (kind, path)
        with self.locks["load"]:
            if key not in self._composers:
                if kind == "template":
                    mod = importlib.import_module("src.generation.grounded.template")
                    self._composers[key] = mod.TemplateComposer()
                else:
                    mod = importlib.import_module("src.generation.grounded.composers")
                    cls = mod.MlxComposer if kind == "mlx" else mod.LlamaComposer
                    self._composers[key] = cls(path)
            return self._composers[key]

    @property
    def composer(self) -> Any:
        return self.get_composer(self.settings.COMPOSER)

    def describe(self) -> dict[str, Any]:
        return {
            "scorer_device": self._device,
            "scorer_device_setting": self.settings.SCORER_DEVICE,
            "reranker_loaded": self._reranker is not None,
            "reranker_attached": self._reranker is not None and not self._owns_reranker,
            "composers": sorted(f"{k}:{p}" for k, p in self._composers),
        }


class _LazyReranker:
    """Defers reranker construction until the first score_pairs call."""

    def __init__(self, owner: GroundedResourcesImpl) -> None:
        self._owner = owner

    def score_pairs(self, query: str, texts: Any) -> list[float]:
        return self._owner._ensure_reranker().score_pairs(query, texts)


def get_resources(settings: GroundedSettings) -> GroundedResourcesImpl:
    with _REGISTRY_LOCK:
        res = _REGISTRY.get(settings)
        if res is None:
            res = _REGISTRY[settings] = GroundedResourcesImpl(settings)
        return res


def reset_resources() -> None:
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
