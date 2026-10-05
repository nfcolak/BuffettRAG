"""Provider ``grounded``: structured, cited answers from the grounded engine.

Heavy modules (resources, engine, composers, torch/mlx) are imported lazily, on
first use, so importing this module or constructing the provider loads nothing.
The service layer calls ``answer_grounded``; ``generate`` exists only for query
expansion and goes to the composer's raw backend.
"""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

_KINDS = ("auto", "mlx", "llama", "template")


class ProviderUnavailable(RuntimeError):
    """The requested backend cannot serve this call (e.g. raw text from the template composer)."""


class GroundedProvider:
    provider_name = "grounded"

    def __init__(
        self,
        settings=None,
        *,
        composer_kind: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> None:
        if settings is None:
            from config import load_grounded_settings

            settings = load_grounded_settings()
        kind = (composer_kind or settings.COMPOSER).strip().lower()
        if kind not in _KINDS:
            raise ValueError(f"composer_kind must be one of {_KINDS}")
        if model_path is not None:
            path = str(Path(model_path).expanduser())
            if kind == "template":
                raise ValueError("model_path does not apply to the template composer")
            if kind == "auto":
                kind = "llama" if path.lower().endswith(".gguf") else "mlx"
            field = "LLAMA_MODEL" if kind == "llama" else "MLX_MODEL"
            settings = dataclasses.replace(settings, **{field: path})
        self.settings = settings
        self.composer_kind = kind
        self.model_path = None if kind in ("template", "auto") else str(
            settings.LLAMA_MODEL if kind == "llama" else settings.MLX_MODEL
        )
        self.model = f"grounded:{kind}" + (f":{Path(self.model_path).name}" if self.model_path else "")
        self._lock = threading.Lock()
        self._resources: Any = None
        self._engine: Any = None
        self._reranker: Any = None

    # ------------------------------------------------------------ lazy state

    def _get_resources(self):
        with self._lock:
            if self._resources is None:
                from src.generation.grounded.resources import get_resources

                self._resources = get_resources(self.settings)
                if self._reranker is not None:
                    self._resources.attach_resources(reranker=self._reranker)
            return self._resources

    def _get_engine(self):
        resources = self._get_resources()
        with self._lock:
            if self._engine is None:
                from src.generation.grounded.engine import GroundedEngine

                self._engine = GroundedEngine(resources, self.settings, composer_kind=self.composer_kind)
            return self._engine

    # ------------------------------------------------------------ public API

    def attach_resources(self, *, reranker: Any) -> None:
        """Reuse the backend retriever's reranker instead of loading a second scorer."""
        if reranker is None:
            return
        with self._lock:
            if self._reranker is reranker:
                return
            self._reranker = reranker
            resources = self._resources
        if resources is not None:
            resources.attach_resources(reranker=reranker)

    def answer_grounded(
        self,
        original_query: str,
        context_hits: Sequence[Any],
        *,
        history: Sequence[Mapping[str, str]] = (),
        max_new_tokens: int = 200,
    ):
        """Return a GroundedAnswer (answer text, citations, plan, trace)."""
        return self._get_engine().answer(
            original_query, context_hits, history=history, max_new_tokens=max_new_tokens
        )

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Raw text for query expansion only; the template composer has no model."""
        if self.composer_kind == "template":
            raise ProviderUnavailable("the template composer has no raw text generation")
        composer = self._get_resources().get_composer(self.composer_kind)
        return composer.raw_generate(prompt, min(max_new_tokens or 200, 200))
