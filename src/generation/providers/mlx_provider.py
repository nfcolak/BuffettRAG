"""MLX generation provider on the OLD prompt path (src/generation/prompt.py).

Same public surface as LlamaCppProvider; shares the lazy MLX loader with MlxComposer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from config import LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE
from src.generation.grounded.composers import infer_lock, load_mlx
from src.generation.providers.llama_provider import split_prompt


class MlxProvider:
    provider_name = "mlx"

    def __init__(self, model_path: str | None = None, temperature: float = LLM_TEMPERATURE) -> None:
        if model_path is None:
            from src.generation.grounded.settings import GroundedSettings
            model_path = str(GroundedSettings.from_env().MLX_MODEL)
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.temperature = temperature
        self.model = Path(self.model_path).name
        self.load_ms = 0.0
        self._lock = infer_lock(self.model_path)

    def _prompt_ids(self, prompt: str):
        model, tok, ms = load_mlx(self.model_path)
        self.load_ms += ms
        system, user = split_prompt(prompt)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return model, tok, ids

    def _stream(self, prompt: str, max_new_tokens: Optional[int]) -> Iterator[str]:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        model, tok, ids = self._prompt_ids(prompt)
        with self._lock:
            for resp in stream_generate(
                model, tok, prompt=ids, max_tokens=max_new_tokens or LLM_MAX_NEW_TOKENS,
                sampler=make_sampler(temp=float(self.temperature)),
            ):
                if resp.text:
                    yield resp.text

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        return "".join(self._stream(prompt, max_new_tokens)).strip()

    def generate_stream(self, prompt: str, max_new_tokens: Optional[int] = None) -> Iterator[str]:
        yield from self._stream(prompt, max_new_tokens)
