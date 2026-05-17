"""Local LLM loader for Mistral-7B-Instruct and Llama 3.1 8B Instruct.

On Nuvolos GPU we load with `torch_dtype=bfloat16` and `device_map='auto'`.
On CPU (e.g. the maintainer's laptop) we switch to fp32 and warn -- this is
slow but useful for smoke tests.

The class exposes a single `.generate(prompt)` method so it stays trivially
swappable for an OpenAI-compatible client later if needed.
"""

from __future__ import annotations

import os
from typing import Optional

from config import (
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_PRIMARY,
    LLM_TEMPERATURE,
    LLM_TOP_P,
)


class LocalLLM:
    def __init__(
        self,
        model_name: str = LLM_MODEL_PRIMARY,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        top_p: float = LLM_TOP_P,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError("transformers and torch are required.") from e

        self._torch = torch
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if dtype is None:
            dtype = "bfloat16" if device == "cuda" else "float32"
        torch_dtype = getattr(torch, dtype)

        print(f"Loading {model_name} on {device} (dtype={dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.getenv("HF_TOKEN"),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            token=os.getenv("HF_TOKEN"),
        )
        if device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate a completion. Returns only the new text (prompt stripped)."""
        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        # Decode only the new tokens (after the prompt)
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
