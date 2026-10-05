"""Validated grounded knobs, without importing config or any ML library.

Fields keep section 3's uppercase names; their environment keys are GROUNDED_
plus the field name. from_env reads only when called. Relative paths resolve
against this checkout, never the caller's working directory. In a .worktrees
checkout the default model paths point to the main checkout's read-only models;
CACHE_DIR always defaults to this checkout. No constructor creates directories
or checks model existence (fake/template use does not need weights).

SCORER_DEVICE="auto" defers the mps-if-available, otherwise CPU decision to lazy
resource loading, keeping this module torch-free. NLI_MODEL=None is valid only
when NLI_VERIFY is false; enabling the dev-only ablation requires an explicit
local model path. MAX_UNITS is the calibration range [4, 8], MAX_PERIODS <= 4,
MAX_PER_HIT <= 2, MAX_WINDOWS <= 48, and GROUP_MAX_TOKENS <= 200.
NUMERIC_GUARD selects the verifier's R1 rule: "verbatim" (a sentence with any
quantity must equal a cited source sentence) or "bound" (numeric paraphrase
allowed when every quantity, its order, qualifiers and context stay bound to
one cited source sentence). Reservations
must fit the window budget, and units must fit that budget as well.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping

__all__ = ["GroundedSettings"]

_CHECKOUT = Path(__file__).resolve().parents[3]
_MODEL_CHECKOUT = next(
    (parent.parent for parent in _CHECKOUT.parents if parent.name == ".worktrees"),
    _CHECKOUT,
)


def _path(value: Path | str, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty local path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else _CHECKOUT / path).resolve()


def _integer(name: str, value: int, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _probability(name: str, value: float) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or not 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class GroundedSettings:
    COMPOSER: str = "auto"
    MLX_MODEL: Path | str = _MODEL_CHECKOUT / "models/teacher-qwen2.5-7b-mlx4"
    LLAMA_MODEL: Path | str = (
        _MODEL_CHECKOUT / "models/qwen2.5-1.5b-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    )
    SCORER_DEVICE: str = "auto"
    MAX_WINDOWS: int = 48
    SLOT_RESERVE: int = 6
    MAX_UNITS: int = 6
    MAX_PER_HIT: int = 2
    MAX_PERIODS: int = 4
    T_RELEVANT: float = 0.01
    T_SLOT: float = 0.001
    DEDUPE_JACCARD: float = 0.8
    GROUP_MAX_TOKENS: int = 200
    NLI_VERIFY: bool = False
    NLI_MODEL: Path | str | None = None
    NUMERIC_GUARD: str = "bound"
    CACHE_DIR: Path | str = _CHECKOUT / "data/evaluation/grounded_cache"

    def __post_init__(self) -> None:
        if self.COMPOSER not in ("auto", "mlx", "llama", "template"):
            raise ValueError("COMPOSER must be auto, mlx, llama or template")
        if not isinstance(self.SCORER_DEVICE, str) or not re.fullmatch(
            r"auto|cpu|mps|cuda(?::\d+)?", self.SCORER_DEVICE
        ):
            raise ValueError("SCORER_DEVICE must be auto, cpu, mps, cuda or cuda:<index>")
        if self.NUMERIC_GUARD not in ("verbatim", "bound"):
            raise ValueError("NUMERIC_GUARD must be verbatim or bound")
        _integer("MAX_WINDOWS", self.MAX_WINDOWS, 1, 48)
        _integer("SLOT_RESERVE", self.SLOT_RESERVE, 1, self.MAX_WINDOWS)
        _integer("MAX_UNITS", self.MAX_UNITS, 4, 8)
        _integer("MAX_PER_HIT", self.MAX_PER_HIT, 1, 2)
        _integer("MAX_PERIODS", self.MAX_PERIODS, 1, 4)
        _integer("GROUP_MAX_TOKENS", self.GROUP_MAX_TOKENS, 1, 200)
        if self.MAX_UNITS > self.MAX_WINDOWS:
            raise ValueError("MAX_UNITS must fit MAX_WINDOWS")
        if self.MAX_PERIODS * self.SLOT_RESERVE > self.MAX_WINDOWS:
            raise ValueError("MAX_PERIODS * SLOT_RESERVE must fit MAX_WINDOWS")
        for name in ("T_RELEVANT", "T_SLOT", "DEDUPE_JACCARD"):
            _probability(name, getattr(self, name))
        if type(self.NLI_VERIFY) is not bool:
            raise ValueError("NLI_VERIFY must be a bool (environment: 0 or 1)")
        if self.NLI_VERIFY and self.NLI_MODEL is None:
            raise ValueError("NLI_MODEL is required when NLI_VERIFY is enabled")
        for name in ("MLX_MODEL", "LLAMA_MODEL", "CACHE_DIR"):
            object.__setattr__(self, name, _path(getattr(self, name), name))
        if self.NLI_MODEL is not None:
            object.__setattr__(self, "NLI_MODEL", _path(self.NLI_MODEL, "NLI_MODEL"))

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> GroundedSettings:
        env = os.environ if environ is None else environ
        values = {}
        integers = {"MAX_WINDOWS", "SLOT_RESERVE", "MAX_UNITS", "MAX_PER_HIT", "MAX_PERIODS", "GROUP_MAX_TOKENS"}
        probabilities = {"T_RELEVANT", "T_SLOT", "DEDUPE_JACCARD"}
        for item in fields(cls):
            key = "GROUNDED_" + item.name
            if key not in env:
                continue
            raw = env[key]
            if item.name in integers:
                if not re.fullmatch(r"[0-9]+", raw):
                    raise ValueError(f"{key} must be an integer")
                values[item.name] = int(raw)
            elif item.name in probabilities:
                values[item.name] = float(raw)
            elif item.name == "NLI_VERIFY":
                if raw not in ("0", "1"):
                    raise ValueError(f"{key} must be 0 or 1")
                values[item.name] = raw == "1"
            else:
                values[item.name] = raw
        return cls(**values)
