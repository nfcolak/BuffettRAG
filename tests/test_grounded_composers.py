"""Composer tests with fake mlx_lm / llama modules; no model is loaded."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from src.generation.grounded import composers as C
from src.generation.grounded.types import (
    ComposeRequest, EvidenceUnit, GroupEvidence, SourceSpan,
)


def _ev(i, text, score=0.9):
    sp = SourceSpan(i, 0, len(text), text)
    return GroupEvidence(i, text, EvidenceUnit(f"u{i}", i, f"p{i}", 1985, sp, sp, score))


def _req(n=2, **kw):
    return ComposeRequest("SYS", "What?", [_ev(i, f"Sentence number {i} here.") for i in range(n)], **kw)


class FakeTok:
    def encode(self, s):
        return s.split()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = " ".join(m["content"] for m in messages)
        return text.split() if tokenize else text


@pytest.fixture(autouse=True)
def _clean():
    C._MLX_CACHE.clear()
    yield
    C._MLX_CACHE.clear()


@pytest.fixture
def fake_mlx(monkeypatch):
    calls = SimpleNamespace(loads=[], gen=[], temps=[], fail=False)
    mod = types.ModuleType("mlx_lm")

    def load(path):
        calls.loads.append(path)
        return object(), FakeTok()

    def stream_generate(model, tok, prompt, max_tokens, sampler):
        calls.gen.append(max_tokens)
        if calls.fail:
            raise RuntimeError("boom")
        yield SimpleNamespace(text="Hello [1].", finish_reason="stop", generation_tokens=3)

    mod.load, mod.stream_generate = load, stream_generate
    su = types.ModuleType("mlx_lm.sample_utils")

    def make_sampler(temp=0.0):
        calls.temps.append(temp)
        return lambda x: x

    su.make_sampler = make_sampler
    monkeypatch.setitem(sys.modules, "mlx_lm", mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", su)
    return calls


def test_mlx_greedy_max_tokens_and_lazy_single_load(fake_mlx, tmp_path):
    comp = C.MlxComposer(str(tmp_path))
    assert fake_mlx.loads == []  # lazy
    r1 = comp.compose(_req(max_tokens=50, temperature=0.0))
    r2 = comp.compose(_req(max_tokens=77, temperature=0.0))
    assert len(fake_mlx.loads) == 1
    assert fake_mlx.temps == [0.0, 0.0] and fake_mlx.gen == [50, 77]
    assert r1.text == "Hello [1]." and r1.backend == "mlx" and r1.error_type is None
    assert comp.load_ms > 0 or comp.load_ms == 0.0
    C.MlxComposer(str(tmp_path)).compose(_req())  # second instance reuses the cache
    assert len(fake_mlx.loads) == 1


def test_mlx_nonzero_temperature_forwarded(fake_mlx, tmp_path):
    C.MlxComposer(str(tmp_path)).compose(_req(temperature=0.7))
    assert fake_mlx.temps == [0.7]


def test_mlx_error_becomes_error_type(fake_mlx, tmp_path):
    fake_mlx.fail = True
    r = C.MlxComposer(str(tmp_path)).compose(_req())
    assert r.error_type == "RuntimeError" and r.text == ""


def test_mlx_trims_lowest_score_visibly(fake_mlx, tmp_path):
    comp = C.MlxComposer(str(tmp_path), max_input_tokens=C._TEMPLATE_OVERHEAD + 200 + 20)
    req = ComposeRequest("SYS", "What?", [_ev(0, "a b c d", 0.9), _ev(1, "e f g h", 0.1)])
    assert comp.compose(req).error_type is None
    assert comp.last_trimmed == [1]


def test_import_works_without_mlx(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    import importlib
    importlib.reload(C)
    assert C.MlxComposer and C.LlamaComposer


def test_llama_greedy_and_error(monkeypatch, tmp_path):
    from src.generation.providers import llama_provider as lp

    seen = []

    class FakeLlama:
        def tokenize(self, b, add_bos=False):
            return b.split()

        def create_chat_completion(self, **kw):
            seen.append(kw)
            if kw["max_tokens"] == 13:
                raise ValueError("bad")
            return {"choices": [{"message": {"content": " ok [1]. "}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 2}}

    loads = []
    monkeypatch.setattr(lp, "_load_model", lambda *a: loads.append(a) or FakeLlama())
    comp = C.LlamaComposer(str(tmp_path / "m.gguf"))
    r = comp.compose(_req(max_tokens=40))
    assert r.text == "ok [1]." and r.tokens_out == 2 and r.backend == "llama"
    assert seen[0]["temperature"] == 0.0 and seen[0]["max_tokens"] == 40 and seen[0]["top_k"] == 1
    assert [m["role"] for m in seen[0]["messages"]] == ["system", "user"]
    assert comp.compose(_req(max_tokens=13)).error_type == "ValueError"
    comp.compose(_req(max_tokens=40, temperature=0.5))
    assert seen[-1]["temperature"] == 0.5 and "top_k" not in seen[-1]
    assert comp.raw_generate("hi", 5) == "ok [1]."
