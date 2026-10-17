"""pv8 EXP-05 / PW-08: every run JSON carries a `settings` block that identifies the served system."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "run_live_benchmark_pv8", Path(__file__).resolve().parents[1] / "scripts/eval/run_live_benchmark.py")
rlb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rlb)

KEYS = {"prompt_version", "model", "LLM_CONTEXT_PASSAGES", "n_ctx", "temperature", "ANSWER_NEIGHBOR_BEFORE",
        "ANSWER_NEIGHBOR_AFTER", "ANSWER_MAX_SENTENCES", "ANSWER_EXPANDED_MAX_SENTENCES", "ANSWER_REFUSAL_RETRY",
        "COMPARE_ROUTING_V2", "LLM_PASSAGE_ORDER", "corpus", "faiss_index_name", "git_commit"}


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setenv("BUFFETT_SHA_CACHE", str(tmp_path / "sha-cache.json"))
    model = tmp_path / "buffett-qwen2.5-7b-ft-r3-q4_k_m.gguf"
    model.write_bytes(b"gguf-bytes")
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text("{}\n")
    return model, corpus


def test_settings_block_has_every_field(files, monkeypatch):
    model, corpus = files
    for name in ("PROMPT_VERSION", "LLM_PASSAGE_ORDER", "ANSWER_NEIGHBOR_AFTER", "COMPARE_ROUTING_V2"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANSWER_NEIGHBOR_AFTER", "3")
    monkeypatch.setenv("COMPARE_ROUTING_V2", "1")
    s = rlb._run_settings(corpus=corpus, provider="llama", model_path=model, temperature=0.0)
    assert KEYS <= set(s)
    assert s["prompt_version"] == "v2"  # derived from the r3 file name
    assert s["model"] == {"path": str(model), "sha256": hashlib.sha256(b"gguf-bytes").hexdigest()}
    assert s["corpus"] == {"file": str(corpus), "sha256": hashlib.sha256(b"{}\n").hexdigest()}
    assert (s["ANSWER_NEIGHBOR_AFTER"], s["ANSWER_NEIGHBOR_BEFORE"]) == (3, 0)
    assert s["ANSWER_MAX_SENTENCES"] == s["ANSWER_EXPANDED_MAX_SENTENCES"] == 12
    assert s["COMPARE_ROUTING_V2"] == 1 and s["LLM_PASSAGE_ORDER"] == "natural"
    assert s["temperature"] == 0.0 and s["n_ctx"] == 16384 and s["ANSWER_REFUSAL_RETRY"] == "1"
    assert s["git_commit"] and len(s["git_commit"]) >= 7


def test_model_sha_is_cached_by_size_and_mtime(files, monkeypatch):
    model, _ = files
    calls = []
    real = rlb._sha256_of_file
    monkeypatch.setattr(rlb, "_sha256_of_file", lambda p: calls.append(p) or real(p))
    first = rlb._model_sha256(model)
    assert rlb._model_sha256(model) == first and len(calls) == 1
    model.write_bytes(b"other-bytes")  # new size/mtime -> recomputed
    assert rlb._model_sha256(model) == hashlib.sha256(b"other-bytes").hexdigest() and len(calls) == 2


def test_missing_model_file_records_null_sha(tmp_path, files):
    _, corpus = files
    s = rlb._run_settings(corpus=corpus, provider="llama", model_path=tmp_path / "nope.gguf", temperature=0.0)
    assert s["model"]["sha256"] is None


def test_prompt_override_is_recorded(files, monkeypatch):
    model, corpus = files
    monkeypatch.setenv("PROMPT_VERSION", "v3")
    monkeypatch.setenv("ALLOW_PROMPT_MISMATCH", "1")
    assert rlb._run_settings(corpus=corpus, provider="llama", model_path=model, temperature=0.0)["prompt_version"] == "v3"
