"""pv8 PW-01 / EXP-06: the prompt version follows the model file; a contradicting PROMPT_VERSION raises."""
import pytest

from src.generation import prompt
from src.generation.providers.llama_provider import LlamaCppProvider

R1 = "models/ft/7b_r1/buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf"
R3 = "models/ft/7b_r3/buffett-qwen2.5-7b-ft-r3-q4_k_m.gguf"
R4 = "models/ft/7b_r4/buffett-qwen2.5-7b-ft-r4-q4_k_m.gguf"
R1_SHA = "ec41100e1457ffc88289d42c9c56de5bddb648c511db693cb75fd2d628182ec1"
R3_SHA = "b1bc1584a0cc4b029e6217f37feb1657a80215dac6500c8a73322946374c6fd0"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("PROMPT_VERSION", "LLM_MODEL_PATH", "ALLOW_PROMPT_MISMATCH"):
        monkeypatch.delenv(name, raising=False)


def test_registry_by_filename_and_sha():
    assert prompt.expected_prompt_version(R1) == "v1"
    assert prompt.expected_prompt_version(R3) == "v2"
    assert prompt.expected_prompt_version(R4) == "v3"
    assert prompt.expected_prompt_version("/x/some-other.gguf") is None
    assert prompt.expected_prompt_version("/x/renamed.gguf", sha256=R3_SHA) == "v2"
    assert prompt.expected_prompt_version("/x/buffett-qwen2.5-7b-ft-r3-q4_k_m.gguf", sha256=R1_SHA) == "v1"  # the bytes win


@pytest.mark.parametrize("path,version", [(R1, "v1"), (R3, "v2"), (R4, "v3")])
def test_unset_prompt_version_is_derived_from_the_model_path(monkeypatch, path, version):
    monkeypatch.setenv("LLM_MODEL_PATH", path)
    assert prompt.prompt_version() == version


def test_default_model_stays_ft_r1_with_v1():
    assert prompt.prompt_version() == "v1"


def test_unknown_model_falls_back_to_config(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_PATH", "/x/other.gguf")
    assert prompt.prompt_version() == "v1"


def test_contradicting_env_raises_at_startup(monkeypatch):
    monkeypatch.setenv("PROMPT_VERSION", "v1")
    with pytest.raises(ValueError, match="ft-r3"):
        prompt.check_prompt_matches_model(R3)
    with pytest.raises(ValueError):
        LlamaCppProvider(model_path=R3)


def test_matching_env_and_unset_env_pass(monkeypatch):
    prompt.check_prompt_matches_model(R3)  # unset: derived, nothing to contradict
    monkeypatch.setenv("PROMPT_VERSION", "v2")
    prompt.check_prompt_matches_model(R3)
    LlamaCppProvider(model_path=R3)


def test_allow_prompt_mismatch_overrides(monkeypatch):
    monkeypatch.setenv("PROMPT_VERSION", "v1")
    monkeypatch.setenv("ALLOW_PROMPT_MISMATCH", "1")
    prompt.check_prompt_matches_model(R3)
    assert prompt.prompt_version() == "v1"  # the experiment's explicit choice is served


def test_unknown_model_is_never_checked(monkeypatch):
    monkeypatch.setenv("PROMPT_VERSION", "v3")
    prompt.check_prompt_matches_model("/x/other.gguf")


def test_v1_v2_texts_unchanged():
    import hashlib
    assert hashlib.sha256(prompt.PROMPT_VERSIONS["v1"].encode()).hexdigest() == _V1
    assert hashlib.sha256(prompt.PROMPT_VERSIONS["v2"].encode()).hexdigest() == _V2


_V1 = "cbaa5e6a514b906cf1a141148a9e33bd5ec2ae1219b8be707eaa027620438c15"
_V2 = "7344a31edcb30a0bd480b81b0b5439c2465ac60f38f6e91576373103bbbc8a93"
