"""Prompt version switch: v1 is byte-identical to the text FT-r1 was trained on; v2 is the coverage prompt."""
import hashlib

import pytest

from src.generation import prompt as P
from src.storage import SearchHit

# sha256 of SYSTEM_PROMPT at base commit 905b023 (the text FT-r1 was trained on)
V1_SHA256 = "cbaa5e6a514b906cf1a141148a9e33bd5ec2ae1219b8be707eaa027620438c15"
V1_LEN = 4212


def _hits():
    return [SearchHit(id="1995_p0001", text="Berkshire repurchased shares.", metadata={"year": 1995, "source_file": "buffet_1995.txt"}, score=1.0)]


def test_v1_is_byte_identical(monkeypatch):
    monkeypatch.delenv("PROMPT_VERSION", raising=False)
    assert len(P.SYSTEM_PROMPT) == V1_LEN
    assert hashlib.sha256(P.SYSTEM_PROMPT.encode()).hexdigest() == V1_SHA256
    assert P.system_prompt() == P.SYSTEM_PROMPT
    assert P.build_cited_prompt("Q?", _hits()).startswith(P.SYSTEM_PROMPT + "\n\n")


def test_default_is_v1_and_env_is_read_at_call_time(monkeypatch):
    monkeypatch.delenv("PROMPT_VERSION", raising=False)
    assert P.prompt_version() == "v1"
    before = P.build_cited_prompt("Q?", _hits())
    monkeypatch.setenv("PROMPT_VERSION", "v2")
    assert P.prompt_version() == "v2"
    after = P.build_cited_prompt("Q?", _hits())
    assert after.startswith(P.SYSTEM_PROMPT_V2 + "\n\n") and after != before
    assert after[len(P.SYSTEM_PROMPT_V2):] == before[len(P.SYSTEM_PROMPT):]   # only the system prompt differs
    monkeypatch.setenv("PROMPT_VERSION", "v9")
    with pytest.raises(ValueError):
        P.prompt_version()


def test_v2_differs_in_the_three_specified_places():
    v1, v2 = P.SYSTEM_PROMPT, P.SYSTEM_PROMPT_V2
    assert "Concise: one solid sentence beats two vague ones" in v1 and "Concise: one solid sentence beats two vague ones" not in v2
    assert "identify the missing evidence" in v1 and "identify the missing evidence" not in v2
    assert "Answer every part of the question" in v2 and "its own sentence" in v2 and "both sentences when one fact spans two" in v2
    assert "genuinely absent from the passages" in v2
    assert "Warren Buffett's Buffett Partnership letters (1957-1970) and Berkshire Hathaway shareholder letters (1977-2024)" in v2
    assert "annual Berkshire Hathaway" not in v2
    a, b = v1.splitlines(), v2.splitlines()
    assert len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 3


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_split_prompt_uses_the_active_version_and_matches_training_rows(monkeypatch, version):
    """Serving split == training row: system text is the active version's, unstripped; user starts at the passages."""
    from src.generation.providers.llama_provider import split_prompt

    monkeypatch.setenv("PROMPT_VERSION", version)
    prompt = P.build_cited_prompt("Q?", _hits())
    system, user = split_prompt(prompt)
    assert system == P.system_prompt(version)
    assert system.endswith("\n") == P.system_prompt(version).endswith("\n")
    assert user.startswith("BEGIN UNTRUSTED PASSAGES") and user.endswith("\n\nAnswer:")
    assert prompt.startswith(system + "\n\n") and prompt[len(system) + 2:].strip() == user
    other = "v1" if version == "v2" else "v2"
    assert system != P.system_prompt(other)


def test_v3_is_v2_with_the_berkshire_only_corpus_line():
    v1, v2, v3 = P.SYSTEM_PROMPT, P.SYSTEM_PROMPT_V2, P.SYSTEM_PROMPT_V3
    assert "Partnership" not in v3 and "1957" not in v3
    assert "Warren Buffett's annual Berkshire Hathaway shareholder letters (1977-2024)" in v3
    assert "Answer every part of the question" in v3 and "Concise: one solid sentence" not in v3
    a, b, c = v1.splitlines(), v2.splitlines(), v3.splitlines()
    assert len(c) == len(b) and sum(x != y for x, y in zip(b, c)) == 1   # only the corpus line differs from v2
    assert sum(x != y for x, y in zip(a, c)) == 2                         # coverage and tone lines differ from v1
    assert P.system_prompt("v3") == v3
