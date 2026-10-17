"""pv8-retrieval-validator group 3: /ask and the ft builders read the served context settings from one call-time source."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'ft'))

import round4  # noqa: E402
from src.services import ask_flow  # noqa: E402
from src.services.schemas import AskRequest  # noqa: E402
from src.storage import SearchHit  # noqa: E402

KEYS = ('ANSWER_CONTEXT_NEIGHBORS', 'LLM_PASSAGE_MAX_CHARS', 'LLM_N_CTX', 'LLM_CONTEXT_PASSAGES')


def _asked_kwargs(monkeypatch):
    """The kwargs /ask passes to prepare_answer_context."""
    hits = [SearchHit('p0', 'Passage 0 says something about float.', {'year': 2000}, 1.0)]
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return list(hits)

    monkeypatch.setattr(ask_flow, '_server_llm', lambda: object())
    monkeypatch.setattr(ask_flow, '_do_search', lambda req, retrieval_query=None: (hits, None, False))
    monkeypatch.setattr(ask_flow, 'prepare_answer_context', capture)
    monkeypatch.setattr(ask_flow, '_state', {})
    ask_flow._prepare_ask(AskRequest(query='What did Buffett say about float?', expand_query=False))
    return seen


def _same_as_serving(asked, served):
    for key in ('neighbors', 'max_chars', 'max_new_tokens', 'n_ctx', 'max_passages', 'passage_max_chars'):
        assert asked[key] == served[key], key


def test_env_overrides_reach_ask_and_serving_settings_alike(monkeypatch):
    for key, value in zip(KEYS, ('2', '3000', '12288', '12')):
        monkeypatch.setenv(key, value)
    asked = _asked_kwargs(monkeypatch)
    served = round4.serving_settings()
    _same_as_serving(asked, served)
    assert (served['neighbors'], served['passage_max_chars'], served['n_ctx'], served['max_passages']) == (2, 3000, 12288, 12)
    assert round4.passage_max_chars() == 3000


def test_no_env_values_equal_todays_constants(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    asked = _asked_kwargs(monkeypatch)
    served = round4.serving_settings()
    _same_as_serving(asked, served)
    assert served['passage_max_chars'] == 1800 == round4.passage_max_chars() == round4.LLM_PASSAGE_MAX_CHARS
    assert served['neighbors'] == round4.ANSWER_CONTEXT_NEIGHBORS
    assert served['n_ctx'] == 16384 == round4.LLM_N_CTX
    assert served['max_passages'] == round4.LLM_CONTEXT_PASSAGES
    assert served['max_chars'] == round4.ANSWER_CONTEXT_MAX_CHARS


def test_build_ft_v6_reads_passage_cap_at_call_time(monkeypatch):
    import build_ft_v6
    doc = type('Doc', (), {'text': ' '.join(f'Sentence number {i} is here.' for i in range(200))})()
    monkeypatch.setenv('LLM_PASSAGE_MAX_CHARS', '300')
    short = build_ft_v6.sentences_of(doc)
    monkeypatch.setenv('LLM_PASSAGE_MAX_CHARS', '3000')
    assert len(build_ft_v6.sentences_of(doc)) > len(short)
