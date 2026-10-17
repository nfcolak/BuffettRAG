"""pv8 ALT-04: LLM_PASSAGE_ORDER = natural | reverse | shuffle:<seed> on the served passages."""
import pytest

from src.services import ask_flow
from src.services.schemas import AskRequest
from src.storage import SearchHit


def _hits(n=6):
    return [SearchHit(f"p{i}", f"Passage {i} says something about float.", {"year": 2000 + i}, 1.0) for i in range(n)]


def test_natural_is_default_and_identity(monkeypatch):
    monkeypatch.delenv("LLM_PASSAGE_ORDER", raising=False)
    hits = _hits()
    assert ask_flow.order_passages(hits) == hits


def test_reverse(monkeypatch):
    monkeypatch.setenv("LLM_PASSAGE_ORDER", "reverse")
    assert [h.id for h in ask_flow.order_passages(_hits())] == ["p5", "p4", "p3", "p2", "p1", "p0"]


def test_shuffle_is_a_seeded_permutation(monkeypatch):
    monkeypatch.setenv("LLM_PASSAGE_ORDER", "shuffle:7")
    a = [h.id for h in ask_flow.order_passages(_hits())]
    assert sorted(a) == [f"p{i}" for i in range(6)] and a != [f"p{i}" for i in range(6)]
    assert a == [h.id for h in ask_flow.order_passages(_hits())]
    monkeypatch.setenv("LLM_PASSAGE_ORDER", "shuffle:8")
    assert [h.id for h in ask_flow.order_passages(_hits())] != a


def test_unknown_order_raises(monkeypatch):
    monkeypatch.setenv("LLM_PASSAGE_ORDER", "sideways")
    with pytest.raises(ValueError, match="LLM_PASSAGE_ORDER"):
        ask_flow.order_passages(_hits())


def test_prompt_markers_follow_the_new_order(monkeypatch):
    hits = _hits(3)
    monkeypatch.setattr(ask_flow, "_server_llm", lambda: object())
    monkeypatch.setattr(ask_flow, "_do_search", lambda req, retrieval_query=None: (hits, None, False))
    monkeypatch.setattr(ask_flow, "prepare_answer_context", lambda *a, **k: list(hits))
    monkeypatch.setattr(ask_flow, "_state", {})
    req = AskRequest(query="What did Buffett say about float?", expand_query=False)
    monkeypatch.setenv("LLM_PASSAGE_ORDER", "reverse")
    _, _, _, _, context, prompt, _ = ask_flow._prepare_ask(req)
    assert [h.id for h in context] == ["p2", "p1", "p0"]
    assert prompt.index("[1] (year=2002") < prompt.index("[3] (year=2000")
