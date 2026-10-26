"""FINANCE_FOLLOWUP_OWN_TOPIC: a new question with its own topic is not joined to the previous question.

Live refusal (2026-10-26): "What did Buffet said about the oil/petroleum crisis?" asked after a 2017 question was
planned on "... oil/petroleum crisis? important point buffett discussed 2017", got the year 2017 and was served only
2017 passages; the model then (correctly) found no oil-crisis evidence.
"""
import pytest

from src.services import ask_flow

H_2017 = [{"role": "user", "content": "What is the most important point that Buffett discussed at 2017?"},
          {"role": "assistant", "content": "A major portion of the 2017 gain came from the tax code change [1]."}]


def _plan(query, history, monkeypatch, on):
    from src.retrieval.query_expansion import build_followup_retrieval_query
    if on:
        monkeypatch.setenv("FINANCE_FOLLOWUP_OWN_TOPIC", "1")
    else:
        monkeypatch.delenv("FINANCE_FOLLOWUP_OWN_TOPIC", raising=False)
    store = ask_flow._knowledge_store()
    resolved = build_followup_retrieval_query(query, history)
    return ask_flow._plan_with_history(query, resolved, history, store, lambda _t: 0.0)


def test_off_reproduces_the_live_bug(monkeypatch):
    plan = _plan("What did Buffet said about the oil/petroleum crisis?", H_2017, monkeypatch, on=False)
    assert 2017 in plan.years


def test_on_a_new_topic_question_drops_the_previous_year(monkeypatch):
    plan = _plan("What did Buffet said about the oil/petroleum crisis?", H_2017, monkeypatch, on=True)
    assert plan.years == () and plan.year_range is None
    assert "oil_gas" in plan.sectors
    assert "2017" not in plan.query


@pytest.mark.parametrize("follow_up", ["and in 2009?", "what about after that?", "What did he say about derivatives?",
                                       "Tell me more about it."])
def test_on_real_follow_ups_still_use_the_history(monkeypatch, follow_up):
    on = _plan(follow_up, H_2017, monkeypatch, on=True)
    off = _plan(follow_up, H_2017, monkeypatch, on=False)
    assert on == off


def test_on_without_history_changes_nothing(monkeypatch):
    q = "What did Buffett say about the oil crisis?"
    assert _plan(q, [], monkeypatch, on=True) == _plan(q, [], monkeypatch, on=False)
