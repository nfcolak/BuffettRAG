"""/ask and /ask/stream take the same evidence-gate decision, and chat history cannot lower the gate score.

Live bug (2026-10-26): with chat history, /ask/stream refused "What is the main event happened in the year 2008?"
before calling the model (overlap 0.25, below the 0.30 bar) while /ask sent the same question to the model (the soft
band 0.15-0.30). The history terms ("inflation", "purchasing", "power") had pulled the score down from 0.50.
"""
import json

from fastapi.testclient import TestClient

from src.generation.evidence_gate import assess_evidence
from src.retrieval.retriever import RetrievalResult
from src.storage import SearchHit

PASSAGE = "Berkshire bought preferred shares of Goldman Sachs during the credit crisis."
SOFT_QUERY = "preferred shares outlook dividends valuation risks management"  # 2 of 7 terms: inside the soft band
HISTORY = ["Buffett on inflation and purchasing power"]


def _hit(text=PASSAGE):
    return SearchHit("2008_p0001", text, {"year": 2008, "source_file": "2008.pdf"}, 1.0)


def test_the_soft_query_sits_in_the_soft_band():
    overlap = assess_evidence(SOFT_QUERY, [_hit()]).best_overlap
    assert 0.15 <= overlap < 0.30


def test_stream_sends_a_soft_band_question_to_the_model_like_ask(monkeypatch):
    from src.services import ask_flow, backend_app as backend

    calls = []

    class FixedRetriever:
        def search(self, **kwargs):
            return RetrievalResult(kwargs["query"], "hybrid", [_hit()])

    class CaptureLLM:
        def generate(self, *_args, **_kwargs):
            calls.append("ask")
            return "Berkshire bought preferred shares of Goldman Sachs during the credit crisis. [1]"

    monkeypatch.setattr(ask_flow, "_state", {"retriever": FixedRetriever(), "docs_by_id": {}, "llm": CaptureLLM()})
    monkeypatch.setattr(backend, "API_KEYS", ())
    monkeypatch.delenv("FINANCE_PIPELINE", raising=False)
    monkeypatch.delenv("EVIDENCE_GATE_SOFT", raising=False)
    client = TestClient(backend.app)
    payload = {"query": SOFT_QUERY, "expand_query": False}
    client.post("/ask", json=payload)
    assert calls == ["ask"]
    stream = client.post("/ask/stream", json=payload)
    assert stream.status_code == 200
    assert len(calls) == 2, "the stream refused a question that /ask sends to the model"
    events = [json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: ")]
    assert {"stage": "generating"} in events


def test_stream_and_ask_share_one_gate_decision(monkeypatch):
    from src.services import ask_flow

    monkeypatch.delenv("EVIDENCE_GATE_SOFT", raising=False)
    hits = [_hit()]
    assert ask_flow.evidence_decision(SOFT_QUERY, hits).allowed
    assert ask_flow.evidence_decision(SOFT_QUERY, hits).soft
    assert not ask_flow.evidence_decision("moon cheese", hits).allowed
    monkeypatch.setenv("EVIDENCE_GATE_SOFT", "0")
    assert not ask_flow.evidence_decision(SOFT_QUERY, hits).allowed


def test_history_terms_cannot_lower_the_score_with_query_alone():
    hits = [_hit()]
    query = "preferred shares Goldman crisis"
    alone = assess_evidence(query, hits).best_overlap
    diluted = assess_evidence(query, hits, extra_queries=HISTORY).best_overlap
    assert diluted < alone  # today's behaviour: the history drags the score down
    kept = assess_evidence(query, hits, extra_queries=HISTORY, query_alone=True).best_overlap
    assert kept == alone


def test_query_alone_still_lets_history_help_a_short_follow_up():
    hits = [_hit()]
    follow_up = "and the dividend?"
    helped = assess_evidence(follow_up, hits, extra_queries=["Goldman Sachs preferred shares"], query_alone=True)
    assert helped.best_overlap >= assess_evidence(follow_up, hits).best_overlap
    assert helped.sufficient


def test_query_alone_is_read_from_the_finance_setting(monkeypatch):
    from src.services import ask_flow

    hits = [_hit()]
    query = "preferred shares Goldman crisis"
    monkeypatch.delenv("FINANCE_GATE_QUERY_ALONE", raising=False)
    off = ask_flow.evidence_decision(query, hits, HISTORY).evidence.best_overlap
    monkeypatch.setenv("FINANCE_GATE_QUERY_ALONE", "1")
    on = ask_flow.evidence_decision(query, hits, HISTORY).evidence.best_overlap
    assert on > off and on == assess_evidence(query, hits).best_overlap
