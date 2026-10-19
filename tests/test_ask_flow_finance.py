"""finance-query-pipeline group 5: /ask and /ask/stream route one planned question the same way (fake LLM)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.generation.finance_answer import SCOPE_REPLY, TREND_SENTENCE
from src.generation.prompt import REFUSAL_LINE
from src.knowledge import KnowledgeStore
from src.services import ask_flow, backend_app as backend
from src.storage import SearchHit, StoredDoc

FIX = Path(__file__).parent / "fixtures" / "knowledge_mini"
PERSONAL_Q = "How old was Mrs. B when she opened her carpet store?"
NUMERIC_Q = "What was Berkshire's book value per share in 1987?"
DIGEST_Q = "Across all the letters, what did Buffett say about airline stocks?"
TREND_Q = "How did Buffett's view of bonds change over time?"
TREND_IDS = ["2010_p0119", "1990_p0100", "1977_p0026", "2003_p0087", "1986_p0118", "1996_p0039"]


class FakeLLM:
    provider_name = "remote"
    model = "fake"

    def __init__(self, reply="Rates mattered [1] and then again [2]."):
        self.reply, self.prompts = reply, []

    def generate(self, prompt, max_new_tokens=None):
        self.prompts.append(prompt)
        return self.reply


class FakeRetriever:
    """Hybrid ranking = the given hits; no reranker, so finance_search keeps its own order."""

    def __init__(self, hits=()):
        self.ranking, self.reranker, self.calls = list(hits), None, []

    def _hybrid_for_filter(self, query, fetch_k, where):
        self.calls.append(("hybrid", where))
        return list(self.ranking)[:fetch_k]

    def _bm25_search(self, query, top_k, where=None):
        return []

    def search(self, *args, **kwargs):  # the pre-pipeline path
        self.calls.append(("search", None))
        raise AssertionError("the flag-on route must not call Retriever.search")


def _docs():
    rows = [json.loads(line) for line in (FIX / "passages.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: StoredDoc(id=r["id"], text=r["text"], metadata={"year": int(r["id"][:4]), "source_file": f"{r['id'][:4]}.pdf"})
            for r in rows}


def _hits(ids, docs):
    return [SearchHit(i, docs[i].text, dict(docs[i].metadata), 1.0 - n / 100) for n, i in enumerate(ids)]


@pytest.fixture
def env(monkeypatch):
    docs = _docs()
    llm, retriever = FakeLLM(), FakeRetriever(_hits(TREND_IDS, docs))
    monkeypatch.setattr(ask_flow, "_state", {
        "retriever": retriever, "llm": llm, "docs_by_id": docs, "knowledge": KnowledgeStore.load(FIX)})
    monkeypatch.setattr(backend, "API_KEYS", ())
    monkeypatch.setenv("FINANCE_PIPELINE", "1")
    monkeypatch.setenv("RETRIEVAL_DEDUP", "0")
    for name in ("LLM_CONTEXT_PASSAGES", "LLM_PASSAGE_ORDER", "PROMPT_VERSION", "LLM_MODEL_PATH"):
        monkeypatch.delenv(name, raising=False)
    return type("Env", (), {"llm": llm, "retriever": retriever, "docs": docs, "client": TestClient(backend.app)})


def _ask(env, query, **extra):
    return env.client.post("/ask", json={"query": query, "expand_query": False, **extra}).json()


def _stream(env, query, **extra):
    response = env.client.post("/ask/stream", json={"query": query, "expand_query": False, **extra})
    events = [(m.group(1), json.loads(m.group(2))) for m in re.finditer(r"event: (\w+)\ndata: (.*)\n", response.text)]
    return {name: data for name, data in events}, [name for name, _ in events]


def _resolves(answer, hits):
    marks = {int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer) for n in group.split(",")}
    return bool(marks) and all(1 <= n <= len(hits) for n in marks)


def test_personal_question_gets_the_scope_reply_on_both_endpoints_without_retrieval_or_llm(env):
    body = _ask(env, PERSONAL_Q)
    events, order = _stream(env, PERSONAL_Q)
    assert body["answer"] == events["done"]["answer"] == SCOPE_REPLY != REFUSAL_LINE
    assert body["hits"] == [] and body["retrieved_hits"] == [] and body["citations"] == events["done"]["citations"] == []
    assert events["meta"]["hits"] == [] and "generating" not in str(events.get("status"))
    assert env.retriever.calls == [] and env.llm.prompts == []


def test_numeric_question_is_answered_from_the_numbers_table_on_both_endpoints(env):
    body = _ask(env, NUMERIC_Q)
    events, _ = _stream(env, NUMERIC_Q)
    assert body["answer"] == events["done"]["answer"] and "$2,477.47" in body["answer"]
    assert [h["id"] for h in body["hits"]] == [h["id"] for h in events["meta"]["hits"]] == ["1987_p0000"]
    assert _resolves(body["answer"], body["hits"]) and body["citations"] == events["done"]["citations"]
    assert body["citations"][0]["years"] == [1987]
    assert env.llm.prompts == []
    assert env.retriever.calls == [("hybrid", None)] * 2  # only the scope vote of each request, no finance_search


def test_digest_question_is_answered_from_the_topic_digest_on_both_endpoints(env):
    body = _ask(env, DIGEST_Q)
    events, _ = _stream(env, DIGEST_Q)
    assert body["answer"] == events["done"]["answer"] and body["answer"] != REFUSAL_LINE
    assert body["answer"].startswith(ask_flow._state["knowledge"].digest("airlines_aviation")["overview"])
    ids = [h["id"] for h in body["hits"]]
    assert ids[:3] == ["1990_p0101", "1990_p0102", "1990_p0103"]
    assert ids == [h["id"] for h in events["meta"]["hits"]]
    assert _resolves(body["answer"], body["hits"]) and body["citations"] == events["done"]["citations"]
    assert [h["id"] for h in body["retrieved_hits"]] == [h["id"] for h in events["meta"]["retrieved_hits"]]
    assert env.llm.prompts == []


def test_stream_skips_the_evidence_gate_for_a_finished_answer(env, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("assess_evidence must not run for a finished answer")
    monkeypatch.setattr(backend, "assess_evidence", boom)
    monkeypatch.setattr(ask_flow, "_finalize_answer", boom)
    for query in (PERSONAL_Q, NUMERIC_Q, DIGEST_Q):
        events, _ = _stream(env, query)
        assert events["done"]["answer"] and events["done"]["answer"] != REFUSAL_LINE


def test_finished_answers_do_not_go_through_the_validator_or_expansion(env, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("validator / expand_answer must not run")
    monkeypatch.setattr(ask_flow, "validate_and_filter_answer", boom)
    monkeypatch.setattr(ask_flow, "expand_answer", boom)
    for query in (PERSONAL_Q, NUMERIC_Q, DIGEST_Q):
        assert _ask(env, query)["answer"]


def test_reader_route_serves_trend_passages_oldest_first_capped_for_the_remote_provider(env, monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "4")
    body = _ask(env, TREND_Q)
    years = [h["year"] for h in body["hits"]]
    assert len(body["hits"]) == 4 and years == sorted(years)
    assert len(body["retrieved_hits"]) > 4  # the cap is on the served list, not on retrieval
    prompt = env.llm.prompts[0]  # a refusal retry (soft gate) would re-ask with the plain prompt
    assert prompt.count(TREND_SENTENCE) == 1 and "BEGIN UNTRUSTED PASSAGES" in prompt
    assert prompt.index(f"[1] (year={years[0]}") < prompt.index(f"[4] (year={years[-1]}")
    assert all(c["passage_indices"] and max(c["passage_indices"]) < 4 for c in body["citations"])


def test_reader_route_on_the_stream_matches_ask(env, monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "4")
    body = _ask(env, TREND_Q)
    events, _ = _stream(env, TREND_Q)
    assert [h["id"] for h in events["meta"]["hits"]] == [h["id"] for h in body["hits"]]
    assert events["done"]["answer"] == body["answer"]


def test_numeric_question_without_a_matching_row_falls_back_to_the_reader(env):
    body = _ask(env, "How much was the holding cost of Washington Post in 1990?")  # the table has 1977 only
    assert body["hits"] and "as written" not in (body["answer"] or "")
    assert any(call[0] == "hybrid" for call in env.retriever.calls)


def test_mixed_question_reaches_the_reader_without_its_personal_clause(env):
    _ask(env, "How did Buffett's view of bonds change over time, and how old is he?")
    prompt = env.llm.prompts[0]
    assert "how old" not in prompt.lower() and "bonds" in prompt


def test_flag_off_keeps_the_old_path(env, monkeypatch):
    monkeypatch.delenv("FINANCE_PIPELINE", raising=False)
    from src.retrieval.retriever import RetrievalResult

    class Old(FakeRetriever):
        def search(self, *args, **kwargs):
            self.calls.append(("search", None))
            return RetrievalResult(PERSONAL_Q, "hybrid", [])
    old = Old()
    ask_flow._state["retriever"] = old
    body = _ask(env, PERSONAL_Q)
    assert body["answer"] is None and old.calls == [("search", None)]
    assert env.llm.prompts == []
