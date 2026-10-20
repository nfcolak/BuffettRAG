"""finance-recall-75 group 4: adaptive served budget, rerank depth, merged-id accounting, follow-up planning."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from config import FINANCE_DEFAULTS
from src.knowledge import KnowledgeStore
from src.retrieval.finance_policy import (finance_search, kind_budget, rerank_depth, reader_window_chars,
                                          served_budget)
from src.services import ask_flow, backend_app as backend
from src.services.schemas import AskRequest
from src.storage import SearchHit, StoredDoc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "eval"))
import finance_eval  # noqa: E402

from test_ask_flow_finance import FakeLLM, TREND_Q  # noqa: E402
from test_finance_policy import FakeRetriever, Recorder, _hit, _plan, _spread  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "knowledge_mini"
BUDGET_SETTINGS = ("FINANCE_K", "FINANCE_K_TREND", "FINANCE_K_CORPUS", "FINANCE_K_PERIOD", "FINANCE_RERANK_CANDIDATES",
                   "FINANCE_FOLLOWUP_HISTORY", "RERANK_CANDIDATES", "LLM_CONTEXT_PASSAGES", "LLM_N_CTX",
                   "LLM_PASSAGE_ORDER", "FINANCE_DEBUG")


@pytest.fixture(scope="module")
def store():
    return KnowledgeStore.load(FIX)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in BUDGET_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RETRIEVAL_DEDUP", "0")


# --- served budget by plan kind ---
def test_new_follow_up_setting_defaults_off():
    assert FINANCE_DEFAULTS["FINANCE_FOLLOWUP_HISTORY"] == 0


def test_served_budget_follows_the_plan_kind(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "8")
    kinds = ("trend", "corpus_wide", "period", "comparison", "specific", "numeric")
    assert [served_budget(_plan(k)) for k in kinds] == [8] * 6 and [kind_budget(_plan(k)) for k in kinds] == [0] * 6
    monkeypatch.setenv("FINANCE_K_TREND", "24")
    monkeypatch.setenv("FINANCE_K_CORPUS", "30")
    monkeypatch.setenv("FINANCE_K_PERIOD", "20")
    assert [served_budget(_plan(k)) for k in kinds] == [24, 30, 20, 20, 8, 8]
    monkeypatch.setenv("FINANCE_K", "12")  # FINANCE_K is the fallback of every kind without its own setting
    assert [served_budget(_plan(k)) for k in kinds] == [24, 30, 20, 20, 12, 12]
    assert [kind_budget(_plan(k)) for k in kinds] == [24, 30, 20, 20, 0, 0]


def test_finance_search_serves_the_kind_budget(store, monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "6")
    ranking = _spread(range(1980, 2000), per_letter=3)
    hits, debug = finance_search(_plan("trend"), FakeRetriever(ranking), store)
    assert len(hits) == debug["k"] == 6
    monkeypatch.setenv("FINANCE_K_TREND", "12")
    hits, debug = finance_search(_plan("trend"), FakeRetriever(ranking), store)
    assert len(hits) == debug["k"] == 12 and debug["fetch_k"] == 36
    assert len(finance_search(_plan("specific"), FakeRetriever(ranking), store)[0]) == 6  # other kinds keep theirs
    assert len(finance_search(_plan("trend"), FakeRetriever(ranking), store, k=3)[0]) == 3  # an explicit k wins


# --- reranker depth ---
def _window(store, monkeypatch=None):
    ranking = [_hit(f"19{60 + n // 10}_f{n:02d}", 1960 + n // 10) for n in range(90)]
    rec = Recorder()
    finance_search(_plan("specific"), FakeRetriever(ranking, reranker=rec), store, k=20)  # fetch_k 60
    return len(rec.windows[0])


def test_rerank_depth_is_todays_until_the_finance_setting_is_set(store, monkeypatch):
    assert rerank_depth() == 30 and _window(store) == 30
    monkeypatch.setenv("RERANK_CANDIDATES", "10")  # the global setting still governs the finance path
    assert rerank_depth() == 10 and _window(store) == 10
    monkeypatch.setenv("FINANCE_RERANK_CANDIDATES", "45")
    assert rerank_depth() == 45 and _window(store) == 45
    monkeypatch.setenv("FINANCE_RERANK_CANDIDATES", "30")  # explicitly 30 is honoured, not read as "unset"
    assert rerank_depth() == 30


def test_an_adopted_config_default_beats_the_global_setting(store, monkeypatch):
    monkeypatch.setenv("RERANK_CANDIDATES", "10")
    monkeypatch.setitem(FINANCE_DEFAULTS, "FINANCE_RERANK_CANDIDATES", 40)
    assert rerank_depth() == 40 and _window(store) == 40


# --- reader window ---
def test_reader_window_is_the_context_minus_the_answer_budget_at_four_characters_per_token(monkeypatch):
    assert reader_window_chars(900) == (16384 - 900) * 4
    monkeypatch.setenv("LLM_N_CTX", "8192")
    assert reader_window_chars(1000) == (8192 - 1000) * 4
    assert reader_window_chars(20000) == 0


def _long_docs(n=24, chars=1200):
    docs = {}
    for i in range(n):
        pid = f"{1980 + i}_p{i:04d}"
        text = " ".join(f"bonds{i}w{j}" for j in range(chars // 9))
        docs[pid] = StoredDoc(id=pid, text=text, metadata={"year": 1980 + i, "source_file": f"{1980 + i}.pdf"})
    return docs


def _hits_of(docs):
    return [SearchHit(pid, d.text, dict(d.metadata), 1.0 - n / 100) for n, (pid, d) in enumerate(docs.items())]


@pytest.fixture
def flow(monkeypatch):
    docs = _long_docs()
    retriever, llm = FakeRetriever(_hits_of(docs)), FakeLLM()
    monkeypatch.setattr(ask_flow, "_state", {"retriever": retriever, "llm": llm, "docs_by_id": docs,
                                              "knowledge": KnowledgeStore.load(FIX)})
    monkeypatch.setenv("FINANCE_PIPELINE", "1")
    return type("Flow", (), {"docs": docs, "retriever": retriever})


def _route(query=TREND_Q, **kw):
    return ask_flow.finance_route(AskRequest(query=query, **kw))


def test_default_budget_is_todays_and_leaves_no_window_trace(flow, monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "4")
    route = _route()
    assert route.plan.kind == "trend" and len(route.hits) == 4 and route.debug["k"] == 4
    assert "window_trimmed" not in route.debug


def test_a_trend_budget_serves_more_passages(flow, monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "4")
    monkeypatch.setenv("FINANCE_K_TREND", "12")
    route = _route()
    assert len(route.hits) == 12 and route.debug["k"] == 12 and route.debug["window_trimmed"] == 0
    years = [h.metadata["year"] for h in route.hits]
    assert years == sorted(years)  # still oldest first
    monkeypatch.setenv("FINANCE_K_TREND", "2")  # an explicit smaller budget is honoured
    assert len(_route().hits) == 2


def test_the_extra_passages_are_cut_to_fit_the_reader_window_lowest_rank_first(flow, monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "4")
    monkeypatch.setenv("FINANCE_K_TREND", "12")
    full = _route()
    assert len(full.hits) == 12
    monkeypatch.setenv("LLM_N_CTX", str(900 + len(full.prompt) // 4 - 50))  # the window is 200 characters short
    window = reader_window_chars(900)
    route = _route()
    assert 4 <= len(route.hits) < 12 and len(route.prompt) <= window < len(full.prompt)
    assert route.debug["window_chars"] == window and route.debug["window_trimmed"] == 12 - len(route.hits)
    kept = {h.id for h in route.hits}
    assert kept == {h.id for h in route.retrieved_hits[:len(route.hits)]}  # the tail of the ranking went first


def test_the_window_never_cuts_below_todays_budget(flow, monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "4")
    monkeypatch.setenv("FINANCE_K_TREND", "12")
    monkeypatch.setenv("LLM_N_CTX", "1000")  # a 400 character window: nothing fits
    route = _route()
    assert len(route.hits) == 4 and route.debug["window_trimmed"] == 8


def test_the_window_uses_the_requests_answer_budget(flow, monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "4")
    monkeypatch.setenv("FINANCE_K_TREND", "12")
    monkeypatch.setenv("LLM_N_CTX", "6000")
    small = _route(max_new_tokens=100)
    big = _route(max_new_tokens=2000)
    assert len(small.hits) > len(big.hits) >= 4
    assert small.debug["window_chars"] == (6000 - 100) * 4 and big.debug["window_chars"] == (6000 - 2000) * 4


def test_a_trend_budget_does_not_touch_other_plan_kinds(flow, monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "4")
    monkeypatch.setenv("FINANCE_K_TREND", "12")
    route = _route("What did Buffett write about bonds in 1990?")
    assert route.plan.kind != "trend" and len(route.hits) <= 4
    monkeypatch.setenv("FINANCE_K_PERIOD", "9")
    period = _route("What did Buffett write about bonds from 1982 to 1990?")
    assert period.plan.kind in ("period", "trend")
    assert len(period.hits) == (9 if period.plan.kind == "period" else 12)


def test_the_corpus_budget_keeps_more_of_a_digest_answers_passages(monkeypatch):
    from test_ask_flow_finance import DIGEST_Q, FakeRetriever as AskRetriever, TREND_IDS, _docs, _hits
    docs = _docs()
    monkeypatch.setattr(ask_flow, "_state", {"retriever": AskRetriever(_hits(TREND_IDS, docs)), "llm": FakeLLM(),
                                              "docs_by_id": docs, "knowledge": KnowledgeStore.load(FIX)})
    monkeypatch.setenv("FINANCE_PIPELINE", "1")
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "4")
    route = _route(DIGEST_Q)
    assert route.route == "digest" and len(route.hits) == 4  # three quoted passages and one gap passage
    monkeypatch.setenv("FINANCE_K_CORPUS", "6")
    route = _route(DIGEST_Q)
    assert route.route == "digest" and len(route.hits) == 6
    monkeypatch.setenv("FINANCE_K_CORPUS", "1")  # the quoted passages always stay
    assert len(_route(DIGEST_Q).hits) == route.debug["cited"] == 3


# --- merged-id accounting ---
def test_served_ids_lists_a_hits_own_id_then_its_merged_neighbours_without_repeats():
    merged = SearchHit("a", "t", {"merged_ids": ["a", "b", "c"]}, 1.0)
    plain = SearchHit("d", "t", {}, 1.0)
    again = SearchHit("b", "t", {"merged_ids": ["b", "a"]}, 1.0)
    assert ask_flow.served_ids([merged, plain, again]) == ["a", "b", "c", "d"]
    assert ask_flow.served_ids([]) == []


def _linked_docs():
    docs = {}
    for year in (1990, 1995, 2000):
        for n in range(5):
            pid = f"{year}_p{n:04d}"
            meta = {"year": year, "source_file": f"{year}.pdf"}
            if n:
                meta["previous_chunk_id"] = f"{year}_p{n - 1:04d}"
            if n < 4:
                meta["next_chunk_id"] = f"{year}_p{n + 1:04d}"
            docs[pid] = StoredDoc(id=pid, text=f"Passage {pid} discusses bond yields and rates.", metadata=meta)
    return docs


def test_the_api_reports_neighbour_merged_ids_in_served_ids(monkeypatch):
    docs = _linked_docs()
    anchors = [SearchHit(pid, docs[pid].text, dict(docs[pid].metadata), 1.0) for pid in ("1990_p0002", "2000_p0003")]
    monkeypatch.setattr(ask_flow, "_state", {"retriever": FakeRetriever(anchors), "llm": FakeLLM(),
                                              "docs_by_id": docs, "knowledge": KnowledgeStore.load(FIX)})
    monkeypatch.setattr(backend, "API_KEYS", ())
    monkeypatch.setenv("FINANCE_PIPELINE", "1")
    monkeypatch.setenv("FINANCE_DEBUG", "1")
    body = TestClient(backend.app).post("/ask", json={"query": TREND_Q, "expand_query": False}).json()
    assert [h["id"] for h in body["hits"]] == ["1990_p0002", "2000_p0003"]  # the hit list is unchanged
    assert body["debug"]["served_ids"] == ["1990_p0002", "1990_p0001", "1990_p0003",
                                           "2000_p0003", "2000_p0002", "2000_p0004"]
    route = _route(TREND_Q)
    assert route.served_ids == body["debug"]["served_ids"]


def test_the_eval_recorder_scores_the_debug_served_ids_when_the_api_sends_them():
    case = {"qid": "q1"}
    data = {"hits": [{"id": "a"}], "debug": {"served_ids": ["a", "b"]}}
    assert finance_eval.record(case, "dev", None, data, 1.0)["served_ids"] == ["a", "b"]
    assert finance_eval.record(case, "dev", None, {"hits": [{"id": "a"}]}, 1.0)["served_ids"] == ["a"]
    assert finance_eval.record(case, "dev", None, {"hits": [{"id": "a"}], "debug": {"served_ids": []}},
                               1.0)["served_ids"] == ["a"]


# --- follow-up planning ---
NOISY = ("what about then", "tent party")  # the scope vote sees mostly non-financial passages for these texts
NON_FINANCIAL = ("1985_p0013", "1990_p0119", "1994_p0103", "2001_p0111", "2001_p0119", "1996_p0126", "1984_p0029")
FINANCIAL = ("2010_p0119", "1990_p0100", "1977_p0026", "2003_p0087", "1986_p0118", "1996_p0039", "2008_p0011",
             "2008_p0067", "2008_p0093", "1990_p0101")
PREVIOUS = "How did Buffett's view of bonds change over time?"
HISTORY = [{"role": "user", "content": PREVIOUS}, {"role": "assistant", "content": "It shifted over the decades."}]


class VoteRetriever(FakeRetriever):
    """Ranks non-financial passages first for a noisy text, so the scope vote of that text says personal."""

    def __init__(self, docs):
        super().__init__([])
        self.docs = docs

    def _hybrid_for_filter(self, query, fetch_k, where):
        ids = [*NON_FINANCIAL, *FINANCIAL] if any(m in query.lower() for m in NOISY) else list(FINANCIAL)
        return [SearchHit(i, self.docs[i].text, dict(self.docs[i].metadata), 1.0 - n / 100)
                for n, i in enumerate(ids) if i in self.docs][:fetch_k]


@pytest.fixture
def follow(monkeypatch):
    from test_ask_flow_finance import _docs
    docs = _docs()
    assert set(NON_FINANCIAL) <= set(docs)
    monkeypatch.setattr(ask_flow, "_state", {"retriever": VoteRetriever(docs), "llm": FakeLLM(), "docs_by_id": docs,
                                              "knowledge": KnowledgeStore.load(FIX)})
    monkeypatch.setenv("FINANCE_PIPELINE", "1")
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "6")
    return docs


def test_a_short_follow_up_is_scope_replied_by_a_noisy_vote_while_the_setting_is_off(follow):
    route = _route("What about then?", history=HISTORY)
    assert route.route == "scope" and route.plan.scope_source == "vote"


def test_a_short_follow_up_to_a_financial_question_is_planned_on_that_question(follow, monkeypatch):
    monkeypatch.setenv("FINANCE_FOLLOWUP_HISTORY", "1")
    route = _route("What about then?", history=HISTORY)
    assert route.route == "reader" and route.plan.scope == "financial" and route.plan.scope_source == "history"
    assert route.debug["scope_source"] == "history" and route.plan.vote_share >= 0.6  # the vote that was overruled
    assert route.hits and not set(NON_FINANCIAL) & {h.id for h in route.hits}
    assert "bonds" in route.plan.query  # planned on the previous question plus the follow-up


def test_a_personal_rule_in_the_follow_up_still_gives_the_scope_reply(follow, monkeypatch):
    monkeypatch.setenv("FINANCE_FOLLOWUP_HISTORY", "1")
    route = _route("What about his diet then?", history=HISTORY)
    assert route.route == "scope" and route.plan.scope_source == "rules"


def test_a_self_contained_question_keeps_its_own_vote(follow, monkeypatch):
    monkeypatch.setenv("FINANCE_FOLLOWUP_HISTORY", "1")
    route = _route("Which magicians performed inside the tent party outside the store on Sunday?", history=HISTORY)
    assert route.route == "scope" and route.plan.scope_source == "vote"


def test_a_follow_up_to_a_question_that_was_not_financial_keeps_the_scope_reply(follow, monkeypatch):
    monkeypatch.setenv("FINANCE_FOLLOWUP_HISTORY", "1")
    history = [{"role": "user", "content": "Which magicians performed inside the tent party outside the store on Sunday?"},
               {"role": "assistant", "content": "I only answer questions about Buffett's financial views."}]
    route = _route("What about then?", history=history)
    assert route.route == "scope" and route.plan.scope_source == "vote"


def test_without_history_the_setting_changes_nothing(follow, monkeypatch):
    monkeypatch.setenv("FINANCE_FOLLOWUP_HISTORY", "1")
    assert _route("What about then?").route == "scope"
    ask_only_assistant = [{"role": "assistant", "content": "Hello."}]
    assert _route("What about then?", history=ask_only_assistant).route == "scope"


def test_a_financial_follow_up_is_unaffected_by_the_setting(follow, monkeypatch):
    before = _route("And how did bonds change in 2008?", history=HISTORY)
    monkeypatch.setenv("FINANCE_FOLLOWUP_HISTORY", "1")
    after = _route("And how did bonds change in 2008?", history=HISTORY)
    assert before.route == after.route and [h.id for h in before.hits] == [h.id for h in after.hits]
    assert before.plan == after.plan
