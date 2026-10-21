"""finance-recall-75-r4 group 3: the superlative route serves the winner's tagged passages that the cross-encoder
scores highest against the question (FINANCE_SUPERLATIVE_RERANK)."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.generation import finance_answer as fa
from src.knowledge import KnowledgeStore
from src.query.plan import QueryPlan
from src.services import ask_flow
from src.storage import SearchHit, StoredDoc

SETTINGS = ("FINANCE_SUPERLATIVE_RERANK", "FINANCE_SUPERLATIVE_TAGGED", "FINANCE_K_CORPUS", "FINANCE_DIGEST_SOURCE",
            "FINANCE_STANCE_SOURCE")
PLAN = QueryPlan(query="q", kind="corpus_wide", superlative="sector:positive")
RETRIEVER_RERANKER = fa._retriever_reranker  # the autouse fixture below stubs it out for every test


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FINANCE_K_CORPUS", "0")  # the budget is the caller's max_hits
    monkeypatch.setenv("FINANCE_DIGEST_SOURCE", "0")
    monkeypatch.setenv("FINANCE_STANCE_SOURCE", "0")
    monkeypatch.setattr(fa, "_retriever_reranker", lambda: None)  # a test passes its own reranker


class FakeReranker:
    """Orders the candidates by a score table (unlisted ids score 0, ties keep the input order) and records the calls."""

    def __init__(self, scores=None):
        self.scores = scores or {}
        self.calls = []

    def rerank(self, query, candidates, top_k=8):
        self.calls.append((query, [c.id for c in candidates]))
        ranked = sorted(candidates, key=lambda c: -self.scores.get(c.id, 0.0))
        return [SearchHit(id=c.id, text=c.text, metadata=c.metadata, score=self.scores.get(c.id, 0.0))
                for c in ranked[:top_k]]


class BrokenReranker:
    def rerank(self, query, candidates, top_k=8):
        raise RuntimeError("model gone")


def _ann(pid, sectors=("insurance",), stance=(("insurance", "positive"),)):
    rows = [{"target": t, "polarity": p, "quote": f"quote {pid}"} for t, p in stance]
    return {"id": pid, "scope": "financial", "sectors": list(sectors), "companies": [], "stance": rows}


def _store(years=range(1980, 2000), extra=()):
    """Insurance has a positive stance in two passages of every letter in `years`; banking in two letters."""
    annotations = [_ann(f"{y}_p000{i}") for y in years for i in (1, 2)]
    annotations += [_ann(f"{y}_p0009", ("banking",), (("banking", "positive"),)) for y in (1990, 1991)]
    annotations += list(extra)
    taxonomy = {"sectors": {"insurance": "", "banking": ""}, "themes": {}}
    docs = {a["id"]: StoredDoc(id=a["id"], text=f"text {a['id']}", metadata={"year": int(a["id"][:4])})
            for a in annotations}
    return KnowledgeStore(annotations=annotations, taxonomy=taxonomy), docs


def _scores(*ids):
    """The ids score in the order given, best first; every other id scores 0."""
    return {pid: 1.0 - 0.01 * i for i, pid in enumerate(ids)}


FAVOURITES = ["1985_p0002", "1985_p0003", "1992_p0002", "1983_p0002", "1997_p0002", "1988_p0002", "1995_p0002"]


def _more():
    return [_ann("1985_p0003")]  # a third passage in 1985, the letter of the best passage


def _route(store, docs, reranker=None, max_hits=10, plan=PLAN):
    return fa.superlative_answer(plan, store, docs, max_hits=max_hits, reranker=reranker)


def _ids(answer):
    return [h.id for h in answer.hits]


# --- the selection ---
def test_spread_ranked_takes_one_passage_of_each_letter_before_a_second_one():
    docs = {}
    ranked = ["1990_p1", "1990_p2", "1991_p1", "1990_p3", "1992_p1", "1991_p2"]
    assert fa._spread_ranked(ranked, 3, docs) == ["1990_p1", "1991_p1", "1992_p1"]
    assert fa._spread_ranked(ranked, 5, docs) == ["1990_p1", "1991_p1", "1992_p1", "1990_p2", "1991_p2"]
    assert fa._spread_ranked(ranked, 99, docs) == ["1990_p1", "1991_p1", "1992_p1", "1990_p2", "1991_p2", "1990_p3"]


def test_spread_ranked_follows_the_rank_not_the_year():
    ranked = ["2010_p1", "1980_p1", "1995_p1", "1996_p1", "1997_p1"]
    assert fa._spread_ranked(ranked, 3, {}) == ["2010_p1", "1980_p1", "1995_p1"]
    assert fa._spread_ranked(ranked, 0, {}) == [] and fa._spread_ranked(ranked, -2, {}) == []


# --- off, and the cases that skip the setting ---
def test_off_by_default_the_reranker_is_never_called():
    store, docs = _store()
    reranker = FakeReranker(_scores(*FAVOURITES))
    answer = _route(store, docs, reranker)
    assert reranker.calls == [] and "rerank" not in answer.detail
    assert _ids(answer) == _ids(_route(store, docs))


def test_without_a_reranker_the_setting_is_skipped_with_a_note(monkeypatch):
    store, docs = _store()
    off = _route(store, docs)
    monkeypatch.setenv("FINANCE_SUPERLATIVE_RERANK", "4")
    on = _route(store, docs)  # no reranker passed, and the retriever has none
    assert _ids(on) == _ids(off) and on.answer == off.answer
    assert on.detail["rerank"] == "skipped: no reranker"


def test_a_failing_reranker_is_skipped_with_a_note(monkeypatch):
    store, docs = _store()
    off = _route(store, docs)
    monkeypatch.setenv("FINANCE_SUPERLATIVE_RERANK", "4")
    on = _route(store, docs, BrokenReranker())
    assert _ids(on) == _ids(off) and on.answer == off.answer
    assert on.detail["rerank"] == "skipped: RuntimeError"


def test_the_retrievers_reranker_is_the_default(monkeypatch):
    store, docs = _store()
    reranker = FakeReranker(_scores(*FAVOURITES))
    monkeypatch.setattr(fa, "_retriever_reranker", RETRIEVER_RERANKER)
    monkeypatch.setenv("FINANCE_SUPERLATIVE_RERANK", "4")
    monkeypatch.setattr(ask_flow, "_state", {"retriever": SimpleNamespace(reranker=reranker)})
    assert fa._retriever_reranker() is reranker
    assert _route(store, docs).detail["rerank"] == "4 served" and len(reranker.calls) == 1
    other = FakeReranker()
    _route(store, docs, other)  # a reranker passed in wins over the retriever's
    assert len(other.calls) == 1 and len(reranker.calls) == 1
    monkeypatch.setattr(ask_flow, "_state", {})
    assert fa._retriever_reranker() is None


# --- on ---
def test_on_serves_the_best_scored_passages_after_the_cited_ones_and_keeps_the_answer(monkeypatch):
    store, docs = _store(extra=_more())
    off = _route(store, docs)
    cited = off.detail["cited"]
    monkeypatch.setenv("FINANCE_SUPERLATIVE_RERANK", "4")
    on = _route(store, docs, FakeReranker(_scores(*FAVOURITES)))
    served = _ids(on)
    assert served[:cited] == _ids(off)[:cited] and len(served) == len(off.hits) == len(set(served))
    assert on.answer == off.answer and on.citations == off.citations and on.detail["cited"] == cited
    # best first; 1985_p0003 (the second passage of the best letter) waits for the next round
    assert served[cited:cited + 4] == ["1985_p0002", "1992_p0002", "1983_p0002", "1997_p0002"]
    assert on.detail["rerank"] == "4 served"


def test_the_candidates_are_the_winners_pool_without_the_cited_passages_capped_at_60(monkeypatch):
    store, docs = _store(years=range(1930, 2000))  # 70 letters of insurance stance passages
    off = _route(store, docs)
    cited = set(_ids(off)[:off.detail["cited"]])
    reranker = FakeReranker()
    monkeypatch.setenv("FINANCE_SUPERLATIVE_RERANK", "4")
    _route(store, docs, reranker)
    (_, candidates), = reranker.calls
    pool = fa.tagged_stance_ids(PLAN, store, docs, "sector", "insurance", "positive", cited, 10 ** 6)
    assert len(candidates) == fa.SUPERLATIVE_RERANK_POOL == 60
    assert set(candidates) <= set(pool) and not set(candidates) & cited
    assert not [pid for pid in candidates if pid.endswith("_p0009")]  # banking's stance rows are not the winner's


def test_the_candidates_are_scored_against_the_financial_question(monkeypatch):
    store, docs = _store()
    monkeypatch.setenv("FINANCE_SUPERLATIVE_RERANK", "2")
    reranker = FakeReranker()
    _route(store, docs, reranker)
    _route(store, docs, reranker, plan=replace(PLAN, query="which sector, and my dog?", scope="mixed",
                                                financial_query="which sector"))
    assert [q for q, _ in reranker.calls] == ["q", "which sector"]


def test_the_picks_take_their_places_from_the_served_budget(monkeypatch):
    store, docs = _store()
    cited = _route(store, docs).detail["cited"]
    monkeypatch.setenv("FINANCE_SUPERLATIVE_RERANK", "50")
    reranker = FakeReranker(_scores(*FAVOURITES))
    assert len(_route(store, docs, reranker, max_hits=cited + 3).hits) == cited + 3
    assert len(_route(store, docs, reranker, max_hits=cited).hits) == cited  # no room: nothing to score
    assert len(reranker.calls) == 1
    monkeypatch.setenv("FINANCE_K_CORPUS", str(cited + 5))  # the larger corpus budget wins over max_hits
    assert len(_route(store, docs, reranker, max_hits=cited).hits) == cited + 5


def test_tagged_passages_and_the_other_evidence_fill_what_the_picks_leave(monkeypatch):
    store, docs = _store(extra=_more())
    cited = _route(store, docs).detail["cited"]
    reranker = FakeReranker(_scores(*FAVOURITES))
    monkeypatch.setenv("FINANCE_SUPERLATIVE_RERANK", "2")
    monkeypatch.setenv("FINANCE_SUPERLATIVE_TAGGED", "2")
    served = _ids(_route(store, docs, reranker, max_hits=cited + 6))
    assert len(served) == cited + 6 == len(set(served))
    assert served[cited:cited + 2] == ["1985_p0002", "1992_p0002"]  # the picks come first
    tagged = fa.tagged_stance_ids(PLAN, store, docs, "sector", "insurance", "positive",
                                  set(served[:cited + 2]), 2)
    assert served[cited + 2:cited + 4] == tagged


def test_the_window_cut_keeps_the_picks_longest(monkeypatch):
    store, docs = _store(extra=_more())
    cited = _route(store, docs).detail["cited"]
    monkeypatch.setenv("FINANCE_SUPERLATIVE_RERANK", "6")
    monkeypatch.setenv("FINANCE_K_CORPUS", str(cited + 6))
    reranker = FakeReranker(_scores(*FAVOURITES))
    full = _ids(_route(store, docs, reranker, max_hits=cited))
    assert len(full) == cited + 6
    text = len(docs[full[0]].text)
    monkeypatch.setattr(fa, "reader_window_chars", lambda max_new_tokens: text * (cited + 2))
    cut = _ids(_route(store, docs, reranker, max_hits=cited))
    assert cut == full[:cited + 2] and cut[cited:] == ["1985_p0002", "1992_p0002"]
    monkeypatch.setattr(fa, "reader_window_chars", lambda max_new_tokens: 1)
    assert _ids(_route(store, docs, reranker, max_hits=cited)) == full[:cited]  # max_hits stays


def test_a_negative_superlative_scores_the_negative_pool(monkeypatch):
    annotations = [_ann(f"{y}_p000{i}", ("banking",), (("banking", "negative" if i == 1 else "positive"),))
                   for y in range(1980, 1990) for i in (1, 2)]
    docs = {a["id"]: StoredDoc(id=a["id"], text=f"text {a['id']}", metadata={"year": int(a["id"][:4])})
            for a in annotations}
    store = KnowledgeStore(annotations=annotations, taxonomy={"sectors": {"banking": ""}, "themes": {}})
    plan = QueryPlan(query="q", kind="corpus_wide", superlative="sector:negative")
    monkeypatch.setenv("FINANCE_SUPERLATIVE_RERANK", "3")
    reranker = FakeReranker()
    answer = _route(store, docs, reranker, plan=plan)
    (_, candidates), = reranker.calls
    assert candidates and all(pid.endswith("_p0001") for pid in candidates)
    assert answer.detail["rerank"] == "3 served"
