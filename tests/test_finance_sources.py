"""finance-recall-75 group 1: candidate-source registry, the new settings, the loss-stage classifier."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from config import FINANCE_DEFAULTS, finance_setting
from src.knowledge import KnowledgeStore
from src.query.plan import QueryPlan
from src.retrieval import finance_policy, knowledge_source, summary_source
from src.retrieval.finance_policy import candidate_sources, finance_search
from src.storage import SearchHit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "eval"))
import finance_route_debug as frd  # noqa: E402

from test_finance_policy import FakeRetriever, _hit, _plan  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "knowledge_mini"
# the defaults group 5 adopted (finance-recall-75, task 5.1), K_CORPUS 32 since finance-recall-75-r2 (task 4.1),
# K_PERIOD 40 and one multi-query search since finance-recall-75-r3 (task 4.1); the rest stay off or at today's value
NEW_SETTINGS = {"FINANCE_SUMMARY_SOURCE": 1, "FINANCE_TAGGED_SOURCE": 1, "FINANCE_DIGEST_SOURCE": 1,
                "FINANCE_STANCE_SOURCE": 0, "FINANCE_K_TREND": 32, "FINANCE_K_CORPUS": 32, "FINANCE_K_PERIOD": 40,
                "FINANCE_RERANK_CANDIDATES": 30, "FINANCE_MULTI_QUERY": 1, "FINANCE_DIGEST_QUERY": 0}
SOURCE_SETTINGS = ("FINANCE_SUMMARY_SOURCE", "FINANCE_TAGGED_SOURCE", "FINANCE_DIGEST_SOURCE", "FINANCE_STANCE_SOURCE",
                   "FINANCE_MULTI_QUERY", "FINANCE_DIGEST_QUERY")  # the multi-query source is on by default since finance-recall-75-r3


@pytest.fixture(scope="module")
def store():
    return KnowledgeStore.load(FIX)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (*NEW_SETTINGS, "FINANCE_K", "LLM_CONTEXT_PASSAGES"):
        monkeypatch.delenv(name, raising=False)
    for name in SOURCE_SETTINGS:  # the registry tests start with every source off and turn on the one they check
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("RETRIEVAL_DEDUP", "0")


def test_new_settings_default_to_the_adopted_values(monkeypatch):
    for name in SOURCE_SETTINGS:
        monkeypatch.delenv(name)
    for name, value in NEW_SETTINGS.items():
        assert FINANCE_DEFAULTS[name] == value and finance_setting(name) == value
    monkeypatch.setenv("FINANCE_SUMMARY_SOURCE", "0")
    assert finance_setting("FINANCE_SUMMARY_SOURCE") == 0


def test_registry_lists_the_module_sources_and_runs_none_while_their_settings_are_off():
    assert finance_policy.registered_sources() == [*summary_source.SOURCES, *knowledge_source.SOURCES]
    assert candidate_sources() == []


def _register(monkeypatch, name, fn):
    monkeypatch.setattr(finance_policy, "EXTRA_SOURCES", [(name, fn)])


def test_source_runs_only_when_its_setting_is_on(store, monkeypatch):
    calls = []

    def source(plan, st, retriever, query, k):
        calls.append((plan, query, k))
        return [_hit("1999_extra")]

    _register(monkeypatch, "FINANCE_SUMMARY_SOURCE", source)
    ranking = [_hit(f"198{n}_a") for n in range(5)]
    hits, debug = finance_search(_plan("specific"), FakeRetriever(ranking), store, k=4)
    assert calls == [] and debug["source_candidates"] == {} and "1999_extra" not in [h.id for h in hits]
    monkeypatch.setenv("FINANCE_SUMMARY_SOURCE", "1")
    hits, debug = finance_search(_plan("specific"), FakeRetriever(ranking), store, k=4)
    assert len(calls) == 1 and calls[0][1:] == ("q", 4) and debug["source_candidates"] == {"FINANCE_SUMMARY_SOURCE": 1}
    assert "1999_extra" in debug["stage_pool"]


def test_source_hits_join_the_rerank_window_and_can_win(store, monkeypatch):
    class Prefer:
        def __init__(self):
            self.seen = []

        def rerank(self, query, hits, top_k):
            self.seen = [h.id for h in hits]
            return sorted(hits, key=lambda h: h.id != "1999_extra")[:top_k]

    _register(monkeypatch, "FINANCE_TAGGED_SOURCE", lambda *a: [_hit("1999_extra"), _hit("1980_a0")])
    monkeypatch.setenv("FINANCE_TAGGED_SOURCE", "1")
    rr = Prefer()
    ranking = [_hit(f"198{n}_a{n}") for n in range(6)]
    hits, debug = finance_search(_plan("specific"), FakeRetriever(ranking, reranker=rr), store, k=3)
    assert "1999_extra" in rr.seen and hits[0].id == "1999_extra"
    assert debug["source_candidates"] == {"FINANCE_TAGGED_SOURCE": 1}  # the already-ranked 1980_a0 is not new
    assert debug["stage_window"] == rr.seen and debug["stage_ranked"][0] == "1999_extra"


def test_nothing_registered_leaves_the_search_unchanged(store):
    ranking = [_hit(f"198{n}_a{n}") for n in range(6)]
    hits, debug = finance_search(_plan("specific"), FakeRetriever(ranking), store, k=4)
    assert [h.id for h in hits] == [h.id for h in ranking[:4]] and debug["source_candidates"] == {}


def test_stage_debug_lists_pool_window_and_ranked_order(store):
    ranking = [_hit(f"198{n}_a{n}") for n in range(6)]
    _, debug = finance_search(_plan("specific"), FakeRetriever(ranking), store, k=4)
    assert debug["stage_pool"] == [h.id for h in ranking] == debug["stage_window"] == debug["stage_ranked"]


def _dbg(**kw):
    base = {"stage_pool": ["a", "b", "c", "d", "e"], "stage_window": ["a", "b", "c", "d"],
            "stage_ranked": ["a", "b", "c", "d"], "dropped_not_financial": ["e"], "k": 2, "pinned": []}
    return {**base, **kw}


@pytest.mark.parametrize("pid,expected", [
    ("zzz", "not_in_pool"), ("e", "scope_filtered"), ("c", "budget_cut"), ("b", "letter_cap"),
])
def test_loss_stage_classifier(pid, expected):
    assert frd.classify_loss(pid, _dbg(), served=["a"], retrieved=["a", "b"][:1]) == expected


def test_loss_stage_rerank_cut_and_other():
    assert frd.classify_loss("d", _dbg(stage_window=["a", "b", "c"]), ["a"], ["a"]) == "rerank_cut"
    assert frd.classify_loss("b", _dbg(), ["a"], ["a", "b"]) == "other"          # selected, cut after the search
    assert frd.classify_loss("d", _dbg(stage_ranked=["a", "b", "c"]), ["a"], ["a"]) == "other"  # deduplicated
    assert frd.classify_loss("x", {}, [], []) == "other"                          # route without a search
    assert frd.classify_loss("c", _dbg(pinned=["p"]), ["a"], ["a"]) == "budget_cut"  # k=2 minus 1 pin: only rank 0


def test_loss_table_counts_per_group():
    rows = [{"group": 1, "gold": 3, "loss": {"x": "not_in_pool", "y": "budget_cut"}},
            {"group": 1, "gold": 1, "loss": {}}, {"group": 2, "gold": 2, "loss": {"z": "not_in_pool"}}]
    lines = frd.loss_table(rows, [1, 2])
    assert lines[1].split()[:3] == ["ALL", "6", "3"] and lines[2].split()[:3] == ["1", "crises/macro", "4"]
    assert [int(v) for v in lines[1].split()[3:]] == [2, 0, 0, 1, 0, 0]


def test_heldout_v6_cases_merge_a_and_b_and_group_by_type():
    cases = frd.load_cases("heldout_v6")
    assert len(cases) == 95 and all(c["group"] == c["question_type"] for c in cases)
