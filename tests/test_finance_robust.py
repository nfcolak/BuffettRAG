"""finance-recall-75-r2 group 1: candidate sources never fail a request, and the served budget of specific plans."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from config import FINANCE_DEFAULTS
from src.knowledge import KnowledgeStore
from src.retrieval import finance_policy, knowledge_source, summary_source
from src.retrieval.finance_policy import finance_search, kind_budget, served_budget, source_unavailable
from src.services import ask_flow
from src.services.schemas import AskRequest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ask_flow_finance import FakeLLM  # noqa: E402
from test_finance_budget import _hits_of, _long_docs  # noqa: E402
from test_finance_policy import FakeRetriever, _hit, _plan  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "knowledge_mini"
APP = Path(__file__).resolve().parent.parent / "deploy" / "modal" / "app.py"
SOURCE_SETTINGS = ("FINANCE_SUMMARY_SOURCE", "FINANCE_TAGGED_SOURCE", "FINANCE_DIGEST_SOURCE", "FINANCE_STANCE_SOURCE",
                   "FINANCE_MULTI_QUERY", "FINANCE_DIGEST_QUERY")  # the multi-query source is on by default since finance-recall-75-r3
SPECIFIC_Q = "What did Buffett say about bonds?"


@pytest.fixture(scope="module")
def store():
    return KnowledgeStore.load(FIX)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("FINANCE_K", "FINANCE_K_TREND", "FINANCE_K_CORPUS", "FINANCE_K_PERIOD", "FINANCE_K_SPECIFIC",
                 "LLM_CONTEXT_PASSAGES", "LLM_N_CTX", "KNOWLEDGE_DIR"):
        monkeypatch.delenv(name, raising=False)
    for name in SOURCE_SETTINGS:  # every source off; a test turns on the ones it checks
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("RETRIEVAL_DEDUP", "0")


def _register(monkeypatch, *sources, keep_summary=False):
    """Run only these sources (the module sources share the setting names and would run beside them)."""
    monkeypatch.setattr(finance_policy, "EXTRA_SOURCES", list(sources))
    monkeypatch.setattr(knowledge_source, "SOURCES", [])
    if not keep_summary:
        monkeypatch.setattr(summary_source, "SOURCES", [])


def _ranking(n=6):
    return [_hit(f"198{i}_a{i}") for i in range(n)]


# --- 1.1 a source that raises is skipped ---
def test_a_raising_source_is_skipped_and_recorded_and_the_others_still_serve(store, monkeypatch):
    def boom(plan, st, retriever, query, k):
        raise RuntimeError("index exploded")

    _register(monkeypatch, ("FINANCE_TAGGED_SOURCE", boom), ("FINANCE_DIGEST_SOURCE", lambda *a: [_hit("1999_extra")]))
    monkeypatch.setenv("FINANCE_TAGGED_SOURCE", "1")
    monkeypatch.setenv("FINANCE_DIGEST_SOURCE", "1")
    hits, debug = finance_search(_plan("specific"), FakeRetriever(_ranking()), store, k=4)
    assert [h.id for h in hits] == [h.id for h in _ranking()[:4]]  # served from the remaining candidates
    assert debug["source_errors"] == [{"source": "FINANCE_TAGGED_SOURCE", "error_type": "RuntimeError"}]
    assert debug["source_candidates"] == {"FINANCE_DIGEST_SOURCE": 1} and "1999_extra" in debug["stage_pool"]
    assert debug["source_skipped"] == []


def test_a_source_that_fails_while_its_hits_are_read_is_skipped_too(store, monkeypatch):
    def broken_stream(plan, st, retriever, query, k):
        yield _hit("1999_extra")
        raise ValueError("lost the connection")

    _register(monkeypatch, ("FINANCE_STANCE_SOURCE", broken_stream), ("FINANCE_TAGGED_SOURCE", lambda *a: None))
    monkeypatch.setenv("FINANCE_STANCE_SOURCE", "1")
    monkeypatch.setenv("FINANCE_TAGGED_SOURCE", "1")  # returns None: not iterable
    hits, debug = finance_search(_plan("specific"), FakeRetriever(_ranking()), store, k=3)
    assert len(hits) == 3 and "1999_extra" not in debug["stage_pool"]
    assert debug["source_errors"] == [{"source": "FINANCE_STANCE_SOURCE", "error_type": "ValueError"},
                                      {"source": "FINANCE_TAGGED_SOURCE", "error_type": "TypeError"}]
    assert debug["source_candidates"] == {}


def test_a_failing_source_leaves_the_search_as_if_it_were_off(store, monkeypatch):
    ranking = _ranking()
    off_hits, off = finance_search(_plan("specific"), FakeRetriever(ranking), store, k=4)
    _register(monkeypatch, ("FINANCE_TAGGED_SOURCE", lambda *a: 1 / 0))
    monkeypatch.setenv("FINANCE_TAGGED_SOURCE", "1")
    on_hits, on = finance_search(_plan("specific"), FakeRetriever(ranking), store, k=4)
    assert [h.id for h in on_hits] == [h.id for h in off_hits] and on["stage_pool"] == off["stage_pool"]
    assert off["source_errors"] == [] and on["source_errors"] == [{"source": "FINANCE_TAGGED_SOURCE",
                                                                   "error_type": "ZeroDivisionError"}]


# --- 1.1 a missing summary index is skipped with a note ---
def test_a_missing_summary_index_is_skipped_with_a_note(store, monkeypatch, tmp_path):
    monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path))  # a knowledge directory without summary_index/
    monkeypatch.setenv("FINANCE_SUMMARY_SOURCE", "1")
    assert finance_policy.candidate_sources() == summary_source.SOURCES
    ranking = _ranking()
    off_hits, _ = finance_search(_plan("specific"), FakeRetriever(ranking), store, k=4)
    hits, debug = finance_search(_plan("specific"), FakeRetriever(ranking), store, k=4)
    assert [h.id for h in hits] == [h.id for h in off_hits]  # still answered from the remaining candidates
    assert debug["source_errors"] == [] and debug["source_candidates"] == {}
    [skipped] = debug["source_skipped"]
    assert skipped["source"] == "FINANCE_SUMMARY_SOURCE" and "summary index missing" in skipped["note"]
    assert str(tmp_path) in skipped["note"]


def test_the_skip_note_names_the_missing_index_files(monkeypatch, tmp_path):
    monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path))
    fn = dict(summary_source.SOURCES)["FINANCE_SUMMARY_SOURCE"]
    note = source_unavailable("FINANCE_SUMMARY_SOURCE", fn)
    assert note and all(name in note for name in ("index_manifest.json", "index.faiss", "meta.json"))
    index = tmp_path / summary_source.INDEX_DIRNAME
    index.mkdir()
    (index / "index_manifest.json").write_text("{}")
    assert "index_manifest.json" not in source_unavailable("FINANCE_SUMMARY_SOURCE", fn)  # a partial index is missing too
    assert "index.faiss" in source_unavailable("FINANCE_SUMMARY_SOURCE", fn)
    (index / "index.faiss").write_bytes(b"")
    (index / "meta.json").write_text("{}")
    assert source_unavailable("FINANCE_SUMMARY_SOURCE", fn) is None  # present: the source runs (and verifies it itself)


def test_a_complete_index_directory_is_not_skipped(store, monkeypatch, tmp_path):
    index = tmp_path / summary_source.INDEX_DIRNAME
    index.mkdir()
    for name in ("index_manifest.json", "index.faiss", "meta.json"):
        (index / name).write_text("{}")
    monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path))
    monkeypatch.setenv("FINANCE_SUMMARY_SOURCE", "1")
    _, debug = finance_search(_plan("specific"), FakeRetriever(_ranking()), store, k=4)
    assert debug["source_skipped"] == []  # the fake retriever has no embedder, so the source itself returns nothing


def test_only_the_real_summary_source_is_checked_for_its_index(store, monkeypatch, tmp_path):
    monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path))  # no index here
    # a double with the same setting as the real summary source, beside it
    _register(monkeypatch, ("FINANCE_SUMMARY_SOURCE", lambda *a: [_hit("1999_extra")]), keep_summary=True)
    monkeypatch.setenv("FINANCE_SUMMARY_SOURCE", "1")
    _, debug = finance_search(_plan("specific"), FakeRetriever(_ranking()), store, k=4)
    assert [s["source"] for s in debug["source_skipped"]] == ["FINANCE_SUMMARY_SOURCE"]  # the real one: no index
    assert debug["source_candidates"] == {"FINANCE_SUMMARY_SOURCE": 1} and "1999_extra" in debug["stage_pool"]


def test_every_search_reports_empty_source_debug_when_no_source_runs(store):
    _, debug = finance_search(_plan("specific"), FakeRetriever(_ranking()), store, k=4)
    assert debug["source_errors"] == [] and debug["source_skipped"] == [] and debug["source_candidates"] == {}


# --- 1.2 FINANCE_K_SPECIFIC ---
def test_the_specific_budget_defaults_to_the_adopted_value():
    assert FINANCE_DEFAULTS["FINANCE_K_SPECIFIC"] == 80  # finance-recall-75-r4, task 4.1 (32 in round 3, 21 in round 2)


def test_the_specific_budget_is_read_by_specific_and_numeric_plans_only(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "8")
    monkeypatch.setenv("FINANCE_K_CORPUS", "0")    # corpus_wide: the kind without a budget of its own here
    monkeypatch.setenv("FINANCE_K_SPECIFIC", "0")  # off
    kinds = ("trend", "corpus_wide", "period", "comparison", "specific", "numeric")
    # period and comparison plans: FINANCE_K_PERIOD, 40 since finance-recall-75-r3; trend plans 80 since finance-recall-75-r4
    assert [kind_budget(_plan(k)) for k in kinds] == [80, 0, 40, 40, 0, 0]  # off: today's budget
    monkeypatch.setenv("FINANCE_K_SPECIFIC", "24")
    assert [kind_budget(_plan(k)) for k in kinds] == [80, 0, 40, 40, 24, 24]
    assert [served_budget(_plan(k)) for k in kinds] == [80, 8, 40, 40, 24, 24]
    monkeypatch.setenv("FINANCE_K", "12")  # FINANCE_K stays the fallback of a kind without its own setting
    assert [served_budget(_plan(k)) for k in kinds] == [80, 12, 40, 40, 24, 24]
    monkeypatch.setenv("FINANCE_K_SPECIFIC", "0")
    assert [served_budget(_plan(k)) for k in kinds] == [80, 12, 40, 40, 12, 12]


def test_finance_search_serves_the_specific_budget(store, monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "6")
    monkeypatch.setenv("FINANCE_K_SPECIFIC", "0")  # off: today's budget
    ranking = [_hit(f"19{80 + n // 4}_s{n:02d}") for n in range(60)]
    hits, debug = finance_search(_plan("specific"), FakeRetriever(ranking), store)
    assert len(hits) == debug["k"] == 6
    for budget in (16, 24, 32):
        monkeypatch.setenv("FINANCE_K_SPECIFIC", str(budget))
        hits, debug = finance_search(_plan("specific"), FakeRetriever(ranking), store)
        assert len(hits) == debug["k"] == budget
    assert len(finance_search(_plan("specific"), FakeRetriever(ranking), store, k=3)[0]) == 3  # an explicit k wins
    monkeypatch.setenv("FINANCE_K_TREND", "0")
    assert len(finance_search(_plan("trend"), FakeRetriever(ranking), store)[0]) == 6  # other kinds keep theirs


@pytest.fixture
def flow(monkeypatch):
    docs = _long_docs()
    monkeypatch.setattr(ask_flow, "_state", {"retriever": FakeRetriever(_hits_of(docs)), "llm": FakeLLM(),
                                              "docs_by_id": docs, "knowledge": KnowledgeStore.load(FIX)})
    monkeypatch.setenv("FINANCE_PIPELINE", "1")
    return type("Flow", (), {"docs": docs})


def _route(query=SPECIFIC_Q, **kw):
    return ask_flow.finance_route(AskRequest(query=query, **kw))


def test_a_specific_route_serves_the_specific_budget_and_leaves_todays_without_it(flow, monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "4")
    monkeypatch.setenv("FINANCE_K_SPECIFIC", "0")  # off: today's budget
    route = _route()
    assert route.plan.kind == "specific" and len(route.hits) == route.debug["k"] == 4
    assert "window_trimmed" not in route.debug
    monkeypatch.setenv("FINANCE_K_SPECIFIC", "12")
    route = _route()
    assert len(route.hits) == route.debug["k"] == 12 and route.debug["window_trimmed"] == 0


def test_the_specific_budget_is_capped_by_the_reader_window_lowest_rank_first(flow, monkeypatch):
    monkeypatch.setenv("FINANCE_WINDOW_TOKENS", "0")  # the character window (the token window: test_finance_window_tokens.py)
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "4")
    monkeypatch.setenv("FINANCE_K_SPECIFIC", "12")
    full = _route()
    assert len(full.hits) == 12
    monkeypatch.setenv("LLM_N_CTX", str(900 + len(full.prompt) // 4 - 50))  # the window is 200 characters short
    route = _route()
    window = finance_policy.reader_window_chars(900)
    assert 4 <= len(route.hits) < 12 and len(route.prompt) <= window < len(full.prompt)
    assert route.debug["window_chars"] == window and route.debug["window_trimmed"] == 12 - len(route.hits)
    assert {h.id for h in route.hits} == {h.id for h in route.retrieved_hits[:len(route.hits)]}
    monkeypatch.setenv("LLM_N_CTX", "1000")  # nothing fits: never below today's budget
    assert len(_route().hits) == 4


# --- 1.1 deploy-time warning (deploy/modal/app.py cannot be imported without modal: run its function) ---
def _warn(tmp_path, config_text, capsys, make_index=False):
    tree = ast.parse(APP.read_text())
    wanted = {"_shipped_finance_default", "_warn_missing_summary_index"}
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted],
                        type_ignores=[])
    (tmp_path / "config.py").write_text(config_text)
    if make_index:
        (tmp_path / "data/knowledge/summary_index").mkdir(parents=True)
        (tmp_path / "data/knowledge/summary_index/index_manifest.json").write_text("{}")
    namespace = {"ast": ast, "REPO": tmp_path}
    exec(compile(module, str(APP), "exec"), namespace)
    namespace["_warn_missing_summary_index"]()
    return capsys.readouterr().out


ON = 'FINANCE_DEFAULTS: dict = {\n    "FINANCE_SUMMARY_SOURCE": 1,\n    "FINANCE_K": 0,\n}\n'
OFF = ON.replace('"FINANCE_SUMMARY_SOURCE": 1', '"FINANCE_SUMMARY_SOURCE": 0')


def test_the_deploy_warns_when_the_summary_source_is_on_and_the_index_is_missing(tmp_path, capsys):
    out = _warn(tmp_path, ON, capsys)
    assert out.startswith("WARNING:") and "FINANCE_SUMMARY_SOURCE" in out and "data/knowledge/summary_index/" in out


def test_the_deploy_is_silent_with_the_index_or_with_the_source_off(tmp_path, capsys):
    with_index, off = tmp_path / "with_index", tmp_path / "off"
    with_index.mkdir(), off.mkdir()
    assert _warn(with_index, ON, capsys, make_index=True) == ""
    assert _warn(off, OFF, capsys) == ""


def test_the_deploy_check_never_fails_the_deploy(tmp_path, capsys):
    out = _warn(tmp_path, "this is not python (", capsys)  # an unreadable config only prints a warning
    assert out.startswith("WARNING: could not check the summary index")


def test_the_deploy_check_runs_only_at_deploy_time_and_the_real_config_ships_the_source_on():
    text = APP.read_text()
    assert "if modal.is_local():\n    _warn_missing_summary_index()" in text
    assert text.rindex("_warn_missing_summary_index()") > text.index('image.add_local_dir(str(REPO / "data/knowledge")')
    assert FINANCE_DEFAULTS["FINANCE_SUMMARY_SOURCE"] == 1
