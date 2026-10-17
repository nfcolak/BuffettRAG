"""pv8-retrieval-validator group 4: served-recall harness scopes, baseline modes, arms file, training fit."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("served_context_recall", ROOT / "scripts/eval/served_context_recall.py")
harness = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(harness)

ARMS = ROOT / "openspec/changes/pv8-retrieval-validator/arms.txt"
TEMPORAL = "Compare Berkshire's insurance float in 1990 and 2000"
YEAR_ONLY = "What did Buffett say about the Dow in 1995?"
NO_YEAR = "How does Buffett think about insurance float?"
V2_ONLY = "What did Berkshire earn in 1990 and in 1996?"  # routed only by COMPARE_ROUTING_V2 ("in A and in B")


def _scope(**overrides):
    return harness._scope_predicate({k: str(v) for k, v in overrides.items()})


# ----------------------------------------------------------------------------- scope functions
@pytest.mark.parametrize("key,value", [("RETRIEVAL_PERIOD_POOL", 30), ("RETRIEVAL_PERIOD_FOCUS_RERANK", 1)])
def test_period_keys_scope_to_temporal_comparisons(key, value):
    scope = _scope(**{key: value})
    assert scope(TEMPORAL, []) is True
    assert scope(YEAR_ONLY, []) is False
    assert scope(NO_YEAR, []) is False


def test_route2_scope_covers_questions_either_routing_sends_to_comparison(monkeypatch):
    from src.retrieval.retriever import detect_temporal_comparison

    monkeypatch.setenv("COMPARE_ROUTING_V2", "1")
    assert detect_temporal_comparison(V2_ONLY)
    monkeypatch.setenv("COMPARE_ROUTING_V2", "0")
    assert not detect_temporal_comparison(V2_ONLY)  # precondition: only the V2 rules route it
    scope = _scope(COMPARE_ROUTING_V2=1)
    assert scope(V2_ONLY, []) is True   # newly routed by V2
    assert scope(TEMPORAL, []) is True  # routed by both, answer path may still differ
    assert scope(NO_YEAR, []) is False
    import os
    assert os.environ["COMPARE_ROUTING_V2"] == "0"  # the probe restores the environment


@pytest.mark.parametrize("key,value", [("RETRIEVAL_YEAR_FILTER_KEEP", 4), ("RETRIEVAL_YEAR_REF_SKIP", 1)])
def test_year_keys_scope_to_questions_with_a_year(key, value):
    scope = _scope(**{key: value})
    assert scope(YEAR_ONLY, []) is True
    assert scope(TEMPORAL, []) is True
    assert scope(NO_YEAR, []) is False


def test_other_keys_scope_everything_and_defaults_are_skipped():
    assert _scope(ANSWER_CONTEXT_NEIGHBORS=2)(NO_YEAR, []) is True
    assert _scope(LLM_PASSAGE_MAX_CHARS=3000)(NO_YEAR, []) is True
    # a value equal to the config default changes nothing, so it adds no scope (all cases reusable only when no
    # other key is non-default: an empty scope list is "everything", as before)
    assert _scope(RETRIEVAL_PERIOD_POOL=0, RETRIEVAL_YEAR_REF_SKIP=1)(NO_YEAR, []) is False


@pytest.mark.parametrize("key", ["VALIDATOR_SPLIT_V2", "VALIDATOR_VERBATIM_KEEP"])
def test_validator_keys_rejected_with_clear_error(key):
    with pytest.raises(ValueError, match="validator"):
        harness.parse_config(f"x:{key}=1")
    with pytest.raises(ValueError, match=key):
        harness._scope_predicate({key: "1"})
    with pytest.raises(ValueError, match="validator"):
        harness.parse_config(f"x:{key}=0")  # even a default value: the key itself is not a harness key


# ----------------------------------------------------------------------------- baseline modes
def test_legacy_baseline_is_default_and_pins_old_defaults():
    name, overrides = harness.parse_config("baseline")
    assert name == "baseline" and overrides == harness.BASELINE_ENV
    assert harness.parse_config("baseline", baseline="legacy") == (name, overrides)
    assert harness.parse_config("big:LLM_N_CTX=16384,RERANK_CANDIDATES=30")[1]["MAX_NEW_TOKENS"] == "1024"


def test_current_baseline_pins_nothing_and_keeps_answer_room():
    name, overrides = harness.parse_config("baseline", baseline="current")
    assert not set(overrides) & set(harness.BASELINE_ENV)
    assert overrides == {"MAX_NEW_TOKENS": str(harness.LONG_CTX_ANSWER_TOKENS)}  # 16384 window: 1024 tokens free
    _, nb2 = harness.parse_config("nb2:ANSWER_CONTEXT_NEIGHBORS=2,LLM_PASSAGE_MAX_CHARS=3000", baseline="current")
    assert nb2["ANSWER_CONTEXT_NEIGHBORS"] == "2" and "LLM_N_CTX" not in nb2


def test_is_default_follows_the_baseline_mode():
    assert harness._is_default("RERANK_CANDIDATES", "15") is True                              # legacy pin
    assert harness._is_default("RERANK_CANDIDATES", "30") is False
    assert harness._is_default("RERANK_CANDIDATES", "30", pins={}) is True                     # today's default
    assert harness._is_default("RERANK_CANDIDATES", "15", pins={}) is False


def test_serve_one_reports_n_ctx_of_the_window_in_force(monkeypatch):
    monkeypatch.delenv("LLM_N_CTX", raising=False)

    class Flow:
        class AskRequest:
            def __init__(self, **kw):
                self.kw = kw

        @staticmethod
        def _prepare_ask(req):
            return None, [], None, [], [], "prompt", None

    cache = harness._Cache(Path("/nonexistent/cache.pkl"))
    monkeypatch.setattr(harness, "prompt_tokens", lambda tok, prompt: 11265)
    row = harness.serve_one(Flow, cache, None, "q", [])
    assert row["n_ctx"] == 16384  # call-time default, not the legacy 8192
    assert row["fits_train"] is False
    monkeypatch.setattr(harness, "prompt_tokens", lambda tok, prompt: 11264)
    assert harness.serve_one(Flow, cache, None, "q", [])["fits_train"] is True


def _case(qid, query):
    return {"set": "s", "qid": qid, "type": "t", "query": query, "history": [], "claims": [["a"]]}


def _fake_row(case, tokens=100):
    return {"set": case["set"], "qid": case["qid"], "type": case["type"], "claims": case["claims"],
            "passage_ids": ["a"], "merged_ids": ["a"], "retrieved_ids": ["a"], "prompt_chars": 1,
            "prompt_tokens": tokens, "wall_ms": 1.0, "latency_ms": 1.0, "used_filter": None, "n_ctx": 16384,
            "fits_train": True}


def test_run_config_reuses_baseline_rows_under_current_mode(tmp_path, monkeypatch):
    pool = [_case("t1", TEMPORAL), _case("n1", NO_YEAR)]
    served = []
    by_query = {c["query"]: c for c in pool}
    monkeypatch.setattr(harness, "serve_one", lambda flow, cache, tok, q, h, mnt=None: (
        served.append(q), _fake_row(by_query[q]))[1])
    cache = harness._Cache(tmp_path / "cache.pkl")
    _, base_over = harness.parse_config("baseline", baseline="current")
    harness.run_config("baseline", base_over, None, cache, None, pool, [], tmp_path, None, baseline="current")
    assert len(served) == 2
    served.clear()
    _, over = harness.parse_config("pool30:RETRIEVAL_PERIOD_POOL=30", baseline="current")
    harness.run_config("pool30", over, None, cache, None, pool, [], tmp_path, None, baseline="current")
    assert served == [TEMPORAL]  # the non-temporal case reuses its baseline row (MAX_NEW_TOKENS is the same)
    result = json.loads((tmp_path / "served_pool30.json").read_text(encoding="utf-8"))
    assert result["baseline_mode"] == "current" and len(result["rows"]) == 2


def test_run_config_refuses_baseline_of_another_mode(tmp_path, monkeypatch):
    pool = [_case("t1", TEMPORAL)]
    monkeypatch.setattr(harness, "serve_one", lambda *a, **k: _fake_row(pool[0]))
    cache = harness._Cache(tmp_path / "cache.pkl")
    harness.run_config("baseline", harness.parse_config("baseline")[1], None, cache, None, pool, [], tmp_path, None)
    _, over = harness.parse_config("pool30:RETRIEVAL_PERIOD_POOL=30", baseline="current")
    with pytest.raises(ValueError, match="baseline"):
        harness.run_config("pool30", over, None, cache, None, pool, [], tmp_path, None, baseline="current")


# ----------------------------------------------------------------------------- arms file
def test_read_arms_file_skips_blank_and_comment_lines(tmp_path):
    path = tmp_path / "arms.txt"
    path.write_text("# arms\nbaseline\n\n  nb2:ANSWER_CONTEXT_NEIGHBORS=2  \n#off:X=1\npool30:RETRIEVAL_PERIOD_POOL=30\n",
                    encoding="utf-8")
    assert harness.read_arms_file(path) == ["baseline", "nb2:ANSWER_CONTEXT_NEIGHBORS=2", "pool30:RETRIEVAL_PERIOD_POOL=30"]


def test_shipped_arms_file_has_the_eleven_design_arms():
    if not ARMS.exists():  # openspec/ is git-ignored: absent on a fresh clone
        pytest.skip("arms.txt not present")
    arms = harness.read_arms_file(ARMS)
    parsed = [harness.parse_config(a, baseline="current") for a in arms]
    assert [n for n, _ in parsed] == ["baseline", "nb2", "pool30", "res8", "focus", "pool30focus", "ykeep4", "yref",
                                      "slots3", "fu", "route2"]
    by_name = dict(parsed)
    assert by_name["nb2"]["ANSWER_CONTEXT_NEIGHBORS"] == "2" and by_name["nb2"]["LLM_PASSAGE_MAX_CHARS"] == "3000"
    assert by_name["pool30focus"]["RETRIEVAL_PERIOD_POOL"] == "30"
    assert by_name["pool30focus"]["RETRIEVAL_PERIOD_FOCUS_RERANK"] == "1"
    assert by_name["res8"]["RETRIEVAL_PERIOD_RESERVE"] == "8" and by_name["slots3"]["RETRIEVAL_PART_SLOTS"] == "3"
    assert by_name["ykeep4"]["RETRIEVAL_YEAR_FILTER_KEEP"] == "4" and by_name["yref"]["RETRIEVAL_YEAR_REF_SKIP"] == "1"
    assert by_name["fu"]["RETRIEVAL_FOLLOWUP_ALWAYS"] == "1" and by_name["route2"]["COMPARE_ROUTING_V2"] == "1"


# ----------------------------------------------------------------------------- training fit and report
def test_fits_train_boundary():
    assert harness.TRAIN_MAX_SEQ == 12288 and harness.TRAIN_ANSWER_TOKENS == 1024
    assert harness.fits_train(11264) is True
    assert harness.fits_train(11265) is False
    assert harness.fits_train(0) is True


def test_percentile_nearest_rank():
    assert harness.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert harness.percentile(list(range(1, 101)), 95) == 95
    assert harness.percentile([], 95) is None


def test_summarize_token_columns():
    rows = [_fake_row(_case(f"q{i}", "x"), tokens=t) for i, t in enumerate([1000, 2000, 11264, 11265])]
    rows[3].pop("fits_train")  # older result files have no stored column: recomputed from prompt_tokens
    s = harness.summarize({"overrides": {}, "rows": rows, "hard_negatives": []})
    assert s["prompt_tokens_max"] == 11265 and s["prompt_tokens_p95"] == 11265
    assert s["fits_train_all"] is False and s["fits_train_failures"] == 1


def test_report_has_token_and_per_type_delta_tables(tmp_path, monkeypatch):
    def result(name, overrides, hits):
        rows = []
        for i, (typ, hit) in enumerate(hits):
            case = _case(f"q{i}", "x") | {"type": typ}
            rows.append(_fake_row(case, tokens=200 + i) | {"type": typ, "passage_ids": ["a"] if hit else []})
        return {"name": name, "overrides": overrides, "rows": rows, "hard_negatives": []}

    base = result("baseline", {}, [("fact", 0), ("fact", 1), ("temporal", 0), ("temporal", 0)])
    cand = result("pool30", {"RETRIEVAL_PERIOD_POOL": "30"}, [("fact", 0), ("fact", 1), ("temporal", 1), ("temporal", 0)])
    for r in (base, cand):
        (tmp_path / f"served_{r['name']}.json").write_text(json.dumps(r), encoding="utf-8")
    monkeypatch.setattr(harness, "determinism_check", lambda *a, **k: {
        "checked": 0, "exact_order": 0, "same_set_other_order": 0, "same_ids_rate": None, "mismatches": []})
    harness.report(tmp_path, "pool30")
    md = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "tokens p95" in md and "fits train" in md
    assert "## Per question type: served recall and delta vs baseline" in md
    assert "+50.0pp (+1)" in md  # temporal: 0/2 -> 1/2
    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert payload["vs_baseline"]["pool30"]["gate"]["fits_train"] is True
    assert payload["summary"]["pool30"]["prompt_tokens_p95"] == 203
