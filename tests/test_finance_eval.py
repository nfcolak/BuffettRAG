"""finance_v1 eval scripts: scoring on synthetic responses, set guard (no network, no model)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "eval"))
import finance_eval  # noqa: E402
import score_finance as sf  # noqa: E402

CASES = {
    "a1": {"qid": "a1", "group": 1, "question_type": "answerable", "gold_passage_ids": ["p1", "p2"],
           "gold_claims": [{"required_terms": ["1,234", "oil"]}, {"required_terms": ["gold"]}]},
    "u1": {"qid": "u1", "group": 2, "question_type": "unanswerable"},
    "o1": {"qid": "o1", "group": 4, "question_type": "out_of_scope"},
    "m1": {"qid": "m1", "group": 4, "question_type": "mixed_scope", "reject": {"forbidden_terms": ["omaha steak"]}},
}


def resp(qid, answer, served=(), seconds=2.0):
    return {"qid": qid, "answer": answer, "served_ids": list(served), "retrieved_ids": [], "seconds": seconds, "error": None}


def test_scoring_metrics() -> None:
    rows = sf.score_rows([
        resp("a1", "Oil cost $1234 million.", served=["p1"], seconds=4),
        resp("u1", sf.REFUSE.capitalize()),
        resp("o1", "I only answer questions about Buffett's financial views and Berkshire's financial results."),
        resp("m1", "He likes an Omaha steak."),
    ], CASES, "dev")
    by = {r["qid"]: r for r in rows}
    assert by["a1"]["term_cov"] == pytest.approx(2 / 3) and by["a1"]["claim_cov"] == pytest.approx(0.5)
    assert by["a1"]["recall_served"] == 0.5
    assert by["u1"]["refused"] and by["o1"]["scoped"] and by["m1"]["leak"] is True
    line = sf.agg_line(rows, "x")
    assert "scope reply 1/1" in line and "mixed leaks 1/1" in line and "unanswerable refused 1/1" in line
    assert "gold passages served 50%" in line


def test_dev_never_touches_test_path(monkeypatch) -> None:
    opened = []
    real = Path.read_text

    def spy(self, *a, **k):
        opened.append(str(self))
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", spy)
    cases = finance_eval.load_cases("dev")
    assert len(cases) == 40
    assert opened == [str(finance_eval.CASE_FILES["dev"])]
    assert all("finance_v1_test" not in p for p in opened)
    assert finance_eval.CASE_FILES["dev"].is_relative_to(finance_eval.REPO)
    assert not finance_eval.CASE_FILES["test"].is_relative_to(finance_eval.REPO)


def test_test_set_refused_without_flag() -> None:
    with pytest.raises(SystemExit):
        finance_eval.load_cases("test")
    with pytest.raises(SystemExit):
        sf.main(["nonexistent.jsonl", "--set", "test"])
    with pytest.raises(SystemExit):
        finance_eval.main(["--url", "http://x", "--name", "n", "--set", "test"])


def test_scorer_default_set_is_dev(tmp_path) -> None:
    f = tmp_path / "responses.jsonl"
    qid = finance_eval.load_cases("dev")[0]["qid"]
    f.write_text(json.dumps(resp(qid, "x")) + "\n")
    sf.main([str(f)])
    assert json.loads((tmp_path / "scores.json").read_text())[0]["half"] == "dev"


def test_record_stores_debug_only_when_present() -> None:
    case = {"qid": "a1"}
    assert "debug" not in finance_eval.record(case, "dev", None, {"answer": "x"}, 1.0)
    assert finance_eval.record(case, "dev", None, {"answer": "x", "debug": {"route": "reader"}}, 1.0)["debug"] == {"route": "reader"}


def test_record_keeps_anchors_as_served_and_merged_separately():
    from scripts.eval.finance_eval import record

    data = {"answer": "x", "hits": [{"id": "a"}, {"id": "b"}],
            "debug": {"served_ids": ["a", "a_prev", "b"]}}
    rec = record({"qid": "q1"}, "dev", None, data, 1.0)
    assert rec["served_ids"] == ["a", "b"]          # pre-registered measure: anchors only
    assert rec["served_ids_merged"] == ["a", "a_prev", "b"]


def test_v2test_is_sealed_without_the_final_run_flag():
    import pytest
    from scripts.eval.finance_eval import case_file

    with pytest.raises(SystemExit):
        case_file("v2test")
    assert case_file("v2test", final_run=True).name == "finance_v2_test.json"
