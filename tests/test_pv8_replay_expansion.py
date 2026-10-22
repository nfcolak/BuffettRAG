"""pv8 D1: replay_expansion re-runs only validator + expansion from stored raw answers."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "replay_expansion_pv8", Path(__file__).resolve().parents[1] / "scripts/eval/replay_expansion.py")
replay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(replay)

PASSAGE = (
    "Insurance float is money we hold but do not own and it has been very valuable to us. "
    "Our float grew from sixteen million dollars to many billions of dollars over the decades. "
    "We try to write policies only when the premiums are adequate for the risks involved. "
    "Charlie and I regard underwriting discipline as the essential condition for good float."
)
RAW = "Our float grew from sixteen million dollars to many billions of dollars over the decades. [1]"
QUERY = "How did Berkshire's insurance float grow over the decades?"


def _row(qid="q1", raw=(RAW,), query=QUERY, answer="stored", citations=()):
    return {"qid": qid, "query": query, "history": [], "question_type": "answerable", "status": "scored",
            "answer": answer, "citations": list(citations), "raw_answers": list(raw),
            "context_snapshot": [{"id": "1990_p1", "year": 1990, "text": PASSAGE, "text_sha256": "x", "merged_ids": ["1990_p1"]}],
            "score": None, "required_claim_hits": None, "refusal_correct": None}


def _run(**kw):
    return {"rows": [_row(**kw)], "summary": {}, "cases": "nope.json"}


def _stored(after, before=0, cap=12):
    settings = dict(replay.LEGACY_SETTINGS, ANSWER_NEIGHBOR_AFTER=after, ANSWER_NEIGHBOR_BEFORE=before,
                    ANSWER_MAX_SENTENCES=cap, ANSWER_EXPANDED_MAX_SENTENCES=cap)
    new, _ = replay.replay_row(_row(), settings)
    return new["answer"], new["citations"]


def test_replay_reproduces_the_stored_answer_at_the_stored_settings():
    answer, citations = _stored(after=2)
    assert answer.count("[1]") == 3  # the raw sentence + 2 neighbours
    run = _run(answer=answer, citations=citations)
    run["settings"] = {"ANSWER_NEIGHBOR_AFTER": 2, "ANSWER_NEIGHBOR_BEFORE": 0, "ANSWER_MAX_SENTENCES": 12,
                       "ANSWER_EXPANDED_MAX_SENTENCES": 12, "COMPARE_ROUTING_V2": 0, "ANSWER_REFUSAL_RETRY": "1"}
    report = replay.verify(run)
    assert (report["matched"], report["rows"]) == (1, 1) and not report["desync_qids"]


def test_replay_at_other_settings_changes_only_the_expansion():
    answer, citations = _stored(after=2)
    run = _run(answer=answer, citations=citations)
    base = dict(replay.stored_settings(run), ANSWER_NEIGHBOR_AFTER=0, ANSWER_NEIGHBOR_BEFORE=0)
    off, _ = replay.replay_run(run, base)
    assert off["rows"][0]["answer"] == RAW
    both, _ = replay.replay_run(run, dict(base, ANSWER_NEIGHBOR_AFTER=3, ANSWER_NEIGHBOR_BEFORE=3))
    assert both["rows"][0]["answer"].count("[1]") == 4  # all four passage sentences
    assert both["replay"]["model_called"] is False and both["settings"]["ANSWER_NEIGHBOR_BEFORE"] == 3
    assert run["rows"][0]["answer"] == answer  # the input run is not mutated


def test_extra_raw_answers_or_missing_raw_answers_are_flagged_not_hidden():
    _, status = replay.replay_row(_row(raw=()), dict(replay.LEGACY_SETTINGS))
    assert status == "desync"  # a model call the stored run no longer has
    _, status = replay.replay_row(_row(raw=(RAW, RAW)), dict(replay.LEGACY_SETTINGS))
    assert status == "desync"  # an unused stored answer


def test_gate_refusal_needs_no_raw_answer():
    row = _row(raw=(), query="Which colour was the dinosaur parade in Tokyo?", answer="x")
    new, status = replay.replay_row(row, dict(replay.LEGACY_SETTINGS))
    assert status == "ok" and new["answer"] == "The retrieved passages do not provide enough evidence to answer this question."


def test_provider_failure_rows_are_copied():
    row = _row()
    row["status"] = "provider_failure"
    new, status = replay.replay_row(row, dict(replay.LEGACY_SETTINGS))
    assert status == "copied" and new == row


ARTIFACTS = Path(os.environ.get("BUFFETTRAG_ARTIFACTS", Path(__file__).resolve().parents[1].parents[2] / "Artifacts" / "BuffettRAG"))
PV7 = ARTIFACTS / "pv7" / "runs" / "ftr3"


@pytest.mark.skipif(not (PV7 / "ftr3_p16_e3_hv6a.json").is_file(), reason="pv7 runs not on this machine")
@pytest.mark.parametrize("name", ["ftr1_p16_e3_hv6a", "ftr3_p16_e3_hv6a", "ftr3_p16_e3_hv6b", "ftr1_p16_e3_hv6b"])
def test_pv7_runs_are_reproduced_row_for_row(name):
    run = json.loads((PV7 / f"{name}.json").read_text())
    report = replay.verify(run)
    assert report["matched"] == report["rows"], report["mismatched_qids"]
