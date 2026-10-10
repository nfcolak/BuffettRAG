"""fair_metrics.py: k-truncation and supported-sentence precision on a 2-case fixture (stub scorer, no models)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from src.evaluation.supported_claims import _answer_sentences
from src.generation.prompt import REFUSAL_LINE

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fair_metrics", ROOT / "scripts/eval/fair_metrics.py")
fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fm)

CASES = {
    "a": {"qid": "a", "gold_claims": [{"claim": "Float is cheap.", "required_terms": ["float", "cheap"]}]},
    "b": {"qid": "b", "gold_claims": [{"claim": "Berkshire acquired GEICO.", "required_terms": ["acquired", "geico"]}]},
    "u": {"qid": "u", "gold_claims": [], "answerable": False},
}
SNAP = [{"id": "g1", "text": "x", "year": 1999}]
ROWS = [
    {"qid": "a", "answer": "Float is cheap [1]. Unrelated filler [1]. More filler [1].", "citations": [], "passage_ids": ["g1"],
     "context_snapshot": SNAP},
    {"qid": "b", "answer": "Filler comes first [1]. Berkshire acquired GEICO [1].", "citations": [], "passage_ids": ["g1"],
     "context_snapshot": SNAP},
    {"qid": "u", "answer": REFUSAL_LINE, "citations": [], "passage_ids": ["g1"], "context_snapshot": SNAP},
]


def stub_scorer(case, answer, citations, hits):
    """Met when one answer sentence holds all required terms; that sentence is the supporting one."""
    if not case["gold_claims"]:
        return []
    sentences = _answer_sentences(answer)
    out = []
    for claim in case["gold_claims"]:
        support = [s for s in sentences if all(t in s.lower() for t in claim["required_terms"])]
        out.append({"claim": claim["claim"], "met": bool(support), "supporting_sentences": support})
    return out


def test_k_truncation_and_precision_on_two_cases():
    out = fm.fair_metrics({"rows": ROWS}, CASES, stub_scorer, None, (1, 2), "arm")
    assert out["required_claims"] == 2 and out["strict_claims_met"] == 2
    assert out["met_at_1"] == 1  # case a's claim is sentence 1; case b's is sentence 2
    assert out["met_at_2"] == 2
    assert out["answer_sentences"] == 5 and out["sentences_per_answer"] == 2.5
    assert out["supporting_sentences"] == 2 and out["supported_sentence_precision"] == 0.4
    assert out["sentences_per_met_claim"] == 2.5
    assert out["refusals"] == 1  # the unanswerable row's refusal is counted but not scored for precision


def test_truncate_per_answer_and_per_period_paragraph():
    text = "In 1985: First one. Second one.\nIn 1995: Third one. Fourth one."
    assert fm.truncate(text, 1, "answer") == "In 1985: First one."
    assert fm.truncate(text, 1, "line") == "In 1985: First one.\nIn 1995: Third one."
    assert fm.truncate(text, None) == text
