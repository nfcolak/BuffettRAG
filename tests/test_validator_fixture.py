"""Hand-labelled sentence support, with false-block/pass diagnostics."""
import json
from pathlib import Path

from src.evaluation.claim_validator import validate_and_filter_answer
from src.storage import SearchHit

FIXTURE = Path(__file__).parent / "fixtures" / "validator_cases.json"


def measure_fixture():
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
    rows = []
    for case in cases:
        hits = [SearchHit(p["id"], p["text"], {}, 1.0) for p in case["cited_passages"]]
        result = validate_and_filter_answer(case["sentence"], hits)
        kept = result.safe_answer == case["sentence"]
        rows.append({"id": case["id"], "label": case["label"], "kept": kept,
                     "scores": result.validations, "blocked": result.blocked_claims})
    supported = sum(row["label"] == "supported" for row in rows)
    unsupported = len(rows) - supported
    false_blocks = sum(row["label"] == "supported" and not row["kept"] for row in rows)
    false_passes = sum(row["label"] == "unsupported" and row["kept"] for row in rows)
    return {"sentences": len(rows), "supported": supported, "unsupported": unsupported,
            "false_blocks": false_blocks, "false_passes": false_passes, "rows": rows}


def test_hand_labelled_validator_fixture():
    report = measure_fixture()
    print(json.dumps(report, indent=2))
    assert report["sentences"] >= 20
    assert report["supported"] and report["unsupported"]
    assert report["false_passes"] == 0
    assert report["false_blocks"] <= report["supported"] / 4


def test_partially_supported_answer_keeps_supported_sentence_verbatim():
    passage = SearchHit("p", "Operating earnings per share were up 37% from the year before.", {}, 1.0)
    supported = "Operating earnings per share were up 37% from the year before. [1]"
    result = validate_and_filter_answer(supported + " Revenue was $99 billion. [1]", [passage])
    assert result.safe_answer == supported
    assert result.blocked_claims


def test_hard_checks_also_apply_when_using_nli():
    passage = SearchHit("p", "Revenue was $10 million. The firm did not buy preferred shares.", {}, 1.0)
    for sentence in ("Revenue was $12 million. [1]", "The firm did buy preferred shares. [1]"):
        result = validate_and_filter_answer(sentence, [passage], nli_scorer=lambda *args: 1.0)
        assert result.safe_answer == ""


def test_citations_do_not_count_as_claim_quantities_and_typography_is_normalized():
    passage = SearchHit("p", "The company's revenue was $1,250.50 million, up 8 percent.", {}, 1.0)
    sentence = "The company’s revenue was $1,250.50 million, up 8%. [1]"
    assert validate_and_filter_answer(sentence, [passage]).safe_answer == sentence
