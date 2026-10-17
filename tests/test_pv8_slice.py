"""pv8 PG-04/05/07/08: slice selection, prompt-passage mapping and the cheap proxy scorer."""
import importlib.util
import json
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parents[1] / "scripts/eval"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_pv8", _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build, scorer = _load("build_slice_prompts"), _load("score_slice")


def _case(qid, kind, golds, answerable=True):
    return {"qid": qid, "question_type": kind, "answerable": answerable, "query": f"query {qid}",
            "gold_claims": [{"claim": f"claim {qid} {i}", "required_terms": [f"term{qid}{i}"], "gold_passage_ids": [g]}
                            for i, g in enumerate(golds)]}


IDS = [f"p{i}" for i in range(1, 11)]


def _world():
    cases = {"d1": _case("d1", "opinion", ["p5", "p6"]), "d2": _case("d2", "company", ["p7", "p9", "p1"]),
             "d3": _case("d3", "opinion", ["p4", "p8"]), "one": _case("one", "company", ["p6", "p1"]),
             "a1": _case("a1", "company", ["p1", "p2"]), "a2": _case("a2", "company", ["p3", "p1"]),
             "bad": _case("bad", "company", ["p1", "p2"]),
             "u1": _case("u1", "unanswerable", [], answerable=False), "u2": _case("u2", "unanswerable", [], answerable=False)}
    rows = {q: {"passage_ids": IDS} for q in cases}
    met3 = {(q, c["claim"]): "met" for q in ("a1", "a2") for c in cases[q]["gold_claims"]}
    met1 = dict(met3)
    return cases, rows, met3, met1


def test_select_slice_takes_deep_balanced_anchors_and_refusals():
    cases, rows, met3, met1 = _world()
    groups = build.select_slice(cases, rows, met3, met1)
    assert groups["deep"] == ["d2", "d1", "d3"]  # >= 2 claims at rank >= 4 ("one" has only one), types round-robin
    assert groups["anchors"] == ["a1", "a2"]  # all claims rank 1-3 and both models met them; "bad" was not met
    assert groups["refusals"] == ["u1", "u2"]


def test_select_slice_balances_question_types():
    cases, rows, met3, met1 = _world()
    build.N_DEEP = 2
    try:
        assert build.select_slice(cases, rows, met3, met1)["deep"] == ["d2", "d1"]  # one company, then one opinion
    finally:
        build.N_DEEP = 16


def test_claim_ranks():
    case = _case("x", "opinion", ["p5", "p99", "p1"])
    assert build.claim_ranks(case, IDS) == [5, None, 1]


def test_prompt_passage_ids_follow_the_blocks():
    from src.generation.prompt import build_cited_prompt
    from src.storage import SearchHit
    hits = [SearchHit(f"x{i}", f"Passage {i} text about float.\nSecond line.", {"year": 2000 + i, "source_file": "s"}, 0.0)
            for i in range(3)]
    user = build_cited_prompt("What about float?", hits[::-1])
    assert build.prompt_passage_ids(user, hits) == ["x2", "x1", "x0"]
    with pytest.raises(ValueError):
        build.prompt_passage_ids(user, hits[:1])


def _slice_dir(tmp_path):
    claims = lambda rank, gold, terms: [{"claim": "c", "required_terms": terms, "gold_passage_ids": [gold], "rank": rank}]
    doc = {"questions": {
        "deep1": {"group": "deep", "served_passage_ids": IDS, "claims": claims(5, "p5", ["float grew"]) + claims(1, "p1", ["underwriting"])},
        "anch1": {"group": "anchors", "served_passage_ids": IDS, "claims": claims(2, "p2", ["goodwill"])},
        "cmp": {"group": "deep", "served_passage_ids": IDS, "claims": claims(4, "p4", ["tax return"])},
        "ref1": {"group": "refusals", "served_passage_ids": IDS, "claims": []}}}
    (tmp_path / "slice.json").write_text(json.dumps(doc))
    rows = [{"id": "cmp#2006", "qid": "cmp", "passage_ids": ["p9", "p4"]}, {"id": "cmp#2014", "qid": "cmp", "passage_ids": ["p8"]}]
    (tmp_path / "slice_prompts.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def test_score_terms_cite_gold_and_citation_position(tmp_path):
    gen = {"deep1": "Our float grew a lot. [5] Underwriting matters. [1,5]",
           "anch1": "Goodwill is not amortised. [1]",
           "cmp#2006": "The tax return was big. [2]", "cmp#2014": "Another year. [1]",
           "ref1": "The retrieved passages do not provide enough evidence to answer this question."}
    out = scorer.score(gen, *scorer.load_slice(_slice_dir(tmp_path)))
    assert (out["deep_n"], out["deep_terms"], out["deep_cite_gold"]) == (2, 2, 2)  # deep1 claim + cmp (cited via its period prompt)
    assert (out["shallow_n"], out["shallow_terms"], out["shallow_cite_gold"]) == (1, 1, 1)  # [1,5] cites p1
    assert (out["anchor_n"], out["anchor_terms"], out["anchor_cite_gold"]) == (1, 1, 0)  # goodwill claim cites [1], gold is p2
    assert out["refusals_ok"] == 1 and out["refusals_n"] == 1
    assert out["all_total"] == 6 and out["all_rank1"] == 3 and out["all_rank4plus"] == 2 and out["all_cite_1_share"] == 0.5


def test_score_reports_missing_questions_and_bare_qid_keys(tmp_path):
    out = scorer.score({"deep1": "Float grew. [1]"}, *scorer.load_slice(_slice_dir(tmp_path)))
    assert set(out["questions_missing"]) == {"anch1", "cmp", "ref1"}
    assert out["deep_terms"] == 1 and out["shallow_terms"] == 0 and out["all_cite_1_share"] == 1.0
