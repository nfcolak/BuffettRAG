"""score_finance answer-quality fields (year_cov, dup_rate, refused_with_gold, unused_gold, partial_ok) on synthetic responses."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "eval"))
import score_finance as sf  # noqa: E402

GOLD = ["1999_p0001", "2008_p0002", "2009_p0003"]
CASES = {
    "a": {"qid": "a", "group": 1, "question_type": "crisis_macro", "gold_passage_ids": GOLD, "years": [1999, 2008, 2009],
          "gold_claims": [{"required_terms": ["Mr. Market", "panic"]}, {"required_terms": ["$15.5 billion"]}]},
    "p": {"qid": "p", "group": 2, "question_type": "partial", "gold_passage_ids": ["1994_p0007"], "years": [1994],
          "gold_claims": [{"required_terms": ["two oil shocks"]}]},
    "u": {"qid": "u", "group": 3, "question_type": "unanswerable", "years": [1985]},
    "o": {"qid": "o", "group": 4, "question_type": "out_of_scope"},
    "n": {"qid": "n", "group": 5, "question_type": "answerable", "gold_passage_ids": ["p1", "p2"], "years": [1990, 1991],
          "gold_claims": [{"required_terms": ["oil"]}]},
}
REFUSAL = "The retrieved passages do not provide enough evidence to answer this question."


def cite(*ids):
    return [{"marker": f"[{i}]", "passage_ids": [pid], "years": [int(pid[:4])] if pid[:4].isdigit() else []}
            for i, pid in enumerate(ids, 1)]


def resp(qid, answer, served=(), citations=()):
    return {"qid": qid, "answer": answer, "served_ids": list(served), "citations": list(citations),
            "retrieved_ids": [], "seconds": 1.0, "error": None}


def row(qid, answer, served=(), citations=()):
    return sf.score_rows([resp(qid, answer, served, citations)], CASES, "dev")[0]


# year_cov ----------------------------------------------------------------------------------------------------------

def test_year_cov_is_cited_gold_years_over_gold_years() -> None:
    assert row("a", "x [1]", GOLD, cite("2008_p0002"))["year_cov"] == pytest.approx(1 / 3)
    assert row("a", "x [1]", GOLD, cite("1999_p0001", "2008_p0002"))["year_cov"] == pytest.approx(2 / 3)
    assert row("a", "x [1]", GOLD, cite(*GOLD))["year_cov"] == 1.0


def test_year_cov_counts_distinct_years_and_ignores_non_gold_years() -> None:
    cit = cite("2008_p0002", "2008_p0009", "2008_p0010", "2011_p0004")          # 2008 three times, 2011 is not a gold year
    assert row("a", "x", GOLD, cit)["year_cov"] == pytest.approx(1 / 3)


def test_year_cov_zero_for_refusal_and_none_without_gold() -> None:
    assert row("a", REFUSAL, GOLD)["year_cov"] == 0.0
    assert row("u", REFUSAL)["year_cov"] is None                                  # unanswerable: `years` is not gold
    assert row("o", "I only answer questions about Buffett's financial views and Berkshire's financial results.")["year_cov"] is None


def test_year_cov_reads_years_from_passage_ids_when_the_citation_has_no_years_field() -> None:
    cit = [{"marker": "[1]", "passage_ids": ["2009_p0003"]}]
    assert row("a", "x [1]", GOLD, cit)["year_cov"] == pytest.approx(1 / 3)


def test_year_cov_falls_back_to_case_years_when_gold_ids_have_no_year_prefix() -> None:
    r = row("n", "x [1]", ["p1"], [{"marker": "[1]", "passage_ids": ["p1"], "years": [1991]}])
    assert r["year_cov"] == 0.5


def test_missing_citations_field_is_tolerated() -> None:
    r = sf.score_rows([{"qid": "a", "answer": "x", "served_ids": GOLD}], CASES, "dev")[0]
    assert r["year_cov"] == 0.0 and r["unused_gold"] == 3


# dup_rate ----------------------------------------------------------------------------------------------------------

def test_dup_rate_zero_without_repeats() -> None:
    assert row("a", "He bought stocks in 2008. [1] He wrote about fear. [2]")["dup_rate"] == 0.0
    assert row("a", "")["dup_rate"] == 0.0 and row("a", None)["dup_rate"] == 0.0


def test_dup_rate_counts_repeats_ignoring_markers_whitespace_and_case() -> None:
    ans = ("Buffett bought $15.5 billion of stocks in the crisis. [1]\n"
           "He admitted his own mistakes.  [2]\n"
           "buffett bought $15.5 billion of  stocks in the crisis [3]\n"
           "Buffett bought $15.5 billion of stocks in the crisis. [1, 2]")
    assert row("a", ans)["dup_rate"] == pytest.approx(2 / 4)


def test_dup_rate_is_not_fooled_by_curly_quotes_and_dashes() -> None:
    ans = "We don’t know what the market will do – ever. [1] We don't know what the market will do - ever. [2]"
    assert row("a", ans)["dup_rate"] == pytest.approx(1 / 2)


def test_dup_rate_does_not_split_abbreviations() -> None:
    ans = "Mr. Market was on a manic rampage in the U.S. until October. [1] Mr. Market was on a manic rampage in the U.S. until October. [2]"
    assert sf.sentences(ans) == ["Mr. Market was on a manic rampage in the U.S. until October.",
                                 "Mr. Market was on a manic rampage in the U.S. until October."]
    assert row("a", ans)["dup_rate"] == pytest.approx(1 / 2)


def test_dup_rate_splits_after_a_closing_quote() -> None:
    quote = "The watchword became the creed I saw on restaurant walls: \u201cIn God we trust; all others pay cash.\u201d"
    ans = f"{quote} By the fourth quarter fear engulfed the country. [1] {quote} A freefall in business activity ensued. [2]"
    assert sf.sentences(ans) == [quote, "By the fourth quarter fear engulfed the country.", quote, "A freefall in business activity ensued."]
    assert row("a", ans)["dup_rate"] == pytest.approx(1 / 4)
    assert sf.sentences('He said "no way." Then he left (quietly). Done now.') == ['He said "no way."', "Then he left (quietly).", "Done now."]


def test_dup_rate_ignores_tiny_fragments_and_list_bullets() -> None:
    assert row("a", "Yes. [1] Yes. [2] Yes. [3]")["dup_rate"] == 0.0
    ans = "- He kept buying good businesses. [1]\n- He kept buying good businesses. [2]"
    assert row("a", ans)["dup_rate"] == pytest.approx(1 / 2)


def test_dup_rate_different_numbers_are_different_sentences() -> None:
    assert row("a", "Float was $16 billion in 1999. [1] Float was $17 billion in 1999. [2]")["dup_rate"] == 0.0


# refused_with_gold -------------------------------------------------------------------------------------------------

def test_refused_with_gold_needs_the_refusal_line_and_a_served_gold_passage() -> None:
    assert row("a", REFUSAL, served=["2008_p0002", "x"])["refused_with_gold"] is True
    assert row("a", REFUSAL, served=["x", "y"])["refused_with_gold"] is False        # nothing gold was served
    assert row("a", REFUSAL, served=[])["refused_with_gold"] is False
    assert row("u", REFUSAL, served=["1985_p0001"])["refused_with_gold"] is False   # no gold passages: refusing is right


def test_refused_with_gold_is_exact_not_wording_based() -> None:
    assert row("a", "He bought stocks. [1]", served=GOLD)["refused_with_gold"] is False
    assert row("a", "", served=GOLD)["refused_with_gold"] is False                   # an empty answer is an error, not the refusal line
    assert row("a", "I only answer questions about Buffett's financial views and Berkshire's financial results.",
               served=GOLD)["refused_with_gold"] is False
    assert row("a", "The letters do not discuss this. He bought stocks. [1]", served=GOLD)["refused_with_gold"] is False
    assert row("a", REFUSAL.lower() + " [1]", served=GOLD)["refused_with_gold"] is True


# unused_gold -------------------------------------------------------------------------------------------------------

def test_unused_gold_counts_served_gold_that_is_not_cited() -> None:
    r = row("a", "x [1]", served=["1999_p0001", "2008_p0002", "other"], citations=cite("1999_p0001", "other"))
    assert r["unused_gold"] == 1 and r["gold_served"] == 2
    r = row("a", "x [1]", served=GOLD, citations=cite(*GOLD))
    assert r["unused_gold"] == 0 and r["gold_served"] == 3
    r = row("a", REFUSAL, served=GOLD)
    assert r["unused_gold"] == 3


def test_unused_gold_ignores_gold_that_was_not_served_and_none_without_gold() -> None:
    r = row("a", "x [1]", served=["1999_p0001"], citations=cite("1999_p0001", "2008_p0002"))
    assert r["unused_gold"] == 0 and r["gold_served"] == 1
    assert row("u", REFUSAL)["unused_gold"] is None and row("u", REFUSAL)["gold_served"] is None


# partial_ok --------------------------------------------------------------------------------------------------------

def test_partial_ok_needs_a_stated_gap_and_a_gold_term() -> None:
    both = "The 1980s letters do not discuss an oil crisis. The 1994 letter mentions two oil shocks. [1]"
    assert row("p", both, ["1994_p0007"], cite("1994_p0007"))["partial_ok"] is True
    assert row("p", REFUSAL, ["1994_p0007"])["partial_ok"] is False                  # gap, no content
    assert row("p", "The 1994 letter mentions two oil shocks. [1]", ["1994_p0007"], cite("1994_p0007"))["partial_ok"] is False
    assert row("p", "Nothing was said about the cold war.")["partial_ok"] is False
    assert row("p", "")["partial_ok"] is False


def test_partial_ok_is_none_for_other_types() -> None:
    for qid in ("a", "u", "o", "n"):
        assert row(qid, REFUSAL)["partial_ok"] is None


def test_partial_ok_term_match_uses_the_scorer_normalisation() -> None:
    both = "The letters don’t cover the 1980s. Two Oil Shocks hit in the 1970s. [1]"
    assert row("p", both, ["1994_p0007"], cite("1994_p0007"))["partial_ok"] is True


# existing fields and lines -----------------------------------------------------------------------------------------

EXISTING_FIELDS = {"qid", "half", "group", "type", "seconds", "claim_cov", "term_cov", "recall_served", "recall_merged",
                   "recall_retrieved", "refused", "scoped", "leak", "error"}


def test_existing_fields_keep_their_values_and_new_fields_are_added() -> None:
    r = row("a", "Mr. Market panic. The $15.5 billion. [1]", served=["1999_p0001"], citations=cite("1999_p0001"))
    assert EXISTING_FIELDS <= set(r)
    assert {"year_cov", "dup_rate", "refused_with_gold", "unused_gold", "partial_ok"} <= set(r)
    assert r["claim_cov"] == 1.0 and r["term_cov"] == 1.0 and r["recall_served"] == pytest.approx(1 / 3)
    assert r["refused"] is False and r["leak"] is None and r["type"] == "crisis_macro"


def test_agg_line_is_unchanged_by_the_quality_fields() -> None:
    rows = sf.score_rows([resp("a", "Panic. [1]", served=["1999_p0001"], citations=cite("1999_p0001")),
                          resp("u", REFUSAL)], CASES, "dev")
    line = sf.agg_line(rows, "x")
    assert line == ("x                            n= 2 | claim coverage 0% (n=1), terms 33% | gold passages served 33% "
                    "| unanswerable refused 1/1 | partial states gap 0/0 | personal: scope reply 0/0, refused 0/0, answered 0/0 "
                    "| mixed leaks 0/0 | errors 0 | sec 1.0")
    assert "year" not in line and "dup" not in line


# quality line ------------------------------------------------------------------------------------------------------

def quality_rows():
    return sf.score_rows([
        resp("a", "He bought stocks in the crisis. [1] He bought stocks in the crisis. [2]", served=GOLD,
             citations=cite("2008_p0002", "2009_p0003")),
        resp("n", REFUSAL, served=["p1"]),
        resp("p", "The letters do not discuss the 1980s. Two oil shocks hit in 1994. [1]", served=["1994_p0007"],
             citations=cite("1994_p0007")),
        resp("u", REFUSAL),
    ], CASES, "dev")


def test_quality_line_numbers() -> None:
    line = sf.quality_line(quality_rows(), "ALL")
    # year_cov: a 2/3, n 0 (gold ids without year prefix: `years` fallback, nothing cited), p 1  -> mean 55%; multi-year: a, n
    assert "n= 4" in line
    assert "year coverage 56% (n=3), multi-year 33% (n=2)" in line
    # multi-sentence answers: a (dup 1/2) and p (dup 0, its gap sentence counts as content); the refusals have one sentence
    assert "dup sentences 25.0% of multi-sentence answers (n=2), 1 repeat" in line
    assert "refused with gold served 1/3" in line
    # gold served: a 3 (unused 1), n 1 (unused 1), p 1 (unused 0)
    assert "gold served but not cited 2/5 (40%)" in line
    assert "partial gap+content 1/1" in line


def test_quality_line_on_empty_input_does_not_divide_by_zero() -> None:
    line = sf.quality_line([], "none")
    assert "year coverage - (n=0)" in line and "refused with gold served 0/0" in line and "partial gap+content 0/0" in line


# CLI ---------------------------------------------------------------------------------------------------------------

def test_main_prints_old_lines_first_then_the_quality_lines_and_writes_the_fields(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(sf, "load_cases", lambda name, final=False: list(CASES.values()))
    path = tmp_path / "responses.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in [
        resp("a", "He bought stocks in the crisis. [1]", served=GOLD, citations=cite("2008_p0002")),
        resp("u", REFUSAL)]) + "\n")
    sf.main([str(path)])
    out = capsys.readouterr().out.splitlines()
    old = [i for i, l in enumerate(out) if "claim coverage" in l]
    quality = [i for i, l in enumerate(out) if "| year coverage" in l]
    assert len(old) == 6 and len(quality) == 6                           # ALL + 5 groups, once in each block
    assert old[-1] < quality[0] and "year coverage" not in "".join(out[i] for i in old)
    assert out[quality[0] - 1].startswith("quality")
    assert out[old[0]].startswith("ALL (dev)") and out[quality[0]].startswith("ALL (dev)")
    rows = json.loads((tmp_path / "scores.json").read_text())
    assert {"year_cov", "dup_rate", "refused_with_gold", "unused_gold", "partial_ok"} <= set(rows[0])
    assert rows[0]["year_cov"] == pytest.approx(1 / 3)
