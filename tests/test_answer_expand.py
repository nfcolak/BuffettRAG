"""Neighbour-sentence expansion: verbatim only, capped, no duplicates, citation and list format kept."""
import re

from src.generation.answer_expand import expand_answer, is_prose_sentence
from src.storage import SearchHit

PASSAGE = (
    "Insurance float is money we hold but do not own and it has been very valuable to us. "
    "Our float grew from sixteen million dollars to many billions of dollars over the decades. "
    "We try to write policies only when the premiums are adequate for the risks involved. "
    "Charlie and I regard underwriting discipline as the essential condition for good float."
)


def _hits():
    return [SearchHit("p1", PASSAGE, {"year": 1990}, 0.0)]


def _sentences(text):
    return [s for s in re.split(r"(?<=\])\s+", text.strip()) if s]


def test_off_by_default_returns_answer_unchanged():
    answer = "Our float grew from sixteen million dollars to many billions of dollars over the decades. [1]"
    assert expand_answer(answer, _hits(), after=0, before=0) == answer


def test_adds_following_and_preceding_sentences_in_passage_order_with_same_marker():
    answer = "Our float grew from sixteen million dollars to many billions of dollars over the decades. [1]"
    out = expand_answer(answer, _hits(), after=1, before=1, max_sentences=12)
    assert out == (
        "Insurance float is money we hold but do not own and it has been very valuable to us. [1] "
        + answer + " "
        "We try to write policies only when the premiums are adequate for the risks involved. [1]"
    )


def test_every_added_sentence_is_verbatim_passage_text():
    answer = "Our float grew from sixteen million dollars to many billions of dollars over the decades. [1]"
    out = expand_answer(answer, _hits(), after=3, before=3, max_sentences=12)
    for sentence in _sentences(out):
        assert re.sub(r"\s*\[\d+\]$", "", sentence) in PASSAGE


def test_paraphrase_is_not_expanded():
    answer = "The float became much larger over the years, which helped Berkshire. [1]"
    assert expand_answer(answer, _hits(), after=2, before=2) == answer


def test_cap_limits_total_sentences():
    answer = "Our float grew from sixteen million dollars to many billions of dollars over the decades. [1]"
    out = expand_answer(answer, _hits(), after=3, before=3, max_sentences=3)
    assert len(_sentences(out)) == 3
    assert expand_answer(answer, _hits(), after=3, before=3, max_sentences=1) == answer


def test_no_duplicate_when_neighbour_already_in_answer():
    answer = (
        "Our float grew from sixteen million dollars to many billions of dollars over the decades. [1] "
        "We try to write policies only when the premiums are adequate for the risks involved. [1]"
    )
    out = expand_answer(answer, _hits(), after=1, before=0, max_sentences=12)
    sentences = _sentences(out)
    assert len(sentences) == len(set(sentences))
    assert out.count("We try to write policies") == 1
    # the first sentence's following sentence is already in the answer: it is not pulled in a second time
    assert out.startswith(answer)
    assert out.endswith("essential condition for good float. [1]")


def test_list_formatting_is_kept():
    answer = ("- Our float grew from sixteen million dollars to many billions of dollars over the decades. [1]\n"
              "- We try to write policies only when the premiums are adequate for the risks involved. [1]")
    out = expand_answer(answer, _hits(), after=0, before=1, max_sentences=12)
    lines = out.split("\n")
    assert len(lines) == 2 and all(line.startswith("- ") for line in lines)
    assert lines[0].startswith("- Insurance float is money we hold")


def test_citation_marker_of_multi_cite_sentence_is_reused():
    hits = [SearchHit("p0", "Unrelated passage about something else entirely, with some words.", {"year": 1990}, 0.0),
            SearchHit("p1", PASSAGE, {"year": 1990}, 0.0)]
    answer = "Our float grew from sixteen million dollars to many billions of dollars over the decades. [1,2]"
    out = expand_answer(answer, hits, after=1, before=0, max_sentences=12)
    assert "We try to write policies only when the premiums are adequate for the risks involved. [1,2]" in out


def test_table_like_and_heading_lines_are_skipped():
    passage = ("Our float grew from sixteen million dollars to many billions of dollars over the decades. "
               "1999 2000 2001 2002 2003 2004 2005 2006 2007 2008. "
               "INSURANCE OPERATIONS OVERVIEW. "
               "We try to write policies only when the premiums are adequate for the risks involved.")
    hits = [SearchHit("p1", passage, {"year": 1990}, 0.0)]
    answer = "Our float grew from sixteen million dollars to many billions of dollars over the decades. [1]"
    out = expand_answer(answer, hits, after=1, before=0, max_sentences=12)
    assert "1999 2000" not in out and "OVERVIEW" not in out
    assert "We try to write policies" in out
    assert not is_prose_sentence("INSURANCE OPERATIONS OVERVIEW.")


def test_period_label_prefix_is_kept():
    answer = "In 1990: Our float grew from sixteen million dollars to many billions of dollars over the decades. [1]"
    out = expand_answer(answer, _hits(), after=1, before=0, max_sentences=12)
    assert out.startswith("In 1990: Our float grew")
    assert "We try to write policies" in out
