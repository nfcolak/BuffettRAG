"""Fake-scorer checks for grounded evidence planning (no model loads)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.evaluation.claim_validator import evidence_sentences
from src.generation.grounded.evidence import (
    _eligible_quantities, _parse_periods, build_plan, group_evidence, source_spans,
)
from src.generation.grounded.settings import GroundedSettings


def hit(text, year=1985, hid=None):
    return SimpleNamespace(id=hid or f"{year}_p0001", text=text, metadata={"year": year})


class KeywordScorer:
    """p = 0.9 when the text contains any boost word, else 0.05; records calls."""

    def __init__(self, boost=("earn", "insur")):
        self.boost, self.calls = boost, []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        return [0.9 if any(b in t.lower() for b in self.boost) else 0.05 for t in texts]


def plan_for(query, hits, *, history=(), scorer=None, **kw):
    scorer = scorer or KeywordScorer()
    return build_plan(query, hits, history=history, scorer=scorer,
                      settings=GroundedSettings(**kw)), scorer


TEXTS = [
    "Mr. Smith earned $5.5 million in 1985. We bought 3 papers.\nThe\nsecond   line follows here.  In 1986. 12 firms joined.",
    "Insurance Operations\n\nOur insurance float grew to $1,200 million.  GEICO did well!\n\n(1) Footnote text.",
    "We like it. Dr. J. K. Lee said so. Nothing else, however, matters at all today.",
]


@pytest.mark.parametrize("text", TEXTS)
def test_span_offsets_round_trip_and_match_legacy_boundaries(text):
    spans = source_spans(text, 3)
    assert all(s.text == text[s.start:s.end] and s.hit_index == 3 for s in spans)
    assert [s.start for s in spans] == sorted(s.start for s in spans)
    if "Operations" not in text:
        assert [" ".join(s.text.split()) for s in spans] == evidence_sentences(text)


def test_header_is_not_an_anchor_and_stays_inside_spanning_window():
    text = "Earnings were good this year for us.\n\nInsurance Operations\n\nInsurance float grew a lot again."
    assert [s.text for s in source_spans(text)] == [
        "Earnings were good this year for us.", "Insurance float grew a lot again."]
    assert source_spans(text, include_headers=True)[1].text == "Insurance Operations"
    plan, _ = plan_for("earnings float", [hit(text)])
    assert all("Insurance Operations" in u.window.text for u in plan.units)
    assert all(u.anchor.text != "Insurance Operations" for u in plan.units)
    assert all(u.window.text == text[u.window.start:u.window.end] for u in plan.units)


@pytest.mark.parametrize("query,expected", [
    ("What did Buffett say about insurance?", [("all", "general")]),
    ("What happened in 1985?", [("1985", "period")]),
    ("Compare 1985 and 1995 results", [("1985", "period"), ("1995", "period")]),
    ("How did it change in the 1980s?", [("1980s", "period")]),
    ("What about the eighties versus 2000s?", [("1980s", "period"), ("2000s", "period")]),
    ("Float from 1985-1990?", [("1985-1990", "period")]),
    ("1985 through 1988, then 1985 again", [("1985-1988", "period"), ("1985", "period")]),
    ("A 1985, 1986, 1987, 1988, 1989 survey", [(str(y), "period") for y in range(1985, 1990)]),
    ("He paid $1,985 for 12.1985 shares", [("all", "general")]),
])
def test_slot_parsing(query, expected):
    plan, _ = plan_for(query, [hit("Insurance earnings grew a lot this year.")])
    assert [(s.name, s.kind) for s in plan.slots] == expected


def test_too_many_periods_refuses_without_scoring():
    plan, scorer = plan_for("1980 1981 1982 1983 1984 results", [hit("Earnings grew a lot.")])
    assert plan.decision == "refuse" and plan.reasons == ["too_many_periods"]
    assert plan.units == [] and scorer.calls == []
    assert len(_parse_periods("1980 1981 1982 1983 1984")) == 5


def test_no_hits_refuses_with_no_context():
    plan, scorer = plan_for("What is float?", [])
    assert (plan.decision, plan.reasons, scorer.calls) == ("refuse", ["no_context"], [])


@pytest.mark.parametrize("query,history,needs", [
    ("How much did Berkshire earn?", (), True),
    ("What percent was the margin?", (), True),
    ("Why does Buffett avoid debt?", (), False),
    ("And what about GEICO?", [{"role": "user", "content": "How much were earnings and the price?"}], False),
])
def test_needs_number_from_original_question_only(query, history, needs):
    plan, _ = plan_for(query, [hit("Berkshire earnings were $5 million.")], history=history)
    assert plan.slots[0].needs_number is needs
    assert plan.original_query == query
    if history:
        assert plan.scoring_query != query and plan.scoring_query.startswith(query)
    else:
        assert plan.scoring_query == query


def test_eligible_quantities_skip_years_pages_ordinals_markers_dates():
    text = "(1) In 1985, on December 31, page 12, the 3rd firm paid $5 million, 7% and 40 people."
    from decimal import Decimal
    assert _eligible_quantities(text) == {
        (Decimal(5000000), "$"), (Decimal(7), "percent"), (Decimal(40), "number")}


def test_dedupe_same_hit_only_and_keeps_different_number_years():
    same = "Our insurance earnings were stable and good for owners in "
    text = f"{same}1985. {same}1985. {same}1986. Padding words remain here today."
    plan, _ = plan_for("insurance earnings", [hit(text)], MAX_PER_HIT=2, MAX_UNITS=6)
    anchors = [u.anchor.text for u in plan.units]
    assert len(anchors) == len(set(anchors))  # exact repeat dropped
    assert any("1985" in a for a in anchors) and any("1986" in a for a in anchors)
    assert plan.trace["dropped"]["dedupe"] >= 1
    # identical text in different hits is never deduped
    other = "Our insurance earnings were stable and good for owners."
    plan, _ = plan_for("insurance earnings", [hit(other, hid="a"), hit(other, hid="b")])
    assert {u.passage_id for u in plan.units} == {"a", "b"}


def test_max_per_hit_and_max_units_and_floor():
    sentences = " ".join(f"Insurance earnings item number {chr(97 + i)} was recorded here." for i in range(8))
    plan, _ = plan_for("insurance earnings", [hit(sentences)], MAX_PER_HIT=2)
    assert len(plan.units) == 2
    hits = [hit(f"Insurance earnings note {chr(97 + i)} was recorded here.", hid=f"h{i}") for i in range(9)]
    plan, _ = plan_for("insurance earnings", hits, MAX_UNITS=5)
    assert len(plan.units) == 5
    plan, _ = plan_for("insurance earnings", hits[:2] + [hit("Nothing relevant lives in this sentence.")])
    assert len(plan.units) == 2  # fill never admits windows below T_SLOT


def test_selection_is_deterministic_and_slot_reserve_covers_each_period():
    hits = [hit(f"Insurance earnings note {i} for the year was strong here.", year=1990, hid=f"n{i}")
            for i in range(10)]
    hits.append(hit("Only quiet remark about 1985 with some earnings noted.", year=1985, hid="old"))
    scorer = KeywordScorer()
    first, _ = plan_for("Compare 1985 and 1990", hits, scorer=scorer, MAX_WINDOWS=8, SLOT_RESERVE=2)
    second, _ = plan_for("Compare 1985 and 1990", hits, scorer=KeywordScorer(), MAX_WINDOWS=8, SLOT_RESERVE=2)
    assert [u.eid for u in first.units] == [u.eid for u in second.units]
    assert first.trace["candidates"] == second.trace["candidates"]
    assert len(scorer.calls) == 1 and len(scorer.calls[0][1]) <= 8
    assert first.decision == "answer" and set(first.groups) == {"1985", "1990"}
    assert [u.passage_id for u in first.units if u.letter_year == 1985] == ["old"]
    assert first.slots[0].filled_by == [u.eid for u in first.units if u.letter_year == 1985]
    assert all(set(first.groups[s.name]) >= set(s.filled_by) for s in first.slots)


def test_missing_period_gives_partial_and_missing_quantity_reason():
    hits = [hit("Insurance earnings were strong in the year.", year=1990)]
    plan, _ = plan_for("Compare 1985 and 1990", hits)
    assert plan.decision == "partial" and plan.reasons == ["missing_period"]
    plan, _ = plan_for("How much did insurance earnings change?", hits)
    assert plan.decision == "refuse" and plan.reasons == ["missing_quantity"]


def test_low_relevance_refuses():
    plan, _ = plan_for("insurance earnings", [hit("Unrelated sentence about nothing here.")])
    assert plan.decision == "refuse" and plan.reasons == ["low_relevance"]


def test_group_evidence_one_numbered_sentence_each_with_hit_mapping():
    hits = [
        hit("Filler line without a hit here. Insurance earnings rose to $5 million.   Trailing "
            "sentence stays\nwrapped nicely.", year=1985, hid="a"),
        hit("Other passage mentions insurance earnings clearly. Another sentence about nothing.",
            year=1990, hid="b"),
    ]
    plan, _ = plan_for("How much did insurance earnings rise?", hits, MAX_PER_HIT=1)
    evidence = group_evidence(plan, "all", hits)
    assert [e.local_index for e in evidence] == list(range(len(evidence)))
    assert all(e.text == " ".join(e.text.split()) and e.text.endswith(".") for e in evidence)
    assert len({(e.hit_index, e.source_span.start) for e in evidence}) == len(evidence)
    for e in evidence:
        assert e.source_span.text == hits[e.hit_index].text[e.source_span.start:e.source_span.end]
        assert " ".join(e.source_span.text.split()) == e.text
        assert e.id == hits[e.hit_index].id and e.metadata == {"year": hits[e.hit_index].metadata["year"]}
    texts = [e.text for e in evidence]
    assert "Trailing sentence stays wrapped nicely." in texts
    assert texts.index("Insurance earnings rose to $5 million.") < texts.index(
        "Other passage mentions insurance earnings clearly.")
    with pytest.raises(KeyError):
        group_evidence(plan, "1985", hits)


def test_group_evidence_period_groups_do_not_mix_years():
    hits = [hit("Insurance earnings were high.", year=1985, hid="a"),
            hit("Insurance earnings were low.", year=1990, hid="b")]
    plan, _ = plan_for("Compare 1985 and 1990", hits)
    assert [e.hit_index for e in group_evidence(plan, "1985", hits)] == [0]
    assert [e.hit_index for e in group_evidence(plan, "1990", hits)] == [1]
    changed = [hit("Different text entirely.", year=1985, hid="a"), hits[1]]
    with pytest.raises(ValueError):
        group_evidence(plan, "1985", changed)


def test_scorer_length_mismatch_raises():
    class Bad:
        def score(self, query, texts):
            return [0.9]
    with pytest.raises(ValueError):
        plan_for("insurance earnings", [hit("Insurance earnings rose. Earnings stayed strong. Float grew too.")],
                 scorer=Bad())
