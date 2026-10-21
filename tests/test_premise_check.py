"""finance-answer-quality group 2: `plan.premise_note` on synthetic stores and plans, then on the real knowledge layer.

No model, no network. Event and topic names are general (a housing bust, airlines, inflation, the oil shocks).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.knowledge import KnowledgeStore
from src.query import QueryPlan, plan_query
from src.query.plan import premise_note

REAL = Path(__file__).resolve().parents[1] / "data" / "knowledge"
OIL_80S = "What did Buffett say about the energy crisis during the 1980s?"
OIL_NOTE = ("The oil shocks were in 1973-74 and 1979; the letters from the 1980s do not discuss them, "
            "but the 1994 and 2023 letters mention them.")


def _ann(year, n, summary="", **tags):
    row = {"id": f"{year}_p{n:04d}", "scope": "financial", "has_personal": False, "is_table": False, "sectors": [],
           "themes": [], "events": [], "companies": [], "periods": [year], "summary": summary}
    row.update(tags)
    return row


EVENTS = {"events": {
    "housing_bust": {"description": "A housing bust", "event_years": [2007, 2008, 2009],
                     "letter_years": [2008, 2009, 2010], "aliases": ["housing bust"]},
    "port_strike": {"description": "A port strike", "event_years": [1983], "letter_years": [], "aliases": ["port strike"]},
}}
ANNOTATIONS = [
    _ann(2008, 1, events=["housing_bust"]),
    _ann(2012, 1, events=["housing_bust"]),                      # a retrospective passage in a later letter
    _ann(1989, 1, sectors=["airlines_aviation"]),
    _ann(1993, 1, sectors=["airlines_aviation"]),
    _ann(1980, 1, themes=["inflation"]),
    _ann(1981, 1, themes=["inflation"]),
    _ann(1982, 1, themes=["inflation"]),
    _ann(1994, 1, "Berkshire ignores forecasts; the past 30 years brought wars and oil shocks."),
    _ann(2023, 1, "Shale lifted U.S. output; oil prices no longer depend on foreign supply."),
    _ann(1990, 2, "A summary about the oil shockwave of a pipe.", sectors=["banking"]),
]


@pytest.fixture(scope="module")
def store():
    return KnowledgeStore(annotations=ANNOTATIONS, events=EVENTS)


@pytest.fixture(scope="module")
def real():
    return KnowledgeStore.load(REAL)


@pytest.fixture(autouse=True)
def _flag_unset(monkeypatch):
    monkeypatch.delenv("FINANCE_PREMISE_CHECK", raising=False)


def _note(query, store, **kw):
    return plan_query(query, store, premise_check=True, **kw).premise_note


def _explicit(*spans, **kw):
    return QueryPlan(query="q", year_source="explicit", year_spans=tuple(spans), **kw)


# --- the setting ---
def test_premise_note_is_off_by_default(store):
    plan = plan_query(OIL_80S, store)
    assert plan.premise_note is None
    assert plan == plan_query(OIL_80S, store, premise_check=False)


def test_setting_reads_the_environment_at_call_time(store, monkeypatch):
    monkeypatch.setenv("FINANCE_PREMISE_CHECK", "1")
    assert plan_query(OIL_80S, store).premise_note == OIL_NOTE
    assert plan_query(OIL_80S, store, premise_check=False).premise_note is None
    monkeypatch.setenv("FINANCE_PREMISE_CHECK", "0")
    assert plan_query(OIL_80S, store).premise_note is None
    assert plan_query(OIL_80S, store, premise_check=True).premise_note == OIL_NOTE


def test_flag_on_changes_only_the_note(store):
    for query in (OIL_80S, "What did Buffett say about the housing bust in the 1990s?", "What did he say about bonds?"):
        off, on = plan_query(query, store, premise_check=False), plan_query(query, store, premise_check=True)
        assert {**on.__dict__, "premise_note": None} == off.__dict__


# --- events with real years ---
def test_event_outside_the_asked_period_names_real_years_and_letters(store):
    assert _note("What did Buffett say about the housing bust in the 1990s?", store) == (
        "The housing bust was in 2007-09; the letters from the 1990s do not discuss it, "
        "but the 2008-10 and 2012 letters mention it.")


def test_event_without_letters_gets_the_corrected_period_only(store):
    assert _note("What did Buffett say about the port strike in 1999?", store) == (
        "The port strike was in 1983; the letters from 1999 do not discuss it.")


def test_letter_of_a_later_year_that_carries_the_event_is_no_false_premise(store):
    assert _note("What did Buffett say about the housing bust in his 2010 letter?", store) is None
    assert _note("What did Buffett say about the housing bust in 2012?", store) is None  # tagged look-back


def test_event_that_happened_in_the_asked_years_needs_no_note(store):
    assert _note("What did Buffett say about the housing bust in 2007?", store) is None
    assert _note("What did Buffett say about the housing bust in the 2000s?", store) is None
    assert _note("What did Buffett say about the housing bust?", store) is None  # no named period


def test_one_of_several_asked_periods_may_carry_the_event(store):
    assert _note("Compare the housing bust in 1995 and 2009", store) is None


# --- events the store has no id for ---
def test_oil_shocks_asked_for_the_eighties(store):
    assert _note(OIL_80S, store) == OIL_NOTE


def test_oil_shocks_without_mentions_in_the_store():
    assert _note(OIL_80S, KnowledgeStore.empty()) == (
        "The oil shocks were in 1973-74 and 1979; the letters from the 1980s do not discuss them.")


def test_oil_shocks_in_their_own_decade_or_a_decade_that_mentions_them(store):
    assert _note("What did Buffett say about the oil crisis in the 1970s?", store) is None
    assert _note("Did Buffett discuss the oil embargo in the 1990s?", store) is None  # the 1994 summary names oil shocks


def test_oil_shocks_word_boundaries(store):
    assert store.letter_years_mentioning(["oil shock"]) == [1994]  # not the 1990 "oil shockwave"
    assert store.letter_years_mentioning(["oil price", "oil shock"]) == [1994, 2023]
    assert store.letter_years_mentioning([]) == []


# --- sector and theme tags ---
def test_sector_with_no_tagged_passage_in_the_asked_years(store):
    assert _note("What did Buffett say about airlines in 2005?", store) == (
        "The letters from 2005 do not discuss airlines, but the 1989 and 1993 letters do.")
    assert _note("What did Buffett say about airlines in 1993?", store) is None


def test_theme_with_no_tagged_passage_in_the_asked_years(store):
    assert _note("What did Buffett say about inflation in the 2000s?", store) == (
        "The letters from the 2000s do not discuss inflation, but the 1980-82 letters do.")
    assert _note("What did Buffett say about inflation in the early 1980s?", store) is None


def test_topic_that_no_letter_carries_has_nothing_to_name(store):
    assert _note("What did Buffett say about railroads in 2005?", store) is None


def test_listed_letter_years_are_capped():
    many = KnowledgeStore(annotations=[_ann(y, 1, sectors=["airlines_aviation"]) for y in range(1980, 2000, 2)])
    note = _note("What did Buffett say about airlines in 2010?", many)
    assert note == ("The letters from 2010 do not discuss airlines, but the "
                    "1980, 1982, 1984, 1986, 1988, 1990 and other letters do.")


# --- guards against a period that was never asked ---
def test_year_before_the_first_letter_is_no_period_claim(store):
    assert _note("What did Buffett say about airlines in 1972?", store) is None


def test_company_and_figure_questions_are_not_judged_by_sector_tags(store):
    assert _note("What did Buffett say about airlines in 2005?", store) is not None
    assert premise_note(_explicit((2005, 2005), sectors=("airlines_aviation",), companies=("Some Airline",)), store) is None
    assert premise_note(_explicit((2005, 2005), sectors=("airlines_aviation",), numeric=True), store) is None


def test_year_that_dates_an_event_is_not_an_asked_period(store):
    assert _note("How did Buffett assess the 2005 bankruptcy of an airline?", store) is None
    assert _note("What did he say about airlines in 2005, after the 2003 bankruptcy of one?", store) is not None


# --- synthetic plans ---
def test_synthetic_plan_event_note_formats_the_asked_period(store):
    plan = _explicit((1980, 1989), events=("housing_bust",))
    assert premise_note(plan, store).startswith("The housing bust was in 2007-09; the letters from the 1980s do not")
    assert "from 1990-1993 do not" in premise_note(_explicit((1990, 1993), events=("housing_bust",)), store)
    assert "from 1980 and 1995 do not" in premise_note(_explicit((1980, 1980), (1995, 1995), events=("housing_bust",)), store)
    assert premise_note(_explicit((1995, 1995), (2009, 2009), events=("housing_bust",)), store) is None


def test_synthetic_plan_needs_a_period_the_question_named(store):
    assert premise_note(QueryPlan(query="q", events=("housing_bust",)), store) is None
    event_sourced = QueryPlan(query="q", events=("housing_bust",), year_source="event", year_spans=((1990, 1999),))
    assert premise_note(event_sourced, store) is None
    personal = _explicit((1980, 1989), events=("housing_bust",), scope="personal")
    assert premise_note(personal, store) is None


def test_synthetic_plan_with_an_unknown_event_id_or_empty_store():
    assert premise_note(_explicit((1980, 1989), events=("no_such_event",)), KnowledgeStore(events=EVENTS)) is None
    assert premise_note(_explicit((1980, 1989), events=("housing_bust",)), KnowledgeStore.empty()) is None
    assert premise_note(_explicit((1980, 1989), events=("housing_bust",)), None) is None


def test_synthetic_plan_theme_note(store):
    plan = _explicit((2001, 2003), themes=("inflation",))
    assert premise_note(plan, store) == "The letters from 2001-2003 do not discuss inflation, but the 1980-82 letters do."


# --- the real knowledge layer ---
def test_real_layer_oil_shocks_in_the_eighties(real):
    note = _note(OIL_80S, real)
    assert note.startswith("The oil shocks were in 1973-74 and 1979; the letters from the 1980s do not discuss them")
    assert note.endswith("letters mention them.") and "1994" in note and "2023" in note


def test_real_layer_financial_crisis_in_the_nineties(real):
    note = _note("What did Buffett say about the 2008 financial crisis in the 1990s?", real)
    assert note.startswith("The 2008 financial crisis was in 2007-09; the letters from the 1990s do not discuss it, but the ")
    event = real.events()["financial_crisis_2008"]
    years = sorted(set(event["letter_years"]) | set(real.letter_years_with(event="financial_crisis_2008")))
    assert note.split("but the ", 1)[1].startswith(str(years[0])) and note.endswith("letters mention it.")


def test_real_layer_questions_without_a_premise_error(real):
    for query in (
            "What did Buffett say about the 2008 financial crisis?",
            "What did Buffett say about the 2008 financial crisis in his 2010 letter?",
            "What did Buffett say about inflation in the 1970s?",
            "What did Buffett say about the oil crisis in the 1970s?",
            "How did Buffett assess the 2023 collapse of a regional bank?",
            "What did Buffett say about insurance float in the 1990s?",
            "Which sector did Buffett speak most positively about?"):
        assert _note(query, real) is None, query


def test_real_layer_tagged_event_years_match_the_store(real):
    years = real.letter_years_with(event="financial_crisis_2008")
    assert years and years == sorted(set(years)) and all(1977 <= y <= 2024 for y in years)
    assert real.letter_years_with() == []
    assert real.letter_years_with(sector="no_such_sector") == []
