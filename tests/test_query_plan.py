"""finance-query-pipeline group 2: the deterministic query planner (fixtures only, no model, no network)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.knowledge import KnowledgeStore
from src.query import QueryPlan, plan_query
from src.query.plan import SCOPE_VOTE_PERSONAL_SHARE

FIX = Path(__file__).parent / "fixtures" / "knowledge_mini"


@pytest.fixture(scope="module")
def store():
    return KnowledgeStore.load(FIX)


def _plan(query, store, **kw):
    return plan_query(query, store, **kw)


# (query, scope, kind, year_range, year_hard)
CORE = [
    ("Did Buffett say anything about the oil crisis in the 80s?", "financial", "period", (1980, 1989), False),
    ("Give me Buffett's insights related to the 2008 crisis", "financial", "period", None, False),
    ("In 1999, does Buffett suggest any sector that has potential to grow substantially?", "financial", "period",
     (1999, 1999), False),
    ("Across all of these letters, which sector did Buffett speak most positively about?", "financial",
     "corpus_wide", None, False),
    ("What did Buffett write in the 1987 letter about the market crash?", "financial", "specific", (1987, 1987), True),
    ("In the 2003 annual letter, what does he say about derivatives?", "financial", "specific", (2003, 2003), True),
    ("What does the 1995 shareholder letter say about See's Candies?", "financial", "specific", (1995, 1995), True),
    ("What did Buffett think of insurance in the early 1990s?", "financial", "period", (1990, 1993), False),
    ("What was his view on bonds in the mid 1980s?", "financial", "period", (1983, 1986), False),
    ("How did he describe the market in the late 1990s?", "financial", "period", (1996, 1999), False),
    ("What did Buffett say about inflation in the 1970s?", "financial", "period", (1970, 1979), False),
    ("What did Buffett say about newspapers after 2008?", "financial", "period", (2009, 2024), False),
    ("Has Buffett written about airlines since 2010?", "financial", "period", (2010, 2024), False),
    ("What did he say about banks before 1990?", "financial", "period", (1977, 1989), False),
    ("What was Buffett's opinion on utilities between 2000 and 2010?", "financial", "period", (2000, 2010), False),
    ("What did he say about railroads from 1995 to 2005?", "financial", "period", (1995, 2005), False),
    ("Berkshire's view on goodwill 1999-2003?", "financial", "period", (1999, 2003), False),
    ("What did Buffett think about retailers in the '90s?", "financial", "period", (1990, 1999), False),
    ("What does he say about technology stocks in the 2000s?", "financial", "period", (2000, 2009), False),
    ("What is Buffett's view on share buybacks?", "financial", "specific", None, False),
    ("Explain what float is.", "financial", "specific", None, False),
    ("How did Buffett's view of USAir change between 1989 and 1996?", "financial", "trend", (1989, 1996), False),
    ("How has Buffett's attitude toward gold evolved?", "financial", "trend", None, False),
    ("How did his thinking about dividends shift over time?", "financial", "trend", None, False),
    ("Compare how Buffett saw insurance in 1985 and in 2005.", "financial", "comparison", (1985, 2005), False),
    ("Compare Buffett's views on the 80s and the 2000s.", "financial", "comparison", (1980, 2009), False),
    ("Is Coca-Cola a better business than airlines according to Buffett?", "financial", "comparison", None, False),
    ("How much float did Berkshire have in 1999?", "financial", "numeric", (1999, 1999), False),
    ("What was the book value per share in 1990?", "financial", "numeric", (1990, 1990), False),
    ("How many employees did Berkshire have in 2005?", "financial", "numeric", (2005, 2005), False),
    ("What was the net worth of Berkshire at yearend 1985?", "financial", "numeric", (1985, 1985), False),
    ("What percentage of earnings did insurance contribute?", "financial", "numeric", None, False),
    ("Which industries did Buffett always avoid over the years?", "financial", "corpus_wide", None, False),
    ("Which mistake does Buffett return to again and again?", "financial", "corpus_wide", None, False),
    ("Throughout the letters, what does he say about leverage?", "financial", "corpus_wide", None, False),
    ("What is the most important lesson Buffett repeats?", "financial", "corpus_wide", None, False),
    ("How old was Mrs. B when she opened her carpet store?", "personal", "specific", None, False),
    ("What does Buffett eat for breakfast?", "personal", "specific", None, False),
    ("Tell me a joke Buffett told about golf.", "personal", "specific", None, False),
    ("Who is Buffett's wife and how many children does he have?", "personal", "specific", None, False),
    ("Which hotel should I book for the annual meeting weekend?", "personal", "specific", None, False),
    ("How is Buffett's health these days?", "personal", "specific", None, False),
    ("What does Buffett say about health insurance companies?", "financial", "specific", None, False),
    ("What does Buffett think about airlines, and how old is his wife?", "mixed", "specific", None, False),
    ("What did Buffett say about GEICO in 1996, and what does he eat for breakfast?", "mixed", "period",
     (1996, 1996), False),
]


@pytest.mark.parametrize("query,scope,kind,year_range,year_hard", CORE)
def test_scope_kind_years(store, query, scope, kind, year_range, year_hard):
    plan = _plan(query, store, scope_vote=lambda q: 0.0)
    assert (plan.scope, plan.kind, plan.year_hard) == (scope, kind, year_hard)
    if year_range is not None:
        assert plan.year_range == year_range
    elif plan.scope == "financial" and "2008 crisis" not in query:
        assert plan.year_range is None


def test_case_table_has_at_least_forty_cases():
    assert len(CORE) >= 40


def test_the_2008_crisis_maps_to_the_event_and_letters(store):
    plan = _plan("Give me Buffett's insights related to the 2008 crisis", store)
    assert plan.events == ("financial_crisis_2008",)
    assert plan.year_range[0] == 2007 and plan.year_range[1] >= 2010
    assert not plan.year_hard and plan.years == ()  # the "2008" inside the alias is not a separate year


@pytest.mark.parametrize("phrase", ["the great recession", "the global financial crisis", "the credit crisis",
                                    "the subprime meltdown"])
def test_crisis_phrasings_find_the_event(store, phrase):
    plan = _plan(f"What did Buffett learn from {phrase}?", store)
    assert plan.events == ("financial_crisis_2008",)


@pytest.mark.parametrize("phrase,event", [("9/11", "sept11_2001"), ("black monday", "crash_1987"),
                                          ("the dot-com bubble", "dotcom_bubble"),
                                          ("Volcker", "high_inflation_late70s_early80s")])
def test_user_event_aliases(store, phrase, event):
    plan = _plan(f"What does Buffett say about {phrase}?", store)
    assert event in plan.events and plan.year_range is not None


@pytest.mark.parametrize("phrase", ["oil crisis", "oil shock", "energy crisis"])
def test_oil_crisis_has_no_event_but_years_and_terms(store, phrase):
    plan = _plan(f"What did Buffett say about the {phrase}?", store)
    assert plan.events == ()
    assert plan.year_range == (1973, 1982)
    assert {"oil", "energy", "OPEC", "gasoline"} <= set(plan.terms)


def test_explicit_decade_beats_the_alias_default_range(store):
    plan = _plan("Did Buffett say anything about the oil crisis in the 80s?", store)
    assert plan.year_range == (1980, 1989) and "oil" in plan.terms and "oil_gas" in plan.sectors


def test_letter_alias_from_events_json(store):
    plan = _plan("What did Buffett say about Hurricane Andrew?", store)
    assert plan.events == ("hurricane_andrew_1992",)


def test_decade_spellings_agree(store):
    ranges = {q: _plan(f"What did he say about banks in {q}?", store).year_range
              for q in ("the 80s", "the 1980s", "the '80s", "1980s")}
    assert set(ranges.values()) == {(1980, 1989)}


def test_year_hard_only_for_letter_questions(store):
    assert _plan("What did the 1999 letter say about Internet stocks?", store).year_hard
    assert _plan("In the letter for 1999, what did he say about Internet stocks?", store).year_hard
    assert not _plan("In 1999, what did he say about Internet stocks?", store).year_hard
    assert not _plan("What happened in the 1999 market?", store).year_hard
    assert not _plan("Compare the 1999 letter with the 2001 letter.", store).year_hard


def test_sectors_companies_themes(store):
    plan = _plan("How did USAir and GEICO affect Berkshire's float and insurance results?", store)
    assert set(plan.companies) == {"USAir", "GEICO"}
    assert {"insurance"} <= set(plan.sectors)
    assert "float_underwriting" in plan.themes


def test_taxonomy_description_words_find_sectors(store):
    assert "airlines_aviation" in _plan("What does he think of airlines?", store).sectors
    assert "oil_gas" in _plan("What did he think of OPEC?", store).sectors


def test_empty_store_still_plans():
    plan = plan_query("What did Buffett say about the 2008 crisis?", KnowledgeStore.empty())
    assert plan.scope == "financial" and plan.events == () and plan.year_range == (2007, 2010)
    assert plan_query("hi", None).kind == "specific"


def test_mixed_question_drops_the_personal_clause(store):
    plan = _plan("What does Buffett think about airlines, and how old is his wife?", store)
    assert plan.scope == "mixed"
    assert plan.financial_query == "What does Buffett think about airlines?"
    assert plan.personal_clauses == ("How old is his wife?",)
    assert plan.sectors == ("airlines_aviation",)


def test_personal_question_has_no_financial_query(store):
    plan = _plan("How old was Mrs. B when she opened her carpet store?", store)
    assert plan.scope == "personal" and plan.financial_query == "" and plan.scope_source == "rules"


def test_empty_query(store):
    plan = _plan("   ", store)
    assert plan.scope == "financial" and plan.kind == "specific" and plan.year_range is None


# --- scope vote ---
class Vote:
    def __init__(self, share):
        self.share, self.calls = share, []

    def __call__(self, query):
        self.calls.append(query)
        return self.share


@pytest.mark.parametrize("query", [
    "How old was Mrs. B when she opened her carpet store?",
    "What does Buffett think about airlines, and how old is his wife?",
    "Which hotel is closest to the annual meeting?",
    "What does Buffett eat for breakfast?",
])
def test_vote_not_called_when_rules_decide(store, query):
    vote = Vote(0.0)
    plan = _plan(query, store, scope_vote=vote)
    assert vote.calls == [] and plan.scope_source == "rules" and plan.vote_share is None
    assert plan.scope in ("personal", "mixed")


@pytest.mark.parametrize("query", [
    "What does Buffett think about airlines?",
    "How much float did Berkshire have in 1999?",
    "Which wonderful thing did Buffett admire in Omaha?",
])
def test_vote_called_when_rules_are_silent(store, query):
    vote = Vote(0.0)
    plan = _plan(query, store, scope_vote=vote)
    assert vote.calls == [query] and plan.scope == "financial"
    assert plan.scope_source == "vote" and plan.vote_share == 0.0


def test_vote_threshold_is_inclusive(store):
    assert SCOPE_VOTE_PERSONAL_SHARE == 0.6
    q = "Which wonderful thing did Buffett admire in Omaha?"
    assert _plan(q, store, scope_vote=Vote(0.6)).scope == "personal"
    assert _plan(q, store, scope_vote=Vote(0.9)).scope == "personal"
    assert _plan(q, store, scope_vote=Vote(0.5)).scope == "financial"
    assert _plan(q, store, scope_vote=Vote(0.6)).financial_query == ""


def test_vote_failure_falls_back_to_financial(store):
    def boom(query):
        raise RuntimeError("retrieval down")

    plan = _plan("What does Buffett think about airlines?", store, scope_vote=boom)
    assert plan.scope == "financial" and plan.scope_source == "default" and plan.vote_share is None


def test_no_vote_given_is_fine(store):
    plan = _plan("What does Buffett think about airlines?", store)
    assert plan.scope == "financial" and plan.scope_source == "default"


def test_plan_is_frozen(store):
    plan = _plan("What does Buffett think about airlines?", store)
    assert isinstance(plan, QueryPlan)
    with pytest.raises(Exception):
        plan.scope = "personal"  # type: ignore[misc]


# --- finance-dev-fix-1: scope rules and vote, year spans, event years, company aliases, superlatives ---
@pytest.mark.parametrize("query", [
    "How did the Blumkin family's furniture business grow its sales after Berkshire bought it?",
    "What did the founder's company earn in its first year under Berkshire?",
    "At the annual meeting, what did Buffett say about Apple's earnings?",
])
def test_a_business_figure_is_financial_next_to_a_founder_family_or_meeting(store, query):
    plan = _plan(query, store, scope_vote=Vote(0.0))
    assert plan.scope == "financial"
    assert _plan(query, store, scope_vote=Vote(0.9)).scope == "financial"  # the vote cannot make it personal


@pytest.mark.parametrize("query,financial", [
    ("How did the company's revenues grow under Berkshire, and what personal setback led its founder to start it?",
     "How did the company's revenues grow under Berkshire?"),
    ("How much did Nebraska Furniture Mart earn in 1990, and how old was Mrs. B when she founded it?",
     "How much did Nebraska Furniture Mart earn in 1990?"),
])
def test_the_mixed_split_keeps_the_financial_clause(store, query, financial):
    plan = _plan(query, store, scope_vote=Vote(0.0))
    assert plan.scope == "mixed" and plan.financial_query == financial


@pytest.mark.parametrize("query", [
    "What joke did Charlie tell at the 2015 annual meeting, and what did the crowd make of it?",
    "Tell me an anecdote Buffett shared about his first job.",
    "What did Buffett say about Charlie's longevity?",
    "How many people attended the annual meeting in 2010?",
    "How many shareholders showed up at the meeting in 1995, and how many came the year before?",
    "Did Buffett throw the first pitch at the ballgame during meeting weekend?",
    "What bargains could shareholders find shopping at the annual meeting?",
])
def test_jokes_anecdotes_attendance_ballgames_and_shopping_are_personal(store, query):
    vote = Vote(0.0)
    plan = _plan(query, store, scope_vote=vote)
    assert plan.scope == "personal" and plan.scope_source == "rules" and vote.calls == []


def test_the_vote_drops_only_the_clauses_without_a_financial_cue(store):
    query = "How did the company's revenues grow under Berkshire, and who inspired the founder?"
    plan = _plan(query, store, scope_vote=Vote(0.9))
    assert plan.scope == "mixed" and plan.scope_source == "vote"
    assert plan.financial_query == "How did the company's revenues grow under Berkshire?"
    assert plan.personal_clauses == ("Who inspired the founder?",)


@pytest.mark.parametrize("query,year_range,spans", [
    ("How did rates matter in the years around 2020?", (2019, 2021), ((2019, 2021),)),
    ("How did Berkshire build its Apple stake from 2016 onward?", (2016, 2024), ((2016, 2024),)),
    ("What made the jeweler strong at the end of the 1980s?", (1986, 1989), ((1986, 1989),)),
    ("Did he think inflation was over in the early-to-mid 1980s?", (1980, 1986), ((1980, 1986),)),
    ("How did Berkshire build its stake in 1988-89?", (1988, 1989), ((1988, 1989),)),
    ("How much did the float grow after the 1996 buyout?", (1997, 2024), ((1997, 2024),)),
    ("How did he describe banks in 2011, and what did he say in 2016?", (2011, 2016), ((2011, 2011), (2016, 2016))),
])
def test_year_spans(store, query, year_range, spans):
    plan = _plan(query, store, scope_vote=Vote(0.0))
    assert plan.year_range == year_range and plan.year_spans == spans and plan.year_source == "explicit"


def test_event_years_are_marked_as_event_years(store):
    plan = _plan("What did Buffett learn from the credit crisis?", store)
    assert plan.year_source == "event" and plan.year_spans == () and plan.year_range[0] == 2007


def test_a_year_the_event_cannot_have_drops_the_event(store):
    plan = _plan("How did Buffett react to the 1997 Asian financial crisis?", store)
    assert plan.events == () and plan.year_range == (1997, 1997)
    assert _plan("What did he say about the 1998 collapse of Long-Term Capital Management?", store).events == (
        "ltcm_1998",)


def test_event_phrases_do_not_create_themes(store):
    plan = _plan("What did Buffett say about the collapse of Long-Term Capital Management?", store)
    assert plan.events == ("ltcm_1998",) and "management_governance" not in plan.themes


def test_company_aliases_match_with_their_case():
    store = KnowledgeStore(companies={"companies": {"BNSF": {"aliases": ["BNSF", "Burlington Northern Santa Fe"]},
                                                    "Pilot Flying J": {"aliases": ["Pilot"]}}})
    assert plan_query("What did Berkshire pay for Burlington Northern Santa Fe?", store).companies == ("BNSF",)
    assert plan_query("Did Buffett ever work as a pilot for insurers?", store).companies == ()
    assert plan_query("Why did Berkshire buy Pilot?", store).companies == ("Pilot Flying J",)


@pytest.mark.parametrize("query,superlative", [
    ("Across all of these letters, which sector did Buffett speak most positively about?", "sector:positive"),
    ("Which company did Buffett praise the most?", "company:positive"),
    ("Which industry did he criticize most harshly across the letters?", "sector:negative"),
    ("Which businesses was Buffett most negative about?", "company:negative"),
    ("What kind of business does Buffett describe as the best kind to own?", ""),
    ("Which industries did Buffett always avoid over the years?", ""),
])
def test_superlatives(store, query, superlative):
    assert _plan(query, store).superlative == superlative


def test_numeric_intent_is_kept_for_every_kind(store):
    assert _plan("How did the book value per share change between 1990 and 1995?", store).numeric
    assert not _plan("What is Buffett's view on share buybacks?", store).numeric
