"""finance-answer-quality group 4 (retrieval round 5): the plan-query rewrite (FINANCE_PLAN_QUERY), per-letter quotas
(FINANCE_LETTER_QUOTA), follow-up plan carry-over (FINANCE_FOLLOWUP_PLAN) and the gentler follow-up vote
(FINANCE_FOLLOWUP_VOTE). Synthetic stores and rankings only."""
from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from config import FINANCE_DEFAULTS, finance_setting
from src.knowledge import KnowledgeStore
from src.query.plan import QueryPlan, plan_query
from src.retrieval import decade_source as ds
from src.retrieval import finance_policy as fp
from src.retrieval import knowledge_source as ks
from src.retrieval import multiquery_source as mq
from src.retrieval.bm25 import _meta_matches
from src.retrieval.finance_policy import candidate_sources, finance_search
from src.services import ask_flow
from src.storage import SearchHit, StoredDoc

SETTINGS = ("FINANCE_PLAN_QUERY", "FINANCE_LETTER_QUOTA", "FINANCE_FOLLOWUP_PLAN", "FINANCE_FOLLOWUP_VOTE",
            "FINANCE_TABLE_CUE")
OTHER_SOURCES = ("FINANCE_SUMMARY_SOURCE", "FINANCE_TAGGED_SOURCE", "FINANCE_DIGEST_SOURCE", "FINANCE_STANCE_SOURCE",
                 "FINANCE_MULTI_QUERY", "FINANCE_DIGEST_QUERY", "FINANCE_DECADE_BLOCKS")
YEARS = range(1990, 2000)
PER_LETTER_DOCS = 6
THEMES = {"buybacks": "share repurchases. A tie-break note."}
SECTORS = {"insurance": "property-casualty insurers and reinsurance"}
EVENTS = {"crash_x": {"event_years": [1993], "letter_years": [1993, 1994], "aliases": ["the big crash", "black day"]}}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("FINANCE_K", "FINANCE_OVERFETCH", "FINANCE_TAG_DEPTH", "FINANCE_TAG_POOL", "FINANCE_MAX_PER_LETTER",
                 "FINANCE_MIN_LETTERS", "FINANCE_YEAR_BOOST_WEIGHT", "LLM_CONTEXT_PASSAGES",
                 "FINANCE_RERANK_CANDIDATES", "FINANCE_FOLLOWUP_HISTORY"):
        monkeypatch.delenv(name, raising=False)
    for name in (*OTHER_SOURCES, *SETTINGS):  # all off (three of them default on): a test turns on what it checks
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("RETRIEVAL_DEDUP", "0")
    monkeypatch.setenv("FINANCE_TAG_DEPTH", "30")  # = the shallow depth (fetch_k) at k=4: only ranks 0-29 are pooled
    monkeypatch.setattr(mq, "THEME_QUERY", "description")  # the fake rankings below key on the topic description


def _pid(year, n):
    return f"{year}_p{n:04d}"


def _docs(per_letter=PER_LETTER_DOCS):
    return [StoredDoc(id=_pid(y, n), text=f"text {y} {n}", metadata={"year": y}) for n in range(per_letter)
            for y in YEARS]


def _ann(pid, **kw):
    return {"id": pid, "scope": kw.pop("scope", "financial"), "is_table": kw.pop("is_table", False),
            "sectors": [], "themes": [], "events": [], "companies": [], "stance": [], "summary": "", **kw}


def _store(annotations=None):
    anns = annotations if annotations is not None else [_ann(d.id) for d in _docs()]
    return KnowledgeStore(annotations=anns, taxonomy={"themes": THEMES, "sectors": SECTORS}, digests=[],
                          events={"events": EVENTS})


class Retriever:
    """Hybrid ranking per query: the ids of the first `rankings` key the query contains (then the rest of the corpus),
    else corpus order (every letter's n=0 passage first, ranks 0-9, then n=1, 10-19, ...). Records every search."""

    def __init__(self, docs, rankings=None):
        self.docs, self.embedder, self.reranker = list(docs), None, None
        self.by_id = {d.id: d for d in self.docs}
        self.rankings = rankings or {}
        self.calls = []
        self._lock = threading.Lock()

    def _hybrid_for_filter(self, query, fetch_k, where):
        with self._lock:
            self.calls.append((query, where))
        first = next((ids for key, ids in self.rankings.items() if key in query), [])
        order = list(dict.fromkeys([*first, *self.by_id]))
        hits = [SearchHit(id=p, text=self.by_id[p].text, metadata=dict(self.by_id[p].metadata), score=1.0)
                for p in order]
        return [h for h in hits if not where or _meta_matches(h.metadata, where)][:fetch_k]


def _plan(query="Share repurchases at Berkshire", kind="specific", **kw):
    return QueryPlan(query=query, financial_query=query, kind=kind, **kw)


def _ids(hits):
    return [h.id for h in hits]


def _deep(year, n):  # ranks 30-59 of the corpus order: not pooled by finance_search at k=4
    assert n >= 3
    return _pid(year, n)


# --- settings ---
def test_the_round_5_settings_with_the_adopted_defaults(monkeypatch):
    # task 4.1: the follow-up carry-over (2), the gentler follow-up vote (7) and the table cue passed the adoption rule,
    # the plan-query rewrite and the per-letter topic quotas did not; FINANCE_MAX_PER_LETTER went from 3 to 6
    adopted = [0, 0, 2, 7, 1]
    assert [FINANCE_DEFAULTS[name] for name in SETTINGS] == adopted
    assert FINANCE_DEFAULTS["FINANCE_MAX_PER_LETTER"] == 6
    for name in SETTINGS:
        monkeypatch.delenv(name)
    assert [finance_setting(name) for name in SETTINGS] == adopted
    assert "FINANCE_PLAN_QUERY" not in [n for n, _ in candidate_sources()]


# --- FINANCE_TABLE_CUE ---
@pytest.mark.parametrize("query,kind,numeric,allowed", [
    ("What did the performance table of the 2011 letter show for 1977?", "period", False, True),
    ("How did the balance sheet of the group look at yearend?", "period", False, True),
    ("Compare the yearend float in 2006 with 2010: how big was each?", "comparison", True, True),
    ("How did the market value of the stock holdings change between the two stock lists?", "trend", False, True),
    ("How did Buffett's view of airlines change?", "trend", False, False),
    ("What does Buffett think about share repurchases?", "specific", False, False),
])
def test_tables_join_a_figure_question_of_any_kind_only_when_on(monkeypatch, query, kind, numeric, allowed):
    plan = _plan(query, kind=kind, numeric=numeric)
    assert fp.tables_allowed(_plan(query, kind="numeric"))  # a numeric plan always takes tables
    assert not fp.tables_allowed(plan)
    monkeypatch.setenv("FINANCE_TABLE_CUE", "1")
    assert fp.tables_allowed(plan) is allowed


def test_finance_search_keeps_a_figure_question_s_tables_only_when_on(monkeypatch):
    table = _pid(1995, 0)
    anns = [_ann(d.id, is_table=d.id == table) for d in _docs()]
    plan = _plan("What did the table in the 1995 letter show?", kind="period")
    hits, off = finance_search(plan, Retriever(_docs()), _store(anns), k=4)
    assert table in off["dropped_not_financial"] and table not in _ids(hits)
    monkeypatch.setenv("FINANCE_TABLE_CUE", "1")
    _, on = finance_search(plan, Retriever(_docs()), _store(anns), k=4)
    assert table not in on["dropped_not_financial"] and table in on["stage_ranked"]
    view = _plan("What does Buffett think about share repurchases?")
    _, still = finance_search(view, Retriever(_docs()), _store(anns), k=4)
    assert table in still["dropped_not_financial"]


def test_the_sources_follow_the_same_table_rule(monkeypatch):
    monkeypatch.setenv("FINANCE_PLAN_QUERY", "1")
    table, prose = _deep(1995, 5), _deep(1996, 5)
    anns = [_ann(d.id, is_table=d.id == table) for d in _docs()]
    retriever = Retriever(_docs(), {"buybacks:": [table, prose]})
    plan = _plan("What did the buyback table show?", themes=("buybacks",))
    assert _ids(mq.plan_query_source(plan, _store(anns), retriever, plan.query, 4))[:1] == [prose]
    monkeypatch.setenv("FINANCE_TABLE_CUE", "1")
    assert _ids(mq.plan_query_source(plan, _store(anns), retriever, plan.query, 4))[:2] == [table, prose]


def test_the_plan_query_source_is_registered_with_the_multi_query_sources(monkeypatch):
    names = [name for name, _ in ks.SOURCES]
    assert names.index("FINANCE_PLAN_QUERY") == names.index("FINANCE_DIGEST_QUERY") + 1
    assert names[-1] == "FINANCE_DECADE_BLOCKS"
    assert mq.plan_query_source in mq.FAMILY and "FINANCE_PLAN_QUERY" not in [n for n, _ in candidate_sources()]
    assert ks.SOURCES[names.index("FINANCE_PLAN_QUERY")] == mq.PLAN_SOURCES[0]
    monkeypatch.setenv("FINANCE_PLAN_QUERY", "1")
    assert "FINANCE_PLAN_QUERY" in [n for n, _ in candidate_sources()]


# --- FINANCE_PLAN_QUERY: the queries ---
def test_no_plan_queries_while_off():
    plan = _plan(themes=("buybacks",), events=("crash_x",))
    assert mq.plan_queries(plan, _store(), Retriever(_docs()), plan.query, 0) == []


def test_a_question_that_is_not_a_view_question_gets_its_topic_queries(monkeypatch):
    store, retriever = _store(), Retriever(_docs())
    plan = _plan(themes=("buybacks",), sectors=("insurance",))
    core = mq.financial_core(plan)
    assert mq.queries(plan, store, retriever, plan.query, 1, False) == []  # the multi-query source skips it
    assert mq.plan_queries(plan, store, retriever, plan.query, 1) == [f"{core} buybacks: share repurchases."]
    assert mq.plan_queries(plan, store, retriever, plan.query, 2) == [
        f"{core} buybacks: share repurchases.", f"{core} insurance: property-casualty insurers and reinsurance"]


def test_a_view_question_leaves_its_topics_to_the_multi_query_source(monkeypatch):
    store, retriever = _store(), Retriever(_docs())
    view = _plan("What does Buffett think about share repurchases?", themes=("buybacks",))
    monkeypatch.setenv("FINANCE_MULTI_QUERY", "1")
    assert mq.plan_queries(view, store, retriever, view.query, 2) == []
    monkeypatch.setenv("FINANCE_MULTI_QUERY", "0")  # nobody searches its topics otherwise
    assert mq.plan_queries(view, store, retriever, view.query, 2) == [
        f"{mq.financial_core(view)} buybacks: share repurchases."]


def test_an_event_query_carries_the_aliases_and_the_real_years(monkeypatch):
    store, retriever = _store(), Retriever(_docs())
    plan = _plan("What did Buffett do in the crash?", events=("crash_x",))
    core = mq.financial_core(plan)
    assert mq.event_query(core, store, "crash_x") == f"{core} the big crash black day 1993"
    monkeypatch.setattr(mq, "EVENT_ALIASES", 1)
    assert mq.event_query(core, store, "crash_x") == f"{core} the big crash 1993"
    assert mq.event_query(core, store, "unknown_event") == core
    monkeypatch.setenv("FINANCE_MULTI_QUERY", "1")  # a view question still gets its event query
    view = _plan("What did Buffett think about the crash?", events=("crash_x",))
    assert mq.plan_queries(view, store, retriever, view.query, 1) == [
        f"{mq.financial_core(view)} the big crash 1993"]


def test_topic_terms_without_a_store_event_make_one_query():
    plan = _plan("What happened in the oil crisis?", terms=("oil", "OPEC"), year_range=(1973, 1982),
                 year_source="event")
    assert mq.plan_queries(plan, _store(), Retriever(_docs()), plan.query, 1) == [
        f"{mq.financial_core(plan)} oil OPEC"]


# --- FINANCE_PLAN_QUERY: the source ---
def test_the_source_adds_only_new_passages_up_to_its_cap(monkeypatch):
    monkeypatch.setenv("FINANCE_PLAN_QUERY", "1")
    monkeypatch.setattr(mq, "PQ_CAP", 3)
    target = [_pid(1990, 0), _deep(1995, 5), _deep(1996, 5), _deep(1997, 5), _deep(1998, 5)]
    retriever = Retriever(_docs(), {"buybacks:": target})
    plan = _plan(themes=("buybacks",))
    hits = mq.plan_query_source(plan, _store(), retriever, plan.query, 4)
    assert _ids(hits) == [_deep(1995, 5), _deep(1996, 5), _deep(1997, 5)]  # 1990_p0000 is pooled already


def test_the_source_skips_what_the_multi_query_source_added(monkeypatch):
    monkeypatch.setenv("FINANCE_PLAN_QUERY", "1")
    monkeypatch.setenv("FINANCE_MULTI_QUERY", "1")
    a, b = _deep(1991, 5), _deep(1992, 5)
    earlier = lambda *args: [SearchHit(id=a, text="x", metadata={"year": 1991}, score=0.0)]  # noqa: E731
    monkeypatch.setattr(ks, "SOURCES", [("FINANCE_MULTI_QUERY", earlier), ("FINANCE_PLAN_QUERY", mq.plan_query_source)])
    monkeypatch.setattr(mq, "FAMILY", (earlier, mq.plan_query_source))
    retriever = Retriever(_docs(), {"buybacks:": [a, b]})
    plan = _plan(themes=("buybacks",))
    assert _ids(mq.plan_query_source(plan, _store(), retriever, plan.query, 4))[:1] == [b]


def test_no_recursion_when_the_multi_query_sources_are_off(monkeypatch):
    monkeypatch.setenv("FINANCE_PLAN_QUERY", "1")
    monkeypatch.setenv("FINANCE_DECADE_BLOCKS", "2")
    plan = _plan("How did share repurchases change?", kind="trend", themes=("buybacks",))
    retriever = Retriever(_docs(), {"buybacks:": [_deep(1995, 5)]})
    assert _deep(1995, 5) in _ids(mq.plan_query_source(plan, _store(), retriever, plan.query, 4))
    assert ds.pooled(plan, _store(), retriever, plan.query, 4) >= {_deep(1995, 5)}  # the decade source skips them


def test_finance_search_pools_the_plan_query_candidates_only_when_on(monkeypatch):
    target = _deep(1995, 5)
    plan = _plan(themes=("buybacks",))
    _, off = finance_search(plan, Retriever(_docs(), {"buybacks:": [target]}), _store(), k=4)
    assert target not in off["stage_pool"] and "FINANCE_PLAN_QUERY" not in off["source_candidates"]
    monkeypatch.setenv("FINANCE_PLAN_QUERY", "1")
    _, on = finance_search(plan, Retriever(_docs(), {"buybacks:": [target]}), _store(), k=4)
    assert target in on["stage_pool"] and on["source_candidates"]["FINANCE_PLAN_QUERY"] >= 1
    assert on["stage_pool"][:len(off["stage_pool"])] == off["stage_pool"] and on["source_errors"] == []


# --- FINANCE_LETTER_QUOTA ---
def _topic_store(per_year):
    """`per_year` tagged (buybacks) passages in each letter, plus one untagged passage per letter."""
    anns = [_ann(_pid(y, n), themes=["buybacks"]) for y, count in per_year.items() for n in range(count)]
    anns += [_ann(_pid(y, 90)) for y in YEARS]
    return KnowledgeStore(annotations=anns, taxonomy={"themes": THEMES, "sectors": SECTORS})


def test_topic_letters_count_the_servable_tagged_passages():
    anns = [_ann(_pid(1990, 0), themes=["buybacks"]), _ann(_pid(1990, 1), themes=["buybacks"], is_table=True),
            _ann(_pid(1991, 0), themes=["buybacks"], scope="personal"), _ann(_pid(1992, 0), sectors=["insurance"]),
            _ann(_pid(1992, 1), themes=["buybacks"], sectors=["insurance"])]
    store = KnowledgeStore(annotations=anns)
    assert fp.topic_letters(_plan(themes=("buybacks",)), store) == {1990: 1, 1992: 1}
    assert fp.topic_letters(_plan(themes=("buybacks",), sectors=("insurance",)), store) == {1990: 1, 1992: 2}
    assert fp.topic_letters(_plan(), store) == {}


def test_quotas_follow_the_topic_share_and_never_fall_below_the_cap(monkeypatch):
    monkeypatch.setenv("FINANCE_MAX_PER_LETTER", "3")  # the cap these numbers are written for (default 6 since round 5)
    store = _topic_store({1990: 12, 1991: 4, 1992: 4})  # 20 tagged passages
    plan = _plan(kind="trend", themes=("buybacks",))
    assert fp.letter_quotas(plan, store, 20) == {}  # off
    monkeypatch.setenv("FINANCE_LETTER_QUOTA", "1")
    assert fp.letter_quotas(plan, store, 20) == {1990: 12, 1991: 4, 1992: 4}
    assert fp.letter_quotas(plan, store, 10) == {1990: 6, 1991: 3, 1992: 3}  # never below the cap (3)
    assert fp.letter_quotas(_plan(kind="trend"), store, 20) == {}  # no topic


def test_quotas_count_only_the_letters_of_the_question_range(monkeypatch):
    monkeypatch.setenv("FINANCE_MAX_PER_LETTER", "3")  # the cap these numbers are written for (default 6 since round 5)
    monkeypatch.setenv("FINANCE_LETTER_QUOTA", "1")
    store = _topic_store({1990: 30, 1995: 6, 1996: 2})
    plan = _plan(kind="trend", themes=("buybacks",), year_range=(1994, 1999), year_spans=((1994, 1999),),
                 year_source="explicit")
    assert fp.letter_quotas(plan, store, 16) == {1995: 12, 1996: 4}


def test_a_dense_letter_takes_its_quota_in_a_diverse_question(monkeypatch):
    monkeypatch.setenv("FINANCE_MAX_PER_LETTER", "3")  # the cap these numbers are written for (default 6 since round 5)
    years = (1990, 1991, 1992, 1993, 1994)
    ranking = [SearchHit(_pid(1990, n), "x", {"year": 1990}, 1.0) for n in range(8)]
    ranking += [SearchHit(_pid(y, n), "x", {"year": y}, 1.0) for y in years[1:] for n in range(2)]
    store = _topic_store({1990: 8, 1991: 2, 1992: 2, 1993: 2, 1994: 2})

    class Ranked:
        reranker = None

        def _hybrid_for_filter(self, query, fetch_k, where):
            return [h for h in ranking if not where or _meta_matches(h.metadata, where)][:fetch_k]

    plan = _plan(kind="trend", themes=("buybacks",))
    off, debug_off = finance_search(plan, Ranked(), store, k=12)
    assert sum(1 for h in off if h.metadata["year"] == 1990) == 3 and "letter_quota" not in debug_off
    monkeypatch.setenv("FINANCE_LETTER_QUOTA", "1")
    on, debug_on = finance_search(plan, Ranked(), store, k=12)
    assert sum(1 for h in on if h.metadata["year"] == 1990) == 6 and debug_on["letter_quota"] == {1990: 6}
    assert len({h.metadata["year"] for h in on}) >= 4 and len(on) == 12
    specific = _plan(themes=("buybacks",))  # not diverse: no quota, no cap
    assert finance_search(specific, Ranked(), store, k=12)[0] == finance_search(
        specific, Ranked(), store, k=12)[0]


def test_diverse_ok_checks_each_letter_against_its_quota():
    hits = [SearchHit(_pid(1990, n), "x", {"year": 1990}, 1.0) for n in range(5)]
    hits.append(SearchHit(_pid(1991, 0), "x", {"year": 1991}, 1.0))
    assert not fp._diverse_ok(hits, 2, cap=3)
    assert fp._diverse_ok(hits, 2, cap=3, quota={1990: 5})
    assert not fp._diverse_ok(hits, 2, cap=3, quota={1991: 5})


# --- follow-up planning ---
PREVIOUS = "In the 1994 letter, how did GEICO grow its float?"
HISTORY = [{"role": "user", "content": PREVIOUS}, {"role": "assistant", "content": "It grew a lot."}]


def _follow_store():
    anns = [_ann("1994_p0001", companies=["GEICO"], sectors=["insurance"]), _ann("1999_p0001", companies=["Gillette"])]
    return KnowledgeStore(annotations=anns, taxonomy={"themes": {}, "sectors": SECTORS})


def _vote(share):
    return lambda text: share


def _with_history(query, history=HISTORY, share=0.1, store=None):
    from src.retrieval.query_expansion import build_followup_retrieval_query
    store = store or _follow_store()
    return ask_flow._plan_with_history(query, build_followup_retrieval_query(query, history), history, store,
                                       _vote(share))


def test_a_follow_up_without_years_or_topics_carries_the_previous_plan(monkeypatch):
    query = "Who ran the company at the time, and how large did the business become over the following years?"
    off = _with_history(query)
    assert off.year_range is None and off.companies == ()
    monkeypatch.setenv("FINANCE_FOLLOWUP_PLAN", "1")
    on = _with_history(query)
    assert on.year_range == (1994, 1994) and on.year_spans == ((1994, 1994),) and on.year_source == "explicit"
    assert on.year_hard is False  # the previous "1994 letter" becomes a boost
    assert on.companies == ("GEICO",) and on.sectors == ("insurance",)
    assert on.kind == off.kind and on.query == off.query  # the follow-up keeps its own kind and text


def test_the_follow_up_keeps_its_own_years_and_topics(monkeypatch):
    monkeypatch.setenv("FINANCE_FOLLOWUP_PLAN", "1")
    own = _with_history("How did Gillette's results look in the 1999 letter, compared with the first year?")
    assert own.companies == ("Gillette",) and own.year_range == (1999, 1999) and own.year_hard
    two = "How did the business do in 1999 and in 2001, judged against its own past record of results?"
    assert _with_history(two).year_spans == ((1999, 1999), (2001, 2001))
    monkeypatch.setenv("FINANCE_FOLLOWUP_PLAN", "2")  # 2: the previous letter becomes one more span
    both = _with_history(two)
    assert both.year_spans == ((1999, 1999), (2001, 2001), (1994, 1994)) and both.year_range == (1994, 2001)
    assert not both.year_hard
    assert _with_history("How did Gillette's results look in the 1999 letter, compared with the first year?"
                         ).year_spans == ((1999, 1999),)  # a "YYYY letter" question is never widened


def test_nothing_is_carried_from_a_previous_question_that_is_not_financial(monkeypatch):
    monkeypatch.setenv("FINANCE_FOLLOWUP_PLAN", "1")
    history = [{"role": "user", "content": "How old was Buffett in 1994?"}, {"role": "assistant", "content": "64."}]
    plan = _with_history("Who ran the company at the time, and how large did the business become over the years?",
                         history)
    assert plan.year_range is None
    assert _with_history("Who ran the company, and how large did it become over the years?", []).year_range is None


def test_carried_event_years_come_from_the_events_not_from_spans(monkeypatch):
    monkeypatch.setenv("FINANCE_FOLLOWUP_PLAN", "1")
    plan = replace(_plan("q"), year_range=(2007, 2011), year_source="event", events=("financial_crisis_2008",))
    before = replace(_plan("p"), year_range=(2004, 2004), year_spans=((2004, 2004),), years=(2004,), year_hard=True,
                     year_source="explicit", companies=("GEICO",))
    carried = ask_flow._carry_plan(plan, before, 1)
    assert carried.year_spans == ((2004, 2004),) and carried.year_range == (2004, 2011)
    assert carried.events == ("financial_crisis_2008",) and carried.companies == ()  # it has a topic of its own
    event_only = replace(_plan("p"), year_range=(2007, 2011), year_source="event", events=("financial_crisis_2008",))
    assert ask_flow._carry_plan(_plan("q"), event_only, 1).year_range is None  # its events carry the years
    assert ask_flow._carry_plan(_plan("q"), event_only, 1).events == ("financial_crisis_2008",)


def test_the_gentler_vote_keeps_a_short_follow_up_financial(monkeypatch):
    short = "What about that year?"
    history = [{"role": "user", "content": "In the 1994 letter, what did Buffett write about GEICO's managers?"},
               {"role": "assistant", "content": "He praised them."}]  # no business figure: the vote decides alone
    assert _with_history(short, history, share=0.6).scope == "personal"
    monkeypatch.setenv("FINANCE_FOLLOWUP_VOTE", "7")
    kept = _with_history(short, history, share=0.6)
    assert kept.scope == "financial" and kept.scope_source == "vote" and kept.vote_share == 0.6
    assert _with_history(short, history, share=0.8).scope == "personal"  # at or above 0.7 the vote still decides
    long_q = "Which magicians performed inside the tent party outside the jewelry store on the Sunday afternoon?"
    assert _with_history(long_q, history, share=0.6).scope == "personal"  # a self-contained question keeps its vote
    assert _with_history(short, [], share=0.6).scope == "personal"  # no history: not a follow-up
    assert _with_history("What about his diet that year?", history, share=0.1).scope == "personal"  # a personal rule


def test_all_round_5_follow_up_settings_off_give_the_plan_query_plan():
    from src.retrieval.query_expansion import build_followup_retrieval_query
    for query in ("Who ran the company then, and how large did the business become over the following years?",
                  "Who ran the company at the time, and how large did the business become over the following years?"):
        resolved = build_followup_retrieval_query(query, HISTORY)
        assert _with_history(query) == plan_query(resolved, _follow_store(), scope_vote=_vote(0.1))
