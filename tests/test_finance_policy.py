"""finance-query-pipeline group 4: finance_search over a fake retriever and the knowledge_mini fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

from config import llm_context_passages
from src.knowledge import KnowledgeStore
from src.query.plan import QueryPlan
from src.retrieval.bm25 import _meta_matches
from src.retrieval.finance_policy import finance_search
from src.storage import SearchHit

FIX = Path(__file__).parent / "fixtures" / "knowledge_mini"


def _hit(pid, year=None):
    return SearchHit(id=pid, text=f"text of {pid}", metadata={"year": year or int(pid[:4])}, score=1.0)


class FakeRetriever:
    """Hybrid ranking = the given order; a filter keeps the matching hits in that order."""

    def __init__(self, ranking, bm25=None, reranker=None):
        self.ranking, self.bm25_ranking, self.reranker = list(ranking), list(bm25 or []), reranker
        self.calls = []

    def _hybrid_for_filter(self, query, fetch_k, where):
        self.calls.append(("hybrid", where))
        return [h for h in self.ranking if not where or _meta_matches(h.metadata, where)][:fetch_k]

    def _bm25_search(self, query, top_k, where=None):
        self.calls.append(("bm25", where))
        return [h for h in self.bm25_ranking if not where or _meta_matches(h.metadata, where)][:top_k]


@pytest.fixture(scope="module")
def store():
    return KnowledgeStore.load(FIX)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("FINANCE_K", "FINANCE_OVERFETCH", "FINANCE_YEAR_BOOST_WEIGHT", "FINANCE_ENTITY_BOOST_WEIGHT",
                 "FINANCE_MAX_PER_LETTER", "FINANCE_MIN_LETTERS", "FINANCE_YEAR_KEEP", "FINANCE_TAG_DEPTH",
                 "FINANCE_TAG_POOL", "LLM_CONTEXT_PASSAGES", "RETRIEVAL_DEDUP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RETRIEVAL_DEDUP", "0")  # the synthetic texts are not near-duplicates; keep ranks exact


def _plan(kind="specific", **kw):
    return QueryPlan(query="q", financial_query="q", kind=kind, **kw)


def _spread(years, per_letter=5):
    """Hits grouped by letter: every letter has `per_letter` passages, letters interleaved best-first."""
    return [_hit(f"{y}_s{n:02d}") for n in range(per_letter) for y in years]


def _clustered():
    """The top 8 ranks all sit in one letter, then other letters follow."""
    return ([_hit(f"1999_c{n:02d}") for n in range(8)]
            + [_hit(f"{y}_c{n:02d}") for y in (1981, 1987, 1992, 1996, 2003, 2008) for n in range(2)])


def test_trend_plan_serves_at_least_four_letters_and_at_most_two_per_letter(store):
    for ranking in (_spread(range(1980, 1990)), _clustered()):
        # at today's budget: a trend plan's own default (FINANCE_K_TREND) is larger
        hits, debug = finance_search(_plan("trend"), FakeRetriever(ranking), store, k=llm_context_passages())
        years = [h.metadata["year"] for h in hits]
        assert len(hits) == min(llm_context_passages(), 2 * len({h.metadata["year"] for h in ranking}))
        assert len(set(years)) >= 4
        assert max(years.count(y) for y in set(years)) <= 2
        assert debug["letters"] == len(set(years))


@pytest.mark.parametrize("kind,year_range", [("period", (1980, 2010)), ("specific", (1980, 2010)), ("corpus_wide", None)])
def test_questions_spanning_several_letters_are_diverse_too(store, kind, year_range):
    hits, _ = finance_search(_plan(kind, year_range=year_range), FakeRetriever(_clustered()), store)
    years = [h.metadata["year"] for h in hits]
    assert len(set(years)) >= 4 and max(years.count(y) for y in set(years)) <= 2


def test_specific_plan_is_not_capped(store):
    hits, _ = finance_search(_plan("specific"), FakeRetriever(_clustered()), store)
    assert [h.id for h in hits] == [h.id for h in _clustered()[:llm_context_passages()]]


def test_min_letters_stops_at_what_is_available(store):
    ranking = [_hit(f"{y}_x{n}") for y in (1990, 1991) for n in range(6)]
    hits, _ = finance_search(_plan("trend"), FakeRetriever(ranking), store)
    assert {h.metadata["year"] for h in hits} == {1990, 1991} and len(hits) == 4  # 2 letters x cap 2


def test_k_defaults_to_the_reader_context_and_finance_k_and_the_argument_win(store, monkeypatch):
    ranking = _spread(range(1980, 1995))
    assert len(finance_search(_plan("specific"), FakeRetriever(ranking), store)[0]) == llm_context_passages() == 16
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "10")
    assert len(finance_search(_plan("specific"), FakeRetriever(ranking), store)[0]) == 10
    monkeypatch.setenv("FINANCE_K", "7")
    assert len(finance_search(_plan("specific"), FakeRetriever(ranking), store)[0]) == 7
    assert len(finance_search(_plan("specific"), FakeRetriever(ranking), store, k=5)[0]) == 5


def test_retrospective_gold_outside_the_year_range_stays_among_the_results(store):
    # "oil crisis in the 70s": the gold passage is a 1994 letter looking back; the in-range letters rank below it.
    gold = _hit("1994_gold")
    ranking = [_hit("1978_a"), gold] + [_hit(f"197{y}_b{n}") for y in (3, 4, 5, 6, 7, 8) for n in range(2)]
    plan = _plan("trend", year_range=(1973, 1982))
    hits, debug = finance_search(plan, FakeRetriever(ranking), store)
    assert "1994_gold" in [h.id for h in hits]
    assert debug["year_range"] == [1973, 1982]


def test_gold_deep_in_the_unfiltered_ranking_survives_the_boost(store):
    gold = _hit("1994_gold")
    ranking = [_hit("1973_b0"), _hit("1974_b0"), gold] + [_hit(f"197{y}_b{n}") for n in range(1, 4) for y in (3, 4, 5, 6, 7, 8)]
    hits, debug = finance_search(_plan("specific", year_range=(1973, 1982)), FakeRetriever(ranking), store, k=30)
    assert "1994_gold" in [h.id for h in hits]
    assert "1994_gold" in debug["year_filtered_out"]  # down-ranked behind the in-range passages, reported for M2


def test_year_boost_keeps_the_top_unfiltered_hits_even_when_k_is_small(store):
    ranking = [_hit("1994_gold")] + [_hit(f"197{y}_b{n}") for n in range(3) for y in (3, 4, 5, 6, 7, 8)]
    for kind in ("specific", "trend"):
        hits, _ = finance_search(_plan(kind, year_range=(1973, 1982)), FakeRetriever(ranking), store, k=4)
        assert "1994_gold" in [h.id for h in hits] and len(hits) == 4
    # a hard filter (an "in the YYYY letter" question) does remove it, by design
    hits, _ = finance_search(_plan("specific", year_range=(1973, 1982), year_hard=True), FakeRetriever(ranking), store, k=4)
    assert "1994_gold" not in [h.id for h in hits]


def test_hard_year_filter_removes_and_reports_out_of_range_hits(store):
    ranking = [_hit("1999_a"), _hit("1994_gold"), _hit("1994_b"), _hit("1999_c")]
    plan = _plan("specific", year_range=(1994, 1994), year_hard=True)
    hits, debug = finance_search(plan, FakeRetriever(ranking), store)
    assert {h.id for h in hits} == {"1994_gold", "1994_b"}
    assert debug["year_filtered_out"] == ["1999_a", "1999_c"]


def test_year_boost_weight_setting_changes_the_merge(store, monkeypatch):
    ranking = [_hit("1999_out")] + [_hit(f"1978_in{n}") for n in range(3)]
    plan = _plan("specific", year_range=(1973, 1982))
    monkeypatch.setenv("FINANCE_YEAR_BOOST_WEIGHT", "1")
    flat = [h.id for h in finance_search(plan, FakeRetriever(ranking), store)[0]]
    monkeypatch.setenv("FINANCE_YEAR_BOOST_WEIGHT", "5")
    boosted = [h.id for h in finance_search(plan, FakeRetriever(ranking), store)[0]]
    assert flat[0] == "1978_in0" or flat.index("1999_out") <= 1
    assert boosted.index("1999_out") >= flat.index("1999_out")


def test_event_tagged_passages_come_first_and_event_years_are_the_boost_range(store):
    ranking = [_hit("2003_p0087"), _hit("1986_p0118"), _hit("2008_p0011"), _hit("2010_p0119"), _hit("2008_p0067")]
    plan = _plan("specific", events=("financial_crisis_2008",))
    retriever = FakeRetriever(ranking)
    hits, debug = finance_search(plan, retriever, store)
    ids = [h.id for h in hits]
    assert set(ids[:4]) >= {"2008_p0011", "2010_p0119", "2008_p0067"}
    assert ids.index("2003_p0087") > ids.index("2008_p0011")
    assert debug["event_tagged"] == 3 and debug["year_range"] is not None
    assert any(w for kind, w in retriever.calls if kind == "hybrid" and w)  # a year-filtered call was made


def test_sector_entities_are_boosted(store):
    ranking = [_hit("2003_p0087"), _hit("1982_p0086"), _hit("1990_p0100")]
    hits, debug = finance_search(_plan("specific", sectors=("airlines_aviation",)), FakeRetriever(ranking), store)
    assert hits[0].id == "1990_p0100" and debug["entity_tagged"] == 1


def test_terms_add_a_lexical_ranking_inside_the_range(store):
    ranking = [_hit(f"{y}_g{n}") for n in range(3) for y in (1990, 1995, 2000)]
    oil = _hit("1980_oil")
    retriever = FakeRetriever(ranking, bm25=[oil])
    hits, _ = finance_search(_plan("specific", year_range=(1973, 1982), terms=("oil", "OPEC")), retriever, store)
    assert "1980_oil" in [h.id for h in hits]


def test_non_financial_passages_are_dropped_and_tables_only_for_numeric_plans(store):
    ranking = [_hit("1985_p0013"), _hit("1990_p0119"), _hit("1991_p0083"), _hit("1991_p0072"), _hit("1990_p0100")]
    hits, debug = finance_search(_plan("specific"), FakeRetriever(ranking), store)
    assert [h.id for h in hits] == ["1991_p0072", "1990_p0100"]
    assert set(debug["dropped_not_financial"]) == {"1985_p0013", "1990_p0119", "1991_p0083"}
    numeric, _ = finance_search(_plan("numeric"), FakeRetriever(ranking), store)
    assert [h.id for h in numeric] == ["1991_p0083", "1991_p0072", "1990_p0100"]


def test_empty_store_passes_everything_through_and_reports_zero_coverage():
    ranking = _spread(range(1980, 1990))
    hits, debug = finance_search(_plan("trend"), FakeRetriever(ranking), KnowledgeStore.empty(), k=llm_context_passages())
    assert len(hits) == llm_context_passages() and debug["coverage"] == 0.0 and debug["annotated"] == 0
    assert debug["pool_size"] == debug["fetch_k"] == 3 * llm_context_passages()


def test_partial_store_reports_its_coverage(store):
    ranking = [_hit("2008_p0011"), _hit("2005_p0127"), _hit("1989_p0088"), _hit("1990_p0100")]  # 2 annotated of 4
    hits, debug = finance_search(_plan("specific"), FakeRetriever(ranking), store)
    assert [h.id for h in hits] == [h.id for h in ranking]
    assert debug["coverage"] == pytest.approx(0.5) and debug["annotated"] == 2


def test_period_plan_reserves_each_named_period(store):
    plan = QueryPlan(query="Compare Berkshire in 1985 and 2005", financial_query="Compare Berkshire in 1985 and 2005",
                     kind="comparison", years=(1985, 2005))
    ranking = [_hit(f"2005_r{n}") for n in range(8)] + [_hit("1985_r0")]
    hits, _ = finance_search(plan, FakeRetriever(ranking), store, k=4)
    assert "1985_r0" in [h.id for h in hits]


def test_reranker_order_is_used(store):
    class Rev:
        def rerank(self, query, hits, top_k):
            return list(reversed(hits))[:top_k]

    ranking = [_hit("1980_a"), _hit("1981_b"), _hit("1982_c")]
    hits, debug = finance_search(_plan("specific"), FakeRetriever(ranking, reranker=Rev()), store)
    assert [h.id for h in hits] == ["1982_c", "1981_b", "1980_a"] and debug["reranked"] is True


def test_empty_retrieval_returns_no_hits(store):
    hits, debug = finance_search(_plan("trend", year_range=(1980, 1989)), FakeRetriever([]), store)
    assert hits == [] and debug["coverage"] == 0.0 and debug["year_filtered_out"] == []


class PublicSearchOnly:
    """A retriever with no hybrid primitive: only the public search (what simple fakes and wrappers expose)."""

    def __init__(self, ranking):
        self.ranking, self.calls = list(ranking), []

    def search(self, query, **kwargs):
        from src.retrieval.retriever import RetrievalResult
        where = kwargs.get("where")
        self.calls.append(kwargs)
        hits = [h for h in self.ranking if not where or _meta_matches(h.metadata, where)]
        return RetrievalResult(query, "hybrid", hits[:kwargs["top_k"]])


def test_retriever_without_hybrid_primitive_falls_back_to_public_search_and_keeps_finance_filtering(store):
    ranking = [_hit("1985_p0013"), _hit("1990_p0119"), _hit("1991_p0083"), _hit("1991_p0072"), _hit("1990_p0100")]
    retriever = PublicSearchOnly(ranking)
    hits, debug = finance_search(_plan("specific"), retriever, store)
    assert [h.id for h in hits] == ["1991_p0072", "1990_p0100"]  # same result as the primitive path
    assert set(debug["dropped_not_financial"]) == {"1985_p0013", "1990_p0119", "1991_p0083"}
    assert retriever.calls and all(c["strategy"] == "hybrid" and c["rerank"] is False
                                   and c["auto_year_filter"] is False for c in retriever.calls)
    numeric, _ = finance_search(_plan("numeric"), PublicSearchOnly(ranking), store)
    assert [h.id for h in numeric] == ["1991_p0083", "1991_p0072", "1990_p0100"]


def test_fallback_keeps_trend_diversity(store):
    hits, _ = finance_search(_plan("trend"), PublicSearchOnly(_clustered()), store)
    years = [h.metadata["year"] for h in hits]
    assert len(set(years)) >= 4 and max(years.count(y) for y in set(years)) <= 2


def test_fallback_applies_the_year_range_as_a_where_filter(store):
    retriever = PublicSearchOnly(_spread(range(1980, 1990)))
    finance_search(_plan("period", year_range=(1983, 1985)), retriever, store)
    wheres = [c.get("where") for c in retriever.calls]
    assert None in wheres and {"year": {"$gte": 1983, "$lte": 1985}} in wheres


# --- finance-dev-fix-1: reader-context k, diversity only across letters, narrow ranges dominate, deep tags, pins ---
def _explicit(kind, *spans, **kw):
    lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
    return _plan(kind, year_range=(lo, hi), year_spans=tuple(spans), year_source="explicit", **kw)


def test_a_single_year_is_never_capped_and_its_letters_y_and_y_plus_1_come_first(store):
    # "In year 1999 ...": out-of-range passages rank first, then 1999 and 2000; nothing caps the 1999 letter
    ranking = ([_hit(f"{y}_o{n}") for n in range(2) for y in (2015, 2016, 2017, 2018)]
               + [_hit(f"1999_i{n:02d}") for n in range(8)] + [_hit(f"2000_i{n:02d}") for n in range(4)]
               + [_hit(f"2001_i{n:02d}") for n in range(4)])
    hits, debug = finance_search(_explicit("period", (1999, 1999)), FakeRetriever(ranking), store,
                                 k=llm_context_passages())  # today's budget, not the period default FINANCE_K_PERIOD
    ids = [h.id for h in hits]
    in_block = [i for i in ids if i[:4] in ("1999", "2000")]
    assert len(hits) == llm_context_passages()
    assert sum(i.startswith("1999") for i in ids) == 8  # no per-letter cap for a single year
    assert ids[:len(in_block)] == in_block and len(in_block) == 12  # Y and Y+1 first ...
    assert {"2015_o0", "2016_o0", "2017_o0"} <= set(ids)  # ... the best unfiltered passages still stay
    assert debug["narrow"] is True and debug["diverse"] is False and debug["year_blocks"] == [[1999, 2000]]


def test_two_separate_years_are_two_narrow_blocks_not_one_wide_range(store):
    ranking = [_hit(f"{y}_x{n}") for n in range(6) for y in (2011, 2013, 2014, 2016)]
    hits, debug = finance_search(_explicit("period", (2011, 2011), (2016, 2016)), FakeRetriever(ranking), store)
    years = [h.metadata["year"] for h in hits]
    assert debug["year_blocks"] == [[2011, 2012], [2016, 2017]] and not debug["diverse"]
    assert years.count(2011) == 6 and years.count(2016) == 6  # each letter fully served, no cap


def test_a_range_of_three_letters_is_diverse_and_the_cap_grows_so_k_is_filled(store):
    ranking = [_hit(f"{y}_x{n:02d}") for y in (1986, 1987, 1988, 1989, 1990) for n in range(8)]
    hits, debug = finance_search(_explicit("period", (1986, 1990)), FakeRetriever(ranking), store, k=16)
    years = [h.metadata["year"] for h in hits]
    assert debug["diverse"] and len(hits) == 16  # cap = ceil(16 / 5) = 4, not 2 (which would serve 10)
    assert max(years.count(y) for y in set(years)) <= 4 and len(set(years)) >= 4
    two, _ = finance_search(_explicit("period", (1986, 1987)), FakeRetriever(ranking), store)
    assert [h.metadata["year"] for h in two].count(1986) == 8  # a 2-letter range is narrow: not capped


def test_an_event_question_serves_its_core_letters_first_without_a_cap(store):
    ranking = [_hit(f"1996_o{n}") for n in range(4)] + [_hit(f"1987_c{n:02d}") for n in range(10)]
    plan = _plan("period", events=("crash_1987",), year_range=(1987, 1987), year_source="event")
    hits, debug = finance_search(plan, FakeRetriever(ranking), store)
    ids = [h.id for h in hits]
    assert sum(i.startswith("1987") for i in ids) == 10 and all(i.startswith("1987") for i in ids[:10])
    assert debug["year_blocks"] == [[1987, 1987]] and debug["narrow"] and not debug["diverse"]


def test_an_explicit_year_and_an_event_with_later_core_letters_both_lead(store):
    # "the 1998 collapse of LTCM": letters 1998-1999 and the event's core letter 2002 lead the served list
    ranking = [_hit(f"2010_o{n}") for n in range(6)] + [_hit("2002_c0"), _hit("2002_c1"), _hit("1998_y0")]
    plan = _explicit("period", (1998, 1998), events=("ltcm_1998",))
    hits, debug = finance_search(plan, FakeRetriever(ranking), store)
    ids = [h.id for h in hits]
    assert debug["year_blocks"] == [[1998, 1999], [2002, 2002]]
    assert set(ids[:3]) == {"2002_c0", "2002_c1", "1998_y0"}


class Recorder:
    """A reranker that keeps the order and records the window it was given."""

    def __init__(self, first=None):
        self.windows, self.first = [], first

    def rerank(self, query, hits, top_k):
        self.windows.append([h.id for h in hits])
        hits = list(hits)
        if self.first:
            hits.sort(key=lambda h: h.id != self.first)
        return hits[:top_k]


def test_tagged_passages_deeper_than_the_fetch_reach_the_reranker(store, monkeypatch):
    monkeypatch.setenv("RERANK_CANDIDATES", "10")
    monkeypatch.setenv("FINANCE_ENTITY_BOOST_WEIGHT", "0")  # no boost: only the tag pool can bring it in
    filler = [_hit(f"19{60 + n // 10}_f{n:02d}", 1960 + n // 10) for n in range(80)]  # untagged
    ranking = filler + [_hit("1990_p0102")]  # tagged airlines_aviation, rank 80 (> fetch_k 48)
    plan = _plan("specific", sectors=("airlines_aviation",))
    rec = Recorder(first="1990_p0102")
    hits, debug = finance_search(plan, FakeRetriever(ranking, reranker=rec), store)
    assert rec.windows[0][:10] == [h.id for h in filler[:10]] and rec.windows[0][10:] == ["1990_p0102"]
    assert hits[0].id == "1990_p0102" and debug["tag_injected"] == 1
    monkeypatch.setenv("FINANCE_TAG_POOL", "0")
    rec = Recorder()
    hits, _ = finance_search(plan, FakeRetriever(ranking, reranker=rec), store)
    assert len(rec.windows[0]) == 10 and "1990_p0102" not in [h.id for h in hits]


def test_pinned_hits_are_served_first_whatever_their_flags(store):
    ranking = [_hit(f"{y}_x{n}") for n in range(5) for y in (1990, 1995, 2000)]
    table = _hit("1991_p0083")  # a table: a non-numeric plan would drop it
    hits, debug = finance_search(_plan("specific"), FakeRetriever(ranking), store, k=6, pinned=[table])
    assert hits[0].id == "1991_p0083" and len(hits) == 6 and debug["pinned"] == ["1991_p0083"]
    assert [h.id for h in hits[1:]] == [h.id for h in ranking[:5]]
