"""finance-query-pipeline group 4: finance_search over a fake retriever and the knowledge_mini fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

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
                 "FINANCE_MAX_PER_LETTER", "FINANCE_MIN_LETTERS", "RETRIEVAL_DEDUP"):
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
        hits, debug = finance_search(_plan("trend"), FakeRetriever(ranking), store)
        years = [h.metadata["year"] for h in hits]
        assert len(hits) == 12
        assert len(set(years)) >= 4
        assert max(years.count(y) for y in set(years)) <= 2
        assert debug["letters"] == len(set(years))


@pytest.mark.parametrize("kind", ["period", "corpus_wide"])
def test_period_and_corpus_wide_plans_are_diverse_too(store, kind):
    hits, _ = finance_search(_plan(kind), FakeRetriever(_clustered()), store)
    years = [h.metadata["year"] for h in hits]
    assert len(set(years)) >= 4 and max(years.count(y) for y in set(years)) <= 2


def test_specific_plan_is_not_capped(store):
    hits, _ = finance_search(_plan("specific"), FakeRetriever(_clustered()), store)
    assert [h.id for h in hits] == [h.id for h in _clustered()[:12]]


def test_min_letters_stops_at_what_is_available(store):
    ranking = [_hit(f"{y}_x{n}") for y in (1990, 1991) for n in range(6)]
    hits, _ = finance_search(_plan("trend"), FakeRetriever(ranking), store)
    assert {h.metadata["year"] for h in hits} == {1990, 1991} and len(hits) == 4  # 2 letters x cap 2


def test_k_defaults_to_finance_k_and_the_argument_wins(store, monkeypatch):
    ranking = _spread(range(1980, 1995))
    assert len(finance_search(_plan("specific"), FakeRetriever(ranking), store)[0]) == 12
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
    hits, debug = finance_search(_plan("trend"), FakeRetriever(ranking), KnowledgeStore.empty())
    assert len(hits) == 12 and debug["coverage"] == 0.0 and debug["annotated"] == 0
    assert debug["pool_size"] == debug["fetch_k"] == 36


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
