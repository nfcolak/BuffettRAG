"""pv8-retrieval-validator group 1: period pool, focused period rerank, year filter keep, reference-year skip.

Synthetic in-memory corpus, stub reranker; no models. Every flag is a call-time setting (env), default = old path.
"""
from __future__ import annotations

from src.generation.compare import period_focused_query
from src.retrieval.retriever import Retriever, detect_year_filter
from src.storage import SearchHit, StoredDoc

COMPARE_Q = "Compare how Buffett discussed insurance float in 2005 and how he discussed it in 2010."


def _hit(i, year, text=None):
    return SearchHit(id=i, text=text or i, metadata={"year": year}, score=1.0)


class _Rerank:
    """Gold ('g*' ids) first, otherwise input order; records every call."""
    def __init__(self):
        self.calls = []

    def rerank(self, query, hits, top_k):
        self.calls.append((query, [h.id for h in hits]))
        return sorted(hits, key=lambda h: not h.id.startswith("g"))[:top_k]


class _Corpus(Retriever):
    def __init__(self, by_year, unfiltered=None, reranker=None):
        super().__init__(None, None, [StoredDoc("d", "dummy", {"year": 2000})], reranker or _Rerank())
        self.by_year, self.unfiltered = by_year, unfiltered or []

    def _hybrid_for_filter(self, query, fetch_k, where):
        if where is None:
            return list(self.unfiltered)
        year = where["year"]
        if isinstance(year, dict):
            year = year["$gte"]
        return list(self.by_year[year])


def _two_periods(gold_rank):
    a = [_hit(f"a{i}", 2005) for i in range(30)]
    a[gold_rank] = _hit("g05", 2005)
    return {2005: a, 2010: [_hit(f"b{i}", 2010) for i in range(30)]}


def _ids(result):
    return [h.id for h in result.hits]


def test_pool_default_is_old_formula_and_zero_is_identical(monkeypatch):
    for k in ("RETRIEVAL_PERIOD_POOL", "RETRIEVAL_PERIOD_FOCUS_RERANK"):
        monkeypatch.delenv(k, raising=False)
    old = _Corpus(_two_periods(20)).search(COMPARE_Q, top_k=4, rerank=True)
    reranker = _Rerank()
    _Corpus(_two_periods(20), reranker=reranker).search(COMPARE_Q, top_k=4, rerank=True)
    assert len(reranker.calls) == 1 and len(reranker.calls[0][1]) == 30  # 15 per period (30 // 2)
    assert "g05" not in _ids(old)  # starved: gold at rank 20 sits outside the 15-deep period pool
    monkeypatch.setenv("RETRIEVAL_PERIOD_POOL", "0")
    assert _ids(_Corpus(_two_periods(20)).search(COMPARE_Q, top_k=4, rerank=True)) == _ids(old)


def test_pool_30_rescues_starved_period_and_total_is_capped(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_PERIOD_POOL", "30")
    reranker = _Rerank()
    result = _Corpus(_two_periods(20), reranker=reranker).search(COMPARE_Q, top_k=4, rerank=True)
    assert "g05" in _ids(result)
    assert len(reranker.calls[0][1]) == 60
    monkeypatch.setenv("RETRIEVAL_PERIOD_POOL", "50")  # 2 periods: min(50, max(60 // 2, 4)) = 30 each
    reranker = _Rerank()
    _Corpus(_two_periods(20), reranker=reranker).search(COMPARE_Q, top_k=4, rerank=True)
    assert len(reranker.calls[0][1]) == 60


def test_focus_rerank_one_call_per_period_with_its_query_and_round_robin(monkeypatch):
    plain = _Corpus(_two_periods(20)).search(COMPARE_Q, top_k=4, rerank=True)
    monkeypatch.setenv("RETRIEVAL_PERIOD_FOCUS_RERANK", "1")
    reranker = _Rerank()
    result = _Corpus(_two_periods(20), reranker=reranker).search(COMPARE_Q, top_k=4, rerank=True)
    periods = [{"year": 2005}, {"year": 2010}]
    assert [q for q, _ in reranker.calls] == [period_focused_query(COMPARE_Q, p, periods) for p in periods]
    assert all(h.startswith("a") for h in reranker.calls[0][1]) and all(h.startswith("b") for h in reranker.calls[1][1])
    # reserve takes a0,b0; the fill follows the per-period round-robin (a1,b1), not the one-sided order (a1,a2)
    assert _ids(result) == ["a0", "b0", "a1", "b1"]
    assert _ids(plain) == ["a0", "b0", "a1", "a2"]


def test_period_reserve_still_honoured(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_PERIOD_RESERVE", "2")
    for env in ({}, {"RETRIEVAL_PERIOD_POOL": "30", "RETRIEVAL_PERIOD_FOCUS_RERANK": "1"}):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        result = _Corpus(_two_periods(20)).search(COMPARE_Q, top_k=4, rerank=True)
        years = [h.metadata["year"] for h in result.hits]
        assert years.count(2005) >= 2 and years.count(2010) >= 2


def _keep_corpus():
    filtered = [_hit(f"f{i}", 2005) for i in range(40)]
    unfiltered = [_hit("g_wrong_year", 2004)] + [_hit(f"u{i}", 2004) for i in range(9)]
    return {2005: filtered}, unfiltered


def test_year_filter_keep_rescues_gold_removed_by_wrong_year_guess(monkeypatch):
    q = "What did Buffett say about insurance float in 2005?"
    by_year, unfiltered = _keep_corpus()
    base = _Corpus(by_year, unfiltered).search(q, top_k=4, rerank=True)
    assert "g_wrong_year" not in _ids(base)
    monkeypatch.setenv("RETRIEVAL_YEAR_FILTER_KEEP", "1")
    reranker = _Rerank()
    result = _Corpus(by_year, unfiltered, reranker).search(q, top_k=4, rerank=True)
    assert "g_wrong_year" in _ids(result)
    window = reranker.calls[0][1]
    assert len(window) == 30 and window[-1] == "g_wrong_year"  # at the end of the rerank window
    assert window[:-1] == [f"f{i}" for i in range(29)]  # lowest merged hit pushed out, order otherwise unchanged
    monkeypatch.setenv("RETRIEVAL_YEAR_FILTER_KEEP", "0")
    assert _ids(_Corpus(by_year, unfiltered).search(q, top_k=4, rerank=True)) == _ids(base)


def test_year_ref_skip(monkeypatch):
    q = "Since 1990 how has the 2005 letter described float?"
    assert detect_year_filter(q) == {"year": {"$gte": 1990, "$lte": 2005}}
    monkeypatch.setenv("RETRIEVAL_YEAR_REF_SKIP", "1")
    assert detect_year_filter(q) == {"year": 2005}
    assert detect_year_filter("What was float at the end of 2001?") is None
    assert detect_year_filter("How did it look as of 1999 versus the 2003 letter?") == {"year": 2003}
    assert detect_year_filter("Was 2004 the same as in 1998?") == {"year": 2004}
    assert detect_year_filter("What changed from 1980 through 1990?") == {"year": {"$gte": 1980, "$lte": 1990}}
    assert detect_year_filter("What did he say in 1999?") == {"year": 1999}
