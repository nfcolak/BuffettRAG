"""finance-recall-75-r4 group 2: decade-block candidates for corpus-wide and trend plans (FINANCE_DECADE_BLOCKS)."""
from __future__ import annotations

import threading

import pytest

from config import FINANCE_DEFAULTS, finance_setting
from src.knowledge import KnowledgeStore
from src.query.plan import QueryPlan
from src.retrieval import decade_source as ds
from src.retrieval import finance_policy
from src.retrieval import knowledge_source as ks
from src.retrieval import multiquery_source as mq
from src.retrieval.bm25 import _meta_matches
from src.retrieval.finance_policy import candidate_sources, finance_search
from src.storage import SearchHit, StoredDoc

SETTING = "FINANCE_DECADE_BLOCKS"
OTHER_SOURCES = ("FINANCE_SUMMARY_SOURCE", "FINANCE_TAGGED_SOURCE", "FINANCE_DIGEST_SOURCE", "FINANCE_STANCE_SOURCE",
                 "FINANCE_MULTI_QUERY", "FINANCE_DIGEST_QUERY")
YEARS = range(1980, 2020)   # 40 letters, 10 in each of the four blocks (1980-89, 1990-99, 2000-09, 2010-19)
PER_LETTER_DOCS = 4
TREND = "How did Buffett's view of share repurchases change over the decades?"
THEMES = {"buybacks": "share repurchases. A tie-break note."}
BLOCK_FILTERS = [{"year": {"$gte": a, "$lte": b}} for a, b in ds.BLOCKS]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("FINANCE_K", "FINANCE_OVERFETCH", "FINANCE_TAG_DEPTH", "FINANCE_TAG_POOL",
                 "FINANCE_YEAR_BOOST_WEIGHT", "LLM_CONTEXT_PASSAGES", "FINANCE_RERANK_CANDIDATES", SETTING):
        monkeypatch.delenv(name, raising=False)
    for name in OTHER_SOURCES:  # the other sources off: a test sees only what this source adds
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("RETRIEVAL_DEDUP", "0")
    monkeypatch.setattr(mq, "THEME_QUERY", "description")  # the fake rankings below key on the topic description


def _pid(year, n):
    return f"{year}_p{n:04d}"


def _docs():
    return [StoredDoc(id=_pid(y, n), text=f"text {y} {n}", metadata={"year": y}) for n in range(PER_LETTER_DOCS)
            for y in YEARS]


def _ann(pid, **kw):
    return {"id": pid, "scope": kw.pop("scope", "financial"), "is_table": kw.pop("is_table", False),
            "sectors": [], "themes": [], "events": [], "companies": [], "stance": [], "summary": "", **kw}


def _store(annotations=None):
    anns = annotations if annotations is not None else [_ann(d.id) for d in _docs()]
    return KnowledgeStore(annotations=anns, taxonomy={"themes": THEMES, "sectors": {}}, digests=[])


class Retriever:
    """Hybrid ranking per query: the ids of the first `rankings` key the query contains (then the rest of the corpus),
    else corpus order (every letter's n=0 passage first, ranks 0-39, then n=1, ...). Records every search."""

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


def _plan(query=TREND, kind="trend", **kw):
    return QueryPlan(query=query, financial_query=query, kind=kind, **kw)


def _ids(hits):
    return [h.id for h in hits]


def _block_calls(retriever):
    return [(q, where) for q, where in retriever.calls if where]


# --- registry and setting ---
def test_registered_last_and_off_by_default(monkeypatch):
    assert ks.SOURCES[-1:] == ds.SOURCES == [(SETTING, ds.decade_candidates)]
    assert FINANCE_DEFAULTS[SETTING] == 0 and finance_setting(SETTING) == 0
    assert SETTING in [name for name, _ in finance_policy.registered_sources()]
    assert SETTING not in [name for name, _ in candidate_sources()]
    monkeypatch.setenv(SETTING, "2")
    assert [name for name, _ in candidate_sources()] == [SETTING]


# --- when it fires ---
@pytest.mark.parametrize("kind,year_range,fires", [
    ("corpus_wide", None, True),
    ("trend", None, True),
    ("specific", None, False),
    ("period", None, False),
    ("comparison", None, False),
    ("numeric", None, False),
    ("trend", (1990, 1999), False),     # a narrow range
    ("trend", (1980, 1994), False),     # 15 letters: not more than 15
    ("trend", (1980, 1995), True),      # 16 letters
    ("corpus_wide", (1977, 2024), True),
    ("corpus_wide", (1995, 1980), True),  # a reversed range counts the same
    ("trend", (2007, 2010), False),
])
def test_fires_only_for_corpus_wide_and_trend_plans_without_a_narrow_range(kind, year_range, fires):
    assert ds.applies(_plan(kind=kind, year_range=year_range)) is fires


def test_blocks_cover_the_corpus_or_are_cut_to_the_range():
    assert ds.blocks_of(_plan()) == [(1977, 1989), (1990, 1999), (2000, 2009), (2010, 2024)]
    assert ds.blocks_of(_plan(year_range=(1980, 2000))) == [(1980, 1989), (1990, 1999), (2000, 2000)]
    assert ds.blocks_of(_plan(year_range=(2015, 1991))) == [(1991, 1999), (2000, 2009), (2010, 2015)]


@pytest.mark.parametrize("value,plan", [
    ("0", _plan()),                               # the setting is off
    ("2", _plan("Share repurchases in 1995", kind="specific")),    # a plan the source does not serve
    ("2", _plan(year_range=(1990, 1999))),        # a narrow range
])
def test_nothing_runs_when_off_or_when_the_plan_does_not_fire(monkeypatch, value, plan):
    monkeypatch.setenv(SETTING, value)
    retriever = Retriever(_docs())
    assert ds.decade_candidates(plan, _store(), retriever, plan.query, 4) == []
    assert retriever.calls == []


# --- the searches ---
def test_one_search_per_block_with_the_financial_core_and_the_year_filter(monkeypatch):
    monkeypatch.setenv(SETTING, "2")
    retriever = Retriever(_docs())
    plan = _plan()
    ds.decade_candidates(plan, _store(), retriever, plan.query, 4)
    calls = _block_calls(retriever)
    assert sorted(where["year"]["$gte"] for _, where in calls) == [1977, 1990, 2000, 2010]
    assert sorted((w for _, w in calls), key=lambda w: w["year"]["$gte"]) == BLOCK_FILTERS
    core = mq.financial_core(plan)
    assert "Buffett" not in core and "repurchases" in core
    assert {q for q, _ in calls} == {core}  # the financial core, not the opinion frame


def test_a_year_range_searches_only_the_blocks_inside_it_cut_to_the_range(monkeypatch):
    monkeypatch.setenv(SETTING, "2")
    retriever = Retriever(_docs())
    plan = _plan(year_range=(1980, 2000))
    hits = ds.decade_candidates(plan, _store(), retriever, plan.query, 4)
    core = mq.financial_core(plan)
    wheres = sorted((w for q, w in _block_calls(retriever) if q == core), key=lambda w: w["year"]["$gte"])
    assert wheres == [{"year": {"$gte": 1980, "$lte": 1989}}, {"year": {"$gte": 1990, "$lte": 1999}},
                      {"year": {"$gte": 2000, "$lte": 2000}}]  # (the pool's own range ranking uses the full query)
    assert all(1980 <= h.metadata["year"] <= 2000 for h in hits)


# --- the picks ---
def test_top_m_per_block_skip_the_pool_and_the_best_of_every_block_comes_first(monkeypatch):
    monkeypatch.setenv(SETTING, "2")
    retriever = Retriever(_docs())
    plan = _plan()
    hits = ds.decade_candidates(plan, _store(), retriever, plan.query, 4)
    # the unfiltered ranking pools ranks 0-29 (fetch_k): every n=0 passage of 1980-2009. The last block's n=0 passages
    # (2010-2019) are not pooled; the others take the next passage of the two best letters.
    assert _ids(hits) == [_pid(1980, 1), _pid(1990, 1), _pid(2000, 1), _pid(2010, 0),
                          _pid(1981, 1), _pid(1991, 1), _pid(2001, 1), _pid(2011, 0)]
    assert all(h.text == retriever.by_id[h.id].text and h.metadata == retriever.by_id[h.id].metadata for h in hits)


@pytest.mark.parametrize("m", [1, 3, 6])
def test_m_sets_the_places_per_block(monkeypatch, m):
    monkeypatch.setenv(SETTING, str(m))
    plan = _plan()
    hits = ds.decade_candidates(plan, _store(), Retriever(_docs()), plan.query, 4)
    assert len(hits) == 4 * m and len(set(_ids(hits))) == 4 * m
    per_block = [sum(1 for h in hits if a <= h.metadata["year"] <= b) for a, b in ds.BLOCKS]
    assert per_block == [m] * 4
    assert len({h.metadata["year"] for h in hits}) == 4 * m  # one passage per letter while the block has letters


def test_a_block_takes_at_most_one_passage_per_letter(monkeypatch):
    monkeypatch.setenv(SETTING, "2")
    plan = _plan()
    first = [_pid(1995, 3), _pid(1995, 2), _pid(1995, 1), _pid(1996, 3), _pid(1997, 3)]
    retriever = Retriever(_docs(), {mq.financial_core(plan): first})
    hits = ds.decade_candidates(plan, _store(), retriever, plan.query, 4)
    assert [i for i in _ids(hits) if i[:4] in ("1995", "1996", "1997")] == [_pid(1995, 3), _pid(1996, 3)]


def test_a_block_with_fewer_letters_than_m_takes_several_per_letter(monkeypatch):
    monkeypatch.setenv(SETTING, "3")
    plan = _plan(year_range=(1980, 2000))  # the last block is the letter 2000 alone
    hits = ds.decade_candidates(plan, _store(), Retriever(_docs()), plan.query, 4)
    last = [i for i in _ids(hits) if i.startswith("2000_")]
    assert last == [_pid(2000, 1), _pid(2000, 2), _pid(2000, 3)]  # 2000_p0000 is pooled


def test_non_financial_and_table_passages_are_skipped(monkeypatch):
    monkeypatch.setenv(SETTING, "1")
    plan = _plan()
    ranking = [_pid(1995, 3), _pid(1996, 3), _pid(1997, 3)]
    anns = [_ann(d.id, scope="personal" if d.id == ranking[0] else "financial", is_table=d.id == ranking[1])
            for d in _docs()]
    retriever = Retriever(_docs(), {mq.financial_core(plan): ranking})
    hits = ds.decade_candidates(plan, _store(anns), retriever, plan.query, 4)
    assert [i for i in _ids(hits) if i[:4] in ("1995", "1996", "1997")] == [_pid(1997, 3)]


def test_nothing_without_a_corpus(monkeypatch):
    monkeypatch.setenv(SETTING, "2")
    retriever = Retriever([])
    assert ds.decade_candidates(_plan(), _store([]), retriever, TREND, 4) == []


# --- the pool ---
def test_passages_the_multi_query_source_adds_are_not_added_again(monkeypatch):
    monkeypatch.setenv(SETTING, "1")
    plan = _plan(themes=("buybacks",))
    key = "did of share"  # the core has it ("did Buffett's view of share" without the frame), the finance query has not
    assert key in mq.financial_core(plan) and key not in plan.query
    target, other = _pid(1995, 3), _pid(1996, 3)
    retriever = Retriever(_docs(), {key: [target, other]})
    store = _store()
    alone = ds.decade_candidates(plan, store, retriever, plan.query, 4)
    assert target in _ids(alone)  # not pooled by the unfiltered ranking: the first of its block

    monkeypatch.setenv("FINANCE_MULTI_QUERY", "1")
    added = {h.id for h in mq.multi_query_source(plan, store, retriever, plan.query, 4)}
    assert target in added
    hits = ds.decade_candidates(plan, store, retriever, plan.query, 4)
    assert len(hits) == 4 and not added & set(_ids(hits))  # the places go to passages no earlier source holds
    assert ds.pooled(plan, store, retriever, plan.query, 4) >= added


def test_no_recursion_when_the_multi_query_sources_are_off(monkeypatch):
    """With FINANCE_MULTI_QUERY and FINANCE_DIGEST_QUERY off, `multiquery_source.pooled` runs every registered source,
    this one too: the guard makes that inner call return nothing."""
    monkeypatch.setenv(SETTING, "2")
    plan = _plan()
    got = ds.pooled(plan, _store(), Retriever(_docs()), plan.query, 4)
    assert _pid(1980, 0) in got and not any(pid.endswith("p0001") for pid in got)
    assert not getattr(ds._RUNNING, "on", False)


def test_the_guard_is_released_when_the_pool_computation_fails(monkeypatch):
    monkeypatch.setenv(SETTING, "2")

    def boom(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(mq, "pooled", boom)
    with pytest.raises(RuntimeError):
        ds.decade_candidates(_plan(), _store(), Retriever(_docs()), TREND, 4)
    assert not getattr(ds._RUNNING, "on", False)


# --- through finance_search ---
def test_finance_search_pools_the_decade_passages_only_when_on(monkeypatch):
    plan = _plan()
    store = _store()
    _, off = finance_search(plan, Retriever(_docs()), store, k=4)
    assert SETTING not in off["source_candidates"]
    monkeypatch.setenv(SETTING, "2")
    _, on = finance_search(plan, Retriever(_docs()), store, k=4)
    assert on["source_candidates"][SETTING] == 8 and on["source_errors"] == []
    assert _pid(1980, 1) in on["stage_pool"] and _pid(1980, 1) not in off["stage_pool"]
    assert on["stage_pool"][:len(off["stage_pool"])] == off["stage_pool"]
    narrow = _plan("Share repurchases in the 1990s", kind="period", year_range=(1990, 1999))
    _, plain = finance_search(narrow, Retriever(_docs()), store, k=4)
    assert plain["source_candidates"][SETTING] == 0
