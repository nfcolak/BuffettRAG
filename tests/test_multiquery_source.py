"""finance-recall-75-r3 group 2: multi-query candidates for view questions (FINANCE_MULTI_QUERY, FINANCE_DIGEST_QUERY)."""
from __future__ import annotations

import math
import threading

import pytest

from config import FINANCE_DEFAULTS, finance_setting
from src.knowledge import KnowledgeStore
from src.query.plan import QueryPlan
from src.retrieval import finance_policy
from src.retrieval import knowledge_source as ks
from src.retrieval import multiquery_source as mq
from src.retrieval.bm25 import _meta_matches
from src.retrieval.finance_policy import candidate_sources, finance_search
from src.storage import SearchHit, StoredDoc

SETTINGS = ("FINANCE_MULTI_QUERY", "FINANCE_DIGEST_QUERY")
OTHER_SOURCES = ("FINANCE_SUMMARY_SOURCE", "FINANCE_TAGGED_SOURCE", "FINANCE_DIGEST_SOURCE", "FINANCE_STANCE_SOURCE")
YEARS = range(1990, 2000)
PER_LETTER_DOCS = 6
VIEW = "What does Warren Buffett think about share repurchases?"
THEMES = {"buybacks": "share repurchases. A tie-break note.", "dividends": "dividends and retained earnings.",
          "hub": "Berkshire's yearly results."}
SECTORS = {"insurance": "property-casualty insurers and reinsurance"}
OVERVIEW = "Buffett backs repurchases below value. He refuses them above it. A third line."


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("FINANCE_K", "FINANCE_OVERFETCH", "FINANCE_TAG_DEPTH", "FINANCE_TAG_POOL",
                 "FINANCE_YEAR_BOOST_WEIGHT", "LLM_CONTEXT_PASSAGES", "FINANCE_RERANK_CANDIDATES"):
        monkeypatch.delenv(name, raising=False)
    for name in SETTINGS:  # both off (FINANCE_MULTI_QUERY defaults to 1 since finance-recall-75-r3): a test turns on what it checks
        monkeypatch.setenv(name, "0")
    for name in OTHER_SOURCES:  # the other sources off: a test sees only what this source adds
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


def _store(annotations=None, digests=None):
    anns = annotations if annotations is not None else [_ann(d.id) for d in _docs()]
    return KnowledgeStore(annotations=anns, taxonomy={"themes": THEMES, "sectors": SECTORS},
                          digests=digests if digests is not None else
                          [{"topic": "buybacks", "kind": "theme", "overview": OVERVIEW, "timeline": []}])


class Retriever:
    """Hybrid ranking per query: the ids of the first `rankings` key the query contains (then the rest of the corpus),
    else corpus order (every letter's n=0 passage first, ranks 0-9, then n=1, 10-19, ...). Records every search."""

    def __init__(self, docs, rankings=None, embedder=None):
        self.docs, self.embedder, self.reranker = list(docs), embedder, None
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


def _plan(query=VIEW, kind="specific", **kw):
    return QueryPlan(query=query, financial_query=query, kind=kind, **kw)


def _ids(hits):
    return [h.id for h in hits]


def _deep(year, n):  # ranks 30-59 of the corpus order: not pooled by finance_search at k=4
    assert n >= 3
    return _pid(year, n)


# --- registry and settings ---
def test_both_settings_registered_last_with_the_adopted_defaults(monkeypatch):
    first = ks.SOURCES.index(mq.SOURCES[0])  # later rounds append theirs after these (finance-recall-75-r4: the decade source)
    assert ks.SOURCES[first:first + len(mq.SOURCES)] == mq.SOURCES
    assert [name for name, _ in mq.SOURCES] == list(SETTINGS)
    for name in SETTINGS:
        monkeypatch.delenv(name)
    # finance-recall-75-r3 group 4: one multi-query search passed the adoption rule, the digest query did not
    assert [FINANCE_DEFAULTS[name] for name in SETTINGS] == [1, 0]
    assert [finance_setting(name) for name in SETTINGS] == [1, 0]
    assert [name for name, _ in candidate_sources() if name in SETTINGS] == ["FINANCE_MULTI_QUERY"]
    for name in SETTINGS:
        monkeypatch.setenv(name, "0")
    assert not [name for name, _ in candidate_sources() if name in SETTINGS]
    monkeypatch.setenv("FINANCE_MULTI_QUERY", "2")
    assert [name for name, _ in candidate_sources() if name in SETTINGS] == ["FINANCE_MULTI_QUERY"]
    monkeypatch.setenv("FINANCE_DIGEST_QUERY", "1")
    assert [name for name, _ in candidate_sources() if name in SETTINGS] == list(SETTINGS)
    assert all(name in [n for n, _ in finance_policy.registered_sources()] for name in SETTINGS)


# --- view questions and their queries ---
def test_only_non_numeric_view_questions_run():
    assert mq.is_view(_plan())
    assert not mq.is_view(_plan(kind="numeric"))
    assert not mq.is_view(_plan("How large were Berkshire's share repurchases in 2012?"))
    store = _store()
    retriever = Retriever(_docs())
    plain = _plan("Share repurchases at Berkshire", themes=("buybacks",))
    assert mq.queries(plain, store, retriever, plain.query, 3, True) == []


def test_financial_core_drops_the_opinion_frame():
    core = mq.financial_core(_plan())
    assert "Buffett" not in core and "think" not in core and "share repurchases" in core
    frame_only = _plan("What did Buffett say?")
    assert mq.financial_core(frame_only) == "What did Buffett say?"  # nothing left: the query itself


def test_topic_queries_are_core_plus_description_in_topic_order_up_to_n(monkeypatch):
    store = _store()
    retriever = Retriever(_docs())
    plan = _plan(themes=("buybacks",), sectors=("insurance",))
    core = mq.financial_core(plan)
    one = mq.queries(plan, store, retriever, plan.query, 1, False)
    assert one == [f"{core} buybacks: share repurchases."]
    two = mq.queries(plan, store, retriever, plan.query, 2, False)
    assert two == [one[0], f"{core} insurance: property-casualty insurers and reinsurance"]
    assert mq.queries(plan, store, retriever, plan.query, 3, False) == two  # only two topics
    monkeypatch.setattr(mq, "MQ_SECTORS", False)
    assert mq.queries(plan, store, retriever, plan.query, 3, False) == one


def test_the_digest_query_is_the_first_overview_lines():
    store = _store()
    plan = _plan(themes=("buybacks",))
    texts = mq.queries(plan, store, Retriever(_docs()), plan.query, 0, True)
    assert texts == ["Buffett backs repurchases below value. He refuses them above it."]
    assert mq.queries(_plan(themes=("dividends",)), store, Retriever(_docs()), VIEW, 0, True) == []  # no digest


class Embedder:
    """Theme descriptions: buybacks e1, dividends e2, hub e3; the anchor sits next to the hub."""
    DOCS = {"buybacks": [1, 0, 0], "dividends": [0, 1, 0], "hub": [0, 0, 1]}

    def embed_documents(self, texts):
        return [self.DOCS[t.split(":")[0]] for t in texts]

    def embed_query(self, text):
        if text == ks.THEME_ANCHOR:
            v = [0.1, 0.1, 0.9]
        else:
            v = [0.0, 0.8, 0.6]  # dividends
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v]


def test_inferred_themes_follow_the_plan_themes(monkeypatch, tmp_path):
    monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path))  # no summary index: themes match by their descriptions
    store = _store()
    retriever = Retriever(_docs(), embedder=Embedder())
    plan = _plan(themes=("buybacks",), sectors=("insurance",))
    assert mq.topics(plan, store, retriever, plan.query) == [
        ("themes", "buybacks"), ("themes", "dividends"), ("sectors", "insurance")]


def test_topic_keywords_set_the_tagged_summaries_apart(monkeypatch):
    anns = [_ann(_pid(1990 + i, 0), themes=["buybacks"], summary=f"Berkshire repurchased shares below value {i}.")
            for i in range(4)]
    anns += [_ann(_pid(1990 + i, 1), summary="Berkshire earned more on insurance float and value.") for i in range(6)]
    store = _store(anns)
    words = mq.topic_keywords(store, "themes", "buybacks")
    assert words[:2] == ("below", "repurchased") and "berkshire" not in words
    monkeypatch.undo()  # the module defaults: keywords, chosen on dev
    assert mq.THEME_QUERY == "keywords" and mq.MQ_SUMMARY is False
    text = mq.topic_query("core", store, "themes", "buybacks")
    assert text.startswith("core below repurchased") and "share repurchases." not in text
    monkeypatch.setattr(mq, "THEME_QUERY", "both")
    assert mq.topic_query("core", store, "themes", "buybacks").startswith("core buybacks: share repurchases. below")


# --- the source ---
def test_source_adds_only_new_passages_capped_and_spread_across_letters(monkeypatch):
    monkeypatch.setenv("FINANCE_MULTI_QUERY", "1")
    monkeypatch.setattr(mq, "MQ_CAP", 4)
    target = [_deep(1995, 5), _deep(1995, 4), _deep(1995, 3), _pid(1990, 0), _deep(1996, 5), _deep(1997, 5)]
    retriever = Retriever(_docs(), {"buybacks:": target})
    plan = _plan(themes=("buybacks",))
    hits = mq.multi_query_source(plan, _store(), retriever, plan.query, 4)
    # 1990_p0000 is pooled already (rank 0); at most PER_LETTER (2) of 1995 while other letters are there
    assert _ids(hits) == [_deep(1995, 5), _deep(1995, 4), _deep(1996, 5), _deep(1997, 5)]
    assert all(h.text == retriever.by_id[h.id].text for h in hits)


def test_searches_are_fused_by_rrf(monkeypatch):
    monkeypatch.setenv("FINANCE_MULTI_QUERY", "2")
    a, b, c = _deep(1991, 5), _deep(1992, 5), _deep(1993, 5)
    retriever = Retriever(_docs(), {"buybacks:": [a, b], "insurance:": [b, c]})
    plan = _plan(themes=("buybacks",), sectors=("insurance",))
    hits = mq.multi_query_source(plan, _store(), retriever, plan.query, 4)
    assert _ids(hits)[:3] == [b, a, c]  # found high by both searches first, then by the better rank
    searched = {q for q, _ in retriever.calls}
    assert any("buybacks:" in q for q in searched) and any("insurance:" in q for q in searched)


def test_a_hard_year_searches_only_its_range():
    docs = _docs(40)  # the 1995 ranking pools 30 of the 40 1995 passages (fetch_k); 1995_p0030-39 are new
    retriever = Retriever(docs, {"buybacks:": [_pid(1990, 39), _pid(1995, 39), _pid(1995, 38)]})
    plan = _plan(themes=("buybacks",), year_range=(1995, 1995), year_hard=True, years=(1995,),
                 year_spans=((1995, 1995),), year_source="explicit", kind="period")
    store = _store([_ann(d.id) for d in docs])
    hits = mq._source(plan, store, retriever, plan.query, 4, 1, False)
    assert _ids(hits)[:2] == [_pid(1995, 39), _pid(1995, 38)] and all(h.metadata["year"] == 1995 for h in hits)
    topic_calls = [where for q, where in retriever.calls if "buybacks:" in q]
    assert topic_calls == [{"year": {"$gte": 1995, "$lte": 1995}}]


def test_a_boost_range_serves_its_letters_first_and_keeps_the_rest():
    docs = _docs(20)  # the 1996-1997 ranking pools 30 of their 40 passages (n < 15)
    retriever = Retriever(docs, {"buybacks:": [_pid(1991, 19), _pid(1996, 19), _pid(1997, 18), _pid(1996, 3)]})
    plan = _plan(themes=("buybacks",), year_range=(1996, 1997), years=(), year_spans=((1996, 1997),),
                 year_source="explicit", kind="period")
    hits = mq._source(plan, _store([_ann(d.id) for d in docs]), retriever, plan.query, 4, 1, False)
    assert _ids(hits)[:2] == [_pid(1996, 19), _pid(1997, 18)] and _pid(1991, 19) in _ids(hits)
    assert _pid(1996, 3) not in _ids(hits)  # pooled by the 1996-1997 ranking already
    wheres = [where for q, where in retriever.calls if "buybacks:" in q]
    assert {"year": {"$gte": 1996, "$lte": 1997}} in wheres and None in wheres


def test_hits_of_the_sources_before_it_are_not_added_again(monkeypatch):
    monkeypatch.setenv("FINANCE_MULTI_QUERY", "1")
    monkeypatch.setenv("FINANCE_TAGGED_SOURCE", "1")
    a, b = _deep(1991, 5), _deep(1992, 5)
    earlier = lambda *args: [SearchHit(id=a, text="x", metadata={"year": 1991}, score=0.0)]  # noqa: E731
    monkeypatch.setattr(ks, "SOURCES", [("FINANCE_TAGGED_SOURCE", earlier), *mq.SOURCES])
    retriever = Retriever(_docs(), {"buybacks:": [a, b]})
    plan = _plan(themes=("buybacks",))
    assert mq.pooled(plan, _store(), retriever, plan.query, 4) >= {a, _pid(1990, 0)}
    assert _ids(mq.multi_query_source(plan, _store(), retriever, plan.query, 4))[:1] == [b]


def test_the_summary_index_rankings_join_the_fusion_only_with_mq_summary(monkeypatch):
    a, b = _deep(1991, 5), _deep(1992, 5)
    retriever = Retriever(_docs(), {"buybacks:": [a]})
    plan = _plan(themes=("buybacks",))
    seen = []

    def summary_ranking(plan_, store_, retriever_, text):
        seen.append(text)
        return [SearchHit(id=b, text="summary", metadata={"year": 1992}, score=1.0)]

    monkeypatch.setattr(mq.summary_source, "summary_ranking", summary_ranking)
    texts = mq.queries(plan, _store(), retriever, plan.query, 1, False)
    assert _ids(mq.fused_ranking(plan, _store(), retriever, texts))[0] == a and seen == []
    monkeypatch.setattr(mq, "MQ_SUMMARY", True)
    assert set(_ids(mq.fused_ranking(plan, _store(), retriever, texts))[:2]) == {a, b} and seen == texts


def test_non_financial_and_table_passages_are_skipped():
    a, b, c = _deep(1991, 5), _deep(1992, 5), _deep(1993, 5)
    anns = [_ann(d.id, scope="personal" if d.id == a else "financial", is_table=d.id == b) for d in _docs()]
    retriever = Retriever(_docs(), {"buybacks:": [a, b, c]})
    plan = _plan(themes=("buybacks",))
    assert _ids(mq._source(plan, _store(anns), retriever, plan.query, 4, 1, False))[:1] == [c]


def test_the_digest_source_runs_alone_and_defers_to_the_multi_query_source(monkeypatch):
    target = [_deep(1994, 5), _deep(1998, 5)]
    retriever = Retriever(_docs(), {"backs repurchases below value": target})
    plan = _plan(themes=("buybacks",))
    monkeypatch.setenv("FINANCE_DIGEST_QUERY", "1")
    assert _ids(mq.digest_query_source(plan, _store(), retriever, plan.query, 4))[:2] == target
    monkeypatch.setenv("FINANCE_MULTI_QUERY", "1")
    assert mq.digest_query_source(plan, _store(), retriever, plan.query, 4) == []
    both = mq.multi_query_source(plan, _store(), retriever, plan.query, 4)
    assert set(target) <= set(_ids(both))  # the digest search runs inside the multi-query source


# --- through finance_search ---
def test_finance_search_pools_the_new_candidates_only_when_on(monkeypatch):
    target = _deep(1995, 5)
    plan = _plan(themes=("buybacks",))
    store = _store()
    _, off = finance_search(plan, Retriever(_docs(), {"buybacks:": [target]}), store, k=4)
    assert target not in off["stage_pool"] and "FINANCE_MULTI_QUERY" not in off["source_candidates"]
    monkeypatch.setenv("FINANCE_MULTI_QUERY", "1")
    _, on = finance_search(plan, Retriever(_docs(), {"buybacks:": [target]}), store, k=4)
    assert target in on["stage_pool"] and on["source_candidates"]["FINANCE_MULTI_QUERY"] >= 1
    assert on["stage_pool"][:len(off["stage_pool"])] == off["stage_pool"] and on["source_errors"] == []
    not_view = _plan("Share repurchases at Berkshire", themes=("buybacks",))
    _, plain = finance_search(not_view, Retriever(_docs(), {"buybacks:": [target]}), store, k=4)
    assert plain["source_candidates"]["FINANCE_MULTI_QUERY"] == 0
