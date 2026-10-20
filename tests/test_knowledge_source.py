"""finance-recall-75 group 3: knowledge-guided candidate sources and the superlative route's served evidence."""
from __future__ import annotations

import math

import pytest

from config import FINANCE_DEFAULTS, finance_setting
from src.generation import finance_answer as fa
from src.knowledge import KnowledgeStore
from src.query.plan import QueryPlan
from src.retrieval import finance_policy
from src.retrieval import knowledge_source as ks
from src.retrieval.bm25 import _meta_matches
from src.retrieval.finance_policy import candidate_sources, finance_search
from src.storage import SearchHit, StoredDoc

SETTINGS = ("FINANCE_TAGGED_SOURCE", "FINANCE_DIGEST_SOURCE", "FINANCE_STANCE_SOURCE")
YEARS = range(1990, 2000)
PER_LETTER_DOCS = 6


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (*SETTINGS, "FINANCE_K", "FINANCE_OVERFETCH", "FINANCE_TAG_DEPTH", "FINANCE_TAG_POOL",
                 "LLM_CONTEXT_PASSAGES", "FINANCE_RERANK_CANDIDATES"):
        monkeypatch.delenv(name, raising=False)
    for name in (*SETTINGS, "FINANCE_SUMMARY_SOURCE"):  # every source off; a test turns on the one it checks
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("RETRIEVAL_DEDUP", "0")
    monkeypatch.setenv("FINANCE_TAG_DEPTH", "30")  # = the shallow depth (fetch_k) at k=4: only ranks 0-29 are pooled


def _pid(year, n):
    return f"{year}_p{n:04d}"


def _docs():
    return [StoredDoc(id=_pid(y, n), text=f"text {y} {n}", metadata={"year": y}) for n in range(PER_LETTER_DOCS)
            for y in YEARS]


def _ann(pid, **kw):
    return {"id": pid, "scope": kw.pop("scope", "financial"), "is_table": kw.pop("is_table", False),
            "sectors": [], "themes": [], "events": [], "companies": [], "stance": [], **kw}


class Retriever:
    """Hybrid ranking = corpus order (every letter's n=0 passage first, ranks 0-9, then n=1, 10-19, ...) or `order`."""

    def __init__(self, docs, embedder=None, reranker=None, order=None):
        self.docs, self.embedder, self.reranker = list(docs), embedder, reranker
        by_id = {d.id: d for d in self.docs}
        self.ranking = [SearchHit(id=pid, text=by_id[pid].text, metadata=dict(by_id[pid].metadata), score=1.0)
                        for pid in (order or by_id)]

    def _hybrid_for_filter(self, query, fetch_k, where):
        return [h for h in self.ranking if not where or _meta_matches(h.metadata, where)][:fetch_k]


THEMES = {"buybacks": "share repurchases. A tie-break note.", "dividends": "dividends and retained earnings.",
          "hub": "Berkshire's yearly results."}


def _unit(v):
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


class Embedder:
    """Theme descriptions: buybacks e1, dividends e2, hub e3; the anchor sits next to the hub."""
    DOCS = {"buybacks": [1, 0, 0], "dividends": [0, 1, 0], "hub": [0, 0, 1]}
    QUERIES = {ks.THEME_ANCHOR: _unit([0.1, 0.1, 0.9]),
               "repurchase question": _unit([0.8, 0.0, 0.6]),     # buybacks; the hub's 0.6 is below its anchor cosine
               "payout question": _unit([0.6, 0.55, 0.58]),       # buybacks and dividends
               "vague question": _unit([0.45, 0.0, 0.89])}        # nothing: buybacks under THEME_MIN_SIM

    def __init__(self):
        self.documents = []

    def embed_documents(self, texts):
        self.documents.append(list(texts))
        return [self.DOCS[t.split(":")[0]] for t in texts]

    def embed_query(self, text):
        return self.QUERIES[text]


def _plan(kind="specific", query="q", **kw):
    return QueryPlan(query=query, financial_query=query, kind=kind, **kw)


def _ids(hits):
    return [h.id for h in hits]


# --- registry and settings ---
def test_three_sources_registered_each_with_its_setting(monkeypatch):
    assert [name for name, _ in ks.SOURCES] == list(SETTINGS)
    # group 5 adopted the tagged and digest sources (finance-recall-75, task 5.1); the stance source stays off
    assert [FINANCE_DEFAULTS[name] for name in SETTINGS] == [1, 1, 0]
    assert all(finance_setting(name) == 0 for name in SETTINGS)  # switched off by the fixture
    assert not [name for name, _ in candidate_sources() if name in SETTINGS]
    monkeypatch.setenv("FINANCE_DIGEST_SOURCE", "1")
    assert [name for name, _ in candidate_sources() if name in SETTINGS] == ["FINANCE_DIGEST_SOURCE"]
    assert all(name in [n for n, _ in finance_policy.registered_sources()] for name in SETTINGS)


# --- helpers ---
def test_spread_letters_takes_one_per_letter_per_round_evenly_over_time():
    ids = [_pid(y, n) for y in YEARS for n in range(2)]
    letter = lambda pid: int(pid[:4])  # noqa: E731
    assert ks.spread_letters(ids, 4, letter) == [_pid(1990, 0), _pid(1993, 0), _pid(1996, 0), _pid(1999, 0)]
    twelve = ks.spread_letters(ids, 12, letter)
    assert twelve[:10] == [_pid(y, 0) for y in YEARS] and twelve[10:] == [_pid(1990, 1), _pid(1999, 1)]
    assert ks.spread_letters(ids[:3], 10, letter) == [_pid(1990, 0), _pid(1991, 0), _pid(1990, 1)]


def test_stance_passages_by_target_and_polarity():
    store = KnowledgeStore(annotations=[
        _ann("1990_p0001", stance=[{"target": "insurance", "polarity": "positive"},
                                   {"target": "GEICO", "polarity": "negative"}]),
        _ann("1991_p0001", stance=[{"target": "insurance", "polarity": "negative"}])])
    assert ks.stance_passages(store, ["insurance", "GEICO"]) == {"insurance": ["1990_p0001", "1991_p0001"],
                                                                 "GEICO": ["1990_p0001"]}
    assert ks.stance_passages(store, ["insurance"], "negative") == {"insurance": ["1991_p0001"]}


# --- theme inference ---
def test_infer_themes_threshold_margin_over_anchor_and_top():
    store = KnowledgeStore(taxonomy={"themes": THEMES})
    embedder = Embedder()
    assert ks.infer_themes("repurchase question", store, embedder) == ("buybacks",)  # the hub is near the anchor
    assert ks.infer_themes("payout question", store, embedder) == ("buybacks", "dividends")
    assert ks.infer_themes("vague question", store, embedder) == ()
    assert embedder.documents == [["buybacks: share repurchases.", "dividends: dividends and retained earnings.",
                                   "hub: Berkshire's yearly results."]]  # first sentence only, embedded once
    assert ks.infer_themes("payout question", KnowledgeStore(), embedder) == ()
    assert ks.infer_themes("payout question", store, None) == ()


# --- tagged source ---
def _tag_store(**extra):
    annotations = []
    for y in YEARS:
        for n in range(PER_LETTER_DOCS):
            themes = ["buybacks"] if n in (1, 3, 4, 5) else (["dividends"] if n == 2 else [])
            annotations.append(_ann(_pid(y, n), themes=themes))
    annotations[[a["id"] for a in annotations].index("1995_p0004")]["is_table"] = True
    annotations[[a["id"] for a in annotations].index("1996_p0005")]["scope"] = "personal_anecdote"
    return KnowledgeStore(annotations=annotations, taxonomy={"themes": THEMES}, **extra)


def test_tagged_source_skips_pooled_passages_tables_and_non_financial(monkeypatch):
    monkeypatch.setenv("FINANCE_TAG_DEPTH", "40")  # plan-tagged passages above rank 40 are pooled by finance_search
    store = _tag_store()
    hits = ks.tagged_source(_plan(themes=("buybacks",)), store, Retriever(_docs()), "q", 4)
    expected = [_pid(y, 4) for y in YEARS if y != 1995] + [_pid(y, 5) for y in YEARS if y != 1996]
    assert _ids(hits) == expected                      # n=1 (rank 10-19) and n=3 (rank 30-39) are in the pool
    assert hits[0].text == "text 1990 4" and hits[0].metadata == {"year": 1990}
    numeric = ks.tagged_source(_plan("numeric", themes=("buybacks",)), store, Retriever(_docs()), "q", 4)
    assert "1995_p0004" in _ids(numeric)               # tables only for numeric plans


def test_tagged_source_cap_and_per_letter(monkeypatch):
    store = _tag_store()
    monkeypatch.setattr(ks, "TAGGED_CAP", 5)
    hits = ks.tagged_source(_plan(themes=("buybacks",)), store, Retriever(_docs()), "q", 4)
    assert _ids(hits) == [_pid(y, 3) for y in range(1990, 1995)]
    # one letter tops the ranking: at most PER_LETTER of it, the rest of the cap goes to the next letters
    pooled = [d.id for d in _docs() if int(d.id[-1]) < 3]
    letter_1993_first = pooled + [_pid(1993, n) for n in (3, 4, 5)] + [d.id for d in _docs()
                                                                       if int(d.id[-1]) >= 3 and d.id[:4] != "1993"]
    hits = ks.tagged_source(_plan(themes=("buybacks",)), store, Retriever(_docs(), order=letter_1993_first), "q", 4)
    assert _ids(hits) == [_pid(1993, 3), _pid(1993, 4), _pid(1990, 3), _pid(1991, 3), _pid(1992, 3)]


def test_tagged_source_uses_inferred_themes_and_nothing_without_tags(monkeypatch):
    monkeypatch.setenv("FINANCE_THEME_CENTROIDS", "0")  # the descriptions only (centroids: test_candidate_fusion_centroids.py)
    store = _tag_store()
    retriever = Retriever(_docs(), embedder=Embedder())
    assert ks.tagged_source(_plan(query="vague question"), store, retriever, "vague question", 4) == []
    hits = ks.tagged_source(_plan(query="payout question"), store, retriever, "payout question", 4)
    # inferred buybacks + dividends (n=1..5); the plan names no entity, so only ranks 0-29 (n=0..2) are pooled
    assert _ids(hits) == ([_pid(y, 3) for y in YEARS] + [_pid(y, 4) for y in YEARS if y != 1995]
                          + [_pid(y, 5) for y in range(1990, 1995)])


def test_tagged_source_year_range_first_and_hard_range_only():
    store = _tag_store()
    narrow = _plan(themes=("buybacks",), years=(1997,), year_spans=((1997, 1997),), year_range=(1997, 1997))
    hits = ks.tagged_source(narrow, store, Retriever(_docs()), "q", 4)
    assert _ids(hits)[:6] == [_pid(1997, 3), _pid(1998, 3), _pid(1997, 4), _pid(1998, 4), _pid(1997, 5),
                              _pid(1998, 5)]       # the letters 1997 and 1998 first
    assert len(_ids(hits)) == 24
    hard = _plan(themes=("buybacks",), years=(1997,), year_spans=((1997, 1997),), year_range=(1997, 1997),
                 year_hard=True)
    assert _ids(ks.tagged_source(hard, store, Retriever(_docs()), "q", 4)) == [_pid(1997, n) for n in (3, 4, 5)]


# --- digest source ---
def _digest_store(**kw):
    timeline = [{"year": y, "point": "p", "quote": "q", "source_id": _pid(y, 4)} for y in YEARS]
    timeline.append({"year": 1995, "point": "p", "quote": "q", "source_id": _pid(1995, 5)})
    timeline.append({"year": 1991, "point": "p", "quote": "q", "source_id": _pid(1991, 0)})  # pooled already
    digests = [{"topic": "buybacks", "kind": "theme", "overview": "o", "timeline": timeline}]
    annotations = [_ann(d.id) for d in _docs()]
    return KnowledgeStore(annotations=annotations, digests=digests, taxonomy={"themes": THEMES}, **kw)


def test_digest_source_trend_and_corpus_wide_only_spread_across_letters(monkeypatch):
    store = _digest_store()
    retriever = Retriever(_docs())
    assert ks.digest_source(_plan("specific", themes=("buybacks",)), store, retriever, "q", 4) == []
    hits = ks.digest_source(_plan("trend", themes=("buybacks",)), store, retriever, "q", 4)
    assert _ids(hits) == [_pid(y, 4) for y in YEARS] + [_pid(1995, 5)]  # timeline order; 1991_p0000 is pooled
    monkeypatch.setattr(ks, "DIGEST_CAP", 4)
    hits = ks.digest_source(_plan("corpus_wide", themes=("buybacks",)), store, retriever, "q", 4)
    assert _ids(hits) == [_pid(1990, 4), _pid(1993, 4), _pid(1996, 4), _pid(1999, 4)]


def test_digest_source_year_range_first(monkeypatch):
    monkeypatch.setattr(ks, "DIGEST_CAP", 3)
    plan = _plan("trend", themes=("buybacks",), year_range=(1995, 1999), year_spans=((1995, 1999),))
    hits = ks.digest_source(plan, _digest_store(), Retriever(_docs()), "q", 4)
    assert _ids(hits) == [_pid(1995, 4), _pid(1997, 4), _pid(1999, 4)]


def test_digest_source_inferred_theme_only_without_a_named_digest():
    store = _digest_store()

    class Exploding(Embedder):
        def embed_query(self, text):
            raise AssertionError("a named digest topic needs no inference")

    named = ks.digest_source(_plan("trend", themes=("buybacks",)), store, Retriever(_docs(), Exploding()), "q", 4)
    assert named
    inferred = ks.digest_source(_plan("trend", query="repurchase question"), store,
                                Retriever(_docs(), Embedder()), "repurchase question", 4)
    assert _ids(inferred) == _ids(named)
    assert ks.digest_source(_plan("trend", query="vague question"), store, Retriever(_docs(), Embedder()),
                            "vague question", 4) == []


# --- stance source ---
def _stance_store():
    annotations = [_ann(d.id) for d in _docs()]
    by_id = {a["id"]: a for a in annotations}
    for y in YEARS:
        by_id[_pid(y, 4)]["stance"] = [{"target": "insurance", "polarity": "positive" if y % 2 else "negative"}]
    by_id[_pid(1990, 5)]["stance"] = [{"target": "GEICO", "polarity": "positive"}]
    by_id[_pid(1991, 5)]["stance"] = [{"target": "railroads", "polarity": "positive"}]
    return KnowledgeStore(annotations=annotations, taxonomy={"sectors": {"insurance": "", "railroads": ""}})


def test_stance_source_needs_a_view_question_on_a_sector_or_company():
    store = _stance_store()
    view = "What does Buffett think of insurance?"
    hits = ks.stance_source(_plan(query=view, sectors=("insurance",), companies=("GEICO",)), store,
                            Retriever(_docs()), view, 4)
    assert _ids(hits) == [_pid(y, 4) for y in YEARS] + [_pid(1990, 5)]
    figure = "How large was the insurance float?"
    assert ks.stance_source(_plan(query=figure, sectors=("insurance",)), store, Retriever(_docs()), figure, 4) == []
    assert ks.stance_source(_plan(query=view), store, Retriever(_docs()), view, 4) == []  # no target named


def test_stance_source_superlative_keeps_the_asked_polarity():
    plan = _plan("corpus_wide", query="q", sectors=("insurance",), superlative="sector:positive")
    hits = ks.stance_source(plan, _stance_store(), Retriever(_docs()), "q", 4)
    assert _ids(hits) == [_pid(y, 4) for y in YEARS if y % 2]


# --- finance_search integration ---
class Prefer:
    def __init__(self, best):
        self.best, self.seen = best, []

    def rerank(self, query, hits, top_k):
        self.seen = [h.id for h in hits]
        return sorted(hits, key=lambda h: h.id != self.best)[:top_k]


def test_tagged_source_hits_reach_the_rerank_window_only_when_on(monkeypatch):
    monkeypatch.setenv("FINANCE_PLAN_TERM_BOOST", "0")  # Prefer keeps the retrieval scores: a score bonus would re-sort them
    store = _tag_store()
    plan = _plan(themes=("buybacks",))
    reranker = Prefer("1990_p0005")  # rank 50: beyond the pool (fetch_k 30)
    hits, debug = finance_search(plan, Retriever(_docs(), reranker=reranker), store, k=4)
    assert "1990_p0005" not in reranker.seen and "FINANCE_TAGGED_SOURCE" not in debug["source_candidates"]
    monkeypatch.setenv("FINANCE_TAGGED_SOURCE", "1")
    hits, debug = finance_search(plan, Retriever(_docs(), reranker=reranker), store, k=4)
    assert "1990_p0005" in reranker.seen and hits[0].id == "1990_p0005"
    assert debug["source_candidates"]["FINANCE_TAGGED_SOURCE"] > 0 and "1990_p0005" in debug["stage_window"]


# --- superlative route (task 3.1b) ---
def _superlative_store():
    def ann(pid, target, polarity="positive"):
        return {"id": pid, "scope": "financial", "stance": [{"target": target, "polarity": polarity,
                                                             "quote": f"quote {pid}"}]}
    annotations = [ann(f"{y}_p0001", "insurance") for y in (1980, 1985, 1990, 1995, 2000)]
    annotations += [ann(f"{y}_p0002", "railroads") for y in (2010, 2011, 2012)]
    annotations += [ann("2015_p0003", "banking"), ann("2016_p0003", "banking", "negative")]
    timeline = [{"year": 1980, "quote": "digest quote", "source_id": "1980_p0001"},
                {"year": 1982, "quote": "not a stance passage", "source_id": "1982_p0009"},
                {"year": 1988, "quote": "nor this", "source_id": "1988_p0009"}]
    digest = {"topic": "insurance", "kind": "sector", "overview": "o", "timeline": timeline}
    taxonomy = {"sectors": {"insurance": "", "railroads": "", "banking": ""}}
    ids = [a["id"] for a in annotations] + ["1982_p0009", "1988_p0009"]
    docs = {pid: StoredDoc(id=pid, text=f"text {pid}", metadata={"year": int(pid[:4])}) for pid in ids}
    return KnowledgeStore(annotations=annotations, digests=[digest], taxonomy=taxonomy), docs


PLAN_SUPERLATIVE = QueryPlan(query="q", kind="corpus_wide", superlative="sector:positive")


def test_superlative_without_the_sources_serves_only_the_winners_stance_passages():
    store, docs = _superlative_store()
    result = fa.superlative_answer(PLAN_SUPERLATIVE, store, docs, max_hits=16)
    assert [t for t, _ in result.detail["ranking"]] == ["insurance", "railroads", "banking"]
    assert {h.id for h in result.hits} == {f"{y}_p0001" for y in (1980, 1985, 1990, 1995, 2000)}


def test_superlative_stance_source_serves_the_top_targets_within_the_budget(monkeypatch):
    monkeypatch.setenv("FINANCE_STANCE_SOURCE", "1")
    monkeypatch.setenv("FINANCE_K_CORPUS", "0")  # the budget is the caller's max_hits
    store, docs = _superlative_store()
    result = fa.superlative_answer(PLAN_SUPERLATIVE, store, docs, max_hits=16)
    served = [h.id for h in result.hits]
    assert set(served) == {a["id"] for a in store.annotations() if a["stance"][0]["polarity"] == "positive"}
    assert len(served) == len(set(served))
    cited = result.detail["cited"]
    assert served[cited:cited + 3] == ["2010_p0002", "2015_p0003", "2011_p0002"]  # other targets, in turns
    small = fa.superlative_answer(PLAN_SUPERLATIVE, store, docs, max_hits=cited + 2)
    assert len(small.hits) == cited + 2 and small.answer == result.answer  # the budget; the answer is unchanged


def test_superlative_digest_source_serves_the_winners_timeline(monkeypatch):
    store, docs = _superlative_store()
    off = fa.superlative_answer(PLAN_SUPERLATIVE, store, docs, max_hits=16)
    monkeypatch.setenv("FINANCE_DIGEST_SOURCE", "1")
    on = fa.superlative_answer(PLAN_SUPERLATIVE, store, docs, max_hits=16)
    assert {"1982_p0009", "1988_p0009"} <= {h.id for h in on.hits}
    assert not {"1982_p0009", "1988_p0009"} & {h.id for h in off.hits}
    assert on.answer == off.answer and on.citations == off.citations
    ranged = fa.superlative_answer(QueryPlan(query="q", kind="corpus_wide", superlative="sector:positive",
                                             year_range=(1984, 2005)), store, docs, max_hits=16)
    assert "1982_p0009" not in {h.id for h in ranged.hits}
