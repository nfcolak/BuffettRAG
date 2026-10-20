"""finance-recall-75-r3 group 3: the superlative route serves the winner's tagged passages with a stance
(FINANCE_SUPERLATIVE_TAGGED)."""
from __future__ import annotations

import pytest

from src.generation import finance_answer as fa
from src.knowledge import KnowledgeStore
from src.query.plan import QueryPlan
from src.storage import StoredDoc


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("FINANCE_SUPERLATIVE_TAGGED", "FINANCE_K_CORPUS", "FINANCE_DIGEST_SOURCE", "FINANCE_STANCE_SOURCE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FINANCE_K_CORPUS", "0")  # the budget is the caller's max_hits
    monkeypatch.setenv("FINANCE_DIGEST_SOURCE", "0")
    monkeypatch.setenv("FINANCE_STANCE_SOURCE", "0")
    monkeypatch.setenv("FINANCE_SUPERLATIVE_TAGGED", "0")  # a test turns the setting on


def _ann(pid, sectors=(), stance=(), companies=(), scope="financial"):
    rows = [{"target": t, "polarity": p, "quote": f"quote {pid}"} for t, p in stance]
    return {"id": pid, "scope": scope, "sectors": list(sectors), "companies": list(companies), "stance": rows}


def _store(extra=()):
    ins = ["insurance"]
    annotations = [
        _ann("1980_p0001", ins, [("insurance", "positive")]),
        _ann("1985_p0003", ins, [("GEICO", "positive")]),                       # a company of the sector: 1 point
        _ann("1985_p0004", ["railroads"], [("BNSF", "positive")]),              # tagged with another sector
        _ann("1990_p0001", ins, [("insurance", "positive")]),
        _ann("1990_p0002", ins, [("insurance", "positive"), ("GEICO", "positive")]),  # stronger than 1990_p0001
        _ann("1995_p0005", ins, [("insurance", "negative"), ("GEICO", "positive")]),  # the opposite on the winner
        _ann("1995_p0006", ins, [("float", "positive")]),                       # a theme
        _ann("1996_p0007", ins, [("banking", "positive")]),                     # another sector
        _ann("1997_p0008", ins, [("GEICO", "positive")], scope="non_financial"),
        _ann("1998_p0009", ins, [("GEICO", "positive")]),                       # not among the docs
        _ann("1999_p0010", ["banking"], [("insurance", "positive")]),           # a stance on the winner, tagged elsewhere
        _ann("2000_p0001", ins, [("insurance", "positive")]),
        _ann("2010_p0001", ins, [("insurance", "positive")]),
        _ann("2012_p0001", ["banking"], [("banking", "positive")]),
        _ann("2013_p0001", ["banking"], [("banking", "positive")]),
        *extra,
    ]
    taxonomy = {"sectors": {"insurance": "", "railroads": "", "banking": ""}, "themes": {"float": ""}}
    docs = {a["id"]: StoredDoc(id=a["id"], text=f"text {a['id']}", metadata={"year": int(a["id"][:4])})
            for a in annotations if a["id"] != "1998_p0009"}
    return KnowledgeStore(annotations=annotations, taxonomy=taxonomy), docs


def _pool(store, docs, n=99, plan=None, kind="sector", winner="insurance", polarity="positive", skip=()):
    plan = plan or QueryPlan(query="q", kind="corpus_wide", superlative=f"{kind}:{polarity}")
    return fa.tagged_stance_ids(plan, store, docs, kind, winner, polarity, set(skip), n)


# --- the pool ---
def test_pool_is_the_winners_tagged_passages_with_a_stance_of_the_asked_polarity():
    store, docs = _store()
    pool = _pool(store, docs)
    assert set(pool) == {"1980_p0001", "1985_p0003", "1990_p0001", "1990_p0002", "1999_p0010", "2000_p0001", "2010_p0001"}
    assert len(pool) == len(set(pool))


def test_pool_leaves_out_other_sectors_themes_opposite_polarity_non_financial_and_missing_docs():
    store, docs = _store()
    pool = set(_pool(store, docs))
    assert "1985_p0004" not in pool    # tagged railroads, positive on a company of railroads
    assert "1995_p0005" not in pool    # negative on insurance itself
    assert "1995_p0006" not in pool    # positive on a theme only
    assert "1996_p0007" not in pool    # positive on another sector only
    assert "1997_p0008" not in pool    # annotated non-financial
    assert "1998_p0009" not in pool    # not a served document
    assert not pool & {"2012_p0001", "2013_p0001"}


def test_strongest_passage_comes_first_within_a_letter_and_letters_come_in_rounds():
    store, docs = _store()
    pool = _pool(store, docs)
    assert pool[:6] == ["1980_p0001", "1985_p0003", "1990_p0002", "1999_p0010", "2000_p0001", "2010_p0001"]
    assert pool[6:] == ["1990_p0001"]  # the second round: the letters' second passages


def test_a_short_list_is_spread_across_the_years():
    store, docs = _store()
    assert _pool(store, docs, n=3) == ["1980_p0001", "1990_p0002", "2010_p0001"]
    assert _pool(store, docs, n=0) == [] and _pool(store, docs, n=-2) == []


def test_skip_and_year_range_limit_the_pool():
    store, docs = _store()
    assert "1990_p0002" not in _pool(store, docs, skip={"1990_p0002"})
    ranged = _pool(store, docs, plan=QueryPlan(query="q", superlative="sector:positive", year_range=(1984, 1995)))
    assert ranged == ["1985_p0003", "1990_p0002", "1990_p0001"]


def test_negative_superlative_takes_negative_stance_passages():
    annotations = [_ann("1980_p0001", ["banking"], [("banking", "negative")]),
                   _ann("1990_p0001", ["banking"], [("banking", "negative")]),
                   _ann("1990_p0002", ["banking"], [("banking", "positive")]),
                   _ann("2000_p0001", ["banking"], [("Wells", "negative")])]
    taxonomy = {"sectors": {"banking": ""}}
    docs = {a["id"]: StoredDoc(id=a["id"], text="t", metadata={"year": int(a["id"][:4])}) for a in annotations}
    store = KnowledgeStore(annotations=annotations, taxonomy=taxonomy)
    assert _pool(store, docs, kind="sector", winner="banking", polarity="negative") == [
        "1980_p0001", "1990_p0001", "2000_p0001"]


def test_company_winner_takes_only_stances_on_the_company_itself():
    annotations = [_ann("1980_p0001", [], [("GEICO", "positive")], companies=["GEICO"]),
                   _ann("1985_p0001", [], [("BNSF", "positive")], companies=["GEICO", "BNSF"]),
                   _ann("1990_p0001", [], [("GEICO", "positive")]),
                   _ann("1995_p0001", ["insurance"], [("GEICO", "positive")])]
    docs = {a["id"]: StoredDoc(id=a["id"], text="t", metadata={"year": int(a["id"][:4])}) for a in annotations}
    store = KnowledgeStore(annotations=annotations, taxonomy={"sectors": {"insurance": ""}})
    assert _pool(store, docs, kind="company", winner="GEICO") == ["1980_p0001", "1990_p0001", "1995_p0001"]


# --- the superlative route ---
PLAN = QueryPlan(query="q", kind="corpus_wide", superlative="sector:positive")
CITED = ["1980_p0001", "1990_p0001", "1999_p0010", "2000_p0001", "2010_p0001"]  # one stance quote per letter


def _more():
    return [_ann(f"{y}_p0011", ["insurance"], [("GEICO", "positive")]) for y in (1983, 1986, 1987, 1992, 1994, 2003)]


def test_off_by_default_the_route_is_unchanged(monkeypatch):
    store, docs = _store(_more())
    off = fa.superlative_answer(PLAN, store, docs, max_hits=10)
    monkeypatch.delenv("FINANCE_SUPERLATIVE_TAGGED")
    assert [h.id for h in fa.superlative_answer(PLAN, store, docs, max_hits=10).hits] == [h.id for h in off.hits]
    assert [h.id for h in off.hits] == CITED


def test_on_serves_tagged_passages_after_the_cited_ones_and_keeps_the_answer(monkeypatch):
    store, docs = _store(_more())
    off = fa.superlative_answer(PLAN, store, docs, max_hits=10)
    monkeypatch.setenv("FINANCE_SUPERLATIVE_TAGGED", "3")
    on = fa.superlative_answer(PLAN, store, docs, max_hits=10)
    served = [h.id for h in on.hits]
    assert served[:len(CITED)] == CITED and len(served) == len(CITED) + 3 == len(set(served))
    assert on.answer == off.answer and on.citations == off.citations and on.detail["cited"] == len(CITED)
    pool = set(_pool(store, docs, skip=set(CITED)))
    assert set(served[len(CITED):]) <= pool
    assert len({int(pid[:4]) for pid in served[len(CITED):]}) == 3  # three different letters


def test_the_tagged_passages_take_their_places_from_the_served_budget(monkeypatch):
    store, docs = _store(_more())
    monkeypatch.setenv("FINANCE_SUPERLATIVE_TAGGED", "50")
    assert len(fa.superlative_answer(PLAN, store, docs, max_hits=len(CITED) + 3).hits) == len(CITED) + 3
    assert len(fa.superlative_answer(PLAN, store, docs, max_hits=len(CITED)).hits) == len(CITED)
    monkeypatch.setenv("FINANCE_K_CORPUS", str(len(CITED) + 5))  # the larger corpus budget wins over max_hits
    assert len(fa.superlative_answer(PLAN, store, docs, max_hits=len(CITED)).hits) == len(CITED) + 5


def test_the_tagged_passages_come_before_the_digest_timeline_ones(monkeypatch):
    digest_ids = ["1984_p0020", "1989_p0021", "1993_p0022", "2005_p0023"]  # tagged, but without a stance
    store, docs = _store(_more() + [_ann(pid, ["insurance"]) for pid in digest_ids])
    store._digests = [{"topic": "insurance", "kind": "sector", "overview": "o",
                       "timeline": [{"year": int(pid[:4]), "quote": "q", "source_id": pid} for pid in digest_ids]}]
    monkeypatch.setenv("FINANCE_DIGEST_SOURCE", "1")
    budget = len(CITED) + 3
    off = [h.id for h in fa.superlative_answer(PLAN, store, docs, max_hits=budget).hits]
    assert len(off) == budget and set(off[len(CITED):]) <= set(digest_ids)
    monkeypatch.setenv("FINANCE_SUPERLATIVE_TAGGED", "1")
    served = [h.id for h in fa.superlative_answer(PLAN, store, docs, max_hits=budget).hits]
    assert served[len(CITED)] in _pool(store, docs, skip=set(CITED)) and len(served) == budget
    assert len(set(served[len(CITED) + 1:]) & set(digest_ids)) == 2  # the rest of the budget goes to the digest


def test_the_window_cut_drops_the_lowest_ranked_passages_beyond_max_hits_first(monkeypatch):
    store, docs = _store(_more())
    monkeypatch.setenv("FINANCE_SUPERLATIVE_TAGGED", "6")
    monkeypatch.setenv("FINANCE_K_CORPUS", str(len(CITED) + 6))
    full = [h.id for h in fa.superlative_answer(PLAN, store, docs, max_hits=len(CITED)).hits]
    assert len(full) == len(CITED) + 6
    text = len(docs[full[0]].text)
    monkeypatch.setattr(fa, "reader_window_chars", lambda max_new_tokens: text * (len(CITED) + 2))
    cut = [h.id for h in fa.superlative_answer(PLAN, store, docs, max_hits=len(CITED)).hits]
    assert cut == full[:len(CITED) + 2]
    monkeypatch.setattr(fa, "reader_window_chars", lambda max_new_tokens: 1)
    assert [h.id for h in fa.superlative_answer(PLAN, store, docs, max_hits=len(CITED)).hits] == CITED  # max_hits stays
