"""finance-recall-75-r2 group 3: source slots below the rerank cut, the plan-term rerank boost, the superlative budget."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from config import FINANCE_DEFAULTS
from src.generation import finance_answer as fa
from src.knowledge import KnowledgeStore
from src.query.plan import QueryPlan
from src.retrieval import finance_policy, knowledge_source, summary_source
from src.retrieval.finance_policy import _reserve_slots, boost_plan_terms, finance_search
from src.storage import SearchHit, StoredDoc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_finance_policy import FakeRetriever, _hit, _plan  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "knowledge_mini"
SOURCE_SETTINGS = ("FINANCE_SUMMARY_SOURCE", "FINANCE_TAGGED_SOURCE", "FINANCE_DIGEST_SOURCE", "FINANCE_STANCE_SOURCE")


@pytest.fixture(scope="module")
def store():
    return KnowledgeStore.load(FIX)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("FINANCE_K", "FINANCE_K_TREND", "FINANCE_K_CORPUS", "FINANCE_K_PERIOD", "FINANCE_K_SPECIFIC",
                 "FINANCE_SOURCE_SLOTS", "FINANCE_PLAN_TERM_BOOST", "FINANCE_MAX_PER_LETTER", "FINANCE_MIN_LETTERS",
                 "LLM_CONTEXT_PASSAGES", "LLM_N_CTX", "KNOWLEDGE_DIR"):
        monkeypatch.delenv(name, raising=False)
    for name in SOURCE_SETTINGS:  # every source off; a test turns on the ones it checks
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("RETRIEVAL_DEDUP", "0")


class ScoreReranker:
    """Orders the window by the given raw scores and min-max normalises them, like the cross-encoder reranker."""

    def __init__(self, scores):
        self.scores = scores

    def rerank(self, query, candidates, top_k=8):
        raw = [self.scores.get(h.id, 0.0) for h in candidates]
        lo, span = min(raw), (max(raw) - min(raw)) or 1.0
        order = sorted(range(len(candidates)), key=lambda i: -raw[i])
        return [SearchHit(candidates[i].id, candidates[i].text, candidates[i].metadata, (raw[i] - lo) / span)
                for i in order[:top_k]]


def _register(monkeypatch, source):
    """One source on the FINANCE_TAGGED_SOURCE setting; the module sources are not run."""
    monkeypatch.setattr(finance_policy, "EXTRA_SOURCES", [("FINANCE_TAGGED_SOURCE", source)])
    monkeypatch.setattr(knowledge_source, "SOURCES", [])
    monkeypatch.setattr(summary_source, "SOURCES", [])
    monkeypatch.setenv("FINANCE_TAGGED_SOURCE", "1")


BASE = [_hit(f"198{n}_a{n}") for n in range(6)]
SCORES = {"1980_a0": .9, "1981_a1": .8, "1982_a2": .7, "1983_a3": .6, "1984_a4": .5, "1985_a5": .4,
          "1999_s1": .45, "1999_s2": .35, "1999_s3": .3}
SOURCES = [_hit("1999_s1"), _hit("1999_s2"), _hit("1999_s3")]


def _search(store, k=4, scores=SCORES, **kw):
    hits, debug = finance_search(_plan("specific"), FakeRetriever(BASE, reranker=ScoreReranker(scores)), store, k=k, **kw)
    return [h.id for h in hits], debug


# --- 3.1 FINANCE_SOURCE_SLOTS ---
def test_the_new_settings_default_to_the_adopted_values():
    # finance-recall-75-r2 group 4: the plan-term boost passed the adoption rule, the source slots did not
    assert FINANCE_DEFAULTS["FINANCE_SOURCE_SLOTS"] == 0 and FINANCE_DEFAULTS["FINANCE_PLAN_TERM_BOOST"] == 5


def test_no_slots_serve_the_rerank_order_and_report_none(store, monkeypatch):
    _register(monkeypatch, lambda *a: list(SOURCES))
    served, debug = _search(store)
    assert served == ["1980_a0", "1981_a1", "1982_a2", "1983_a3"]
    assert debug["source_slots"] == [] and debug["plan_term_boosted"] == 0


def test_slots_serve_the_best_cut_source_candidates_in_cross_encoder_order(store, monkeypatch):
    _register(monkeypatch, lambda *a: list(SOURCES))
    monkeypatch.setenv("FINANCE_SOURCE_SLOTS", "2")
    served, debug = _search(store)
    # the two best cut source candidates (s1 .45, s2 .35) take the places of the lowest-ranked other hits (a3, a2)
    assert served == ["1980_a0", "1981_a1", "1999_s1", "1999_s2"]
    assert debug["source_slots"] == ["1999_s1", "1999_s2"] and len(served) == debug["k"] == 4
    assert debug["stage_ranked"][:4] == ["1980_a0", "1981_a1", "1982_a2", "1983_a3"]  # the rerank order is unchanged


def test_slots_never_exceed_the_candidates_or_the_budget(store, monkeypatch):
    _register(monkeypatch, lambda *a: list(SOURCES))
    monkeypatch.setenv("FINANCE_SOURCE_SLOTS", "9")
    served, debug = _search(store)
    assert len(served) == 4 and debug["source_slots"] == ["1999_s1", "1999_s2", "1999_s3"]
    assert served[0] == "1980_a0"  # the best hit is never a victim


def test_a_source_candidate_the_budget_kept_uses_no_slot_and_is_no_victim(store, monkeypatch):
    _register(monkeypatch, lambda *a: list(SOURCES))
    monkeypatch.setenv("FINANCE_SOURCE_SLOTS", "1")
    served, debug = _search(store, scores={**SCORES, "1999_s1": .75})  # s1 ranks third: inside the budget
    assert served == ["1980_a0", "1981_a1", "1999_s1", "1999_s2"] and debug["source_slots"] == ["1999_s2"]


def test_slots_leave_a_search_without_source_candidates_unchanged(store, monkeypatch):
    _register(monkeypatch, lambda *a: [])
    plain, _ = _search(store)
    monkeypatch.setenv("FINANCE_SOURCE_SLOTS", "3")
    served, debug = _search(store)
    assert served == plain and debug["source_slots"] == []


def test_slots_keep_the_pinned_hits_first_and_the_budget(store, monkeypatch):
    _register(monkeypatch, lambda *a: list(SOURCES))
    monkeypatch.setenv("FINANCE_SOURCE_SLOTS", "2")
    served, debug = _search(store, pinned=[_hit("1970_pin")])
    assert served[0] == "1970_pin" and len(served) == 4 and debug["pinned"] == ["1970_pin"]
    assert debug["source_slots"] == ["1999_s1", "1999_s2"]


def test_reserve_slots_leaves_protected_hits_in_place():
    pool = [_hit("1980_a"), _hit("1981_b"), _hit("1982_c"), _hit("1999_s")]
    hits = _reserve_slots(pool[:3], pool, ["1982_c"], ["1999_s"], 1, 3, diverse=False)
    assert [h.id for h in hits] == ["1980_a", "1982_c", "1999_s"]  # b, not the protected c, was the victim


def test_reserve_slots_respects_the_per_letter_cap_of_a_diverse_question(monkeypatch):
    monkeypatch.setenv("FINANCE_MIN_LETTERS", "1")
    pool = [_hit("1980_a"), _hit("1981_b"), _hit("1980_s")]
    added: list = []
    hits = _reserve_slots(pool[:2], pool, [], ["1980_s"], 1, 2, diverse=True, cap=1, added=added)
    # replacing b (the lowest-ranked) would put two passages into 1980; replacing a keeps one per letter
    assert [h.id for h in hits] == ["1981_b", "1980_s"] and added == ["1980_s"]
    none = _reserve_slots(pool[:2], pool, ["1980_a", "1981_b"], ["1980_s"], 1, 2, diverse=True, cap=1)
    assert [h.id for h in none] == ["1980_a", "1981_b"]  # no victim: both are protected


def test_reserve_slots_fills_a_free_place_before_replacing():
    pool = [_hit("1980_a"), _hit("1981_b"), _hit("1999_s")]
    assert [h.id for h in _reserve_slots(pool[:2], pool, [], ["1999_s"], 1, 3, diverse=False)] \
        == ["1980_a", "1981_b", "1999_s"]


# --- 3.1 FINANCE_PLAN_TERM_BOOST ---
BANKING_PLAN = _plan("specific", sectors=("banking",))


def _ranked():
    """Reranked window: a housing passage first, the banking passage just behind it, an untagged passage last."""
    return [SearchHit("2008_p0067", "t", {"year": 2008}, 1.0), SearchHit("2008_p0011", "t", {"year": 2008}, .93),
            SearchHit("1986_p0118", "t", {"year": 1986}, .5)]


def test_plan_term_boost_adds_hundredths_to_tagged_candidates_for_the_order_only(store, monkeypatch):
    monkeypatch.setenv("FINANCE_PLAN_TERM_BOOST", "0")  # off
    plain, count = boost_plan_terms(BANKING_PLAN, store, _ranked())
    assert [h.id for h in plain] == ["2008_p0067", "2008_p0011", "1986_p0118"] and count == 0
    monkeypatch.setenv("FINANCE_PLAN_TERM_BOOST", "5")  # .93 + .05 = .98 < 1.0
    small, count = boost_plan_terms(BANKING_PLAN, store, _ranked())
    assert [h.id for h in small] == ["2008_p0067", "2008_p0011", "1986_p0118"] and count == 1
    monkeypatch.setenv("FINANCE_PLAN_TERM_BOOST", "10")  # .93 + .10 = 1.03 > 1.0
    big, count = boost_plan_terms(BANKING_PLAN, store, _ranked())
    assert [h.id for h in big] == ["2008_p0011", "2008_p0067", "1986_p0118"] and count == 1
    assert [h.score for h in big] == [.93, 1.0, .5]  # the scores themselves are unchanged


def test_plan_term_boost_covers_themes_companies_and_events(store, monkeypatch):
    monkeypatch.setenv("FINANCE_PLAN_TERM_BOOST", "100")
    for field, value in (("themes", ("debt_leverage",)), ("companies", ("Clayton Homes",)),
                         ("events", ("financial_crisis_2008",))):
        ranked = [SearchHit("1986_p0118", "t", {"year": 1986}, 1.0), SearchHit("2008_p0067", "t", {"year": 2008}, .2)]
        boosted, count = boost_plan_terms(_plan("specific", **{field: value}), store, ranked)
        assert boosted[0].id == "2008_p0067" and count == 1, field


def test_plan_term_boost_needs_plan_entities_and_a_store(store, monkeypatch):
    monkeypatch.setenv("FINANCE_PLAN_TERM_BOOST", "100")
    assert boost_plan_terms(_plan("specific"), store, _ranked())[0] == _ranked()  # nothing named: unchanged
    assert boost_plan_terms(BANKING_PLAN, KnowledgeStore.empty(), _ranked())[0] == _ranked()


def test_plan_term_boost_reaches_the_served_order_of_finance_search(store, monkeypatch):
    ranking = [_hit("2008_p0067"), _hit("2008_p0011"), _hit("1986_p0118")]
    scores = {"2008_p0067": 1.0, "2008_p0011": .93, "1986_p0118": 0.0}  # min-max normalised: 1.0, .93, 0

    def run():
        hits, debug = finance_search(BANKING_PLAN, FakeRetriever(ranking, reranker=ScoreReranker(scores)), store, k=1)
        return [h.id for h in hits], debug

    assert run()[0] == ["2008_p0067"]
    monkeypatch.setenv("FINANCE_PLAN_TERM_BOOST", "10")
    served, debug = run()
    assert served == ["2008_p0011"] and debug["plan_term_boosted"] == 1


# --- 3.2 superlative route: the stance passages up to FINANCE_K_CORPUS or the reader window ---
TARGETS = {"insurance": range(1980, 1990), "railroads": range(1990, 1998), "banking": range(2000, 2006)}
PLAN_SUPERLATIVE = QueryPlan(query="q", kind="corpus_wide", superlative="sector:positive")


def _stance_store(polarity="positive"):
    annotations = [{"id": f"{y}_p0001", "scope": "financial",
                    "stance": [{"target": target, "polarity": polarity, "quote": f"quote {y}"}]}
                   for target, years in TARGETS.items() for y in years]
    docs = {a["id"]: StoredDoc(id=a["id"], text=f"text {a['id']}", metadata={"year": int(a["id"][:4])})
            for a in annotations}
    return KnowledgeStore(annotations=annotations, taxonomy={"sectors": {t: "" for t in TARGETS}}), docs


def test_superlative_budget_is_the_callers_without_a_corpus_budget(monkeypatch):
    monkeypatch.setenv("FINANCE_STANCE_SOURCE", "1")
    monkeypatch.setenv("FINANCE_K_CORPUS", "0")  # no corpus budget
    store, docs = _stance_store()
    result = fa.superlative_answer(PLAN_SUPERLATIVE, store, docs, max_hits=16)
    assert len(result.hits) == 16 and result.detail["evidence"] == 16 - result.detail["cited"]


def test_superlative_corpus_budget_serves_more_stance_passages_of_the_top_targets(monkeypatch):
    monkeypatch.setenv("FINANCE_STANCE_SOURCE", "1")
    monkeypatch.setenv("FINANCE_K_CORPUS", "0")  # today's 16 first
    store, docs = _stance_store()
    before = fa.superlative_answer(PLAN_SUPERLATIVE, store, docs, max_hits=16)
    monkeypatch.setenv("FINANCE_K_CORPUS", "24")
    after = fa.superlative_answer(PLAN_SUPERLATIVE, store, docs, max_hits=16)
    served = [h.id for h in after.hits]
    assert len(served) == len(set(served)) == 24  # every positive stance passage of the three targets
    assert {h.id for h in before.hits} <= set(served)  # the 16 of today are still served
    assert after.answer == before.answer and after.citations == before.citations  # only uncited evidence grows
    assert after.detail["cited"] == before.detail["cited"] and after.detail["evidence"] == 24 - after.detail["cited"]
    assert {r[0] for r in after.detail["ranking"]} == set(TARGETS)


def test_superlative_corpus_budget_below_the_callers_does_not_lower_it(monkeypatch):
    monkeypatch.setenv("FINANCE_STANCE_SOURCE", "1")
    monkeypatch.setenv("FINANCE_K_CORPUS", "8")
    store, docs = _stance_store()
    assert len(fa.superlative_answer(PLAN_SUPERLATIVE, store, docs, max_hits=16).hits) == 16


def test_superlative_corpus_budget_without_the_stance_source_adds_the_winners_letters(monkeypatch):
    store, docs = _stance_store()
    monkeypatch.setenv("FINANCE_K_CORPUS", "0")
    off = fa.superlative_answer(PLAN_SUPERLATIVE, store, docs, max_hits=16)
    monkeypatch.setenv("FINANCE_K_CORPUS", "24")
    on = fa.superlative_answer(PLAN_SUPERLATIVE, store, docs, max_hits=16)
    assert {h.id for h in off.hits} == {h.id for h in on.hits} == {f"{y}_p0001" for y in TARGETS["insurance"]}


def test_superlative_extra_passages_must_fit_the_reader_window(monkeypatch):
    monkeypatch.setenv("FINANCE_STANCE_SOURCE", "1")
    monkeypatch.setenv("FINANCE_K_CORPUS", "24")
    store, docs = _stance_store()
    text = len("text 1980_p0001")  # every passage has this length
    monkeypatch.setenv("LLM_N_CTX", str(900 + 20 * text // 4))  # a window of 20 passages
    assert len(fa.superlative_answer(PLAN_SUPERLATIVE, store, docs, max_hits=16).hits) == 20
    monkeypatch.setenv("LLM_N_CTX", str(900 + 10 * text // 4))  # a window of 10: today's budget stays
    assert len(fa.superlative_answer(PLAN_SUPERLATIVE, store, docs, max_hits=16).hits) == 16
    monkeypatch.setenv("LLM_N_CTX", "16384")
    assert len(fa.superlative_answer(PLAN_SUPERLATIVE, store, docs, max_hits=16).hits) == 24


def test_superlative_negative_polarity_spreads_the_negative_passages_too(monkeypatch):
    monkeypatch.setenv("FINANCE_STANCE_SOURCE", "1")
    monkeypatch.setenv("FINANCE_K_CORPUS", "24")
    store, docs = _stance_store("negative")
    plan = QueryPlan(query="q", kind="corpus_wide", superlative="sector:negative")
    result = fa.superlative_answer(plan, store, docs, max_hits=16)
    assert len(result.hits) == 24 and "negatively" in result.answer
    years = sorted(int(h.id[:4]) for h in result.hits[result.detail["cited"]:])
    assert years[0] < 1990 and years[-1] >= 2000  # evidence from the first and the last target's letters
