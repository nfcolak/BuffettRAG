"""finance-query-pipeline group 5: scope, numeric and digest answers and the trend prompt (knowledge_mini fixtures)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.generation import finance_answer as fa
from src.generation.prompt import SYSTEM_PROMPT, build_cited_prompt, system_prompt
from src.knowledge import KnowledgeStore
from src.query.plan import QueryPlan, plan_query
from src.storage import SearchHit, StoredDoc

FIX = Path(__file__).parent / "fixtures" / "knowledge_mini"


@pytest.fixture(scope="module")
def store():
    return KnowledgeStore.load(FIX)


@pytest.fixture(scope="module")
def docs():
    rows = [json.loads(line) for line in (FIX / "passages.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: StoredDoc(id=r["id"], text=r["text"], metadata={"year": int(r["id"][:4]), "source_file": f"{r['id'][:4]}.pdf"})
            for r in rows}


def _markers(answer):
    return sorted({int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer) for n in group.split(",")})


def _resolves(result):
    return all(1 <= n <= len(result.hits) for n in _markers(result.answer))


def test_scope_answer_is_the_fixed_reply_without_hits():
    result = fa.scope_answer()
    assert result.answer == "I only answer questions about Buffett's financial views and Berkshire's financial results."
    assert result.hits == [] and result.citations == [] and result.route == "scope"


def test_numeric_answer_gives_value_year_as_written_and_a_citation_to_its_source(store, docs):
    plan = plan_query("What was Berkshire's book value per share in 1987?", store)
    result = fa.numeric_answer(plan, store, docs)
    assert result is not None and result.route == "numeric"
    assert "1987" in result.answer and "$2,477.47" in result.answer and 'as written: "$2,477.47"' in result.answer
    assert [h.id for h in result.hits] == ["1987_p0000"]
    assert _markers(result.answer) == [1] and _resolves(result)
    assert result.citations and result.citations[0]["passage_indices"] == [0]


def test_numeric_answer_shares_one_hit_between_rows_of_the_same_passage(store, docs):
    plan = plan_query("How much did Berkshire's net worth gain in 1987?", store)
    result = fa.numeric_answer(plan, store, docs)
    assert result is not None
    assert "$464 million" in result.answer and len(result.hits) == 1 and _markers(result.answer) == [1]


def test_numeric_answer_lists_years_in_order_when_the_question_names_none(store, docs):
    plan = plan_query("What was the holding cost of USAir?", store)
    result = fa.numeric_answer(plan, store, docs)
    assert result is not None
    assert [h.id for h in result.hits] == ["1990_p0100", "1991_p0083"]
    assert result.answer.index("1990") < result.answer.index("1991") and _resolves(result)


def test_numeric_answer_is_none_when_nothing_matches_or_the_source_is_not_served(store, docs):
    assert fa.numeric_answer(plan_query("How much was the widget backlog in 1987?", store), store, docs) is None
    assert fa.numeric_answer(plan_query("What was Berkshire's book value per share in 1987?", store), store, {}) is None
    assert fa.numeric_answer(plan_query("What was Berkshire's book value per share in 1977?", store), store, docs) is None
    assert fa.numeric_answer(QueryPlan(query="q", kind="numeric"), KnowledgeStore.empty(), docs) is None


def test_numeric_answer_does_not_use_a_row_of_another_company(store, docs):
    plan = plan_query("How much was the holding cost of Washington Post in 1990?", store)
    assert fa.numeric_answer(plan, store, docs) is None  # USAir has a 1990 row, Washington Post only 1977


def test_digest_answer_orders_quotes_by_year_and_numbers_hits_in_that_order(store, docs):
    plan = plan_query("Across all the letters, how did Buffett see the airlines_aviation sector, the airline business?", store)
    plan = QueryPlan(query=plan.query, kind="corpus_wide", sectors=("airlines_aviation",))
    result = fa.digest_answer(plan, store, docs)
    assert result is not None and result.route == "digest"
    digest = store.digest("airlines_aviation")
    assert result.answer.startswith(digest["overview"])
    assert [h.id for h in result.hits] == ["1990_p0101", "1990_p0102", "1990_p0103"]
    assert _markers(result.answer) == [1, 2, 3] and _resolves(result)
    for entry in digest["timeline"]:
        assert f'{entry["year"]}: "{entry["quote"]}"' in result.answer
    assert [c["passage_indices"] for c in result.citations] == [[0], [1], [2]]


def test_digest_answer_sorts_a_timeline_that_is_out_of_order_and_shares_repeated_sources(store, docs):
    digest = dict(store.digest("financial_crisis_2008"))
    timeline = list(reversed(digest["timeline"]))  # 2010_p0142, 2010_p0119, 2008_p0093, 2008_p0011 (ties keep timeline order)
    timeline.append(dict(timeline[0], year=2010, quote="again"))
    scrambled = KnowledgeStore(digests=[dict(digest, timeline=timeline)])
    plan = QueryPlan(query="q", kind="corpus_wide", events=("financial_crisis_2008",))
    result = fa.digest_answer(plan, scrambled, docs)
    years = [int(y) for y in re.findall(r"^- (\d{4}):", result.answer, flags=re.M)]
    assert years == sorted(years)
    assert [h.id for h in result.hits[:4]] == ["2008_p0093", "2008_p0011", "2010_p0142", "2010_p0119"]
    assert _resolves(result) and len(result.hits) == 4  # the repeated source reuses its number


def test_digest_answer_appends_reader_passages_for_uncovered_letters_after_the_cited_ones(store, docs):
    plan = QueryPlan(query="q", kind="corpus_wide", sectors=("airlines_aviation",))
    reader = [SearchHit(pid, docs[pid].text, dict(docs[pid].metadata), 1.0)
              for pid in ("1990_p0102", "1991_p0072", "1994_p0004", "2003_p0087")]
    result = fa.digest_answer(plan, store, docs, reader, gap_passages=2)
    assert [h.id for h in result.hits] == ["1990_p0101", "1990_p0102", "1990_p0103", "1991_p0072", "1994_p0004"]
    assert _markers(result.answer) == [1, 2, 3] and _resolves(result)
    assert result.detail["cited"] == 3 and result.detail["gaps"] == 2


def test_digest_answer_is_none_without_a_digest_or_without_servable_sources(store, docs):
    assert fa.digest_answer(QueryPlan(query="q", kind="corpus_wide"), store, docs) is None
    assert fa.digest_answer(QueryPlan(query="q", kind="corpus_wide", sectors=("banking",)), store, docs) is None
    plan = QueryPlan(query="q", kind="corpus_wide", sectors=("airlines_aviation",))
    assert fa.digest_answer(plan, store, {}) is None


def test_trend_hits_are_served_oldest_first_and_the_markers_follow_that_order():
    hits = [SearchHit(f"{y}_x", f"text {y}", {"year": y}, 1.0) for y in (2008, 1981, 1999, 1981)]
    ordered = fa.sort_by_year(hits)
    assert [h.metadata["year"] for h in ordered] == [1981, 1981, 1999, 2008]
    prompt = fa.trend_prompt("How did it change?", ordered)
    assert prompt.index("[1] (year=1981") < prompt.index("[3] (year=1999") < prompt.index("[4] (year=2008")


def test_trend_prompt_keeps_system_prompt_v1_and_adds_exactly_one_sentence(monkeypatch):
    monkeypatch.delenv("PROMPT_VERSION", raising=False)
    monkeypatch.delenv("LLM_MODEL_PATH", raising=False)
    hits = [SearchHit("1990_x", "text", {"year": 1990, "source_file": "a.pdf"}, 1.0)]
    plain = build_cited_prompt("How did it change?", hits)
    trend = fa.trend_prompt("How did it change?", hits)
    assert trend.startswith(SYSTEM_PROMPT) and system_prompt("v1") == SYSTEM_PROMPT
    assert trend.endswith("\n\nAnswer:") and fa.TREND_SENTENCE in trend
    assert trend.replace(" " + fa.TREND_SENTENCE, "") == plain
    assert len(re.findall(r"[.!?](?:\s|$)", fa.TREND_SENTENCE)) == 1
