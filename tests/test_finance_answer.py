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


# --- finance-dev-fix-1: plain lookups only, entity-safe rows, figure rows for the reader, superlatives ---
@pytest.mark.parametrize("query,plain", [
    ("What was Berkshire's book value per share in 1987?", True),
    ("How much float did Berkshire have in 1999?", True),
    ("Why did Berkshire's book value per share rise in 1987?", False),
    ("How did book value per share grow after 1987?", False),
    ("How much did Berkshire pay for USAir, and how did Buffett later judge it?", False),
    ("What did Berkshire pay for USAir in 1989? And what did he think of airlines?", False),
])
def test_plain_figure_lookup(store, query, plain):
    assert fa.plain_figure_lookup(plan_query(query, store)) is plain


def _rows_store(*rows, companies=None):
    base = {"unit": "usd_millions", "as_written": "x", "source_id": "1990_p0100"}
    return KnowledgeStore(numbers=[dict(base, **r) for r in rows], companies=companies)


def test_a_metric_named_only_in_part_and_tied_is_not_a_plain_lookup_but_both_groups_reach_the_reader():
    store = _rows_store({"metric": "insurance_float", "year": 1999, "value": 25298, "entity": "Berkshire"},
                        {"metric": "cost_of_float_pct", "year": 1999, "value": 5.8, "unit": "percent",
                         "entity": "Berkshire"})
    plan = plan_query("How much float did Berkshire have in 1999?", store)
    assert not fa.metric_named(plan, store)
    assert {r["metric"] for r in fa._match_numbers(plan, store)} == {"insurance_float", "cost_of_float_pct"}
    named = plan_query("What was Berkshire's insurance float in 1999?", store)
    assert fa.metric_named(named, store) and [r["metric"] for r in fa._match_numbers(named, store)] == ["insurance_float"]


def test_rows_of_another_entity_are_never_used():
    companies = {"companies": {"BNSF": {"aliases": ["BNSF", "Burlington Northern Santa Fe"]}}}
    store = _rows_store({"metric": "acquisition_price", "year": 2009, "value": 100, "unit": "usd", "entity": "BNSF"},
                        {"metric": "cash_and_equivalents", "year": 2009, "value": 30558, "entity": "Berkshire"},
                        companies=companies)
    plan = plan_query("What price did Berkshire pay for Burlington Northern Santa Fe in 2009, in cash and stock?", store)
    assert [r["entity"] for r in fa._match_numbers(plan, store)] == ["BNSF"]
    plain = plan_query("How much cash did Berkshire hold in 2009?", store)  # no other company: Berkshire's rows
    assert [r["entity"] for r in fa._match_numbers(plain, store)] == ["Berkshire"]


def test_figures_prompt_adds_the_rows_before_the_question_citing_the_served_passages(store, docs):
    plan = plan_query("Why did Berkshire's book value per share reach its 1987 level?", store)
    rows = fa.figure_rows(plan, store, docs)
    pinned = fa.figure_source_hits(rows, docs)
    assert [h.id for h in pinned] == ["1987_p0000"]
    other = SearchHit("1990_p0100", docs["1990_p0100"].text, dict(docs["1990_p0100"].metadata), 1.0)
    hits = [other, *pinned]
    prompt = fa.figures_prompt(build_cited_prompt(plan.query, hits), rows, hits)
    block = prompt[prompt.index(fa.FIGURES_BEGIN):prompt.index(fa.FIGURES_END)]
    assert '$2,477.47; as written: "$2,477.47" [2]' in block  # the row cites its passage's served number
    assert prompt.index(fa.FIGURES_END) < prompt.index("BEGIN USER QUESTION")
    assert fa.figures_prompt(build_cited_prompt(plan.query, [other]), rows, [other]) == build_cited_prompt(
        plan.query, [other])  # a row whose passage is not served is left out
    assert fa.figures_prompt("no question block", rows, hits) == "no question block"


def _stance_store():
    def ann(pid, target, polarity, quote="q"):
        return {"id": pid, "scope": "financial", "stance": [{"target": target, "polarity": polarity, "quote": quote}]}
    annotations = [ann("1980_p0001", "insurance", "positive", "insurance is a fine business"),
                   ann("1980_p0002", "insurance", "positive"), ann("1980_p0003", "insurance", "positive"),
                   ann("1990_p0001", "insurance", "positive", "we like insurance"),
                   ann("2000_p0001", "insurance", "positive", "insurance shines"),
                   ann("2000_p0002", "inflation", "positive")]  # a theme: never a sector target
    annotations += [ann(f"2010_p{n:04d}", "railroads", "positive") for n in range(1, 7)]  # 6 rows, one letter
    annotations += [ann("2011_p0001", "railroads", "positive"), ann("1995_p0001", "USAir", "positive")]
    digest = {"topic": "insurance", "kind": "sector", "overview": "o",
              "timeline": [{"year": 1990, "quote": "from the digest", "source_id": "1990_p0001"},
                           {"year": 1995, "quote": "not a stance passage", "source_id": "1995_p0001"}]}
    taxonomy = {"sectors": {"insurance": "", "railroads": ""}, "themes": {"inflation": ""}}
    docs = {a["id"]: StoredDoc(id=a["id"], text=f"text {a['id']}", metadata={"year": int(a["id"][:4])})
            for a in annotations}
    return KnowledgeStore(annotations=annotations, digests=[digest], taxonomy=taxonomy), docs


def test_superlative_counts_letters_not_stance_rows():
    store, docs = _stance_store()
    plan = QueryPlan(query="q", kind="corpus_wide", superlative="sector:positive")
    ranking = fa.stance_letters(plan, store, docs)
    assert [(t, years) for t, years, _ in ranking] == [("insurance", [1980, 1990, 2000]), ("railroads", [2010, 2011])]
    result = fa.superlative_answer(plan, store, docs)
    assert result.route == "superlative"
    assert "1. Insurance: 3 letters (1980–2000)" in result.answer and "2. Railroads: 2 letters (2010–2011)" in result.answer
    assert '1990: "from the digest" [' in result.answer and "not a stance passage" not in result.answer
    assert '"insurance is a fine business"' in result.answer  # letters the timeline misses: the stance quote
    assert _resolves(result) and result.detail["cited"] == len(result.citations)
    cited = {pid for c in result.citations for pid in c["passage_ids"]}
    assert cited and cited <= {h.id for h in result.hits[:result.detail["cited"]]}


def test_superlative_company_targets_exclude_sectors_and_a_year_range_limits_the_letters():
    store, docs = _stance_store()
    company = fa.stance_letters(QueryPlan(query="q", superlative="company:positive"), store, docs)
    assert [t for t, _, _ in company] == ["USAir"]
    ranged = fa.stance_letters(QueryPlan(query="q", superlative="sector:positive", year_range=(2005, 2015)), store, docs)
    assert [t for t, _, _ in ranged] == ["railroads"]
    assert fa.superlative_answer(QueryPlan(query="q", superlative="sector:negative"), store, docs) is None


def test_drop_personal_sentences_removes_family_story_cited_to_personal_passage():
    from src.generation.finance_answer import drop_personal_sentences
    from src.storage import SearchHit

    class Store:
        def has_personal(self, pid):
            return pid == "p_personal"

        def is_financial(self, pid):
            return True

    hits = [SearchHit("p_personal", "", {}, 0.0), SearchHit("p_fin", "", {}, 0.0)]
    answer = ("With his first child due soon, he used $500 of his savings to found the company. [1] "
              "First-year sales totaled $112,000. [1] Sales rose 21% above the previous high. [2]")
    out = drop_personal_sentences(answer, hits, Store())
    assert "first child" not in out
    assert "First-year sales totaled $112,000. [1]" in out and "Sales rose 21%" in out
    # a personal cue cited to a purely financial passage stays
    kept = drop_personal_sentences("The company paid its founder's son $1 million in salary. [2]", hits, Store())
    assert "son" in kept
    # nothing financial left -> refusal line
    from src.generation.prompt import REFUSAL_LINE
    assert drop_personal_sentences("He told a joke about his age. [1]", hits, Store()) == REFUSAL_LINE


def test_drop_mixed_personal_sentences_removes_story_without_business_figure():
    from src.generation.finance_answer import drop_mixed_personal_sentences
    from src.generation.prompt import REFUSAL_LINE
    from src.storage import SearchHit

    class Store:
        def has_personal(self, pid):
            return pid.startswith("p_personal")

        def is_financial(self, pid):
            return True

    hits = [SearchHit("p_personal_1", "", {}, 0.0), SearchHit("p_fin", "", {}, 0.0),
            SearchHit("p_personal_2", "", {}, 0.0)]
    answer = ("In 1971, Paul was working as a purchasing agent for General Dynamics when the roof fell in. [1] "
              "The company set itself up to distribute small electronic components, and first-year sales totaled "
              "$112,000. [1] Earnings have increased 673%. [3]\nHe moved to Omaha. [2]")
    out = drop_mixed_personal_sentences(answer, hits, Store())
    assert "General Dynamics" not in out                       # personal source, no business figure (a year is not one)
    assert "first-year sales totaled $112,000. [1]" in out     # personal source, but a business figure
    assert "Earnings have increased 673%. [3]" in out
    assert "He moved to Omaha. [2]" in out                     # cited to a financial passage: kept
    assert drop_mixed_personal_sentences("He was born in Latvia. [1]", hits, Store()) == REFUSAL_LINE
    assert drop_mixed_personal_sentences(REFUSAL_LINE, hits, Store()) == REFUSAL_LINE


def test_finance_mixed_filter_only_for_mixed_reader_routes_with_the_flag(monkeypatch):
    from src.query.plan import FINANCIAL, MIXED, QueryPlan
    from src.services import ask_flow
    from src.storage import SearchHit

    class Store:
        def has_personal(self, pid):
            return pid == "p_personal"

        def is_financial(self, pid):
            return True

    monkeypatch.setattr(ask_flow, "_knowledge_store", lambda: Store())
    hits = [SearchHit("p_personal", "story", {"year": 2021}, 0.0), SearchHit("p_fin", "figures", {"year": 2021}, 0.0)]
    answer = "Paul was working for General Dynamics when the roof fell in. [1] Sales rose 21%. [2]"
    mixed = ask_flow.FinanceRoute("reader", QueryPlan(query="q", scope=MIXED), "q", hits=hits)
    financial = ask_flow.FinanceRoute("reader", QueryPlan(query="q", scope=FINANCIAL), "q", hits=hits)
    monkeypatch.delenv("FINANCE_MIXED_STRICT", raising=False)
    assert ask_flow.finance_mixed_filter(mixed, answer, ["c"]) == (answer, ["c"])      # flag off: unchanged
    monkeypatch.setenv("FINANCE_MIXED_STRICT", "1")
    assert ask_flow.finance_mixed_filter(financial, answer, ["c"]) == (answer, ["c"])  # not a mixed plan
    assert ask_flow.finance_mixed_filter(None, answer, ["c"]) == (answer, ["c"])       # legacy route
    out, cites = ask_flow.finance_mixed_filter(mixed, answer, ["c"])
    assert out == "Sales rose 21%. [2]"
    assert [c["passage_ids"] for c in cites] == [["p_fin"]]
