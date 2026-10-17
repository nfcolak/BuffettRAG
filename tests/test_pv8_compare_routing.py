"""pv8 SPECIAL-04: COMPARE_ROUTING_V2 (default off) fixes missed comparisons and wrongly routed fact questions."""
import pytest

from src.generation.compare import comparison_periods, period_question
from src.retrieval.retriever import detect_temporal_comparison

# true comparisons the v1 rules miss (v6 hv6a_32, hv6b_25 verbatim; others are the unit's shapes)
MISSED = {
    "hv6a_32": "How did Buffett's statement of Berkshire's basic goal in his 2023 letter compare with his 2031 letter?",
    "hv6b_25": "Compare what Berkshire paid in income tax in 2024 and in 2032, and how the dividend history frames it.",
    "no_cue": "What did Buffett say about the Buffalo News in 1987 and in 1995?",
    "in_a_and_in_b": "In 1990 and in 1996, what did Buffett say about GEICO's float?",
    "year_later": "What did Buffett say about GEICO's float in 1996 and a year later in 1997?",
}
# plain fact questions with one event year / table-row years that v1 routes into up to 6 calls
WRONG = {
    "hv6b_05": "Using the annual comparison printed in the 2012 letter, what happened to Berkshire's book value versus the index, dividends included, in each of 2001, 2002 and 2003?",
    "hv6b_17": "In 2012, how had Buffett's Berkshire share count and the book value of his stake changed since 2005, how much did bolt-on purchases cost, and how much free money did insurance provide?",
    "hv6b_27": "What do the 1995, 1996 and 1998 rows of the table in the 2014 letter show for the stock price gain of Berkshire shares compared with the S&P 500?",
    "rows": "What were the 1995, 1996 and 1998 figures for the stock price gain shown in the 2014 letter table, compared with the S&P 500?",
}
STILL_ROUTED = [
    "How did Buffett's assessment of the textile business change between his 1977 and 1979 letters?",
    "Compare Gen Re Securities in the 2002 and 2004 letters: what remained of the business in 2002 versus 2004?",
    "How much cash equivalents did Berkshire hold at yearend in 2004 compared with 2010, and how did Buffett feel about it each time?",
]


def _years(periods):
    return [p["year"] for p in periods or []]


def test_default_is_off_and_v1_behaviour_is_unchanged(monkeypatch):
    monkeypatch.delenv("COMPARE_ROUTING_V2", raising=False)
    assert detect_temporal_comparison(MISSED["hv6a_32"]) is None
    assert detect_temporal_comparison(MISSED["hv6b_25"]) is None
    assert len(detect_temporal_comparison(WRONG["hv6b_05"])) == 4


@pytest.mark.parametrize("name", sorted(MISSED))
def test_v2_routes_the_missed_comparisons(monkeypatch, name):
    monkeypatch.setenv("COMPARE_ROUTING_V2", "1")
    periods = detect_temporal_comparison(MISSED[name])
    assert periods and len(periods) == 2


def test_v2_out_of_corpus_year_is_an_unanswerable_half(monkeypatch):
    monkeypatch.setenv("COMPARE_ROUTING_V2", "1")
    assert _years(detect_temporal_comparison(MISSED["hv6a_32"])) == [2023, 2031]
    assert _years(detect_temporal_comparison(MISSED["hv6b_25"])) == [2024, 2032]


@pytest.mark.parametrize("name", sorted(WRONG))
def test_v2_does_not_route_fact_questions(monkeypatch, name):
    monkeypatch.setenv("COMPARE_ROUTING_V2", "1")
    assert detect_temporal_comparison(WRONG[name]) is None


@pytest.mark.parametrize("query", STILL_ROUTED)
def test_v2_keeps_true_comparisons_and_period_count(monkeypatch, query):
    monkeypatch.delenv("COMPARE_ROUTING_V2", raising=False)
    v1 = detect_temporal_comparison(query)
    monkeypatch.setenv("COMPARE_ROUTING_V2", "1")
    assert detect_temporal_comparison(query) == v1


def test_out_of_corpus_period_gets_a_whole_question(monkeypatch):
    monkeypatch.setenv("COMPARE_ROUTING_V2", "1")
    query = MISSED["hv6a_32"]
    periods = comparison_periods(query)
    question = period_question(query, periods[1], periods)
    assert question.startswith("How did Buffett's statement of Berkshire's basic goal") and question.endswith("(focus: 2031)")
    assert "2023" not in question


def test_out_of_corpus_half_is_reported_not_covered_and_costs_no_call(monkeypatch):
    from src.generation.compare import generate_comparison_answer
    from src.storage import SearchHit
    monkeypatch.setenv("COMPARE_ROUTING_V2", "1")

    class LLM:
        provider_name = "stub"
        prompts = []

        def generate(self, prompt, max_new_tokens=None):
            self.prompts.append(prompt)
            return "Our basic goal is to maximize the average annual rate of gain in intrinsic business value. [1]"

    text = "Our basic goal is to maximize the average annual rate of gain in intrinsic business value per share."
    hits = [SearchHit(f"2023_p{i}", text, {"year": 2023, "source_file": "buffet_2023.txt"}, 0.0) for i in range(3)]
    llm = LLM()
    result = generate_comparison_answer(llm, MISSED["hv6a_32"], hits, max_new_tokens=300)
    assert len(llm.prompts) == 1
    assert "In 2031: The retrieved passages for this period do not cover this question." in result.answer
    assert result.answer.startswith("In 2023:")
