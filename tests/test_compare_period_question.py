"""Compare path: each period's model call gets the period-neutral question; a refused period is retried once."""
from src.generation import compare
from src.generation.compare import comparison_periods, generate_comparison_answer, period_question
from src.generation.prompt import REFUSAL_LINE
from src.storage import SearchHit

QUERY = "How did Buffett's view of insurance float in 2003 compare with 2008?"


def _periods():
    periods = comparison_periods(QUERY)
    assert len(periods) == 2
    return periods


def test_period_question_drops_the_other_period_and_the_comparison():
    periods = _periods()
    question = period_question(QUERY, periods[0], periods)
    assert "2003" in question and "2008" not in question
    assert "compare" not in question.lower() or "2008" not in question
    assert question.endswith("(focus: 2003)")


def _hits():
    text = "Insurance float is money we hold but do not own and it has been very valuable to us."
    return [SearchHit(f"{year}_p000{i}", f"{text} Year {year} detail number {i} about float and underwriting.",
                      {"year": year, "source_file": f"buffet_{year}.txt"}, 0.0)
            for year in (2003, 2008) for i in range(1, 6)]


class _LLM:
    provider_name = "stub"

    def __init__(self, refuse_first=False):
        self.prompts, self.refuse_first = [], refuse_first

    def generate(self, prompt, max_new_tokens=None):
        self.prompts.append(prompt)
        if self.refuse_first and len(self.prompts) == 1:
            return REFUSAL_LINE
        return "Insurance float is money we hold but do not own and it has been very valuable to us. [1]"


def test_each_period_call_asks_the_neutral_question():
    llm = _LLM()
    generate_comparison_answer(llm, QUERY, _hits(), max_new_tokens=300)
    assert len(llm.prompts) == 2
    assert "compare with" not in llm.prompts[0] and "compare with" not in llm.prompts[1]
    # passages are period-isolated, so a year in the prompt can only come from the question
    assert "2008" not in llm.prompts[0] and "2003" not in llm.prompts[1]


def test_refused_period_is_retried_once_with_top_three_passages():
    llm = _LLM(refuse_first=True)
    result = generate_comparison_answer(llm, QUERY, _hits(), max_new_tokens=300)
    assert len(llm.prompts) == 3
    retry = llm.prompts[1]
    assert retry.count("(year=") == 3
    assert "The retrieved passages for this period do not cover" not in result.answer
