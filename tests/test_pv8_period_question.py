"""pv8 SPECIAL-03: every per-period call gets a well-formed question, never a headless fragment."""
import re

import pytest

from src.generation.compare import comparison_periods, period_question

# (query, {year: exact expected question without the "(focus: Y)" note})
CASES = {
    "hv6a_27": ("How did Buffett's assessment of the textile business change between his 1977 and 1979 letters?",
                {1977: "How did Buffett's assessment of the textile business change in his 1977 letters?",
                 1979: "How did Buffett's assessment of the textile business change in his 1979 letters?"}),
    "hv6a_29": ("How did the Berkadia guarantee for FINOVA look in Buffett's 2001 letter compared with his 2003 letter?",
                {2001: "How did the Berkadia guarantee for FINOVA look in Buffett's 2001 letter?",
                 2003: "How did the Berkadia guarantee for FINOVA look in Buffett's 2003 letter?"}),
    "hv6a_31": ("How did the size of Berkshire's federal income tax return and its number of state filings differ between the 2015 and 2017 letters?",
                {2015: "How did the size of Berkshire's federal income tax return and its number of state filings differ in the 2015 letters?",
                 2017: "How did the size of Berkshire's federal income tax return and its number of state filings differ in the 2017 letters?"}),
    "hv6b_39": ("How much cash equivalents did Berkshire hold at yearend in 2004 compared with 2010, and how did Buffett feel about it each time?",
                {2004: "How much cash equivalents did Berkshire hold at yearend in 2004, and how did Buffett feel about it each time?",
                 2010: "How much cash equivalents did Berkshire hold at yearend in 2010, and how did Buffett feel about it each time?"}),
    "hv6b_31": ("How did GEICO's position compare between the 2008 and 2016 letters, including market share in 2008 and how its scale was described in 2016?",
                {2008: "How did GEICO's position compare in the 2008 letters, including market share in 2008?",
                 2016: "How did GEICO's position compare in the 2016 letters, and how its scale was described in 2016?"}),
    "hv6b_46": ("Compare the size of Berkshire's home-office operation in 2006 and 2014: how many people worked there and how large was the federal tax return each year?",
                {2006: "The size of Berkshire's home-office operation in 2006: how many people worked there and how large was the federal tax return each year?",
                 2014: "The size of Berkshire's home-office operation in 2014: how many people worked there and how large was the federal tax return each year?"}),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_period_question_is_the_whole_ask_for_that_period(name):
    query, expected = CASES[name]
    periods = comparison_periods(query)
    assert len(periods) == 2
    for period in periods:
        year = period["year"]
        assert period_question(query, period, periods) == f"{expected[year]} (focus: {year})"


def test_old_headless_fragments_are_gone():
    # dpl_31-style: the whole ask used to be cut to "How did Buffett explain in his 1987 (focus: 1987)"
    query = "How did Buffett explain in his 1987 letter and again in his 1990 letter that he should do better in falling markets than in rising ones?"
    periods = comparison_periods(query)
    assert len(periods) == 2
    for period in periods:
        question = period_question(query, period, periods)
        stem = question.rsplit(" (focus:", 1)[0]
        assert "falling markets" in stem and stem.endswith("?")
        assert len(re.findall(r"\b(?:19|20)\d{2}\b", stem)) == 1


def test_no_dangling_connectors_or_orphan_compare():
    for query, _expected in CASES.values():
        periods = comparison_periods(query)
        for period in periods:
            stem = period_question(query, period, periods).rsplit(" (focus:", 1)[0]
            assert not re.search(r"\b(?:compared|between|versus|and|with|to)\s*[,?]", stem)
            assert " compared " not in stem and " between " not in stem


def test_stub_sub_ask_falls_back_to_the_whole_ask():
    query = "What do the 1995, 1996 and 1998 rows of the table in the 2014 letter show for the stock price gain of Berkshire shares compared with the S&P 500?"
    periods = comparison_periods(query)
    stem = period_question(query, periods[0], periods).rsplit(" (focus:", 1)[0]
    assert "stock price gain" in stem and "table" in stem
