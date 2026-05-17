"""Gold evaluation set.

Each entry is a query with expected provenance. Because we don't have
human-annotated chunk IDs, the gold standard is expressed in terms the
ingestion pipeline guarantees: target year(s), and a small list of
required keywords / phrases that should appear in any genuinely relevant
passage.

This is honest about its limits: it's a structural gold set, not a
human-annotated one. Retrieval metrics are computed against it, and the
limitations are documented in the eval report.

Adding new queries: keep `target_years` tight (the actual year(s) the
relevant content lives in) and `must_contain_any` to specific terms
that wouldn't match unrelated passages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class GoldQuery:
    qid: str
    query: str
    target_years: Sequence[int]                # passages from these years are "correct"
    must_contain_any: Sequence[str] = field(default_factory=list)
    # Optional: phrases that, if present in the answer, indicate correctness
    answer_keywords: Sequence[str] = field(default_factory=list)
    # Optional decade hint for cross-decade queries
    cross_decade: bool = False


GOLD_QUERIES: List[GoldQuery] = [
    GoldQuery(
        qid="g01",
        query="How did Buffett react to the 2008 financial crisis in his shareholder letter?",
        target_years=[2008, 2009],
        must_contain_any=["financial", "crisis", "Lehman", "panic", "2008"],
        answer_keywords=["2008", "financial crisis"],
    ),
    GoldQuery(
        qid="g02",
        query="What did Buffett say about derivatives being weapons of mass destruction?",
        target_years=[2002, 2003],
        must_contain_any=["derivative", "weapons of mass destruction", "mass destruction"],
        answer_keywords=["weapons of mass destruction", "derivatives"],
    ),
    GoldQuery(
        qid="g03",
        query="What is Buffett's view on share repurchases and buybacks?",
        target_years=[2011, 2016, 2020, 2022],
        must_contain_any=["repurchas", "buyback", "share"],
        answer_keywords=["repurchase", "buyback"],
    ),
    GoldQuery(
        qid="g04",
        query="What did Buffett write about See's Candies as an investment?",
        target_years=[1983, 1989, 1991, 2007],
        must_contain_any=["See's", "candy", "candies"],
        answer_keywords=["See's"],
    ),
    GoldQuery(
        qid="g05",
        query="What does Buffett look for when acquiring a company?",
        target_years=[1982, 1986, 1990, 1994, 2014],
        must_contain_any=["acquisition", "purchase", "criteria", "businesses"],
        answer_keywords=["acquisition", "criteria"],
    ),
    GoldQuery(
        qid="g06",
        query="Buffett's discussion of inflation in the late 1970s and early 1980s",
        target_years=[1977, 1978, 1979, 1980, 1981, 1982, 1983],
        must_contain_any=["inflation", "purchasing power", "inflationary"],
        answer_keywords=["inflation"],
        cross_decade=True,
    ),
    GoldQuery(
        qid="g07",
        query="What did Buffett say about GEICO and the insurance float?",
        target_years=[1995, 1996, 2007, 2009, 2014],
        must_contain_any=["GEICO", "float", "insurance"],
        answer_keywords=["GEICO", "float"],
    ),
    GoldQuery(
        qid="g08",
        query="How has Buffett's view on technology stocks changed from the 1990s to the 2020s?",
        target_years=[1999, 2000, 2016, 2017, 2018, 2020, 2021, 2022],
        must_contain_any=["technology", "tech", "Apple", "internet"],
        answer_keywords=["Apple", "technology"],
        cross_decade=True,
    ),
    GoldQuery(
        qid="g09",
        query="Buffett on succession planning at Berkshire",
        target_years=[2014, 2015, 2018, 2021, 2022, 2023],
        must_contain_any=["succession", "successor", "Greg Abel", "Ajit"],
        answer_keywords=["succession"],
    ),
    GoldQuery(
        qid="g10",
        query="What does Buffett mean by intrinsic value and margin of safety?",
        target_years=[1983, 1989, 1992, 1996, 2014],
        must_contain_any=["intrinsic value", "margin of safety", "Mr. Market"],
        answer_keywords=["intrinsic value"],
    ),
    GoldQuery(
        qid="g11",
        query="Buffett's advice to individual investors about index funds",
        target_years=[1996, 2007, 2013, 2016],
        must_contain_any=["index fund", "S&P 500", "individual investor"],
        answer_keywords=["index fund"],
    ),
    GoldQuery(
        qid="g12",
        query="What did Buffett write about the BNSF railroad acquisition?",
        target_years=[2009, 2010, 2011],
        must_contain_any=["BNSF", "Burlington Northern", "railroad"],
        answer_keywords=["BNSF", "Burlington Northern"],
    ),
]


def get_gold_queries() -> List[GoldQuery]:
    return list(GOLD_QUERIES)
