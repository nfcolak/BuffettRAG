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
    # ---- expansion batch (g13-g50), validated against chunks_v2 on 2026-07-06 ----
    GoldQuery(
        qid="g13",
        query="What is insurance float and why does Buffett love it?",
        target_years=[1995, 1996, 1997, 2002, 2009, 2013],
        must_contain_any=["float"],
        answer_keywords=["float"],
    ),
    GoldQuery(
        qid="g14",
        query="What did Buffett write about Ajit Jain and reinsurance?",
        target_years=[1996, 2001, 2009, 2011, 2014],
        must_contain_any=["Ajit"],
        answer_keywords=["Ajit Jain", "reinsurance"],
    ),
    GoldQuery(
        qid="g15",
        query="What did Buffett say about Rose Blumkin and Nebraska Furniture Mart?",
        target_years=[1983, 1984, 2013],
        must_contain_any=["Blumkin", "Furniture Mart", "Mrs. B"],
        answer_keywords=["Blumkin", "Nebraska Furniture Mart"],
    ),
    GoldQuery(
        qid="g16",
        query="Buffett on the Coca-Cola investment",
        target_years=[1988, 1989, 1996, 2022],
        must_contain_any=["Coca-Cola", "Coke"],
        answer_keywords=["Coca-Cola"],
    ),
    GoldQuery(
        qid="g17",
        query="What happened with the Salomon Brothers crisis?",
        target_years=[1987, 1991, 1992],
        must_contain_any=["Salomon"],
        answer_keywords=["Salomon"],
    ),
    GoldQuery(
        qid="g18",
        query="Why did Buffett call the US Air investment a mistake?",
        target_years=[1989, 1994, 1996],
        must_contain_any=["USAir", "US Air"],
        answer_keywords=["USAir"],
    ),
    GoldQuery(
        qid="g19",
        query="What did Buffett admit about the Dexter Shoe acquisition?",
        target_years=[1993, 2007, 2014],
        must_contain_any=["Dexter"],
        answer_keywords=["Dexter"],
    ),
    GoldQuery(
        qid="g20",
        query="What did Buffett write about the General Re acquisition?",
        target_years=[1998, 1999, 2001],
        must_contain_any=["General Re", "Gen Re"],
        answer_keywords=["General Re"],
    ),
    GoldQuery(
        qid="g21",
        query="Buffett on MidAmerican Energy and the utility business",
        target_years=[1999, 2000, 2002, 2006],
        must_contain_any=["MidAmerican", "utility"],
        answer_keywords=["MidAmerican"],
    ),
    GoldQuery(
        qid="g22",
        query="What did Buffett write about ISCAR and its Israeli founders?",
        target_years=[2006, 2007, 2011, 2012],
        must_contain_any=["ISCAR", "Iscar", "Wertheimer"],
        answer_keywords=["ISCAR", "Israel"],
    ),
    GoldQuery(
        qid="g23",
        query="Buffett on the Heinz and Kraft investments",
        target_years=[2013, 2015, 2019],
        must_contain_any=["Heinz", "Kraft"],
        answer_keywords=["Heinz"],
    ),
    GoldQuery(
        qid="g24",
        query="What did Buffett write about Apple as an investment?",
        target_years=[2016, 2017, 2018, 2020, 2021, 2022, 2023],
        must_contain_any=["Apple"],
        answer_keywords=["Apple"],
    ),
    GoldQuery(
        qid="g25",
        query="Why did Buffett hire Todd Combs and Ted Weschler?",
        target_years=[2010, 2011, 2012],
        must_contain_any=["Todd Combs", "Ted Weschler", "Combs", "Weschler"],
        answer_keywords=["Combs", "Weschler"],
    ),
    GoldQuery(
        qid="g26",
        query="What is Buffett's argument against gold as an investment?",
        target_years=[2011, 2012],
        must_contain_any=["gold"],
        answer_keywords=["gold"],
    ),
    GoldQuery(
        qid="g27",
        query="What did Buffett say about the airline business?",
        target_years=[1989, 1996, 2007, 2016, 2020],
        must_contain_any=["airline"],
        answer_keywords=["airline"],
    ),
    GoldQuery(
        qid="g28",
        query="Where does Buffett say his favorite holding period is forever?",
        target_years=[1988, 1990],
        must_contain_any=["favorite holding period", "forever"],
        answer_keywords=["forever"],
    ),
    GoldQuery(
        qid="g29",
        query="What is the context of being fearful when others are greedy?",
        target_years=[1986, 2004],
        must_contain_any=["fearful", "greedy"],
        answer_keywords=["fearful", "greedy"],
    ),
    GoldQuery(
        qid="g30",
        query="What are look-through earnings?",
        target_years=[1990, 1991, 1992],
        must_contain_any=["look-through", "lookthrough", "look through"],
        answer_keywords=["look-through"],
    ),
    GoldQuery(
        qid="g31",
        query="Who is Mr. Market in Buffett's letters?",
        target_years=[1987, 1997],
        must_contain_any=["Mr. Market"],
        answer_keywords=["Mr. Market"],
    ),
    GoldQuery(
        qid="g32",
        query="What does Buffett mean by an economic moat?",
        target_years=[1995, 2000, 2005, 2007],
        must_contain_any=["moat"],
        answer_keywords=["moat"],
    ),
    GoldQuery(
        qid="g33",
        query="What did Buffett write about the newspaper business?",
        target_years=[1984, 1990, 2006, 2012],
        must_contain_any=["newspaper"],
        answer_keywords=["newspaper"],
    ),
    GoldQuery(
        qid="g34",
        query="Buffett on Clayton Homes and manufactured housing",
        target_years=[2003, 2008, 2009, 2010, 2011],
        must_contain_any=["Clayton"],
        answer_keywords=["Clayton"],
    ),
    GoldQuery(
        qid="g35",
        query="What did Buffett say about his bet with Protege Partners on index funds versus hedge funds?",
        target_years=[2016, 2017],
        must_contain_any=["Protégé", "Protege", "wager", "hedge fund"],
        answer_keywords=["index fund", "hedge fund"],
    ),
    GoldQuery(
        qid="g36",
        query="Buffett reflecting on 50 years of Berkshire in the anniversary letter",
        target_years=[2014],
        must_contain_any=["50 years", "fifty years", "next 50"],
        answer_keywords=["50"],
    ),
    GoldQuery(
        qid="g37",
        query="What did Buffett write about Charlie Munger's contribution to Berkshire?",
        target_years=[2014, 2015, 2023],
        must_contain_any=["Charlie", "Munger"],
        answer_keywords=["Munger"],
    ),
    GoldQuery(
        qid="g38",
        query="What is Buffett's criticism of EBITDA as a metric?",
        target_years=[2000, 2002, 2013],
        must_contain_any=["EBITDA"],
        answer_keywords=["EBITDA"],
    ),
    GoldQuery(
        qid="g39",
        query="Buffett on expensing stock options",
        target_years=[1998, 2002, 2003, 2004],
        must_contain_any=["option"],
        answer_keywords=["option"],
    ),
    GoldQuery(
        qid="g40",
        query="Why did Berkshire close its textile operation?",
        target_years=[1978, 1985],
        must_contain_any=["textile"],
        answer_keywords=["textile"],
    ),
    GoldQuery(
        qid="g41",
        query="What is cigar-butt investing according to Buffett?",
        target_years=[1989],
        must_contain_any=["cigar"],
        answer_keywords=["cigar"],
    ),
    GoldQuery(
        qid="g42",
        query="Buffett on super-cat insurance and catastrophe risk",
        target_years=[1992, 1993, 1994, 1995, 1996, 2005],
        must_contain_any=["super-cat", "catastrophe", "mega-cat"],
        answer_keywords=["catastrophe"],
    ),
    GoldQuery(
        qid="g43",
        query="How did September 11 affect Berkshire's insurance business?",
        target_years=[2001],
        must_contain_any=["September 11", "terrorism", "terrorist"],
        answer_keywords=["September 11", "terrorism"],
    ),
    GoldQuery(
        qid="g44",
        query="Why does Berkshire not pay dividends to its shareholders?",
        target_years=[1984, 2012, 2013],
        must_contain_any=["dividend"],
        answer_keywords=["dividend"],
    ),
    GoldQuery(
        qid="g45",
        query="What does Buffett think about issuing Berkshire shares to make acquisitions?",
        target_years=[1982, 1993, 1997],
        must_contain_any=["issuing shares", "share issuance", "issue shares", "stock for stock", "issued shares"],
        answer_keywords=["shares", "issuance"],
    ),
    GoldQuery(
        qid="g46",
        query="What did Buffett say about winding down the General Re derivatives book?",
        target_years=[2002, 2003, 2005, 2008],
        must_contain_any=["derivative"],
        answer_keywords=["derivative", "General Re"],
    ),
    GoldQuery(
        qid="g47",
        query="Buffett on the housing bubble and mortgage lending",
        target_years=[2007, 2008, 2009, 2010, 2011],
        must_contain_any=["housing", "mortgage"],
        answer_keywords=["housing", "mortgage"],
    ),
    GoldQuery(
        qid="g48",
        query="What did Buffett write about retained earnings and Edgar Lawrence Smith?",
        target_years=[2019],
        must_contain_any=["retained earnings", "Edgar"],
        answer_keywords=["retained earnings"],
    ),
    GoldQuery(
        qid="g49",
        query="What did Buffett write about the Precision Castparts acquisition?",
        target_years=[2015, 2016, 2020],
        must_contain_any=["Precision Castparts", "PCC"],
        answer_keywords=["Precision Castparts"],
    ),
    GoldQuery(
        qid="g50",
        query="What did Buffett write about Japanese trading companies?",
        target_years=[2020, 2023],
        must_contain_any=["Japan", "Japanese", "Itochu", "Mitsubishi"],
        answer_keywords=["Japan"],
    ),
]


def get_gold_queries() -> List[GoldQuery]:
    return list(GOLD_QUERIES)
