"""Deterministic query planner (finance-query-pipeline, group 2).

`plan_query(query, store, scope_vote=None)` turns a question into a `QueryPlan`: scope (financial, personal,
mixed), kind, year range, events, sectors, companies and themes. No model, no retrieval except the optional
`scope_vote` callable the caller passes in. Callers gate it behind `FINANCE_PIPELINE=1`; this module does not
read the flag.

Years are a boost range by default; `year_hard` is True only for "in the YYYY letter" questions. `year_spans` keeps
each year mention of the question apart (two single years are two spans, not one range), so retrieval can tell a
narrow question from one that spans several letters.

Scope (finance-dev-fix-1): a clause about a business's sales, earnings, growth or price is financial even when it
names a founder, a family or a meeting; jokes, anecdotes, longevity, attendance, ballgames and shopping at the
annual meeting are personal; in a question with a personal clause, a clause with no financial cue at all is a
personal follow-up ("... and what did he credit for their longevity?"). The scope vote cannot make a question
about a business's figures personal: it only drops the clauses that carry no financial cue.

Premise check (finance-answer-quality, group 2): with `FINANCE_PREMISE_CHECK=1` (or `premise_check=True`) a question that
names a period the topic does not belong to gets a one-sentence `premise_note`: the event's real years miss the asked
period, or the knowledge store has no passage on the topic in the asked years but has some in other years. The note names
the letter years that do carry the topic. Off by default: `premise_note` stays None and the plan is unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from config import finance_setting
from src.generation.decompose import split_question
from src.knowledge import KnowledgeStore

FINANCIAL, PERSONAL, MIXED = "financial", "personal", "mixed"
SPECIFIC, PERIOD, TREND, COMPARISON, NUMERIC, CORPUS_WIDE = (
    "specific", "period", "trend", "comparison", "numeric", "corpus_wide")

SCOPE_VOTE_PERSONAL_SHARE = 0.6  # at this share of non-financial / has_personal passages the question is personal
FIRST_LETTER_YEAR = 1977
LAST_LETTER_YEAR = 2024

# --- user event phrasings the letters never use (events.json keeps only phrases found in the letters) ---
# phrase -> event id (used only if the store knows it), fallback letter years, topic terms for the boost.
# An entry without a store event also carries what the premise check needs: how the note names the event, the real
# years of the event as inclusive spans, and the words a passage summary uses when a letter does mention it.
_OIL_SHOCKS: Dict[str, Any] = {
    "event": None, "years": (1973, 1982), "terms": ("oil", "energy", "OPEC", "gasoline"),
    "label": "the oil shocks", "plural": True, "real_years": ((1973, 1974), (1979, 1979)),
    "letter_phrases": ("oil shock", "oil crisis", "oil embargo", "energy crisis", "oil price", "price of oil",
                       "foreign oil", "energy dependence"),
}
USER_EVENT_ALIASES: Dict[str, Dict[str, Any]] = {
    "2008 crisis": {"event": "financial_crisis_2008", "years": (2007, 2010)},
    "2008 financial crisis": {"event": "financial_crisis_2008", "years": (2007, 2010)},
    "2008 recession": {"event": "financial_crisis_2008", "years": (2007, 2010)},
    "global financial crisis": {"event": "financial_crisis_2008", "years": (2007, 2010)},
    "great recession": {"event": "financial_crisis_2008", "years": (2007, 2010)},
    "credit crunch": {"event": "financial_crisis_2008", "years": (2007, 2010)},
    "subprime": {"event": "financial_crisis_2008", "years": (2007, 2010)},
    "lehman": {"event": "financial_crisis_2008", "years": (2007, 2010)},
    "9/11": {"event": "sept11_2001", "years": (2001, 2002)},
    "sept 11": {"event": "sept11_2001", "years": (2001, 2002)},
    "sept. 11": {"event": "sept11_2001", "years": (2001, 2002)},
    "2001 terrorist attacks": {"event": "sept11_2001", "years": (2001, 2002)},
    "black monday": {"event": "crash_1987", "years": (1987, 1987)},
    "1987 crash": {"event": "crash_1987", "years": (1987, 1987)},
    "1987 stock market crash": {"event": "crash_1987", "years": (1987, 1987)},
    "stock market crash of 1987": {"event": "crash_1987", "years": (1987, 1987)},
    "dot-com": {"event": "dotcom_bubble", "years": (1999, 2002)},
    "dot com": {"event": "dotcom_bubble", "years": (1999, 2002)},
    "dotcom": {"event": "dotcom_bubble", "years": (1999, 2002)},
    "tech bubble": {"event": "dotcom_bubble", "years": (1999, 2002)},
    "oil crisis": _OIL_SHOCKS,
    "oil shock": _OIL_SHOCKS,
    "oil embargo": _OIL_SHOCKS,
    "energy crisis": _OIL_SHOCKS,
    "volcker": {"event": "high_inflation_late70s_early80s", "years": (1979, 1984),
                "terms": ("Volcker", "interest rates", "inflation")},
}

# How the premise note names the events of events.json (an id missing here is named "the <id with spaces>").
EVENT_NAMES: Dict[str, str] = {
    "bear_market_1973_74": "the 1973-74 bear market",
    "high_inflation_late70s_early80s": "the high inflation of the late 1970s and early 1980s",
    "crash_1987": "the October 1987 market crash",
    "banking_real_estate_1990": "the 1989-91 bank and real-estate troubles",
    "salomon_1991": "the 1991 Salomon scandal",
    "hurricane_andrew_1992": "Hurricane Andrew",
    "dotcom_bubble": "the dot-com bubble",
    "ltcm_1998": "the 1998 collapse of Long-Term Capital Management",
    "accounting_scandals_2002": "the 2001-02 accounting scandals",
    "sept11_2001": "the September 11 attacks",
    "katrina_2005": "Hurricane Katrina",
    "financial_crisis_2008": "the 2008 financial crisis",
    "tax_act_2017": "the 2017 tax act",
}
_PLURAL_EVENT_NAMES = {"accounting_scandals_2002", "sept11_2001"}

# --- entities ---
SECTOR_TERMS: Dict[str, Tuple[str, ...]] = {
    "insurance": ("insurance", "insurer", "insurers", "reinsurance", "property-casualty", "geico", "underwriting"),
    "banking": ("bank", "banks", "banking", "thrift", "thrifts", "savings and loan"),
    "financial_services_other": ("credit card", "credit cards", "consumer finance", "investment bank",
                                 "investment banks", "asset manager", "rating agency"),
    "railroads": ("railroad", "railroads", "railway", "bnsf"),
    "utilities_energy": ("utility", "utilities", "electric utility", "pipeline", "pipelines", "renewable",
                         "renewables", "power company"),
    "oil_gas": ("oil", "oil and gas", "natural gas", "petroleum", "refiner", "refiners", "opec", "gasoline"),
    "airlines_aviation": ("airline", "airlines", "aviation", "aircraft", "airplane"),
    "newspapers_publishing": ("newspaper", "newspapers", "publishing", "publisher"),
    "tv_broadcasting_media": ("television", "broadcasting", "broadcaster", "media", "advertising"),
    "retail": ("retail", "retailer", "retailers", "department store", "department stores"),
    "consumer_brands": ("consumer brand", "consumer brands", "consumer goods", "branded products"),
    "manufacturing_industrial": ("manufacturing", "manufacturer", "manufacturers", "industrial"),
    "housing_construction": ("housing", "home building", "homebuilding", "manufactured housing", "construction",
                             "building products"),
    "textiles_apparel_shoes": ("textile", "textiles", "apparel", "shoe", "shoes"),
    "technology": ("technology", "tech sector", "tech stocks", "software", "semiconductor", "semiconductors",
                   "telecom"),
    "real_estate": ("real estate", "reit", "reits", "commercial property"),
    "commodities_gold": ("gold", "silver", "commodity", "commodities", "farmland"),
    "japan_trading_houses": ("trading house", "trading houses"),
    "transportation_other": ("trucking", "logistics", "car dealership", "car dealerships"),
    "healthcare_pharma": ("healthcare", "pharmaceutical", "pharmaceuticals", "pharma", "drug maker", "drugmakers"),
    "steel_metals_mining": ("steel", "metals", "mining", "aluminum"),
}
_SECTOR_STOP = {"general", "other", "companies", "company", "industry", "sector", "business", "businesses", "and its"}

THEME_TERMS: Dict[str, Tuple[str, ...]] = {
    "inflation": ("inflation", "purchasing power"),
    "interest_rates": ("interest rate", "interest rates", "the fed", "federal reserve", "monetary policy"),
    "bonds_fixed_income": ("bond", "bonds", "treasuries", "treasury bills", "junk bond", "junk bonds"),
    "preferred_convertible_investments": ("preferred stock", "preferred stocks", "convertible", "convertibles"),
    "arbitrage_workouts": ("arbitrage",),
    "market_valuation": ("market valuation", "overvalued", "overvaluation", "market bubble", "stock prices"),
    "market_timing_forecasts": ("market timing", "forecast", "forecasts", "forecasting", "predict the market"),
    "cash_liquidity": ("cash holdings", "liquidity", "cash pile"),
    "debt_leverage": ("leverage", "borrowing", "margin debt"),
    "derivatives": ("derivative", "derivatives", "credit default", "equity put", "equity puts"),
    "buybacks": ("buyback", "buybacks", "repurchase", "repurchases", "share repurchase"),
    "dividends_capital_allocation": ("dividend", "dividends", "capital allocation", "retained earnings"),
    "acquisitions_deals": ("acquisition", "acquisitions", "acquire businesses", "takeover"),
    "intrinsic_value_book_value": ("intrinsic value", "book value"),
    "accounting_reporting": ("accounting", "gaap", "goodwill", "look-through earnings"),
    "pensions_retirement_accounting": ("pension", "pensions"),
    "taxes": ("tax", "taxes", "taxation"),
    "float_underwriting": ("float", "underwriting", "combined ratio"),
    "catastrophe_risk": ("catastrophe", "catastrophes", "hurricane", "earthquake"),
    "management_governance": ("management", "governance", "executive compensation", "ceo pay"),
    "succession_culture": ("succession", "successor"),
    "shareholder_programs": ("shareholder-designated", "designated contributions"),
    "index_funds_fees": ("index fund", "index funds", "hedge fund fees", "hedge funds", "money managers"),
    "currency_trade_deficit": ("trade deficit", "currency", "currencies", "dollar"),
    "government_fiscal_policy": ("fiscal policy", "federal deficit", "national debt", "government debt"),
    "regulation_politics": ("regulation", "regulators", "politics"),
    "investing_principles": ("margin of safety", "circle of competence", "value investing", "mr. market"),
    "moats_competitive_advantage": ("moat", "moats", "competitive advantage", "franchise value", "pricing power"),
    "mistakes_lessons": ("mistake", "mistakes", "regret"),
    "berkshire_performance": ("berkshire's performance", "berkshire performance", "annual performance"),
    "stock_issuance_splits": ("stock split", "stock splits", "issuing stock", "class b shares"),
    "climate_esg": ("climate", "esg", "climate change"),
    "american_economy_outlook": ("economic outlook", "economy's prospects", "american economy"),
}
_COMPANY_SUFFIX = re.compile(r"\s+(?:international|inc\.?|corp\.?|corporation|company|co\.?|group|holdings)$", re.I)

# --- patterns ---
_YEAR = r"(?:1[89]\d{2}|20\d{2})"
_MEETING = r"(?:annual meeting|shareholder weekend|shareholders' meeting|meeting weekend|the meeting)"
# meeting logistics and side events: tickets, attendance, shopping, ballgames
_MEETING_SIDE = (r"(?:tickets?|schedule|location|venue|seating|registration|what time|exhibit\w*|shop\w*|bargains?|"
                 r"merchandise|discounts?|attend\w*|showed up|show up|turn(?:ed)? out|turnout|crowds?|came back|"
                 r"ball ?games?|games?)")

_PERSONAL_RULES: Tuple[re.Pattern, ...] = tuple(re.compile(p, re.I) for p in (
    r"\bhow old\b", r"\byears? old\b", r"\b(?:his|her|their|buffett's|munger's)\s+age\b",
    r"\bage (?:of|at which) (?:buffett|munger|mrs?\.?\s*b|he|she)\b", r"\bbirthday\b", r"\bborn\b",
    r"\bfamily\b", r"\b(?:wife|husband|daughters?|sons?|grandchild(?:ren)?|grand(?:father|mother)|mother|father|"
    r"brother|sister|parents|children|kids|childhood|marriage|married|spouse)\b",
    r"\bhealth\b(?!\s*(?:care|insurance|plans?|benefits?))", r"\b(?:surgery|illness|sick|medical condition)\b",
    r"\bdiet\b", r"\b(?:eats?|eating|breakfast|lunch|dinner|hamburgers?|favou?rite food)\b",
    r"\b(?:jokes?|joking|funny|humou?r|witty|laugh\w*|anecdotes?)\b",
    r"\b(?:longevity|lifespan|live so long|lived so long|long lives?)\b",
    r"\bpersonal (?:setbacks?|life|lives|story|stories|tragedy|tragedies|loss|losses|struggles?|history|background|"
    r"reasons?|habits?)\b",
    r"\b(?:sports?|baseball|basketball|golf|football|tennis|ping[- ]pong|bowling)\b",
    r"\b(?:first pitch(?:es)?|ball ?games?|ballpark|stadium)\b",
    r"\bplays?\s+bridge\b", r"\bbridge (?:game|player|playing)\b",
    r"\b(?:hotels?|parking|shuttles?|airport|restaurants?|reservations?)\b",
    rf"\b{_MEETING}\b.{{0,60}}\b(?:{_MEETING_SIDE}|where|when is|date)\b",
    rf"\b{_MEETING_SIDE}\b.{{0,60}}\b{_MEETING}\b",
    rf"\bhow many (?:people|shareholders|attendees|guests|visitors)\b.{{0,60}}\b(?:{_MEETING}|meeting)\b",
))
# A business's sales, earnings, growth or price: the clause is financial even next to a founder, family or meeting.
_BUSINESS_FIGURE = re.compile(
    r"\b(?:sales|revenues?|earn(?:s|ed|ing|ings)?|profits?|margins?|growth|grew|grow(?:s|n|ing)?|prices?|priced|"
    r"paid|pay|costs?|"
    r"valuation|book value|float|premiums?|dividends?|market share|operating ratio|returns? on)\b", re.I)

_NUMERIC_METRIC = re.compile(r"\b(?:per[- ]share|float|book value|net worth)\b", re.I)
_NUMERIC_RULES: Tuple[re.Pattern, ...] = tuple(re.compile(p, re.I) for p in (
    r"\bhow (?:much|many)\b", r"\bhow (?:high|large|big)\b", r"\bwhat percent(?:age)?\b", r"\bpercentage of\b",
    r"\bwhat (?:was|were|is|are)\s+(?:the|its|berkshire's|buffett's|that)\s+(?:[\w'$%.,-]+\s+){0,4}?"
    r"(?:value|amount|number|total|level|size|rate|percentage|percent|price|cost|income|earnings|revenues?|sales|"
    r"premiums?|profit|loss|return|gain|ratio|figure|dividend|salary|compensation|market cap|float)\b",
))
_EXPLAIN = re.compile(r"^\s*(?:why|explain|describe|what (?:is|are|does)\b.*\bmean|how does|how do)\b", re.I)

_COMPARISON = re.compile(
    r"\bcompar(?:e|ed|es|ing|ison)\b|\bversus\b|\bvs\.?\b|\bdiffer(?:s|ed|ent|ence|ences)?\b|\bcontrast\b|"
    r"\bsimilarit(?:y|ies)\b|\b(?:better|worse|stronger|weaker)\b.{0,30}\bthan\b", re.I)
_TREND = re.compile(
    r"\b(?:evolv\w*|chang(?:e|ed|es|ing)|trend\w*|shift(?:ed|s|ing)?|progress(?:ion)?|over time|developed over|"
    r"grew over|declin\w+ over|rise and fall|turn(?:ed)? (?:more|less)|become more|became more)\b", re.I)
_CORPUS_STRONG = re.compile(
    r"\bacross (?:all|every|the|these|those|his|buffett's)\b.{0,30}\b(?:letters?|years|decades|reports?)\b|"
    r"\b(?:all|every) (?:of )?(?:the |these |his |buffett's )?(?:annual |shareholder )?letters\b|"
    r"\b(?:over|through|throughout) the (?:years|decades)\b|\bin (?:all|every) (?:the )?letters\b|"
    r"\bagain and again\b|\bevery year\b|\byear after year\b|\bthroughout (?:his |the |all )?(?:career|letters)\b", re.I)
_CORPUS_WEAK = re.compile(r"\balways\b|\bnever\b|\bmost\b(?! of\b)|\bconsistently\b|\bmore often than\b|\boverall\b", re.I)

_FINANCE_CUE = re.compile(
    r"\b(?:berkshire|invest\w*|stocks?|shares?|earnings|profits?|insurance|bonds?|acqui\w+|dividends?|market|"
    r"economy|premiums?|float|price|prices|return|returns|crisis|recession|inflation|tax\w*|debt|sector|"
    r"companies|company|business(?:es)?|revenue|sales|funds?|banks?|cash|buybacks?|repurchas\w*|valuation|"
    r"interest rates?|industr(?:y|ies)|holdings?|stake|subsidiar(?:y|ies)|deal|capital)\b", re.I)

# Corpus-wide superlatives: "which sector did he speak most positively about", "which company did he praise most".
_SUPERLATIVE_TARGET = re.compile(
    r"\b(?:which|what)\s+(?:\w+\s+){0,2}?(sectors?|industr(?:y|ies)|compan(?:y|ies)|business(?:es)?|holdings?|"
    r"investments?|stocks?|subsidiar(?:y|ies))\b", re.I)
_SUPERLATIVE_NEGATIVE = (r"negative(?:ly)?|critical(?:ly)?|pessimistic(?:ally)?|bearish|harsh(?:ly)?|"
                         r"criticiz\w*|criticis\w*|disparag\w*|complain\w*|disliked?")
_SUPERLATIVE_POSITIVE = (r"positive(?:ly)?|favou?rabl[ey]|enthusiastic(?:ally)?|optimistic(?:ally)?|bullish|warmly|"
                         r"prais\w*|admir\w*|lik(?:e|ed|es)|lov(?:e|ed|es)")
_SUPERLATIVE = re.compile(
    rf"\bmost\s+(?:\w+\s+)?(?P<a>{_SUPERLATIVE_POSITIVE}|{_SUPERLATIVE_NEGATIVE})\b|"
    rf"\b(?P<b>{_SUPERLATIVE_POSITIVE}|{_SUPERLATIVE_NEGATIVE})\b[^?.;]{{0,40}}\b(?:the\s+)?most\b", re.I)

_SPAN_SEP = r"(?:\s*(?:-|–|—|to|through|thru|until|and)\s*)"
_RANGE_BETWEEN = re.compile(rf"\b(?:between|from)\s+({_YEAR}){_SPAN_SEP}({_YEAR})\b", re.I)
_RANGE_DASH = re.compile(rf"\b({_YEAR})\s*(?:-|–|—)\s*({_YEAR})\b")
_RANGE_SHORT = re.compile(rf"\b({_YEAR})\s*(?:-|–|—|/)\s*(\d{{2}})\b(?![%.,]\d)")  # "1988-89"
_QUALIFIER = r"(?:(early|mid|late)(?:[- ]?(?:to|or)[- ]?(early|mid|late))?[- ]?)?"
_DECADE_FULL = re.compile(rf"\b{_QUALIFIER}(?:the )?(1[89]|20)(\d)0'?s\b", re.I)
_DECADE_SHORT = re.compile(rf"(?<![\d$]){_QUALIFIER}'?([6-9])0'?s\b", re.I)
# "after 2008", "since the 1996 buyout" (the year of a named event or deal, not a letter)
_AFTER = re.compile(rf"\b(after|since)\s+(?:the\s+|its\s+|his\s+)?({_YEAR})\b(?!\s+(?:annual\s+)?(?:letter|report))", re.I)
_ONWARD = re.compile(rf"\bfrom\s+({_YEAR})\s+on(?:wards?)?\b|\b({_YEAR})\s+(?:onwards?|and (?:later|beyond|after))\b",
                     re.I)
_AROUND = re.compile(rf"\b(?:around|circa|approximately|roughly)\s+({_YEAR})\b", re.I)
_DECADE_EDGE = re.compile(r"\b(?:the\s+)?(end|close|start|beginning)\s+of\s+(?:the\s+)?(?=(?:(?:1[89]|20)\d0|'?[6-9]0)'?s\b)",
                          re.I)
_BEFORE = re.compile(rf"\b(before|prior to|until|up to)\s+(?:the\s+|its\s+|his\s+)?({_YEAR})\b"
                     rf"(?!\s+(?:annual\s+)?(?:letter|report))", re.I)
_YEAR_LETTER = re.compile(
    rf"\b({_YEAR})\s+(?:annual\s+|shareholder(?:s'?)?\s+|berkshire\s+|chairman'?s?\s+)*(?:letter|report)\b|"
    rf"\b(?:letter|report)\s+(?:for|of|from)\s+(?:the\s+year\s+)?({_YEAR})\b|"
    rf"\bin\s+(?:his\s+|buffett's\s+|the\s+)?({_YEAR})\s+(?:\w+\s+){{0,2}}?(?:letter|report)\b", re.I)
_YEAR_SINGLE = re.compile(rf"(?<![\d$.,/])({_YEAR})(?![\d%,.]\d|\d)")
_DECADE_EARLY_MID_LATE = {"early": (0, 3), "mid": (3, 6), "late": (6, 9)}


@dataclass(frozen=True)
class QueryPlan:
    query: str
    scope: str = FINANCIAL                      # financial | personal | mixed
    financial_query: str = ""                   # the query with personal clauses removed
    kind: str = SPECIFIC                        # specific | period | trend | comparison | numeric | corpus_wide
    year_range: Optional[Tuple[int, int]] = None
    year_hard: bool = False                     # True only for "in the YYYY letter" questions
    years: Tuple[int, ...] = ()                 # explicit single years named in the question
    year_spans: Tuple[Tuple[int, int], ...] = ()  # each year mention of the question, in order (a single year: (y, y))
    year_source: str = ""                       # explicit | event | "" (where year_range comes from)
    events: Tuple[str, ...] = ()
    sectors: Tuple[str, ...] = ()
    companies: Tuple[str, ...] = ()
    themes: Tuple[str, ...] = ()
    terms: Tuple[str, ...] = ()                 # topic terms for the boost (oil crisis: oil, energy, OPEC, gasoline)
    personal_clauses: Tuple[str, ...] = ()
    scope_source: str = "rules"                 # rules | vote | default
    vote_share: Optional[float] = None
    numeric: bool = False                       # the question asks for a figure (any kind)
    superlative: str = ""                       # "sector:positive" | "company:negative" | ... | "" (corpus-wide ranking)
    premise_note: Optional[str] = None          # FINANCE_PREMISE_CHECK: one sentence correcting the asked period, else None


def _norm(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def _phrase_re(phrase: str) -> re.Pattern:
    return re.compile(r"(?<![\w])" + re.escape(phrase.lower()) + r"(?:s)?(?![\w])" if phrase[-1:].isalpha()
                      else r"(?<![\w])" + re.escape(phrase.lower()) + r"(?![\w])")


def _find_phrases(text: str, phrases: Sequence[str]) -> List[Tuple[int, int, str]]:
    low = text.lower()
    out = []
    for phrase in phrases:
        for m in _phrase_re(phrase).finditer(low):
            out.append((m.start(), m.end(), phrase))
    return out


# --- scope ---
def _personal_hit(text: str) -> bool:
    return any(rule.search(text) for rule in _PERSONAL_RULES)


def _personal_clause(text: str) -> bool:
    """A personal rule fires and the clause is not about a business's figures."""
    return _personal_hit(text) and not _BUSINESS_FIGURE.search(text)


def _finance_cue(text: str, store: KnowledgeStore) -> bool:
    """Any financial word, business figure, sector, theme, company or event phrase (years alone are not a cue)."""
    if _FINANCE_CUE.search(text) or _BUSINESS_FIGURE.search(text):
        return True
    if _match_keys(text, _sector_terms(store)) or _match_keys(text, THEME_TERMS) or _match_companies(text, store):
        return True
    return bool(_match_events(text, store)[0])


def _split_scope(parts: Sequence[str], personal: Sequence[str]) -> Tuple[str, str, Tuple[str, ...]]:
    financial = [p for p in parts if p not in personal]
    if not financial:
        return PERSONAL, "", tuple(personal)
    return MIXED, " ".join(financial), tuple(personal)


def _scope_from_rules(query: str, store: KnowledgeStore) -> Tuple[str, str, Tuple[str, ...]]:
    """(scope or '' when silent, financial_query, personal clauses)."""
    if not _personal_hit(query):
        return "", query, ()
    parts = [p for p in split_question(query) if p] or [query]
    personal = [p for p in parts if _personal_clause(p)]
    if not personal:  # every personal word sits in a clause about a business's figures
        return "", query, ()
    # a clause with no financial cue next to a personal one is its follow-up ("what did he credit for it?")
    personal += [p for p in parts if p not in personal and not _finance_cue(p, store)]
    return _split_scope(parts, [p for p in parts if p in personal])


def _scope_from_vote(query: str, store: KnowledgeStore) -> Tuple[str, str, Tuple[str, ...]]:
    """The vote said personal: a question about a business's figures keeps its clauses that carry a financial cue."""
    if not _BUSINESS_FIGURE.search(query):
        return PERSONAL, "", (query,)
    parts = [p for p in split_question(query) if p] or [query]
    personal = [p for p in parts if not _finance_cue(p, store)]
    if not personal:
        return FINANCIAL, query, ()
    return _split_scope(parts, personal)


# --- years ---
def _decade_range(century: str, digit: str, qualifier: Optional[str], until: Optional[str] = None) -> Tuple[int, int]:
    start = int(century) * 100 + int(digit) * 10
    lo, hi = _DECADE_EARLY_MID_LATE.get((qualifier or "").lower(), (0, 9))
    if until:  # "early-to-mid 1980s": from the first qualifier's start to the second one's end
        hi = _DECADE_EARLY_MID_LATE[until.lower()][1]
    return start + lo, start + hi


def _short_range(m: re.Match) -> Optional[Tuple[int, int]]:
    start = int(m.group(1))
    end = start // 100 * 100 + int(m.group(2))
    end += 100 if end < start else 0
    return (start, end) if end - start <= 10 else None


def _edge(m: re.Match) -> str:
    return "late " if m.group(1).lower() in ("end", "close") else "early "


def _parse_years(text: str):
    """(year_range, year_hard, explicit single years, spans) from the text with event phrases already blanked out."""
    hard_years = [int(next(g for g in m.groups() if g)) for m in _YEAR_LETTER.finditer(text)]
    work = _DECADE_EDGE.sub(_edge, text)  # "the end of the 1980s" -> "late 1980s"
    ranges: List[Tuple[int, int]] = []
    singles: List[int] = []

    def take(pattern: re.Pattern, handler) -> None:
        nonlocal work
        for m in list(pattern.finditer(work)):
            rng = handler(m)
            if rng:
                ranges.append(rng)
        work = pattern.sub(" ", work)

    take(_RANGE_BETWEEN, lambda m: tuple(sorted((int(m.group(1)), int(m.group(2))))))
    take(_RANGE_DASH, lambda m: tuple(sorted((int(m.group(1)), int(m.group(2))))))
    take(_RANGE_SHORT, _short_range)
    take(_DECADE_FULL, lambda m: _decade_range(m.group(3), m.group(4), m.group(1), m.group(2)))
    take(_DECADE_SHORT, lambda m: _decade_range("19", m.group(3), m.group(1), m.group(2)))
    take(_ONWARD, lambda m: (int(m.group(1) or m.group(2)), LAST_LETTER_YEAR))
    take(_AROUND, lambda m: (int(m.group(1)) - 1, int(m.group(1)) + 1))
    take(_AFTER, lambda m: (int(m.group(2)) + (1 if m.group(1).lower() == "after" else 0), LAST_LETTER_YEAR))
    take(_BEFORE, lambda m: (FIRST_LETTER_YEAR, int(m.group(2)) - (1 if m.group(1).lower() in ("before", "prior to") else 0)))
    for m in _YEAR_SINGLE.finditer(work):
        year = int(m.group(1))
        singles.append(year)
        ranges.append((year, year))
    if not ranges and not hard_years:
        return None, False, (), ()
    for year in hard_years:
        if year not in singles:
            singles.append(year)
            ranges.append((year, year))
    lo, hi = min(r[0] for r in ranges), max(r[1] for r in ranges)
    hard = bool(hard_years) and len(set(singles)) == 1 and len(ranges) == 1
    return (lo, hi), hard, tuple(sorted(set(singles))), tuple(dict.fromkeys(ranges))


# --- events ---
def _match_events(text: str, store: KnowledgeStore):
    """(event ids, fallback/event year range, terms, text with the matched phrases blanked out)."""
    return _scan_events(text, store)[:4]


def _scan_events(text: str, store: KnowledgeStore):
    """`_match_events` plus the alias entries that name an event the store has no id for (they carry `real_years`)."""
    known = store.events()
    phrases: Dict[str, Dict[str, Any]] = {}
    for key, ev in known.items():
        for alias in ev.get("aliases", []):
            phrases[alias.lower()] = {"event": key}
    for alias, spec in USER_EVENT_ALIASES.items():
        phrases.setdefault(alias, spec)
    found = _find_phrases(text, list(phrases))
    # longest phrases first so "2008 financial crisis" wins over "financial crisis"
    found.sort(key=lambda t: (-(t[1] - t[0]), t[0]))
    taken: List[Tuple[int, int]] = []
    events: List[str] = []
    terms: List[str] = []
    fallback: List[Tuple[int, int]] = []
    unknown: List[Dict[str, Any]] = []
    blanked = list(text)
    for start, end, phrase in found:
        if any(start < b and a < end for a, b in taken):
            continue
        spec = phrases[phrase]
        if _year_contradicts(text[:start], known.get(spec.get("event") or ""), spec.get("years")):
            continue  # "the 1997 Asian financial crisis" is not the 2008 one
        taken.append((start, end))
        for i in range(start, end):
            blanked[i] = " "
        event = spec.get("event")
        if event and event in known:
            if event not in events:
                events.append(event)
            fallback.append(_event_range(known[event], spec.get("years")))
        elif spec.get("years"):
            fallback.append(tuple(spec["years"]))
        if not (event and event in known) and spec.get("real_years") and spec not in unknown:
            unknown.append(spec)
        for term in spec.get("terms", ()):
            if term not in terms:
                terms.append(term)
    rng = (min(r[0] for r in fallback), max(r[1] for r in fallback)) if fallback else None
    return tuple(events), rng, tuple(terms), "".join(blanked), unknown


_YEAR_BEFORE = re.compile(rf"\b({_YEAR})\s+(?:[\w'-]+\s+){{0,2}}$")


def _year_contradicts(before: str, event: Optional[Dict[str, Any]], fallback) -> bool:
    """A year just before the phrase (at most two words between) outside the event's years and letters, +-1."""
    m = _YEAR_BEFORE.search(before)
    if not m:
        return False
    years = list((event or {}).get("event_years") or []) + list((event or {}).get("letter_years") or [])
    years += list(fallback or ())
    return bool(years) and not min(years) - 1 <= int(m.group(1)) <= max(years) + 1


def _event_range(event: Dict[str, Any], fallback: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    years = event.get("letter_years") or event.get("event_years") or ()
    if years:
        return min(years), max(years)
    return tuple(fallback) if fallback else (FIRST_LETTER_YEAR, LAST_LETTER_YEAR)  # type: ignore[return-value]


# --- entities ---
def _sector_terms(store: KnowledgeStore) -> Dict[str, List[str]]:
    terms = {k: list(v) for k, v in SECTOR_TERMS.items()}
    for key, desc in (store.taxonomy.get("sectors") or {}).items():
        bucket = terms.setdefault(key, [])
        for piece in re.split(r"[,;()]", str(desc)):
            piece = re.sub(r"^(?:the|and|a)\s+", "", piece.strip().lower())
            words = piece.split()
            if 1 <= len(words) <= 3 and len(piece) >= 5 and piece not in _SECTOR_STOP \
                    and not any(w in _SECTOR_STOP for w in words) and piece not in bucket:
                bucket.append(piece)
    return terms


def _match_keys(text: str, table: Dict[str, Sequence[str]]) -> Tuple[str, ...]:
    return tuple(key for key, phrases in table.items() if _find_phrases(text, phrases))


def _match_companies(text: str, store: KnowledgeStore) -> Tuple[str, ...]:
    """Canonical company names named in the text; aliases ("Burlington Northern Santa Fe" for BNSF) must keep their case."""
    low = text.lower()
    found = []
    aliases = store.company_aliases()
    for name in sorted(set(store.company_names()) | set(aliases)):
        variants = {name.lower(), _COMPANY_SUFFIX.sub("", name).lower()}
        if any(len(v) >= 3 and re.search(r"(?<![\w])" + re.escape(v) + r"(?:'s)?(?![\w])", low) for v in variants) \
                or any(len(a) >= 3 and re.search(r"(?<![\w])" + re.escape(a) + r"(?:'s)?(?![\w])", text)
                       for a in aliases.get(name, ())):
            found.append(name)
    return tuple(found)


# --- kind ---
def _numeric_intent(query: str) -> bool:
    if any(rule.search(query) for rule in _NUMERIC_RULES):
        return True
    return bool(_NUMERIC_METRIC.search(query)) and not _EXPLAIN.search(query)


def _kind(query: str, year_range, year_hard: bool, events, years) -> str:
    has_period = year_range is not None or bool(events)
    if _COMPARISON.search(query) or (len(years) >= 2 and re.search(r"\b(?:and|or|versus|vs\.?)\b", query, re.I)
                                     and re.search(r"\b(?:how|what|which)\b.*\b(?:compare|differ|better|worse)", query, re.I)):
        return COMPARISON
    if _TREND.search(query):
        return TREND
    if not has_period and (_CORPUS_STRONG.search(query) or _CORPUS_WEAK.search(query)) and not _numeric_intent(query):
        return CORPUS_WIDE
    if _numeric_intent(query):
        return NUMERIC
    if year_hard:
        return SPECIFIC
    if has_period:
        return PERIOD
    if _CORPUS_STRONG.search(query):
        return CORPUS_WIDE
    return SPECIFIC


def _superlative(query: str) -> str:
    """"sector:positive", "company:negative", ...: a question that ranks sectors or companies by Buffett's stance."""
    target, stance = _SUPERLATIVE_TARGET.search(query), _SUPERLATIVE.search(query)
    if not target or not stance:
        return ""
    kind = "sector" if target.group(1).lower().startswith(("sector", "industr")) else "company"
    word = stance.group("a") or stance.group("b")
    polarity = "negative" if re.fullmatch(_SUPERLATIVE_NEGATIVE, word, re.I) else "positive"
    return f"{kind}:{polarity}"


def _vote_share(scope_vote: Callable[[str], float], text: str) -> Optional[float]:
    try:
        share = float(scope_vote(text))
    except Exception:
        return None
    return share


# --- premise check ---
PREMISE_MAX_LISTED = 6  # letter-year runs the note names at most; more end with "and other"
# "the 2023 collapse of ...": the year dates the event; it does not ask for the letters of that year
_EVENT_NOUN = (r"(?:collapse|crash|crisis|bankruptcy|failure|buyout|acquisition|purchase|merger|takeover|deal|scandal|"
               r"attacks?|bubble|recession|panic|meltdown|bailout|default|earthquake|hurricane|storm|bid)")
_DATED_EVENT = re.compile(rf"\b(?:the|its|his|her|their|an?)\s+({_YEAR})\s+(?:[a-z-]+\s+){{0,2}}?{_EVENT_NOUN}\b", re.I)


def _runs(years: Sequence[int]) -> List[Tuple[int, int]]:
    """Consecutive years as inclusive spans: [1994, 2007, 2008] -> [(1994, 1994), (2007, 2008)]."""
    out: List[List[int]] = []
    for year in sorted(set(years)):
        if out and year == out[-1][1] + 1:
            out[-1][1] = year
        else:
            out.append([year, year])
    return [(a, b) for a, b in out]


def _span_text(a: int, b: int) -> str:
    if a == b:
        return str(a)
    return f"{a}-{b % 100:02d}" if a // 100 == b // 100 else f"{a}-{b}"


def _join(items: Sequence[str], tail: str = "and") -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {tail} " + items[-1]


def _asked_text(spans: Sequence[Tuple[int, int]]) -> str:
    """"the 1980s", "1999", "1990-1993"; several mentions are joined with "and"."""
    return _join([f"the {a}s" if a % 10 == 0 and b == a + 9 else str(a) if a == b else f"{a}-{b}" for a, b in spans])


def _letters_text(years: Sequence[int]) -> str:
    """"1994 and 2023", "2007-11"; past PREMISE_MAX_LISTED runs the rest is "other"."""
    runs = [_span_text(a, b) for a, b in _runs(years)]
    return _join(runs[:PREMISE_MAX_LISTED] + (["other"] if len(runs) > PREMISE_MAX_LISTED else []))


def _overlaps(spans: Sequence[Tuple[int, int]], years: Sequence[int]) -> bool:
    return any(a <= year <= b for a, b in spans for year in years)


def _span_years(spans: Sequence[Tuple[int, int]]) -> List[int]:
    return [year for a, b in spans for year in range(a, b + 1)]


def _topic_label(text: str, key: str, table: Dict[str, Sequence[str]]) -> str:
    """The words the question uses for a sector or theme ("airlines"); the key with spaces when none match."""
    hits = _find_phrases(text, table.get(key, ()))
    if not hits:
        return key.replace("_", " ")
    start, end, _ = min(hits, key=lambda hit: (hit[0], hit[0] - hit[1]))  # earliest, then longest ("oil and gas")
    return text[start:end].lower()


def _event_subjects(plan: QueryPlan, store: KnowledgeStore, text: str) -> List[Dict[str, Any]]:
    """Events the question names: label, plural, real-year spans (empty when unknown) and the letter years that carry it."""
    known = store.events()
    out: List[Dict[str, Any]] = []
    for key in plan.events:
        ev = known.get(key)
        if ev is None:
            continue
        out.append({"label": EVENT_NAMES.get(key) or "the " + key.replace("_", " "), "plural": key in _PLURAL_EVENT_NAMES,
                    "real": _runs(ev["event_years"]),
                    "letters": sorted(set(ev["letter_years"]) | set(store.letter_years_with(event=key)))})
    for spec in _scan_events(text, store)[4]:
        out.append({"label": spec["label"], "plural": bool(spec.get("plural")), "real": list(spec["real_years"]),
                    "letters": store.letter_years_mentioning(spec.get("letter_phrases", ()))})
    return out


def premise_note(plan: QueryPlan, store: Optional[KnowledgeStore] = None) -> Optional[str]:
    """One plain sentence when the asked period does not fit the question's topic, else None.

    Needs a period the question names itself (`year_source == "explicit"`). With a named event the plan is checked
    against the event: its real years miss the asked period, or no passage tagged with the event (or, for an event
    the store has no id for, no passage summary naming it) lies in the asked years while some lie in other years.
    A letter from the asked years that carries the event, or an event that did happen in the asked years, means there is
    nothing to correct. A question that names no event, company or figure is checked against its sectors and themes by
    the tag test alone. A year that dates an event in the question ("the 2023 collapse of ...") is not an asked period.
    The note names the letter years that do carry the topic.
    """
    store = store if store is not None else KnowledgeStore.empty()
    text = plan.financial_query or plan.query
    dated = {int(m.group(1)) for m in _DATED_EVENT.finditer(text)}
    spans = tuple(span for span in plan.year_spans if not (span[0] == span[1] and span[0] in dated))
    if plan.scope == PERSONAL or plan.year_source != "explicit" or not spans:
        return None
    asked, asked_years = _asked_text(spans), _span_years(spans)
    events = _event_subjects(plan, store, text)
    for ev in events:
        if _overlaps(spans, ev["letters"]):
            continue
        pronoun = "them" if ev["plural"] else "it"
        letters = _letters_text(ev["letters"]) if ev["letters"] else ""
        if ev["real"] and not _overlaps(ev["real"], asked_years):
            when = _join([_span_text(a, b) for a, b in ev["real"]])
            name = ev["label"][0].upper() + ev["label"][1:]
            mention = f", but the {letters} letters mention {pronoun}" if letters else ""
            return f"{name} {'were' if ev['plural'] else 'was'} in {when}; " \
                   f"the letters from {asked} do not discuss {pronoun}{mention}."
        if letters and not ev["real"]:  # the event did happen in the asked years: its date is right, nothing to correct
            return f"The letters from {asked} do not discuss {ev['label']}, but the {letters} letters do."
    letter_range = any(a <= LAST_LETTER_YEAR and b >= FIRST_LETTER_YEAR for a, b in spans)
    if events or plan.companies or plan.numeric or not letter_range:
        return None  # no tag claim for a company or figure question, or for years before the first letter
    sector_table = _sector_terms(store) if plan.sectors else {}
    topics = [(_topic_label(text, key, sector_table), store.letter_years_with(sector=key)) for key in plan.sectors]
    topics += [(_topic_label(text, key, THEME_TERMS), store.letter_years_with(theme=key)) for key in plan.themes]
    topics = [(label, letters) for label, letters in topics if letters]  # a topic no letter carries has nothing to name
    if not topics or any(_overlaps(spans, letters) for _, letters in topics):
        return None
    label, letters = topics[0]
    return f"The letters from {asked} do not discuss {label}, but the {_letters_text(letters)} letters do."


def plan_query(query: str, store: Optional[KnowledgeStore] = None,
               scope_vote: Optional[Callable[[str], float]] = None,
               premise_check: Optional[bool] = None) -> QueryPlan:
    """Plan one question. `scope_vote(query)` is called only when the scope rules are silent.

    `premise_check` fills `premise_note` (see `premise_note`); None reads FINANCE_PREMISE_CHECK at call time.
    """
    store = store if store is not None else KnowledgeStore.empty()
    query = _norm(str(query or ""))
    scope, financial_query, personal = _scope_from_rules(query, store)
    source, share = "rules", None
    if not scope:
        scope, source = FINANCIAL, "default"
        if scope_vote is not None and query:
            share = _vote_share(scope_vote, query)
            if share is not None:
                source = "vote"
                if share >= SCOPE_VOTE_PERSONAL_SHARE:
                    scope, financial_query, personal = _scope_from_vote(query, store)
    if scope == FINANCIAL:
        financial_query = query

    if scope == PERSONAL:  # nothing financial to plan; the caller gives the scope reply
        return QueryPlan(query=query, scope=PERSONAL, financial_query="", kind=SPECIFIC,
                         personal_clauses=personal or (query,), scope_source=source, vote_share=share)
    text = financial_query or query
    events, event_range, terms, blanked = _match_events(text, store)
    year_range, year_hard, years, spans = _parse_years(blanked)
    year_source = "explicit" if year_range is not None else ""
    if year_range is None and event_range is not None:
        year_range, year_source = event_range, "event"
    sectors = _match_keys(text, _sector_terms(store))
    themes = _match_keys(blanked, THEME_TERMS)  # "Long-Term Capital Management" is an event, not the management theme
    companies = _match_companies(text, store)
    kind = _kind(text, year_range, year_hard, events, years)
    plan = QueryPlan(
        query=query, scope=scope, financial_query=financial_query, kind=kind, year_range=year_range,
        year_hard=year_hard, years=years, year_spans=spans, year_source=year_source, events=events, sectors=sectors,
        companies=companies, themes=themes, terms=terms, personal_clauses=personal, scope_source=source,
        vote_share=share, numeric=_numeric_intent(text), superlative=_superlative(text))
    if premise_check is None:
        premise_check = bool(finance_setting("FINANCE_PREMISE_CHECK"))
    note = premise_note(plan, store) if premise_check else None
    return replace(plan, premise_note=note) if note else plan


# --- the site's Year / Decade scope (the request's `where`) ---
def where_year_range(where: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    """The inclusive year range a request's metadata filter allows (the site's Year / Decade scope), else None.

    `year` wins over `decade`. A year is an integer or an operator dict ($eq, $gte, $lte, $gt, $lt, $in); a decade
    (1990) is the years 1990-1999. Open ends fall back to the corpus bounds."""
    if not where:
        return None

    def bounds(cond, width: int) -> Optional[Tuple[int, int]]:
        if isinstance(cond, int) and not isinstance(cond, bool):
            return cond, cond + width
        if not isinstance(cond, dict):
            return None
        lo, hi = FIRST_LETTER_YEAR, LAST_LETTER_YEAR + width
        if "$eq" in cond:
            lo, hi = cond["$eq"], cond["$eq"] + width
        if "$in" in cond:
            values = [v for v in cond["$in"] if isinstance(v, int)]
            if not values:
                return None
            lo, hi = min(values), max(values) + width
        if "$gte" in cond:
            lo = max(lo, cond["$gte"])
        if "$gt" in cond:
            lo = max(lo, cond["$gt"] + 1)
        if "$lte" in cond:
            hi = min(hi, cond["$lte"] + width)
        if "$lt" in cond:
            hi = min(hi, cond["$lt"] - 1 + width)
        return (lo, hi) if lo <= hi else None

    if "year" in where:
        return bounds(where["year"], 0)
    if "decade" in where:
        return bounds(where["decade"], 9)
    return None


def scope_plan(plan: QueryPlan, year_range: Tuple[int, int], store: Optional[KnowledgeStore] = None,
               premise_check: Optional[bool] = None) -> QueryPlan:
    """`plan` restricted to the letters the user picked in the site's Year / Decade scope.

    The scope is the user's explicit choice, like "in the YYYY letter", so it becomes a hard filter. A question with
    explicit years of its own keeps the part of them inside the scope (1985-1995 with the 1990s picked: 1990-1995). A
    question whose own years lie wholly outside the scope ("book value in 1987" with 1990 picked) is searched in the
    scope's letters but loses its figure lookup, so no other year's figure is given as the answer to it. The kind is
    planned again with the new years (a corpus-wide question becomes a period question; the superlative ranking, which
    is corpus-wide by design, is not used), and so is the premise note: "What happened in 2008?" with the 2018 scope
    gets a note when the 2018 letter does not carry the topic."""
    if plan.scope == PERSONAL:
        return plan
    lo, hi = sorted(year_range)
    text = plan.financial_query or plan.query
    changes: Dict[str, Any] = {}
    if plan.year_source == "explicit" and plan.year_range:
        own_lo, own_hi = sorted(plan.year_range)
        if max(lo, own_lo) <= min(hi, own_hi):  # the question's own years overlap the scope: keep that part
            lo, hi = max(lo, own_lo), min(hi, own_hi)
            years = tuple(y for y in plan.years if lo <= y <= hi)
            spans = tuple((max(a, lo), min(b, hi)) for a, b in plan.year_spans if max(a, lo) <= min(b, hi))
        else:  # wholly outside the scope: no figure of another year may answer it
            years, spans = ((lo,) if lo == hi else ()), ((lo, hi),)
            changes.update(numeric=False)
    else:
        years, spans = ((lo,) if lo == hi else ()), ((lo, hi),)
    kind = _kind(text, (lo, hi), True, plan.events, years)
    if changes.get("numeric") is False and kind == NUMERIC:
        kind = PERIOD
    scoped = replace(plan, year_range=(lo, hi), year_hard=True, years=years, year_spans=spans or ((lo, hi),),
                     year_source="explicit", superlative="", premise_note=None, kind=kind, **changes)
    if premise_check is None:
        premise_check = bool(finance_setting("FINANCE_PREMISE_CHECK"))
    note = premise_note(scoped, store) if premise_check else None
    return replace(scoped, premise_note=note) if note else scoped
