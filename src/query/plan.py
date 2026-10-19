"""Deterministic query planner (finance-query-pipeline, group 2).

`plan_query(query, store, scope_vote=None)` turns a question into a `QueryPlan`: scope (financial, personal,
mixed), kind, year range, events, sectors, companies and themes. No model, no retrieval except the optional
`scope_vote` callable the caller passes in. Callers gate it behind `FINANCE_PIPELINE=1`; this module does not
read the flag.

Years are a boost range by default; `year_hard` is True only for "in the YYYY letter" questions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    "oil crisis": {"event": None, "years": (1973, 1982), "terms": ("oil", "energy", "OPEC", "gasoline")},
    "oil shock": {"event": None, "years": (1973, 1982), "terms": ("oil", "energy", "OPEC", "gasoline")},
    "oil embargo": {"event": None, "years": (1973, 1982), "terms": ("oil", "energy", "OPEC", "gasoline")},
    "energy crisis": {"event": None, "years": (1973, 1982), "terms": ("oil", "energy", "OPEC", "gasoline")},
    "volcker": {"event": "high_inflation_late70s_early80s", "years": (1979, 1984),
                "terms": ("Volcker", "interest rates", "inflation")},
}

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

_PERSONAL_RULES: Tuple[re.Pattern, ...] = tuple(re.compile(p, re.I) for p in (
    r"\bhow old\b", r"\byears? old\b", r"\b(?:his|her|their|buffett's|munger's)\s+age\b",
    r"\bage (?:of|at which) (?:buffett|munger|mrs?\.?\s*b|he|she)\b", r"\bbirthday\b", r"\bborn\b",
    r"\bfamily\b", r"\b(?:wife|husband|daughters?|sons?|grandchild(?:ren)?|grand(?:father|mother)|mother|father|"
    r"brother|sister|parents|children|kids|childhood|marriage|married|spouse)\b",
    r"\bhealth\b(?!\s*(?:care|insurance|plans?|benefits?))", r"\b(?:surgery|illness|sick|medical condition)\b",
    r"\bdiet\b", r"\b(?:eats?|eating|breakfast|lunch|dinner|hamburgers?|favou?rite food)\b",
    r"\b(?:jokes?|joking|funny|humou?r|witty|laugh\w*)\b",
    r"\b(?:sports?|baseball|basketball|golf|football|tennis|ping[- ]pong|bowling)\b",
    r"\bplays?\s+bridge\b", r"\bbridge (?:game|player|playing)\b",
    r"\b(?:hotels?|parking|shuttles?|airport|restaurants?|reservations?)\b",
    r"\b(?:annual meeting|shareholder weekend|shareholders' meeting|the meeting)\b.{0,60}\b(?:tickets?|schedule|"
    r"location|venue|seating|registration|what time|where|when is|exhibit\w*|shopping|discounts?|attend\w*|date)\b",
    r"\b(?:tickets?|schedule|location|venue|seating|registration|what time|exhibit\w*|shopping day)\b.{0,60}"
    r"\b(?:annual meeting|shareholder weekend|shareholders' meeting|the meeting)\b",
))

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
    r"companies|company|business(?:es)?|revenue|sales)\b", re.I)

_SPAN_SEP = r"(?:\s*(?:-|–|—|to|through|thru|until|and)\s*)"
_RANGE_BETWEEN = re.compile(rf"\b(?:between|from)\s+({_YEAR}){_SPAN_SEP}({_YEAR})\b", re.I)
_RANGE_DASH = re.compile(rf"\b({_YEAR})\s*(?:-|–|—)\s*({_YEAR})\b")
_DECADE_FULL = re.compile(r"\b(?:(early|mid|late)[- ]?)?(?:the )?(1[89]|20)(\d)0'?s\b", re.I)
_DECADE_SHORT = re.compile(r"(?<![\d$])(?:(early|mid|late)[- ]?)?'?([6-9])0'?s\b", re.I)
_AFTER = re.compile(rf"\b(after|since)\s+({_YEAR})\b", re.I)
_BEFORE = re.compile(rf"\b(before|prior to|until|up to)\s+({_YEAR})\b", re.I)
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
    events: Tuple[str, ...] = ()
    sectors: Tuple[str, ...] = ()
    companies: Tuple[str, ...] = ()
    themes: Tuple[str, ...] = ()
    terms: Tuple[str, ...] = ()                 # topic terms for the boost (oil crisis: oil, energy, OPEC, gasoline)
    personal_clauses: Tuple[str, ...] = ()
    scope_source: str = "rules"                 # rules | vote | default
    vote_share: Optional[float] = None


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


def _scope_from_rules(query: str) -> Tuple[str, str, Tuple[str, ...]]:
    """(scope or '' when silent, financial_query, personal clauses)."""
    if not _personal_hit(query):
        return "", query, ()
    parts = [p for p in split_question(query) if p] or [query]
    personal = tuple(p for p in parts if _personal_hit(p))
    financial = [p for p in parts if not _personal_hit(p)]
    if not financial:
        return PERSONAL, "", personal
    return MIXED, " ".join(financial), personal


# --- years ---
def _decade_range(century: str, digit: str, qualifier: Optional[str]) -> Tuple[int, int]:
    start = int(century) * 100 + int(digit) * 10
    lo, hi = _DECADE_EARLY_MID_LATE.get((qualifier or "").lower(), (0, 9))
    return start + lo, start + hi


def _parse_years(text: str) -> Tuple[Optional[Tuple[int, int]], bool, Tuple[int, ...]]:
    """(year_range, year_hard, explicit single years) from the text with event phrases already blanked out."""
    hard_years = [int(next(g for g in m.groups() if g)) for m in _YEAR_LETTER.finditer(text)]
    work = text
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
    take(_DECADE_FULL, lambda m: _decade_range(m.group(2), m.group(3), m.group(1)))
    take(_DECADE_SHORT, lambda m: _decade_range("19", m.group(2), m.group(1)))
    take(_AFTER, lambda m: (int(m.group(2)) + (1 if m.group(1).lower() == "after" else 0), LAST_LETTER_YEAR))
    take(_BEFORE, lambda m: (FIRST_LETTER_YEAR, int(m.group(2)) - (1 if m.group(1).lower() in ("before", "prior to") else 0)))
    for m in _YEAR_SINGLE.finditer(work):
        year = int(m.group(1))
        singles.append(year)
        ranges.append((year, year))
    if not ranges and not hard_years:
        return None, False, ()
    for year in hard_years:
        if year not in singles:
            singles.append(year)
            ranges.append((year, year))
    lo, hi = min(r[0] for r in ranges), max(r[1] for r in ranges)
    hard = bool(hard_years) and len(set(singles)) == 1 and len(ranges) == 1
    return (lo, hi), hard, tuple(sorted(set(singles)))


# --- events ---
def _match_events(text: str, store: KnowledgeStore):
    """(event ids, fallback/event year range, terms, text with the matched phrases blanked out)."""
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
    blanked = list(text)
    for start, end, phrase in found:
        if any(start < b and a < end for a, b in taken):
            continue
        taken.append((start, end))
        for i in range(start, end):
            blanked[i] = " "
        spec = phrases[phrase]
        event = spec.get("event")
        if event and event in known:
            if event not in events:
                events.append(event)
            fallback.append(_event_range(known[event], spec.get("years")))
        elif spec.get("years"):
            fallback.append(tuple(spec["years"]))
        for term in spec.get("terms", ()):
            if term not in terms:
                terms.append(term)
    rng = (min(r[0] for r in fallback), max(r[1] for r in fallback)) if fallback else None
    return tuple(events), rng, tuple(terms), "".join(blanked)


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
    low = text.lower()
    found = []
    for name in store.company_names():
        variants = {name.lower(), _COMPANY_SUFFIX.sub("", name).lower()}
        if any(len(v) >= 3 and re.search(r"(?<![\w])" + re.escape(v) + r"(?:'s)?(?![\w])", low) for v in variants):
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


def _vote_share(scope_vote: Callable[[str], float], text: str) -> Optional[float]:
    try:
        share = float(scope_vote(text))
    except Exception:
        return None
    return share


def plan_query(query: str, store: Optional[KnowledgeStore] = None,
               scope_vote: Optional[Callable[[str], float]] = None) -> QueryPlan:
    """Plan one question. `scope_vote(query)` is called only when the scope rules are silent."""
    store = store if store is not None else KnowledgeStore.empty()
    query = _norm(str(query or ""))
    scope, financial_query, personal = _scope_from_rules(query)
    source, share = "rules", None
    if not scope:
        scope, source = FINANCIAL, "default"
        if scope_vote is not None and query:
            share = _vote_share(scope_vote, query)
            if share is not None:
                source = "vote"
                if share >= SCOPE_VOTE_PERSONAL_SHARE:
                    scope, financial_query = PERSONAL, ""
    if scope == FINANCIAL:
        financial_query = query

    if scope == PERSONAL:  # nothing financial to plan; the caller gives the scope reply
        return QueryPlan(query=query, scope=PERSONAL, financial_query="", kind=SPECIFIC,
                         personal_clauses=personal or (query,), scope_source=source, vote_share=share)
    text = financial_query or query
    events, event_range, terms, blanked = _match_events(text, store)
    year_range, year_hard, years = _parse_years(blanked)
    if year_range is None:
        year_range = event_range
    sectors = _match_keys(text, _sector_terms(store))
    themes = _match_keys(text, THEME_TERMS)
    companies = _match_companies(text, store)
    kind = _kind(text, year_range, year_hard, events, years)
    return QueryPlan(
        query=query, scope=scope, financial_query=financial_query, kind=kind, year_range=year_range,
        year_hard=year_hard, years=years, events=events, sectors=sectors, companies=companies, themes=themes,
        terms=terms, personal_clauses=personal, scope_source=source, vote_share=share)
