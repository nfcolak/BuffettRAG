"""Validate the frozen Buffett Partnership evaluation sets (devpl, heldout_v7, heldout_v8) without a model or retriever.

Usage:
    check_pl_sets.py --set devpl|heldout_v7|heldout_v8  -> prints "OK set=<s> n=<n> sha256=<sha256 of that set's file>"
    check_pl_sets.py --all                              -> checks every present set plus the cross-set rules and prints
                                                           "OK all n=<total> sets=<names> sha256_<set>=<..> ..."
    check_pl_sets.py --excluded-passage-ids             -> prints ONLY a sorted JSON list of every gold/relevant passage id
                                                           of all present sets plus same-letter +-1 neighbours

Set years (partnership rows only, field year = the year a letter reports on; the three sets are disjoint):
    devpl       1957 1960 1961 1964 1967 1970
    heldout_v7  1958 1962 1965 1968
    heldout_v8  1959 1963 1966 1969

Required terms are checked verbatim (case-insensitive) against the raw passage text. Only chunks_v5_pl.jsonl is read.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
MAIN_CHECKOUT = Path("/Users/necatifurkancolak/AI-Workplace/Projects/current/BuffettRAG")
CORPUS_PATH = "data/processed/chunks_v5_pl.jsonl"
CORPUS_SHA256 = "42478c6a7327cc139f34678c7abd2884447f1f109de285474a6fbb7799740c7c"
SET_PATHS = {
    "devpl": ROOT / "data/evaluation/devpl/devpl.json",
    "heldout_v7": ROOT / "data/evaluation/heldout_v7/answer_benchmark_heldout_v7.json",
    "heldout_v8": ROOT / "data/evaluation/heldout_v8/answer_benchmark_heldout_v8.json",
}
SET_YEARS = {
    "devpl": (1957, 1960, 1961, 1964, 1967, 1970),
    "heldout_v7": (1958, 1962, 1965, 1968),
    "heldout_v8": (1959, 1963, 1966, 1969),
}
QID_PREFIX = {"devpl": None, "heldout_v7": "hv7_", "heldout_v8": "hv8_"}
SIZE = 45
TYPE_MIX = {"opinion": 8, "company": 7, "fact_number": 9, "follow_up": 6, "temporal_comparison": 7, "unanswerable": 8}
CREATED = "2026-10-12"
MIN_WORDS = 40
MIN_CASES_PER_YEAR = 3
MAX_CLAIM_CHARS = 400
REFUSAL = "The retrieved passages do not provide enough evidence to answer this question."
REFUSAL_CATEGORIES = {"out_of_domain", "unsupported_figure", "future_absent_year", "false_premise", "both_sides_unsupported"}
REQUIRED_FINANCE_CATEGORIES = ("unsupported_figure", "future_absent_year", "false_premise", "both_sides_unsupported")
TOP_KEYS = {"schema_version", "purpose", "frozen", "created", "source_corpus", "corpus_sha256", "set", "cases"}
BASE_KEYS = {"qid", "question_type", "answerable", "years", "query", "gold_passage_ids", "relevant_ids", "gold_claims",
             "accept", "reject", "verification_note", "ft_context_overlap"}
ANSWERABLE_KEYS = BASE_KEYS | {"multi_part"}
REFUSAL_KEYS = BASE_KEYS | {"refusal_category", "expected_answer", "absence_search_patterns"}
TOKEN_RE = re.compile(r"[a-z0-9]+")
ID_RE = re.compile(r"(.+_p)(\d{4})")
PASSAGE_WORDING = ("the passage", "this passage", "the passages", "the letter says", "the letter states",
                   "according to the letter", "the excerpt", "the text says")
COPY_NGRAM = 8
YEAR_PATTERN = re.compile(r"(?:\\b|\^)?(\d{4})(?:\\b|\$)?")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def token_list(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def tokens(text: str) -> set[str]:
    return set(token_list(text))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def neighbours(passage_id: str) -> set[str]:
    """Same-letter paragraph ids at +-1 (pure id arithmetic; non-existent ids are harmless)."""
    match = ID_RE.fullmatch(passage_id)
    if match is None:
        return set()
    prefix, number = match.group(1), int(match.group(2))
    return {f"{prefix}{other:04d}" for other in (number - 1, number + 1) if other >= 0}


def evidence_ids(cases: list[dict]) -> set[str]:
    found: set[str] = set()
    for case in cases:
        for field in ("gold_passage_ids", "relevant_ids"):
            require(isinstance(case.get(field), list), f"{case.get('qid')}: invalid {field}")
            found.update(case[field])
    return found


def corpus_bytes() -> bytes:
    for base in (ROOT, MAIN_CHECKOUT):
        path = base / CORPUS_PATH
        if path.exists():
            return path.read_bytes()
    raise AssertionError("Frozen corpus file not found")


def load_corpus() -> dict[str, dict]:
    raw = corpus_bytes()
    require(hashlib.sha256(raw).hexdigest() == CORPUS_SHA256, "Corpus hash changed")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    corpus = {row["id"]: row for row in rows}
    require(len(corpus) == len(rows), "Duplicate corpus IDs")
    return corpus


def present_sets() -> list[str]:
    return [name for name in SET_PATHS if SET_PATHS[name].exists()]


def set_evidence(name: str) -> set[str]:
    return evidence_ids(json.loads(SET_PATHS[name].read_bytes())["cases"])


def excluded_passage_ids() -> list[str]:
    """Every gold/relevant id of all present sets plus their same-letter +-1 neighbours, sorted."""
    found: set[str] = set()
    for name in present_sets():
        found |= set_evidence(name)
    result = set(found)
    for passage_id in found:
        result |= neighbours(passage_id)
    return sorted(result)


def check_set(name: str, context: dict | None = None) -> tuple[str, dict]:
    raw = SET_PATHS[name].read_bytes()
    fixture = json.loads(raw)
    set_years = set(SET_YEARS[name])
    require(set(fixture) == TOP_KEYS, "Top-level keys differ from the heldout_v6 schema (+set)")
    require(type(fixture["schema_version"]) is int and fixture["schema_version"] == 1, "Schema must be 1")
    require(fixture["frozen"] is True and fixture["created"] == CREATED, "Wrong freeze metadata")
    require(fixture["set"] == name, "Wrong set marker")
    require(fixture["source_corpus"] == CORPUS_PATH and fixture["corpus_sha256"] == CORPUS_SHA256, "Wrong corpus metadata")
    require(isinstance(fixture["purpose"], str) and bool(fixture["purpose"].strip()), "Missing purpose")
    corpus = context if context is not None else load_corpus()
    corpus_years = {row["year"] for row in corpus.values()}
    all_text = "\n".join(row["text"] for row in corpus.values())
    cases = fixture["cases"]
    require(isinstance(cases, list) and len(cases) == SIZE, f"Need exactly {SIZE} cases")
    qids = [case["qid"] for case in cases]
    prefix = QID_PREFIX[name]
    if prefix:
        require(qids == [f"{prefix}{n:02d}" for n in range(1, SIZE + 1)], f"Qids must be {prefix}01..{SIZE}")
    else:
        stem = qids[0].rsplit("_", 1)[0]
        require(re.fullmatch(r"[a-z0-9]+", stem) is not None and
                qids == [f"{stem}_{n:02d}" for n in range(1, SIZE + 1)], f"Qids must be <prefix>_01..{SIZE}")
    type_counts = Counter(case["question_type"] for case in cases)
    require(dict(type_counts) == TYPE_MIX, f"Wrong type mix {dict(type_counts)}")
    require(len({case["query"].casefold() for case in cases}) == SIZE, "Duplicate queries")
    token_sets = [(case["qid"], tokens(case["query"])) for case in cases]
    for i, (qid, left) in enumerate(token_sets):
        for other_qid, right in token_sets[i + 1:]:
            require(jaccard(left, right) < 0.5, f"{qid}: question overlaps {other_qid}")
    gold_owner: dict[str, str] = {}
    year_cases: Counter[int] = Counter()
    refusal_categories: Counter[str] = Counter()
    partials: list[str] = []
    total_claims = 0
    for case in cases:
        qid, kind, query = case["qid"], case["question_type"], case["query"]
        require(isinstance(query, str) and bool(query.strip()), f"{qid}: empty query")
        require(not any(phrase in query.casefold() for phrase in PASSAGE_WORDING), f"{qid}: passage wording")
        note = case.get("verification_note", "")
        require(isinstance(note, str) and bool(note.strip()) and "\n" not in note, f"{qid}: invalid one-line verification")
        require(type(case.get("ft_context_overlap")) is bool, f"{qid}: context-overlap flag missing")
        require(case["accept"]["min_claim_coverage"] == 1.0, f"{qid}: require all claims")
        require(isinstance(case["reject"]["forbidden_terms"], list), f"{qid}: invalid reject terms")
        years = case["years"]
        require(isinstance(years, list) and len(set(years)) == len(years) and all(type(y) is int for y in years),
                f"{qid}: invalid years")
        for field in ("gold_passage_ids", "relevant_ids"):
            ids = case[field]
            require(isinstance(ids, list) and len(set(ids)) == len(ids), f"{qid}: invalid {field}")
            require(set(ids) <= corpus.keys(), f"{qid}: unknown {field}")
            for pid in ids:
                row = corpus[pid]
                require(pid.startswith("bpl") and row.get("collection", "partnership") == "partnership",
                        f"{qid}: {pid} is not a partnership row")
                require(row["year"] in set_years, f"{qid}: {pid} outside the years of {name}")
                require(row["year"] in years, f"{qid}: {pid} year not among case years")
        gold_ids = set(case["gold_passage_ids"])
        require(gold_ids <= set(case["relevant_ids"]), f"{qid}: missing relevant gold IDs")
        for pid in gold_ids:
            require(len(corpus[pid]["text"].split()) >= MIN_WORDS, f"{qid}: gold {pid} has fewer than {MIN_WORDS} words")
            require(pid not in gold_owner, f"{qid}: gold {pid} already gold for {gold_owner.get(pid)}")
            gold_owner[pid] = qid
        query_tokens = token_list(query)
        for pid in gold_ids:
            passage_tokens = token_list(corpus[pid]["text"])
            grams = {tuple(passage_tokens[i:i + COPY_NGRAM]) for i in range(len(passage_tokens) - COPY_NGRAM + 1)}
            require(not any(tuple(query_tokens[i:i + COPY_NGRAM]) in grams
                            for i in range(len(query_tokens) - COPY_NGRAM + 1)), f"{qid}: query copies a passage")
        if kind == "unanswerable":
            require(set(case) <= REFUSAL_KEYS | {"unsupported_sides", "side_patterns"} and REFUSAL_KEYS <= set(case),
                    f"{qid}: wrong refusal fields")
            category = case["refusal_category"]
            require(category in REFUSAL_CATEGORIES, f"{qid}: wrong refusal category")
            refusal_categories[category] += 1
            require(case["answerable"] is False and years == [], f"{qid}: incorrect refusal shape")
            require(case["gold_passage_ids"] == case["relevant_ids"] == case["gold_claims"] == [], f"{qid}: invented evidence")
            require(case["expected_answer"] == REFUSAL and case["accept"].get("require_exact_refusal") is True and
                    case["accept"].get("require_valid_citation") is False, f"{qid}: wrong exact-refusal acceptance")
            patterns = case["absence_search_patterns"]
            require(isinstance(patterns, list) and bool(patterns) and all(isinstance(p, str) and p for p in patterns),
                    f"{qid}: unaudited absence searches")
            for pattern in patterns:
                require(re.search(pattern, all_text, re.I) is None, f"{qid}: absence search has corpus support")
            require(any(re.search(pattern, query, re.I) for pattern in patterns),
                    f"{qid}: no absence pattern matches the query's unsupported subject")
            if category == "future_absent_year":
                named = {int(m.group(1)) for m in (YEAR_PATTERN.fullmatch(p) for p in patterns) if m}
                require(bool(named) and not named & corpus_years, f"{qid}: future year pattern must name an absent year")
            if category == "both_sides_unsupported":
                sides = case.get("unsupported_sides", [])
                require(len(sides) == 2 and len({s.casefold() for s in sides}) == 2 and
                        all(side.casefold() in query.casefold() for side in sides), f"{qid}: must identify both unsupported sides")
                side_patterns = case.get("side_patterns")
                require(isinstance(side_patterns, dict) and set(side_patterns) == set(sides), f"{qid}: side patterns")
                require(all(re.search(side_patterns[s], all_text, re.I) is None and re.search(side_patterns[s], query, re.I)
                            for s in sides), f"{qid}: one comparison side has corpus support or is not in the query")
            else:
                require("unsupported_sides" not in case and "side_patterns" not in case, f"{qid}: unexpected sides")
            continue
        require(set(case) <= ANSWERABLE_KEYS | {"history", "expected_coverage_note"} and ANSWERABLE_KEYS <= set(case),
                f"{qid}: wrong answerable fields")
        require(case["answerable"] is True and bool(years) and bool(gold_ids) and bool(case["gold_claims"]),
                f"{qid}: missing answerable evidence")
        require(case["accept"].get("require_valid_citation") is True and
                not case["accept"].get("require_exact_refusal", False), f"{qid}: wrong answer acceptance")
        require(case["multi_part"] is True, f"{qid}: multi_part must be true")
        supported_years = {corpus[pid]["year"] for pid in gold_ids}
        year_cases.update(supported_years)
        if kind == "temporal_comparison":
            require(len(years) == 2 and all(str(year) in query for year in years), f"{qid}: comparison needs two named years")
            if "expected_coverage_note" in case:
                partials.append(qid)
                missing = set(years) - supported_years
                require(len(missing) == 1 and len(supported_years) == 1 and supported_years <= set_years,
                        f"{qid}: partial needs exactly one supported side")
                missing_year = missing.pop()
                coverage = case["expected_coverage_note"]
                require(isinstance(coverage, str) and str(missing_year) in coverage and "missing" in coverage.casefold() and
                        "\n" not in coverage, f"{qid}: missing-period coverage note")
                require(missing_year not in corpus_years and missing_year not in set_years, f"{qid}: absent side must truly be absent")
                require(re.search(rf"\b{missing_year}\b", all_text) is None, f"{qid}: missing period has corpus text")
            else:
                require(set(years) <= set_years and supported_years == set(years), f"{qid}: full comparison missing one side")
        else:
            require("expected_coverage_note" not in case, f"{qid}: partial is comparison-only")
            require(set(years) <= set_years, f"{qid}: years outside {name}")
            require(len(years) == 1 or (kind in {"fact_number", "company"} and len(years) == 2), f"{qid}: wrong year count")
            require(supported_years == set(years), f"{qid}: missing evidence for declared period")
        assigned_ids: set[str] = set()
        claim_term_sets = []
        claims = case["gold_claims"]
        require(2 <= len(claims) <= 3, f"{qid}: need 2-3 gold claims")
        for claim in claims:
            require(set(claim) == {"claim", "required_terms", "gold_passage_ids", "verification_note"}, f"{qid}: claim fields")
            require(isinstance(claim["claim"], str) and bool(claim["claim"].strip()) and len(claim["claim"]) <= MAX_CLAIM_CHARS,
                    f"{qid}: empty or overlong claim")
            terms, ids = claim["required_terms"], claim["gold_passage_ids"]
            require(isinstance(terms, list) and 2 <= len(terms) <= 3 and
                    all(isinstance(t, str) and bool(t.strip()) for t in terms) and
                    len({t.casefold() for t in terms}) == len(terms), f"{qid}: invalid 2-3 terms")
            require(isinstance(ids, list) and bool(ids) and len(set(ids)) == len(ids) and set(ids) <= gold_ids,
                    f"{qid}: invalid claim IDs")
            assigned_ids.update(ids)
            claim_note = claim["verification_note"]
            require(isinstance(claim_note, str) and bool(claim_note.strip()) and "\n" not in claim_note,
                    f"{qid}: claim needs a one-line verification_note")
            numeric = False
            for term in terms:
                require(term.casefold() in claim["claim"].casefold(), f"{qid}: term absent from claim: {term}")
                require(all(term.casefold() in corpus[pid]["text"].casefold() for pid in ids),
                        f"{qid}: term not verbatim in evidence: {term}")
                numeric = numeric or bool(re.search(r"\d", term))
            if kind == "fact_number":
                require(numeric, f"{qid}: fact_number claim needs an exact number in a required term")
            claim_term_sets.append(frozenset(t.casefold() for t in terms))
        total_claims += len(claims)
        require(assigned_ids == gold_ids, f"{qid}: unassigned gold evidence")
        require(len(set(claim_term_sets)) == len(claims) and len({c["claim"] for c in claims}) == len(claims),
                f"{qid}: duplicate claims")
        if kind == "follow_up":
            history = case.get("history")
            require(isinstance(history, list) and len(history) == 2 and
                    [m.get("role") for m in history] == ["user", "assistant"], f"{qid}: wrong follow-up history")
            require(all(set(m) == {"role", "content"} and isinstance(m["content"], str) and bool(m["content"].strip())
                        for m in history), f"{qid}: invalid history")
            history_text = " ".join(m["content"] for m in history).casefold()
            require(all(not all(t.casefold() in history_text for t in claim["required_terms"]) for claim in claims),
                    f"{qid}: history reveals scored claim")
        else:
            require("history" not in case, f"{qid}: history on non-follow-up")
    unanswerable = type_counts["unanswerable"]
    require(sum(refusal_categories.values()) == unanswerable and refusal_categories["out_of_domain"] == 2 and
            all(refusal_categories[c] >= 1 for c in REQUIRED_FINANCE_CATEGORIES), "Wrong refusal category mix")
    require(len(partials) == 1, "Need exactly one answerable one-side-absent comparison")
    require(all(year_cases[y] >= MIN_CASES_PER_YEAR for y in set_years),
            f"Each year needs >= {MIN_CASES_PER_YEAR} cases: {dict(year_cases)}")
    require(set(year_cases) <= set_years, "Evidence outside the set years")
    return hashlib.sha256(raw).hexdigest(), {"claims": total_claims, "size": len(cases), "year_cases": dict(year_cases),
                                              "queries": [(c["qid"], c["query"]) for c in cases]}


def check_all() -> str:
    names = present_sets()
    require(bool(names), "No set files present")
    corpus = load_corpus()
    digests, stats = {}, {}
    for name in names:
        digests[name], stats[name] = check_set(name, corpus)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            require(not set(SET_YEARS[left]) & set(SET_YEARS[right]), f"{left}/{right}: years overlap")
            ids_left, ids_right = set_evidence(left), set_evidence(right)
            barred = set(ids_left)
            for passage_id in ids_left:
                barred |= neighbours(passage_id)
            require(not ids_right & barred, f"{left}/{right}: sets share passage ids or neighbours")
            queries_left = [(qid, tokens(q)) for qid, q in stats[left]["queries"]]
            for qid_right, query_right in stats[right]["queries"]:
                current = tokens(query_right)
                for qid_left, earlier in queries_left:
                    require(jaccard(current, earlier) < 0.5, f"{qid_right}: question overlaps {qid_left}")
    total = sum(stats[name]["size"] for name in names)
    digest_text = " ".join(f"sha256_{name}={digests[name]}" for name in names)
    return f"OK all n={total} sets={','.join(names)} {digest_text}"


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--excluded-passage-ids" in args:
        print(json.dumps(excluded_passage_ids()))
    elif "--all" in args:
        print(check_all())
    elif "--set" in args and args.index("--set") + 1 < len(args) and args[args.index("--set") + 1] in SET_PATHS:
        selected = args[args.index("--set") + 1]
        digest, set_stats = check_set(selected)
        print(f"OK set={selected} n={set_stats['size']} sha256={digest}")
    else:
        raise SystemExit("usage: check_pl_sets.py --set devpl|heldout_v7|heldout_v8 | --all | --excluded-passage-ids")
