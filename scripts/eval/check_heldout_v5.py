"""Validate the frozen blind heldout_v5 fixture (parts a and b) without loading a model or retriever.

Usage:
    check_heldout_v5.py --part a|b            -> prints "OK part=<p> n=60 sha256=<sha256 of that part's file>"
    check_heldout_v5.py --all                 -> checks both parts and the cross-part rules, prints
                                                 "OK n=120 sha256_a=<..> sha256_b=<..>"
    check_heldout_v5.py --excluded-passage-ids -> prints ONLY a sorted JSON list of every gold/relevant
                                                 passage id of all present parts plus same-letter +-1 neighbours

Part a uses only ODD letter years (1977..2023), part b only EVEN ones (1978..2024), so the parts are disjoint.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
MAIN_CHECKOUT = Path("/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG")
PART_PATHS = {
    "a": ROOT / "data/evaluation/heldout_v5/answer_benchmark_heldout_v5_a.json",
    "b": ROOT / "data/evaluation/heldout_v5/answer_benchmark_heldout_v5_b.json",
}
CORPUS_PATH = "data/processed/chunks_v3_paragraph.jsonl"
CORPUS_SHA256 = "8263f08958729a77febf0922e7bbc4161bedd5fdb3e9a8b3fc9a64ab681ab5f0"
FORBIDDEN_PATH = Path("/Users/necatifurkancolak/AI-Workplace/Artifacts/BuffettRAG/round3/v2_forbidden_passages.json")
V1_PATH = "data/evaluation/heldout_v1/answer_benchmark_heldout_v1.json"
V2_PATH = "data/evaluation/heldout_v2/answer_benchmark_heldout_v2.json"
V3_PATH = "data/evaluation/heldout_v3/answer_benchmark_heldout_v3.json"
V4_PATH = "data/evaluation/heldout_v4/answer_benchmark_heldout_v4.json"
DEV_PATH = "data/evaluation/answer_quality_program/answer_benchmark_v3.json"
DEV_CASES_PATH = "data/evaluation/grounded_dev_v1/confirm/dev_cases.json"
HN_PATHS = (
    "data/evaluation/answer_quality_program/hard_negatives_v2.json",
    "data/evaluation/answer_quality_program/hard_negatives_v3.json",
    "data/evaluation/hard_negatives_v4/hard_negatives_v4.json",
)
FT_PATHS = ("data/ft/examples.jsonl", "data/ft_v2/examples.jsonl", "data/ft_v3/examples.jsonl",
            "data/ft_v4/examples.jsonl")
PART_TYPE_COUNTS = {"opinion": 11, "company": 11, "fact_number": 11,
                    "temporal_comparison": 10, "follow_up": 9, "unanswerable": 8}
PART_EXACT_TYPE_COUNTS = {"opinion": 11, "follow_up": 9, "unanswerable": 8}
MIN_TEMPORAL, MAX_TEMPORAL = 6, 10
MIN_COMPANY_FACT = 11
FLEX_TYPE_TOTAL = 32
PART_SIZE = 60
PART_ANSWERABLE = 52
MIN_PART_CLAIMS = 110
MIN_NUMERIC = 14
MIN_YEARS = 15
MAX_CASES_PER_YEAR = 5
MAX_DOUBLE_NONCOMPARISON = 3
MIN_TOTAL_ANSWERABLE = 104
MIN_TOTAL_CLAIMS = 220
PART_PARITY = {"a": 1, "b": 0}  # a: odd letter years 1977..2023, b: even letter years 1978..2024
REFUSAL = "The retrieved passages do not provide enough evidence to answer this question."
SCOPE = "source-heldout (FT answer passages excluded), not unseen-evidence (FT context overlap allowed)"
REFUSAL_CATEGORIES = {"out_of_domain", "unsupported_figure", "future_absent_year", "false_premise",
                      "both_sides_unsupported"}
REQUIRED_FINANCE_CATEGORIES = ("unsupported_figure", "future_absent_year", "false_premise", "both_sides_unsupported")
TOKEN_RE = re.compile(r"[a-z0-9]+")
ID_RE = re.compile(r"(\d{4})_p(\d{4})")
PASSAGE_WORDING = ("the passage", "this passage", "the passages", "the letter says", "the letter states",
                   "according to the letter", "the excerpt", "the text says")
COPY_NGRAM = 8
YEAR_PATTERN = re.compile(r"\\b(\d{4})\\b")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


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
    year, number = match.group(1), int(match.group(2))
    return {f"{year}_p{other:04d}" for other in (number - 1, number + 1) if other >= 0}


def evidence_ids(value: object) -> set[str]:
    """Collect gold/relevant IDs only; never consume result/prediction fixtures."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"gold_passage_ids", "relevant_ids"}:
                require(isinstance(child, list), f"Invalid reference field {key}")
                found.update(child)
            elif isinstance(child, (dict, list)):
                found.update(evidence_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(evidence_ids(child))
    return found


def gold_queries() -> list[dict]:
    """Read literal GoldQuery declarations with AST; do not import app code."""
    tree = ast.parse((ROOT / "src/evaluation/gold_set.py").read_text(encoding="utf-8"))
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "GoldQuery":
            result.append({keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords})
    require(len(result) == 50, "Gold-query declarations changed; re-audit custody")
    return result


def corpus_bytes() -> bytes:
    # Model weights and data/processed are git-ignored; worktrees use the main checkout's copy.
    for base in (ROOT, MAIN_CHECKOUT):
        path = base / CORPUS_PATH
        if path.exists():
            return path.read_bytes()
    raise AssertionError("Frozen corpus file not found")


def ft_file(path: str) -> Path:
    for base in (ROOT, MAIN_CHECKOUT):
        candidate = base / path
        if candidate.exists():
            return candidate
    raise AssertionError(f"FT examples file not found: {path}")


def sources() -> tuple[dict[str, dict], set[str], set[str], list[tuple[str, str]]]:
    raw = corpus_bytes()
    require(hashlib.sha256(raw).hexdigest() == CORPUS_SHA256, "Corpus hash changed")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    corpus = {row["id"]: row for row in rows}
    require(len(corpus) == len(rows), "Duplicate corpus IDs")
    deny = json.loads(FORBIDDEN_PATH.read_text(encoding="utf-8"))
    forbidden = set(deny["ids"])
    require(len(forbidden) == len(deny["ids"]) == deny["count"], "Forbidden-list count mismatch")
    require(forbidden <= corpus.keys(), "Unknown forbidden-list IDs")
    prior = []
    for path in (V1_PATH, V2_PATH, V3_PATH, V4_PATH):
        fixture = load_json(path)
        forbidden.update(evidence_ids(fixture))
        prior.extend((case["qid"], case["query"]) for case in fixture["cases"])
    # Development evidence/answers are not inspected: only question texts.
    prior.extend((case["qid"], case["query"]) for case in load_json(DEV_PATH)["cases"])
    prior.extend((case["qid"], case["query"]) for case in load_json(DEV_CASES_PATH)["cases"])
    gold = gold_queries()
    prior.extend((case["qid"], case["query"]) for case in gold)
    for row in rows:
        if any(row["year"] in case["target_years"] and
               any(term.casefold() in row["text"].casefold() for term in case["must_contain_any"])
               for case in gold):
            forbidden.add(row["id"])
    negatives = [load_json(path) for path in HN_PATHS]
    # Legacy v2 IDs are left unchanged. Its explicitly migrated v3 case with
    # the same qid supplies paragraph IDs; never guess a legacy index mapping.
    v3_cases = {case["qid"]: case for case in negatives[1]["cases"]}
    for case in negatives[0]["cases"]:
        require(case["qid"] in v3_cases, "Legacy hard-negative case lacks its v3 companion")
        forbidden.update(evidence_ids(v3_cases[case["qid"]]))
    for fixture in negatives:
        forbidden.update(evidence_ids(fixture))
    # FT: ONLY answer-source ids are excluded (source_id, source_ids, evidence[*].source_id). The retrieved
    # context (passage_ids and context_passage_ids) is only recorded as ft_context_overlap, never excluded.
    contexts: set[str] = set()
    for path in FT_PATHS:
        with ft_file(path).open(encoding="utf-8") as stream:
            for line in stream:
                example = json.loads(line)
                ids = example.get("passage_ids", [])
                require(isinstance(ids, list), f"Invalid FT passage_ids in {path}")
                contexts.update(item for item in ids if isinstance(item, str))
                source = example.get("source_id")
                if source is not None:
                    require(isinstance(source, str), f"Invalid FT source_id in {path}")
                    forbidden.add(source)
                extra = example.get("source_ids", [])
                require(isinstance(extra, list), f"Invalid FT source_ids in {path}")
                forbidden.update(item for item in extra if isinstance(item, str))
                evidence = example.get("evidence", [])
                require(isinstance(evidence, list), f"Invalid FT evidence in {path}")
                forbidden.update(item["source_id"] for item in evidence
                                 if isinstance(item, dict) and isinstance(item.get("source_id"), str))
                context = example.get("context_passage_ids", [])
                require(isinstance(context, list), f"Invalid FT context IDs in {path}")
                contexts.update(context)
    return corpus, forbidden, contexts, prior


def present_parts() -> list[str]:
    return [part for part in ("a", "b") if PART_PATHS[part].exists()]


def part_evidence(part: str) -> set[str]:
    return evidence_ids(json.loads(PART_PATHS[part].read_bytes())["cases"])


def excluded_passage_ids() -> list[str]:
    """Every gold/relevant id of all present parts plus their same-letter +-1 neighbours, sorted."""
    found: set[str] = set()
    for part in present_parts():
        found |= part_evidence(part)
    result = set(found)
    for passage_id in found:
        result |= neighbours(passage_id)
    return sorted(result)


def expected_qids(part: str) -> list[str]:
    return [f"hv5{part}_{n:02d}" for n in range(1, PART_SIZE + 1)]


def check_part(part: str, context: tuple | None = None) -> tuple[str, dict]:
    raw = PART_PATHS[part].read_bytes()
    fixture = json.loads(raw)
    require(set(fixture) == set(load_json(V4_PATH)) | {"part"}, "Top-level keys differ from v4 (+part)")
    require(type(fixture["schema_version"]) is int and fixture["schema_version"] == 1, "Schema must be 1")
    require(fixture["frozen"] is True and fixture["created"] == "2026-10-10", "Wrong freeze metadata")
    require(fixture["part"] == part, "Wrong part marker")
    require(fixture["source_corpus"] == CORPUS_PATH and fixture["corpus_sha256"] == CORPUS_SHA256,
            "Wrong corpus metadata")
    purpose = fixture["purpose"].casefold()
    require(all(term in purpose for term in ("blind", "measured once per system", "never used for tuning")),
            "Missing blind custody purpose")
    require(SCOPE.casefold() in purpose, "Missing exact source-heldout scope")
    corpus, forbidden, contexts, prior = context or sources()
    barred = set(forbidden)
    for passage_id in forbidden:
        barred |= neighbours(passage_id)
    previous = [(qid, tokens(query)) for qid, query in prior]
    cases = fixture["cases"]
    require(isinstance(cases, list) and len(cases) == PART_SIZE, f"Need {PART_SIZE} cases")
    require([case["qid"] for case in cases] == expected_qids(part), f"Qids must be unique hv5{part}_01..60")
    type_counts = Counter(case["question_type"] for case in cases)
    require(set(type_counts) <= set(PART_TYPE_COUNTS), "Unknown question types")
    require(all(type_counts[kind] == count for kind, count in PART_EXACT_TYPE_COUNTS.items()),
            "Wrong opinion/follow_up/unanswerable counts")
    require(MIN_TEMPORAL <= type_counts["temporal_comparison"] <= MAX_TEMPORAL, "Wrong temporal_comparison count")
    require(type_counts["company"] >= MIN_COMPANY_FACT and type_counts["fact_number"] >= MIN_COMPANY_FACT,
            "company and fact_number need >= 11 each")
    require(type_counts["company"] + type_counts["fact_number"] + type_counts["temporal_comparison"]
            == FLEX_TYPE_TOTAL, "company + fact_number + temporal_comparison must be 32")
    require(len({case["query"].casefold() for case in cases}) == PART_SIZE, "Duplicate queries")
    answer_years: set[int] = set()
    year_cases: Counter[int] = Counter()
    numeric_qids: set[str] = set()
    multi_part_qids: set[str] = set()
    partials = []
    refusal_categories: Counter[str] = Counter()
    double_noncomparison = 0
    total_claims = 0
    all_text = "\n".join(row["text"] for row in corpus.values())
    corpus_years = {row["year"] for row in corpus.values()}
    for case in cases:
        qid, kind = case["qid"], case["question_type"]
        query = case["query"]
        require(isinstance(query, str) and bool(query.strip()), f"{qid}: empty query")
        require(not any(phrase in query.casefold() for phrase in PASSAGE_WORDING), f"{qid}: passage wording")
        note = case.get("verification_note", "")
        require(isinstance(note, str) and bool(note.strip()) and "\n" not in note, f"{qid}: invalid one-line verification")
        require(type(case.get("ft_context_overlap")) is bool, f"{qid}: context-overlap flag missing")
        years = case["years"]
        require(isinstance(years, list) and len(set(years)) == len(years) and
                all(type(year) is int for year in years), f"{qid}: invalid years")
        require(all(year % 2 == PART_PARITY[part] for year in years), f"{qid}: letter year outside part {part} parity")
        require(case["accept"]["min_claim_coverage"] == 1.0, f"{qid}: require all claims")
        require(isinstance(case["reject"]["forbidden_terms"], list), f"{qid}: invalid reject terms")
        require(isinstance(case["gold_claims"], list), f"{qid}: invalid claims")
        for field in ("gold_passage_ids", "relevant_ids"):
            ids = case[field]
            require(isinstance(ids, list) and len(set(ids)) == len(ids), f"{qid}: invalid {field}")
            require(set(ids) <= corpus.keys(), f"{qid}: unknown {field}")
            require(not set(ids) & barred, f"{qid}: {field} is forbidden or adjacent to an earlier evidence passage")
            require(all(corpus[pid]["year"] in years for pid in ids),
                    f"{qid}: evidence/year mismatch or wrong parity")
        gold_ids = set(case["gold_passage_ids"])
        require(gold_ids <= set(case["relevant_ids"]), f"{qid}: missing relevant gold IDs")
        require(case["ft_context_overlap"] == bool(gold_ids & contexts), f"{qid}: incorrect FT context flag")
        current = tokens(query)
        for previous_qid, earlier in previous:
            require(jaccard(current, earlier) < 0.5, f"{qid}: question overlaps {previous_qid}")
        query_tokens = token_list(query)
        for pid in gold_ids:
            passage_tokens = token_list(corpus[pid]["text"])
            grams = {tuple(passage_tokens[i:i + COPY_NGRAM]) for i in range(len(passage_tokens) - COPY_NGRAM + 1)}
            require(not any(tuple(query_tokens[i:i + COPY_NGRAM]) in grams
                            for i in range(len(query_tokens) - COPY_NGRAM + 1)), f"{qid}: query copies a passage")
        if kind == "unanswerable":
            category = case.get("refusal_category")
            require(category in REFUSAL_CATEGORIES, f"{qid}: wrong refusal category")
            refusal_categories[category] += 1
            require(case.get("answerable") is False and years == [], f"{qid}: incorrect refusal shape")
            require(case["gold_passage_ids"] == case["relevant_ids"] == case["gold_claims"] == [], f"{qid}: invented evidence")
            require(case["expected_answer"] == REFUSAL and case["accept"].get("require_exact_refusal") is True and
                    case["accept"].get("require_valid_citation") is False, f"{qid}: wrong exact-refusal acceptance")
            patterns = case.get("absence_search_patterns")
            require(isinstance(patterns, list) and bool(patterns) and all(isinstance(p, str) and p for p in patterns),
                    f"{qid}: unaudited absence searches")
            for pattern in patterns:
                require(re.search(pattern, all_text, re.I) is None, f"{qid}: absence search has corpus support")
            require(any(re.search(pattern, query, re.I) for pattern in patterns),
                    f"{qid}: no absence pattern matches the query's unsupported subject")
            require("history" not in case and "expected_coverage_note" not in case and "multi_part" not in case,
                    f"{qid}: invalid refusal additions")
            if category == "future_absent_year":
                named = {int(m.group(1)) for m in (YEAR_PATTERN.fullmatch(p) for p in patterns) if m}
                require(bool(named) and not named & corpus_years, f"{qid}: future year pattern must name an absent year")
            if category == "both_sides_unsupported":
                sides = case.get("unsupported_sides", [])
                require(len(sides) == 2 and len({s.casefold() for s in sides}) == 2 and
                        all(side.casefold() in query.casefold() for side in sides),
                        f"{qid}: must identify both unsupported sides")
                side_patterns = case.get("side_patterns")
                require(isinstance(side_patterns, dict) and set(side_patterns) == set(sides), f"{qid}: side patterns")
                require(all(re.search(side_patterns[s], all_text, re.I) is None and
                            re.search(side_patterns[s], query, re.I) for s in sides),
                        f"{qid}: one comparison side has corpus support or is not in the query")
            else:
                require("unsupported_sides" not in case and "side_patterns" not in case, f"{qid}: unexpected sides")
            continue
        require(case.get("answerable") is True and bool(years) and bool(gold_ids) and bool(case["gold_claims"]),
                f"{qid}: missing answerable evidence")
        require(case["accept"].get("require_valid_citation") is True and
                not case["accept"].get("require_exact_refusal", False), f"{qid}: wrong answer acceptance")
        require("expected_answer" not in case and "refusal_category" not in case, f"{qid}: answerable refusal metadata")
        supported_years = {corpus[pid]["year"] for pid in gold_ids}
        answer_years.update(supported_years)
        year_cases.update(supported_years)
        if kind == "temporal_comparison":
            require(len(years) == 2 and all(str(year) in query for year in years),
                    f"{qid}: comparison needs exactly two named letter years")
            if "expected_coverage_note" in case:
                partials.append(qid)
                missing = set(years) - supported_years
                require(len(missing) == 1 and len(supported_years) == 1, f"{qid}: partial needs exactly one supported side")
                missing_year = missing.pop()
                coverage = case["expected_coverage_note"]
                require(isinstance(coverage, str) and str(missing_year) in coverage and
                        "missing" in coverage.casefold() and "\n" not in coverage, f"{qid}: missing-period coverage note")
                require(missing_year not in corpus_years, f"{qid}: absent side must truly be absent")
                require(re.search(rf"\b{missing_year}\b", all_text) is None, f"{qid}: missing period has corpus text")
            else:
                require(supported_years == set(years), f"{qid}: full comparison missing one side")
        else:
            require("expected_coverage_note" not in case, f"{qid}: partial is comparison-only")
            require(len(years) == 1 or (kind in {"fact_number", "company"} and len(years) == 2), f"{qid}: wrong year count")
            double_noncomparison += len(years) == 2
            require(supported_years == set(years), f"{qid}: missing evidence for declared period")
        assigned_ids = set()
        claim_term_sets = []
        for claim in case["gold_claims"]:
            require(isinstance(claim["claim"], str) and bool(claim["claim"].strip()), f"{qid}: empty claim")
            terms, ids = claim["required_terms"], claim["gold_passage_ids"]
            require(isinstance(terms, list) and 2 <= len(terms) <= 3 and
                    all(isinstance(term, str) and bool(term.strip()) for term in terms) and
                    len({term.casefold() for term in terms}) == len(terms), f"{qid}: invalid 2-3 terms")
            require(isinstance(ids, list) and bool(ids) and len(set(ids)) == len(ids) and set(ids) <= gold_ids,
                    f"{qid}: invalid claim IDs")
            assigned_ids.update(ids)
            claim_note = claim.get("verification_note", "")
            require(isinstance(claim_note, str) and bool(claim_note.strip()) and "\n" not in claim_note and
                    "unit=" in claim_note and "definition=" in claim_note and
                    all(str(corpus[pid]["year"]) in claim_note for pid in ids),
                    f"{qid}: claim needs one-line year/unit/definition verification")
            for term in terms:
                require(term.casefold() in claim["claim"].casefold(), f"{qid}: term absent from claim")
                require(all(term.casefold() in corpus[pid]["text"].casefold() for pid in ids),
                        f"{qid}: term not verbatim in evidence")
                if re.search(r"\d", term):
                    numeric_qids.add(qid)
            claim_term_sets.append(frozenset(term.casefold() for term in terms))
        total_claims += len(case["gold_claims"])
        require(assigned_ids == gold_ids, f"{qid}: unassigned gold evidence")
        require(case.get("multi_part") is True and len(case["gold_claims"]) >= 2 and
                len(set(claim_term_sets)) == len(claim_term_sets) and
                len({claim["claim"] for claim in case["gold_claims"]}) == len(case["gold_claims"]),
                f"{qid}: every answerable case needs multi_part true and two or more distinct gold claims")
        multi_part_qids.add(qid)
        if kind == "follow_up":
            history = case.get("history")
            require(isinstance(history, list) and len(history) == 2 and
                    [message.get("role") for message in history] == ["user", "assistant"], f"{qid}: wrong follow-up history")
            require(all(set(message) == {"role", "content"} and isinstance(message["content"], str) and
                        bool(message["content"].strip()) for message in history), f"{qid}: invalid history")
            history_text = " ".join(message["content"] for message in history).casefold()
            require(all(not all(term.casefold() in history_text for term in claim["required_terms"])
                        for claim in case["gold_claims"]), f"{qid}: history reveals scored claim")
        else:
            require("history" not in case, f"{qid}: history on non-follow-up")
    require(refusal_categories["out_of_domain"] == 2 and sum(refusal_categories.values()) == 8 and
            all(refusal_categories[c] >= 1 for c in REQUIRED_FINANCE_CATEGORIES), "Wrong refusal category mix")
    require(len(partials) == 1, "Need exactly one answerable one-side-absent comparison")
    require(double_noncomparison <= MAX_DOUBLE_NONCOMPARISON, "Too many two-year fact/company cases")
    require(len(multi_part_qids) == PART_ANSWERABLE, f"All {PART_ANSWERABLE} answerable cases must be multi_part")
    require(total_claims >= MIN_PART_CLAIMS, f"Need at least {MIN_PART_CLAIMS} gold claims")
    require(len(numeric_qids) >= MIN_NUMERIC, f"Need at least {MIN_NUMERIC} answerable cases with exact numeric terms")
    require(len(answer_years) >= MIN_YEARS, f"Need at least {MIN_YEARS} distinct supported letter years")
    require(max(year_cases.values()) <= MAX_CASES_PER_YEAR, f"A letter year is shared by more than {MAX_CASES_PER_YEAR} cases")
    return hashlib.sha256(raw).hexdigest(), {"claims": total_claims, "answerable": len(multi_part_qids),
                                             "queries": [(case["qid"], case["query"]) for case in cases]}


def check_all() -> str:
    require(all(PART_PATHS[part].exists() for part in ("a", "b")), "Both part files must be present")
    context = sources()
    digests, stats = {}, {}
    for part in ("a", "b"):
        digests[part], stats[part] = check_part(part, context)
    ids_a, ids_b = part_evidence("a"), part_evidence("b")
    barred_a = set(ids_a)
    for passage_id in ids_a:
        barred_a |= neighbours(passage_id)
    require(not ids_b & barred_a, "Parts share passage ids or neighbours")
    queries_a = [(qid, tokens(query)) for qid, query in stats["a"]["queries"]]
    for qid_b, query_b in stats["b"]["queries"]:
        current = tokens(query_b)
        for qid_a, earlier in queries_a:
            require(jaccard(current, earlier) < 0.5, f"{qid_b}: question overlaps {qid_a}")
    require(stats["a"]["answerable"] + stats["b"]["answerable"] >= MIN_TOTAL_ANSWERABLE, "Too few answerable cases")
    require(stats["a"]["claims"] + stats["b"]["claims"] >= MIN_TOTAL_CLAIMS, "Too few gold claims")
    return f"OK n=120 sha256_a={digests['a']} sha256_b={digests['b']}"


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--excluded-passage-ids" in args:
        print(json.dumps(excluded_passage_ids()))
    elif "--all" in args:
        print(check_all())
    elif "--part" in args and args.index("--part") + 1 < len(args) and args[args.index("--part") + 1] in PART_PATHS:
        selected = args[args.index("--part") + 1]
        print(f"OK part={selected} n={PART_SIZE} sha256={check_part(selected)[0]}")
    else:
        raise SystemExit("usage: check_heldout_v5.py --part a|b | --all | --excluded-passage-ids")
