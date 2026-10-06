"""Validate the frozen source-heldout fixture without loading a model or retriever."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / "data/evaluation/heldout_v3/answer_benchmark_heldout_v3.json"
CORPUS_PATH = "data/processed/chunks_v3_paragraph.jsonl"
CORPUS_SHA256 = "8263f08958729a77febf0922e7bbc4161bedd5fdb3e9a8b3fc9a64ab681ab5f0"
FORBIDDEN_PATH = Path("/Users/necatifurkancolak/AI-Workplace/Artifacts/BuffettRAG/round3/v2_forbidden_passages.json")
V1_PATH = "data/evaluation/heldout_v1/answer_benchmark_heldout_v1.json"
V2_PATH = "data/evaluation/heldout_v2/answer_benchmark_heldout_v2.json"
DEV_PATH = "data/evaluation/answer_quality_program/answer_benchmark_v3.json"
HN_PATHS = (
    "data/evaluation/answer_quality_program/hard_negatives_v2.json",
    "data/evaluation/answer_quality_program/hard_negatives_v3.json",
    "data/evaluation/hard_negatives_v4/hard_negatives_v4.json",
)
FT_PATHS = ("data/ft/examples.jsonl", "data/ft_v2/examples.jsonl", "data/ft_v3/examples.jsonl")
TYPE_COUNTS = {"opinion": 9, "company": 8, "fact_number": 8,
               "temporal_comparison": 5, "follow_up": 4, "unanswerable": 6}
REFUSAL = "The retrieved passages do not provide enough evidence to answer this question."
SCOPE = "source-heldout (FT answer passages excluded), not unseen-evidence (FT context overlap allowed)"
TOKEN_RE = re.compile(r"[a-z0-9]+")
# These patterns check both spellings/paraphrases of each absent concept, not
# merely the absence of a fabricated answer number. Context reviews are recorded
# in each refusal's verification_note, including real topic-word corpus hits.
ABSENCE_PATTERNS = {
    "hv3_35": ("out_of_domain", (r"sourdough", r"focaccia")),
    "hv3_36": ("out_of_domain", (r"bearded dragon", r"pogona")),
    "hv3_37": ("unsupported_figure", (r"debt[\s-]+service\s+coverage", r"\bdscr\b")),
    "hv3_38": ("future_absent_year", (r"\b2031\b",)),
    "hv3_39": ("false_premise", (r"spotify",)),
    "hv3_40": ("both_sides_unsupported", (r"customer\s+acquisition\s+cost", r"cost\s+(?:of|per)\s+(?:acquiring|acquisition)")),
}
REFUSAL_QUERY_ANCHORS = {
    "hv3_35": ("sourdough", "focaccia"),
    "hv3_36": ("bearded dragon",),
    "hv3_37": ("apple", "2019", "debt-service coverage"),
    "hv3_38": ("2031", "underwriting profit"),
    "hv3_39": ("spotify", "acquired"),
    "hv3_40": ("2023", "customer acquisition cost", "geico", "progressive"),
}
# In 2023, even the broader acquisition-cost paraphrases are absent. GEICO
# itself is mentioned, so absence of the entity name is NOT our refusal test.
COMPARISON_ABSENCE = (r"acquisition[\s-]+cost", r"acquir\w*.{0,40}cost", r"\bprogressive\b")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.casefold()))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


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


def sources() -> tuple[dict[str, dict], set[str], set[str], list[tuple[str, str]]]:
    raw = (ROOT / CORPUS_PATH).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CORPUS_SHA256, "Corpus hash changed")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    corpus = {row["id"]: row for row in rows}
    require(len(corpus) == len(rows), "Duplicate corpus IDs")
    deny = json.loads(FORBIDDEN_PATH.read_text(encoding="utf-8"))
    forbidden = set(deny["ids"])
    require(len(forbidden) == len(deny["ids"]) == deny["count"], "Forbidden-list count mismatch")
    require(forbidden <= corpus.keys(), "Unknown forbidden-list IDs")
    prior = []
    for path in (V1_PATH, V2_PATH):
        fixture = load_json(path)
        forbidden.update(evidence_ids(fixture))
        prior.extend((case["qid"], case["query"]) for case in fixture["cases"])
    # Development evidence/answers are not inspected: only question texts.
    prior.extend((case["qid"], case["query"]) for case in load_json(DEV_PATH)["cases"])
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
    contexts: set[str] = set()
    for path in FT_PATHS:
        with (ROOT / path).open(encoding="utf-8") as stream:
            for line in stream:
                example = json.loads(line)
                ids = example.get("passage_ids", [])
                require(isinstance(ids, list), f"Invalid FT passage_ids in {path}")
                forbidden.update(ids)
                source = example.get("source_id")
                if source is not None:
                    require(isinstance(source, str), f"Invalid FT source_id in {path}")
                    forbidden.add(source)
                context = example.get("context_passage_ids", [])
                require(isinstance(context, list), f"Invalid FT context IDs in {path}")
                contexts.update(context)
    return corpus, forbidden, contexts, prior


def check() -> str:
    raw = CASES_PATH.read_bytes()
    fixture = json.loads(raw)
    require(set(fixture) == set(load_json(V2_PATH)), "Top-level keys differ from v2")
    require(type(fixture["schema_version"]) is int and fixture["schema_version"] == 1, "Schema must be 1")
    require(fixture["frozen"] is True and fixture["created"] == "2026-10-05", "Wrong freeze metadata")
    require(fixture["source_corpus"] == CORPUS_PATH and fixture["corpus_sha256"] == CORPUS_SHA256,
            "Wrong corpus metadata")
    purpose = fixture["purpose"].casefold()
    require(all(term in purpose for term in ("blind", "measured once per system", "never used for tuning")),
            "Missing blind custody purpose")
    require(SCOPE.casefold() in purpose, "Missing exact source-heldout scope")
    corpus, forbidden, contexts, prior = sources()
    previous = [(qid, tokens(query)) for qid, query in prior]
    cases = fixture["cases"]
    require(isinstance(cases, list) and len(cases) == 40, "Need 40 cases")
    require([case["qid"] for case in cases] == [f"hv3_{n:02d}" for n in range(1, 41)], "Qids must be unique hv3_01..hv3_40")
    require(Counter(case["question_type"] for case in cases) == Counter(TYPE_COUNTS), "Wrong type counts")
    require(len({case["query"].casefold() for case in cases}) == 40, "Duplicate queries")
    answer_years: set[int] = set()
    numeric_qids: set[str] = set()
    partials = []
    refusals = set()
    double_noncomparison = 0
    all_text = "\n".join(row["text"] for row in corpus.values())
    for case in cases:
        qid, kind = case["qid"], case["question_type"]
        require(isinstance(case["query"], str) and bool(case["query"].strip()), f"{qid}: empty query")
        note = case.get("verification_note", "")
        require(isinstance(note, str) and bool(note.strip()) and "\n" not in note, f"{qid}: invalid one-line verification")
        require(type(case.get("ft_context_overlap")) is bool, f"{qid}: context-overlap flag missing")
        years = case["years"]
        require(isinstance(years, list) and len(set(years)) == len(years) and
                all(type(year) is int for year in years), f"{qid}: invalid years")
        require(case["accept"]["min_claim_coverage"] == 1.0, f"{qid}: require all claims")
        require(isinstance(case["reject"]["forbidden_terms"], list), f"{qid}: invalid reject terms")
        require(isinstance(case["gold_claims"], list), f"{qid}: invalid claims")
        for field in ("gold_passage_ids", "relevant_ids"):
            ids = case[field]
            require(isinstance(ids, list) and len(set(ids)) == len(ids), f"{qid}: invalid {field}")
            require(set(ids) <= corpus.keys() and not set(ids) & forbidden, f"{qid}: missing/forbidden {field}")
            require(all(corpus[pid]["year"] in years for pid in ids), f"{qid}: evidence/year mismatch")
        gold_ids = set(case["gold_passage_ids"])
        require(gold_ids <= set(case["relevant_ids"]), f"{qid}: missing relevant gold IDs")
        require(case["ft_context_overlap"] == bool(gold_ids & contexts), f"{qid}: incorrect FT context flag")
        current = tokens(case["query"])
        for previous_qid, earlier in previous:
            require(jaccard(current, earlier) < 0.5, f"{qid}: question overlaps {previous_qid}")
        if kind == "unanswerable":
            refusals.add(qid)
            require(qid in ABSENCE_PATTERNS, f"{qid}: no audited absence search")
            subtype, patterns = ABSENCE_PATTERNS[qid]
            require(case.get("refusal_category") == subtype, f"{qid}: wrong refusal category")
            require(all(anchor in case["query"].casefold() for anchor in REFUSAL_QUERY_ANCHORS[qid]),
                    f"{qid}: query does not match audited unsupported subject")
            require(case.get("answerable") is False and years == [], f"{qid}: incorrect refusal shape")
            require(case["gold_passage_ids"] == case["relevant_ids"] == case["gold_claims"] == [], f"{qid}: invented evidence")
            require(case["expected_answer"] == REFUSAL and case["accept"].get("require_exact_refusal") is True and
                    case["accept"].get("require_valid_citation") is False, f"{qid}: wrong exact-refusal acceptance")
            require(case.get("absence_search_patterns") == list(patterns), f"{qid}: unaudited absence searches")
            for pattern in patterns:
                require(re.search(pattern, all_text, re.I) is None, f"{qid}: absence search has corpus support")
            require("history" not in case and "expected_coverage_note" not in case, f"{qid}: invalid refusal additions")
            if subtype == "both_sides_unsupported":
                require(len(case.get("unsupported_sides", [])) == 2 and
                        len(set(case["unsupported_sides"])) == 2 and
                        all(side.casefold() in case["query"].casefold() for side in case["unsupported_sides"]),
                        f"{qid}: must identify both unsupported sides")
                period_text = "\n".join(row["text"] for row in corpus.values() if row["year"] == 2023)
                require(all(re.search(pattern, period_text, re.I | re.S) is None
                            for pattern in COMPARISON_ABSENCE), f"{qid}: one comparison side has period support")
            continue
        require(case.get("answerable") is True and bool(years) and bool(gold_ids) and bool(case["gold_claims"]), f"{qid}: missing answerable evidence")
        require(case["accept"].get("require_valid_citation") is True and
                not case["accept"].get("require_exact_refusal", False), f"{qid}: wrong answer acceptance")
        require("expected_answer" not in case and "refusal_category" not in case, f"{qid}: answerable refusal metadata")
        supported_years = {corpus[pid]["year"] for pid in gold_ids}
        answer_years.update(supported_years)
        if kind == "temporal_comparison":
            require(len(years) == 2 and all(str(year) in case["query"] for year in years),
                    f"{qid}: comparison needs exactly two named letter years")
            if "expected_coverage_note" in case:
                partials.append(qid)
                missing = set(years) - supported_years
                require(len(missing) == 1 and len(supported_years) == 1, f"{qid}: partial needs exactly one supported side")
                missing_year = missing.pop()
                coverage = case["expected_coverage_note"]
                require(isinstance(coverage, str) and str(missing_year) in coverage and
                        "missing" in coverage.casefold() and "\n" not in coverage, f"{qid}: missing-period coverage note")
                require(missing_year not in {row["year"] for row in corpus.values()}, f"{qid}: absent side must truly be absent")
                require(re.search(rf"\b{missing_year}\b", all_text) is None, f"{qid}: missing period has corpus text")
                require(str(missing_year) in case["query"], f"{qid}: missing period absent from query")
            else:
                require(supported_years == set(years), f"{qid}: full comparison missing one side")
        else:
            require("expected_coverage_note" not in case, f"{qid}: partial is comparison-only")
            require(len(years) == 1 or (kind in {"fact_number", "company"} and len(years) == 2), f"{qid}: wrong year count")
            double_noncomparison += len(years) == 2
            require(supported_years == set(years), f"{qid}: missing evidence for declared period")
        assigned_ids = set()
        for claim in case["gold_claims"]:
            require(isinstance(claim["claim"], str) and bool(claim["claim"].strip()), f"{qid}: empty claim")
            terms, ids = claim["required_terms"], claim["gold_passage_ids"]
            require(isinstance(terms, list) and 2 <= len(terms) <= 3 and
                    all(isinstance(term, str) and bool(term.strip()) for term in terms) and
                    len({term.casefold() for term in terms}) == len(terms), f"{qid}: invalid 2-3 terms")
            require(isinstance(ids, list) and bool(ids) and len(set(ids)) == len(ids) and set(ids) <= gold_ids, f"{qid}: invalid claim IDs")
            assigned_ids.update(ids)
            claim_note = claim.get("verification_note", "")
            require(isinstance(claim_note, str) and bool(claim_note.strip()) and "\n" not in claim_note and
                    "unit=" in claim_note and "definition=" in claim_note and
                    all(str(corpus[pid]["year"]) in claim_note for pid in ids), f"{qid}: claim needs one-line year/unit/definition verification")
            for term in terms:
                require(term.casefold() in claim["claim"].casefold(), f"{qid}: term absent from claim")
                require(all(term.casefold() in corpus[pid]["text"].casefold() for pid in ids), f"{qid}: term not verbatim in evidence")
                if re.search(r"\d", term):
                    numeric_qids.add(qid)
        require(assigned_ids == gold_ids, f"{qid}: unassigned gold evidence")
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
    require(refusals == set(ABSENCE_PATTERNS), "Wrong refusal distribution")
    require(len(partials) == 1, "Need exactly one answerable one-side-absent comparison")
    require(double_noncomparison <= 3, "Too many two-year fact/company cases")
    require(len(numeric_qids) >= 8, "Need at least eight answerable cases with exact numeric terms")
    require(len(answer_years) >= 25, "Need at least 25 distinct supported letter years")
    return hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    print(f"OK n=40 sha256={check()}")
