"""Validate blind heldout_v2 custody and fixtures without loading an answer system."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / "data/evaluation/heldout_v2/answer_benchmark_heldout_v2.json"
CORPUS_RELATIVE = "data/processed/chunks_v3_paragraph.jsonl"
CORPUS_SHA256 = "8263f08958729a77febf0922e7bbc4161bedd5fdb3e9a8b3fc9a64ab681ab5f0"
DEFAULT_FORBIDDEN = Path(
    "/Users/necatifurkancolak/AI-Workplace/Artifacts/BuffettRAG/round3/"
    "v2_forbidden_passages.json"
)
REFERENCE_PATHS = (
    ROOT / "data/evaluation/heldout_v1/answer_benchmark_heldout_v1.json",
    ROOT / "data/evaluation/answer_quality_program/answer_benchmark_v3.json",
)
TYPE_COUNTS = {
    "opinion": 10,
    "company": 9,
    "fact_number": 8,
    "temporal_comparison": 5,
    "follow_up": 4,
    "unanswerable": 4,
}
REFUSAL = "The retrieved passages do not provide enough evidence to answer this question."
ABSENCE_SEARCHES = {
    "hv37": ("crispr", "cas9", "gene editing"),
    "hv38": ("sudoku", "sudoku puzzle"),
    "hv39": ("exoplanet", "trappist", "exoplanet atmosphere"),
    "hv40": ("pickleball", "pickleball paddle"),
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def require(condition: bool, message: str) -> None:
    """Assertions remain active even if the interpreter is invoked with -O."""
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tokens(text: str) -> set[str]:
    """Lowercase alphanumeric token sets; keep years and do not remove stopwords."""
    return set(TOKEN_RE.findall(text.casefold()))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def check(forbidden_path: Path) -> str:
    raw_cases = CASES_PATH.read_bytes()
    fixture = json.loads(raw_cases)
    require(set(fixture) == set(load_json(REFERENCE_PATHS[0])), "Top-level keys differ from v1")
    require(type(fixture["schema_version"]) is int and fixture["schema_version"] == 1,
            "schema_version must be 1")
    require(fixture["frozen"] is True, "Fixture must be frozen")
    require(fixture["created"] == "2026-10-04", "Incorrect creation date")
    require(fixture["source_corpus"] == CORPUS_RELATIVE, "Unexpected source corpus")
    require(fixture["corpus_sha256"] == CORPUS_SHA256, "Incorrect declared corpus hash")
    purpose = fixture["purpose"].casefold()
    require(all(term in purpose for term in ("blind", "once per system", "never", "tuning")),
            "Purpose must state blind, once-per-system, never-for-tuning custody")

    raw_corpus = (ROOT / CORPUS_RELATIVE).read_bytes()
    require(hashlib.sha256(raw_corpus).hexdigest() == CORPUS_SHA256,
            "Actual corpus hash does not match the frozen corpus")
    rows = [json.loads(line) for line in raw_corpus.decode("utf-8").splitlines()]
    corpus = {row["id"]: row for row in rows}
    require(len(corpus) == len(rows), "Duplicate passage IDs in corpus")

    forbidden_fixture = load_json(forbidden_path)
    forbidden = set(forbidden_fixture["ids"])
    require(len(forbidden) == len(forbidden_fixture["ids"]) == forbidden_fixture["count"],
            "Forbidden-list count or uniqueness mismatch")
    require(forbidden <= corpus.keys(), "Forbidden list contains IDs absent from corpus")

    cases = fixture["cases"]
    require(isinstance(cases, list) and len(cases) == 40, "Exactly 40 cases are required")
    qids = [case["qid"] for case in cases]
    require(len(set(qids)) == 40, "Duplicate qids")
    require(qids == [f"hv{number:02d}" for number in range(1, 41)],
            "Qids must be ordered hv01..hv40")
    require(Counter(case["question_type"] for case in cases) == Counter(TYPE_COUNTS),
            "Incorrect question-type counts")
    require(len({case["query"] for case in cases}) == 40, "Duplicate queries")

    reference_queries = [
        (str(path.relative_to(ROOT)), case["qid"], tokens(case["query"]))
        for path in REFERENCE_PATHS
        for case in load_json(path)["cases"]
    ]
    require(bool(reference_queries), "No reference queries loaded")
    letter_years: set[int] = set()
    unanswerable_qids: set[str] = set()
    texts_lower = [row["text"].casefold() for row in rows]

    for case in cases:
        qid = case["qid"]
        kind = case["question_type"]
        require(isinstance(case["query"], str) and bool(case["query"].strip()),
                f"{qid}: empty query")
        require(isinstance(case["verification_note"], str) and bool(case["verification_note"].strip()),
                f"{qid}: missing verification note")
        require(isinstance(case["years"], list) and len(set(case["years"])) == len(case["years"]),
                f"{qid}: invalid or duplicate letter years")
        require(all(type(year) is int and 1977 <= year <= 2024 for year in case["years"]),
                f"{qid}: letter year outside 1977-2024")
        require(case["accept"]["min_claim_coverage"] == 1.0,
                f"{qid}: every gold claim must be required")
        require(isinstance(case["reject"]["forbidden_terms"], list),
                f"{qid}: invalid reject criteria")
        require(isinstance(case["gold_claims"], list), f"{qid}: invalid claims")
        for field in ("gold_passage_ids", "relevant_ids"):
            ids = case[field]
            require(isinstance(ids, list) and len(set(ids)) == len(ids),
                    f"{qid}: invalid or duplicate {field}")
            require(set(ids) <= corpus.keys(), f"{qid}: nonexistent {field}")
            require(not set(ids) & forbidden, f"{qid}: forbidden {field}")
            require(all(corpus[pid]["year"] in case["years"] for pid in ids),
                    f"{qid}: {field} do not match declared letter years")
        require(set(case["gold_passage_ids"]) <= set(case["relevant_ids"]),
                f"{qid}: relevant IDs must include all gold passages")

        query_tokens = tokens(case["query"])
        require(bool(query_tokens), f"{qid}: no query tokens")
        for path, reference_qid, reference_tokens in reference_queries:
            similarity = jaccard(query_tokens, reference_tokens)
            require(similarity < 0.5,
                    f"{qid}: Jaccard={similarity:.6f} vs {path}:{reference_qid}, must be < 0.5")

        if kind == "unanswerable":
            unanswerable_qids.add(qid)
            require(case.get("answerable") is False, f"{qid}: missing unanswerable flag")
            require(case["years"] == [], f"{qid}: unanswerable case has letter years")
            require(case["gold_passage_ids"] == case["relevant_ids"] == case["gold_claims"] == [],
                    f"{qid}: unanswerable case invents evidence")
            require(case["expected_answer"] == REFUSAL, f"{qid}: incorrect exact refusal")
            require(case["accept"].get("require_exact_refusal") is True,
                    f"{qid}: exact refusal must be required")
            require(case["accept"].get("require_valid_citation") is False,
                    f"{qid}: refusal must not require citations")
            require(qid in ABSENCE_SEARCHES, f"{qid}: no reproducible absence searches")
            for needle in ABSENCE_SEARCHES[qid]:
                matches = sum(needle.casefold() in text for text in texts_lower)
                require(matches == 0, f"{qid}: {needle!r} has {matches} corpus matches")
                require(needle in case["verification_note"].casefold(),
                        f"{qid}: absence search missing from verification note")
            require("history" not in case, f"{qid}: unexpected refusal history")
            continue

        require(case.get("answerable", True) is True, f"{qid}: inconsistent answerability")
        require(bool(case["years"]) and bool(case["gold_passage_ids"]) and bool(case["gold_claims"]),
                f"{qid}: answerable case lacks evidence or years")
        require(case["accept"].get("require_valid_citation") is True,
                f"{qid}: answerable claims require valid citations")
        require(not case["accept"].get("require_exact_refusal", False),
                f"{qid}: answerable case requires refusal")
        require("expected_answer" not in case, f"{qid}: unexpected fixed answer")
        letter_years.update(case["years"])
        require(len(case["years"]) >= 2 if kind == "temporal_comparison" else len(case["years"]) == 1,
                f"{qid}: incorrect number of letter years for type")

        for claim in case["gold_claims"]:
            require(isinstance(claim["claim"], str) and bool(claim["claim"].strip()),
                    f"{qid}: empty claim")
            terms = claim["required_terms"]
            require(isinstance(terms, list) and 2 <= len(terms) <= 3,
                    f"{qid}: every claim needs 2-3 required terms")
            require(all(isinstance(term, str) and bool(term.strip()) for term in terms),
                    f"{qid}: empty required term")
            require(len({term.casefold() for term in terms}) == len(terms),
                    f"{qid}: duplicate required terms")
            ids = claim["gold_passage_ids"]
            require(isinstance(ids, list) and bool(ids) and len(set(ids)) == len(ids),
                    f"{qid}: invalid claim-level gold IDs")
            require(set(ids) <= set(case["gold_passage_ids"]),
                    f"{qid}: claim IDs must be assigned case gold IDs")
            require(set(ids) <= corpus.keys() and not set(ids) & forbidden,
                    f"{qid}: nonexistent or forbidden claim-level IDs")
            for term in terms:
                require(term.casefold() in claim["claim"].casefold(),
                        f"{qid}: {term!r} absent from claim")
                for pid in ids:
                    require(term.casefold() in corpus[pid]["text"].casefold(),
                            f"{qid}: {term!r} not copied verbatim from {pid}")

        if kind == "follow_up":
            history = case.get("history")
            require(isinstance(history, list) and len(history) == 2,
                    f"{qid}: follow-up needs the v1 two-message history format")
            require([message.get("role") for message in history] == ["user", "assistant"],
                    f"{qid}: history must contain a user then assistant turn")
            require(all(set(message) == {"role", "content"} and
                        isinstance(message["content"], str) and bool(message["content"].strip())
                        for message in history), f"{qid}: invalid history message")
            history_text = " ".join(message["content"] for message in history).casefold()
            for claim in case["gold_claims"]:
                require(not all(term.casefold() in history_text for term in claim["required_terms"]),
                        f"{qid}: history reveals a complete scored claim")
        else:
            require("history" not in case, f"{qid}: only follow-ups should have history")

    require(unanswerable_qids == set(ABSENCE_SEARCHES), "Absence-search cases do not match refusals")
    require(len(letter_years) >= 25, "Answerable cases need at least 25 distinct letter years")
    target_years = {year for year in letter_years if 1998 <= year <= 2003 or 2008 <= year <= 2023}
    require(len(target_years) >= 10, "Need at least 10 distinct years from the requested eras")
    return hashlib.sha256(raw_cases).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forbidden-passages", type=Path, default=DEFAULT_FORBIDDEN,
                        help="Read-only round-3 forbidden passage list")
    args = parser.parse_args()
    print(f"OK sha256={check(args.forbidden_passages)}")


if __name__ == "__main__":
    main()
