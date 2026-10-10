"""Near-verbatim, correctly cited sentences the validator used to drop, and the guards that must still hold.

Fixture cases are real model sentences with the cited passage text copied verbatim (see the fixture's note).
"""
import json
import re
from pathlib import Path

import pytest

from src.evaluation.claim_validator import validate_and_filter_answer
from src.storage import SearchHit

CASES = json.loads((Path(__file__).parent / "fixtures" / "validator_fix.json").read_text(encoding="utf-8"))["cases"]
IDS = [case["id"] for case in CASES]
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
CITATION = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


def hits_of(case, passages=None):
    return [SearchHit(p["id"], p["text"], {"year": p["year"]}, 1.0) for p in (passages or case["passages"])]


def kept(sentence, hits):
    result = validate_and_filter_answer(sentence, hits)
    return result.safe_answer.strip() == sentence.strip() and not result.blocked_claims


def number_mutants(sentence):
    """One mutant per number in the sentence (citation markers excluded), each number shifted by 7."""
    markers = [m.span() for m in CITATION.finditer(sentence)]
    for match in NUMBER.finditer(sentence):
        if any(start <= match.start() < end for start, end in markers):
            continue
        raw = match.group(0)
        value = float(raw.replace(",", ""))
        shifted = f"{int(value) + 7:,}" if "," in raw else (str(int(value) + 7) if "." not in raw else f"{value + 7:.1f}")
        yield sentence[:match.start()] + shifted + sentence[match.end():]


def test_fixture_has_enough_real_cases():
    assert len(CASES) >= 8
    assert all(case["sentence"].strip() and case["passages"] for case in CASES)
    assert sum(bool(list(number_mutants(case["sentence"]))) for case in CASES) >= 8


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_correctly_cited_sentence_is_kept(case):
    result = validate_and_filter_answer(case["sentence"], hits_of(case))
    assert result.safe_answer == case["sentence"], result.blocked_claims


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_changed_number_is_dropped(case):
    for mutant in number_mutants(case["sentence"]):
        assert not kept(mutant, hits_of(case)), mutant


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_inserted_negation_is_dropped(case):
    old, new = case["negate"]
    mutant = case["sentence"].replace(old, new, 1)
    assert mutant != case["sentence"]
    assert not kept(mutant, hits_of(case)), mutant


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_swapped_entity_is_dropped(case):
    old, new = case["entity"]
    mutant = case["sentence"].replace(old, new, 1)
    assert mutant != case["sentence"]
    assert not kept(mutant, hits_of(case)), mutant


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_citation_to_another_passage_is_dropped(index):
    case, other = CASES[index], CASES[(index + 1) % len(CASES)]
    assert other["passages"][0]["id"] not in {p["id"] for p in case["passages"]}
    sentence = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "[1]", case["sentence"])
    assert not kept(sentence, hits_of(case, other["passages"][:1]))


def test_passage_year_only_licenses_that_year():
    case = next(c for c in CASES if c["id"].endswith("1983_p0028"))
    assert kept(case["sentence"], hits_of(case))
    other_year = [{**p, "year": 1984} for p in case["passages"]]
    assert not kept(case["sentence"], hits_of(case, other_year))
    no_year = [SearchHit(p["id"], p["text"], {}, 1.0) for p in case["passages"]]
    assert not kept(case["sentence"], no_year)
