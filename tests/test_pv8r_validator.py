"""pv8-retrieval-validator group 2: VALIDATOR_SPLIT_V2 and VALIDATOR_VERBATIM_KEEP (call-time, default off)."""
import json

import pytest

from src.evaluation.claim_validator import _verbatim_span, split_claims, validate_and_filter_answer
from src.storage import SearchHit
from test_validator_fixture import FIXTURE, measure_fixture

FLAGS = ("VALIDATOR_SPLIT_V2", "VALIDATOR_VERBATIM_KEEP")


def _hits(*texts):
    return [SearchHit(f"p{i}", t, {}, 1.0) for i, t in enumerate(texts)]


def _run(sentence, *passages, flags=()):
    mp = pytest.MonkeyPatch()
    for flag in flags:
        mp.setenv(flag, "1")
    try:
        return validate_and_filter_answer(sentence, _hits(*passages))
    finally:
        mp.undo()


# (sentence, cited passage): stopword-only fragments after and/but ("on", "that") sink the whole verbatim sentence
# today; "down from 23,218" is a passes-anyway guard that a number-bearing span stays kept with the flags
PATTERNS = {
    "goes_on_and_on": ("The float, in a sense, goes on and on. [1]",
                       "The float, in a sense, goes on and on."),
    "on_and_on_and_on": ("The float goes on and on and on, as we expect. [1]",
                         "The float goes on and on and on, as we expect."),
    "it_is_that": ("It is what it is, and it is that. [1]", "It is what it is, and it is that."),
    "down_from": ("Employment at the textile mill fell, down from 23,218 at the peak. [1]",
                  "Employment at the textile mill fell, down from 23,218 at the peak."),
}
DROPPED_TODAY = ("goes_on_and_on", "on_and_on_and_on", "it_is_that")


def test_split_v2_default_unchanged(monkeypatch):
    monkeypatch.delenv("VALIDATOR_SPLIT_V2", raising=False)
    assert split_claims("The float goes on and on") == ["The float goes on", "on"]


def test_split_v2_keeps_short_fragment_joined(monkeypatch):
    monkeypatch.setenv("VALIDATOR_SPLIT_V2", "1")
    assert split_claims("The float goes on and on") == ["The float goes on and on"]
    assert split_claims("Buffett's huge and growing float") == ["Buffett's huge and growing float"]
    assert split_claims("It is what it is, and it is that") == ["It is what it is, and it is that"]
    # parts with >= 3 content tokens each still split
    assert split_claims("Revenue rose sharply this year and costs fell sharply this year") == [
        "Revenue rose sharply this year", "costs fell sharply this year"]


@pytest.mark.parametrize("name", DROPPED_TODAY)
def test_patterns_dropped_today(name):
    sentence, passage = PATTERNS[name]
    assert _run(sentence, passage).safe_answer == ""


@pytest.mark.parametrize("name", sorted(PATTERNS))
def test_patterns_kept_with_flags(name):
    sentence, passage = PATTERNS[name]
    result = _run(sentence, passage, flags=FLAGS)
    assert result.safe_answer == sentence and not result.blocked_claims


def test_split_v2_alone_fixes_goes_on_and_on():
    sentence, passage = PATTERNS["goes_on_and_on"]
    assert _run(sentence, passage).safe_answer == ""
    assert _run(sentence, passage, flags=("VALIDATOR_SPLIT_V2",)).safe_answer == sentence


def test_verbatim_keep_alone_keeps_span_of_cited_passage():
    sentence, passage = PATTERNS["it_is_that"]
    assert _run(sentence, passage, flags=("VALIDATOR_VERBATIM_KEEP",)).safe_answer == sentence


def test_verbatim_keep_normalises_quotes_dashes_whitespace_but_keeps_case():
    passage = "He wrote that  the plan\nwas ‘sound’ — and costly for owners."
    assert _verbatim_span("He wrote that the plan was 'sound' - and costly for owners. [1]", [passage]) is True
    assert _verbatim_span("he wrote that the plan was 'sound' - and costly for owners. [1]", [passage]) is False


SRC = "Operating earnings were $21 million last year, and the firm did not buy preferred shares at all."


@pytest.mark.parametrize("sentence,passages", [
    ("Operating earnings were $22 million last year, and the firm did not buy preferred shares at all. [1]", [SRC]),
    ("Operating earnings were $21 million last year, and the firm did buy preferred shares at all. [1]", [SRC]),
    ("Operating earnings were $21 million last year, and the firm did not buy preferred shares at all. [2]",
     [SRC, "Insurance float grew steadily during the decade."]),
    ("Operating earnings were $21 million last year, and the firm did not buy preferred shares at all. [1]",
     ["Insurance float grew steadily during the decade.", SRC]),
], ids=["changed_number", "added_negation", "wrong_passage", "uncited_passage"])
def test_negative_controls_still_dropped_with_both_flags(sentence, passages):
    assert _run(sentence, *passages, flags=FLAGS).safe_answer == ""


def test_five_token_span_is_not_a_verbatim_keep():
    # a short span must earn its keep through the normal claim checks, not the verbatim shortcut
    passage = "Operating earnings were 21 million last year."
    assert _verbatim_span("Operating earnings were 21 million", [passage]) is False
    assert _verbatim_span("Operating earnings were 21 million last", [passage]) is True


def test_off_flags_byte_identical_on_fixture(monkeypatch):
    for flag in FLAGS:
        monkeypatch.delenv(flag, raising=False)
    base = measure_fixture()
    for flag in FLAGS:
        monkeypatch.setenv(flag, "0")
    assert measure_fixture() == base


def test_flags_on_fixture_keeps_false_passes_zero(monkeypatch):
    for flag in FLAGS:
        monkeypatch.delenv(flag, raising=False)
    base = measure_fixture()
    for flag in FLAGS:
        monkeypatch.setenv(flag, "1")
    on = measure_fixture()
    assert on["sentences"] == base["sentences"] == len(json.loads(FIXTURE.read_text())["cases"])
    assert on["false_passes"] == 0
    assert on["false_blocks"] <= base["false_blocks"]
