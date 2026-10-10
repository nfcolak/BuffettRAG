"""Strict-scorer rules a-d (polarity, merged neighbours, year label, alias): one accepted and one rejected case each.

The rejected cases are the point: every rule is a narrow relaxation and must not loosen anything else.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation.answer_benchmark import _best_clauses, _polarity_and_numbers_agree
from src.evaluation.supported_claims import _definitions, _label_year, supported_claims_met

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "scorer_rules.json").read_text(encoding="utf-8"))


class _AcceptAll:
    """Verifier stub: isolates the lexical rules (polarity, numbers, terms) from deterministic support."""

    def verify(self, group_text, units):
        return [object()]


class _InSource:
    """Accepts a sentence only if all its words occur in one source sentence (support really matters)."""

    def verify(self, group_text, units):
        body = re.sub(r"\[\d+\]", "", group_text).lower()
        words = [w.strip(".,") for w in re.findall(r"[a-z0-9$.,%]+", body) if w.strip(".,")]
        return [object()] if any(all(w in u.text.lower() for w in words) for u in units) else []


def _case(claim, terms, gold=("g1",)):
    return {"qid": "q", "query": "x", "gold_passage_ids": list(gold),
            "gold_claims": [{"claim": claim, "required_terms": terms, "gold_passage_ids": list(gold)}],
            "accept": {"min_claim_coverage": 1.0, "require_valid_citation": True}, "reject": {}}


def _met(case, answer, hits, verifier=None, **kwargs):
    citations = [{"id": h["id"] if isinstance(h, dict) else h.id} for h in hits]
    rows = supported_claims_met(case, answer, citations, hits, verifier=verifier or _AcceptAll(), **kwargs)
    return rows[0]["met"]


def _hit(hid="g1", text="Source text.", year=1999, **extra):
    return {"id": hid, "text": text, "year": year, **extra}


# --- a. polarity -------------------------------------------------------------------------------

POSITIVE = _case("Float is profitable.", ["float", "profitable"])
NEGATIVE = _case("These tools do not create households.", ["create", "households"])


@pytest.mark.parametrize("answer", [
    "These tools don't create households [1].",
    "These tools don\u2019t create households [1].",       # typographic apostrophe
    "These tools do not create households [1].",
])
def test_a_contraction_matches_negative_claim(answer):
    assert _met(NEGATIVE, answer, [_hit()])


@pytest.mark.parametrize("claim,answer", [
    ("The fund will not buy stocks.", "The fund won't buy stocks [1]."),
    ("The fund can not buy stocks.", "The fund can\u2019t buy stocks [1]."),
])
def test_a_wont_and_cant_are_negations(claim, answer):
    assert _met(_case(claim, ["fund", "buy stocks"]), answer, [_hit()])


@pytest.mark.parametrize("answer", [
    "No matter the cycle, float is profitable [1].",
    "There is no doubt that float is profitable [1].",
    "Float is of no importance to critics but float is profitable [1].",
])
def test_a_idioms_are_not_predicate_negation(answer):
    assert _met(POSITIVE, answer, [_hit()])


@pytest.mark.parametrize("answer", [
    "Float is never profitable [1].",
    "Float isn't profitable [1].",
    "Float isn\u2019t profitable [1].",
    "Float is no longer profitable [1].",
    "Float cannot be profitable [1].",
])
def test_a_real_negation_still_rejects_positive_claim(answer):
    assert not _met(POSITIVE, answer, [_hit()])


def test_a_negative_claim_still_requires_negation():
    assert not _met(NEGATIVE, "These tools create households [1].", [_hit()])
    assert not _met(NEGATIVE, "These tools never matter, but they create households [1].", [_hit()])


def test_a_no_counts_only_directly_before_a_required_term():
    claim = _case("They had no insights into tech.", ["no insights", "tech"])
    assert _met(claim, "They had no insights into tech [1].", [_hit()])
    assert not _met(claim, "They had insights into tech [1].", [_hit()])
    claim = _case("Float had no cost.", ["float", "cost"])
    assert _met(claim, "Float had no cost [1].", [_hit()])
    assert not _met(claim, "Float had a cost [1].", [_hit()])
    assert not _met(_case("Float had a cost.", ["float", "cost"]), "Float had no cost [1].", [_hit()])


def test_a_polarity_is_judged_in_the_clause_with_most_terms():
    # Negation in another clause does not flip a positive claim ...
    assert _met(POSITIVE, "Insurance is not growing, but float is profitable [1].", [_hit()])
    assert _met(POSITIVE, "Float is profitable (although costs are not low) [1].", [_hit()])
    # ... but negation inside the term-bearing clause does, and ties never relax the check.
    assert not _met(POSITIVE, "Insurance is growing, but float is not profitable [1].", [_hit()])
    assert not _met(POSITIVE, "Float, while volatile, is not profitable [1].", [_hit()])
    # A thousands separator is not a clause boundary.
    claim = _case("Earnings were $21,904,000.", ["earnings", "$21,904,000"])
    assert _met(claim, "Earnings were $21,904,000 [1].", [_hit()])
    assert _best_clauses("Earnings were $21,904,000, up a lot", ["earnings", "$21,904,000"]) == [
        "Earnings were $21,904,000"]


def test_a_direct_helper():
    assert _polarity_and_numbers_agree("A do not B.", "A doesn't B.", ["a", "b"])
    assert not _polarity_and_numbers_agree("A is B.", "A is never B.", ["a", "b"])


# --- b. merged neighbour ids -------------------------------------------------------------------

B_CASE = _case("Float reached $10 billion.", ["float", "$10 billion"], gold=("g2",))
B_ANSWER = "Float reached $10 billion [1]."
B_OWN = {"g2": "Float reached $10 billion in 1999.", "a1": "Insurance is growing."}


def _merged_hit(**extra):
    return _hit("a1", "Insurance is growing. Float reached $10 billion in 1999.", **extra)


def test_b_sentence_in_gold_neighbour_own_text_is_met():
    hit = _merged_hit(merged_ids=["a1", "g2"])
    assert _met(B_CASE, B_ANSWER, [hit], _InSource(), chunk_texts=B_OWN)
    # SearchHit objects carry merged_ids in metadata; a callable lookup is accepted too.
    hit = SimpleNamespace(id="a1", text=hit["text"], year=1999, metadata={"merged_ids": ["a1", "g2"]})
    assert _met(B_CASE, B_ANSWER, [hit], _InSource(), chunk_texts=B_OWN.get)


def test_b_sentence_only_in_non_gold_merged_neighbour_is_rejected():
    own = {"g2": "Float reached $9 billion in 1999.", "a1": "Float reached $10 billion in 1999."}
    hit = _hit("a1", "Float reached $10 billion in 1999. Float reached $9 billion in 1999.",
               merged_ids=["a1", "g2"])
    assert not _met(B_CASE, B_ANSWER, [hit], _InSource(), chunk_texts=own)


def test_b_no_merged_ids_or_no_gold_in_them_or_no_own_text_is_rejected():
    assert not _met(B_CASE, B_ANSWER, [_merged_hit()], _InSource(), chunk_texts=B_OWN)              # default [hit id]
    assert not _met(B_CASE, B_ANSWER, [_merged_hit(merged_ids=["a1", "n3"])], _InSource(), chunk_texts=B_OWN)
    assert not _met(B_CASE, B_ANSWER, [_merged_hit(merged_ids=["a1", "g2"])], _InSource(), chunk_texts={})
    # The citation list still gates: a hit that was not cited by the answer's own citations is ignored.
    rows = supported_claims_met(B_CASE, B_ANSWER, [{"id": "other"}], [_merged_hit(merged_ids=["a1", "g2"])],
                                verifier=_InSource(), chunk_texts=B_OWN)
    assert not rows[0]["met"]


# --- c. year label -----------------------------------------------------------------------------

C_CASE = _case("Float rose to $5 billion in 2012.", ["float", "$5 billion"])


def _real_verifier():
    from src.generation.grounded.verify import DeterministicVerifier
    return DeterministicVerifier()


def test_c_label_year_counts_when_label_stripped_and_gold_hit_year_matches():
    hit = _hit(text="Float rose to $5 billion in 2012.", year=2012)
    assert _met(C_CASE, "In 2012: Float rose to $5 billion [1].", [hit], _real_verifier())
    assert _met(C_CASE, "In the 2012 letter: Float rose to $5 billion [1].", [hit], _real_verifier())


def test_c_label_year_does_not_count_for_a_different_hit_year():
    hit = _hit(text="Float rose to $5 billion in 2013.", year=2013)
    assert not _met(C_CASE, "In 2012: Float rose to $5 billion [1].", [hit], _real_verifier())
    assert not _met(C_CASE, "In the 2012 letter: Float rose to $5 billion [1].", [hit], _real_verifier())


def test_c_decade_and_range_labels_add_nothing():
    hit = _hit(text="Float rose to $5 billion in 2012.", year=2012)
    assert not _met(C_CASE, "In 2010s: Float rose to $5 billion [1].", [hit], _real_verifier())
    assert not _met(C_CASE, "In 2011\u20132013: Float rose to $5 billion [1].", [hit], _real_verifier())
    assert _label_year("In 2010s: Float rose [1].", [hit]) is None
    assert _label_year("In the 2011-2013 letters: Float rose [1].", [hit]) is None
    assert _label_year("In 2012: Float rose [1].", [hit]) == "2012"
    assert _label_year("In 2013: Float rose [1].", [hit]) is None      # label not stripped


def test_c_label_year_never_substitutes_for_other_numbers():
    hit = _hit(text="Float rose to $4 billion in 2012.", year=2012)
    assert not _met(C_CASE, "In 2012: Float rose to $4 billion [1].", [hit], _real_verifier())


# --- d. alias ----------------------------------------------------------------------------------

D_CASE = _case("Berkshire Hathaway Specialty Insurance wrote property policies.",
               ["berkshire hathaway specialty insurance", "property"])
D_DEFINITION = "Berkshire formed Berkshire Hathaway Specialty Insurance (\u201cBHSI\u201d) last June [1]. "


@pytest.mark.parametrize("definition", [
    D_DEFINITION,
    "Berkshire formed Berkshire Hathaway Specialty Insurance (BHSI) last June [1]. ",
    "Berkshire formed Berkshire Hathaway Specialty Insurance (\"BHSI\") last June [1]. ",
])
def test_d_abbreviation_defined_earlier_counts_as_full_name(definition):
    assert _met(D_CASE, definition + "BHSI wrote property policies [1].", [_hit()])


def test_d_undefined_or_later_defined_abbreviation_is_rejected():
    assert not _met(D_CASE, "BHSI wrote property policies [1].", [_hit()])
    assert not _met(D_CASE, "BHSI wrote property policies [1]. " + D_DEFINITION, [_hit()])
    assert not _met(D_CASE, "Berkshire formed a unit (\u201cBHSI\u201d) last June [1]. BHSI wrote property policies [1].",
                    [_hit()])


def test_d_alias_affects_required_terms_only_not_the_verifier():
    hit = _hit(text="BHSI wrote auto policies.")
    assert not _met(D_CASE, D_DEFINITION + "BHSI wrote property policies [1].", [hit], _InSource())
    assert _definitions(D_DEFINITION) == {"BHSI": "Berkshire Hathaway Specialty Insurance"}
    assert _definitions("That year, Berkshire purchased Precision Castparts (\u201cPCC\u201d) [1].") == {
        "PCC": "Precision Castparts"}


# --- polarity mutation on real dev claims -------------------------------------------------------

_AUX_RE = re.compile(r"\b(was|were|is|are|had|has|have|did|does|do)\b")


def _real_claims():
    for case in FIXTURE["cases"]:
        for index, claim in enumerate(case["gold_claims"]):
            gold = FIXTURE["gold_sentences"][f"{case['qid']}:{index}"]
            yield pytest.param(case, claim, gold, id=f"{case['qid']}:{index}")


def _mutations(sentence, terms):
    """Negated variants of a real supporting sentence, each negating inside the term-bearing clause."""
    lowered = sentence.lower()
    first = min(lowered.index(t) for t in terms)
    yield "never-before-term", sentence[:first] + "never " + sentence[first:]
    clause = _best_clauses(sentence, terms)[0]
    match = _AUX_RE.search(clause)
    if match:
        for label, replacement in (("not", match.group(0) + " not"),
                                   ("contraction", match.group(0) + "n't")):
            yield label, sentence.replace(clause, clause[:match.start()] + replacement + clause[match.end():], 1)


def test_fixture_is_five_real_dev_claims_from_two_cases():
    assert len(FIXTURE["cases"]) == 2
    assert sum(len(c["gold_claims"]) for c in FIXTURE["cases"]) == 5


@pytest.mark.parametrize("case,claim,gold", list(_real_claims()))
def test_polarity_mutation_of_real_gold_sentences_is_never_met(case, claim, gold):
    gid = gold["passage_id"]
    one = {**case, "gold_claims": [claim]}
    terms = [t.lower() for t in claim["required_terms"]]
    hits = [_hit(gid, FIXTURE["chunks"][gid]["text"], FIXTURE["chunks"][gid]["year"])]
    base = f"{gold['sentence']} [1]"
    assert _met(one, base, hits), "unmutated real sentence must be met, else the mutation test is vacuous"
    mutants = list(_mutations(gold["sentence"], terms))
    assert mutants
    for label, mutant in mutants:
        assert mutant != gold["sentence"]
        assert not _met(one, f"{mutant} [1]", hits), f"{label}: {mutant}"


def test_polarity_mutation_covers_aux_negation_for_most_claims():
    covered = 0
    for case in FIXTURE["cases"]:
        for index, claim in enumerate(case["gold_claims"]):
            sentence = FIXTURE["gold_sentences"][f"{case['qid']}:{index}"]["sentence"]
            labels = [lab for lab, _ in _mutations(sentence, [t.lower() for t in claim["required_terms"]])]
            covered += "contraction" in labels
    assert covered >= 3
