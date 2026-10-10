"""Scoped negation: a negation counts against a non-negated claim only when it scopes a required term.

The six W5T-06 sentences are verbatim source sentences that carry all required terms; the claim is worded
without the negation the source uses elsewhere in the clause. A negated claim still needs a negation.
"""
from __future__ import annotations

import pytest

from src.evaluation.answer_benchmark import _best_clauses, _has_negation, _polarity_and_numbers_agree

W5T_06 = [
    ("hv15", "R. C. Willey's stores were closed on Sunday because Bill and most managers were Mormons.",
     ["R. C. Willey", "Sunday", "Mormons"],
     "Bill and most of his managers are Mormons, and for this reason R. C. Willey\u2019s stores have never "
     "operated on Sunday."),
    ("hv20", "Berkshire purchased 90% of NFM.", ["90%", "NFM"],
     "Our evaluation of the integrity of Mrs. B and her family was demonstrated when we purchased 90% of the "
     "business: NFM had never had an audit and we did not request one."),
    ("hv31c1", "The GEICO purchase turned the holding into a wholly-owned operating business.",
     ["GEICO", "wholly-owned operating business"],
     "At the beginning of 1996, we acquired the half of GEICO we didn\u2019t already own, a cash transaction that "
     "changed our holding from a portfolio investment into a wholly-owned operating business."),
    ("hv31c2", "Berkshire acquired the remaining half at the beginning of 1996.", ["half", "beginning of 1996"],
     "At the beginning of 1996, we acquired the half of GEICO we didn\u2019t already own, a cash transaction that "
     "changed our holding from a portfolio investment into a wholly-owned operating business."),
    ("hv35", "Todd Combs brought the company to Buffett's attention.", ["Todd Combs", "attention"],
     "The PCC acquisition would not have happened without the input and assistance of our own Todd Combs, who "
     "brought the company to my attention a few years ago and went on to educate me about both the business."),
    ("adv02", "The industry combined ratio range is 100 - 120.", ["industry range", "100 - 120"],
     "Therefore, our yearly combined ratio on this business will almost never fall in the industry range of "
     "100 - 120, but will instead be close to either zero or 300%."),
]


@pytest.mark.parametrize("name,claim,terms,sentence", W5T_06, ids=[w[0] for w in W5T_06])
def test_w5t_06_sentences_are_judged_on_the_claims_own_terms(name, claim, terms, sentence):
    assert _polarity_and_numbers_agree(claim, sentence, terms)


@pytest.mark.parametrize("name,claim,terms,sentence", W5T_06, ids=[w[0] for w in W5T_06])
def test_negation_inserted_before_a_required_term_is_still_rejected(name, claim, terms, sentence):
    checked = 0
    judged = " ".join(_best_clauses(sentence, terms)).lower()  # polarity is judged in the clause(s) holding most terms
    for term in terms:
        index = sentence.lower().find(term.lower())
        assert index != -1
        if term.lower() not in judged:
            continue
        for negator in ("never ", "is not "):
            mutated = sentence[:index] + negator + sentence[index:]
            assert not _polarity_and_numbers_agree(claim, mutated, terms), (name, term, negator)
            checked += 1
    assert checked >= 2


def test_negated_verb_of_the_claim_is_still_rejected():
    claim = "The dividend increased to $704 million."
    terms = ["dividend", "$704 million"]
    assert _polarity_and_numbers_agree(claim, "By 2022, the dividend had increased to $704 million.", terms)
    assert not _polarity_and_numbers_agree(claim, "By 2022, the dividend had not increased to $704 million.", terms)
    assert not _polarity_and_numbers_agree(claim, "By 2022, the dividend hadn't increased to $704 million.", terms)


def test_scope_window_is_short():
    terms = ["float", "profitable"]
    assert not _polarity_and_numbers_agree("Float is profitable.", "Float is not profitable.", terms)
    assert not _polarity_and_numbers_agree("Float is profitable.", "Float has never been profitable.", terms)
    assert not _polarity_and_numbers_agree("Float is profitable.", "Float isn't profitable.", terms)
    assert _polarity_and_numbers_agree("Float is profitable.", "Float is profitable, but we did not buy more.", terms)


def test_negated_claim_still_needs_a_negation():
    terms = ["create", "households"]
    claim = "These tools do not create households."
    assert _polarity_and_numbers_agree(claim, "These tools don't create households.", terms)
    assert not _polarity_and_numbers_agree(claim, "These tools create households.", terms)


def test_has_negation_scoped_flag():
    sentence = "Our stores have never operated on Sunday"
    assert _has_negation(sentence, ["sunday"])  # default: any predicate negation
    assert not _has_negation(sentence, ["sunday"], scoped=True)
    assert _has_negation("Stores never operated", ["operated"], scoped=True)
