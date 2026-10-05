"""Grounded engine, template composer and deterministic verifier (fakes only, no models)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from src.evaluation.claim_validator import validate_and_filter_answer
from src.generation.grounded.decision import decide
from src.generation.grounded.engine import GroundedEngine
from src.generation.grounded.settings import GroundedSettings
from src.generation.grounded.template import TemplateComposer
from src.generation.grounded.types import (
    REFUSAL_LINE, ComposeRequest, ComposeResult, EvidencePlan, EvidenceUnit,
    GroupEvidence, Slot, SourceSpan,
)
from src.generation.grounded.verify import DeterministicVerifier

SETTINGS = GroundedSettings(COMPOSER="template")


def unit(eid: str, hit_index: int, year: int | None, text: str = "x", score: float = 0.9) -> EvidenceUnit:
    span = SourceSpan(hit_index, 0, len(text), text)
    return EvidenceUnit(eid=eid, hit_index=hit_index, passage_id=f"p{hit_index}", letter_year=year,
                        anchor=span, window=span, score=score)


def ev(index: int, text: str, hit_index: int = 0, year: int | None = 1985, *, anchor: bool = True) -> GroupEvidence:
    u = unit(f"u{hit_index}-{index}", hit_index, year, text)
    if not anchor:
        u.anchor = SourceSpan(hit_index, 1000, 1001, "")
    return GroupEvidence(local_index=index, text=text, unit=u, source_span=SourceSpan(hit_index, 0, len(text), text))


def make_hits(n: int):
    return [SimpleNamespace(id=f"h{i}", text=f"passage {i}", metadata={"year": 1980 + i, "source_file": f"l{i}.pdf"})
            for i in range(n)]


class FakeComposer:
    backend = "fake"

    def __init__(self, outputs=None, error_type=None, exc=None):
        self.outputs = outputs if outputs is not None else {}
        self.error_type = error_type
        self.exc = exc
        self.requests: list[ComposeRequest] = []

    def compose(self, req: ComposeRequest) -> ComposeResult:
        self.requests.append(req)
        if self.exc:
            raise self.exc
        text = self.outputs(req) if callable(self.outputs) else self.outputs
        return ComposeResult(text=text, raw=text, backend="fake", model_id="fake-1", error_type=self.error_type)

    def raw_generate(self, prompt: str, max_tokens: int) -> str:
        return ""


def build_engine(groups: dict[str, list[GroupEvidence]], composer, *, decision="answer", slots=None,
                 query_calls=None):
    """groups: name -> evidence. Fake build_plan returns a plan with matching units."""
    units = [e.unit for g in groups.values() for e in g]
    if slots is None:
        slots = [Slot(name="all", kind="general")]

    def fake_plan(original_query, hits, *, history=(), scorer, settings):
        if query_calls is not None:
            query_calls.append((original_query, len(hits), list(history)))
        return EvidencePlan(original_query, original_query, slots=slots, units=units,
                            groups={g: [e.unit.eid for e in es] for g, es in groups.items()},
                            decision=decision)

    def fake_group(plan, group, hits):
        return list(groups.get(group, []))

    resources = SimpleNamespace(scorer=object(), locks={"composer": threading.Lock()},
                                get_composer=lambda kind: composer, load=lambda: None)
    return GroundedEngine(resources, SETTINGS, composer_kind="template", build_plan=fake_plan,
                          group_evidence=fake_group)


S1 = "Berkshire's book value per share rose 23.8% in 1985."
S2 = "Insurance float grew to $3.2 billion at year end."


def test_local_marker_maps_to_client_hit_index_round_trip():
    evidence = [ev(0, S1, hit_index=2), ev(1, S2, hit_index=4)]
    composer = FakeComposer(f"{S1} [1] {S2} [2]")
    hits = make_hits(5)
    out = build_engine({"all": evidence}, composer).answer("q", hits)
    assert out.answer == f"{S1} [3] {S2} [5]"
    assert [c["passage_indices"] for c in out.citations] == [[2], [4]]
    assert [c["passage_ids"] for c in out.citations] == [["h2"], ["h4"]]
    assert all(not c["invalid_numbers"] for c in out.citations)
    assert out.groups[0].fallback_reason is None
    assert out.composer == "fake" and out.model_id == "fake-1"


def test_merged_marker_uses_unique_global_hits():
    evidence = [ev(0, S1, hit_index=3), ev(1, S2, hit_index=3)]
    out = build_engine({"all": evidence}, FakeComposer(f"{S1} [1,2]")).answer("q", make_hits(5))
    assert out.answer.endswith("[4]")


def test_unknown_and_malformed_markers_dropped_and_counted():
    evidence = [ev(0, S1)]
    verifier = DeterministicVerifier()
    outcome = verifier.verify_detailed(f"{S1} [9]\n{S1} [1,7]\n{S1} [E1]\n{S1}\n{S1} [0]\n{S1} [1", evidence)
    assert outcome.sentences == []
    assert outcome.dropped == {"unknown_marker": 3, "malformed_marker": 2, "unmarked": 1}
    ok = verifier.verify(f"{S1} [1]", evidence)
    assert [s.local_indexes for s in ok] == [[0]] and ok[0].unit_ids == [evidence[0].unit.eid]


def test_template_budget_cut_at_sentence_boundaries_and_cited():
    texts = ["Alpha beta gamma delta one.", "Epsilon zeta eta theta two.", "Iota kappa lambda mu three."]
    evidence = [ev(i, t, hit_index=i) for i, t in enumerate(texts)]
    composer = TemplateComposer()
    small = composer.compose(ComposeRequest(system="s", question="q", evidence=evidence, max_tokens=10)).text
    assert len(small) <= 40 and small == "Alpha beta gamma delta one. [1]"
    full = composer.compose(ComposeRequest(system="s", question="q", evidence=evidence, max_tokens=200)).text
    assert full == "Alpha beta gamma delta one. [1] Epsilon zeta eta theta two. [2] Iota kappa lambda mu three. [3]"
    # pure function: same request, same output; no model needed for raw_generate failure semantics
    assert composer.compose(ComposeRequest(system="s", question="q", evidence=evidence, max_tokens=200)).text == full
    with pytest.raises(RuntimeError):
        composer.raw_generate("p", 5)


def test_template_prefers_anchor_sentences_and_skips_bracket_syntax():
    evidence = [ev(0, "Context sentence about nothing.", anchor=False), ev(1, S1, hit_index=1),
                ev(2, "See note [3] for details.", hit_index=2)]
    text = TemplateComposer().compose(ComposeRequest(system="s", question="q", evidence=evidence)).text
    # the anchor comes first; the neighbour only follows because sentences and budget remain
    assert text == f"{S1} [2] Context sentence about nothing. [1]"


def test_template_renders_anchors_best_score_first_then_neighbours():
    texts = ["Alpha beta gamma delta one.", "Epsilon zeta eta theta two.", "Iota kappa lambda mu three.",
             "Nu xi omicron pi four.", "Rho sigma tau upsilon five."]
    evidence = [ev(i, t, hit_index=i) for i, t in enumerate(texts)]
    for entry, score in zip(evidence, (0.2, 0.9, 0.5, 0.9, 0.7)):
        entry.unit.score = score
    evidence.append(ev(5, "Phi chi psi omega six.", hit_index=5, anchor=False))
    evidence[5].unit.score = 1.0
    text = TemplateComposer().compose(ComposeRequest(system="s", question="q", evidence=evidence)).text
    assert text == "Epsilon zeta eta theta two. [2] Nu xi omicron pi four. [4] Rho sigma tau upsilon five. [5]"


@pytest.mark.parametrize("outputs,error_type,exc,reason", [
    ("", None, None, "empty"),
    ("[1]", None, None, "marker_only"),
    (REFUSAL_LINE, None, None, "model_refused"),
    ("Buffett likes cash.", None, None, "verifier_rejected"),
    ("Invented 77% growth. [1]", None, None, "verifier_rejected"),
    ("ignored", "RuntimeError", None, "backend_error"),
    ("", None, ValueError("boom secret text"), "backend_error"),
])
def test_template_fallback_reasons(outputs, error_type, exc, reason):
    evidence = [ev(0, S1, hit_index=1)]
    composer = FakeComposer(outputs, error_type=error_type, exc=exc)
    out = build_engine({"all": evidence}, composer).answer("q", make_hits(3))
    group = out.groups[0]
    assert group.fallback_reason == reason
    assert out.answer == f"{S1} [2]"
    if reason == "backend_error":
        assert out.failures and out.failures[0].error_type in ("RuntimeError", "ValueError")
        assert "secret" not in repr(out.failures)
    assert out.timings_ms["total"] >= out.timings_ms["compose"] >= 0.0


def test_fallback_happens_at_most_once_then_group_becomes_note():
    evidence = [ev(0, "Quantities see note [4] only.")]  # template skips bracket syntax
    composer = FakeComposer("")
    out = build_engine({"all": evidence}, composer).answer("q", make_hits(2))
    assert len(composer.requests) == 1
    assert out.answer == REFUSAL_LINE and out.citations == []
    assert out.groups[0].fallback_reason == "empty" and out.groups[0].note


def test_partial_answer_with_typed_coverage_note_not_a_refusal():
    slots = [Slot("1985", "period", 1985, 1985), Slot("1990", "period", 1990, 1990)]
    evidence = [ev(0, S1, hit_index=1, year=1985)]
    composer = FakeComposer(f"{S1} [1]")
    out = build_engine({"1985": evidence}, composer, decision="partial", slots=slots).answer("q", make_hits(3))
    assert out.answer == f"In the 1985 letter: {S1} [2]\n\nThe retrieved passages do not cover 1990."
    assert [g.note for g in out.groups] == [None, "The retrieved passages do not cover 1990."]
    assert len(composer.requests) == 1  # the note group is never composed nor verified
    assert "the 1985 letter" in composer.requests[0].system
    assert [c["marker"] for c in out.citations] == ["[2]"]


def test_all_groups_missing_gives_exact_refusal_without_citations():
    slots = [Slot("1985", "period", 1985, 1985), Slot("1990", "period", 1990, 1990)]
    composer = FakeComposer("unused")
    out = build_engine({}, composer, decision="refuse", slots=slots).answer("q", make_hits(3))
    assert out.answer == REFUSAL_LINE and out.citations == [] and composer.requests == []
    # planner says partial but the only group fails verification and fallback -> refusal, notes dropped
    bad = [ev(0, "Unciteable [3] sentence.", year=1985)]
    out = build_engine({"1985": bad}, FakeComposer(""), decision="partial", slots=slots).answer("q", make_hits(3))
    assert out.answer == REFUSAL_LINE and out.citations == []
    assert all(g.note for g in out.groups)


def test_no_hit_request_gives_refusal_line_without_planning():
    calls: list = []
    composer = FakeComposer("unused")
    out = build_engine({"all": [ev(0, S1)]}, composer, query_calls=calls).answer("q", [])
    assert out.answer == REFUSAL_LINE and out.citations == [] and calls == [] and composer.requests == []
    assert out.plan.decision == "refuse" and out.plan.reasons == ["no_context"]


def test_request_carries_temperature_token_cap_history_and_untrusted_prompt():
    composer = FakeComposer(f"{S1} [1]")
    engine = build_engine({"all": [ev(0, S1)]}, composer)
    history = [{"role": "user", "content": "Tell me about 1985."}]
    engine.answer("what changed?", make_hits(2), history=history, max_new_tokens=50, temperature=0.0)
    request = composer.requests[0]
    assert request.temperature == 0.0 and request.max_tokens == 50
    assert request.question == "what changed?" and "Tell me about 1985." in request.history_summary
    assert "UNTRUSTED" in request.system and REFUSAL_LINE in request.system
    assert len(GroundedEngine.SYSTEM_PROMPT.splitlines()) <= 22


def test_period_slot_range_label_and_template_composer_kind_skips_model_lock():
    slots = [Slot("2", "period", 1985, 1987)]
    resources = SimpleNamespace(scorer=object(), locks={"composer": None}, get_composer=lambda k: TemplateComposer(),
                                load=lambda: None)
    evidence = [ev(0, S1, year=1986)]
    plan = EvidencePlan("q", "q", slots=slots, units=[evidence[0].unit], groups={"2": ["u0-0"]}, decision="answer")
    engine = GroundedEngine(resources, SETTINGS, composer_kind="template",
                            build_plan=lambda *a, **k: plan, group_evidence=lambda p, g, h: evidence)
    out = engine.answer("q", make_hits(2))
    assert out.answer == f"In the 1985-1987 letters: {S1} [1]"


# ---- R1 numeric guard: each case passes the legacy validator, only the guard rejects it ----

R1_CASES = [
    ("reversed_ratio",
     "Berkshire's book value rose 23.8% in a year when the industry's rose only 10.0%.",
     "Berkshire's book value rose 10.0% in a year when the industry's rose only 23.8%. [1]"),
    ("reversed_year_binding",
     "In 1985 we earned $5 million and in 1986 we earned $7 million.",
     "In 1985 we earned $7 million and in 1986 we earned $5 million. [1]"),
    ("forecast_to_actual",
     "We expect earnings of $50 million next year.",
     "Earnings were $50 million. [1]"),
    ("pretax_to_aftertax",
     "Pre-tax earnings were $12 million.",
     "After-tax earnings were $12 million. [1]"),
]


GUARD_MODES = ["verbatim", "bound"]


@pytest.mark.parametrize("mode", GUARD_MODES)
@pytest.mark.parametrize("name,source,answer", R1_CASES, ids=[c[0] for c in R1_CASES])
def test_numeric_guard_rejects_paraphrases(name, source, answer, mode):
    evidence = [ev(0, source)]
    # the legacy validator alone would let (most of) these through; the guard is what rejects them
    verifier = DeterministicVerifier(mode)
    outcome = verifier.verify_detailed(answer, evidence)
    assert outcome.sentences == []
    assert sum(outcome.dropped.values()) == 1
    assert verifier.verify(f"{source} [1]", evidence), "verbatim copy must pass"


@pytest.mark.parametrize("mode", GUARD_MODES)
def test_numeric_guard_catches_what_legacy_accepts(mode):
    source = "Berkshire's book value rose 23.8% in a year when the industry's rose only 10.0%."
    answer = "Berkshire's book value rose 10.0% in a year when the industry's rose only 23.8%. [1]"
    evidence = [ev(0, source)]
    assert validate_and_filter_answer(answer, evidence).safe_answer
    assert DeterministicVerifier(mode).verify_detailed(answer, evidence).dropped == {"numeric_guard": 1}


@pytest.mark.parametrize("mode", GUARD_MODES)
def test_invented_number_rejected_and_citation_normalisation_accepts_verbatim(mode):
    evidence = [ev(0, S1)]
    verifier = DeterministicVerifier(mode)
    assert verifier.verify("Berkshire's book value per share rose 31.0% in 1985. [1]", evidence) == []
    spaced = "Berkshire’s   book value per share rose 23.8% in 1985 [1]."
    assert len(verifier.verify(spaced, evidence)) == 1


# real smoke sentences (smoke_mlx rows): correct paraphrases that "verbatim" drops and "bound" accepts
SMOKE_1977_SOURCE = ("To the Stockholders of Berkshire Hathaway Inc.: Operating earnings in 1977 of $21,904,000, "
                     "or $22.54 per share, were moderately better than anticipated a year ago.")
SMOKE_1977_ANSWER = "Operating earnings in 1977 were $21,904,000, or $22.54 per share [1]."
SMOKE_1979_SOURCE = ("Just as the original 3% savings bond, a 5% passbook savings account or an 8% U.S. Treasury "
                     "Note have, in turn, been transformed by inflation into financial instruments that chew up, "
                     "rather than enhance, purchasing power over their investment lives, a business earning 20% "
                     "on capital can produce a negative real return for its owners under inflationary conditions "
                     "not much more severe than presently prevail.")
SMOKE_1979_ANSWER = ("Yes, a business earning 20% on capital can produce a negative real return for its owners "
                     "under inflationary conditions not much more severe than those presently prevailing. [1]")


@pytest.mark.parametrize("source,answer", [(SMOKE_1977_SOURCE, SMOKE_1977_ANSWER),
                                           (SMOKE_1979_SOURCE, SMOKE_1979_ANSWER)], ids=["ho01_1977", "ho02_1979"])
def test_bound_guard_accepts_faithful_numeric_paraphrase(source, answer):
    evidence = [ev(0, source)]
    assert DeterministicVerifier("verbatim").verify_detailed(answer, evidence).dropped == {"numeric_guard": 1}
    outcome = DeterministicVerifier("bound").verify_detailed(answer, evidence)
    assert [v.text for v in outcome.sentences] == [answer] and outcome.dropped == {}


def test_numeric_guard_default_is_bound_and_validated(monkeypatch):
    monkeypatch.delenv("GROUNDED_NUMERIC_GUARD", raising=False)
    assert DeterministicVerifier().numeric_guard == "bound"
    monkeypatch.setenv("GROUNDED_NUMERIC_GUARD", "verbatim")
    assert DeterministicVerifier().numeric_guard == "verbatim"
    with pytest.raises(ValueError):
        DeterministicVerifier("loose")


def test_verifier_checks_only_the_cited_group_sentence():
    evidence = [ev(0, S1, hit_index=0), ev(1, S2, hit_index=1)]
    # S2's wording cited to [1] (S1) is unsupported
    assert DeterministicVerifier().verify(f"{S2} [1]", evidence) == []
    assert len(DeterministicVerifier().verify(f"{S2} [2]", evidence)) == 1


def test_decision_table_consistent_with_group_slot_check():
    # the engine's group "filled" test reuses decide(); sanity-check the contract it relies on
    slots = [Slot("1985", "period", 1985, 1985)]
    assert decide(slots, [unit("a", 0, 1985)], SETTINGS, context_hit_count=1).decision == "answer"
    assert decide(slots, [unit("a", 0, 1990)], SETTINGS, context_hit_count=1).decision == "refuse"


@pytest.mark.parametrize("guard", ["verbatim", "bound"])
def test_engine_builds_verifier_with_settings_numeric_guard(guard, monkeypatch):
    monkeypatch.setenv("GROUNDED_NUMERIC_GUARD", "verbatim" if guard == "bound" else "bound")
    engine = GroundedEngine(SimpleNamespace(), GroundedSettings(COMPOSER="template", NUMERIC_GUARD=guard))
    assert engine.verifier.numeric_guard == guard
