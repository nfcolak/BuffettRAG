"""Fake-only checks for the frozen U0 contract; no model loads or downloads."""

from __future__ import annotations

import copy
import math
import os
import subprocess
import sys
import threading
from dataclasses import fields
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.generation.grounded.decision import decide
from src.generation.grounded.protocols import Composer, Engine, GroundedResources, SentenceScorer, Verifier
from src.generation.grounded.settings import GroundedSettings
from src.generation.grounded.types import (
    ComposeRequest, ComposeResult, DecisionResult, EvidencePlan, EvidenceUnit,
    GroundedAnswer, GroupEvidence, GroupResult, Slot, SourceSpan, StageFailure,
    VerifiedSentence,
)


def _unit(eid="u1", year=1985, score=0.9, *, quantity=True, hit_index=0):
    text = "Berkshire earned $5 million." if quantity else "Berkshire retained the funds."
    span = SourceSpan(hit_index, 0, len(text), text)
    return EvidenceUnit(
        eid, hit_index, f"passage-{eid}", year, span, span, score,
        {(Decimal("5000000"), "$")} if quantity else set(), False, [],
    )


def _period(year, *, number=False, filled_by=()):
    return Slot(str(year), "period", year, year, number, list(filled_by))


def _exercise_fakes():
    class FakeScorer:
        def score(self, query, texts):
            scores = [0.9 for _ in texts]
            assert len(scores) == len(texts) and all(math.isfinite(p) for p in scores)
            return scores

    class FakeComposer:
        def compose(self, req):
            assert req.temperature == 0.0
            text = f"{req.evidence[0].text} [1]"
            return ComposeResult(text, text, "fake", "no-model", "stop", 20, 8)

        def raw_generate(self, prompt, max_tokens):
            return prompt[:max_tokens]

    class FakeVerifier:
        def verify(self, group_text, units):
            return [VerifiedSentence(group_text, [0], [units[0].unit.eid], [units[0].hit_index])]

    class FakeResources:
        def __init__(self):
            self.scorer = FakeScorer()
            self.composer = FakeComposer()
            self.locks = {name: threading.Lock() for name in ("load", "scorer", "composer")}
            self.reranker = None
            self.loads = 0

        def attach_resources(self, *, reranker):
            self.reranker = reranker

        def load(self):
            with self.locks["load"]:
                if not self.loads:
                    self.loads = 1

    resources = FakeResources()
    unit = _unit()
    adapter = GroupEvidence(0, unit.anchor.text, unit, unit.anchor)
    plan = EvidencePlan("earnings?", "earnings?", [Slot("all")], [unit], {"all": [unit.eid]}, "answer")
    verifier = FakeVerifier()

    class FakeEngine:
        def answer(self, original_query, context_hits, *, history=(), max_new_tokens=200):
            resources.load()
            req = ComposeRequest("source only", original_query, [adapter], max_tokens=max_new_tokens)
            result = resources.composer.compose(req)
            sentences = verifier.verify(result.text, req.evidence)
            group = GroupResult("all", result.text, sentences)
            return GroundedAnswer(result.text, [], plan, [group], result.backend,
                                  result.model_id, {"all": result}, [StageFailure("fake", "ExampleError")])

    assert isinstance(resources, GroundedResources)
    assert isinstance(resources.scorer, SentenceScorer)
    assert isinstance(resources.composer, Composer)
    assert isinstance(verifier, Verifier)
    engine = FakeEngine()
    assert isinstance(engine, Engine)
    shared = object()
    resources.attach_resources(reranker=shared)
    resources.load()
    answer = engine.answer("earnings?", [SimpleNamespace(text=unit.window.text)])
    assert resources.loads == 1 and resources.reranker is shared
    assert resources.scorer.score("earnings?", [unit.window.text]) == [0.9]
    assert resources.composer.raw_generate("test", 3) == "tes"
    assert answer.groups[0].sentences[0].unit_ids == [unit.eid]
    assert answer.raw_outputs["all"].raw == answer.answer
    assert answer.failures == [StageFailure("fake", "ExampleError")]
    assert set(answer.timings_ms) == {
        "load_scorer", "load_composer", "load_nli", "score", "plan", "compose", "verify", "total",
    }
    assert all(value == 0.0 for value in answer.timings_ms.values())
    assert GroundedSettings(COMPOSER="template").MLX_MODEL.is_absolute()


def test_construct_with_fakes():
    _exercise_fakes()


def test_import_and_construct_without_optional_ml_dependencies():
    script = '''
import importlib.abc
import runpy
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in {'torch', 'mlx', 'mlx_lm', 'sentence_transformers', 'transformers'}
                or '.nli' in fullname
                or fullname in {'src.generation.grounded.engine', 'src.generation.grounded.resources',
                                'src.generation.grounded.composers', 'src.generation.prompt', 'config'}):
            raise ImportError('intentionally unavailable: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import src.generation.grounded
assert src.generation.grounded.__all__ == []
namespace = runpy.run_path(sys.argv[1])
namespace['_exercise_fakes']()
assert not any(name.split('.')[0] in {'torch', 'mlx', 'mlx_lm', 'sentence_transformers', 'transformers'}
               for name in sys.modules)
print('offline import/construct OK')
'''
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(PYTHON_DOTENV_DISABLED="1", HF_HUB_OFFLINE="1")
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve())],
                            cwd=Path(__file__).resolve().parents[1], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "offline import/construct OK"


def test_refusal_is_the_existing_constant():
    from src.generation.grounded.types import REFUSAL_LINE
    from src.generation.prompt import REFUSAL_LINE as legacy_refusal
    assert REFUSAL_LINE is legacy_refusal


def test_r2_adapter_real_validator_and_client_citation_round_trip():
    from src.evaluation.claim_validator import evidence_sentences, validate_and_filter_answer
    from src.generation.prompt import format_answer_markdown, parse_citations

    source = "Berkshire earned $5\nmillion. It retained the funds."
    first_end = source.index(".") + 1
    first = SourceSpan(1, 0, first_end, source[:first_end])
    second = SourceSpan(1, first_end + 1, len(source), source[first_end + 1:])
    window = SourceSpan(1, 0, len(source), source)
    unit = EvidenceUnit("u", 1, "actual-hit", 1985, first, window, 0.9,
                        {(Decimal("5000000"), "$")})
    rendered = evidence_sentences(source)
    local = [GroupEvidence(i, text, unit, span) for i, (text, span) in enumerate(zip(rendered, [first, second]))]
    request = ComposeRequest("sources only", "earnings?", local)
    assert request.temperature == 0.0
    assert request.max_tokens == 200
    for entry in local:
        span = entry.source_span
        assert span.text == source[span.start:span.end]
        assert entry.text == " ".join(span.text.split())
    answer = " ".join(f"{entry.text} [{entry.local_index + 1}]" for entry in local)
    checked = validate_and_filter_answer(answer, local)
    assert checked.safe_answer == answer
    assert checked.blocked_claims == []
    assert all(item["supported"] for item in checked.validations)
    assert [item["cited_indexes"] for item in checked.validations] == [[0], [1]]
    assert [(entry.local_index, entry.unit.eid, entry.hit_index) for entry in local] == [(0, "u", 1), (1, "u", 1)]
    hits = [SimpleNamespace(id="irrelevant", text="Unrelated.", metadata={"year": 1999}),
            SimpleNamespace(id="actual-hit", text=source, metadata={"year": 1985, "source_file": "letter-1985"})]
    global_answer = format_answer_markdown(" ".join(f"{entry.text} [{entry.hit_index + 1}]" for entry in local))
    citations = parse_citations(global_answer, hits)
    assert len(citations) == 2
    assert all(cite["passage_indices"] == [1] and cite["passage_ids"] == ["actual-hit"] and
               cite["years"] == [1985] and not cite["invalid_numbers"] for cite in citations)


def test_adapter_escapes_only_prompt_copy():
    unit = _unit()
    text = "The quoted notation is [1] and [E2] and [k]."
    entry = GroupEvidence(0, text, unit)
    assert entry.text == text
    assert entry.prompt_text == "The quoted notation is (1) and (E2) and (k)."
    assert entry.id == unit.passage_id
    assert entry.metadata == {"year": 1985}


@pytest.mark.parametrize("kwargs", [
    {"local_index": -1}, {"local_index": True}, {"text": "two\nlines."}, {"text": " leading."},
    {"text": ""}, {"source_span": SourceSpan(2, 0, 4, "Text")},
])
def test_adapter_rejects_invalid_mapping_or_rendering(kwargs):
    values = {"local_index": 0, "text": "A source sentence.", "unit": _unit()}
    values.update(kwargs)
    with pytest.raises(ValueError):
        GroupEvidence(**values)


@pytest.mark.parametrize("kwargs", [
    {"temperature": -0.1}, {"temperature": 2.1}, {"temperature": float("nan")},
    {"temperature": float("inf")}, {"temperature": True},
    {"max_tokens": 0}, {"max_tokens": 201}, {"max_tokens": True},
])
def test_compose_request_range_validation(kwargs):
    with pytest.raises(ValueError):
        ComposeRequest("system", "question", [], **kwargs)


def test_compose_request_rejects_noncontiguous_local_indexes():
    with pytest.raises(ValueError, match="contiguous"):
        ComposeRequest("system", "question", [GroupEvidence(1, "A source sentence.", _unit())])


def test_dataclass_containers_are_request_local():
    first, second = EvidencePlan("q", "q"), EvidencePlan("q", "q")
    first.slots.append(Slot("all"))
    first.groups["all"] = ["u1"]
    first.trace["request"] = 1
    assert not second.slots and not second.groups and not second.trace
    left, right = GroundedAnswer("", [], first), GroundedAnswer("", [], second)
    left.timings_ms["total"] = 12.0
    left.failures.append(StageFailure("compose", "FakeFailure"))
    assert right.timings_ms["total"] == 0.0 and right.failures == []
    one, two = _unit(), _unit()
    one.quantities.clear()
    one.slots.append("s")
    assert two.quantities and not two.slots


@pytest.mark.parametrize("composer", ["auto", "mlx", "llama", "template"])
def test_settings_composers(composer):
    assert GroundedSettings(COMPOSER=composer).COMPOSER == composer


@pytest.mark.parametrize("device", ["auto", "cpu", "mps", "cuda", "cuda:0"])
def test_settings_devices(device):
    assert GroundedSettings(SCORER_DEVICE=device).SCORER_DEVICE == device


@pytest.mark.parametrize("kwargs", [
    {"COMPOSER": "unknown"}, {"SCORER_DEVICE": "metal"}, {"SCORER_DEVICE": "cuda:-1"},
    {"MAX_WINDOWS": 0}, {"MAX_WINDOWS": 49}, {"MAX_WINDOWS": True},
    {"SLOT_RESERVE": 0}, {"SLOT_RESERVE": 49},
    {"MAX_UNITS": 3}, {"MAX_UNITS": 9}, {"MAX_UNITS": 6.0},
    {"MAX_PER_HIT": 0}, {"MAX_PER_HIT": 3},
    {"MAX_PERIODS": 0}, {"MAX_PERIODS": 5},
    {"GROUP_MAX_TOKENS": 0}, {"GROUP_MAX_TOKENS": 201},
    {"T_RELEVANT": -0.01}, {"T_RELEVANT": 1.01}, {"T_RELEVANT": float("nan")},
    {"T_SLOT": -0.01}, {"T_SLOT": 1.01}, {"T_SLOT": float("inf")},
    {"DEDUPE_JACCARD": -0.01}, {"DEDUPE_JACCARD": 1.01}, {"DEDUPE_JACCARD": True},
    {"NLI_VERIFY": 1}, {"NLI_VERIFY": True},
    {"NUMERIC_GUARD": "strict"}, {"NUMERIC_GUARD": ""}, {"NUMERIC_GUARD": None},
    {"MLX_MODEL": ""}, {"LLAMA_MODEL": " "}, {"NLI_MODEL": ""}, {"CACHE_DIR": ""},
    {"MAX_WINDOWS": 8},
    {"MAX_WINDOWS": 4, "MAX_PERIODS": 1, "SLOT_RESERVE": 1, "MAX_UNITS": 8},
])
def test_settings_invalid_ranges(kwargs):
    with pytest.raises(ValueError):
        GroundedSettings(**kwargs)


def test_settings_boundary_values_and_paths(tmp_path, monkeypatch):
    low = GroundedSettings(MAX_WINDOWS=4, SLOT_RESERVE=1, MAX_UNITS=4, MAX_PER_HIT=1,
                           MAX_PERIODS=1, T_RELEVANT=0.0, T_SLOT=0.0,
                           DEDUPE_JACCARD=0.0, GROUP_MAX_TOKENS=1)
    high = GroundedSettings(MAX_WINDOWS=48, SLOT_RESERVE=12, MAX_UNITS=8, MAX_PER_HIT=2,
                            MAX_PERIODS=4, T_RELEVANT=1.0, T_SLOT=1.0,
                            DEDUPE_JACCARD=1.0, GROUP_MAX_TOKENS=200)
    assert low.MAX_UNITS == 4 and high.MAX_UNITS == 8
    monkeypatch.chdir(tmp_path)
    repo = Path(__file__).resolve().parents[1]
    settings = GroundedSettings(MLX_MODEL="relative/mlx", LLAMA_MODEL="relative/llama.gguf",
                                NLI_VERIFY=True, NLI_MODEL="relative/nli", CACHE_DIR="relative/cache")
    for name, suffix in (("MLX_MODEL", "mlx"), ("LLAMA_MODEL", "llama.gguf"),
                         ("NLI_MODEL", "nli"), ("CACHE_DIR", "cache")):
        assert getattr(settings, name) == (repo / "relative" / suffix).resolve()
    assert not (repo / "relative").exists()
    defaults = GroundedSettings()
    model_repo = next((parent.parent for parent in repo.parents if parent.name == ".worktrees"), repo)
    assert defaults.MLX_MODEL == model_repo / "models/teacher-qwen2.5-7b-mlx4"
    assert defaults.LLAMA_MODEL.is_absolute()
    assert defaults.CACHE_DIR.is_relative_to(repo)


def test_settings_has_every_frozen_knob_and_loads_env():
    assert {item.name for item in fields(GroundedSettings)} == {
        "COMPOSER", "MLX_MODEL", "LLAMA_MODEL", "SCORER_DEVICE", "MAX_WINDOWS", "SLOT_RESERVE",
        "MAX_UNITS", "MAX_PER_HIT", "MAX_PERIODS", "T_RELEVANT", "T_SLOT", "DEDUPE_JACCARD",
        "GROUP_MAX_TOKENS", "NLI_VERIFY", "NLI_MODEL", "NUMERIC_GUARD", "CACHE_DIR",
    }
    assert GroundedSettings().NUMERIC_GUARD == "bound"
    # frozen by the dev calibration (data/evaluation/grounded_dev_v1/chosen.json)
    frozen = GroundedSettings()
    assert (frozen.T_RELEVANT, frozen.T_SLOT, frozen.MAX_UNITS) == (0.01, 0.001, 6)
    assert GroundedSettings(NUMERIC_GUARD="verbatim").NUMERIC_GUARD == "verbatim"
    settings = GroundedSettings.from_env({
        "GROUNDED_COMPOSER": "template", "GROUNDED_MLX_MODEL": "local/mlx",
        "GROUNDED_LLAMA_MODEL": "local/model.gguf", "GROUNDED_SCORER_DEVICE": "cpu",
        "GROUNDED_MAX_WINDOWS": "48", "GROUNDED_SLOT_RESERVE": "6", "GROUNDED_MAX_UNITS": "8",
        "GROUNDED_MAX_PER_HIT": "1", "GROUNDED_MAX_PERIODS": "4", "GROUNDED_T_RELEVANT": "0.3",
        "GROUNDED_T_SLOT": "0.6", "GROUNDED_DEDUPE_JACCARD": "0.8", "GROUNDED_GROUP_MAX_TOKENS": "100",
        "GROUNDED_NLI_VERIFY": "1", "GROUNDED_NLI_MODEL": "local/nli", "GROUNDED_CACHE_DIR": "local/cache",
        "GROUNDED_NUMERIC_GUARD": "verbatim",
    })
    assert settings.NUMERIC_GUARD == "verbatim"
    assert settings.COMPOSER == "template" and settings.MAX_UNITS == 8
    assert settings.T_RELEVANT == 0.3 and settings.T_SLOT == 0.6
    assert settings.NLI_VERIFY is True and settings.NLI_MODEL.is_absolute()


@pytest.mark.parametrize("env", [
    {"GROUNDED_NLI_VERIFY": "true"}, {"GROUNDED_MAX_UNITS": "6.5"},
    {"GROUNDED_MAX_UNITS": "-1"}, {"GROUNDED_T_SLOT": "nan"},
    {"GROUNDED_COMPOSER": " MLX "}, {"GROUNDED_NUMERIC_GUARD": "loose"},
])
def test_settings_rejects_bad_env(env):
    with pytest.raises(ValueError):
        GroundedSettings.from_env(env)


@pytest.mark.parametrize("slots,units,count,expected", [
    ([Slot("all")], [], 0, DecisionResult("refuse", ["no_context"])),
    ([Slot("all")], [], None, DecisionResult("refuse", ["no_context"])),
    ([Slot("all")], [], 2, DecisionResult("refuse", ["low_relevance"])),
    ([Slot("all")], [_unit(score=0.49)], 1, DecisionResult("refuse", ["low_relevance"])),
    ([Slot("all")], [_unit(score=0.5)], 1, DecisionResult("answer", [])),
    ([Slot("all", needs_number=True)], [_unit(quantity=False)], 1, DecisionResult("refuse", ["missing_quantity"])),
    ([_period(1990)], [_unit(year=1985)], 1, DecisionResult("refuse", ["missing_period"])),
    ([_period(1985), _period(1990)], [_unit(year=1985)], 1, DecisionResult("partial", ["missing_period"])),
    ([_period(1985, number=True), _period(1990, number=True)],
     [_unit(year=1985), _unit("u2", 1990, quantity=False)], 2, DecisionResult("partial", ["missing_quantity"])),
    ([_period(1985), _period(1990)], [_unit(year=1985), _unit("u2", 1990)], 2, DecisionResult("answer", [])),
    ([_period(year) for year in range(1980, 1985)],
     [_unit(str(year), year) for year in range(1980, 1985)], 5, DecisionResult("refuse", ["too_many_periods"])),
    ([_period(1985), _period(1990, filled_by=["removed"])], [_unit(year=1985)],
     2, DecisionResult("partial", ["capacity"])),
    ([_period(1990, filled_by=["removed"])], [_unit(year=1985)], 2, DecisionResult("refuse", ["capacity"])),
    ([_period(1985, filled_by=["removed"])], [_unit(year=1985)], 1, DecisionResult("answer", [])),
    ([Slot("range", "period", 1980, 1989)], [_unit(year=1985)], 1, DecisionResult("answer", [])),
    ([_period(1985, number=True)], [_unit(year=None)], 1, DecisionResult("refuse", ["missing_period"])),
    ([_period(1990), _period(1995)], [_unit(year=1985)], 1, DecisionResult("refuse", ["missing_period"])),
    ([_period(1990), Slot("amount", needs_number=True)], [_unit(year=1985, quantity=False)],
     1, DecisionResult("refuse", ["missing_period", "missing_quantity"])),
    ([], [_unit()], 1, DecisionResult("refuse", ["low_relevance"])),
])
def test_decision_truth_table_is_pure(slots, units, count, expected):
    settings = GroundedSettings(T_RELEVANT=0.5, T_SLOT=0.5)  # the table's cases are written at 0.5
    before = copy.deepcopy((slots, units, settings))
    assert decide(slots, units, settings, context_hit_count=count) == expected
    assert (slots, units, settings) == before


def test_decision_recomputes_final_coverage_at_slot_threshold():
    settings = GroundedSettings(T_RELEVANT=0.4, T_SLOT=0.7)
    assert decide([Slot("all", filled_by=["u1"])], [_unit(score=0.6)], settings) == DecisionResult("refuse", ["low_relevance"])
    assert decide([Slot("all")], [_unit(score=0.7)], settings) == DecisionResult("answer", [])
    assert decide([_period(1985, number=True)], [_unit(quantity=False)], settings) == DecisionResult("refuse", ["missing_quantity"])


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_decision_rejects_invalid_scores(score):
    with pytest.raises(ValueError):
        decide([Slot("all")], [_unit(score=score)], GroundedSettings())


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_decision_rejects_invalid_context_count(count):
    with pytest.raises(ValueError):
        decide([Slot("all")], [], GroundedSettings(), context_hit_count=count)
