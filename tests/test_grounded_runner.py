"""Grounded benchmark runner and strict supported-claim metric, with fakes only."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.eval import run_live_benchmark as runner
from src.evaluation.supported_claims import supported_claims_met


class _Provider:
    provider_name = "mlx"
    model = "fake-mlx"
    temperature = None

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def generate(self, prompt, max_new_tokens=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("composer unavailable: secret body")
        return "Insurance float is cheap capital [1]."


def _run(monkeypatch, provider, tmp_path, **kwargs):
    monkeypatch.setattr(runner, "create_llm_provider", lambda **_k: provider)
    return runner.run(provider="mlx", retrieval="bm25", output=tmp_path / "out.json", **kwargs)


def test_manifest_rows_and_resume_skip_completed(monkeypatch, tmp_path):
    first = _Provider()
    result = _run(monkeypatch, first, tmp_path, max_cases=2, temperature=0.0)
    out = tmp_path / "out.json"
    rows_path, manifest_path = tmp_path / "out.json.rows.jsonl", tmp_path / "out.json.manifest.json"
    assert len(rows_path.read_text().splitlines()) == 2
    manifest = json.loads(manifest_path.read_text())
    assert manifest["corpus_sha256"] == result["corpus_sha256"] and manifest["cases_sha256"] == result["cases_sha256"]
    assert manifest["temperature"]["requested"] == 0.0 and manifest["temperature"]["asserted_at_provider"]
    assert manifest["resumes"] == [] and "library_versions" in manifest and "code_sha" in manifest
    assert result["answer_engine"]["temperature_asserted"] is True
    assert not out.exists()  # the final JSON is written by main(); only sidecars exist here

    second = _Provider()
    resumed = _run(monkeypatch, second, tmp_path, max_cases=3, resume=True)
    done = {row["qid"] for row in result["rows"]}
    assert second.calls >= 1 and len(resumed["rows"]) == 3
    assert {row["qid"] for row in resumed["rows"]} >= done
    qids = [json.loads(line)["qid"] for line in rows_path.read_text().splitlines()]
    assert len(qids) == 3 and len(set(qids)) == 3  # completed qids were not regenerated
    manifest = json.loads(manifest_path.read_text())
    assert manifest["resumes"][0]["skipped_completed_qids"] == sorted(done, key=[r["qid"] for r in result["rows"]].index)
    assert manifest["resumes"][0]["generated_new"] == 1


def test_unavailable_composer_is_recorded_failure_not_substituted(monkeypatch, tmp_path):
    result = _run(monkeypatch, _Provider(fail=True), tmp_path, max_cases=1)
    row = result["rows"][0]
    assert row["status"] == "provider_failure" and row["answer"] is None and row["score"] is None
    assert row["provider_failures"] == [{"stage": "generating", "error_type": "RuntimeError"}]
    assert "secret body" not in json.dumps(result)
    saved = json.loads((tmp_path / "out.json.rows.jsonl").read_text().splitlines()[0])
    assert saved["status"] == "provider_failure"


def test_grounded_requires_explicit_composer_and_absolute_model_path(tmp_path):
    with pytest.raises(ValueError, match="--composer"):
        runner.run(provider="grounded", retrieval="bm25", max_cases=1)
    with pytest.raises(ValueError, match="absolute"):
        runner.run(provider="mlx", retrieval="bm25", max_cases=1, model_path=runner.Path("models/x"))
    with pytest.raises(ValueError, match="--resume"):
        runner.run(provider="local", retrieval="bm25", max_cases=1, resume=True)


def test_tracked_provider_records_grounded_call():
    from src.generation.grounded.types import ComposeResult, EvidencePlan, GroundedAnswer, GroupResult, StageFailure

    class P:
        provider_name, model = "grounded", "m"

        def answer_grounded(self, q, hits, *, history=(), max_new_tokens=200):
            return GroundedAnswer(
                answer="a", citations=[], plan=EvidencePlan(q, q, trace={"k": 1}),
                groups=[GroupResult("all", "a", fallback_reason="verifier_rejected")], composer="template",
                model_id="m", raw_outputs={"all": ComposeResult("t", "RAW", "template", "m")},
                failures=[StageFailure("compose", "OSError")], timings_ms={"load_composer": 5.0, "compose": 2.0})

    tracked = runner._TrackedProvider(P(), 0.0, "template")
    hit = SimpleNamespace(id="p1", text="Text here.", year=1990)
    tracked.answer_grounded("q", [hit])
    call = tracked.grounded_calls[0]
    assert call["raw_outputs"]["all"]["raw"] == "RAW" and call["fallback_reasons"] == {"all": "verifier_rejected"}
    assert call["cold_load_ms"] == {"load_composer": 5.0} and call["inference_ms"] == {"compose": 2.0}
    assert call["context"][0]["id"] == "p1" and call["context"][0]["text"] == "Text here."
    assert tracked.failures == [{"stage": "compose", "error_type": "OSError"}]
    assert not hasattr(runner._TrackedProvider(_Provider(), 0.0), "answer_grounded")


# --- strict metric -----------------------------------------------------------------------------

class _FakeVerifier:
    """Accepts a sentence when its words all occur in some source sentence (numbers included)."""

    def verify(self, group_text, units):
        import re
        body = re.sub(r"\[\d+\]", "", group_text).lower()
        words = [w.strip(".,") for w in re.findall(r"[a-z0-9$.,%]+", body)]
        words = [w for w in words if w]
        return [object()] if any(all(w in u.text.lower() for w in words) for u in units) else []


CASE = {"qid": "q", "query": "x", "gold_passage_ids": ["g1", "g2"],
        "gold_claims": [{"claim": "Float reached $10 billion.", "required_terms": ["float", "$10 billion"],
                         "gold_passage_ids": ["g1"]}],
        "accept": {"min_claim_coverage": 1.0, "require_valid_citation": True}, "reject": {}}
HITS = [SimpleNamespace(id="g1", text="Float reached $10 billion in 1999.", year=1999),
        SimpleNamespace(id="g2", text="Float reached $10 billion in 2000.", year=2000)]


def _met(answer, hits=HITS):
    return supported_claims_met(CASE, answer, [{"id": h.id} for h in hits], hits, verifier=_FakeVerifier())[0]["met"]


def test_strict_metric_correct_is_met():
    assert _met("Float reached $10 billion [1].")


def test_strict_metric_non_gold_citation_not_met():
    assert not _met("Float reached $10 billion [2].")  # lexical match, but cites a non-gold hit


def test_strict_metric_numeric_paraphrase_not_met():
    assert not _met("Float reached ten billion dollars [1].")  # lexical numbers/terms fail
    # Lexical claim match but the source says something different: deterministic support rejects it.
    hits = [SimpleNamespace(id="g1", text="Float reached $9 billion in 1999. The $10 billion figure was a target.", year=1999)]
    assert not _met("Float reached $10 billion [1].", hits)


def test_strict_metric_label_stripped_only_when_year_supported():
    assert _met("In the 1999 letter: Float reached $10 billion [1].")
    assert not _met("In the 1985 letter: Float reached $10 billion [1].")


def test_strict_label_strip_exact_engine_form_only():
    from src.evaluation.supported_claims import _strip_label
    h = [SimpleNamespace(id="g1", text="", year=2002)]
    assert _strip_label("In the 2002 letter: Float rose [1].", h) == "Float rose [1]."
    assert _strip_label("In 2002, float rose [1].", h) == "In 2002, float rose [1]."
    assert _strip_label("In 2002: float rose [1].", h) == "In 2002: float rose [1]."
    assert _strip_label("In the 1985 letter: Float rose [1].", h) == "In the 1985 letter: Float rose [1]."


def test_rescore_adds_strict_column_and_checks_identity(tmp_path, monkeypatch):
    import hashlib
    from scripts.eval import rescore_nli

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"cases": [CASE]}))
    sha = hashlib.sha256(cases.read_bytes()).hexdigest()
    row = {"qid": "q", "status": "scored", "answer": "Float reached $10 billion [1].", "citations": [{"id": "g1"}],
           "context_snapshot": [{"id": "g1", "text": HITS[0].text, "year": 1999}], "provider_failures": [],
           "score": {"accepted": True}}
    result = {"cases": "cases.json", "cases_sha256": sha, "corpus_sha256": "c", "summary": {"accepted": 1}, "rows": [row]}
    path = tmp_path / "res.json"
    path.write_text(json.dumps(result))

    class FakeNli:
        device = "cpu"

        def score_row(self, r, c):
            return {"qid": r["qid"], "unscored": False, "nli_accepted": True, "nli_claims": [], "nli_claim_coverage": 1.0,
                    "cited_sentences": 0, "supported_cited_sentences": 0}

    monkeypatch.setattr("src.evaluation.supported_claims._default_verifier", lambda: _FakeVerifier())
    summary = rescore_nli.rescore(path, cases, scorer=FakeNli())
    assert summary["strict_supported_claims_met"] == 1 and summary["strict_required_claims"] == 1
    result["cases_sha256"] = "bad"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="sha256"):
        rescore_nli.rescore(path, cases, scorer=FakeNli())
