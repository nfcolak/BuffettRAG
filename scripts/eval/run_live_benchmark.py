"""Run the real /ask path in-process; BM25 mode needs no downloaded models.

The `local` provider is extractive, not a live LLM; `llama` is the embedded GGUF model
(prompts are fitted by backend_app via fit_context_to_llm, as in the server). Scores are the existing
claim-term/number/polarity and claim-specific citation checks, not an LLM judge.
Provider failures are unscored and never enter the answer-quality denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import CHUNKS_V3_FILE, DEFAULT_LLM_PROVIDER
from src.evaluation.answer_benchmark import (
    evaluate_live_answer,
    is_unanswerable_case,
    validate_fixture_ids,
)
from src.generation.providers import create_llm_provider
from src.generation.prompt import REFUSAL_LINE
from src.retrieval.context import build_doc_lookup
from src.retrieval.retriever import Retriever
from src.storage import SearchHit, load_chunks_as_docs

GROUNDED_PROVIDERS = frozenset({"grounded", "mlx"})
COMPOSERS = ("mlx", "llama", "template")
DEFAULT_CASES = ROOT / "data/evaluation/heldout_v1/answer_benchmark_heldout_v1.json"


class FixtureValidationError(ValueError):
    """Only fixture validation errors are safe to render as diagnostic text."""


class BM25OnlyRetriever(Retriever):
    """Use the production fusion, year handling and temporal decomposition.

    Replace only each hybrid candidate source with genuine BM25 retrieval;
    never construct a vector store, embedder or cross-encoder in offline mode.
    """

    def __init__(self, docs):
        # No vector method can run; _vector_search below fails explicitly.
        unused_embedder: Any = None
        super().__init__(vector_store=None, embedder=unused_embedder, docs=docs, reranker=None)

    def _hybrid_for_filter(self, query, fetch_k, where):
        return self.bm25.search(query, top_k=fetch_k, where=where)

    def _vector_search(self, *_args, **_kwargs):
        raise RuntimeError("Vector retrieval is disabled in BM25 mode")


def _context_snapshot(context_hits) -> List[Dict[str, Any]]:
    out = []
    for hit in context_hits:
        text = getattr(hit, "text", "") or ""
        meta = getattr(hit, "metadata", None) or {}
        year = getattr(hit, "year", None)
        out.append({"id": getattr(hit, "id", None), "year": year if year is not None else meta.get("year"),
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "text": text})
    return out


class _TrackedProvider:
    """Capture provider errors even when /ask or query expansion swallows them."""

    def __init__(self, provider, temperature, composer=None):
        self._provider = provider
        self.provider_name = provider.provider_name
        self.model = provider.model
        self.failures = []
        self.requests = []
        self.call_count = 0
        self.temperature = temperature
        self.composer = composer
        self.raw_outputs = []  # (stage, raw model text) for --save-raw; pre-validation
        self.grounded_calls = []  # one record per answer_grounded call (grounded/mlx runs)
        self.notes = []  # non-fatal events, e.g. expansion unavailable with the template composer
        if hasattr(provider, "answer_grounded"):
            self.answer_grounded = self._answer_grounded

    def __getattr__(self, name):
        # Forward optional hooks (attach_resources, ...) without exposing answer_grounded early.
        if name.startswith("_") or name == "answer_grounded":
            raise AttributeError(name)
        return getattr(self._provider, name)

    def _answer_grounded(self, original_query, context_hits, *, history=(), max_new_tokens=200):
        self.call_count += 1
        self.requests.append({"model": self.model, "temperature": self.temperature})
        snapshot = _context_snapshot(context_hits)
        try:
            result = self._provider.answer_grounded(original_query, context_hits, history=history,
                                                    max_new_tokens=max_new_tokens)
        except Exception as exc:
            # Unavailable composer/model: a provider failure, never a silent substitution.
            self.failures.append({"stage": "grounded", "error_type": type(exc).__name__})
            self.grounded_calls.append({"composer": self.composer, "model_id": self.model,
                                        "context": snapshot, "exception_type": type(exc).__name__})
            raise RuntimeError("Benchmark provider request failed") from None
        raw = {}
        for group, item in (getattr(result, "raw_outputs", None) or {}).items():
            raw[group] = {"raw": getattr(item, "raw", None), "text": getattr(item, "text", None),
                          "finish_reason": getattr(item, "finish_reason", None),
                          "tokens_in": getattr(item, "tokens_in", None),
                          "tokens_out": getattr(item, "tokens_out", None),
                          "error_type": getattr(item, "error_type", None)}
            if getattr(item, "error_type", None):
                self.failures.append({"stage": f"compose:{group}", "error_type": item.error_type})
        stage_failures = [{"stage": f.stage, "error_type": f.error_type} for f in getattr(result, "failures", [])]
        for failure in stage_failures:
            if failure not in self.failures:
                self.failures.append(dict(failure))
        groups = getattr(result, "groups", []) or []
        timings = dict(getattr(result, "timings_ms", {}) or {})
        plan = getattr(result, "plan", None)
        self.grounded_calls.append({
            "composer": getattr(result, "composer", self.composer),
            "model_id": getattr(result, "model_id", self.model),
            "raw_outputs": raw, "stage_failures": stage_failures,
            "fallback_reasons": {g.group: g.fallback_reason for g in groups if getattr(g, "fallback_reason", None)},
            "trace": getattr(plan, "trace", None),
            "decision": getattr(plan, "decision", None), "reasons": list(getattr(plan, "reasons", []) or []),
            "timings_ms": timings,
            "cold_load_ms": {k: v for k, v in timings.items() if k.startswith("load_")},
            "inference_ms": {k: v for k, v in timings.items() if not k.startswith("load_")},
            "context": snapshot})
        return result

    def generate(self, prompt, max_new_tokens=None):
        self.call_count += 1
        stage = "expanding" if prompt.startswith(("Return JSON only:", "You expand search queries")) else "generating"
        self.requests.append({"model": self.model, "temperature": self.temperature})
        try:
            output = self._provider.generate(prompt, max_new_tokens=max_new_tokens)
            self.raw_outputs.append((stage, output))
            return output
        except Exception as exc:
            if stage == "expanding" and self.composer == "template":
                self.notes.append({"stage": stage, "error_type": type(exc).__name__, "effect": "no_query_expansion"})
            else:
                self.failures.append({"stage": stage, "error_type": type(exc).__name__})
            # Production debug paths must not accidentally print provider error
            # bodies, headers or API keys; the original exception is not exposed.
            raise RuntimeError("Benchmark provider request failed") from None


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                              timeout=20, check=True).stdout.strip()
    except Exception:
        return None


def _model_fingerprint(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Whole-file sha256 for a file; for a directory, a sha over (relative name, size) + small-file content."""
    if path is None or not Path(path).exists():
        return None
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        return {"path": str(path), "kind": "file", "sha256": digest.hexdigest()}
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        size = item.stat().st_size
        digest.update(f"{item.relative_to(path)}:{size}\n".encode())
        if size < 1 << 20:
            digest.update(item.read_bytes())
    return {"path": str(path), "kind": "dir", "sha256": digest.hexdigest()}


def _library_versions() -> Dict[str, Optional[str]]:
    from importlib import metadata

    out: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for name in ("mlx", "mlx-lm", "llama-cpp-python", "torch", "transformers", "sentence-transformers"):
        try:
            out[name] = metadata.version(name)
        except Exception:
            out[name] = None
    return out


def _device() -> Dict[str, Any]:
    info: Dict[str, Any] = {"machine": platform.machine(), "platform": platform.platform()}
    try:
        import torch
        info["torch_mps"] = bool(torch.backends.mps.is_available())
    except Exception:
        info["torch_mps"] = None
    return info


class _Sidecar:
    """<output>.rows.jsonl (appended per finished row) and <output>.manifest.json; inert when disabled."""

    def __init__(self, output: Optional[Path], enabled: bool):
        self.enabled = bool(enabled and output is not None)
        if self.enabled:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            self.rows_path = output.with_name(output.name + ".rows.jsonl")
            self.manifest_path = output.with_name(output.name + ".manifest.json")
        self.manifest: Dict[str, Any] = {}

    def load_resume(self, corpus_hash: str, cases_hash: str) -> Dict[str, Dict[str, Any]]:
        if not self.manifest_path.exists():
            raise FixtureValidationError("--resume needs an existing manifest next to the rows file")
        prior = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if prior.get("corpus_sha256") != corpus_hash or prior.get("cases_sha256") != cases_hash:
            raise FixtureValidationError("--resume identity mismatch (corpus or cases sha256 changed)")
        self.manifest = prior
        rows: Dict[str, Dict[str, Any]] = {}
        if self.rows_path.exists():
            for line in self.rows_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows[row["qid"]] = row
        return rows

    def write_manifest(self, *, resume, started, corpus, corpus_hash, cases_path, cases_hash, provider, composer,
                       model_path, temperature, retrieval, llm, initialization_failure) -> None:
        if not self.enabled:
            return
        resumes = self.manifest.get("resumes", [])
        settings: Any = None
        try:
            import dataclasses
            from src.generation.grounded.settings import GroundedSettings
            settings = {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in dataclasses.asdict(GroundedSettings.from_env()).items()}
        except Exception as exc:
            settings = {"unavailable": type(exc).__name__}
        status = _git("status", "--porcelain")
        paths = {"model_path": model_path}
        if model_path is None and provider == "grounded" and isinstance(settings, dict):
            paths = {"model_path": settings.get({"mlx": "MLX_MODEL", "llama": "LLAMA_MODEL"}.get(composer or "", ""))}
        self.manifest = {
            "schema_version": 1, "started_utc": self.manifest.get("started_utc", started),
            "code_sha": _git("rev-parse", "HEAD"), "code_dirty": bool(status) if status is not None else None,
            "corpus": str(corpus), "corpus_sha256": corpus_hash, "cases": str(cases_path), "cases_sha256": cases_hash,
            "provider": provider, "composer": composer, "retrieval": retrieval,
            "model": _model_fingerprint(Path(paths["model_path"])) if paths["model_path"] else None,
            "effective_settings": settings,
            "temperature": {"requested": temperature, "asserted_at_provider": llm is not None,
                            "provider_attr": getattr(llm._provider, "temperature", None) if llm is not None else None},
            "library_versions": _library_versions(), "device": _device(),
            "initialization_failure": initialization_failure, "resumes": resumes}
        self._flush()
        if not resume and self.rows_path.exists():
            self.rows_path.unlink()  # a fresh run never inherits rows from an earlier one

    def _flush(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False, default=str) + "\n",
                                      encoding="utf-8")

    def append_row(self, row: Dict[str, Any]) -> None:
        if self.enabled:
            with self.rows_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def record_resume(self, resumed_qids, new_rows: int) -> None:
        if self.enabled:
            self.manifest.setdefault("resumes", []).append({
                "at_utc": datetime.now(timezone.utc).isoformat(), "skipped_completed_qids": list(resumed_qids),
                "generated_new": new_rows})
            self._flush()


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run(
    corpus: Path = CHUNKS_V3_FILE,
    cases_path: Path = DEFAULT_CASES,
    *,
    provider: str = DEFAULT_LLM_PROVIDER,
    temperature: float = 0.0,
    max_cases: Optional[int] = None,
    retrieval: str = "hybrid",
    save_raw: bool = False,
    composer: Optional[str] = None,
    model_path: Optional[Path] = None,
    output: Optional[Path] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """Validate every fixture first, then call backend.ask without HTTP/startup."""
    if retrieval not in {"bm25", "hybrid"}:
        raise ValueError("retrieval must be bm25 or hybrid")
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be finite and between 0 and 2")
    if max_cases is not None and max_cases < 1:
        raise ValueError("max_cases must be positive")
    grounded_run = provider in GROUNDED_PROVIDERS
    if provider == "grounded" and composer not in COMPOSERS:
        raise ValueError("--composer {mlx,llama,template} is required with the grounded provider (no auto)")
    if composer is not None and provider != "grounded":
        raise ValueError("--composer applies only to the grounded provider")
    if model_path is not None:
        if provider not in {"llama", "mlx", "grounded"} or (provider == "grounded" and composer == "template"):
            raise ValueError("--model-path applies only to llama, mlx or grounded (mlx/llama composer)")
        if not Path(model_path).is_absolute():
            raise ValueError("--model-path must be absolute")
    if resume and (output is None or not grounded_run):
        raise ValueError("--resume needs --output and a grounded or mlx provider")
    corpus, cases_path = corpus.resolve(), cases_path.resolve()
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    docs = load_chunks_as_docs(corpus)
    try:
        fixture_check = validate_fixture_ids(cases, docs)
    except ValueError as exc:
        raise FixtureValidationError(str(exc)) from None
    corpus_hash = hashlib.sha256(corpus.read_bytes()).hexdigest()
    if payload.get("corpus_sha256") and payload["corpus_sha256"] != corpus_hash:
        raise FixtureValidationError("Fixture corpus SHA256 does not match the active corpus")
    selected_cases = cases[:max_cases] if max_cases is not None else cases
    cases_hash = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    sidecar = _Sidecar(output, grounded_run)
    prior_rows: Dict[str, Dict[str, Any]] = {}
    if resume:
        prior_rows = sidecar.load_resume(corpus_hash, cases_hash)

    from src.services import backend_app as backend, ask_flow

    llm = None
    initialization_failure = None
    setup_start = time.perf_counter()
    try:
        # Temperature goes through config LLM_TEMPERATURE (env + attr) before creation.
        os.environ["LLM_TEMPERATURE"] = repr(temperature)
        if composer:
            os.environ["GROUNDED_COMPOSER"] = composer
        if model_path is not None:
            kind = composer or provider
            os.environ["GROUNDED_LLAMA_MODEL" if kind == "llama" else "GROUNDED_MLX_MODEL"] = str(model_path)
            if kind == "llama":
                os.environ["LLM_MODEL_PATH"] = str(model_path)
        import config
        config.LLM_TEMPERATURE = temperature
        if model_path is not None and (composer or provider) == "llama":
            config.LLM_MODEL_PATH = Path(model_path)
        created = create_llm_provider(provider=provider)
        if hasattr(created, "temperature"):
            created.temperature = temperature  # default arg was bound at import time
            if created.temperature != temperature:
                raise RuntimeError("temperature did not reach the provider")
        llm = _TrackedProvider(created, temperature, composer)
    except Exception as exc:
        initialization_failure = {"stage": "initializing_provider", "error_type": type(exc).__name__}

    retriever = None
    if llm is not None:
        if retrieval == "bm25":
            retriever = BM25OnlyRetriever(docs)
        else:
            from src.pipeline import BuffettRAGPipeline, PipelineConfig
            pipeline = BuffettRAGPipeline.build(PipelineConfig(chunks_file=corpus, use_llm=False))
            retriever = pipeline.retriever
    setup_latency_ms = (time.perf_counter() - setup_start) * 1000

    old_state, old_debug = ask_flow._state, ask_flow.EXPOSE_DEBUG_STATUS
    ask_flow._state = {"retriever": retriever, "docs": docs, "docs_by_id": build_doc_lookup(docs), "llm": llm}
    ask_flow.EXPOSE_DEBUG_STATUS = False
    rows = []
    resumed_qids = []
    run_started = datetime.now(timezone.utc).isoformat()
    sidecar.write_manifest(resume=resume, started=run_started, corpus=corpus, corpus_hash=corpus_hash,
                           cases_path=cases_path, cases_hash=cases_hash, provider=provider, composer=composer,
                           model_path=model_path, temperature=temperature, retrieval=retrieval,
                           llm=llm, initialization_failure=initialization_failure)
    try:
        for case in selected_cases:
            if case["qid"] in prior_rows:
                rows.append(prior_rows[case["qid"]])
                resumed_qids.append(case["qid"])
                continue
            start = time.perf_counter()
            row = {"qid": case["qid"], "query": case["query"],
                   "history": case.get("history", []),
                   "question_type": case.get("question_type", "answerable"),
                   "answer": None, "citations": [], "passage_ids": [], "retrieved_passage_ids": [],
                   "required_claim_hits": None, "refusal_correct": None, "score": None,
                   "provider_failures": [], "provider_calls": 0, "model_requests": []}
            if initialization_failure is not None:
                row["status"] = "provider_failure"
                row["provider_failures"] = [dict(initialization_failure)]
            else:
                assert llm is not None
                failure_offset = len(llm.failures)
                request_offset = len(llm.requests)
                call_offset = llm.call_count
                raw_offset = len(llm.raw_outputs)
                grounded_offset = len(llm.grounded_calls)
                note_offset = len(llm.notes)
                req = backend.AskRequest(query=case["query"], history=case.get("history", []),
                                         strategy="hybrid", rerank=retrieval == "hybrid")
                response = backend.ask(req)
                row.update({"answer": response.answer, "citations": response.citations,
                            "passage_ids": [hit.id for hit in response.hits],
                            "retrieved_passage_ids": [hit.id for hit in response.retrieved_hits],
                            "used_filter": response.used_filter, "reranked": response.reranked,
                            "provider_calls": llm.call_count - call_offset,
                            "provider_failures": llm.failures[failure_offset:],
                            "model_requests": llm.requests[request_offset:]})
                if grounded_run:
                    row["grounded_calls"] = llm.grounded_calls[grounded_offset:]
                    row["notes"] = llm.notes[note_offset:]
                    row["context_snapshot"] = row["grounded_calls"][-1]["context"] if row["grounded_calls"] else []
                if save_raw:
                    row["raw_answers"] = [text for stage, text in llm.raw_outputs[raw_offset:] if stage == "generating"]
                if row["provider_failures"] or (response.answer or "").startswith("[LLM unavailable"):
                    row["status"] = "provider_failure"
                    if not row["provider_failures"]:
                        row["provider_failures"] = [{"stage": "generating", "error_type": "ProviderUnavailable"}]
                    # Error marker text is not a generated or scored answer.
                    row["answer"] = None
                    row["citations"] = []
                else:
                    row["status"] = "scored"
                    hits = [SearchHit(hit.id, hit.text, {"year": hit.year, "source_file": hit.source_file}, hit.score)
                            for hit in response.hits]
                    score = evaluate_live_answer(response.answer or "", hits, case)
                    row.update({"score": score, "required_claim_hits": score["claims"],
                                "refusal_correct": score["refusal_correct"]})
            row["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
            rows.append(row)
            sidecar.append_row(row)
    finally:
        ask_flow._state, ask_flow.EXPOSE_DEBUG_STATUS = old_state, old_debug
    if resume:
        sidecar.record_resume(resumed_qids, len(rows) - len(resumed_qids))

    scored = [row for row in rows if row["status"] == "scored"]
    accepted = sum(row["score"]["accepted"] for row in scored)
    unanswerable_qids = {case["qid"] for case in selected_cases if is_unanswerable_case(case)}
    return {
        "schema_version": 1,
        "mode": "offline_bm25_extractive_not_live_llm_quality" if provider == "local" and retrieval == "bm25" else "offline_bm25_embedded_llama" if provider == "llama" and retrieval == "bm25" else "in_process_ask_benchmark",
        "execution_path": "src.services.backend_app.ask (no HTTP server)",
        "live_provider_run": llm is not None and any(row["model_requests"] for row in rows) and (
            llm.provider_name == "llama" or (grounded_run and composer != "template")),
        "corpus": _relative_or_absolute(corpus), "corpus_sha256": corpus_hash,
        "cases": _relative_or_absolute(cases_path), "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "frozen": bool(payload.get("frozen", False)),
        "answer_engine": {"provider": provider, "model": llm.model if llm is not None else None,
                          "resolved_provider": llm.provider_name if llm is not None else None,
                          "temperature": temperature,
                          "temperature_applies": provider == "llama" or (grounded_run and composer != "template"),
                          **({"composer": composer, "model_path": str(model_path) if model_path else None,
                              "temperature_asserted": True} if grounded_run else {})},
        "retrieval": retrieval, "setup_latency_ms": round(setup_latency_ms, 3),
        "fixture_validation": fixture_check,
        "summary": {"fixture_cases": len(cases), "cases_run": len(rows),
                    "cases_skipped": len(cases) - len(rows), "scored_answers": len(scored),
                    "provider_failures": len(rows) - len(scored),
                    "provider_error_events": sum(len(row["provider_failures"]) for row in rows),
                    "accepted": accepted, "rejected": len(scored) - accepted,
                    "acceptance_rate": accepted / len(scored) if scored else None,
                    "unanswerable_cases": len(unanswerable_qids),
                    "unanswerable_scored": sum(row["qid"] in unanswerable_qids for row in scored),
                    "correct_refusals": sum(row["refusal_correct"] is True for row in scored),
                    "unexpected_refusals": sum(row["qid"] not in unanswerable_qids and row["answer"] == REFUSAL_LINE for row in scored),
                    "required_claims": sum(len(row["required_claim_hits"]) for row in scored),
                    "required_claims_met": sum(claim["met"] for row in scored for claim in row["required_claim_hits"]),
                    "missing_fixture_ids": fixture_check["missing_fixture_id_count"],
                    "mean_latency_ms": round(sum(row["latency_ms"] for row in rows) / len(rows), 3) if rows else None},
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--corpus", type=Path, default=CHUNKS_V3_FILE)
    parser.add_argument("--provider", choices=("llama", "local", "grounded", "mlx"), default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--composer", choices=COMPOSERS, default=None,
                        help="required with --provider grounded; there is no auto")
    parser.add_argument("--model-path", type=Path, default=None, help="absolute model path (llama/mlx/grounded)")
    parser.add_argument("--resume", action="store_true",
                        help="grounded/mlx: skip qids already in <output>.rows.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--retrieval", choices=("bm25", "hybrid"), default="hybrid")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-raw", action="store_true", help="store each raw (pre-validation) model answer per case as rows[].raw_answers")
    args = parser.parse_args()
    try:
        result = run(args.corpus, args.cases, provider=args.provider,
                     temperature=args.temperature, max_cases=args.max_cases, retrieval=args.retrieval, save_raw=args.save_raw,
                     composer=args.composer, model_path=args.model_path, output=args.output, resume=args.resume)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except FixtureValidationError as exc:
        # Only our fixture errors can be printed; provider exception bodies
        # and arbitrary SDK/retrieval errors are never rendered here.
        print(f"Benchmark validation failed: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        if getattr(exc, "args", None) and str(exc).startswith("--"):
            print(f"Benchmark arguments invalid: {exc}", file=sys.stderr)
            return 2
        print(f"Benchmark could not complete ({type(exc).__name__}); no answer-quality result claimed.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Benchmark could not complete ({type(exc).__name__}); no answer-quality result claimed.", file=sys.stderr)
        return 2
    print(json.dumps({"summary": result["summary"], "fixture_validation": result["fixture_validation"]}, indent=2))
    return 0 if result["summary"]["provider_failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
