"""Run the real /ask path in-process; BM25 mode needs no downloaded models.

`llama` is the embedded GGUF model
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
import subprocess
import time
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
from src.generation.decompose import split_question
from src.generation.prompt import REFUSAL_LINE, check_prompt_matches_model, expected_prompt_version, prompt_version
from src.retrieval.context import build_doc_lookup
from src.retrieval.retriever import Retriever
from src.storage import SearchHit, load_chunks_as_docs

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
        hit_id = getattr(hit, "id", None)
        out.append({"id": hit_id, "year": year if year is not None else meta.get("year"),
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "text": text,
                    "merged_ids": list(meta.get("merged_ids") or [hit_id])})
    return out


class _TrackedProvider:
    """Capture provider errors even when /ask or query expansion swallows them."""

    def __init__(self, provider, temperature):
        self._provider = provider
        self.provider_name = provider.provider_name
        self.model = provider.model
        self.failures = []
        self.requests = []
        self.call_count = 0
        self.temperature = temperature
        self.raw_outputs = []  # (stage, raw model text) for --save-raw; pre-validation

    def __getattr__(self, name):
        # Forward optional provider hooks.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._provider, name)

    def generate(self, prompt, max_new_tokens=None):
        self.call_count += 1
        stage = "expanding" if prompt.startswith(("Return JSON only:", "You expand search queries")) else "generating"
        self.requests.append({"model": self.model, "temperature": self.temperature})
        try:
            output = self._provider.generate(prompt, max_new_tokens=max_new_tokens)
            self.raw_outputs.append((stage, output))
            return output
        except Exception as exc:
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


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_sha256(path: Path) -> Optional[str]:
    """sha256 of a model file, cached on disk by (path, size, mtime) so a 4.7 GB gguf is hashed once."""
    path = Path(path)
    if not path.is_file():
        return None
    cache_file = Path(os.environ.get("BUFFETT_SHA_CACHE") or Path.home() / ".cache/buffettrag/sha256_cache.json")
    stat = path.stat()
    key = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    if key not in cache:
        cache[key] = _sha256_of_file(path)
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        except OSError:
            pass
    return cache[key]


def _run_settings(*, corpus: Path, provider: str, model_path: Optional[Path], temperature: float) -> Dict[str, Any]:
    """What was actually served: every setting that changes the answer, at the moment the run starts."""
    import config
    from src.generation.answer_expand import _setting as expansion_setting

    if model_path is None and provider == "llama":
        model_path = Path(os.environ.get("LLM_MODEL_PATH") or config.LLM_MODEL_PATH)
    explicit = (os.environ.get("PROMPT_VERSION") or "").strip()
    version = prompt_version() if explicit or model_path is None else (
        expected_prompt_version(model_path) or prompt_version())
    corpus = Path(corpus)
    return {
        "prompt_version": version,
        "model": {"path": str(model_path) if model_path else None,
                  "sha256": _model_sha256(model_path) if model_path else None},
        "LLM_CONTEXT_PASSAGES": config.llm_context_passages(),
        "n_ctx": config.llm_n_ctx(),
        "temperature": temperature,
        "ANSWER_NEIGHBOR_BEFORE": expansion_setting("ANSWER_NEIGHBOR_BEFORE"),
        "ANSWER_NEIGHBOR_AFTER": expansion_setting("ANSWER_NEIGHBOR_AFTER"),
        "ANSWER_MAX_SENTENCES": expansion_setting("ANSWER_MAX_SENTENCES"),
        "ANSWER_EXPANDED_MAX_SENTENCES": expansion_setting("ANSWER_EXPANDED_MAX_SENTENCES"),
        "ANSWER_REFUSAL_RETRY": os.environ.get("ANSWER_REFUSAL_RETRY", str(config.ANSWER_REFUSAL_RETRY)),
        "COMPARE_ROUTING_V2": config.retrieval_setting("COMPARE_ROUTING_V2"),
        "LLM_PASSAGE_ORDER": (os.environ.get("LLM_PASSAGE_ORDER") or "natural").strip().lower(),
        "corpus": {"file": str(corpus), "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest()},
        "faiss_index_name": config.FAISS_DIR.name,
        "git_commit": _git("rev-parse", "HEAD"),
    }


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
    model_path: Optional[Path] = None,
    qids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate every fixture first, then call backend.ask without HTTP/startup."""
    if retrieval not in {"bm25", "hybrid"}:
        raise ValueError("retrieval must be bm25 or hybrid")
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be finite and between 0 and 2")
    if max_cases is not None and max_cases < 1:
        raise ValueError("max_cases must be positive")
    if model_path is not None:
        if provider != "llama":
            raise ValueError("--model-path applies only to the llama provider")
        if not Path(model_path).is_absolute():
            raise ValueError("--model-path must be absolute")
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
    if qids:
        unknown = sorted(set(qids) - {case["qid"] for case in cases})
        if unknown:
            raise FixtureValidationError(f"--qids not in the cases file: {unknown}")
        selected_cases = [case for case in selected_cases if case["qid"] in set(qids)]
    cases_hash = hashlib.sha256(cases_path.read_bytes()).hexdigest()

    from src.services import backend_app as backend, ask_flow

    # PROMPT_VERSION must not contradict the model's training prompt (raises; ALLOW_PROMPT_MISMATCH=1 overrides).
    import config as _config
    check_prompt_matches_model(model_path if model_path is not None and provider == "llama"
                               else _config.LLM_MODEL_PATH if provider == "llama" else "")
    llm = None
    initialization_failure = None
    setup_start = time.perf_counter()
    try:
        # Temperature goes through config LLM_TEMPERATURE (env + attr) before creation.
        os.environ["LLM_TEMPERATURE"] = repr(temperature)
        if model_path is not None:
            os.environ["LLM_MODEL_PATH"] = str(model_path)
        import config
        config.LLM_TEMPERATURE = temperature
        if model_path is not None:
            config.LLM_MODEL_PATH = Path(model_path)
        created = create_llm_provider(provider=provider)
        if hasattr(created, "temperature"):
            created.temperature = temperature  # default arg was bound at import time
            if created.temperature != temperature:
                raise RuntimeError("temperature did not reach the provider")
        llm = _TrackedProvider(created, temperature)
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
    old_prepare = ask_flow._prepare_ask
    served_context: List[Any] = []  # context_hits exactly as handed to the model (merge + truncation applied)

    def _capturing_prepare(req):
        prepared = old_prepare(req)
        served_context[:] = list(prepared[4])
        return prepared

    ask_flow._prepare_ask = _capturing_prepare
    ask_flow._state = {"retriever": retriever, "docs": docs, "docs_by_id": build_doc_lookup(docs), "llm": llm}
    ask_flow.EXPOSE_DEBUG_STATUS = False
    rows = []
    try:
        for case in selected_cases:
            start = time.perf_counter()
            row = {"qid": case["qid"], "query": case["query"],
                   "history": case.get("history", []),
                   "question_type": case.get("question_type", "answerable"),
                   "answer": None, "citations": [], "passage_ids": [], "retrieved_passage_ids": [],
                   "required_claim_hits": None, "refusal_correct": None, "score": None,
                   "provider_failures": [], "provider_calls": 0, "model_requests": []}
            if os.getenv("ANSWER_DECOMPOSE", "0") == "1":
                row["decompose_parts"] = split_question(case["query"])
            if initialization_failure is not None:
                row["status"] = "provider_failure"
                row["provider_failures"] = [dict(initialization_failure)]
            else:
                assert llm is not None
                failure_offset = len(llm.failures)
                request_offset = len(llm.requests)
                call_offset = llm.call_count
                raw_offset = len(llm.raw_outputs)
                served_context.clear()
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
                row["context_snapshot"] = _context_snapshot(served_context)
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
    finally:
        ask_flow._state, ask_flow.EXPOSE_DEBUG_STATUS = old_state, old_debug
        ask_flow._prepare_ask = old_prepare

    scored = [row for row in rows if row["status"] == "scored"]
    accepted = sum(row["score"]["accepted"] for row in scored)
    unanswerable_qids = {case["qid"] for case in selected_cases if is_unanswerable_case(case)}
    return {
        "schema_version": 1,
        "mode": "offline_bm25_embedded_llama" if provider == "llama" and retrieval == "bm25" else "in_process_ask_benchmark",
        "execution_path": "src.services.backend_app.ask (no HTTP server)",
        "live_provider_run": llm is not None and any(row["model_requests"] for row in rows) and (
            llm.provider_name == "llama"),
        "corpus": _relative_or_absolute(corpus), "corpus_sha256": corpus_hash,
        "cases": _relative_or_absolute(cases_path), "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "frozen": bool(payload.get("frozen", False)),
        "answer_engine": {"provider": provider, "model": llm.model if llm is not None else None,
                          "resolved_provider": llm.provider_name if llm is not None else None,
                          "temperature": temperature,
                          "temperature_applies": provider == "llama"},
        "settings": _run_settings(corpus=corpus, provider=provider, model_path=model_path, temperature=temperature),
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
    parser.add_argument("--provider", choices=("llama",), default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--model-path", type=Path, default=None, help="absolute llama model path")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--qids", default=None, help="comma-separated qids; only these cases are run")
    parser.add_argument("--retrieval", choices=("bm25", "hybrid"), default="hybrid")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-raw", action="store_true", help="store each raw (pre-validation) model answer per case as rows[].raw_answers")
    args = parser.parse_args()
    try:
        result = run(args.corpus, args.cases, provider=args.provider,
                     temperature=args.temperature, max_cases=args.max_cases, retrieval=args.retrieval, save_raw=args.save_raw,
                     model_path=args.model_path,
                     qids=[q.strip() for q in args.qids.split(",") if q.strip()] if args.qids else None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except FixtureValidationError as exc:
        # Only our fixture errors can be printed; provider exception bodies
        # and arbitrary SDK/retrieval errors are never rendered here.
        print(f"Benchmark validation failed: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        if getattr(exc, "args", None) and str(exc).startswith(("--", "PROMPT_VERSION")):
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
