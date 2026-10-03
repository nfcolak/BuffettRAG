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
import time
from pathlib import Path
from typing import Any, Dict, Optional

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
) -> Dict[str, Any]:
    """Validate every fixture first, then call backend.ask without HTTP/startup."""
    if retrieval not in {"bm25", "hybrid"}:
        raise ValueError("retrieval must be bm25 or hybrid")
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be finite and between 0 and 2")
    if max_cases is not None and max_cases < 1:
        raise ValueError("max_cases must be positive")
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

    from src.services import backend_app as backend, ask_flow

    llm = None
    initialization_failure = None
    setup_start = time.perf_counter()
    try:
        # Temperature goes through config LLM_TEMPERATURE (env + attr) before creation.
        os.environ["LLM_TEMPERATURE"] = repr(temperature)
        import config
        config.LLM_TEMPERATURE = temperature
        created = create_llm_provider(provider=provider)
        if hasattr(created, "temperature"):
            created.temperature = temperature  # default arg was bound at import time
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
            if initialization_failure is not None:
                row["status"] = "provider_failure"
                row["provider_failures"] = [dict(initialization_failure)]
            else:
                assert llm is not None
                failure_offset = len(llm.failures)
                request_offset = len(llm.requests)
                call_offset = llm.call_count
                raw_offset = len(llm.raw_outputs)
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

    scored = [row for row in rows if row["status"] == "scored"]
    accepted = sum(row["score"]["accepted"] for row in scored)
    unanswerable_qids = {case["qid"] for case in selected_cases if is_unanswerable_case(case)}
    return {
        "schema_version": 1,
        "mode": "offline_bm25_extractive_not_live_llm_quality" if provider == "local" and retrieval == "bm25" else "offline_bm25_embedded_llama" if provider == "llama" and retrieval == "bm25" else "in_process_ask_benchmark",
        "execution_path": "src.services.backend_app.ask (no HTTP server)",
        "live_provider_run": llm is not None and llm.provider_name == "llama" and any(row["model_requests"] for row in rows),
        "corpus": _relative_or_absolute(corpus), "corpus_sha256": corpus_hash,
        "cases": _relative_or_absolute(cases_path), "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "frozen": bool(payload.get("frozen", False)),
        "answer_engine": {"provider": provider, "model": llm.model if llm is not None else None,
                          "resolved_provider": llm.provider_name if llm is not None else None,
                          "temperature": temperature, "temperature_applies": provider == "llama"},
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
    parser.add_argument("--provider", choices=("llama", "local"), default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--retrieval", choices=("bm25", "hybrid"), default="hybrid")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-raw", action="store_true", help="store each raw (pre-validation) model answer per case as rows[].raw_answers")
    args = parser.parse_args()
    try:
        result = run(args.corpus, args.cases, provider=args.provider,
                     temperature=args.temperature, max_cases=args.max_cases, retrieval=args.retrieval, save_raw=args.save_raw)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except FixtureValidationError as exc:
        # Only our fixture errors can be printed; provider exception bodies
        # and arbitrary SDK/retrieval errors are never rendered here.
        print(f"Benchmark validation failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Benchmark could not complete ({type(exc).__name__}); no answer-quality result claimed.", file=sys.stderr)
        return 2
    print(json.dumps({"summary": result["summary"], "fixture_validation": result["fixture_validation"]}, indent=2))
    return 0 if result["summary"]["provider_failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
