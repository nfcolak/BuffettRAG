"""TDD contract tests for the sequential answer-quality programme."""
from __future__ import annotations

import json
from pathlib import Path

from src.vector_store import SearchHit, load_chunks_as_docs


ROOT = Path(__file__).resolve().parents[1]


def _hit(hid: str, text: str, year: int = 2008) -> SearchHit:
    return SearchHit(hid, text, {"year": year, "source_file": f"buffet_{year}.pdf"}, 1.0)


def test_answer_benchmark_accepts_all_gold_claims_with_valid_passages():
    from src.evaluation.answer_benchmark import evaluate_answer, validate_benchmark_case

    case = {
        "qid": "unit-1", "query": "What happened?", "gold_passage_ids": ["2008_p1"],
        "gold_claims": [{"claim": "Berkshire bought preferred shares.", "required_terms": ["preferred", "shares"]}],
        "accept": {"min_claim_coverage": 1.0, "require_valid_citation": True},
        "reject": {"forbidden_terms": ["moon cheese"]},
    }
    validate_benchmark_case(case)
    result = evaluate_answer("Berkshire bought preferred shares. [1]", [_hit("2008_p1", "Berkshire bought preferred shares.")], case)
    assert result["accepted"] is True
    assert result["claim_coverage"] == 1.0


def test_claim_validator_splits_compound_claim_and_blocks_unsupported_part():
    from src.evaluation.claim_validator import validate_and_filter_answer

    answer = "Berkshire bought preferred shares and guaranteed moon cheese. [1]"
    result = validate_and_filter_answer(answer, [_hit("p1", "Berkshire bought preferred shares.")])
    assert result.blocked_claims == ["guaranteed moon cheese"]
    assert result.safe_answer == "Berkshire bought preferred shares. [1]"


def test_paragraph_chunking_records_source_provenance_and_stable_ids(tmp_path):
    from src.ingestion.paragraph_index import build_paragraph_records

    source = tmp_path / "buffet_2024.txt"
    source.write_text("A heading\n\nFirst evidence paragraph.\n\nSecond evidence paragraph.", encoding="utf-8")
    records = build_paragraph_records(source, year=2024, chunk_size=80, overlap=0, min_chunk_chars=1)
    assert records[0]["id"] == "2024_p0000"
    assert records[0]["provenance"]["source_sha256"]
    assert records[0]["provenance"]["paragraph_count"] >= 1


def test_hard_negative_case_requires_relevant_ahead_of_named_decoy():
    from src.evaluation.hard_negatives import score_hard_negative_case

    case = {"qid": "hn1", "relevant_ids": ["right"], "hard_negative_ids": ["wrong"]}
    assert score_hard_negative_case(case, ["right", "wrong"])["passed"] is True
    assert score_hard_negative_case(case, ["wrong", "right"])["passed"] is False


def test_v3_bm25_ranks_every_curated_passage_ahead_of_its_decoy():
    from src.evaluation.hard_negatives import score_hard_negative_case
    from src.retrieval.bm25 import BM25Retriever

    cases_path = ROOT / "data/evaluation/answer_quality_program/hard_negatives_v3.json"
    corpus_path = ROOT / "data/processed/chunks_v3_paragraph.jsonl"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    retriever = BM25Retriever(load_chunks_as_docs(corpus_path))

    failed = []
    for case in cases:
        ranked_ids = [hit.id for hit in retriever.search(case["query"], top_k=20)]
        result = score_hard_negative_case(case, ranked_ids)
        if not result["passed"]:
            failed.append(result)

    assert failed == []


def test_runtime_defaults_to_v3_corpus():
    from config import CHUNKS_V3_FILE, FAISS_DIR
    from src.pipeline import PipelineConfig
    from src.services.backend_app import _resolve_chunks_path

    assert CHUNKS_V3_FILE.name == "chunks_v3_paragraph.jsonl"
    assert FAISS_DIR.name == "faiss_v3"
    assert PipelineConfig().chunks_file == CHUNKS_V3_FILE
    assert _resolve_chunks_path() == CHUNKS_V3_FILE


def test_index_manifest_binds_corpus_model_and_document_ids(tmp_path):
    from scripts.build_index import build_index_manifest
    from src.vector_store import StoredDoc

    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text('{"id":"a","text":"evidence","year":2024}\n', encoding="utf-8")
    docs = [StoredDoc("a", "evidence", {"year": 2024})]

    manifest = build_index_manifest(
        corpus=corpus,
        docs=docs,
        backend="faiss",
        model_name="test-embedder",
        dimension=3,
    )

    assert manifest["corpus_sha256"]
    assert manifest["document_ids_sha256"]
    assert manifest["document_count"] == 1
    assert manifest["embedding"] == {"model": "test-embedder", "dimension": 3}


def test_index_manifest_rejects_tampered_or_extra_artifacts(tmp_path):
    from src.index_manifest import load_and_validate_index_manifest, write_index_manifest
    from src.vector_store import StoredDoc

    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text('{"id":"a"}\n', encoding="utf-8")
    docs = [StoredDoc(id="a", text="alpha", metadata={})]
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_bytes(b"index")
    (index_dir / "meta.json").write_text("[]", encoding="utf-8")
    write_index_manifest(
        index_dir, corpus=corpus, docs=docs, backend="faiss",
        model_name="model", dimension=3, artifact_names=("index.faiss", "meta.json"),
    )
    load_and_validate_index_manifest(
        index_dir, corpus=corpus, docs=docs, backend="faiss",
        model_name="model", dimension=3,
    )
    (index_dir / "meta.json").write_text("[{}]", encoding="utf-8")
    try:
        load_and_validate_index_manifest(
            index_dir, corpus=corpus, docs=docs, backend="faiss",
            model_name="model", dimension=3,
        )
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered metadata must be rejected")


def test_ablation_rejects_index_built_for_different_corpus(tmp_path):
    from scripts.run_ablation import validate_index_manifest

    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text('{"id":"a","text":"evidence"}\n', encoding="utf-8")
    manifest = {
        "corpus_sha256": "0" * 64,
        "document_ids_sha256": "0" * 64,
        "document_count": 1,
        "embedding": {"model": "test", "dimension": 3},
    }

    try:
        validate_index_manifest(manifest, corpus=corpus, document_ids=["a"],
                                model_name="test", dimension=3)
    except ValueError as exc:
        assert "corpus hash" in str(exc)
    else:
        raise AssertionError("stale index manifest must be rejected")


def test_ablation_orchestrator_marks_missing_optional_models_unavailable():
    from src.evaluation.ablation import benchmark_configurations

    cases = [{"qid": "case", "gold_passage_ids": ["a"]}]
    report = benchmark_configurations(cases, runners={"bm25": lambda _: (["a"], 1.0)})
    assert report["configurations"]["embedding_only"]["status"] == "unavailable"
    assert report["configurations"]["bm25"]["status"] == "ok"
    assert report["configurations"]["bm25"]["summary"]["recall_at_8"] == 1.0
    assert report["configurations"]["bm25"]["summary"]["mean_latency_ms"] == 1.0


def test_structured_expansion_preserves_year_and_named_entity():
    from src.retrieval.query_expansion import expand_query_structured

    class FakeLLM:
        provider_name = "test"
        def generate(self, *_args, **_kwargs):
            return '{"terms":["investment"],"entities":["Apple"],"years":[2020]}'

    expansion = expand_query_structured("What did Apple do in 2020?", FakeLLM())
    assert expansion is not None
    assert expansion.entities == ["Apple"]
    assert expansion.years == [2020]
    assert "Apple" in expansion.retrieval_query and "2020" in expansion.retrieval_query


def test_evidence_gate_refuses_when_hits_do_not_support_query():
    from src.generation.evidence_gate import assess_evidence

    result = assess_evidence("What did Buffett say about moon cheese?", [_hit("p", "Berkshire insurance operations produced float.")])
    assert result.sufficient is False
    assert result.reason == "no_lexical_evidence"


def test_pipeline_blocks_generation_when_evidence_gate_fails():
    from src.pipeline import BuffettRAGPipeline
    from src.retrieval.retriever import RetrievalResult

    class FixedRetriever:
        def search(self, query, **_kwargs):
            return RetrievalResult(query, "hybrid", [_hit("p", "Insurance float was valuable.")])

    class ShouldNotRun:
        def generate(self, *_args, **_kwargs):
            raise AssertionError("generation must be blocked before provider call")

    result = BuffettRAGPipeline(FixedRetriever(), {}, ShouldNotRun()).ask("What did Buffett say about moon cheese?")
    from src.generation.prompt import REFUSAL_LINE
    assert result["answer"] == REFUSAL_LINE
    assert result["evidence"]["reason"] == "no_lexical_evidence"


def test_pipeline_post_validation_blocks_unsupported_cited_claim():
    from src.pipeline import BuffettRAGPipeline
    from src.retrieval.retriever import RetrievalResult

    class FixedRetriever:
        def search(self, query, **_kwargs):
            return RetrievalResult(query, "hybrid", [_hit("p", "Berkshire bought preferred shares.")])

    class FakeLLM:
        def generate(self, *_args, **_kwargs):
            return "Berkshire bought preferred shares and guaranteed moon cheese. [1]"

    result = BuffettRAGPipeline(FixedRetriever(), {}, FakeLLM()).ask("preferred shares")
    assert "moon cheese" not in result["answer"]
    assert result["citation_validation"]["blocked_claims"] == ["guaranteed moon cheese"]


def test_backend_generation_gate_blocks_provider_before_generation():
    from src.services.backend_app import _generate_answer
    from src.vector_store import SearchHit

    class ShouldNotRun:
        def generate(self, *_args, **_kwargs):
            raise AssertionError("provider must not run")

    answer, citations = _generate_answer(ShouldNotRun(), "unused", [_hit("p", "Insurance float was valuable.")], 50,
                                         query="moon cheese")
    assert "enough evidence" in answer
    assert citations == []


def test_streaming_gate_does_not_open_provider_when_evidence_is_insufficient(monkeypatch):
    import json
    from fastapi.testclient import TestClient
    from src.services import backend_app as backend
    from src.retrieval.retriever import RetrievalResult

    calls = []
    class FixedRetriever:
        def search(self, **kwargs):
            return RetrievalResult(kwargs["query"], "hybrid", [_hit("p", "Insurance float was valuable.")])
    class CaptureLLM:
        def generate(self, *_args, **_kwargs):
            calls.append(True)
            return "Moon cheese. [1]"

    monkeypatch.setattr(backend, "_state", {"retriever": FixedRetriever(), "docs_by_id": {}, "llm": CaptureLLM()})
    monkeypatch.setattr(backend, "API_KEYS", ())
    response = TestClient(backend.app).post("/ask/stream", json={"query": "moon cheese", "expand_query": False})
    assert response.status_code == 200
    assert calls == []
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert "enough evidence" in events[-1]["answer"]


def test_streaming_never_emits_raw_unvalidated_llm_deltas(monkeypatch):
    import json
    from fastapi.testclient import TestClient
    from src.services import backend_app as backend
    from src.retrieval.retriever import RetrievalResult

    class FixedRetriever:
        def search(self, **kwargs):
            return RetrievalResult(kwargs["query"], "hybrid", [_hit("p", "Berkshire bought preferred shares.")])
    class RawStreamingLLM:
        def generate_stream(self, *_args, **_kwargs):
            yield "Berkshire bought preferred shares and guaranteed moon cheese. [1]"

    monkeypatch.setattr(backend, "_state", {"retriever": FixedRetriever(), "docs_by_id": {}, "llm": RawStreamingLLM()})
    monkeypatch.setattr(backend, "API_KEYS", ())
    response = TestClient(backend.app).post("/ask/stream", json={"query": "preferred shares", "expand_query": False})
    assert "moon cheese" not in response.text
    assert "event: delta" not in response.text


def test_v3_offline_answer_benchmark_accepts_all_curated_cases():
    from scripts.run_answer_benchmark import run

    result = run(
        Path("data/processed/chunks_v3_paragraph.jsonl"),
        Path("data/evaluation/answer_quality_program/answer_benchmark_v3.json"),
    )
    assert result["n_cases"] == 8
    assert result["accepted"] == 8
    assert len(result["corpus_sha256"]) == 64
    assert len(result["cases_sha256"]) == 64
    assert result["answer_engine"] == {"provider": "local", "model": "embedded-extractive-v1"}


def test_backend_prefers_structured_expansion_without_replacing_original_query(monkeypatch):
    from types import SimpleNamespace
    from src.services import backend_app as backend
    from src.retrieval.retriever import RetrievalResult

    seen = []
    class FixedRetriever:
        def search(self, **kwargs):
            seen.append(kwargs.get("retrieval_query"))
            return RetrievalResult(kwargs["query"], "hybrid", [_hit("p", "Apple repurchases in 2020.", 2020)])
    class Local:
        provider_name = "local"
    monkeypatch.setattr(backend, "_state", {"retriever": FixedRetriever(), "docs_by_id": {}, "llm": Local()})
    monkeypatch.setattr(backend, "expand_query_structured", lambda *_args, **_kwargs: SimpleNamespace(retrieval_query="What did Apple do in 2020? investment Apple 2020"))
    backend._prepare_ask(backend.AskRequest(query="What did Apple do in 2020?"))
    assert seen == ["What did Apple do in 2020? investment Apple 2020"]


def test_demo_security_rejects_private_backend_and_enforces_request_budget(monkeypatch):
    import socket
    from src.services.demo_security import DemoRequestBudget, validate_demo_backend_url

    monkeypatch.setattr(socket, "getaddrinfo", lambda *_args, **_kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
    ])
    assert validate_demo_backend_url("https://demo.example.com", allowed_hosts={"demo.example.com"}) == "https://demo.example.com"
    assert validate_demo_backend_url("http://127.0.0.1:8000", allowed_hosts={"127.0.0.1"}) is None
    assert validate_demo_backend_url("https://2130706433", allowed_hosts={"2130706433"}) is None
    assert validate_demo_backend_url("https://foo.localhost", allowed_hosts={"foo.localhost"}) is None
    assert validate_demo_backend_url("https://other.example.com", allowed_hosts={"demo.example.com"}) is None
    budget = DemoRequestBudget(max_requests=1, max_tokens=10)
    assert budget.allow("session", tokens=5) is True
    assert budget.allow("session", tokens=5) is False


def test_backend_request_validation_rejects_unsafe_payloads(monkeypatch):
    from pydantic import ValidationError
    from src.services import backend_app as backend

    for query in (" ", "\n\t"):
        try:
            backend.AskRequest(query=query)
        except ValidationError:
            pass
        else:
            raise AssertionError("whitespace-only query must be rejected")

    try:
        backend.AskRequest(query="evidence", where={"year": {"$in": list(range(100))}})
    except ValidationError:
        pass
    else:
        raise AssertionError("oversized metadata filter must be rejected")

    monkeypatch.setattr(backend, "PUBLIC_DEMO_MODE", True)
    monkeypatch.setattr(backend, "ALLOW_LLM_REQUEST_OVERRIDES", True)
    for field, value in (("llm_provider", "openai"), ("llm_model", "expensive"),
                         ("llm_api_key", "visitor-key")):
        try:
            backend.AskRequest.model_validate({"query": "evidence", field: value})
        except ValidationError:
            pass
        else:
            raise AssertionError(f"public demo must reject {field}")

    monkeypatch.setattr(backend, "PUBLIC_DEMO_MODE", False)
    monkeypatch.setattr(backend, "ALLOW_LLM_REQUEST_OVERRIDES", False)
    for field, value in (("llm_provider", "openai"), ("llm_model", "expensive"),
                         ("llm_api_key", "visitor-key")):
        try:
            backend.AskRequest.model_validate({"query": "evidence", field: value})
        except ValidationError:
            pass
        else:
            raise AssertionError(f"disabled request overrides must reject {field}")


def test_public_backend_requires_auth_and_reports_readiness(monkeypatch):
    from fastapi.testclient import TestClient
    from src.services import backend_app as backend

    monkeypatch.setattr(backend, "PUBLIC_DEMO_MODE", True)
    monkeypatch.setattr(backend, "API_KEYS", ())
    monkeypatch.setattr(backend, "_state", {})
    client = TestClient(backend.app)

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.get("/stats").status_code == 401


def test_public_security_configuration_fails_closed_without_key():
    from src.services.backend_app import validate_deployment_security

    try:
        validate_deployment_security(public_demo=True, api_keys=(), cors_origins=(), debug=False)
    except RuntimeError as exc:
        assert "API_KEYS" in str(exc)
    else:
        raise AssertionError("public backend without API keys must fail closed")

    try:
        validate_deployment_security(
            public_demo=True, api_keys=("key",), cors_origins=(), debug=False,
            trust_proxy_headers=True,
        )
    except RuntimeError as exc:
        assert "trusted proxy" in str(exc)
    else:
        raise AssertionError("public backend must not trust forwarding headers")

    try:
        validate_deployment_security(
            public_demo=True, api_keys=("key",), cors_origins=(), debug=False,
            allow_llm_request_overrides=True,
        )
    except RuntimeError as exc:
        assert "LLM request overrides" in str(exc)
    else:
        raise AssertionError("public backend must not allow LLM request overrides")


def test_backend_rejects_oversized_body_before_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    from src.services import backend_app as backend

    monkeypatch.setattr(backend, "API_KEYS", ())
    monkeypatch.setattr(backend, "MAX_REQUEST_BODY_BYTES", 1024)
    response = TestClient(backend.app).post(
        "/ask", content=b"x" * 1025, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413


def test_demo_token_reservation_includes_context_and_output_allowance():
    from src.services.demo_security import estimate_token_reservation

    reserved = estimate_token_reservation("short question", max_output_tokens=500,
                                          context_token_allowance=1500)
    assert reserved >= 2000


def test_claim_validator_rejects_negation_number_and_citation_line_smuggling():
    from src.evaluation.claim_validator import validate_and_filter_answer

    hits = [
        _hit("p1", "Berkshire did not buy preferred shares. Revenue was $10."),
        _hit("p2", "The moon is not discussed in this evidence."),
    ]
    answer = (
        "Berkshire bought preferred shares. [1]\n"
        "Revenue was $12. [1]\n"
        "The moon is cheese [1]"
    )
    result = validate_and_filter_answer(answer, hits)
    assert result.safe_answer == ""
    assert len(result.blocked_claims) == 3

    for claim, evidence in (
        ("Berkshire did buy preferred shares. [1]", "Berkshire didn't buy preferred shares."),
        ("Berkshire bought preferred shares. [1]", "Berkshire sold preferred shares."),
    ):
        contradicted = validate_and_filter_answer(claim, [_hit("p", evidence)])
        assert contradicted.safe_answer == ""


def test_answer_benchmark_requires_claim_level_gold_citation_and_polarity():
    from src.evaluation.answer_benchmark import evaluate_answer

    case = {
        "qid": "q", "query": "q", "gold_passage_ids": ["gold"],
        "gold_claims": [{"claim": "Revenue was $10", "required_terms": ["revenue", "$10"]}],
        "accept": {"min_claim_coverage": 1.0, "require_valid_citation": True},
        "reject": {"forbidden_terms": []},
    }
    hits = [_hit("gold", "Revenue was $10."), _hit("wrong", "Costs were $10.")]
    assert not evaluate_answer("Revenue was $10. [2] Unrelated. [1]", hits, case)["accepted"]
    assert not evaluate_answer("Revenue was not $10. [1]", hits, case)["accepted"]
    assert evaluate_answer("Revenue was $10. [1]", hits, case)["accepted"]

    paired_case = {
        "qid": "paired", "query": "q", "gold_passage_ids": ["revenue", "costs"],
        "gold_claims": [
            {"claim": "Revenue was $10", "required_terms": ["revenue", "$10"],
             "gold_passage_ids": ["revenue"]},
            {"claim": "Costs were $7", "required_terms": ["costs", "$7"],
             "gold_passage_ids": ["costs"]},
        ],
        "accept": {"min_claim_coverage": 1.0, "require_valid_citation": True},
        "reject": {"forbidden_terms": []},
    }
    paired_hits = [_hit("revenue", "Revenue was $10."), _hit("costs", "Costs were $7.")]
    assert not evaluate_answer("Revenue was $10. [2] Costs were $7. [1]", paired_hits, paired_case)["accepted"]
    assert evaluate_answer("Revenue was $10. [1] Costs were $7. [2]", paired_hits, paired_case)["accepted"]


def test_local_provider_honors_tiny_output_budget():
    from src.generation.providers.local_provider import LocalProvider

    long_sentence = "Insurance " + ("performed strongly " * 300) + "."
    prompt = (
        "BEGIN USER QUESTION\nQuestion: How did insurance perform?\n\n"
        f"[1] (year=2024)\n{long_sentence}\n\nEND UNTRUSTED PASSAGES"
    )
    answer = LocalProvider().generate(prompt, max_new_tokens=1)
    assert len(answer) <= 4 or "enough evidence" in answer

    bounded_prompt = (
        "BEGIN USER QUESTION\nQuestion: How did insurance performance improve?\n\n"
        "[1] (year=2024)\n"
        "Insurance performance improved improved . "
        "Insurance performance improved improved improved .\n\n"
        "END UNTRUSTED PASSAGES"
    )
    bounded = LocalProvider().generate(bounded_prompt, max_new_tokens=25)
    assert len(bounded) <= 25 * 4


def test_demo_evidence_html_escapes_backend_supplied_text():
    from src.services.demo_security import render_evidence_html

    rendered = render_evidence_html({
        "year": "2024<script>alert(1)</script>",
        "source_file": "letter <img src=x>",
        "text": "Evidence <script>alert(1)</script>",
    }, 1)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "<img" not in rendered
