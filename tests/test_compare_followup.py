"""Synthetic contracts for temporal isolation and history-aware retrieval."""
from __future__ import annotations

import json

import pytest

from src.generation.compare import generate_comparison_answer, prepare_answer_context
from src.generation.prompt import REFUSAL_LINE, build_cited_prompt
from src.retrieval.context import build_doc_lookup, expand_hits_with_neighbors, fit_context_to_llm
from src.retrieval.query_expansion import build_followup_retrieval_query
from src.retrieval.retriever import Retriever, detect_temporal_comparison
from src.storage import SearchHit, StoredDoc


def hit(name, text, year):
    return SearchHit(name, text, {"year": year, "source_file": f"letter-{year}"}, 1.0)


EARLY = "Insurance float was free when underwriting results broke even."
LATE = "Insurance float had a negative cost when underwriting produced a profit."
QUERY = "How did insurance float costs differ in 2041 and 2053?"


def test_detects_all_years_without_comparison_verb_and_mixed_decades():
    assert detect_temporal_comparison("Discuss margins in 2041, 2053 and 2067") == [
        {"year": 2041}, {"year": 2053}, {"year": 2067},
    ]
    assert detect_temporal_comparison("Margins in the 2040s and 2053") == [
        {"year": {"$gte": 2040, "$lte": 2049}}, {"year": 2053},
    ]
    assert detect_temporal_comparison("2041 and again 2041") is None


@pytest.mark.parametrize("strategy", ["bm25", "hybrid"])
def test_retrieval_reserves_periods_even_with_biased_reranker(strategy):
    docs = [StoredDoc("early", EARLY, {"year": 2041}),
            StoredDoc("late", LATE, {"year": 2053}),
            StoredDoc("late-other", "Insurance float and underwriting premium volume rose.", {"year": 2053})]
    class LexicalHybrid(Retriever):
        def _hybrid_for_filter(self, query, fetch_k, where):
            return self.bm25.search(query, top_k=fetch_k, where=where)
    class BiasedReranker:
        def rerank(self, query, hits, top_k):
            return sorted(hits, key=lambda item: item.metadata["year"], reverse=True)[:top_k]
    retriever = LexicalHybrid(None, None, docs, BiasedReranker())
    result = retriever.search(QUERY, strategy=strategy, top_k=1, rerank=True)
    assert {item.metadata["year"] for item in result.hits} == {2041, 2053}
    assert len(result.used_filter["multi_subquery"]) == 2


def test_llm_cut_trims_other_passages_before_reserved_periods():
    passages = [hit("a", EARLY * 60, 2041), hit("b", "Distracting insurance data. " * 60, 2041),
                hit("c", LATE * 60, 2053)]
    periods = detect_temporal_comparison(QUERY)
    fitted = fit_context_to_llm(passages, passages, QUERY, max_new_tokens=100, n_ctx=1800,
                                max_passages=1, passage_max_chars=1000, periods=periods)
    assert {item.metadata["year"] for item in fitted} == {2041, 2053}
    assert len(build_cited_prompt(QUERY, fitted)) <= (1800 - 100) * 4


def test_neighbor_evidence_has_its_own_citation_identity():
    docs = [StoredDoc("anchor", "Insurance float discussion.",
                      {"year": 2041, "source_file": "letter", "next_chunk_id": "detail"}),
            StoredDoc("detail", EARLY, {"year": 2041, "source_file": "letter"})]
    anchors = [SearchHit(docs[0].id, docs[0].text, docs[0].metadata, 1.0)]
    context = expand_hits_with_neighbors(anchors, build_doc_lookup(docs), separate_neighbors=True)
    assert [item.id for item in context] == ["anchor", "detail"]
    assert EARLY not in context[0].text and context[1].text == EARLY


@pytest.mark.parametrize("provider_name", ["local", "llama"])
def test_comparison_isolates_generation_validates_and_renumbers(provider_name):
    class Capture:
        def generate(self, prompt, max_new_tokens=None):
            self.prompts.append(prompt)
            if "(focus: 2041)" in prompt:
                assert LATE not in prompt
                return EARLY + " [1]"
            assert EARLY not in prompt
            return LATE + " [1]"
    llm = Capture()
    llm.provider_name, llm.prompts = provider_name, []
    context = [hit("late", LATE, 2053), hit("early", EARLY, 2041)]
    result = generate_comparison_answer(llm, QUERY, context, max_new_tokens=200)
    assert result.answer == f"In 2041: {EARLY} [2]\n\nIn 2053: {LATE} [1]"
    assert [item["passage_ids"] for item in result.citations] == [["early"], ["late"]]
    assert len(llm.prompts) == 2 and not result.validation.blocked_claims


def test_missing_or_unsupported_period_does_not_refuse_supported_part():
    class Fabricates:
        def generate(self, prompt, **kwargs):
            return EARLY + " [1]" if "(focus: 2041)" in prompt else EARLY + " [1]"
    context = [hit("early", EARLY, 2041), hit("late", LATE, 2053)]
    result = generate_comparison_answer(Fabricates(), QUERY, context, max_new_tokens=200)
    assert "In 2041:" in result.answer and "[1]" in result.answer
    assert "In 2053: The retrieved passages for this period do not cover" in result.answer
    assert REFUSAL_LINE != result.answer
    missing = generate_comparison_answer(Fabricates(), QUERY, context[:1], max_new_tokens=200)
    assert "In 2053: The retrieved passages for this period do not cover" in missing.answer


def test_followup_uses_last_user_terms_and_year_not_assistant_facts():
    history = [{"role": "user", "content": "Earlier widget margins in 2007."},
               {"role": "user", "content": "Discuss insurance float costs in 2009."},
               {"role": "assistant", "content": "Moon cheese in 2020."}]
    resolved = build_followup_retrieval_query("What about that?", history)
    assert all(word in resolved for word in ["insurance", "float", "costs", "2009"])
    assert all(word not in resolved for word in ["widget", "2007", "moon", "2020"])
    assert "2009" not in build_followup_retrieval_query("What about 2011?", history)
    long_query = "Explain insurance premium reserve accounting treatment under statutory capital standards"
    assert build_followup_retrieval_query(long_query, history) == long_query
    docs = [StoredDoc("right", "Insurance float costs were low.", {"year": 2009}),
            StoredDoc("wrong", "Insurance float costs were low.", {"year": 2007})]
    result = Retriever(None, None, docs).search(resolved, strategy="bm25")
    assert result.used_filter == {"year": 2009}
    assert [item.id for item in result.hits] == ["right"]


def test_http_stream_and_pipeline_share_comparison_answers(monkeypatch):
    from fastapi.testclient import TestClient
    from src.generation.providers.local_provider import LocalProvider
    from src.pipeline import BuffettRAGPipeline
    from src.retrieval.retriever import RetrievalResult
    from src.services import backend_app as backend, ask_flow
    passages = [hit("early", EARLY, 2041), hit("late", LATE, 2053)]
    class Fixed:
        def search(self, *args, **kwargs):
            return RetrievalResult(QUERY, "hybrid", passages)
    llm = LocalProvider()
    monkeypatch.setattr(ask_flow, "_state", {"retriever": Fixed(), "llm": llm, "docs_by_id": {}})
    monkeypatch.setattr(backend, "API_KEYS", ())
    payload = {"query": QUERY, "expand_query": False}
    client = TestClient(backend.app)
    ordinary = client.post("/ask", json=payload).json()
    stream = client.post("/ask/stream", json=payload)
    events = [json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: ")]
    pipeline = BuffettRAGPipeline(Fixed(), {}, llm).ask(QUERY)
    assert events[-1]["answer"] == ordinary["answer"] == pipeline["answer"]
    assert events[-1]["citations"] == ordinary["citations"] == pipeline["citations"]
    assert {year for item in ordinary["citations"] for year in item["years"]} == {2041, 2053}


def test_backend_and_pipeline_followup_preserve_original_prompt(monkeypatch):
    from src.pipeline import BuffettRAGPipeline
    from src.retrieval.retriever import RetrievalResult
    from src.services import backend_app as backend, ask_flow
    history = [{"role": "user", "content": "Discuss insurance float costs in 2009."}]
    query = "What about that?"
    calls, prompts = [], []
    class Fixed:
        def search(self, *args, **kwargs):
            calls.append(args[0] if args else kwargs["query"])
            return RetrievalResult(calls[-1], "hybrid", [hit("a", EARLY, 2009)])
    class Capture:
        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            return EARLY + " [1]"
    llm = Capture()
    monkeypatch.setattr(ask_flow, "_state", {"retriever": Fixed(), "llm": llm, "docs_by_id": {}})
    backend.ask(backend.AskRequest(query=query, history=history, expand_query=False))
    BuffettRAGPipeline(Fixed(), {}, llm).ask(query, history=history)
    assert calls[0] == calls[1] == build_followup_retrieval_query(query, history)
    assert len(prompts) == 2
    assert all(f"Question: {query}\n" in prompt and "User: Discuss insurance" in prompt for prompt in prompts)
