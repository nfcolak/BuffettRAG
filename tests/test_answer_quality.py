"""Offline regressions. Corpus fixtures are read from tracked real letters/chunks."""
from pathlib import Path

from src.retrieval.bm25 import BM25Retriever
from src.vector_store import load_chunks_as_docs

ROOT = Path(__file__).resolve().parents[1]


def test_bm25_no_lexical_evidence_does_not_vote_in_rrf():
    docs = load_chunks_as_docs(ROOT / 'data/processed/chunks_v2.jsonl')
    assert BM25Retriever(docs).search('xyzzyqzzzzz') == []


def test_context_budget_preserves_anchor_without_repeating_real_overlap():
    from src.retrieval.context import _compose_context
    docs = load_chunks_as_docs(ROOT / 'data/processed/chunks_v2.jsonl')
    # Real 1977 letter fragment, repeated as overlapping windows.
    anchor = docs[0].text
    before = anchor[:120]
    assert _compose_context(before=[before], current=anchor, after=[],
                            max_chars=len(anchor) + 1) == anchor
    text = _compose_context(before=['prefix ' * 100], current=anchor,
                            after=['suffix ' * 100], max_chars=len(anchor) + 40)
    assert anchor in text
    assert len(text) <= len(anchor) + 40


def test_context_never_crosses_letter_boundary():
    from src.retrieval.context import build_doc_lookup, expand_hits_with_neighbors
    from src.vector_store import SearchHit, StoredDoc
    a = StoredDoc('a', 'Original evidence', {'year': 2008, 'source_file': '2008.pdf', 'next_chunk_id': 'b'})
    b = StoredDoc('b', 'Unrelated later letter', {'year': 2009, 'source_file': '2009.pdf'})
    hit = SearchHit(a.id, a.text, a.metadata, 1.0)
    expanded = expand_hits_with_neighbors([hit], build_doc_lookup([a, b]))
    assert expanded[0].text == a.text


def test_comparison_retains_both_periods_after_one_sided_reranking():
    from src.retrieval.retriever import Retriever
    from src.vector_store import SearchHit
    docs = load_chunks_as_docs(ROOT / 'data/processed/chunks_v2.jsonl')
    period_docs = {y: [d for d in docs if d.metadata['year'] == y][:5] for y in (1990, 2020)}

    class PeriodRetriever(Retriever):
        def _hybrid_for_filter(self, query, fetch_k, where):
            year = where['year']['$gte']
            return [SearchHit(d.id, d.text, d.metadata, 1.0) for d in period_docs[year]]

    class OneSidedReranker:
        def rerank(self, query, hits, top_k):
            return sorted(hits, key=lambda h: h.metadata['year'])[:top_k]

    retriever = PeriodRetriever(None, None, docs, OneSidedReranker())
    result = retriever.search('Compare earnings in the 1990s and 2020s', top_k=2, rerank=True)
    assert {h.metadata['year'] for h in result.hits} == {1990, 2020}


def test_repeated_decade_does_not_crash_or_create_two_periods():
    from src.retrieval.retriever import detect_temporal_comparison
    assert detect_temporal_comparison('Compare the 1990s with the 1990s') is None


def test_explicit_year_comparison_uses_requested_years_not_intervening_decades():
    from src.retrieval.retriever import detect_temporal_comparison
    assert detect_temporal_comparison('Compare 2008 versus 2020') == [{'year': 2008}, {'year': 2020}]


def test_expansion_does_not_change_temporal_intent_or_reranker_question():
    from src.retrieval.retriever import Retriever
    from src.vector_store import SearchHit
    docs = load_chunks_as_docs(ROOT / 'data/processed/chunks_v2.jsonl')[:10]
    calls = []

    class CaptureRetriever(Retriever):
        def _hybrid_for_filter(self, query, fetch_k, where):
            calls.append((query, where))
            return [SearchHit(docs[0].id, docs[0].text, docs[0].metadata, 1.0)]

    class CaptureReranker:
        def rerank(self, query, candidates, top_k):
            assert query == 'What are operating earnings?'
            return candidates[:top_k]

    retriever = CaptureRetriever(None, None, docs, CaptureReranker())
    result = retriever.search('What are operating earnings?', rerank=True,
                              retrieval_query='What are operating earnings? 2008 accounting')
    assert result.used_filter is None
    assert calls == [('What are operating earnings?', None),
                     ('What are operating earnings? 2008 accounting', None)]


def test_citation_at_end_does_not_cover_next_uncited_claim():
    from src.evaluation.citation_faithfulness import evaluate_faithfulness
    from src.vector_store import SearchHit
    passage = SearchHit('real', 'Operating earnings increased.', {'year': 1977}, 1.0)
    report = evaluate_faithfulness('Operating earnings increased. [1] The moon is cheese.', [passage])
    assert report.n_sentences == 2
    assert report.citation_coverage == 0.5
    assert report.per_sentence[1]['cited_passages'] == []
    assert report.per_sentence[0]['support'] == 1.0


def test_pipeline_exposes_exact_prompt_evidence_and_citation_ids():
    from src.pipeline import BuffettRAGPipeline
    from src.retrieval.retriever import RetrievalResult
    from src.retrieval.context import build_doc_lookup
    from src.vector_store import SearchHit
    docs = load_chunks_as_docs(ROOT / 'data/processed/chunks_v2.jsonl')[:3]
    hit = SearchHit(docs[1].id, docs[1].text, docs[1].metadata, 1.0)

    class FixedRetriever:
        def search(self, *args, **kwargs):
            return RetrievalResult(args[0], 'hybrid', [hit])

    class EvidenceLLM:
        def generate(self, prompt):
            self.prompt = prompt
            return 'Operating earnings. [1,99]'

    llm = EvidenceLLM()
    result = BuffettRAGPipeline(FixedRetriever(), build_doc_lookup(docs), llm).ask('Earnings?')
    assert result['passages'][0]['text'] != hit.text
    assert result['passages'][0]['text'] in llm.prompt
    assert result['retrieved_passages'][0]['text'] == hit.text
    assert result['citations'][0]['passage_ids'] == [hit.id]
    assert result['citations'][0]['invalid_numbers'] == [99]


def test_answer_evaluation_persists_evidence_and_scoring_method(monkeypatch):
    from src.evaluation import run_eval
    from src.evaluation.gold_set import get_gold_queries
    passage = {'id': '1977_0', 'text': 'Operating earnings increased.', 'year': 1977}

    class OfflinePipeline:
        llm = object()
        def ask(self, *args, **kwargs):
            return {'answer': 'Operating earnings increased. [1]', 'passages': [passage],
                    'citations': [], 'retrieved_passages': [passage]}

    monkeypatch.setattr(run_eval, 'get_gold_queries', lambda: get_gold_queries()[:1])
    monkeypatch.setattr(run_eval.time, 'sleep', lambda _: None)
    row = run_eval.evaluate_answers(OfflinePipeline())[0]
    assert row['passages'] == [passage]
    assert row['faithfulness']['support_method'] == 'lexical_bigram_proxy_not_entailment'
    assert row['faithfulness']['per_sentence'][0]['cited_passages'] == [0]


def test_backend_refusal_is_not_retried_until_it_hallucinates():
    from src.services.backend_app import _finalize_answer
    from src.generation.prompt import REFUSAL_LINE
    # Catch swallowed retry exceptions too.
    calls = []
    class Capture:
        def generate(self, *args, **kwargs):
            calls.append(True)
            return 'Unsupported alternative answer. [1]'
    answer, citations = _finalize_answer(Capture(), 'prompt', [], REFUSAL_LINE, 100)
    assert answer == REFUSAL_LINE
    assert calls == []
    assert citations == []


def test_backend_returns_prompt_passages_in_citation_order(monkeypatch):
    from src.services import backend_app as backend
    from src.vector_store import SearchHit
    raw = SearchHit('1977_0', 'anchor', {'year': 1977}, 1.0)
    expanded = SearchHit('1977_0', 'before anchor after', {'year': 1977}, 1.0)
    monkeypatch.setitem(backend._state, 'test_ready', True)
    monkeypatch.setattr(backend, '_prepare_ask', lambda req: (None, [raw], None, False, [expanded], 'prompt'))
    monkeypatch.setattr(backend, '_generate_answer', lambda *args: ('Evidence. [1]', []))
    result = backend.ask(backend.AskRequest(query='Earnings?'))
    assert result.hits[0].text == expanded.text
    assert result.retrieved_hits[0].text == raw.text


def test_dedup_retains_different_years_numbers_and_negation():
    from src.retrieval.retriever import deduplicate_hits
    from src.vector_store import SearchHit
    docs = load_chunks_as_docs(ROOT / 'data/processed/chunks_v2.jsonl')
    original = docs[0].text
    a = SearchHit('a', original, {'year': 1977}, 1.0)
    changed_number = SearchHit('b', original.replace('21,904,000', '31,904,000'), {'year': 1977}, 0.9)
    changed_negation = SearchHit('c', original.replace('are not included', 'are included'), {'year': 1977}, 0.8)
    different_year = SearchHit('d', original, {'year': 1978}, 0.7)
    assert len(deduplicate_hits([a, changed_number, changed_negation, different_year])) == 4


def test_bm25_retains_real_matches_even_when_idf_is_negative():
    from src.vector_store import StoredDoc
    docs = [StoredDoc(str(i), 'insurance float', {}) for i in range(3)]
    assert len(BM25Retriever(docs).search('insurance')) == 3


def test_decade_aliases_are_one_period():
    from src.retrieval.retriever import detect_temporal_comparison
    assert detect_temporal_comparison('Compare the 1990s with the nineties') is None


def test_http_and_streaming_use_the_same_real_evidence(monkeypatch):
    import json
    from fastapi.testclient import TestClient
    from src.services import backend_app as backend
    from src.generation.providers.local_provider import LocalProvider
    from src.retrieval.context import build_doc_lookup
    from src.retrieval.retriever import RetrievalResult
    from src.vector_store import SearchHit
    docs = load_chunks_as_docs(ROOT / 'data/processed/chunks_v2.jsonl')[:3]
    hit = SearchHit(docs[1].id, docs[1].text, docs[1].metadata, 1.0)
    class FixedRetriever:
        def search(self, **kwargs):
            return RetrievalResult(kwargs['query'], 'hybrid', [hit])
    monkeypatch.setattr(backend, '_state', {'retriever': FixedRetriever(),
                        'docs_by_id': build_doc_lookup(docs), 'llm': LocalProvider()})
    monkeypatch.setattr(backend, 'API_KEYS', ())
    # No lifespan context: startup would load models/DB. HTTP serialization,
    # real prompt/context, local generation, citation parsing and SSE are live.
    client = TestClient(backend.app)
    payload = {'query': 'How did insurance operations perform?', 'expand_query': False}
    response = client.post('/ask', json=payload)
    assert response.status_code == 200
    result = response.json()
    stream = client.post('/ask/stream', json=payload)
    assert stream.status_code == 200
    events = [json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith('data: ')]
    assert events[0]['hits'] == result['hits']
    assert events[-1]['answer'] == result['answer']
    assert events[-1]['citations'] == result['citations']
    assert result['hits'][0]['text'] != result['retrieved_hits'][0]['text']
    assert result['citations'][0]['passage_ids'] == [hit.id]


def test_context_budget_all_small_boundaries():
    from src.retrieval.context import _compose_context
    for budget in range(1, 80):
        result = _compose_context(before=['preceding words ' * 3], current='anchor sentence',
                                  after=['following words ' * 3], max_chars=budget)
        assert len(result) <= budget
        if budget >= len('anchor sentence'):
            assert 'anchor sentence' in result
