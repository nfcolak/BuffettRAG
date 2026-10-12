"""Retrieval settings are read at call time; the harness metric helpers are exact."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import config
from src.retrieval.query_expansion import build_followup_retrieval_query
from src.retrieval.retriever import Retriever, mmr_select, reciprocal_rank_fusion, reserve_period_hits
from src.storage import SearchHit, StoredDoc

_SPEC = importlib.util.spec_from_file_location(
    "served_context_recall", Path(__file__).resolve().parents[1] / "scripts/eval/served_context_recall.py")
harness = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(harness)


def _hit(i, year=2000, text="text", score=1.0):
    return SearchHit(id=i, text=text, metadata={"year": year}, score=score)


def test_settings_read_env_at_call_time(monkeypatch):
    monkeypatch.delenv("RERANK_CANDIDATES", raising=False)
    monkeypatch.delenv("LLM_CONTEXT_PASSAGES", raising=False)
    default_passages = config.llm_context_passages()
    assert config.retrieval_setting("RERANK_CANDIDATES") == config.RETRIEVAL_DEFAULTS["RERANK_CANDIDATES"]
    monkeypatch.setenv("RERANK_CANDIDATES", "25")
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "12")
    monkeypatch.setenv("RETRIEVAL_MMR_LAMBDA", "0.9")
    assert config.retrieval_setting("RERANK_CANDIDATES") == 25
    assert config.llm_context_passages() == 12
    assert config.retrieval_setting("RETRIEVAL_MMR_LAMBDA") == 0.9
    monkeypatch.delenv("LLM_CONTEXT_PASSAGES")
    assert config.llm_context_passages() == default_passages


def test_rrf_weights_and_k(monkeypatch):
    a, b = [_hit("x"), _hit("y")], [_hit("y"), _hit("x")]
    assert reciprocal_rank_fusion([a, b], top_k=2)[0].id == "x"  # tie broken by first ranking
    assert reciprocal_rank_fusion([a, b], top_k=2, weights=[1.0, 3.0])[0].id == "y"
    monkeypatch.setenv("RRF_K", "10")
    assert abs(reciprocal_rank_fusion([a], top_k=1)[0].score - 1 / 11) < 1e-12


def test_period_reserve_per_period():
    hits = [_hit("a1", 2000), _hit("a2", 2000), _hit("a3", 2000), _hit("b1", 2001), _hit("b2", 2001)]
    periods = [{"year": 2000}, {"year": 2001}]
    assert [h.id for h in reserve_period_hits(hits, periods, 0)] == ["a1", "b1"]
    assert [h.id for h in reserve_period_hits(hits, periods, 0, per_period=2)] == ["a1", "a2", "b1", "b2"]


def test_mmr_prefers_diverse_second_pick():
    hits = [_hit("a", text="berkshire insurance float grows", score=1.0),
            _hit("b", text="berkshire insurance float grows", score=0.95),
            _hit("c", text="railroad capital spending plan", score=0.8)]
    assert [h.id for h in mmr_select(hits, 2, 0.5)] == ["a", "c"]
    assert [h.id for h in mmr_select(hits, 2, 1.0)] == ["a", "b"]


def test_followup_turns_and_always(monkeypatch):
    history = [{"role": "user", "content": "Describe insurance float accounting"},
               {"role": "assistant", "content": "ok"},
               {"role": "user", "content": "Discuss GEICO underwriting profits"}]
    assert "float" not in build_followup_retrieval_query("What about it?", history)
    monkeypatch.setenv("RETRIEVAL_FOLLOWUP_TURNS", "2")
    assert "float" in build_followup_retrieval_query("What about it?", history)
    long_query = "How does Buffett evaluate the long term economics of owning newspapers today"
    assert build_followup_retrieval_query(long_query, history) == long_query
    monkeypatch.setenv("RETRIEVAL_FOLLOWUP_ALWAYS", "1")
    assert build_followup_retrieval_query(long_query, history) != long_query


def test_neighbor_anchors_append_same_letter_neighbours(monkeypatch):
    docs = [StoredDoc(id=f"2000_p{i}", text=f"t{i}", metadata={
        "year": 2000, "source_file": "s", "previous_chunk_id": f"2000_p{i - 1}" if i else None,
        "next_chunk_id": f"2000_p{i + 1}" if i < 3 else None}) for i in range(4)]
    retriever = Retriever(vector_store=None, embedder=None, docs=docs)
    hits = [SearchHit("2000_p1", "t1", docs[1].metadata, 1.0)]
    assert retriever._append_neighbor_anchors(hits) == hits
    monkeypatch.setenv("RETRIEVAL_NEIGHBOR_ANCHORS", "1")
    assert [h.id for h in retriever._append_neighbor_anchors(hits)] == ["2000_p1", "2000_p0", "2000_p2"]


def test_multipart_reserves_slots_per_part(monkeypatch):
    docs = [StoredDoc(id=f"d{i}", text=f"w{i}", metadata={"year": 1990}) for i in range(8)]
    retriever = Retriever(vector_store=None, embedder=None, docs=docs)
    def ranking(query):
        return ["d0", "d1", "d2", "d3"] if "Candies" in query else ["d6", "d7", "d4"]

    def fake_single(self, query, **kwargs):
        from src.retrieval.retriever import RetrievalResult
        ids = ranking(query)
        return RetrievalResult(query=query, strategy="hybrid", hits=[_hit(i) for i in ids[:kwargs["top_k"]]])

    monkeypatch.setattr(Retriever, "search", fake_single)
    monkeypatch.setenv("RETRIEVAL_PART_SLOTS", "1")
    result = Retriever._multipart_search(
        retriever, "Why did Berkshire buy See's Candies, and what did it earn in 1990?", "hybrid", 3, 30, True, True,
        None, None)
    assert result is not None and {h.id for h in result.hits} == {"d0", "d6", "d1"}
    assert Retriever._multipart_search(retriever, "What did it earn in 1990?", "hybrid", 3, 30, True, True,
                                       None, None) is None


def test_harness_metric_and_bootstrap():
    row = {"claims": [["a"], ["b", "c"], ["z"]], "passage_ids": ["a", "c"], "merged_ids": ["a", "c", "z"]}
    assert harness.claim_hits(row) == [True, True, False]
    assert harness.claim_hits(row, "merged_ids") == [True, True, True]
    worse = dict(row, passage_ids=["a"])
    boot = harness.paired_bootstrap([worse] * 5, [row] * 5, draws=200)
    assert abs(boot["delta"] - 1 / 3) < 1e-9 and boot["ci95"][0] > 0
    case = {"relevant_ids": ["r"], "hard_negative_ids": ["n"], "ranked_ids": ["n", "r"]}
    assert harness.hn_pass(case) is False
    assert harness.hn_pass(dict(case, ranked_ids=["r", "n"])) is True


def test_n_ctx_read_at_call_time(monkeypatch):
    from src.generation.providers.llama_provider import LlamaCppProvider

    monkeypatch.delenv("LLM_N_CTX", raising=False)
    assert config.llm_n_ctx() == 16384
    provider = LlamaCppProvider(model_path="x.gguf")
    monkeypatch.setenv("LLM_N_CTX", "16384")
    assert config.llm_n_ctx() == 16384 and provider.n_ctx == 16384
    assert LlamaCppProvider(model_path="x.gguf", n_ctx=4096).n_ctx == 4096


def test_year_filter_min_and_scaffold_strip(monkeypatch):
    from src.retrieval.query_expansion import strip_question_scaffolding
    from src.retrieval.retriever import detect_year_filter

    monkeypatch.setenv("RETRIEVAL_YEAR_FILTER_MIN", "1970")
    assert detect_year_filter("What did Buffett say in 1961 about the Dow?") is None
    monkeypatch.setenv("RETRIEVAL_YEAR_FILTER_MIN", "1957")
    assert detect_year_filter("What did Buffett say in 1961 about the Dow?") == {"year": 1961}
    assert strip_question_scaffolding("What did Buffett say about the Dow in 1961?") == "Dow 1961"
    assert strip_question_scaffolding("Why does he use 3 years as a test?") == "use 3 years test"
