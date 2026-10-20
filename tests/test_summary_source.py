"""finance-recall-75 group 2: the annotation-summary index (builder) and its candidate source (FINANCE_SUMMARY_SOURCE)."""
from __future__ import annotations

import json
import logging
import re
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import sentence_transformers  # noqa: F401  (macOS: imported first, torch and faiss share one libomp; else FAISS search aborts)

from config import finance_setting
from src.knowledge import KnowledgeStore, default_dir
from src.query.plan import NUMERIC, QueryPlan
from src.retrieval import finance_policy, summary_source
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.finance_policy import candidate_sources, finance_search
from src.retrieval.summary_source import INDEX_DIRNAME, summary_candidates, summary_docs
from src.storage import SearchHit, StoredDoc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "index"))
import build_summary_index as bsi  # noqa: E402

from test_finance_policy import FakeRetriever  # noqa: E402

# Passage summaries use words the raw passage text does not ("kamikaze pricing", "freefall"): the paraphrase case.
ANN = [
    {"id": "1990_p0001", "scope": "financial", "is_table": False,
     "summary": "Airline investment soured soon after purchase because of kamikaze pricing in a commodity business."},
    {"id": "1990_p0002", "scope": "financial", "is_table": False,
     "summary": "Airline carriers kept cutting fares, kamikaze pricing that hurt every commodity producer."},
    {"id": "1990_p0003", "scope": "financial", "is_table": False,
     "summary": "Airline pricing stayed kamikaze while the commodity cycle turned."},
    {"id": "2008_p0001", "scope": "financial", "is_table": False,
     "summary": "Credit markets froze in the panic and business activity went into freefall."},
    {"id": "2008_p0002", "scope": "financial", "is_table": False,
     "summary": "Airline stocks fell once more that autumn."},
    {"id": "2008_p0003", "scope": "financial", "is_table": True,
     "summary": "Table: yearend 2008 holdings with cost and market value."},
    {"id": "1995_p0001", "scope": "meeting_logistics", "is_table": False, "summary": ""},
    {"id": "1995_p0002", "scope": "personal_anecdote", "is_table": False, "summary": "A story about a paper route."},
    {"id": "1996_p0001", "scope": "financial", "is_table": False, "summary": ""},
]
RAW = {  # the raw passage texts share no content word with the summaries above
    "1990_p0001": "We bought the preferred just as trouble started, and the plane business then turned against us.",
    "1990_p0002": "Rivals slashed ticket fares again and every seat was sold at a loss.",
    "1990_p0003": "Little changed that year in how the carriers behaved.",
    "2008_p0001": "By the fourth quarter fear engulfed the country and spending stopped almost overnight.",
    "2008_p0002": "Shares of the carriers dropped again that autumn.",
    "2008_p0003": "Holdings at year end: cost 100, market 140.",
    "1995_p0001": "The annual meeting is held in Omaha.",
    "1995_p0002": "I delivered newspapers as a boy.",
    "1996_p0001": "A filler paragraph.",
}
QUERY = "kamikaze pricing commodity airline"


class BagEmbedder:
    """Deterministic bag-of-words embedder (hashed tokens, L2-normalised); no model download."""
    model_name = "fake-bag-of-words"
    dimension = 64

    def _vec(self, text: str):
        out = np.zeros(self.dimension, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            out[zlib.crc32(token.encode()) % self.dimension] += 1.0
        norm = float(np.linalg.norm(out))
        return (out / norm if norm else out).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, query):
        return self._vec(query)


def _main_docs():
    return [StoredDoc(id=pid, text=text, metadata={"year": int(pid[:4])}) for pid, text in RAW.items()]


class IndexedRetriever(FakeRetriever):
    """finance_search's fake retriever plus what the summary source reads: the embedder and the corpus documents."""

    def __init__(self, ranking=(), embedder=None, docs=None, **kw):
        super().__init__(ranking, **kw)
        self.embedder = embedder or BagEmbedder()
        self.bm25 = SimpleNamespace(docs=_main_docs() if docs is None else docs)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("FINANCE_SUMMARY_SOURCE", "FINANCE_YEAR_BOOST_WEIGHT", "FINANCE_K", "LLM_CONTEXT_PASSAGES",
                 "HYBRID_VECTOR_WEIGHT", "HYBRID_BM25_WEIGHT", "KNOWLEDGE_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RETRIEVAL_DEDUP", "0")
    summary_source.reset_cache()
    yield
    summary_source.reset_cache()


@pytest.fixture
def kdir(tmp_path, monkeypatch):
    """A knowledge directory (annotations + a built summary index) that KNOWLEDGE_DIR points at."""
    root = tmp_path / "knowledge"
    root.mkdir()
    (root / "annotations.jsonl").write_text("\n".join(json.dumps(a) for a in ANN) + "\n", encoding="utf-8")
    bsi.build(root / "annotations.jsonl", None, root / INDEX_DIRNAME, BagEmbedder())
    monkeypatch.setenv("KNOWLEDGE_DIR", str(root))
    return root


@pytest.fixture(scope="module")
def store():
    return KnowledgeStore(annotations=ANN)


def _plan(query=QUERY, **kw):
    return QueryPlan(query=query, financial_query=query, **kw)


def _ids(hits):
    return [h.id for h in hits]


# --- the builder -------------------------------------------------------------------------------------------------
def test_summary_docs_keep_financial_passages_with_a_summary_and_their_year():
    docs = summary_docs(ANN)
    assert _ids(docs) == ["1990_p0001", "1990_p0002", "1990_p0003", "2008_p0001", "2008_p0002", "2008_p0003"]
    assert docs[0].text == ANN[0]["summary"] and docs[0].metadata == {"year": 1990}
    assert summary_docs(ANN, {"1990_p0001": 1991})[0].metadata == {"year": 1991}  # the main corpus' year wins
    assert summary_docs([{"id": "x", "scope": "financial", "summary": "s"}])[0].metadata == {}


def test_build_writes_the_index_and_a_manifest(kdir):
    out = kdir / INDEX_DIRNAME
    manifest = json.loads((out / "index_manifest.json").read_text(encoding="utf-8"))
    assert sorted(p.name for p in out.iterdir()) == ["index.faiss", "index_manifest.json", "meta.json"]
    assert manifest["document_count"] == 6 and manifest["kind"] == "annotation_summary_index"
    assert manifest["embedding"] == {"model": "fake-bag-of-words", "dimension": 64}
    assert set(manifest["artifacts_sha256"]) == {"index.faiss", "meta.json"}
    assert manifest["bm25"]["documents"] == 6 and manifest["bm25"]["vocabulary"] > 10
    index = summary_source.load_index(out, kdir / "annotations.jsonl", "fake-bag-of-words")
    assert len(index) == 6 and index.bm25.search("freefall", 3)[0].id == "2008_p0001"


def test_build_takes_the_year_from_the_chunk_corpus_and_checks_the_ids(tmp_path):
    ann = tmp_path / "annotations.jsonl"
    ann.write_text(json.dumps(ANN[0]) + "\n" + json.dumps(ANN[3]) + "\n", encoding="utf-8")
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(json.dumps({"id": "1990_p0001", "year": 1991, "text": "t"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not in the chunk corpus"):
        bsi.build(ann, chunks, tmp_path / "idx", BagEmbedder())
    chunks.write_text(chunks.read_text(encoding="utf-8") + json.dumps({"id": "2008_p0001", "year": 2008}) + "\n",
                      encoding="utf-8")
    bsi.build(ann, chunks, tmp_path / "idx", BagEmbedder())
    meta = json.loads((tmp_path / "idx" / "meta.json").read_text(encoding="utf-8"))
    assert {d["id"]: d["metadata"] for d in meta} == {"1990_p0001": {"year": 1991}, "2008_p0001": {"year": 2008}}


def test_build_refuses_a_non_empty_directory_without_overwrite(kdir):
    out = kdir / INDEX_DIRNAME
    with pytest.raises(SystemExit, match="--overwrite"):
        bsi.build(kdir / "annotations.jsonl", None, out, BagEmbedder())
    manifest = bsi.build(kdir / "annotations.jsonl", None, out, BagEmbedder(), overwrite=True)
    assert manifest["document_count"] == 6 and len(list(out.iterdir())) == 3


def test_build_without_a_financial_summary_fails(tmp_path):
    ann = tmp_path / "annotations.jsonl"
    ann.write_text(json.dumps(ANN[6]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no financial passage"):
        bsi.build(ann, None, tmp_path / "idx", BagEmbedder())


# --- loading and verification ------------------------------------------------------------------------------------
def test_load_rejects_an_index_that_does_not_match_its_inputs(kdir):
    out, ann = kdir / INDEX_DIRNAME, kdir / "annotations.jsonl"
    with pytest.raises(ValueError, match="embedding configuration"):
        summary_source.load_index(out, ann, "some-other-model")
    meta = out / "meta.json"
    original = meta.read_text(encoding="utf-8")
    meta.write_text(original.replace("freefall", "free fall"), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch: meta.json"):
        summary_source.load_index(out, ann)
    meta.write_text(original, encoding="utf-8")
    (out / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact set"):
        summary_source.load_index(out, ann)
    (out / "stray.txt").unlink()
    ann.write_text(ann.read_text(encoding="utf-8") + json.dumps(ANN[6]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corpus hash"):  # annotations changed after the index was built
        summary_source.load_index(out, ann)
    ann.unlink()
    with pytest.raises(ValueError, match="annotations"):
        summary_source.load_index(out, ann)


def test_a_missing_index_gives_no_candidates_and_creates_nothing(tmp_path, monkeypatch, caplog, store):
    root = tmp_path / "knowledge"
    root.mkdir()
    monkeypatch.setenv("KNOWLEDGE_DIR", str(root))
    assert summary_source.load_index(root / INDEX_DIRNAME, root / "annotations.jsonl") is None
    with caplog.at_level(logging.WARNING, logger=summary_source.log.name):
        assert summary_candidates(_plan(), store, IndexedRetriever(), QUERY, 8) == []
        assert summary_candidates(_plan(), store, IndexedRetriever(), QUERY, 8) == []
    assert not (root / INDEX_DIRNAME).exists()
    assert len([r for r in caplog.records if "no summary index" in r.getMessage()]) == 1  # warned once


def test_a_retriever_that_cannot_embed_or_has_no_corpus_gives_no_candidates(kdir, store):
    assert summary_candidates(_plan(), store, FakeRetriever([]), QUERY, 8) == []
    assert summary_candidates(_plan(), store, IndexedRetriever(docs=[]), QUERY, 8) == []
    assert summary_candidates(_plan(), store, IndexedRetriever(), "  ", 8) == []


def test_the_source_checks_the_embedding_model_of_the_retriever(kdir, store):
    other = BagEmbedder()
    other.model_name = "not-the-index-model"
    with pytest.raises(ValueError, match="embedding configuration"):
        summary_candidates(_plan(), store, IndexedRetriever(embedder=other), QUERY, 8)


# --- the source --------------------------------------------------------------------------------------------------
def test_the_source_returns_the_passages_not_the_summaries(kdir, store):
    hits = summary_candidates(_plan(), store, IndexedRetriever(), QUERY, 8)
    assert {"1990_p0001", "1990_p0002", "1990_p0003"} <= set(_ids(hits)) and _ids(hits)[0].startswith("1990")
    for hit in hits:
        assert hit.text == RAW[hit.id] and hit.metadata == {"year": int(hit.id[:4])}
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert "1995_p0002" not in _ids(hits) and "1996_p0001" not in _ids(hits)  # not financial / no summary: not indexed


def test_the_source_finds_a_paraphrase_the_raw_text_misses(kdir, store):
    assert BM25Retriever(_main_docs()).search(QUERY, 5) == []            # the raw passages share no query word
    assert "1990_p0001" in _ids(summary_candidates(_plan(), store, IndexedRetriever(), QUERY, 8))
    freefall = summary_candidates(_plan("business activity freefall"), store, IndexedRetriever(), "freefall", 8)
    assert _ids(freefall)[0] == "2008_p0001"


def test_the_candidate_count_is_capped(kdir, store, monkeypatch):
    monkeypatch.setattr(summary_source, "SUMMARY_CANDIDATES", 2)
    assert len(summary_candidates(_plan(), store, IndexedRetriever(), QUERY, 8)) == 2


def test_tables_are_candidates_only_for_numeric_plans(kdir, store):
    query = "yearend holdings cost market value"
    assert "2008_p0003" not in _ids(summary_candidates(_plan(query), store, IndexedRetriever(), query, 8))
    numeric = summary_candidates(_plan(query, kind=NUMERIC), store, IndexedRetriever(), query, 8)
    assert _ids(numeric)[0] == "2008_p0003"


def test_a_hard_year_never_leaves_its_range(kdir, store):
    free = summary_candidates(_plan(), store, IndexedRetriever(), QUERY, 8)
    assert "2008_p0002" in _ids(free)
    hard = _plan(year_range=(1990, 1990), year_hard=True, year_spans=((1990, 1990),), year_source="explicit")
    got = summary_candidates(hard, store, IndexedRetriever(), QUERY, 8)
    assert got and {h.metadata["year"] for h in got} == {1990}


def test_a_year_block_reaches_in_range_passages_the_unfiltered_top_misses(kdir, store, monkeypatch):
    monkeypatch.setattr(summary_source, "SUMMARY_CANDIDATES", 2)
    assert "2008_p0002" not in _ids(summary_candidates(_plan(), store, IndexedRetriever(), QUERY, 8))
    boosted = _plan(year_range=(2008, 2008), year_spans=((2008, 2008),), year_source="explicit")
    got = summary_candidates(boosted, store, IndexedRetriever(), QUERY, 8)
    assert "2008_p0002" in _ids(got) and len(got) == 2
    monkeypatch.setenv("FINANCE_YEAR_BOOST_WEIGHT", "1")  # the block weight follows the policy setting
    assert summary_candidates(boosted, store, IndexedRetriever(), QUERY, 8)


# --- registry, finance_search and deploy -------------------------------------------------------------------------
def test_the_source_is_registered_and_gated_by_its_setting(monkeypatch):
    assert ("FINANCE_SUMMARY_SOURCE", summary_candidates) in finance_policy.registered_sources()
    assert summary_source.SOURCES == [("FINANCE_SUMMARY_SOURCE", summary_candidates)]
    assert finance_setting("FINANCE_SUMMARY_SOURCE") == 0 and candidate_sources() == []
    monkeypatch.setenv("FINANCE_SUMMARY_SOURCE", "1")
    assert candidate_sources() == [("FINANCE_SUMMARY_SOURCE", summary_candidates)]


def test_finance_search_adds_the_summary_passages_to_the_pool(kdir, store, monkeypatch):
    main = [SearchHit(id="1990_p0002", text=RAW["1990_p0002"], metadata={"year": 1990}, score=1.0)]
    plan = _plan()
    _, off = finance_search(plan, IndexedRetriever(main), store, k=4)
    assert off["source_candidates"] == {} and "1990_p0001" not in off["stage_pool"]
    monkeypatch.setenv("FINANCE_SUMMARY_SOURCE", "1")
    hits, on = finance_search(plan, IndexedRetriever(main), store, k=4)
    assert on["source_candidates"]["FINANCE_SUMMARY_SOURCE"] >= 1       # the ids the pool did not have yet
    assert "1990_p0001" in on["stage_pool"] and "1990_p0002" == on["stage_pool"][0]
    assert "1990_p0001" in _ids(hits) and all(h.text == RAW[h.id] for h in hits)


def test_the_modal_image_ships_the_summary_index_with_the_knowledge_directory():
    app = (ROOT / "deploy" / "modal" / "app.py").read_text(encoding="utf-8")
    call = re.search(r'image\.add_local_dir\(str\(REPO / "data/knowledge"\), f"\{APP_DIR\}/data/knowledge"([^)]*)\)', app)
    assert call and "ignore" not in call.group(1)                  # the whole directory, nothing filtered out
    assert default_dir() == ROOT / "data" / "knowledge"           # where the builder writes: <knowledge dir>/summary_index
    assert summary_source.index_dir() == ROOT / "data" / "knowledge" / INDEX_DIRNAME
