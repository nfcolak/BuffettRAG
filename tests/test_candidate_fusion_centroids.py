"""finance-recall-75-r2 group 2: the summary ranking fused into the candidate order (FINANCE_SUMMARY_FUSE_WEIGHT) and
theme inference by summary centroids (FINANCE_THEME_CENTROIDS)."""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import sentence_transformers  # noqa: F401  (macOS: imported first, torch and faiss share one libomp; else FAISS search aborts)

from config import FINANCE_DEFAULTS
from src.knowledge import KnowledgeStore
from src.query.plan import QueryPlan
from src.retrieval import knowledge_source as ks
from src.retrieval import summary_source
from src.retrieval.bm25 import _meta_matches
from src.retrieval.summary_source import INDEX_DIRNAME, fuse_summary, summary_vectors
from src.storage import SearchHit, StoredDoc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "index"))
import build_summary_index as bsi  # noqa: E402

N_THEMES = 10  # a theme stands out by at most sqrt(n - 1) standard deviations: 10 themes allow 3 > THEME_CENTROID_MIN_Z
THEMES = {f"theme{i}": f"desc{i} desc{i} desc{i}. A second sentence." for i in range(N_THEMES)}
THEMES["theme_bare"] = "descbare descbare descbare."  # no summarised passage carries it: no centroid


def _pid(i, n):
    return f"{1990 + i}_p{n:04d}"


# Two passages per theme; a summary says "summ<i>", a word no theme description uses (the paraphrase case).
ANN = [{"id": _pid(i, n), "scope": "financial", "is_table": False, "themes": [f"theme{i}"],
        "sectors": [], "events": [], "companies": [], "stance": [], "summary": f"summ{i} alpha"}
       for i in range(N_THEMES) for n in (1, 2)]
ANN.append({"id": "2005_p0001", "scope": "financial", "is_table": False, "themes": ["theme_bare"], "summary": ""})


class ConceptEmbedder:
    """Every concept word is its own axis, any other word adds to axis 0 (generic); L2-normalised. No model."""
    model_name = "fake-concepts"

    def __init__(self):
        words = [f"{kind}{i}" for kind in ("desc", "summ", "noise") for i in range(N_THEMES)] + ["descbare"]
        self.axis = {w: i + 1 for i, w in enumerate(words)}
        self.dimension = len(words) + 1

    def _vec(self, text):
        out = np.zeros(self.dimension, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            out[self.axis.get(token, 0)] += 1.0
        norm = float(np.linalg.norm(out))
        return (out / norm if norm else out).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, query):
        return self._vec(query)


FILLER = 40  # untagged passages ranked first, so the tagged ones lie below the pooled depth (fetch_k = 30)


def _docs():
    return [StoredDoc(id=f"2010_p{n:04d}", text=f"filler {n}", metadata={"year": 2010}) for n in range(FILLER)] + \
           [StoredDoc(id=a["id"], text=f"passage {a['id']}", metadata={"year": int(a["id"][:4])}) for a in ANN]


class Retriever:
    """Hybrid ranking = corpus order; what the summary source and the knowledge sources read."""

    def __init__(self, embedder=None, docs=None):
        self.embedder = embedder or ConceptEmbedder()
        self.docs = _docs() if docs is None else docs
        self.ranking = [SearchHit(id=d.id, text=d.text, metadata=dict(d.metadata), score=1.0) for d in self.docs]

    @property
    def bm25(self):
        return type("BM25", (), {"docs": self.docs})()

    def _hybrid_for_filter(self, query, fetch_k, where):
        return [h for h in self.ranking if not where or _meta_matches(h.metadata, where)][:fetch_k]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("FINANCE_SUMMARY_FUSE_WEIGHT", "FINANCE_THEME_CENTROIDS", "FINANCE_YEAR_BOOST_WEIGHT", "FINANCE_K",
                 "FINANCE_OVERFETCH", "FINANCE_TAG_DEPTH", "HYBRID_VECTOR_WEIGHT", "HYBRID_BM25_WEIGHT",
                 "KNOWLEDGE_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ks, "_CENTROID_WARNED", set())
    summary_source.reset_cache()
    yield
    summary_source.reset_cache()


@pytest.fixture
def kdir(tmp_path, monkeypatch):
    """A knowledge directory (annotations + a summary index built with ConceptEmbedder) that KNOWLEDGE_DIR points at."""
    root = tmp_path / "knowledge"
    root.mkdir()
    (root / "annotations.jsonl").write_text("\n".join(json.dumps(a) for a in ANN) + "\n", encoding="utf-8")
    bsi.build(root / "annotations.jsonl", None, root / INDEX_DIRNAME, ConceptEmbedder())
    monkeypatch.setenv("KNOWLEDGE_DIR", str(root))
    return root


@pytest.fixture
def store():
    return KnowledgeStore(annotations=ANN, taxonomy={"themes": THEMES})


def _plan(query):
    return QueryPlan(query=query, financial_query=query)


def _hit(pid):
    return SearchHit(id=pid, text=f"pool text {pid}", metadata={"year": int(pid[:4])}, score=1.0)


def test_the_new_settings_default_off():
    assert FINANCE_DEFAULTS["FINANCE_SUMMARY_FUSE_WEIGHT"] == 0
    assert FINANCE_DEFAULTS["FINANCE_THEME_CENTROIDS"] == 0


# --- FINANCE_SUMMARY_FUSE_WEIGHT -----------------------------------------------------------------------------------
def test_fusion_promotes_a_pooled_passage_whose_summary_matches(kdir, store, monkeypatch):
    pool = [_hit(_pid(i, 1)) for i in range(5)]  # theme 4's passage is pooled last
    assert fuse_summary(pool, _plan("summ4"), store, Retriever(), "summ4") == pool  # weight 0: unchanged
    monkeypatch.setenv("FINANCE_SUMMARY_FUSE_WEIGHT", "1")
    fused = fuse_summary(pool, _plan("summ4"), store, Retriever(), "summ4")
    assert fused[0].id == _pid(4, 1)
    assert sorted(h.id for h in fused) == sorted(h.id for h in pool)  # _pid(4, 2) matches too, but it is not pooled
    assert all(h.text == f"pool text {h.id}" for h in fused)          # the pool's hits, not the summaries


def test_fusion_keeps_the_order_without_a_summary_match_or_an_index(kdir, store, tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCE_SUMMARY_FUSE_WEIGHT", "2")
    outside = [_hit("2010_p0001"), _hit("2010_p0002")]  # not in the summary index
    assert fuse_summary(outside, _plan("summ4"), store, Retriever(), "summ4") == outside
    assert fuse_summary([], _plan("summ4"), store, Retriever(), "summ4") == []
    pool = [_hit(_pid(i, 1)) for i in range(5)]
    assert fuse_summary(pool, _plan("summ4"), store, Retriever(embedder=None), "  ") == pool
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("KNOWLEDGE_DIR", str(empty))
    summary_source.reset_cache()
    assert fuse_summary(pool, _plan("summ4"), store, Retriever(), "summ4") == pool


def test_the_weight_sets_how_far_a_summary_match_moves(kdir, store, monkeypatch):
    pool = [_hit(_pid(i, 1)) for i in range(N_THEMES)] + [_hit(f"2010_p{n:04d}") for n in range(60)]
    pool.append(pool.pop(4))  # theme 4's passage last: rank 70
    places = []
    for weight in ("1", "3"):
        monkeypatch.setenv("FINANCE_SUMMARY_FUSE_WEIGHT", weight)
        places.append([h.id for h in fuse_summary(pool, _plan("summ4"), store, Retriever(), "summ4")].index(_pid(4, 1)))
    assert places[1] <= places[0] < 70


# --- the summary embeddings and the theme centroids ---------------------------------------------------------------
def test_summary_vectors_are_the_stored_unit_embeddings(kdir, tmp_path, monkeypatch):
    ids, matrix = summary_vectors(ConceptEmbedder.model_name)
    assert ids == [a["id"] for a in ANN if a["summary"]]
    embedder = ConceptEmbedder()
    assert np.allclose(matrix, np.asarray(embedder.embed_documents([f"summ{int(p[:4]) - 1990} alpha" for p in ids])))
    assert summary_vectors(ConceptEmbedder.model_name)[1] is matrix  # read back once
    monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path / "nowhere"))
    assert summary_vectors(ConceptEmbedder.model_name) is None


def test_a_theme_centroid_is_the_mean_of_its_tagged_summaries(kdir, store):
    keys, _ = ks.theme_texts(store)
    rows, has = ks.theme_centroids(store, ConceptEmbedder(), keys)
    expected = np.asarray(ConceptEmbedder().embed_query("summ3 alpha"))
    assert np.allclose(rows[keys.index("theme3")], expected / np.linalg.norm(expected))
    assert has[keys.index("theme3")] and not has[keys.index("theme_bare")]
    assert not rows[keys.index("theme_bare")].any()


def test_centroids_reach_a_theme_its_description_misses(kdir, store, monkeypatch):
    embedder = ConceptEmbedder()
    assert ks.infer_themes("summ3", store, embedder) == ()  # off: the description shares no word with the question
    monkeypatch.setenv("FINANCE_THEME_CENTROIDS", "1")
    assert ks.infer_themes("summ3", store, embedder) == ("theme3",)
    assert ks.infer_themes("desc5", store, embedder) == ("theme5",)  # the description still matches on its own
    both = "desc5 desc5 summ3 summ3 alpha alpha"
    assert ks.infer_themes(both, store, embedder) == ("theme5", "theme3")  # ranked by score over threshold


def test_a_centroid_matches_only_a_close_question_that_stands_out(kdir, store, monkeypatch):
    monkeypatch.setenv("FINANCE_THEME_CENTROIDS", "1")
    embedder = ConceptEmbedder()
    assert ks.infer_themes("alpha alpha", store, embedder) == ()  # as close to every centroid: none stands out
    vague = "summ3 " + " ".join(f"noise{n}" for n in range(8))     # theme 3 stands out, but its cosine is too low
    rows, _ = ks.theme_centroids(store, embedder, ks.theme_texts(store)[0])
    question = np.asarray(embedder.embed_query(vague))
    assert float(rows[ks.theme_texts(store)[0].index("theme3")] @ question) < ks.THEME_CENTROID_MIN_SIM
    assert ks.infer_themes(vague, store, embedder) == ()


def test_without_a_usable_index_the_descriptions_still_work(kdir, store, tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("FINANCE_THEME_CENTROIDS", "1")
    other = ConceptEmbedder()
    other.model_name = "not-the-index-model"
    with caplog.at_level(logging.WARNING, logger=ks.log.name):
        assert ks.infer_themes("desc5 summ3", store, other) == ("theme5",)
        assert ks.infer_themes("desc5 summ3", store, other) == ("theme5",)
    assert len([r for r in caplog.records if "theme centroids off" in r.getMessage()]) == 1  # warned once
    monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path / "nowhere"))
    summary_source.reset_cache()
    assert ks.infer_themes("desc5 summ3", store, ConceptEmbedder()) == ("theme5",)


def test_the_tagged_source_serves_the_theme_a_centroid_inferred(kdir, store, monkeypatch):
    plan = _plan("summ3")
    assert ks.tagged_source(plan, store, Retriever(), "summ3", 4) == []
    monkeypatch.setenv("FINANCE_THEME_CENTROIDS", "1")
    got = ks.tagged_source(plan, store, Retriever(), "summ3", 4)
    assert sorted(h.id for h in got) == [_pid(3, 1), _pid(3, 2)]
