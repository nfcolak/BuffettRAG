"""Finance candidate source: annotation-summary index (finance-recall-75, group 2).

`SOURCES` lists (setting name, callable) pairs. `finance_policy.candidate_sources()` runs a callable
`(plan, store, retriever, query, k) -> List[SearchHit]` only when its `finance_setting(name)` is non-zero; the hits join
the candidate pool before the cross-encoder rerank.

`FINANCE_SUMMARY_SOURCE` (default 0) searches a second index built over the one-sentence annotation `summary` of every
financial passage (`scripts/index/build_summary_index.py` writes it to `<knowledge dir>/summary_index/`). A summary
restates the passage in plain words, so it finds a view the raw text puts in other words (a paraphrase the main
dense and BM25 rankings miss). The source:
  - embeds the finance query with the retriever's embedder (the main index's bge-base) and runs a hybrid search over the
    summaries (dense + BM25, weighted RRF with the retriever's hybrid weights);
  - follows the year logic of `finance_search`: a hard year ("in the 1987 letter") searches only that range, a boost
    plan adds one ranking per block with `FINANCE_YEAR_BOOST_WEIGHT`, so the source never serves a letter the plan
    excludes and still reaches in-block passages when the unfiltered top is dominated by other years;
  - returns the PASSAGES (main-corpus text and metadata), not the summaries, best first; tables only for numeric plans.
The policy keeps only ids that are not in the pool yet and sends them to the reranker with the rest.

`fuse_summary` (FINANCE_SUMMARY_FUSE_WEIGHT, default 0, finance-recall-75-r2) promotes the ids the pool already has: it
fuses the same summary ranking into the candidate order before the rerank (weighted RRF), so a pooled passage whose
summary matches the question can move into the reranker window. Its call site is `finance_search`, right after the
entity boost and before the sources run: `fused = fuse_summary(fused, plan, store, retriever, query)`.

`summary_vectors` hands the stored summary embeddings to the theme centroids of `knowledge_source.infer_themes`.

The BM25 side is rebuilt from the stored summaries at load time with the main BM25 class (the main index does the same),
so both sides tokenise identically. A missing index logs a warning and returns no candidates (like a missing knowledge
directory); an index that does not match `annotations.jsonl` or the embedder raises ValueError.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import weakref
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from config import finance_setting, retrieval_setting
from src.knowledge import default_dir
from src.query.plan import NUMERIC, QueryPlan
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.retriever import reciprocal_rank_fusion
from src.storage import SearchHit, StoredDoc
from src.storage.faiss_store import FaissStore
from src.storage.index_manifest import MANIFEST_FILE, validate_index_manifest

log = logging.getLogger(__name__)

INDEX_DIRNAME = "summary_index"     # under the knowledge directory
SUMMARY_DEPTH = 60                  # hits fetched per ranking (dense, BM25) before the fusion
SUMMARY_CANDIDATES = 24             # passages the source returns (the policy keeps the ones not in the pool yet)

_YEAR_RE = re.compile(r"^(\d{4})_")


def summary_docs(annotations: Sequence[Mapping[str, Any]],
                 years: Optional[Mapping[str, int]] = None) -> List[StoredDoc]:
    """One StoredDoc per financial annotation with a summary: id = passage id, text = summary, metadata = year.

    `years` maps passage id -> letter year (the main corpus' metadata); an id missing from it takes the year in its
    `YYYY_pNNNN` prefix.
    """
    docs: List[StoredDoc] = []
    for ann in annotations:
        summary = str(ann.get("summary") or "").strip()
        if ann.get("scope") != "financial" or not summary:
            continue
        pid = str(ann["id"])
        year = (years or {}).get(pid)
        if year is None:
            match = _YEAR_RE.match(pid)
            year = int(match.group(1)) if match else None
        docs.append(StoredDoc(id=pid, text=summary, metadata={} if year is None else {"year": int(year)}))
    return docs


class SummaryIndex:
    """The dense (FAISS) and BM25 indexes over the passage summaries."""

    def __init__(self, vectors: FaissStore, docs: Sequence[StoredDoc], manifest: Mapping[str, Any]) -> None:
        self.vectors = vectors
        self.manifest = dict(manifest)
        self.bm25 = BM25Retriever(docs)
        self._embeddings: Optional[Tuple[List[str], np.ndarray]] = None

    def __len__(self) -> int:
        return len(self.vectors)

    def embeddings(self) -> Tuple[List[str], np.ndarray]:
        """(passage ids, their unit summary embeddings), read back from the FAISS index once."""
        if self._embeddings is None:
            matrix = np.asarray(self.vectors._index.reconstruct_n(0, len(self.vectors)), dtype=np.float32)
            self._embeddings = ([d.id for d in self.vectors._docs], matrix)
        return self._embeddings

    def hybrid(self, query_vector: Sequence[float], query: str, depth: int,
               where: Optional[Dict[str, Any]] = None) -> List[SearchHit]:
        """Dense and BM25 rankings over the summaries, fused like the main index's hybrid pass."""
        dense = self.vectors.search(query_vector, top_k=depth, where=where)
        lexical = self.bm25.search(query, top_k=depth, where=where)
        weights = (retrieval_setting("HYBRID_VECTOR_WEIGHT"), retrieval_setting("HYBRID_BM25_WEIGHT"))
        return reciprocal_rank_fusion([dense, lexical], top_k=depth, weights=None if weights == (1.0, 1.0) else weights)


def load_index(directory: Path, annotations: Path, model_name: Optional[str] = None) -> Optional[SummaryIndex]:
    """Load and verify the summary index in `directory`; None when it has not been built.

    Raises ValueError when the manifest does not match the stored artifacts, the current `annotations.jsonl` or the
    embedding model (`model_name`, when given).
    """
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not annotations.is_file():
        raise ValueError(f"summary index needs {annotations}, which is missing")
    vectors = FaissStore(persist_dir=directory)
    docs = list(vectors._docs)  # the stored summaries (the FaissStore keeps them next to the vectors)
    embedding = manifest.get("embedding") or {}
    validate_index_manifest(manifest, corpus=annotations, docs=docs, backend="faiss",
                            model_name=model_name or str(embedding.get("model")),
                            dimension=int(embedding.get("dimension", -1)), persist_dir=directory)
    if len(vectors) != len(docs) or vectors._index.d != int(embedding["dimension"]):
        raise ValueError("summary index vectors do not match its documents")
    return SummaryIndex(vectors, docs, manifest)


_CACHE: Dict[Tuple[str, int, Optional[str]], Optional[SummaryIndex]] = {}
_LOCK = threading.Lock()
_WARNED: set = set()
_PASSAGES: "weakref.WeakKeyDictionary[Any, Dict[str, StoredDoc]]" = weakref.WeakKeyDictionary()


def reset_cache() -> None:
    """Forget the loaded index and the passage lookups (tests, a rebuilt index)."""
    with _LOCK:
        _CACHE.clear()
        _WARNED.clear()
        _PASSAGES.clear()


def index_dir() -> Path:
    return default_dir() / INDEX_DIRNAME


def _summary_index(model_name: Optional[str]) -> Optional[SummaryIndex]:
    directory = index_dir()
    manifest = directory / MANIFEST_FILE
    stamp = manifest.stat().st_mtime_ns if manifest.is_file() else -1
    key = (str(directory.resolve()), stamp, model_name)
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = load_index(directory, default_dir() / "annotations.jsonl", model_name)
        index = _CACHE[key]
        if index is None and key not in _WARNED:
            _WARNED.add(key)
            log.warning("FINANCE_SUMMARY_SOURCE is on but %s has no summary index; "
                        "run scripts/index/build_summary_index.py", directory)
    return index


def summary_vectors(model_name: Optional[str]) -> Optional[Tuple[List[str], np.ndarray]]:
    """(passage ids, unit summary embeddings) of the summary index; None when it has not been built. Raises
    ValueError like the source when the index does not match its inputs or `model_name`."""
    index = _summary_index(model_name)
    return None if index is None else index.embeddings()


def _passages(retriever: Any) -> Dict[str, StoredDoc]:
    """Passage id -> the main corpus' StoredDoc (text and metadata), from the retriever's BM25 documents."""
    try:
        found = _PASSAGES.get(retriever)
    except TypeError:  # an unhashable retriever
        found = None
    if found is None:
        bm25 = getattr(retriever, "bm25", None)
        found = {d.id: d for d in getattr(bm25, "docs", ())}
        try:
            _PASSAGES[retriever] = found
        except TypeError:  # a retriever that cannot be weakly referenced: look it up each time
            pass
    return found


def _rankings(index: SummaryIndex, plan: QueryPlan, store: Any, vector: Sequence[float], query: str
              ) -> Tuple[List[Sequence[SearchHit]], List[float]]:
    """The summary rankings and weights, following finance_search: an unfiltered one plus one per year block."""
    from src.retrieval.finance_policy import _boost, _year_filter  # lazy: finance_policy loads this module

    blocks, _ = _boost(plan, store)
    boost = float(max(1, finance_setting("FINANCE_YEAR_BOOST_WEIGHT")))
    rankings: List[Sequence[SearchHit]] = []
    weights: List[float] = []
    if blocks:
        for block in blocks:
            rankings.append(index.hybrid(vector, query, SUMMARY_DEPTH, _year_filter(block)))
            weights.append(1.0 if plan.year_hard else boost)
    if not (blocks and plan.year_hard):  # a hard year never leaves its range
        rankings.append(index.hybrid(vector, query, SUMMARY_DEPTH, None))
        weights.append(1.0)
    return rankings, weights


def summary_ranking(plan: QueryPlan, store: Any, retriever: Any, query: str) -> List[SearchHit]:
    """The summary hits of the finance query, best first (id = passage id, text = its summary): the rankings of
    `_rankings` fused by weighted RRF. [] without an embedder, a query or a built index."""
    embedder = getattr(retriever, "embedder", None)
    if embedder is None or not str(query).strip():
        return []
    index = _summary_index(getattr(embedder, "model_name", None))
    if index is None:
        return []
    rankings, weights = _rankings(index, plan, store, embedder.embed_query(query), query)
    total = sum(len(r) for r in rankings)
    return reciprocal_rank_fusion(rankings, top_k=max(total, 1), weights=weights) if total else []


def summary_candidates(plan: QueryPlan, store: Any, retriever: Any, query: str, k: int) -> List[SearchHit]:
    """Passages whose annotation summary matches the finance query (see the module docstring).

    `k` (the served budget) is not used: the source returns SUMMARY_CANDIDATES passages whatever the budget, so the
    reranker's work does not grow with it.
    """
    passages = _passages(retriever)
    if not passages:
        return []
    numeric = plan.kind == NUMERIC
    out: List[SearchHit] = []
    for hit in summary_ranking(plan, store, retriever, query):
        doc = passages.get(hit.id)
        if doc is None or (store.is_table(hit.id) and not numeric):
            continue
        out.append(SearchHit(id=doc.id, text=doc.text, metadata=doc.metadata, score=hit.score))
        if len(out) >= SUMMARY_CANDIDATES:
            break
    return out


def fuse_summary(fused: Sequence[SearchHit], plan: QueryPlan, store: Any, retriever: Any, query: str
                 ) -> List[SearchHit]:
    """The candidate order `fused` with the summary ranking fused in (FINANCE_SUMMARY_FUSE_WEIGHT, default 0 = off).

    The source above only adds ids the pool does not have. This promotes the pooled ones: `fused` and the summary
    ranking go through weighted RRF (1 : FINANCE_SUMMARY_FUSE_WEIGHT, each id at its rank in the full summary ranking),
    so a pooled passage whose summary matches the question moves up, into the reranker window. Only pooled ids are
    returned, with their pool hits: the summary's new ids stay with the source. Returns `fused` (a copy) when the
    weight is 0, the index is missing or no pooled id has a summary hit.
    """
    weight = float(max(0, finance_setting("FINANCE_SUMMARY_FUSE_WEIGHT")))
    if not weight or not fused:
        return list(fused)
    pooled = {h.id for h in fused}
    ranking = summary_ranking(plan, store, retriever, query)
    if not any(h.id in pooled for h in ranking):
        return list(fused)
    both = reciprocal_rank_fusion([fused, ranking], top_k=len(pooled) + len(ranking), weights=[1.0, weight])
    return [h for h in both if h.id in pooled]  # the pool's hit wins for a shared id (fused is the first ranking)


SOURCES: List[Tuple[str, Callable]] = [("FINANCE_SUMMARY_SOURCE", summary_candidates)]
