"""Compare embedding models and rerankers on the gold-50 set and hard negatives.

Vector search is in-memory numpy cosine; embeddings are cached per
(model, corpus sha256) under indices/retrieval_compare/. Hybrid = BM25 + vector
fused with the production reciprocal_rank_fusion. Neither year filters, query
expansion nor dedup are applied (pure retrieval comparison).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import BGE_QUERY_INSTRUCTION, RERANK_CANDIDATES, RETRIEVAL_FETCH_K  # noqa: E402
from src.evaluation.gold_set import get_gold_queries  # noqa: E402
from src.evaluation.hard_negatives import score_hard_negative_case  # noqa: E402
from src.evaluation.retrieval_metrics import aggregate_metrics, per_query_metrics  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.retriever import reciprocal_rank_fusion  # noqa: E402
from src.storage import SearchHit, load_chunks_as_docs  # noqa: E402

RANK_DEPTH = 20
# name -> (query prefix, passage prefix)
EMBEDDERS = {
    "BAAI/bge-base-en-v1.5": (BGE_QUERY_INSTRUCTION, ""),
    "BAAI/bge-small-en-v1.5": (BGE_QUERY_INSTRUCTION, ""),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "sentence-transformers/all-MiniLM-L6-v2": ("", ""),
}
RERANKERS = ["BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-base"]


def device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _rel(p):
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def corpus_embeddings(model_name, docs, corpus_sha, cache_dir, batch_size):
    from sentence_transformers import SentenceTransformer
    slug = model_name.replace("/", "__")
    path = cache_dir / f"{slug}__{corpus_sha[:16]}.npy"
    model = SentenceTransformer(model_name, device=device())
    model.max_seq_length = min(model.max_seq_length or 512, 512)
    if path.exists():
        return model, np.load(path)
    pre = EMBEDDERS[model_name][1]
    emb = model.encode([pre + d.text for d in docs], batch_size=batch_size,
                       normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False).astype(np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return model, emb


class VectorIndex:
    def __init__(self, model_name, model, emb, docs):
        self.model, self.emb, self.docs = model, emb, docs
        self.qpre = EMBEDDERS[model_name][0]

    def search(self, query, top_k):
        q = self.model.encode([self.qpre + query], normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)[0]
        scores = self.emb @ q
        idx = np.argsort(-scores)[:top_k]
        return [SearchHit(id=self.docs[i].id, text=self.docs[i].text,
                          metadata=self.docs[i].metadata, score=float(scores[i])) for i in idx]


def rerank(ce, query, cands, depth):
    """Rerank the first RERANK_CANDIDATES; keep the rest below them, in order."""
    head, tail = cands[:RERANK_CANDIDATES], cands[RERANK_CANDIDATES:]
    if not head:
        return cands[:depth]
    raw = np.asarray(ce.predict([(query, c.text) for c in head], batch_size=16,
                                show_progress_bar=False), dtype=float)
    return [head[int(i)] for i in np.argsort(-raw)] + tail


def evaluate(name, kind, rank_fn, gold, hn_sets):
    rows, lat, ranked_by_q = {}, [], {}
    queries = [(g.qid, g.query) for g in gold]
    for g in gold:
        t = time.perf_counter()
        hits = rank_fn(g.query)[:RANK_DEPTH]
        lat.append((time.perf_counter() - t) * 1000)
        rows[g.qid] = per_query_metrics(hits, g)
        ranked_by_q[g.qid] = [h.id for h in hits]
    m = aggregate_metrics(rows).means
    out = {"config": name, "kind": kind, "n_queries": len(gold),
           "mrr": m["mrr"], "recall@1": m["recall@1"], "recall@5": m["recall@5"],
           "recall@10": m["recall@10"], "per_query": rows, "hard_negatives": {}}
    for label, cases in hn_sets.items():
        res = []
        for case in cases:
            t = time.perf_counter()
            ids = [h.id for h in rank_fn(case["query"])[:RANK_DEPTH]]
            lat.append((time.perf_counter() - t) * 1000)
            res.append({**score_hard_negative_case(case, ids), "ranked_ids": ids})
        out["hard_negatives"][label] = {"n_cases": len(res),
                                        "passed": sum(r["passed"] for r in res), "rows": res}
    out["mean_latency_ms"] = float(np.mean(lat))
    print(f"{name}: mrr={out['mrr']:.3f} r@10={out['recall@10']:.3f} "
          f"hn={ {k: v['passed'] for k, v in out['hard_negatives'].items()} } "
          f"lat={out['mean_latency_ms']:.0f}ms", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=ROOT / "data/processed/chunks_v3_paragraph.jsonl")
    ap.add_argument("--hn-cases", type=Path, action="append", default=[])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "indices/retrieval_compare")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    corpus_sha = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    docs = load_chunks_as_docs(args.corpus)
    gold = get_gold_queries()
    hn_sets = {p.stem: json.loads(p.read_text(encoding="utf-8"))["cases"] for p in args.hn_cases}
    id_map_path = args.corpus.parent / "chunk_id_map_v2_to_v3.jsonl"
    corpus_ids = {d.id for d in docs}
    id_map = {}
    if id_map_path.exists():
        for line in id_map_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            id_map[row["old_id"]] = row["new_ids"]
    for cases in hn_sets.values():  # v2-era ids -> v3 ids via the recorded map
        for case in cases:
            for key in ("relevant_ids", "hard_negative_ids"):
                if any(i not in corpus_ids for i in case[key]):
                    case[key] = sorted({n for i in case[key]
                                        for n in (id_map.get(i, []) if i not in corpus_ids else [i])})
    bm25 = BM25Retriever(docs)
    results = []

    def bm_search(q):
        return bm25.search(q, top_k=RETRIEVAL_FETCH_K)

    results.append(evaluate("bm25", "bm25", bm_search, gold, hn_sets))

    hybrid_bge = None
    bge_model = None
    for name in EMBEDDERS:
        model, emb = corpus_embeddings(name, docs, corpus_sha, args.cache_dir, args.batch_size)
        vi = VectorIndex(name, model, emb, docs)
        short = name.split("/")[-1]
        results.append(evaluate(f"vector:{short}", "vector",
                                lambda q, vi=vi: vi.search(q, RETRIEVAL_FETCH_K), gold, hn_sets))

        def hybrid(q, vi=vi):
            return reciprocal_rank_fusion([vi.search(q, RETRIEVAL_FETCH_K), bm_search(q)],
                                          top_k=RETRIEVAL_FETCH_K)
        results.append(evaluate(f"hybrid:{short}", "hybrid", hybrid, gold, hn_sets))
        if name == "BAAI/bge-base-en-v1.5":
            hybrid_bge = hybrid
            bge_model = model

    results.append(evaluate("hybrid:bge-base-en-v1.5+rerank:none", "hybrid+rerank",
                            hybrid_bge, gold, hn_sets))
    from sentence_transformers import CrossEncoder
    for rname in RERANKERS:
        try:
            ce = CrossEncoder(rname, device=device())
        except Exception as exc:  # weights missing from the offline HF cache
            print(f"{rname}: unavailable ({type(exc).__name__}: {exc})", flush=True)
            results.append({"config": f"hybrid:bge-base-en-v1.5+rerank:{rname.split('/')[-1]}",
                            "kind": "hybrid+rerank", "unavailable": str(exc)})
            continue
        results.append(evaluate(
            f"hybrid:bge-base-en-v1.5+rerank:{rname.split('/')[-1]}", "hybrid+rerank",
            lambda q, ce=ce: rerank(ce, q, hybrid_bge(q), RANK_DEPTH), gold, hn_sets))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps({
        "corpus": _rel(args.corpus), "corpus_sha256": corpus_sha, "rank_depth": RANK_DEPTH,
        "fetch_k": RETRIEVAL_FETCH_K, "rerank_candidates": RERANK_CANDIDATES,
        "hn_cases": [str(p) for p in args.hn_cases], "configs": results,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    hn_labels = list(hn_sets)
    head = ["config", "MRR", "R@1", "R@5", "R@10"] + [f"HN {l}" for l in hn_labels] + ["latency ms"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results:
        if "unavailable" in r:
            lines.append("| " + " | ".join([r["config"] + " (weights not in offline cache)"]
                                           + ["n/a"] * (len(head) - 1)) + " |")
            continue
        hn = [f"{r['hard_negatives'][l]['passed']}/{r['hard_negatives'][l]['n_cases']}" for l in hn_labels]
        lines.append("| " + " | ".join([r["config"], f"{r['mrr']:.3f}", f"{r['recall@1']:.3f}",
                                        f"{r['recall@5']:.3f}", f"{r['recall@10']:.3f}"] + hn
                                       + [f"{r['mean_latency_ms']:.0f}"]) + " |")
    (args.out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
