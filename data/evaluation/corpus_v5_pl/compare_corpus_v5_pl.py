"""Corpus v5_pl (v3 + Buffett Partnership letters) vs v3: retrieval regression on the old labels.

  retrieve --corpus v3|v5_pl   (child process; FAISS_INDEX_NAME picks the index) -> raw_<corpus>.json
  report                       -> results.json + comparison.md (ends with REGRESSION: OK|FAIL)

Serving retrieval = Retriever.search as ask_flow._do_search calls it for the hybrid benchmark
(strategy hybrid, top_k 8, fetch_k 30, rerank on, auto year filter, follow-up query resolution,
no query expansion). Label sets: gold_set.py 50 queries (year + keyword relevance, text based),
grounded_dev_v1/confirm/dev_cases.json (answerable cases, gold passage ids), hard_negatives_v4
(relevant / hard-negative ids). All old labels are v3 ids, which v5_pl keeps unchanged.
The v3 baseline runs with the year bounds that served v3 (1977-2024); v5_pl uses the new
1957-2024 bounds, so the comparison includes the effect of year filters on 1957-1976 years.
Rule: pooled recall@8 drop <= 0.01 and hard-negative passes not lower -> REGRESSION: OK.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("PYTHON_DOTENV_DISABLED", "1")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
MAIN = Path(os.getenv("BUFFETTRAG_MAIN", "/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG"))
OUT_DIR = ROOT / "data/evaluation/corpus_v5_pl"
TOP_K, FETCH_K = 8, 30
RERANK = os.getenv("CMP_RERANK", "1") != "0"  # CMP_RERANK=0: hybrid RRF order without the cross-encoder
CORPUS_FILES = {"v3": "data/processed/chunks_v3_paragraph.jsonl", "v5_pl": "data/processed/chunks_v5_pl.jsonl"}
INDEX_NAMES = {"v3": "faiss_v3", "v5_pl": "faiss_v5_pl"}
SETS = ("gold50", "dev", "hard_neg")


def _p(rel: str) -> Path:
    return ROOT / rel if (ROOT / rel).exists() else MAIN / rel


def load_cases():
    dev = json.loads(_p("data/evaluation/grounded_dev_v1/confirm/dev_cases.json").read_text(encoding="utf-8"))["cases"]
    dev = [c for c in dev if c.get("question_type") != "unanswerable" and c.get("gold_passage_ids")]
    hn = json.loads(_p("data/evaluation/hard_negatives_v4/hard_negatives_v4.json").read_text(encoding="utf-8"))["cases"]
    return dev, hn


def retrieve(corpus: str, raw_out: Path) -> None:
    import config
    from src.evaluation.gold_set import get_gold_queries
    from src.pipeline import BuffettRAGPipeline, PipelineConfig
    from src.retrieval import retriever as retriever_mod
    from src.retrieval.query_expansion import build_followup_retrieval_query

    if config.FAISS_DIR.name != INDEX_NAMES[corpus]:
        raise SystemExit(f"FAISS_INDEX_NAME must be {INDEX_NAMES[corpus]} (got {config.FAISS_DIR.name})")
    if corpus == "v3":
        retriever_mod.CORPUS_YEAR_MIN = 1977  # the bounds that served v3
    chunks = _p(CORPUS_FILES[corpus])
    pipeline = BuffettRAGPipeline.build(PipelineConfig(chunks_file=chunks, use_llm=False))
    retriever = pipeline.retriever
    dev, hn = load_cases()
    queries = ([("gold50", g.qid, g.query, []) for g in get_gold_queries()]
               + [("dev", c["qid"], c["query"], c.get("history", [])) for c in dev]
               + [("hard_neg", c["qid"], c["query"], []) for c in hn])
    rows = []
    for i, (label_set, qid, query, history) in enumerate(queries, 1):
        res = retriever.search(query=build_followup_retrieval_query(query, history), strategy="hybrid",
                               top_k=TOP_K, fetch_k=FETCH_K, rerank=RERANK, where=None, auto_year_filter=True,
                               retrieval_query=None)
        rows.append({"set": label_set, "qid": qid, "reranked": res.reranked, "used_filter": res.used_filter,
                     "hits": [{"id": h.id, "year": h.metadata.get("year"), "text": h.text} for h in res.hits]})
        print(f"[{corpus}] {i}/{len(queries)} {qid}", flush=True)
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw_out.write_text(json.dumps({"corpus": corpus, "faiss_dir": str(config.FAISS_DIR.resolve()),
                                   "chunks": str(chunks), "top_k": TOP_K, "fetch_k": FETCH_K, "rerank": RERANK, "rows": rows},
                                  ensure_ascii=False) + "\n", encoding="utf-8")


def report(raw_dir: Path, out_dir: Path, tag: str = "") -> None:
    import numpy as np

    from src.evaluation.gold_set import get_gold_queries
    from src.evaluation.hard_negatives import score_hard_negative_case
    from src.evaluation.retrieval_metrics import _is_relevant
    from src.storage import SearchHit

    raw = {c: json.loads((raw_dir / f"raw_{c}{tag}.json").read_text(encoding="utf-8")) for c in ("v3", "v5_pl")}
    rerank = all(r.get("rerank", True) for r in raw.values())
    gold = {g.qid: g for g in get_gold_queries()}
    dev, hn = load_cases()
    dev_by, hn_by = {c["qid"]: c for c in dev}, {c["qid"]: c for c in hn}

    def per_corpus(corpus: str):
        per_q, hn_rows = {}, {}
        for row in raw[corpus]["rows"]:
            hits = row["hits"]
            if row["set"] == "gold50":
                flags = [_is_relevant(SearchHit(h["id"], h["text"], {"year": h["year"]}, 0.0), gold[row["qid"]])
                         for h in hits]
            elif row["set"] == "dev":
                rel = set(dev_by[row["qid"]]["gold_passage_ids"])
                flags = [h["id"] in rel for h in hits]
            else:
                rel = set(hn_by[row["qid"]]["relevant_ids"])
                flags = [h["id"] in rel for h in hits]
                hn_rows[row["qid"]] = score_hard_negative_case(hn_by[row["qid"]], [h["id"] for h in hits])
            rank = next((i for i, f in enumerate(flags, 1) if f), None)
            ids = [h["id"] for h in hits]
            per_q[(row["set"], row["qid"])] = {
                "r8": float(any(flags[:8])), "rr": 1.0 / rank if rank else 0.0, "ids": ids,
                "pl_slots": sum(i.startswith("bpl") for i in ids), "filter": row["used_filter"]}
        return per_q, hn_rows

    data = {c: per_corpus(c) for c in raw}

    def agg(per_q, sets):
        sel = [v for (s, _), v in per_q.items() if s in sets]
        return {"n": len(sel), "recall@8": round(float(np.mean([v["r8"] for v in sel])), 4),
                "mrr": round(float(np.mean([v["rr"] for v in sel])), 4),
                "pl_slots": int(sum(v["pl_slots"] for v in sel)), "top8_slots": int(sum(len(v["ids"]) for v in sel)),
                "queries_with_pl": int(sum(v["pl_slots"] > 0 for v in sel))}

    metrics = {c: {name: agg(data[c][0], sets) for name, sets in
                   (("gold50", {"gold50"}), ("dev", {"dev"}), ("hard_neg", {"hard_neg"}), ("pooled", set(SETS)))}
               for c in data}
    hn_passed = {c: {"passed": sum(r["passed"] for r in data[c][1].values()), "n": len(data[c][1])} for c in data}
    a, b = data["v3"][0], data["v5_pl"][0]
    changed = []
    for k in sorted(a):
        if a[k]["r8"] != b[k]["r8"] or a[k]["rr"] != b[k]["rr"]:
            changed.append({"query": f"{k[0]}:{k[1]}", "v3_r8": a[k]["r8"], "v5_pl_r8": b[k]["r8"],
                            "v3_rr": round(a[k]["rr"], 4), "v5_pl_rr": round(b[k]["rr"], 4),
                            "v5_pl_pl_slots": b[k]["pl_slots"], "v5_pl_filter": b[k]["filter"]})
    hn_changed = [q for q in data["v3"][1] if data["v3"][1][q]["passed"] != data["v5_pl"][1][q]["passed"]]
    drop = metrics["v3"]["pooled"]["recall@8"] - metrics["v5_pl"]["pooled"]["recall@8"]
    ok = drop <= 0.01 + 1e-12 and hn_passed["v5_pl"]["passed"] >= hn_passed["v3"]["passed"]
    results = {"schema_version": 1, "cross_encoder_rerank": rerank, "top_k": TOP_K, "fetch_k": FETCH_K, "metrics": metrics,
               "hard_neg_passed": hn_passed, "hard_neg_changed": hn_changed,
               "pooled_recall@8_drop": round(drop, 4), "queries_with_changed_metrics": changed,
               "per_query_pl_slots": {f"{s}:{q}": v["pl_slots"] for (s, q), v in b.items() if v["pl_slots"]},
               "regression": "OK" if ok else "FAIL"}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"results{tag}.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / f"comparison{tag}.md").write_text(render(results), encoding="utf-8")
    print(results["regression"], json.dumps(metrics["v3"]["pooled"]), json.dumps(metrics["v5_pl"]["pooled"]))


def render(res: Dict[str, Any]) -> str:
    L: List[str] = [
        "# Corpus v5_pl (v3 + Buffett Partnership letters 1957-1970) vs v3: retrieval regression", "",
        "Serving hybrid retrieval as in `run_live_benchmark.py --retrieval hybrid` (FAISS bge-base-en-v1.5 + BM25 + RRF, "
        f"{'bge-reranker-v2-m3' if res['cross_encoder_rerank'] else 'NO cross-encoder rerank (hybrid RRF order only)'}, "
        f"auto year filter, top {res['top_k']}, fetch_k {res['fetch_k']}). Labels are the old v3 ids: "
        "gold-50 (year + keyword, text based), dev (gold passages of the answerable grounded_dev_v1 cases), "
        "hard_negatives_v4 (relevant ids). v3 baseline uses the 1977-2024 year bounds that served v3; v5_pl uses 1957-2024. "
        "Pooled = all three sets; MRR is over the returned top 8.", "",
        "| set | n | corpus | recall@8 | MRR | top-8 slots on partnership passages | queries with >=1 partnership hit |",
        "|---|---|---|---|---|---|---|"]
    for name in ("gold50", "dev", "hard_neg", "pooled"):
        for corpus in ("v3", "v5_pl"):
            m = res["metrics"][corpus][name]
            L.append(f"| {name} | {m['n']} | {corpus} | {m['recall@8']:.4f} | {m['mrr']:.4f} | "
                     f"{m['pl_slots']} / {m['top8_slots']} | {m['queries_with_pl']} |")
    hp = res["hard_neg_passed"]
    L += ["", f"hard_negatives_v4 passes (best relevant rank < best hard-negative rank): v3 {hp['v3']['passed']}/{hp['v3']['n']}, "
              f"v5_pl {hp['v5_pl']['passed']}/{hp['v5_pl']['n']}; cases that changed: {res['hard_neg_changed'] or 'none'}.", "",
          f"Pooled recall@8 drop (v3 - v5_pl): {res['pooled_recall@8_drop']:+.4f}.", "",
          "Queries whose recall@8 or reciprocal rank changed:", ""]
    if res["queries_with_changed_metrics"]:
        L += ["| query | recall@8 v3 -> v5_pl | RR v3 -> v5_pl | partnership slots (v5_pl) | year filter (v5_pl) |", "|---|---|---|---|---|"]
        for c in res["queries_with_changed_metrics"]:
            L.append(f"| {c['query']} | {c['v3_r8']:.0f} -> {c['v5_pl_r8']:.0f} | {c['v3_rr']:.3f} -> {c['v5_pl_rr']:.3f} | "
                     f"{c['v5_pl_pl_slots']} | {c['v5_pl_filter']} |")
    else:
        L.append("none")
    L += ["", f"REGRESSION: {res['regression']}", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieve")
    r.add_argument("--corpus", choices=tuple(CORPUS_FILES), required=True)
    r.add_argument("--raw-dir", type=Path, required=True)
    rep = sub.add_parser("report")
    rep.add_argument("--raw-dir", type=Path, required=True)
    rep.add_argument("--tag", default="", help="raw/result file suffix, e.g. _norerank")
    args = ap.parse_args()
    if args.cmd == "retrieve":
        if os.getenv("FAISS_INDEX_NAME") != INDEX_NAMES[args.corpus]:
            raise SystemExit(f"run with FAISS_INDEX_NAME={INDEX_NAMES[args.corpus]}")
        retrieve(args.corpus, args.raw_dir / f"raw_{args.corpus}{'' if RERANK else '_norerank'}.json")
    else:
        report(args.raw_dir, OUT_DIR, args.tag)


if __name__ == "__main__":
    main()
