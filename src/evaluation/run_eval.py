"""Run the evaluation matrix.

Compares the four retrieval strategies x {with rerank, without rerank},
plus an LLM answer-correctness pass on the top configuration.

Outputs:
    eval_results/retrieval_<timestamp>.json  -- full per-query metrics
    eval_results/retrieval_<timestamp>.csv   -- mean metrics per config
    eval_results/faithfulness_<timestamp>.json  (only if --with-llm)

Usage:
    python -m src.evaluation.run_eval
    python -m src.evaluation.run_eval --with-llm
    python -m src.evaluation.run_eval --strategies hybrid metadata --no-rerank
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EVAL_DIR, EVAL_TOP_K_VALUES
from src.evaluation.citation_faithfulness import evaluate_faithfulness
from src.evaluation.gold_set import get_gold_queries
from src.evaluation.retrieval_metrics import (
    aggregate_metrics,
    per_query_metrics,
)
from src.pipeline import BuffettRAGPipeline, PipelineConfig


STRATEGIES = ["naive", "metadata", "vector", "hybrid"]


def evaluate_config(
    pipeline: BuffettRAGPipeline,
    strategy: str,
    rerank: bool,
    top_k: int = 10,
) -> Dict:
    gold = get_gold_queries()
    rows: Dict[str, Dict[str, float]] = {}
    for g in gold:
        # naive should bypass auto year filter to remain a true baseline
        result = pipeline.retriever.search(
            g.query,
            strategy=strategy,
            top_k=top_k,
            rerank=rerank,
            auto_year_filter=(strategy != "naive"),
        )
        rows[g.qid] = per_query_metrics(result.hits, g, ks=EVAL_TOP_K_VALUES)
    agg = aggregate_metrics(rows)
    return {
        "strategy": strategy,
        "rerank": rerank,
        "n_queries": agg.n_queries,
        "means": agg.means,
        "per_query": agg.per_query,
    }


def evaluate_answers(
    pipeline: BuffettRAGPipeline,
    strategy: str = "hybrid",
    rerank: bool = True,
    top_k: int = 8,
) -> List[Dict]:
    """Run end-to-end answer generation + faithfulness scoring on the gold set."""
    if pipeline.llm is None:
        raise RuntimeError("Pipeline was built without an LLM; cannot evaluate answers.")
    gold = get_gold_queries()
    out: List[Dict] = []
    for g in gold:
        result = pipeline.ask(
            g.query, strategy=strategy, top_k=top_k, rerank=rerank
        )
        report = evaluate_faithfulness(result["answer"] or "", _hits_from_passages(result["passages"]))
        # Crude answer-correctness signal: do gold answer keywords appear?
        ak = [k.lower() for k in g.answer_keywords]
        ans_lower = (result["answer"] or "").lower()
        keyword_hit_rate = (
            sum(1 for k in ak if k in ans_lower) / len(ak) if ak else None
        )
        out.append(
            {
                "qid": g.qid,
                "query": g.query,
                "answer": result["answer"],
                "citations": result["citations"],
                "answer_keyword_hit_rate": keyword_hit_rate,
                "faithfulness": {
                    "n_sentences": report.n_sentences,
                    "citation_coverage": report.citation_coverage,
                    "invalid_citation_rate": report.invalid_citation_rate,
                    "mean_support": report.mean_support,
                },
            }
        )
    return out


def _hits_from_passages(passages: List[Dict]):
    """Re-wrap pipeline.ask passages into SearchHit-like duck-typed objects."""
    class _Hit:
        def __init__(self, p):
            self.text = p["text"]
            self.metadata = {
                "year": p.get("year"),
                "source_file": p.get("source_file"),
                "topics": p.get("topics", ""),
            }
            self.id = p.get("id")
            self.score = p.get("score", 0.0)
    return [_Hit(p) for p in passages]


def write_outputs(
    rows: List[Dict],
    answer_rows: List[Dict] | None,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = out_dir / f"retrieval_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # CSV summary -- one row per (strategy, rerank)
    csv_path = out_dir / f"retrieval_{ts}.csv"
    if rows:
        keys = sorted(set().union(*[r["means"].keys() for r in rows]))
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["strategy", "rerank", "n_queries", *keys])
            for r in rows:
                w.writerow(
                    [r["strategy"], r["rerank"], r["n_queries"]]
                    + [round(r["means"].get(k, 0.0), 4) for k in keys]
                )

    if answer_rows is not None:
        ans_path = out_dir / f"faithfulness_{ts}.json"
        with open(ans_path, "w", encoding="utf-8") as f:
            json.dump(answer_rows, f, indent=2, ensure_ascii=False)
        print(f"Wrote {ans_path}")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BuffettRAG eval matrix")
    p.add_argument(
        "--strategies",
        nargs="+",
        default=STRATEGIES,
        choices=STRATEGIES,
    )
    p.add_argument("--no-rerank", action="store_true",
                   help="Skip the rerank=True configurations")
    p.add_argument("--with-llm", action="store_true",
                   help="Also run end-to-end answer generation + faithfulness")
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(use_llm=args.with_llm, use_reranker=True)
    pipeline = BuffettRAGPipeline.build(cfg)

    rerank_options = [False] if args.no_rerank else [False, True]

    rows: List[Dict] = []
    print("\n=== Retrieval evaluation ===")
    for strategy in args.strategies:
        for rerank in rerank_options:
            print(f"  -> strategy={strategy:<10s}  rerank={rerank}")
            row = evaluate_config(pipeline, strategy, rerank, top_k=args.top_k)
            rows.append(row)
            for k, v in row["means"].items():
                print(f"        {k:<22s}: {v:.4f}")

    answer_rows: List[Dict] | None = None
    if args.with_llm:
        print("\n=== Answer faithfulness (hybrid + rerank) ===")
        answer_rows = evaluate_answers(pipeline, strategy="hybrid", rerank=True)

    write_outputs(rows, answer_rows, EVAL_DIR)


if __name__ == "__main__":
    main()
