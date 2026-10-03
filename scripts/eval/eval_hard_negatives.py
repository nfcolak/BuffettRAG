"""Evaluate named hard negatives using the local deterministic BM25 retrieval path."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.hard_negatives import score_hard_negative_case
from src.retrieval.bm25 import BM25Retriever
from src.storage import load_chunks_as_docs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    bm25 = BM25Retriever(load_chunks_as_docs(args.corpus))
    rows = []
    for case in cases:
        ranked_ids = [hit.id for hit in bm25.search(case["query"], top_k=20)]
        rows.append({**score_hard_negative_case(case, ranked_ids), "ranked_ids": ranked_ids})
    result = {"schema_version": 1, "mode": "offline_bm25", "n_cases": len(rows),
              "passed": sum(row["passed"] for row in rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
