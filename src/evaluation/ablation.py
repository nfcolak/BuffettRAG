"""Model-ablation orchestration; unavailable local models are reported, never guessed."""
from __future__ import annotations

from time import perf_counter
from typing import Callable, Dict, Sequence

_CONFIGS = ("bm25", "embedding_only", "embedding_reranker")


def benchmark_configurations(cases: Sequence[Dict], runners: Dict[str, Callable]) -> Dict:
    configurations = {}
    for name in _CONFIGS:
        runner = runners.get(name)
        if runner is None:
            configurations[name] = {"status": "unavailable", "reason": "runner/model not available locally"}
            continue
        started = perf_counter()
        rows = []
        for case in cases:
            ranked_ids, latency_ms = runner(case)
            relevant = set(case.get("relevant_ids", case.get("gold_passage_ids", [])))
            rows.append({"qid": case.get("qid"), "recall_at_8": float(bool(relevant & set(ranked_ids[:8]))),
                         "citation_support": float(bool(relevant & set(ranked_ids))), "latency_ms": float(latency_ms)})
        summary = {
            "n_cases": len(rows),
            "recall_at_8": sum(row["recall_at_8"] for row in rows) / len(rows) if rows else 0.0,
            "citation_support": sum(row["citation_support"] for row in rows) / len(rows) if rows else 0.0,
            "mean_latency_ms": sum(row["latency_ms"] for row in rows) / len(rows) if rows else 0.0,
        }
        configurations[name] = {
            "status": "ok",
            "summary": summary,
            "rows": rows,
            "wall_time_ms": round((perf_counter() - started) * 1000, 3),
        }
    return {"schema_version": 1, "configurations": configurations}
