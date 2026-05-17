"""Retrieval evaluation metrics.

We compute three things per query:
    - Year hit rate @ k:   fraction of top-k passages whose year is in the
                           gold target_years set. Coarse but reliable.
    - Recall @ k:          1 if at least one top-k passage has the right year
                           AND contains a gold keyword; 0 otherwise.
                           This is the harder, semantically grounded metric.
    - MRR:                 mean reciprocal rank of the first passage that is
                           "relevant" (year + keyword match).

Aggregation across queries reports mean and per-query breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from src.evaluation.gold_set import GoldQuery
from src.vector_store import SearchHit


def _is_relevant(hit: SearchHit, gold: GoldQuery) -> bool:
    """A hit is relevant iff its year is in target_years AND its text contains
    any of the gold keywords (case-insensitive)."""
    year = hit.metadata.get("year")
    if gold.target_years and year not in set(gold.target_years):
        return False
    if not gold.must_contain_any:
        return True
    text_lower = hit.text.lower()
    return any(kw.lower() in text_lower for kw in gold.must_contain_any)


def _year_hit(hit: SearchHit, gold: GoldQuery) -> bool:
    return hit.metadata.get("year") in set(gold.target_years)


# -----------------------------------------------------------------------------
# Per-query metrics
# -----------------------------------------------------------------------------

def per_query_metrics(
    hits: Sequence[SearchHit],
    gold: GoldQuery,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # Year hit rate @ k
    for k in ks:
        topk = hits[:k]
        out[f"year_hit_rate@{k}"] = (
            sum(1 for h in topk if _year_hit(h, gold)) / max(len(topk), 1)
        )

    # Recall @ k -- a binary "did we get a truly relevant passage in top k"
    for k in ks:
        topk = hits[:k]
        out[f"recall@{k}"] = float(any(_is_relevant(h, gold) for h in topk))

    # MRR: rank of first relevant; 0 if none in returned hits
    rr = 0.0
    for rank, h in enumerate(hits, start=1):
        if _is_relevant(h, gold):
            rr = 1.0 / rank
            break
    out["mrr"] = rr

    return out


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

@dataclass
class AggregatedMetrics:
    n_queries: int
    means: Dict[str, float] = field(default_factory=dict)
    per_query: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "n_queries": self.n_queries,
            "means": self.means,
            "per_query": self.per_query,
        }


def aggregate_metrics(
    rows: Dict[str, Dict[str, float]],
) -> AggregatedMetrics:
    if not rows:
        return AggregatedMetrics(n_queries=0)
    keys = next(iter(rows.values())).keys()
    means = {
        k: sum(r.get(k, 0.0) for r in rows.values()) / len(rows)
        for k in keys
    }
    return AggregatedMetrics(
        n_queries=len(rows),
        means=means,
        per_query=rows,
    )


def evaluate_retrieval(
    pipeline_search_fn,
    gold_queries: Sequence[GoldQuery],
    ks: Sequence[int] = (1, 3, 5, 10),
    top_k: int = 10,
) -> AggregatedMetrics:
    """Run a retrieval-only evaluation.

    `pipeline_search_fn(query)` must return a list of SearchHit.
    """
    rows: Dict[str, Dict[str, float]] = {}
    for gold in gold_queries:
        hits = pipeline_search_fn(gold.query)
        rows[gold.qid] = per_query_metrics(hits, gold, ks=ks)
        rows[gold.qid]["__query"] = gold.query  # type: ignore[assignment]
    return aggregate_metrics({k: {kk: vv for kk, vv in v.items() if kk != "__query"}
                              for k, v in rows.items()})
