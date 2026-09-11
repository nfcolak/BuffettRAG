"""Explicit hard-negative retrieval evaluation helpers."""
from __future__ import annotations

from typing import Dict, Sequence


def score_hard_negative_case(case: Dict, ranked_ids: Sequence[str]) -> Dict:
    relevant = set(case["relevant_ids"])
    negatives = set(case["hard_negative_ids"])
    relevant_ranks = [i for i, passage_id in enumerate(ranked_ids, 1) if passage_id in relevant]
    negative_ranks = [i for i, passage_id in enumerate(ranked_ids, 1) if passage_id in negatives]
    best_relevant = min(relevant_ranks, default=None)
    best_negative = min(negative_ranks, default=None)
    return {"qid": case["qid"], "best_relevant_rank": best_relevant, "best_negative_rank": best_negative,
            "passed": best_relevant is not None and (best_negative is None or best_relevant < best_negative)}
