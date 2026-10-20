"""Finance candidate sources: knowledge (finance-recall-75).

`SOURCES` lists (setting name, callable) pairs. `finance_policy.candidate_sources()` runs a callable
`(plan, store, retriever, query, k) -> List[SearchHit]` only when its `finance_setting(name)` is non-zero; the hits join
the candidate pool before the cross-encoder rerank. Nothing is registered yet.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

SOURCES: List[Tuple[str, Callable]] = []
