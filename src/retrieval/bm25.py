"""BM25 retriever over chunk corpus.

Used as the lexical signal in hybrid retrieval. We rebuild from chunks at
construction time -- the corpus is small (~7K docs) so this is cheap.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from rank_bm25 import BM25Okapi

from src.vector_store import SearchHit, StoredDoc


_TOKEN = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def _meta_matches(meta: Dict[str, Any], where: Dict[str, Any]) -> bool:
    for key, cond in where.items():
        value = meta.get(key)

        if isinstance(cond, dict):
            for op, target in cond.items():
                if op == "$eq" and value != target:
                    return False
                if op == "$gte" and not (value is not None and value >= target):
                    return False
                if op == "$lte" and not (value is not None and value <= target):
                    return False
                if op == "$gt" and not (value is not None and value > target):
                    return False
                if op == "$lt" and not (value is not None and value < target):
                    return False
                if op == "$in" and value not in target:
                    return False
        else:
            if value != cond:
                return False

    return True


class BM25Retriever:
    def __init__(self, docs: Sequence[StoredDoc]) -> None:
        self.docs = list(docs)
        self._tokens = [tokenize(d.text) for d in self.docs]
        self._bm25 = BM25Okapi(self._tokens)

    def search(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)

        candidate_idxs = range(len(self.docs))
        if where:
            candidate_idxs = [
                i for i in candidate_idxs if _meta_matches(self.docs[i].metadata, where)
            ]

        ranked = sorted(candidate_idxs, key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            SearchHit(
                id=self.docs[i].id,
                text=self.docs[i].text,
                metadata=self.docs[i].metadata,
                score=float(scores[i]),
            )
            for i in ranked
        ]