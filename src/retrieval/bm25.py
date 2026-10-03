"""BM25 retriever over chunk corpus.

Used as the lexical signal in hybrid retrieval. We rebuild from chunks at
construction time -- the corpus is small (~7K docs) so this is cheap.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from rank_bm25 import BM25Okapi

from src.storage import SearchHit, StoredDoc


_TOKEN = re.compile(r"[A-Za-z0-9']+")
_SEARCH_STOPWORDS = frozenset(
    """a about after all also an and any are as at be because been but buffett by can
    could did do does for from had has have how i if in into is it its just like more most
    of on or over said say says she so some such than that the their them then there these
    they this to was we were what when which who why will with would you your berkshire
    letter letters shareholder""".split()
)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def _normalize_search_token(token: str) -> str:
    """Normalize common English inflections without external model data."""
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _search_tokens(text: str) -> List[str]:
    normalized = (_normalize_search_token(token) for token in tokenize(text))
    return [token for token in normalized if token and token not in _SEARCH_STOPWORDS]


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
        self._tokens = [_search_tokens(d.text) for d in self.docs]
        self._bm25 = BM25Okapi(self._tokens)

    def search(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        q_tokens = _search_tokens(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)

        candidate_idxs = range(len(self.docs))
        if where:
            candidate_idxs = [
                i for i in candidate_idxs if _meta_matches(self.docs[i].metadata, where)
            ]

        # RRF turns even a zero score into a positive rank vote. Do not
        # promote arbitrary corpus-order documents without lexical evidence.
        # Match tokens directly: BM25 IDF can be zero/negative in small corpora.
        query_terms = set(q_tokens)
        candidate_idxs = [i for i in candidate_idxs
                          if query_terms.intersection(self._bm25.doc_freqs[i])]
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