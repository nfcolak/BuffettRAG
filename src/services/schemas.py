"""Request and response schemas for the backend service."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from config import DEFAULT_TOP_K, RETRIEVAL_FETCH_K


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    strategy: Literal["hybrid", "vector", "metadata", "naive"] = "hybrid"
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    fetch_k: int = Field(default=RETRIEVAL_FETCH_K, ge=1, le=100)
    rerank: bool = True
    where: Optional[Dict[str, Any]] = None
    auto_year_filter: bool = True

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must contain non-whitespace characters")
        return value

    @field_validator("where")
    @classmethod
    def validate_where(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        allowed_fields = {"year", "decade", "source_file", "chunk_index", "topics"}
        allowed_ops = {"$eq", "$gte", "$lte", "$gt", "$lt", "$in"}
        if len(value) > len(allowed_fields) or set(value) - allowed_fields:
            raise ValueError("unsupported metadata filter field")
        for field, condition in value.items():
            if isinstance(condition, dict):
                if not condition or set(condition) - allowed_ops:
                    raise ValueError("unsupported metadata filter operator")
                for operator, target in condition.items():
                    if operator == "$in":
                        if not isinstance(target, list) or not target or len(target) > 50:
                            raise ValueError("metadata $in requires 1 to 50 scalar values")
                        targets = target
                    else:
                        targets = [target]
                    if any(isinstance(item, (dict, list)) or item is None for item in targets):
                        raise ValueError("metadata filters require scalar values")
            elif isinstance(condition, (dict, list)) or condition is None:
                raise ValueError("metadata filters require scalar values")
            if field in {"year", "decade", "chunk_index"}:
                raw_values = condition.values() if isinstance(condition, dict) else [condition]
                for raw in raw_values:
                    values = raw if isinstance(raw, list) else [raw]
                    if any(not isinstance(item, int) or isinstance(item, bool) for item in values):
                        raise ValueError(f"{field} filters require integers")
        return value


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class AskRequest(SearchRequest):
    max_new_tokens: int = Field(default=900, ge=1, le=2000)
    expand_query: bool = True
    history: List[HistoryTurn] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_ask(self):
        if sum(len(turn.content) for turn in self.history) > 8000:
            raise ValueError("conversation history is too large")
        return self


class HitOut(BaseModel):
    id: str
    score: float
    year: Optional[int] = None
    source_file: Optional[str] = None
    topics: str = ""
    text: str


class SearchResponse(BaseModel):
    query: str
    strategy: str
    reranked: bool
    used_filter: Optional[Dict[str, Any]] = None
    hits: List[HitOut]


class AskResponse(SearchResponse):
    retrieved_hits: List[HitOut] = Field(default_factory=list)
    answer: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
