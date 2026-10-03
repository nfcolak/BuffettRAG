"""Metadata validation and filtering shared by vector stores."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

ALLOWED_METADATA_FIELDS = {
    "year",
    "decade",
    "source_file",
    "chunk_index",
    "topics",
}


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif isinstance(value, list):
            out[key] = ",".join(str(x) for x in value)
        else:
            out[key] = str(value)
    return out


def _chroma_where(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Translate the app's filter format into Chroma's.

    Chroma requires operator expressions to hold exactly one operator, so a
    range like {"year": {"$gte": a, "$lte": b}} must become
    {"$and": [{"year": {"$gte": a}}, {"year": {"$lte": b}}]}.
    """
    if not where:
        return None

    clauses: List[Dict[str, Any]] = []
    for field, cond in where.items():
        if isinstance(cond, dict) and len(cond) > 1:
            clauses.extend({field: {op: value}} for op, value in cond.items())
        else:
            clauses.append({field: cond})

    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def validate_where_filter(where: Optional[Dict[str, Any]]) -> None:
    if not where:
        return

    for field, cond in where.items():
        if field not in ALLOWED_METADATA_FIELDS:
            raise ValueError(f"Unsupported metadata filter field: {field}")

        if isinstance(cond, dict):
            for op, value in cond.items():
                if op not in {"$eq", "$gte", "$lte", "$gt", "$lt", "$in"}:
                    raise ValueError(f"Unsupported metadata filter operator: {op}")
                if op == "$in" and not isinstance(value, (list, tuple, set)):
                    raise ValueError("$in metadata filter value must be a list, tuple, or set")


def _meta_matches(meta: Dict[str, Any], where: Dict[str, Any]) -> bool:
    validate_where_filter(where)

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
