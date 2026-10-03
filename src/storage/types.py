"""Document and search-result types shared by vector stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class StoredDoc:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float
