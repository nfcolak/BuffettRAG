"""Read-only lookups over `data/knowledge/` (annotations, numbers, digests, taxonomy, events, companies).

A passage without an annotation counts as financial, so an empty or partial store lets every passage pass through.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

FINANCIAL = "financial"
_FILES = ("annotations.jsonl", "numbers.jsonl", "digests.jsonl", "taxonomy.json", "events.json", "companies.json")


def default_dir() -> Path:
    """`KNOWLEDGE_DIR` (read at call time) or `data/knowledge/` under the repo root."""
    env = os.environ.get("KNOWLEDGE_DIR", "").strip()
    return Path(env) if env else Path(__file__).resolve().parents[2] / "data" / "knowledge"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path) -> Dict[str, str]:
    """File name -> sha256. Accepts `{"files": {name: sha}}`, `{name: sha}` or `sha  name` lines."""
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict):
        files = data.get("files", data)
        return {str(name): str(sha).lower() for name, sha in files.items() if isinstance(sha, str)}
    out: Dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1].lstrip("*")] = parts[0].lower()
    return out


def _verify_manifest(root: Path) -> None:
    manifest = root / "MANIFEST"
    if not manifest.is_file():
        return
    for name, expected in _read_manifest(manifest).items():
        target = root / name
        if not target.is_file():
            raise ValueError(f"knowledge MANIFEST lists {name} but it is missing in {root}")
        if _sha256(target) != expected:
            raise ValueError(f"knowledge file {name} does not match its MANIFEST sha256")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _as_set(value: Optional[Any]) -> Optional[set]:
    if value is None:
        return None
    return {value} if isinstance(value, (str, int)) else set(value)


class KnowledgeStore:
    def __init__(
        self,
        annotations: Optional[Iterable[Dict[str, Any]]] = None,
        numbers: Optional[Iterable[Dict[str, Any]]] = None,
        digests: Optional[Iterable[Dict[str, Any]]] = None,
        taxonomy: Optional[Dict[str, Any]] = None,
        events: Optional[Dict[str, Any]] = None,
        companies: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._ann: Dict[str, Dict[str, Any]] = {a["id"]: a for a in (annotations or [])}
        self._numbers: List[Dict[str, Any]] = [n for n in (numbers or []) if "metric" in n]
        self._digests: List[Dict[str, Any]] = list(digests or [])
        self.taxonomy: Dict[str, Any] = taxonomy or {}
        self._events: Dict[str, Any] = (events or {}).get("events", events or {}) if events else {}
        self._companies: Dict[str, Any] = (companies or {}).get("companies", companies or {}) if companies else {}

    @classmethod
    def empty(cls) -> "KnowledgeStore":
        """A store whose lookups are all empty: every passage passes through as financial."""
        return cls()

    @classmethod
    def load(cls, directory: Optional[os.PathLike] = None) -> "KnowledgeStore":
        """Load a knowledge directory (default `default_dir()`); a missing one gives `empty()`.

        Raises ValueError when a MANIFEST is present and a listed file is missing or its sha256 differs.
        """
        root = Path(directory) if directory is not None else default_dir()
        if not root.is_dir():
            return cls.empty()
        _verify_manifest(root)
        return cls(
            annotations=_read_jsonl(root / "annotations.jsonl"),
            numbers=_read_jsonl(root / "numbers.jsonl"),
            digests=_read_jsonl(root / "digests.jsonl"),
            taxonomy=_read_json(root / "taxonomy.json"),
            events=_read_json(root / "events.json"),
            companies=_read_json(root / "companies.json"),
        )

    def __len__(self) -> int:
        return len(self._ann)

    # passages
    def annotation(self, pid: str) -> Optional[Dict[str, Any]]:
        return self._ann.get(pid)

    def is_financial(self, pid: str) -> bool:
        ann = self._ann.get(pid)
        return ann is None or ann.get("scope") == FINANCIAL

    def has_personal(self, pid: str) -> bool:
        ann = self._ann.get(pid)
        return bool(ann and ann.get("has_personal"))

    def annotations(self) -> Iterable[Dict[str, Any]]:
        return self._ann.values()

    def is_table(self, pid: str) -> bool:
        ann = self._ann.get(pid)
        return bool(ann and ann.get("is_table"))

    def passages_with(self, sector=None, theme=None, event=None, company=None) -> List[str]:
        """Ids of annotated passages tagged with every given value (each may be one value or a collection: any of)."""
        wanted = (("sectors", _as_set(sector)), ("themes", _as_set(theme)), ("events", _as_set(event)),
                  ("companies", _as_set(company)))
        wanted = tuple((field, values) for field, values in wanted if values is not None)
        if not wanted:
            return []
        return [pid for pid, ann in self._ann.items()
                if all(values.intersection(ann.get(field) or ()) for field, values in wanted)]

    def coverage(self, pids: Iterable[str]) -> float:
        """Share of the given passages that have an annotation (0.0 for none given)."""
        pids = list(pids)
        return sum(1 for pid in pids if pid in self._ann) / len(pids) if pids else 0.0

    # numbers
    def numbers(self, metric=None, year=None, entity=None) -> List[Dict[str, Any]]:
        want_year = None if year is None else int(year)
        return [n for n in self._numbers
                if (metric is None or n.get("metric") == metric)
                and (want_year is None or n.get("year") == want_year)
                and (entity is None or str(n.get("entity", "")).lower() == str(entity).lower())]

    # digests
    def digest(self, topic: str) -> Optional[Dict[str, Any]]:
        """The final digest of a topic (not a decade part); `kind:topic` also works."""
        kind, _, name = topic.partition(":") if ":" in topic else (None, "", topic)
        for row in self._digests:
            if row.get("part") or row.get("topic") != name:
                continue
            if kind is None or row.get("kind") == kind:
                return row
        return None

    def topics(self) -> List[str]:
        seen: List[str] = []
        for row in self._digests:
            if not row.get("part") and row.get("topic") not in seen:
                seen.append(row["topic"])
        return seen

    # events and companies
    def events(self) -> Dict[str, Dict[str, Any]]:
        """Event id -> {description, event_years, letter_years, aliases}."""
        return {key: {"description": ev.get("description", ""), "event_years": list(ev.get("event_years", [])),
                      "letter_years": list(ev.get("letter_years", [])), "aliases": list(ev.get("aliases", []))}
                for key, ev in self._events.items()}

    def company_names(self) -> List[str]:
        return sorted({name for ann in self._ann.values() for name in ann.get("companies") or ()})

    def company_aliases(self) -> Dict[str, List[str]]:
        """Canonical company name -> the other ways the letters write it (`companies.json`; empty without it)."""
        return {name: [a for a in (row.get("aliases") or []) if a and a != name]
                for name, row in self._companies.items() if isinstance(row, dict)}
