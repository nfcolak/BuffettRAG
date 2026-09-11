"""Context expansion helpers for answer generation."""

from __future__ import annotations

from typing import Dict, Iterable, List

from src.vector_store import SearchHit, StoredDoc


def build_doc_lookup(docs: Iterable[StoredDoc]) -> Dict[str, StoredDoc]:
    return {doc.id: doc for doc in docs}


def expand_hits_with_neighbors(
    hits: List[SearchHit],
    docs_by_id: Dict[str, StoredDoc],
    *,
    neighbors: int = 1,
    max_chars: int = 2600,
) -> List[SearchHit]:
    """Return hits with adjacent chunks included in text for LLM context.

    Retrieval should stay precise, so we search over compact chunks. Generation
    benefits from a little surrounding context, so this expands each selected
    hit using `previous_chunk_id` / `next_chunk_id` metadata from the chunk file.
    """
    if neighbors <= 0:
        return hits

    expanded: List[SearchHit] = []
    for hit in hits:
        doc = docs_by_id.get(hit.id)
        if doc is None:
            expanded.append(hit)
            continue

        before_docs = _walk_neighbors(doc, docs_by_id, "previous_chunk_id", neighbors)
        after_docs = _walk_neighbors(doc, docs_by_id, "next_chunk_id", neighbors)
        before_docs.reverse()

        context_text = _compose_context(
            before=[d.text for d in before_docs],
            current=doc.text,
            after=[d.text for d in after_docs],
            max_chars=max_chars,
        )
        metadata = dict(hit.metadata)
        metadata.setdefault("section_title", doc.metadata.get("section_title", ""))
        metadata["context_expanded"] = True
        expanded.append(
            SearchHit(
                id=hit.id,
                text=context_text,
                metadata=metadata,
                score=hit.score,
            )
        )

    return expanded


def _walk_neighbors(
    doc: StoredDoc,
    docs_by_id: Dict[str, StoredDoc],
    field: str,
    limit: int,
) -> List[StoredDoc]:
    out: List[StoredDoc] = []
    current = doc
    seen = {doc.id}
    for _ in range(limit):
        next_id = current.metadata.get(field)
        if not next_id or next_id in seen:
            break
        neighbor = docs_by_id.get(str(next_id))
        if neighbor is None:
            break
        if any(neighbor.metadata.get(key) != doc.metadata.get(key)
               for key in ("year", "source_file")):
            break
        out.append(neighbor)
        seen.add(neighbor.id)
        current = neighbor
    return out


def _compose_context(
    *,
    before: List[str],
    current: str,
    after: List[str],
    max_chars: int,
) -> str:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    current_text = current.strip()
    if len(current_text) >= max_chars:
        return current_text[:max_chars].rstrip()

    # Remove exact window overlap, not token-set similarity (which can erase
    # changed numbers or negations). The anchor is never displaced by neighbors.
    before_text = "\n\n".join(before).strip()
    after_text = "\n\n".join(after).strip()
    for size in range(min(len(before_text), len(current_text)), 19, -1):
        if before_text[-size:] == current_text[:size]:
            before_text = before_text[:-size].rstrip()
            break
    for size in range(min(len(after_text), len(current_text)), 19, -1):
        if current_text[-size:] == after_text[:size]:
            after_text = after_text[size:].lstrip()
            break

    remaining = max_chars - len(current_text)
    # Count separators as well as text; never use [-0:] (the whole string).
    slots = int(bool(before_text)) + int(bool(after_text))
    available = max(0, remaining - 2 * slots)
    before_budget = min(len(before_text), available // slots) if slots and before_text else 0
    after_budget = min(len(after_text), available - before_budget)
    before_budget = min(len(before_text), available - after_budget)
    before_text = before_text[-before_budget:].lstrip() if before_budget else ""
    after_text = after_text[:after_budget].rstrip() if after_budget else ""
    return "\n\n".join(part for part in (before_text, current_text, after_text) if part)
