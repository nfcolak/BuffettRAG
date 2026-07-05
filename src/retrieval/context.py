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
    before_text = "\n\n".join(before).strip()
    after_text = "\n\n".join(after).strip()
    current_text = current.strip()

    if len(current_text) >= max_chars:
        return current_text[:max_chars].rstrip()

    remaining = max_chars - len(current_text)
    before_budget = remaining // 2
    after_budget = remaining - before_budget

    before_text = before_text[-before_budget:].lstrip() if before_text else ""
    after_text = after_text[:after_budget].rstrip() if after_text else ""

    parts = []
    if before_text:
        parts.append(before_text)
    parts.append(current_text)
    if after_text:
        parts.append(after_text)

    return "\n\n".join(parts)
