"""Context expansion helpers for answer generation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from src.storage import SearchHit, StoredDoc


def build_doc_lookup(docs: Iterable[StoredDoc]) -> Dict[str, StoredDoc]:
    return {doc.id: doc for doc in docs}


def expand_hits_with_neighbors(
    hits: List[SearchHit],
    docs_by_id: Dict[str, StoredDoc],
    *,
    neighbors: int = 1,
    max_chars: int = 2600,
    separate_neighbors: bool = False,
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

        if separate_neighbors:
            # Evidence from a neighbor must retain that neighbor's identity.
            # Interleave its own text after the anchor, rather than attributing
            # every adjacent claim to the anchor's citation number.
            existing = {item.id for item in expanded}
            for item in (doc, *before_docs, *after_docs):
                if item.id not in existing:
                    expanded.append(SearchHit(item.id, item.text,
                                              {**item.metadata, "merged_ids": [item.id]}, hit.score))
                    existing.add(item.id)
            continue

        context_text, before_spans, after_spans, anchor_span = _compose_context_tracked(
            before=[d.text for d in before_docs],
            current=doc.text,
            after=[d.text for d in after_docs],
            max_chars=max_chars,
        )
        metadata = dict(hit.metadata)
        # Ids of every chunk whose text is in the merged text: own id first, then the
        # neighbours in text order. merged_spans keeps their (start, end) in the text so
        # that a later truncation can drop the ids it cuts off.
        located = [(d.id, sp) for d, sp in zip(before_docs, before_spans) if sp is not None]
        located.append((hit.id, anchor_span))
        located += [(d.id, sp) for d, sp in zip(after_docs, after_spans) if sp is not None]
        metadata["merged_spans"] = [[i, sp[0], sp[1]] for i, sp in located]
        metadata["merged_ids"] = [hit.id] + [i for i, _ in located if i != hit.id]
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
    return _compose_context_tracked(before=before, current=current, after=after, max_chars=max_chars)[0]


def _segment_spans(texts: List[str], lo: int, hi: int, shift: int) -> List[Optional[tuple]]:
    """Per segment of ``"\\n\\n".join(texts)``: its kept (start, end) inside [lo, hi), else None.

    ``shift`` is subtracted from each segment's offset in the joined string, to
    follow the leading-whitespace strip applied to that string. Spans are
    relative to ``lo``.
    """
    spans: List[Optional[tuple]] = []
    pos = 0
    for text in texts:
        start, end = pos - shift, pos - shift + len(text)
        start, end = max(start, lo), min(end, hi)
        spans.append((start - lo, end - lo) if text.strip() and end > start else None)
        pos += len(text) + 2
    return spans


def _compose_context_tracked(
    *,
    before: List[str],
    current: str,
    after: List[str],
    max_chars: int,
) -> tuple:
    """Return (text, before_spans, after_spans, anchor_span).

    The spans locate each neighbour's kept text (None when none of it is in the
    result) and the anchor, as (start, end) offsets into ``text``.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    current_text = current.strip()
    if len(current_text) >= max_chars:
        kept = current_text[:max_chars].rstrip()
        return kept, [None] * len(before), [None] * len(after), (0, len(kept))

    # Remove exact window overlap, not token-set similarity (which can erase
    # changed numbers or negations). The anchor is never displaced by neighbors.
    before_raw = "\n\n".join(before)
    before_shift = len(before_raw) - len(before_raw.lstrip())
    before_text = before_raw.strip()
    after_raw = "\n\n".join(after)
    after_shift = len(after_raw) - len(after_raw.lstrip())
    after_text = after_raw.strip()
    for size in range(min(len(before_text), len(current_text)), 19, -1):
        if before_text[-size:] == current_text[:size]:
            before_text = before_text[:-size].rstrip()
            break
    for size in range(min(len(after_text), len(current_text)), 19, -1):
        if current_text[-size:] == after_text[:size]:
            after_shift += size + (len(after_text[size:]) - len(after_text[size:].lstrip()))
            after_text = after_text[size:].lstrip()
            break

    remaining = max_chars - len(current_text)
    # Count separators as well as text; never use [-0:] (the whole string).
    slots = int(bool(before_text)) + int(bool(after_text))
    available = max(0, remaining - 2 * slots)
    before_budget = min(len(before_text), available // slots) if slots and before_text else 0
    after_budget = min(len(after_text), available - before_budget)
    before_budget = min(len(before_text), available - after_budget)
    before_kept_lo = len(before_text) - before_budget
    if before_budget:
        sliced = before_text[-before_budget:]
        before_kept_lo += len(sliced) - len(sliced.lstrip())
    before_kept_hi = len(before_text)
    before_text = before_text[-before_budget:].lstrip() if before_budget else ""
    after_text = after_text[:after_budget].rstrip() if after_budget else ""
    after_kept_hi = len(after_text)
    text = "\n\n".join(part for part in (before_text, current_text, after_text) if part)
    anchor_start = len(before_text) + 2 if before_text else 0
    anchor_end = anchor_start + len(current_text)
    before_spans = ([_shifted(sp, 0) for sp in _segment_spans(before, before_kept_lo, before_kept_hi, before_shift)]
                    if before_text else [None] * len(before))
    after_spans = ([_shifted(sp, anchor_end + 2) for sp in _segment_spans(after, 0, after_kept_hi, after_shift)]
                   if after_text else [None] * len(after))
    return text, before_spans, after_spans, (anchor_start, anchor_end)


def _shifted(span: Optional[tuple], offset: int) -> Optional[tuple]:
    return None if span is None else (span[0] + offset, span[1] + offset)


_CHARS_PER_TOKEN = 4  # rough estimate used for the prompt budget


def truncate_around_anchor(text: str, anchor: str, max_chars: int) -> str:
    """Cut ``text`` to ``max_chars``, keeping the anchor chunk centred in the window."""
    if len(text) <= max_chars:
        return text
    anchor = anchor.strip()
    start = text.find(anchor) if anchor else -1
    if start < 0:
        return text[:max_chars].rstrip()
    end = start + len(anchor)
    if end - start >= max_chars:
        return text[start:start + max_chars].rstrip()
    spare = max_chars - (end - start)
    lo = max(0, start - spare // 2)
    hi = min(len(text), end + (spare - (start - lo)))
    lo = max(0, hi - max_chars)
    return text[lo:hi].strip()


def _trim_merged(metadata: Dict, original: str, trimmed: str) -> Dict:
    """Copy of ``metadata`` whose merged_ids only name chunks that survive the cut ``original`` -> ``trimmed``."""
    out = dict(metadata)
    spans = out.get("merged_spans")
    if not spans or trimmed == original:
        return out
    offset = original.find(trimmed)
    if offset < 0:
        return out
    lo, hi = offset, offset + len(trimmed)
    kept = [[i, max(a, lo) - lo, min(b, hi) - lo] for i, a, b in spans if min(b, hi) > max(a, lo)]
    out["merged_spans"] = kept
    out["merged_ids"] = [out["merged_ids"][0]] + [i for i, _, _ in kept if i != out["merged_ids"][0]]
    return out


def fit_context_to_llm(
    expanded: Sequence[SearchHit],
    anchors: Sequence[SearchHit],
    query: str,
    *,
    history: Optional[Sequence[Dict[str, str]]] = None,
    max_new_tokens: int,
    n_ctx: int,
    max_passages: int,
    passage_max_chars: int,
    periods: Optional[Sequence[Dict]] = None,
) -> List[SearchHit]:
    """Shrink answer context for a small-window model.

    Keeps the first ``max_passages`` passages, truncates each around its anchor
    chunk and drops trailing passages until the prompt fits
    ``n_ctx - max_new_tokens`` (4 chars/token). Passage order is preserved, so
    the [n] numbering of the returned list is the numbering the model sees.
    """
    from src.generation.prompt import build_cited_prompt
    from src.retrieval.retriever import reserve_period_hits

    periods = list(periods or [])
    selected = reserve_period_hits(expanded, periods, max(1, max_passages))
    protected = {hit.id for hit in reserve_period_hits(expanded, periods, 0)}
    anchor_text = {a.id: a.text for a in anchors}
    kept: List[SearchHit] = []
    for hit in selected:
        text = truncate_around_anchor(hit.text, anchor_text.get(hit.id, ""), passage_max_chars)
        kept.append(SearchHit(id=hit.id, text=text, metadata=_trim_merged(hit.metadata, hit.text, text),
                              score=hit.score))

    budget_chars = max(0, n_ctx - max_new_tokens) * _CHARS_PER_TOKEN
    while kept and len(build_cited_prompt(query, kept, history)) > budget_chars:
        drop = next((i for i in reversed(range(len(kept))) if kept[i].id not in protected), None)
        if len(kept) > max(1, len(protected)) and drop is not None:
            kept.pop(drop)
            continue
        # Once only reserved passages remain, trim text, never a period slot.
        # Fail explicitly when even empty passages cannot fit the fixed prompt.
        lengths = [len(hit.text) for hit in kept]
        index = max(range(len(kept)), key=lambda i: lengths[i])
        if lengths[index] <= 1:
            raise ValueError("LLM context is too small for the grounded prompt")
        hit = kept[index]
        cap = max(1, lengths[index] // 2)
        shrunk = truncate_around_anchor(hit.text, anchor_text.get(hit.id, ""), cap)
        kept[index] = SearchHit(hit.id, shrunk, _trim_merged(hit.metadata, hit.text, shrunk), hit.score)
    return kept
