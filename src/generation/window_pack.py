"""Finance reader window packing (finance-recall-75-r4, group 1): FINANCE_WINDOW_PACK.

When the served passages overflow the reader window, the plain trim drops whole passages, lowest ranks first. Packing
first takes back the room that other text takes, so more distinct ranked passages stay in the same window:

  1: the merged neighbour chunks of the lowest-ranked hits go first. A hit's text shrinks to its own chunk and the
     dropped chunks leave its `merged_ids` (and `merged_spans`), so `served_ids` and the merged measure only name text
     the reader sees. A whole hit is dropped only when no neighbour text is left;
  2: as 1, then the lowest-ranked passages beyond today's budget (`floor`) that are still over the window are shortened
     to their sentences with the most question words (`relevance_words`), kept in their own order, at least
     MIN_SENTENCES each, before a whole hit is dropped.

A hit keeps its id and its place, so the passage order and every [n] marker are those of the plain trim. Every step is
checked with the caller's counter (the reader trim passes `token_count.count_tokens_and_source`). When whole hits still
have to go, they are dropped from the fully packed list, and the kept hits are then packed again from their full text,
so neighbour text and sentences only go where the room is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from src.evaluation.claim_validator import evidence_sentences
from src.generation.text_relevance import relevance_words
from src.storage import SearchHit

MIN_SENTENCES = 2  # a shortened passage keeps at least this many sentences

Compose = Callable[[List[SearchHit]], Tuple[List[SearchHit], str]]  # served (rank order) -> (prompt order, prompt)
Count = Callable[[str], Tuple[int, str]]                            # prompt -> (size, source)


@dataclass
class Packed:
    """A packed served list: `served` in rank order, `ordered` and `prompt` as `compose` built them."""
    served: List[SearchHit]
    ordered: List[SearchHit]
    prompt: str
    used: int
    source: str
    debug: Dict[str, Any] = field(default_factory=dict)


def count_chars(text: str) -> Tuple[int, str]:
    """The character window's counter."""
    return len(text), "chars"


def _with_text(hit: SearchHit, text: str) -> SearchHit:
    """`hit` with `text` as its only chunk: merged_ids and merged_spans name the hit's own id alone."""
    metadata = dict(hit.metadata)
    metadata["merged_ids"] = [hit.id]
    metadata["merged_spans"] = [[hit.id, 0, len(text)]]
    return SearchHit(hit.id, text, metadata, hit.score)


def own_chunk(hit: SearchHit) -> Optional[SearchHit]:
    """`hit` with the merged neighbour text removed, or None when no neighbour text is merged into it."""
    spans = hit.metadata.get("merged_spans") or []
    own = next((span for span in spans if span[0] == hit.id), None)
    if own is None or len(spans) < 2:
        return None
    text = hit.text[own[1]:own[2]].strip()
    return _with_text(hit, text) if text else None


def shortenings(hit: SearchHit, terms: Set[str]) -> List[SearchHit]:
    """Ever shorter copies of `hit`: its sentences with the fewest question words (`terms`) dropped one at a time (the
    later sentence first on a tie), down to MIN_SENTENCES; the kept sentences stay in their order. Empty when the
    passage has MIN_SENTENCES sentences or fewer."""
    sentences = [s.strip() for s in evidence_sentences(hit.text) if s.strip()]
    if len(sentences) <= MIN_SENTENCES:
        return []
    overlap = [len(terms & relevance_words(s)) for s in sentences]
    order = sorted(range(len(sentences)), key=lambda i: (overlap[i], -i))[:len(sentences) - MIN_SENTENCES]
    dropped: Set[int] = set()
    out = []
    for i in order:
        dropped.add(i)
        out.append(_with_text(hit, " ".join(s for j, s in enumerate(sentences) if j not in dropped)))
    return out


class _State:
    """The served list being packed, with its prompt and size."""

    def __init__(self, served: Sequence[SearchHit], compose: Compose, count: Count) -> None:
        self.served, self.compose, self.count = list(served), compose, count
        self.ordered, self.prompt = compose(self.served)
        self.used, self.source = count(self.prompt)

    def refresh(self) -> None:
        self.ordered, self.prompt = self.compose(self.served)
        self.used, self.source = self.count(self.prompt)


def _apply(state: _State, steps: Sequence[Tuple[int, SearchHit]], window: int) -> int:
    """Apply the shortest prefix of `steps` ((index, replacement) pairs) that makes the prompt fit, or all of them;
    returns how many were applied.

    Steps go in batches: as many as the characters they save cover the deficit at the prompt's own size per character,
    then the prompt is built and counted again. A batch that fits is bisected down to its shortest fitting prefix, so
    an estimate that saved too much does not drop text the window has room for."""
    applied = 0
    while state.used > window and applied < len(steps):
        start, before = applied, list(state.served)
        rate = state.used / max(1, len(state.prompt))
        saved = 0.0
        while applied < len(steps) and saved < state.used - window:
            index, hit = steps[applied]
            saved += (len(state.served[index].text) - len(hit.text)) * rate
            state.served[index] = hit
            applied += 1
        state.refresh()
        if state.used <= window and applied - start > 1:
            low, high, at = start + 1, applied, applied  # `high` fits; `at` is the prefix the state holds
            while low < high:
                at = (low + high) // 2
                _prefix(state, before, steps, start, at)
                if state.used <= window:
                    high = at
                else:
                    low = at + 1
            if at != high:
                _prefix(state, before, steps, start, high)
            applied = high
    return applied


def _prefix(state: _State, before: List[SearchHit], steps: Sequence[Tuple[int, SearchHit]], start: int,
            end: int) -> None:
    """`state` as `before` with steps[start:end] applied."""
    state.served = list(before)
    for index, hit in steps[start:end]:
        state.served[index] = hit
    state.refresh()


def _squeeze(served: Sequence[SearchHit], compose: Compose, count: Count, window: int, floor: int,
             terms: Optional[Set[str]]) -> _State:
    """Neighbour text off, lowest ranks first; then (terms given) the passages from `floor` on shortened, lowest ranks
    first; each only until the prompt fits."""
    state = _State(served, compose, count)
    _apply(state, [(i, hit) for i in reversed(range(len(state.served)))
                   for hit in [own_chunk(state.served[i])] if hit is not None], window)
    if terms is not None and state.used > window:
        _apply(state, [(i, hit) for i in reversed(range(floor, len(state.served)))
                       for hit in shortenings(state.served[i], terms)], window)
    return state


def _debug(mode: int, before: Sequence[SearchHit], after: Sequence[SearchHit]) -> Dict[str, Any]:
    """What packing changed in the kept hits: hits that lost neighbour text, neighbour ids gone, passages shortened."""
    stripped = merged = shortened = 0
    for old, new in zip(before, after):
        lost = len(old.metadata.get("merged_ids") or ()) - len(new.metadata.get("merged_ids") or ())
        stripped += lost > 0
        merged += max(0, lost)
        shortened += len(new.text) < len((own_chunk(old) or old).text)
    return {"window_pack": mode, "pack_stripped": stripped, "pack_merged_dropped": merged,
            "pack_shortened": shortened}


def pack(served: Sequence[SearchHit], compose: Compose, count: Count, window: int, floor: int, query: str,
         mode: int) -> Packed:
    """Pack `served` (rank order) into `window` (FINANCE_WINDOW_PACK `mode` 1 or 2, see the module docstring); whole
    hits are dropped, lowest rank first, only when packing alone does not fit, and never below `floor`."""
    terms = relevance_words(query) if mode >= 2 else None
    state = _squeeze(served, compose, count, window, floor, terms)
    if state.used > window and len(state.served) > floor:
        while len(state.served) > floor and state.used > window:
            state.served.pop()
            state.refresh()
        if state.used <= window:  # pack the kept hits again from their full text: only the room they need
            state = _squeeze(served[:len(state.served)], compose, count, window, floor, terms)
    return Packed(state.served, state.ordered, state.prompt, state.used, state.source,
                  _debug(mode, served, state.served))
