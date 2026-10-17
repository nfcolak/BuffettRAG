"""Neighbour-sentence expansion of validated answers.

The 7B quotes one sentence per cited passage while most questions need the sentence next to it.
For every kept answer sentence that is a verbatim span of its cited passage, add up to N following
and M preceding sentences of the same passage, in passage order, with the same citation marker.
Only text found in the passage is ever added; each added sentence must also pass the existing
claim validator on its own. Two caps: a model answer that already has ANSWER_MAX_SENTENCES sentences is not expanded, and the
expanded answer is capped at ANSWER_EXPANDED_MAX_SENTENCES (0 = max(ANSWER_MAX_SENTENCES, 4 x the raw sentences)).

Settings (environment, read at call time; defaults in config.py): ANSWER_NEIGHBOR_AFTER,
ANSWER_NEIGHBOR_BEFORE, ANSWER_MAX_SENTENCES, ANSWER_EXPANDED_MAX_SENTENCES.
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence

from src.evaluation.claim_validator import _split_original, evidence_sentences, validate_and_filter_answer

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_TRAILING_MARKERS_RE = re.compile(r"(?:\s*\[\d+(?:\s*,\s*\d+)*\])+\s*$")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")
_PERIOD_LABEL_RE = re.compile(r"^In [^:\n]{1,40}:\s+")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "−": "-", "–": "-", "—": "-"})
_MIN_SPAN_CHARS = 12
_MIN_WORDS = 6
_MAX_WORDS = 70  # longer "sentences" are mis-split run-ons (section breaks, footnotes)
_SCAN_EXTRA = 3  # skipped (table/heading) lines examined beyond the requested count


def _setting(name: str) -> int:
    raw = os.getenv(name)
    if raw not in (None, ""):
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    import config
    return max(0, int(getattr(config, name)))


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = _CITATION_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", text)


def _core(text: str) -> str:
    """Normalised sentence body without closing punctuation/quotes (the model often changes those)."""
    return _norm(text).strip(" \"'").rstrip(".!?\"' ").strip()


def _hit_text(hit: Any) -> str:
    return (hit.get("text") if isinstance(hit, dict) else getattr(hit, "text", "")) or ""


def is_prose_sentence(sentence: str) -> bool:
    """False for table rows, headings, leader dots and fragments; only prose is added."""
    s = sentence.strip()
    words = s.split()
    if not _MIN_WORDS <= len(words) <= _MAX_WORDS or re.search(r"\*\s*\*", s) or not re.search(r"[.!?][\"”’')\]]*$", s):
        return False
    if s.upper() == s or s[0].islower() or " | " in s or re.search(r"\.{4,}|(?:\.\s){3,}", s):
        return False
    letters = sum(c.isalpha() for c in s)
    if letters / max(1, len(s.replace(" ", ""))) < 0.6:
        return False
    numeric = sum(any(c.isdigit() for c in w) for w in words)
    return numeric / len(words) <= 0.35


class _Passage:
    def __init__(self, text: str):
        self.sentences = evidence_sentences(text)
        norms = [_norm(s) for s in self.sentences]
        self.joined = " ".join(norms)
        self.starts, pos = [], 0
        for n in norms:
            self.starts.append(pos)
            pos += len(n) + 1
        self.ends = [s + len(n) for s, n in zip(self.starts, norms)]

    def locate(self, core: str) -> Optional[tuple]:
        """(first, last) sentence indexes covered by `core` when it is a verbatim span, else None."""
        if len(core) < _MIN_SPAN_CHARS:
            return None
        start = self.joined.find(core)
        if start < 0:
            return None
        end = start + len(core)
        covered = [i for i in range(len(self.sentences)) if self.starts[i] < end and self.ends[i] > start]
        return (covered[0], covered[-1]) if covered else None


def _marker_of(sentence: str) -> str:
    tail = _TRAILING_MARKERS_RE.search(sentence)
    if tail:
        return tail.group(0).strip()
    return "".join(dict.fromkeys(f"[{m}]" for m in _CITATION_RE.findall(sentence)))


def _valid_indexes(sentence: str, n_hits: int) -> List[int]:
    out: List[int] = []
    for raw in _CITATION_RE.findall(sentence):
        for n in raw.split(","):
            if n.strip().isdigit() and 1 <= int(n) <= n_hits and int(n) - 1 not in out:
                out.append(int(n) - 1)
    return out


def _passes_validator(sentence: str, hits: Sequence[Any]) -> bool:
    kept = validate_and_filter_answer(sentence, hits).safe_answer
    return re.sub(r"\s+", " ", kept).strip() == re.sub(r"\s+", " ", sentence).strip()


def expand_answer(answer: str, hits: Sequence[Any], *, after: Optional[int] = None, before: Optional[int] = None,
                  max_sentences: Optional[int] = None, expanded_max_sentences: Optional[int] = None) -> str:
    """Return `answer` with passage neighbours added to verbatim-located sentences (unchanged if none apply).

    `max_sentences` is the model-answer cap and, when `expanded_max_sentences` is not given, the expanded cap too.
    """
    after = _setting("ANSWER_NEIGHBOR_AFTER") if after is None else max(0, after)
    before = _setting("ANSWER_NEIGHBOR_BEFORE") if before is None else max(0, before)
    cap = _setting("ANSWER_MAX_SENTENCES") if max_sentences is None else max_sentences
    if expanded_max_sentences is None:
        expanded_max_sentences = cap if max_sentences is not None else _setting("ANSWER_EXPANDED_MAX_SENTENCES")
    if (after == 0 and before == 0) or not answer or not hits:
        return answer

    lines: List[Dict[str, Any]] = []
    total = 0
    for raw_line in answer.split("\n"):
        if not raw_line.strip():
            lines.append({"blank": True})
            continue
        prefix = _LIST_PREFIX_RE.match(raw_line)
        body = raw_line[prefix.end():] if prefix else raw_line
        items = []
        for index, sentence in enumerate(_split_original(body)):
            label = ""
            if index == 0:
                found = _PERIOD_LABEL_RE.match(sentence)
                label = found.group(0) if found else ""
            items.append({"label": label, "text": sentence[len(label):].strip(), "before": [], "after": []})
        total += len(items)
        lines.append({"blank": False, "prefix": prefix.group(0) if prefix else "", "items": items})
    if total >= cap:
        return answer
    budget = (expanded_max_sentences or max(cap, 4 * total)) - total
    if budget <= 0:
        return answer

    passages: Dict[int, _Passage] = {}
    seen = {_core(item["text"]) for ln in lines if not ln["blank"] for item in ln["items"]}
    anchors = []
    for ln in lines:
        if ln["blank"]:
            continue
        for item in ln["items"]:
            core = _core(item["text"])
            for idx in _valid_indexes(item["text"], len(hits)):
                passage = passages.setdefault(idx, _Passage(_hit_text(hits[idx])))
                span = passage.locate(core)
                if span:
                    anchors.append((item, passage, span, _marker_of(item["text"])))
                    break

    def candidates(passage: _Passage, start: int, step: int, wanted: int) -> List[str]:
        out, i, scanned = [], start, 0
        while 0 <= i < len(passage.sentences) and len(out) < wanted and scanned < wanted + _SCAN_EXTRA:
            sentence = passage.sentences[i]
            if _core(sentence) in seen:
                break  # contiguity: never jump over a sentence the answer already has
            if is_prose_sentence(sentence):
                out.append(sentence)
            i += step
            scanned += 1
        return out

    pools = [(item, marker, candidates(p, hi + 1, 1, after) if after else [],
              candidates(p, lo - 1, -1, before) if before else [])
             for item, p, (lo, hi), marker in anchors]
    for distance in range(max(after, before)):
        for item, marker, forward, backward in pools:
            for pool, bucket in ((forward, item["after"]), (backward, item["before"])):
                if budget <= 0:
                    break
                if distance >= len(pool) or len(bucket) != distance:
                    continue
                sentence = pool[distance]
                key = _core(sentence)
                if key in seen or not _passes_validator(f"{sentence} {marker}", hits):
                    continue
                seen.add(key)
                bucket.append(f"{sentence} {marker}")
                budget -= 1

    out_lines = []
    for ln in lines:
        if ln["blank"]:
            out_lines.append("")
            continue
        parts = []
        for item in ln["items"]:
            parts.append(item["label"] + " ".join(item["before"][::-1] + [item["text"]] + item["after"]))
        out_lines.append(ln["prefix"] + " ".join(parts))
    return "\n".join(out_lines)
