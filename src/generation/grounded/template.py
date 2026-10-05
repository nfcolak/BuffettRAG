"""Model-free composer: a pure function of the request's evidence.

The answer is built from verbatim evidence sentences, each followed by its
local marker [k] (k = local_index + 1). Anchor sentences (those overlapping
their unit's anchor span) come first, best unit score first (ties in document
order); neighbour sentences follow only while sentences and budget remain.
The output obeys a character budget
(max_tokens * CHARS_PER_TOKEN) and at most MAX_SENTENCES sentences; it only
ever cuts at sentence boundaries, never inside a sentence, so numbers and
qualifiers stay exactly as in the source. Sentences that contain citation-like
bracket syntax are skipped because they cannot be cited safely.
"""

from __future__ import annotations

import re

from .types import ComposeRequest, ComposeResult, GroupEvidence

__all__ = ["TemplateComposer", "ProviderUnavailable", "CHARS_PER_TOKEN", "MAX_SENTENCES"]

CHARS_PER_TOKEN = 4
MAX_SENTENCES = 3
_BRACKET_NUMBERS_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")


class ProviderUnavailable(RuntimeError):
    """Raised when a backend cannot serve the request (e.g. template raw_generate)."""


def _is_anchor(entry: GroupEvidence) -> bool:
    span = entry.source_span
    if span is None:
        return True
    anchor = entry.unit.anchor
    return (span.hit_index == anchor.hit_index and span.start < anchor.end
            and anchor.start < span.end)


def _terminated(text: str) -> str:
    return text + "." if text[-1].isalnum() else text


class TemplateComposer:
    backend = "template"
    model_id = "template"

    def compose(self, req: ComposeRequest) -> ComposeResult:
        usable = [e for e in req.evidence if not _BRACKET_NUMBERS_RE.search(e.text)]
        # best unit score first (ties: document order); neighbours only after every anchor
        ranked = sorted(usable, key=lambda e: (-e.unit.score, e.unit.hit_index, e.local_index))
        anchors = [e for e in ranked if e.source_span is not None and _is_anchor(e)]
        chosen_pool = anchors + [e for e in ranked if e not in anchors]
        budget = req.max_tokens * CHARS_PER_TOKEN
        parts: list[str] = []
        seen: set[str] = set()
        used = 0
        for entry in chosen_pool:
            if len(parts) >= MAX_SENTENCES:
                break
            if entry.text in seen:
                continue
            part = f"{_terminated(entry.text)} [{entry.local_index + 1}]"
            cost = len(part) + (1 if parts else 0)
            if used + cost > budget:
                continue
            seen.add(entry.text)
            parts.append(part)
            used += cost
        text = " ".join(parts)
        return ComposeResult(
            text=text, raw=text, backend=self.backend, model_id=self.model_id,
            finish_reason="stop", tokens_in=None, tokens_out=None,
        )

    def raw_generate(self, prompt: str, max_tokens: int) -> str:
        raise ProviderUnavailable("the template composer has no language model")
