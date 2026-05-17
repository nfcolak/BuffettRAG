"""Prompt templates and citation parsing.

The model is instructed to cite passages as [n], where n is a 1-based index
into the provided context. We then verify in `parse_citations` that the
numbers map to real passages.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from src.vector_store import SearchHit


SYSTEM_PROMPT = (
    "You are a careful research assistant answering questions about Warren "
    "Buffett's annual shareholder letters (1977-2024). Use ONLY the provided "
    "passages. If the passages do not contain the answer, say so plainly. "
    "When you make a claim, cite the supporting passage(s) inline using "
    "bracket numbers like [1] or [2,3]. Do not fabricate quotes or years."
)


def _passage_block(hits: Sequence[SearchHit]) -> str:
    """Format passages as a numbered list with year headers."""
    lines: List[str] = []
    for i, h in enumerate(hits, start=1):
        year = h.metadata.get("year", "?")
        src = h.metadata.get("source_file", "?")
        lines.append(f"[{i}] (year={year}, source={src})\n{h.text.strip()}")
    return "\n\n".join(lines)


def build_cited_prompt(
    query: str,
    hits: Sequence[SearchHit],
    *,
    style: str = "mistral",
) -> str:
    """Build a prompt asking for a cited answer.

    `style` controls the chat-template formatting. Both Mistral-Instruct and
    Llama 3.1-Instruct accept the [INST] format for short turns; we keep the
    template simple to stay portable.
    """
    passages = _passage_block(hits)
    user_block = (
        "Passages:\n"
        f"{passages}\n\n"
        f"Question: {query}\n\n"
        "Answer with inline citations [n] referring to the passages above. "
        "If multiple passages support a claim, cite them like [1,3]."
    )

    if style == "qwen":
        # Qwen2.5 ChatML format
        return (
            "<|im_start|>system\n"
            f"{SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_block}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    if style == "mistral":
        # Mistral instruct format
        return f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_block} [/INST]"

    if style == "llama3":
        # Llama 3 instruct format
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_block}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    # Plain fallback.
    return f"{SYSTEM_PROMPT}\n\n{user_block}\n\nAnswer:"


_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_citations(answer: str, hits: Sequence[SearchHit]) -> List[Dict]:
    """Extract citation references from the model's answer.

    Returns a list of dicts:
        {'marker': '[1]', 'passage_indices': [0], 'years': [2008], 'sources': ['buffet_2008.pdf']}

    Indices that fall outside the valid range are dropped (the caller can use
    the count as a faithfulness signal).
    """
    out: List[Dict] = []
    for m in _CITATION_RE.finditer(answer):
        raw = m.group(1)
        nums = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
        valid_idxs = [n - 1 for n in nums if 1 <= n <= len(hits)]
        years = [hits[i].metadata.get("year") for i in valid_idxs]
        sources = [hits[i].metadata.get("source_file") for i in valid_idxs]
        out.append(
            {
                "marker": m.group(0),
                "passage_indices": valid_idxs,
                "raw_numbers": nums,
                "years": years,
                "sources": sources,
            }
        )
    return out


def strip_chat_artifacts(text: str) -> str:
    """Remove common boilerplate the model leaks into the answer."""
    text = re.sub(r"<\|.*?\|>", "", text)        # covers Llama 3 + Qwen ChatML
    text = re.sub(r"\[/?INST\]", "", text)       # Mistral
    text = re.sub(r"<s>|</s>", "", text)         # BOS/EOS that some tokenizers emit
    return text.strip()
