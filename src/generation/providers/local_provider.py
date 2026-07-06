"""Embedded local answer engine (no external API).

A deliberately lightweight extractive engine: it picks the passage sentences
that best match the question and returns them verbatim with [n] citations.
It needs no API key, no network access and no model weights, so the app can
run fully offline. Answer quality is intentionally below cloud LLMs -- it
selects and quotes, it does not reason or paraphrase.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.generation.prompt import REFUSAL_LINE

_PASSAGE_RE = re.compile(
    r"\[(\d+)\]\s*\(year=[^)]*\)\n(.*?)(?=\n\n\[\d+\]\s*\(year=|\n+END UNTRUSTED PASSAGES)",
    re.DOTALL,
)
_QUESTION_RE = re.compile(r"BEGIN USER QUESTION\s*\nQuestion:\s*(.+)")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_WORD_RE = re.compile(r"[a-z']+")

_STOPWORDS = frozenset(
    """a about after all also an and any are as at be because been but buffett buffetts by can
    could did do does for from had has have how i if in into is it its just like more most not
    of on or over said say says she so some such than that the their them then there these they
    this to was we were what when which who why will with would you your berkshire letter
    letters shareholder""".split()
)

_MAX_SENTENCES = 3
_CHARS_PER_TOKEN = 4


def _content_words(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2]


def _split_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENTENCE_RE.split(normalized) if len(s.strip()) >= 40]


class LocalProvider:
    """Extractive fallback provider registered as ``local`` in the factory."""

    provider_name = "local"

    def __init__(self, model: str = "embedded-extractive-v1") -> None:
        self.model = model

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        question_match = _QUESTION_RE.search(prompt)
        passages = _PASSAGE_RE.findall(prompt)
        if not question_match or not passages:
            return REFUSAL_LINE

        query_words = set(_content_words(question_match.group(1)))
        if not query_words:
            return REFUSAL_LINE

        # Score every sentence by content-word overlap with the question,
        # with a small bonus for earlier (higher-ranked) passages.
        scored: List[Tuple[float, int, str]] = []
        for rank, (number, text) in enumerate(passages):
            for sentence in _split_sentences(text):
                overlap = len(query_words & set(_content_words(sentence)))
                if overlap == 0:
                    continue
                score = overlap + (len(passages) - rank) * 0.01
                scored.append((score, int(number), sentence))

        if not scored:
            return REFUSAL_LINE

        scored.sort(key=lambda item: item[0], reverse=True)

        budget = (max_new_tokens or 300) * _CHARS_PER_TOKEN
        picked: List[Tuple[int, str]] = []
        used_chars = 0
        seen_passages = set()
        seen_sentences = set()
        for _, number, sentence in scored:
            # At most one sentence per passage keeps the answer diverse, and
            # overlapping chunks can repeat the same sentence verbatim.
            fingerprint = re.sub(r"\W+", "", sentence.lower())
            if number in seen_passages or fingerprint in seen_sentences:
                continue
            if used_chars + len(sentence) > budget and picked:
                break
            picked.append((number, sentence))
            seen_passages.add(number)
            seen_sentences.add(fingerprint)
            used_chars += len(sentence)
            if len(picked) >= _MAX_SENTENCES:
                break

        picked.sort(key=lambda item: item[0])
        return "\n\n".join(f"{sentence} [{number}]" for number, sentence in picked)
