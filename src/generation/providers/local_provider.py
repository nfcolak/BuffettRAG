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

_MAX_ANCHORS = 3
_MAX_SENTENCES = 6
_CONTEXT_RADIUS = 2
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

        # Select a few high-overlap anchor sentences from distinct passages,
        # then include nearby sentences from the same passage. Facts such as a
        # quantity or consequence often follow the sentence that names the event.
        passage_sentences: List[List[str]] = []
        scored: List[Tuple[float, int, int, int, str]] = []
        for rank, (number, text) in enumerate(passages):
            sentences = _split_sentences(text)
            passage_sentences.append(sentences)
            for position, sentence in enumerate(sentences):
                overlap = len(query_words & set(_content_words(sentence)))
                if overlap == 0:
                    continue
                score = overlap + (len(passages) - rank) * 0.01
                scored.append((score, rank, int(number), position, sentence))

        if not scored:
            return REFUSAL_LINE

        scored.sort(key=lambda item: item[0], reverse=True)
        anchors: List[Tuple[float, int, int, int]] = []
        seen_passages = set()
        seen_sentences = set()
        for score, rank, number, position, sentence in scored:
            fingerprint = re.sub(r"\W+", "", sentence.lower())
            if rank in seen_passages or fingerprint in seen_sentences:
                continue
            anchors.append((score, rank, number, position))
            seen_passages.add(rank)
            seen_sentences.add(fingerprint)
            if len(anchors) >= _MAX_ANCHORS:
                break

        candidates: List[Tuple[float, int, int, int]] = list(anchors)
        for score, rank, number, position in anchors:
            sentences = passage_sentences[rank]
            for distance in range(1, _CONTEXT_RADIUS + 1):
                for neighbor in (position - distance, position + distance):
                    if 0 <= neighbor < len(sentences):
                        candidates.append((score - distance * 0.1, rank, number, neighbor))
        candidates.sort(key=lambda item: item[0], reverse=True)

        budget = (max_new_tokens or 300) * _CHARS_PER_TOKEN
        picked: List[Tuple[int, int, str]] = []
        used_chars = 0
        seen_locations = set()
        seen_sentences = set()
        for _, rank, number, position in candidates:
            location = (rank, position)
            sentence = passage_sentences[rank][position]
            fingerprint = re.sub(r"\W+", "", sentence.lower())
            if location in seen_locations or fingerprint in seen_sentences:
                continue
            rendered_length = len(sentence) + len(str(number)) + 3
            separator_length = 2 if picked else 0
            if used_chars + separator_length + rendered_length > budget:
                continue
            picked.append((number, position, sentence))
            seen_locations.add(location)
            seen_sentences.add(fingerprint)
            used_chars += separator_length + rendered_length
            if len(picked) >= _MAX_SENTENCES:
                break

        if not picked:
            return REFUSAL_LINE
        picked.sort(key=lambda item: (item[0], item[1]))
        return "\n\n".join(f"{sentence} [{number}]" for number, _, sentence in picked)
