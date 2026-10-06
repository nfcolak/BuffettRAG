"""Embedded local answer engine (no external API).

A deliberately lightweight extractive engine: it leads with the passage sentence
that best answers the question, adds at most two supporting sentences, and
returns them verbatim with [n] citations.
It needs no API key, no network access and no model weights, so the app can
run fully offline. Answer quality is intentionally below cloud LLMs -- it
selects and quotes, it does not reason or paraphrase.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from nltk.stem import PorterStemmer

from src.generation.prompt import REFUSAL_LINE
from src.evaluation.claim_validator import evidence_sentences

_PASSAGE_RE = re.compile(
    r"\[(\d+)\]\s*\(year=[^)]*\)\n(.*?)(?=\n\n\[\d+\]\s*\(year=|\n+END UNTRUSTED PASSAGES)",
    re.DOTALL,
)
_QUESTION_RE = re.compile(r"BEGIN USER QUESTION\s*\nQuestion:\s*(.+)")
_WORD_RE = re.compile(r"[a-z']+")
_SALUTATION_RE = re.compile(r"\bTo the (?:Stockholders|Shareholders) of [^:\n]+:\s*", re.I)
_QUANTITY_QUESTION_RE = re.compile(r"\b(?:how much|how many|what percent(?:age)?|what (?:was|were|is|are).{0,50}(?:earnings|cost|amount|total))\b", re.I)
_NUMBER_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")

_STOPWORDS = frozenset(
    """a about after all also an and any are as at be because been but buffett buffetts by can
    could did do does for from had has have how i if in into is it its just like more most not
    of on or over said say says she so some such than that the their them then there these they
    this to was we were what when which who why will with would you your berkshire letter
    letters shareholder should must may might him his her hers our us me much many
    compare comparison toward""".split()
)

_MAX_SUPPORTING = 2
_CONTEXT_RADIUS = 2
_CHARS_PER_TOKEN = 4
_MIN_QUESTION_OVERLAP = 0.30
_RELEVANCE_WINDOW = 3
_STEMMER = PorterStemmer()


def _content_words(text: str) -> List[str]:
    text = text.lower().replace("’", "'")
    text = re.sub(r"\b([a-z]+)'s\b", r"\1", text)
    # Numeric percent notation and the question's unit are the same term;
    # amounts themselves are still checked by the typed claim validator.
    text = text.replace("%", " percent ")
    return ["percent" if w == "percentage" else w
            for w in _WORD_RE.findall(text) if w not in _STOPWORDS and len(w) > 2]


def _relevance_words(text: str) -> set[str]:
    # Stemmed terms drive both the sufficiency check and sentence ranking, so
    # "pay"/"paying" and "fund"/"funds" count as the same question term.
    return {_STEMMER.stem(word) for word in _content_words(text)}


def _best_question_overlap(terms: set[str], sentences: List[str]) -> float:
    """Measure one bounded context window, never union unrelated passages.

    An answer's topic, action and quantity may occupy adjacent sentences. A
    single-sentence denominator wrongly rejects multi-part questions, while
    whole-corpus overlap licenses incidental terms from unrelated evidence.
    """
    if not terms:
        return 0.0
    sentence_terms = [_relevance_words(sentence) for sentence in sentences]
    return max((len(terms & set().union(*sentence_terms[start:start + _RELEVANCE_WINDOW])) / len(terms)
                for start in range(len(sentence_terms))), default=0.0)


def _is_header(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if re.fullmatch(r"[\d\s*•Š-]+", line):
        return True
    if re.match(r"^(?:Chairman of the Board|Vice Chairman|Page\s+\d+)\b", line, re.I):
        return True
    # Standalone signatures, including middle initials.
    if re.fullmatch(r"[A-Z][a-z]+\s+(?:[A-Z]\.\s+)?[A-Z][a-z]+", line):
        return True
    words = line.split()
    return (len(line) <= 80 and len(words) <= 8 and not re.search(r"[.!?:\d]", line)
            and all(w[:1].isupper() or w.lower() in {"of", "the", "and", "to", "a"} for w in words))


def _split_sentences(text: str) -> List[str]:
    text = _SALUTATION_RE.sub("", text)
    prose = "\n".join(line for line in text.splitlines() if not _is_header(line))
    return [s.strip() for s in evidence_sentences(prose) if len(s.strip()) >= 20 and not _is_header(s)]


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

        question = question_match.group(1)
        query_words = set(_content_words(question))
        relevance_words = _relevance_words(question)
        asks_quantity = bool(_QUANTITY_QUESTION_RE.search(question))
        question_numbers = set(_NUMBER_RE.findall(question))
        if not query_words:
            return REFUSAL_LINE

        passage_sentences: List[List[str]] = []
        scored: List[Tuple[float, int, int, int, str]] = []
        best_overlap = 0.0
        for rank, (number, text) in enumerate(passages):
            sentences = _split_sentences(text)
            passage_sentences.append(sentences)
            best_overlap = max(best_overlap, _best_question_overlap(relevance_words, sentences))
            for position, sentence in enumerate(sentences):
                overlap = len(relevance_words & _relevance_words(sentence))
                if overlap == 0:
                    continue
                quantities = set(_NUMBER_RE.findall(sentence))
                # Quantities complement relevance, never substitute for it.
                numeric_bonus = 0.75 if asks_quantity and quantities else 0.0
                number_bonus = 0.25 * len(question_numbers & quantities)
                score = overlap + numeric_bonus + number_bonus + (len(passages) - rank) * 0.01
                scored.append((score, rank, int(number), position, sentence))

        if not scored or best_overlap < _MIN_QUESTION_OVERLAP:
            return REFUSAL_LINE

        scored.sort(key=lambda item: item[0], reverse=True)
        budget = (max_new_tokens or 300) * _CHARS_PER_TOKEN

        def rendered(number: int, sentence: str) -> int:
            return len(sentence) + len(str(number)) + 3

        # Lead with the single best answering sentence, then add at most
        # _MAX_SUPPORTING cited sentences. A longer extract buries the answer.
        lead = None
        for score, rank, number, position, sentence in scored:
            if rendered(number, sentence) <= budget:
                lead = (rank, number, position, sentence)
                break
        if lead is None:
            return REFUSAL_LINE
        lead_rank, lead_number, lead_position, lead_sentence = lead
        used_chars = rendered(lead_number, lead_sentence) + 2  # paragraph break

        # Support candidates, in priority order: the lead's best neighbour (a
        # quantity or consequence often follows the sentence naming the event),
        # then the best sentences from other passages, then any remaining match.
        sentence_scores = {(rank, position): score for score, rank, _, position, _ in scored}
        lead_sentences = passage_sentences[lead_rank]
        neighbors = []
        for distance in range(1, _CONTEXT_RADIUS + 1):
            for neighbor in (lead_position - distance, lead_position + distance):
                if 0 <= neighbor < len(lead_sentences):
                    # The neighbour's own relevance, not the lead's; on ties
                    # prefer nearer context, then a concrete figure, then the
                    # following sentence.
                    neighbors.append((sentence_scores.get((lead_rank, neighbor), 0.0), -distance,
                                      bool(_NUMBER_RE.search(lead_sentences[neighbor])), neighbor, neighbor))
        candidates: List[Tuple[int, int, int]] = []
        best_neighbor = max(neighbors, default=None)
        # A neighbour with no question term and no figure is filler, not support.
        if best_neighbor and (best_neighbor[0] > 0 or best_neighbor[2]):
            candidates.append((lead_rank, lead_number, best_neighbor[4]))
        candidates += [(rank, number, position) for _, rank, number, position, _ in scored
                       if rank != lead_rank]
        candidates += [(rank, number, position) for _, rank, number, position, _ in scored
                       if rank == lead_rank]

        supporting: List[Tuple[int, int, str]] = []
        seen_locations = {(lead_rank, lead_position)}
        seen_sentences = {re.sub(r"\W+", "", lead_sentence.lower())}
        seen_passages = set()
        for rank, number, position in candidates:
            sentence = passage_sentences[rank][position]
            fingerprint = re.sub(r"\W+", "", sentence.lower())
            if (rank, position) in seen_locations or fingerprint in seen_sentences:
                continue
            # One supporting sentence per other passage keeps the support diverse.
            if rank != lead_rank and rank in seen_passages:
                continue
            separator_length = 1 if supporting else 0
            if used_chars + separator_length + rendered(number, sentence) > budget:
                continue
            supporting.append((number, position, sentence))
            seen_locations.add((rank, position))
            seen_sentences.add(fingerprint)
            seen_passages.add(rank)
            used_chars += separator_length + rendered(number, sentence)
            if len(supporting) >= _MAX_SUPPORTING:
                break

        answer = f"{lead_sentence} [{lead_number}]"
        if supporting:
            supporting.sort(key=lambda item: (item[0], item[1]))
            answer += "\n\n" + " ".join(f"{sentence} [{number}]" for number, _, sentence in supporting)
        return answer
