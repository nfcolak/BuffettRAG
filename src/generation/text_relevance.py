"""Lexical relevance helpers shared by the evidence gate.

Question/passage word overlap over a bounded sentence window, plus the sentence
splitter that feeds it. These helpers decide whether retrieved evidence is
relevant enough to attempt an answer; they never produce answer text.
"""

from __future__ import annotations

import re
from typing import List

from nltk.stem import PorterStemmer

from src.evaluation.claim_validator import evidence_sentences

_WORD_RE = re.compile(r"[a-z']+")
_SALUTATION_RE = re.compile(r"\bTo the (?:Stockholders|Shareholders) of [^:\n]+:\s*", re.I)

_STOPWORDS = frozenset(
    """a about after all also an and any are as at be because been but buffett buffetts by can
    could did do does for from had has have how i if in into is it its just like more most not
    of on or over said say says she so some such than that the their them then there these they
    this to was we were what when which who why will with would you your berkshire letter
    letters shareholder should must may might him his her hers our us me much many
    compare comparison toward""".split()
)

MIN_QUESTION_OVERLAP = 0.30
RELEVANCE_WINDOW = 3
_STEMMER = PorterStemmer()


def content_words(text: str) -> List[str]:
    text = text.lower().replace("’", "'")
    text = re.sub(r"\b([a-z]+)'s\b", r"\1", text)
    # Numeric percent notation and the question's unit are the same term;
    # amounts themselves are still checked by the typed claim validator.
    text = text.replace("%", " percent ")
    return ["percent" if w == "percentage" else w
            for w in _WORD_RE.findall(text) if w not in _STOPWORDS and len(w) > 2]


def relevance_words(text: str) -> set[str]:
    # Morphological recall belongs in the sufficiency check.
    return {_STEMMER.stem(word) for word in content_words(text)}


def best_question_overlap(terms: set[str], sentences: List[str]) -> float:
    """Measure one bounded context window, never union unrelated passages.

    An answer's topic, action and quantity may occupy adjacent sentences. A
    single-sentence denominator wrongly rejects multi-part questions, while
    whole-corpus overlap licenses incidental terms from unrelated evidence.
    """
    if not terms:
        return 0.0
    sentence_terms = [relevance_words(sentence) for sentence in sentences]
    return max((len(terms & set().union(*sentence_terms[start:start + RELEVANCE_WINDOW])) / len(terms)
                for start in range(len(sentence_terms))), default=0.0)


def is_header(line: str) -> bool:
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


def split_sentences(text: str) -> List[str]:
    text = _SALUTATION_RE.sub("", text)
    prose = "\n".join(line for line in text.splitlines() if not is_header(line))
    return [s.strip() for s in evidence_sentences(prose) if len(s.strip()) >= 20 and not is_header(s)]
