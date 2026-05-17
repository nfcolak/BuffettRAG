"""Citation faithfulness metric.

Goal: for every cited passage in an answer, check that the answer's claims
about it are actually grounded in the passage text.

We provide two implementations:

    1. Lexical (default) -- compute n-gram overlap between (a) the sentences
       around each citation marker in the answer and (b) the cited passage.
       High overlap == the model is paraphrasing the passage. Low overlap
       == possible hallucination.

       This is fast, has zero dependencies beyond the standard library, and
       correlates reasonably well with human judgment on RAG outputs.

    2. NLI-based (optional, opt-in) -- use a small NLI model
       (cross-encoder/nli-deberta-v3-small) to score entailment of each
       sentence by its citation. More accurate but adds a model dependency.

Both also report:
    - citation_coverage: fraction of factual sentences that carry any citation
    - invalid_citation_rate: fraction of [n] markers pointing to no passage
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from src.generation.prompt import parse_citations
from src.vector_store import SearchHit


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _ngrams(tokens: List[str], n: int) -> set:
    return set(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def lexical_support_score(sentence: str, passage: str, n: int = 2) -> float:
    """Fraction of n-grams in `sentence` that also appear in `passage`."""
    s_tokens = _tokenize(sentence)
    p_tokens = _tokenize(passage)
    if len(s_tokens) < n or len(p_tokens) < n:
        # Fall back to unigram overlap for very short sentences.
        s_set, p_set = set(s_tokens), set(p_tokens)
        if not s_set:
            return 0.0
        return len(s_set & p_set) / len(s_set)
    s_ng = _ngrams(s_tokens, n)
    p_ng = _ngrams(p_tokens, n)
    if not s_ng:
        return 0.0
    return len(s_ng & p_ng) / len(s_ng)


# -----------------------------------------------------------------------------
# Per-answer faithfulness
# -----------------------------------------------------------------------------

@dataclass
class FaithfulnessReport:
    answer: str
    n_sentences: int
    n_cited_sentences: int
    citation_coverage: float
    invalid_citation_rate: float
    mean_support: float
    per_sentence: List[Dict]


_CITATION_IN_SENT_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def evaluate_faithfulness(
    answer: str,
    hits: Sequence[SearchHit],
    *,
    nli_scorer: Optional[object] = None,
) -> FaithfulnessReport:
    """Compute faithfulness metrics for one (answer, hits) pair.

    Parameters
    ----------
    nli_scorer : optional callable (premise, hypothesis) -> entailment_prob
        If supplied, it overrides the lexical signal for the per-sentence
        support score. Use `make_nli_scorer()` to build one.
    """
    sentences = split_sentences(answer)
    if not sentences:
        return FaithfulnessReport(
            answer=answer,
            n_sentences=0,
            n_cited_sentences=0,
            citation_coverage=0.0,
            invalid_citation_rate=0.0,
            mean_support=0.0,
            per_sentence=[],
        )

    # Identify citations we know about overall (for invalid-rate denominator).
    all_citations = parse_citations(answer, hits)
    total_markers = len(_CITATION_IN_SENT_RE.findall(answer))
    invalid_markers = sum(
        1
        for m in _CITATION_IN_SENT_RE.finditer(answer)
        if not _is_marker_valid(m.group(1), len(hits))
    )
    invalid_citation_rate = (invalid_markers / total_markers) if total_markers else 0.0

    per_sentence: List[Dict] = []
    cited_sentences = 0
    support_scores: List[float] = []

    for sent in sentences:
        markers = _CITATION_IN_SENT_RE.findall(sent)
        cited_idxs: List[int] = []
        for m in markers:
            for n in m.split(","):
                n = n.strip()
                if n.isdigit():
                    idx = int(n) - 1
                    if 0 <= idx < len(hits):
                        cited_idxs.append(idx)

        if cited_idxs:
            cited_sentences += 1

        support: Optional[float] = None
        if cited_idxs:
            sub_scores: List[float] = []
            for idx in cited_idxs:
                passage_text = hits[idx].text
                if nli_scorer is not None:
                    sub_scores.append(float(nli_scorer(passage_text, sent)))
                else:
                    sub_scores.append(lexical_support_score(sent, passage_text))
            support = max(sub_scores) if sub_scores else 0.0
            support_scores.append(support)

        per_sentence.append(
            {
                "sentence": sent,
                "cited_passages": cited_idxs,
                "support": support,
            }
        )

    coverage = cited_sentences / len(sentences)
    mean_support = sum(support_scores) / len(support_scores) if support_scores else 0.0

    return FaithfulnessReport(
        answer=answer,
        n_sentences=len(sentences),
        n_cited_sentences=cited_sentences,
        citation_coverage=coverage,
        invalid_citation_rate=invalid_citation_rate,
        mean_support=mean_support,
        per_sentence=per_sentence,
    )


def _is_marker_valid(raw: str, n_hits: int) -> bool:
    nums = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
    return all(1 <= n <= n_hits for n in nums)


# -----------------------------------------------------------------------------
# Optional NLI scorer
# -----------------------------------------------------------------------------

def make_nli_scorer(model_name: str = "cross-encoder/nli-deberta-v3-base"):
    """Return a callable (premise, hypothesis) -> entailment_probability.

    Lazily imports sentence-transformers so the lexical path stays dependency-
    light.
    """
    from sentence_transformers import CrossEncoder
    import numpy as np

    model = CrossEncoder(model_name)
    # Class order for the deberta-v3 NLI checkpoints: [contradiction, entailment, neutral]
    def score(premise: str, hypothesis: str) -> float:
        logits = model.predict([(premise, hypothesis)], show_progress_bar=False)[0]
        # softmax for numerical stability
        e = np.exp(logits - np.max(logits))
        probs = e / e.sum()
        return float(probs[1])  # entailment

    return score
