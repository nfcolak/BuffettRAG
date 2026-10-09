"""R3 strict supported-claim metric shared by the benchmark and calibration.

A gold claim is met only if one answer sentence (a) lexically matches the claim
(required terms, polarity, numbers) and (b) passes the deterministic support
check, including the R1 numeric guard, against a cited hit that is one of the
claim's gold passages, using that hit's saved text. Engine year labels are
stripped only when the cited hit's metadata year supports them; typed coverage
notes carry no citation and so never count.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from src.evaluation.answer_benchmark import _CITATION_RE, _polarity_and_numbers_agree, is_unanswerable_case
from src.evaluation.claim_validator import evidence_sentences

# Exact engine labels (engine.py _label: "In the {y} letter:" and "In the {y1}-{y2} letters:"); colon required,
# so "In 2002, ..." is never stripped. A label is stripped only when a cited hit's year lies in its range.
_LABEL_RE = re.compile(r"^\s*In the ((?:19|20)\d{2})(?: letter:|-((?:19|20)\d{2}) letters:)\s+")
# Comparison paragraph label (compare.py period_label): "In {y}:", "In {y}s:" (decade) or "In {y1}\u2013{y2}:".
_PERIOD_LABEL_RE = re.compile(r"^\s*In ((?:19|20)\d{2})(s?)(?:\u2013((?:19|20)\d{2}))?:\s+")
_MARKERS_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")


def _answer_sentences(text: str) -> List[str]:
    """One sentence per entry, each keeping its trailing [n] marker.

    Uses the verifier's own splitter (evidence_sentences), which does not break after titles
    ("Mr. Market") or initials ("U.S. Treasury Bills"), so the sentence scored here is the
    sentence the verifier sees. Lines (comparison paragraphs, bullets) stay separate.
    """
    return [s.strip() for line in (text or "").splitlines()
            for s in evidence_sentences(_BULLET_RE.sub("", line)) if s.strip()]


def _hit_get(hit: Any, name: str, default: Any = None) -> Any:
    if isinstance(hit, dict):
        return hit.get(name, default)
    return getattr(hit, name, default)


def _hit_year(hit: Any) -> Optional[int]:
    year = _hit_get(hit, "year")
    if year is None:
        year = (_hit_get(hit, "metadata") or {}).get("year")
    try:
        return int(year) if year is not None else None
    except (TypeError, ValueError):
        return None


def _strip_label(sentence: str, cited_hits: Sequence[Any]) -> str:
    match = _LABEL_RE.match(sentence)
    if match:
        low = int(match.group(1))
        high = int(match.group(2) or low)
        if any(y is not None and low <= y <= high for y in (_hit_year(h) for h in cited_hits)):
            return sentence[match.end():]
    return sentence


def _strip_period_label(sentence: str, cited_hits: Sequence[Any]) -> str:
    """Drop the comparison paragraph's "In 1985: " prefix when a cited hit lies in that period.

    The prefix is added by the comparison answer around the model's text; it is not part of the claim.
    """
    match = _PERIOD_LABEL_RE.match(sentence)
    if match:
        low = int(match.group(1))
        high = int(match.group(3) or (low + 9 if match.group(2) else low))
        if any(y is not None and low <= y <= high for y in (_hit_year(h) for h in cited_hits)):
            return sentence[match.end():]
    return sentence


def _default_verifier() -> Any:
    from src.generation.grounded.verify import DeterministicVerifier  # lazy: owned by u2a
    return DeterministicVerifier()


def _group_evidence(hit: Any, hit_index: int) -> List[Any]:
    from src.generation.grounded.types import EvidenceUnit, GroupEvidence, SourceSpan

    text = " ".join(str(_hit_get(hit, "text", "")).split())
    entries = []
    for local, sentence in enumerate(s for s in evidence_sentences(text) if s.strip()):
        span = SourceSpan(hit_index, 0, len(sentence), sentence)
        unit = EvidenceUnit(eid=f"h{hit_index}s{local}", hit_index=hit_index,
                            passage_id=str(_hit_get(hit, "id")), letter_year=_hit_year(hit),
                            anchor=span, window=span, score=1.0)
        entries.append(GroupEvidence(local_index=local, text=sentence, unit=unit, source_span=span))
    return entries


def _supported_by_hit(verifier: Any, sentence: str, hit: Any, hit_index: int) -> bool:
    """Deterministic support: some source sentence of the hit verifies the answer sentence."""
    bare = _MARKERS_RE.sub("", sentence).strip()
    entries = _group_evidence(hit, hit_index)
    for entry in entries:
        marked = f"{bare} [{entry.local_index + 1}]"
        if verifier.verify(marked, entries):
            return True
    return False


def supported_claims_met(case: Dict[str, Any], answer_text: str, citations: Sequence[Any],
                         context_hits: Sequence[Any], *, verifier: Any = None) -> List[Dict[str, Any]]:
    if is_unanswerable_case(case):
        return []
    verifier = verifier or _default_verifier()
    allowed_ids = {str(c.get("id") or c.get("passage_id")) for c in citations
                   if isinstance(c, dict) and (c.get("id") or c.get("passage_id"))}
    sentences = _answer_sentences(answer_text)
    out = []
    for claim in case["gold_claims"]:
        gold = set(claim.get("gold_passage_ids") or case["gold_passage_ids"])
        terms = [t.lower() for t in claim["required_terms"]]
        supporting = []
        for sentence in sentences:
            indexes = sorted({int(n) - 1 for raw in _CITATION_RE.findall(sentence)
                              for n in raw.split(",") if n.strip().isdigit()})
            cited = [(i, context_hits[i]) for i in indexes if 0 <= i < len(context_hits)]
            cited_gold = [(i, h) for i, h in cited if _hit_get(h, "id") in gold
                          and (not allowed_ids or _hit_get(h, "id") in allowed_ids)]
            if not cited_gold:
                continue
            body = _strip_period_label(_strip_label(sentence, [h for _, h in cited_gold]),
                                       [h for _, h in cited_gold])
            if not (all(t in body.lower() for t in terms)
                    and _polarity_and_numbers_agree(claim["claim"], body)):
                continue
            if any(_supported_by_hit(verifier, body, h, i) for i, h in cited_gold):
                supporting.append(sentence)
        out.append({"claim": claim["claim"], "met": bool(supporting), "supporting_sentences": supporting})
    return out
