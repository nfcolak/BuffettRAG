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
from functools import lru_cache
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


def _label_year(sentence: str, cited_hits: Sequence[Any]) -> Optional[str]:
    """YYYY of a single-year "In YYYY:" / "In the YYYY letter:" label that was actually stripped, else None.

    Ranges and decades ("1990s") return None: they name no single year.
    """
    if _strip_period_label(_strip_label(sentence, cited_hits), cited_hits) == sentence:
        return None
    match = _LABEL_RE.match(sentence)
    if match and not match.group(2):
        return match.group(1)
    match = _PERIOD_LABEL_RE.match(sentence)
    if match and not match.group(2) and not match.group(3):
        return match.group(1)
    return None


# "Full Name (\"ABBR\")" / "Full Name (ABBR)": a run of capitalised words (lowercase connectors allowed inside)
# directly before a parenthesised upper-case abbreviation, optionally in straight or curly quotes.
_ALIAS_RE = re.compile(
    r"((?:[A-Z][\w&'.\-]*)(?:\s+(?:(?:of|and|the|for|&)\s+)*[A-Z][\w&'.\-]*)*)"
    r"\s*\(\s*[\"\u201c\u2018']?([A-Z][A-Z0-9&.]{1,9})[\"\u201d\u2019']?\s*\)"
)
_LEADING_WORDS = frozenset({"The", "In", "A", "An", "Its", "Our", "Both", "At", "On", "Of", "For", "And", "But"})


def _definitions(sentence: str) -> Dict[str, str]:
    out = {}
    for full, abbr in _ALIAS_RE.findall(_MARKERS_RE.sub("", sentence)):
        words = full.split()
        while len(words) > 1 and words[0] in _LEADING_WORDS:
            words.pop(0)
        out[abbr] = " ".join(words)
    return out


def _expand_aliases(body: str, aliases: Dict[str, str]) -> str:
    for abbr, full in aliases.items():
        body = re.sub(r"(?<![A-Za-z0-9])" + re.escape(abbr) + r"(?![A-Za-z0-9])", full, body)
    return body


def _default_verifier() -> Any:
    from src.generation.grounded.verify import DeterministicVerifier  # lazy: owned by u2a
    return DeterministicVerifier()


@lru_cache(maxsize=1)
def _corpus_texts() -> Dict[str, str]:
    """Own text of every chunk in the active corpus (empty when the corpus file is absent)."""
    import json

    from config import CHUNKS_V3_FILE

    out: Dict[str, str] = {}
    try:
        with open(CHUNKS_V3_FILE, "r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                out[str(row.get("id"))] = row.get("text") or ""
    except OSError:
        return {}
    return out


def _chunk_text(chunk_texts: Any, chunk_id: str) -> Optional[str]:
    if chunk_texts is None:
        return _corpus_texts().get(chunk_id)
    if callable(chunk_texts):
        return chunk_texts(chunk_id)
    return chunk_texts.get(chunk_id)


def _merged_ids(hit: Any) -> List[str]:
    ids = _hit_get(hit, "merged_ids") or (_hit_get(hit, "metadata") or {}).get("merged_ids")
    if not ids:
        return [str(_hit_get(hit, "id"))]
    return [str(i) for i in ids]


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
                         context_hits: Sequence[Any], *, verifier: Any = None,
                         chunk_texts: Any = None) -> List[Dict[str, Any]]:
    """`chunk_texts` (mapping id -> text, or callable) supplies each chunk's own text for merged neighbours;
    by default it is read from the active corpus file. A neighbour with no own text never counts."""
    if is_unanswerable_case(case):
        return []
    verifier = verifier or _default_verifier()
    allowed_ids = {str(c.get("id") or c.get("passage_id")) for c in citations
                   if isinstance(c, dict) and (c.get("id") or c.get("passage_id"))}
    sentences = _answer_sentences(answer_text)
    # Abbreviations defined by earlier sentences of this answer (rule d: required-terms check only).
    aliases_before: List[Dict[str, str]] = []
    defined: Dict[str, str] = {}
    for sentence in sentences:
        aliases_before.append(dict(defined))
        defined.update(_definitions(sentence))
    out = []
    for claim in case["gold_claims"]:
        gold = set(claim.get("gold_passage_ids") or case["gold_passage_ids"])
        terms = [t.lower() for t in claim["required_terms"]]
        supporting = []
        for position, sentence in enumerate(sentences):
            indexes = sorted({int(n) - 1 for raw in _CITATION_RE.findall(sentence)
                              for n in raw.split(",") if n.strip().isdigit()})
            cited = [(i, context_hits[i]) for i in indexes if 0 <= i < len(context_hits)]
            # (index, hit, gold ids of the hit's merged neighbours that are not the hit itself)
            cited_gold = []
            for i, h in cited:
                hit_id = _hit_get(h, "id")
                if allowed_ids and hit_id not in allowed_ids:
                    continue
                if hit_id in gold:
                    cited_gold.append((i, h, []))
                    continue
                neighbours = [g for g in _merged_ids(h) if g in gold and g != hit_id]
                if neighbours:
                    cited_gold.append((i, h, neighbours))
            if not cited_gold:
                continue
            gold_hits = [h for _, h, _ in cited_gold]
            body = _strip_period_label(_strip_label(sentence, gold_hits), gold_hits)
            label_year = _label_year(sentence, gold_hits)
            year_hit = label_year is not None and any(_hit_year(h) == int(label_year) for h in gold_hits)
            extra_numbers = [label_year] if (label_year and year_hit) else []
            aliased = _expand_aliases(body, aliases_before[position])
            # A term matches the sentence as written or with its abbreviations expanded (a term may be the abbreviation).
            if not (all(t in aliased.lower() or t in body.lower() for t in terms)
                    and _polarity_and_numbers_agree(claim["claim"], body, terms, extra_numbers)):
                continue
            if any(_cited_hit_supports(verifier, body, h, i, neighbours, chunk_texts)
                   for i, h, neighbours in cited_gold):
                supporting.append(sentence)
        out.append({"claim": claim["claim"], "met": bool(supporting), "supporting_sentences": supporting})
    return out


def _cited_hit_supports(verifier: Any, body: str, hit: Any, hit_index: int, neighbours: Sequence[str],
                        chunk_texts: Any) -> bool:
    if not neighbours:
        return _supported_by_hit(verifier, body, hit, hit_index)
    # A merged neighbour counts only when the sentence is found in that gold chunk's OWN text.
    for gold_id in neighbours:
        own = _chunk_text(chunk_texts, gold_id)
        if own and _supported_by_hit(verifier, body, {"id": gold_id, "text": own, "year": _hit_year(hit)},
                                     hit_index):
            return True
    return False
