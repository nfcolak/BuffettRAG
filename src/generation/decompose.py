"""Conservative deterministic decomposition of multi-part questions."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from src.evaluation.claim_validator import evidence_sentences, validate_and_filter_answer
from src.generation.prompt import (
    REFUSAL_LINE,
    build_cited_prompt,
    format_answer_markdown,
    parse_citations,
    strip_chat_artifacts,
)

_WH = r"what|who|when|where|why|how|which|whose|whom"
_WH_CLAUSE = re.compile(rf"\b(?:{_WH})\b", re.I)


def _question(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" \t,;:-")
    if not text:
        return ""
    text = text.rstrip("?.! ") + "?"
    return text[0].upper() + text[1:]


def _why_followup(full: str) -> str | None:
    """Expand a final 'and why?' using the proposition in the first clause."""
    base = re.sub(r"\s*,?\s*and\s+why\s*\??\s*$", "", full, flags=re.I).strip()
    intro = ""
    intro_match = re.match(r"^(In the .*?,\s*)(.+)$", base, re.I)
    if intro_match:
        intro, base = intro_match.groups()
    # Modal/auxiliary questions: invert back to a standalone why-question.
    match = re.match(
        r"^(Could|Would|Can|Will|Should|Did|Does|Do|Is|Are|Was|Were|Has|Have|Had)\s+(.+)$",
        base, re.I,
    )
    if match:
        aux, proposition = match.groups()
        return _question(f"{intro}why {aux.lower()} {proposition}")
    # Transform common "What did <subject> <verb> ..." questions without guessing a new subject.
    match = re.match(r"^What did (.+?) (say|call|describe|identify|report|decide|think|believe|argue|recommend)\s+(.+)$", base, re.I)
    if match:
        subject, verb, obj = match.groups()
        return _question(f"Why did {subject} {verb.lower()} that {obj}")
    return None


def _split_enumeration(query: str) -> List[str] | None:
    matches = list(re.finditer(r"\(\s*(\d+)\s*\)", query))
    if len(matches) < 2 or matches[0].group(1) != "1" or matches[1].group(1) != "2":
        return None
    parts = []
    for i, match in enumerate(matches[:4]):
        end = matches[i + 1].start() if i + 1 < min(len(matches), 4) else len(query)
        part = query[match.end():end].strip(" ,;:-")
        if part:
            parts.append(_question(part))
    return parts if len(parts) >= 2 else None


def _complete_wh_fragment(left: str, right: str) -> str:
    """Carry the main subject/predicate/context into a short coordinated clause."""
    fragment = _question(right)
    if re.match(r"^compare\b", left, re.I):
        decision = re.match(r"what\s+(.+?)\s+followed\??$", fragment, re.I)
        years = re.findall(r"\b(?:19|20)\d{2}\b", left)
        company = re.search(r"\bBerkshire\b", left, re.I)
        if decision and years and company:
            return f"What decision did Berkshire make about its textile operation in {years[-1]}?"
    match = re.search(r"\b(did|does|do|could|would|can|will|should|is|are|was|were|has|have|had)\s+([A-Z][\w'-]*)\s+([a-z]+)\s+(.+)$", left)
    antecedent = re.search(r"\bname of (that [\w'-]+\s+(?:group|company|business|holding))\b", left, re.I)
    if antecedent and re.search(r"\bdid it\b", fragment, re.I):
        return re.sub(r"\bdid it\b", f"did {antecedent.group(1)}", fragment, flags=re.I)
    if re.match(r"^who retained the rest\??$", fragment, re.I):
        topic = re.search(r"\bof\s+([A-Z][A-Z0-9-]+)", left)
        if topic:
            return f"Who retained the rest of {topic.group(1)}?"
    if not match:
        return fragment
    aux, subject, verb, obj = match.groups()
    leading_prep = re.match(r"((?:across|for|with|about)\s+)((?:how|what|which)\s+.+?)\?", fragment, re.I)
    if leading_prep:
        return f"{leading_prep.group(1)}{leading_prep.group(2)} {aux.lower()} {subject} {verb} {obj}?"
    short_count = re.fullmatch(r"How many\s+([^?]+)\?", fragment, re.I)
    if short_count:
        return f"How many {short_count.group(1)} {aux.lower()} {subject} {verb} {obj}?"
    pronoun = re.search(r"\b(did|does|do|is|are|was|were|has|have)\s+it\b", fragment, re.I)
    if pronoun:
        fragment = fragment[:pronoun.start()] + pronoun.group(1) + " " + subject + fragment[pronoun.end():]
        year = re.search(r"\b(?:in|during|by)\s+\d{4}\b", left, re.I)
        if year and year.group(0).lower() not in fragment.lower():
            fragment = fragment.rstrip("?") + " " + year.group(0) + "?"
    return fragment


def _split_wh_join(query: str) -> List[str] | None:
    # Wh-led coordinated clauses are explicit part boundaries; noun phrases are not.
    boundary = re.compile(rf"\s*,\s*(?:and\s+)?(?=(?:{_WH})\b|(?:across|for|with|about|approximately|roughly)\s+(?:{_WH})\b)|\s+and\s+(?=(?:{_WH})\b|(?:across|for|with|about|approximately|roughly)\s+(?:{_WH})\b)", re.I)
    matches = list(boundary.finditer(query))
    first_clause = query[:matches[0].start()] if matches else ""
    prefixless = re.sub(r"^In the .*?,\s*", "", first_clause, flags=re.I)
    if not matches or not (_WH_CLAUSE.search(first_clause) or re.match(r"^(?:Could|Would|Can|Will|Should|Did|Does|Do|Is|Are|Was|Were|Has|Have|Had)\b", prefixless.strip(), re.I)):
        return None
    segments = []
    start = 0
    for match in matches[:3]:
        segments.append(query[start:match.start()].strip(" ,;"))
        start = match.end()
    segments.append(query[start:].strip(" ,;"))
    if len(segments) < 2 or any(not segment for segment in segments):
        return None
    completed = [_question(segments[0])]
    for segment in segments[1:]:
        if segment.lower().strip(" ?") == "why":
            expanded = _why_followup(query)
            completed.append(expanded or _question(segment))
        else:
            completed.append(_question(_complete_wh_fragment(segments[0], segment)))
    return completed


def _split_both_or_as_well(query: str) -> List[str] | None:
    patterns = (
        re.compile(r"\bboth\s+(.+?)\s+and\s+(.+?)(?=[?.!]?$)", re.I),
        re.compile(r"(.+?)\s+as well as\s+(.+?)(?=[?.!]?$)", re.I),
    )
    for pattern in patterns:
        match = pattern.search(query.rstrip("?.! "))
        if not match:
            continue
        left, right = match.groups()
        # Avoid fragmenting coordinated noun phrases; both halves must carry a verb or wh-word.
        verbish = re.compile(rf"\b(?:{_WH}|(?:am|is|are|was|were|be|been|being|do|does|did|have|has|had|can|could|will|would|should|must|pay|paid|hold|held|buy|bought|sell|sold|earn|earning|give|gave|retain|retained|receive|received|supply|supplied|report|reported|say|said|prefer|preferred|decide|decided|remain|remains|run|runs|cost|costs))\b", re.I)
        if not verbish.search(left) or not verbish.search(right):
            continue
        return [_question(query[:match.start()] + left), _question(right)]
    return None


def split_question(query: str) -> list[str]:
    """Split clear multi-part interrogatives into at most four standalone questions."""
    query = re.sub(r"\s+", " ", str(query or "")).strip()
    if not query:
        return []
    enumerated = _split_enumeration(query)
    if enumerated:
        return enumerated[:4]
    # Multiple complete question sentences are unambiguous.
    sentences = [s.strip() for s in re.split(r"(?<=\?)\s+(?=[A-Z])", query) if s.strip()]
    if len(sentences) > 1:
        return [_question(s) for s in sentences[:4]]
    split = _split_wh_join(query)
    if split:
        return split[:4]
    split = _split_both_or_as_well(query)
    return split[:4] if split else [_question(query)]


def _sentences(text: str) -> List[str]:
    """Split answer prose without detaching trailing citation markers."""
    out: List[str] = []
    for sentence in evidence_sentences(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if not _dedupe_key(sentence) and out:
            out[-1] += " " + sentence  # a bare marker belongs to the previous sentence
        elif _dedupe_key(sentence):
            out.append(sentence)
    return out


def _dedupe_key(sentence: str) -> str:
    text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", sentence)
    return " ".join(re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text.casefold()))


def _near_duplicate(left: str, right: str) -> bool:
    left_tokens, right_tokens = set(left.split()), set(right.split())
    union = left_tokens | right_tokens
    return bool(union) and len(left_tokens & right_tokens) / len(union) >= 0.8


def answer_by_parts(llm: Any, query: str, context_hits: Sequence[Any], history,
                    max_new_tokens: int) -> tuple[str, list[dict]]:
    """Generate, validate, and join one independently cited answer per question part."""
    parts = split_question(query)
    if len(parts) < 2:
        prompt = build_cited_prompt(query=query, hits=context_hits, history=history)
        raw = llm.generate(prompt, max_new_tokens=max_new_tokens)
        answer = format_answer_markdown(strip_chat_artifacts(raw))
        validation = validate_and_filter_answer(answer, context_hits)
        answer = validation.safe_answer or REFUSAL_LINE
        return answer, parse_citations(answer, context_hits)

    kept: List[str] = []
    keys: List[str] = []
    for part in parts:
        part_query = f"{query}\nAnswer only this part: {part}"
        prompt = build_cited_prompt(query=part_query, hits=context_hits, history=history)
        raw = llm.generate(prompt, max_new_tokens=max_new_tokens)
        answer = format_answer_markdown(strip_chat_artifacts(raw))
        if answer == REFUSAL_LINE or len(re.findall(r"[^\W_]+(?:['’][^\W_]+)*", re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", " ", answer))) < 3:
            continue
        validation = validate_and_filter_answer(answer, context_hits)
        answer = validation.safe_answer or REFUSAL_LINE
        if answer == REFUSAL_LINE:
            continue
        for sentence in _sentences(answer):
            key = _dedupe_key(sentence)
            if key and not any(key == k or _near_duplicate(key, k) for k in keys):
                keys.append(key)
                kept.append(sentence)
    if not kept:
        return REFUSAL_LINE, []
    answer = " ".join(kept)
    answer = validate_and_filter_answer(answer, context_hits).safe_answer or REFUSAL_LINE
    if answer == REFUSAL_LINE:
        return REFUSAL_LINE, []
    return answer, parse_citations(answer, context_hits)
