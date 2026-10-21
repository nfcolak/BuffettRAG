"""Finance answer routes (finance-query-pipeline, group 5).

Callers gate everything here behind `FINANCE_PIPELINE=1`; this module does not read the flag.

Final answers, returned as they are (no evidence gate, no claim validator, no `expand_answer`):
  - `scope_answer()`: the fixed scope reply for personal plans, no hits;
  - `numeric_answer(plan, store, docs_by_id)`: rows of `store.numbers` that match the question, each with its value,
    year and the `as_written` text, citing the source passage that is served as a hit; None when nothing matches
    (the caller falls back to the reader). The caller uses it only for a plain figure lookup
    (`plain_figure_lookup`): a metric is named, the entity matches, no why/how/explain clause, no second question;
  - `digest_answer(plan, store, docs_by_id, reader_hits)`: the overview of the matching `store.digest` topic, then dated
    verbatim quotes. The hits are the timeline `source_id` passages ordered by year and numbered in that order, then
    reader passages for gaps (letters the timeline does not cover). Every [n] in the answer points at a returned hit;
  - `superlative_answer(plan, store, docs_by_id)`: "which sector did he speak most positively about". Targets are
    ranked by the number of DISTINCT LETTERS with a stance of that polarity in the annotations (letters, not stance
    rows, so a densely annotated letter does not dominate); the answer names the top three with their letter count
    and year span, then quotes the winner's digest timeline (entries on its stance passages; letters the timeline
    does not cover get the annotation's verbatim stance quote), citing the quoted source passages served as hits.
    Uncited evidence fills the rest of the served budget: with FINANCE_DIGEST_SOURCE the winner's digest timeline
    passages and with FINANCE_STANCE_SOURCE the stance passages of the top targets (finance-recall-75), each spread
    across letters; then, as without them, the winner's other stance passages, one per letter. The budget is the
    caller's `max_hits`, raised to FINANCE_K_CORPUS when that is larger (finance-recall-75-r2, default 0 = unchanged);
    the passages beyond `max_hits` must fit the reader window (`reader_window_chars`), lowest ranks cut first.
    With FINANCE_SUPERLATIVE_TAGGED=N (finance-recall-75-r3) up to N passages tagged with the winner and carrying a
    stance of the asked polarity (`tagged_stance_ids`) come first among the uncited evidence, taking their places from
    the same budget.
    With FINANCE_SUPERLATIVE_RERANK=N (finance-recall-75-r4) the cross-encoder scores up to 60 of those tagged passages
    against the question and the best N, spread across letters, come first of all (`reranked_stance_ids`); without a
    reranker the setting is skipped and the answer says so in its detail.

Matching figure rows never use another entity's rows: when the question names a company other than Berkshire, only
that company's rows match; Berkshire's rows are the default only when no other company is named.

Reader route helpers (the answer still goes through `ask_flow._finalize_answer`):
  - `sort_by_year(hits)` and `trend_prompt(query, hits, history)`: trend answers keep the system prompt v1 (FT-r1 was
    trained on it), serve the passages sorted by year (the [n] markers follow the served list) and append one sentence
    to the user block that asks for a year-by-year answer;
  - `reader_prompt(query, hits, history, trend)`: every finance reader prompt. The system prompt, the passage and
    question blocks and the "Answer:" cue stay exactly as FT-r1 was trained on them; `READER_INSTRUCTIONS` is appended
    to the user block (after the trend sentence for a trend). It asks for every relevant passage, the reasons and the
    figures, several cited sentences, a gap sentence for a part the passages do not cover, and the refusal line only
    when no passage answers any part. `is_reader_prompt(prompt)` recognises such a prompt by its tail, so
    `ask_flow._finalize_answer` applies the reader settings below on /ask and /ask/stream alike;
  - reader post-processing: `format_reader_answer` (list formatting that keeps "600,000" whole), `READER_VALIDATION`
    (claim validator options: citation repair, gap sentences via `is_gap_line`, contentless sentences dropped) and
    `READER_EXPANSION` (up to two passage sentences after every verbatim answer sentence, within the existing
    ANSWER_MAX_SENTENCES / ANSWER_EXPANDED_MAX_SENTENCES caps; `expansion_hits` keeps passages with personal facts
    out of it);
  - `figure_rows(plan, store, docs_by_id)` and `figures_prompt(prompt, rows, hits)`: a question with numeric intent
    that is not a plain lookup goes to the reader; its matching figure rows' source passages are served first and
    the rows are added to the user block as extra context, each citing its served passage.

Reader v3 (FINANCE_ANSWER_V3, finance-answer-quality group 3; the caller reads the flag, everything below follows the
prompt, so /ask and /ask/stream behave alike):
  - `reader_prompt_v3(query, hits, history, premise_note)`: the evidence pack by year. The caller serves the passages
    in year order (`sort_by_year`); the user block lists them under letter-year headers (`year_passage_block`) with
    the same `[n] (year=..., source=...)` lines, then the premise correction when the plan has a `premise_note`, then
    V3_INSTRUCTIONS. The system prompt, the question block and the "Answer:" cue stay as for the other prompts;
    `is_reader_prompt` accepts the v3 prompt and its retry prompt, so the reader settings apply;
  - post-processing (`ask_flow._finalize_answer`): `strip_premise_line` takes out the model's own uncited premise
    sentence and `with_premise` puts the planner's note first instead; `dedupe_sentences` drops repeated sentences
    (normalised text, the first kept) before the validator and again after expansion;
  - `partial_retry_prompt`: one retry for the covered part (RETRY_INSTRUCTIONS) when the model refused while the prompt
    carries a premise note or a served hit carries the plan's topic tags (`mark_topic_hits`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from config import finance_setting, retrieval_setting
from src.generation.decompose import split_question
from src.generation.prompt import (REFUSAL_LINE, _passage_block, build_cited_prompt, format_answer_markdown,
                                   parse_citations)
from src.knowledge import KnowledgeStore
from src.query.plan import QueryPlan
from src.retrieval.finance_policy import reader_window_chars
from src.retrieval.knowledge_source import letter_of, spread_letters
from src.storage import SearchHit

SCOPE_REPLY = "I only answer questions about Buffett's financial views and Berkshire's financial results."
TREND_SENTENCE = "Answer year by year, in the order of the passages, and say which year each point comes from."
READER_INSTRUCTIONS = (
    "Use every passage that answers part of the question, not only the first one. "
    "Give the reasons and the figures the passages state. "
    "Write several sentences, one point each, copying the passage sentence when you can, and end every sentence "
    "with the citation of the passage it comes from. "
    "If the passages answer only part of the question, answer that part, then say in one sentence which part they "
    "do not cover. "
    "Write the refusal line only when no passage answers any part of the question; a passage that only shares a "
    "topic word with the question does not answer it."
)
# FINANCE_ANSWER_V3: the user-block instructions of the year-grouped reader prompt, the premise correction around a
# plan's `premise_note`, and the shorter instructions of the one partial retry.
V3_INSTRUCTIONS = (
    "The passages are grouped by letter year, oldest first. "
    "For each letter year or period that answers part of the question, write one or two sentences, copying the "
    "passage sentence when you can, and end every sentence with the citation of the passage it comes from. "
    "Prefer Buffett's views, actions, figures and lessons over narration of events. "
    "Never repeat a sentence. "
    "If the passages answer only part of the question, answer that part, then say in one sentence which part they "
    "do not cover. "
    "Write the refusal line only when no passage answers any part of the question; a passage that only shares a "
    "topic word with the question does not answer it."
)
PREMISE_OPEN = "Correction to the question: "
PREMISE_CLOSE = (" Start the answer with one sentence that makes this correction, then give what the passages say on "
                 "the topic, whatever their letter year.")
RETRY_INSTRUCTIONS = (
    "Some passages answer part of the question. Answer that part in a few sentences, each ending with the citation "
    "of its passage, then say in one sentence which part they do not cover. Do not write the refusal line."
)
TOPIC_FLAG = "plan_topic"     # hit metadata: the served passage carries the plan's topic tags (`mark_topic_hits`)
DEDUPE_MIN_WORDS = 3          # a shorter sentence is never dropped as a repeat
PREMISE_LINE_MAX_WORDS = 60
# Passage sentences added after / before every verbatim reader-answer sentence (staging_v3 replay of the raw answers:
# claim coverage 6.2% -> 14.3% with 2/0, 13.8% with 1/1, 17.5% with 2/1; sentences before an anchor often open
# mid-argument, so none are added before).
READER_NEIGHBOR_AFTER = 2
READER_NEIGHBOR_BEFORE = 0
GAP_LINE_MAX_WORDS = 40
MAX_NUMERIC_ROWS = 6
DIGEST_GAP_PASSAGES = 4  # reader passages appended after the digest's quoted passages, at most
SUPERLATIVE_TOP = 3      # targets named in a superlative answer
SUPERLATIVE_QUOTES = 6   # quotes from the winner's digest timeline, at most
SUPERLATIVE_RERANK_POOL = 60  # tagged passages of the winner the cross-encoder scores (FINANCE_SUPERLATIVE_RERANK)
BERKSHIRE = {"berkshire", "berkshire hathaway"}
FIGURES_BEGIN = "BEGIN FIGURES FROM THE PASSAGES (each row cites the passage it comes from)"
FIGURES_END = "END FIGURES"
_QUESTION_BLOCK = "BEGIN USER QUESTION\n"
# why / explain / describe, or a "how" that is not "how much / how many"
_EXPLAIN_CLAUSE = re.compile(
    r"\b(?:why|explain\w*|describe\w*|reasons?|caus(?:e|ed|es)|what (?:made|led|drove))\b|"
    r"\bhow\b(?!\s+(?:much|many|high|large|big|low|small)\b)", re.I)
_SECOND_ASK = re.compile(r"(?:[,;]|\band\b)\s*(?:and\s+)?(?:how|what|why|which|when|where|who)\b", re.I)

_ANSWER_SUFFIX = "\n\nAnswer:"
_READER_TAIL = f" {READER_INSTRUCTIONS}{_ANSWER_SUFFIX}"
_V3_TAIL = f" {V3_INSTRUCTIONS}{_ANSWER_SUFFIX}"
_RETRY_TAIL = f" {RETRY_INSTRUCTIONS}{_ANSWER_SUFFIX}"
_PASSAGES_BEGIN = "BEGIN UNTRUSTED PASSAGES\n"
_QUESTION_END = "END USER QUESTION"
# a gap sentence: "The (retrieved) passages / letters ... do not ...", or "There is no information ..."
_GAP_START = re.compile(
    r"^(?:(?:however|but),?\s+)?(?:(?:the|these)\s+(?:[a-z-]+\s+){0,3}?(?:passages?|letters?|excerpts?)\b"
    r"|there\s+is\s+no\s+(?:information|evidence|mention)\b)", re.I)
_GAP_NEGATION = re.compile(r"\b(?:do|does|did)\s+not\b|\b(?:don't|doesn't|didn't)\b|\bno\s+(?:information|evidence|mention)\b",
                           re.I)
_GAP_CLAIM = re.compile(r"\b(?:but|although|though|while|whereas|instead)\b|;", re.I)  # a second clause may state a fact
_GAP_FIGURE = re.compile(r"[$£€]\s*\d|\d\s*(?:%|percent\b)", re.I)
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_COMMA_MASK = "\ue000"  # a private-use character, never in model output
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_METRIC_STOP = {"in", "of", "the", "a", "an", "and", "to", "usd", "pct", "per"}
_UNIT_FORMATS = {
    "usd_millions": lambda v: f"${_num(v)} million",
    "usd_billions": lambda v: f"${_num(v)} billion",
    "usd": lambda v: f"${_num(v)}",
    "percent": lambda v: f"{_num(v)}%",
}


@dataclass
class FinanceAnswer:
    """A finished answer: the text, the hits its [n] markers point at, and the parsed citations."""
    answer: str
    hits: List[SearchHit] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    route: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


def _num(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value:,}" if isinstance(value, (int, float)) else str(value)


def _tokens(text: str) -> List[str]:
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in _TOKEN_RE.findall(str(text).lower())]


def _year(hit: SearchHit) -> int:
    value = hit.metadata.get("year")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 9999


def _hit_from_doc(doc: Any, fallback_year: Optional[int] = None) -> SearchHit:
    metadata = dict(getattr(doc, "metadata", None) or {})
    if fallback_year is not None and not metadata.get("year"):
        metadata["year"] = fallback_year
    return SearchHit(id=doc.id, text=doc.text, metadata=metadata, score=1.0)


def sort_by_year(hits: Sequence[SearchHit]) -> List[SearchHit]:
    """Served order for trend answers: oldest letter first, ties keep their rank."""
    return sorted(hits, key=_year)


def trend_prompt(query: str, hits: Sequence[SearchHit], history: Optional[Sequence[Dict[str, str]]] = None) -> str:
    """The v1-compatible cited prompt plus one sentence in the user block asking for a year-by-year answer."""
    prompt = build_cited_prompt(query=query, hits=hits, history=history)
    if not prompt.endswith(_ANSWER_SUFFIX):  # defensive: never rewrite a prompt whose tail we do not know
        return prompt
    return f"{prompt[:-len(_ANSWER_SUFFIX)]} {TREND_SENTENCE}{_ANSWER_SUFFIX}"


# --- reader ---
def reader_prompt(query: str, hits: Sequence[SearchHit], history: Optional[Sequence[Dict[str, str]]] = None,
                  trend: bool = False) -> str:
    """The finance reader prompt: the cited prompt (or the trend prompt) with READER_INSTRUCTIONS at the end of the
    user block, before the "Answer:" cue. The system prompt and the blocks the model was trained on are unchanged."""
    prompt = trend_prompt(query, hits, history) if trend else build_cited_prompt(query=query, hits=hits, history=history)
    if not prompt.endswith(_ANSWER_SUFFIX):  # defensive, as in trend_prompt
        return prompt
    return prompt[:-len(_ANSWER_SUFFIX)] + _READER_TAIL


def is_reader_prompt(prompt: str) -> bool:
    """True for a prompt built by `reader_prompt` or `reader_prompt_v3`, or a v3 retry prompt (the question text cannot
    reach the tail)."""
    return bool(prompt) and prompt.endswith((_READER_TAIL, _V3_TAIL, _RETRY_TAIL))


def is_gap_line(text: str) -> bool:
    """A sentence that only says which part of the question the passages do not cover: it opens with the passages
    (or "There is no information"), negates, has no second clause and no money or percent figure, and is not the
    refusal line itself (beside an answer that line would read as a refusal)."""
    s = " ".join(str(text).replace("’", "'").split())
    if not s or s.rstrip(".") == REFUSAL_LINE.rstrip(".") or len(s.split()) > GAP_LINE_MAX_WORDS:
        return False
    negation = _GAP_NEGATION.search(s)
    return bool(_GAP_START.match(s) and negation and not _GAP_CLAIM.search(s, negation.end())
                and not _GAP_FIGURE.search(s))


def expansion_hits(hits: Sequence[SearchHit], store: KnowledgeStore) -> List[SearchHit]:
    """The served hits as neighbour expansion sees them: a passage with personal facts in any of its merged chunks
    (annotations) is blanked, so expansion never adds a sentence nobody asked for from a personal story. The [n]
    positions stay; citations are still parsed against the served hits."""
    return [SearchHit(h.id, "", dict(h.metadata), h.score)
            if any(store.has_personal(pid) for pid in (h.metadata.get("merged_ids") or [h.id])) else h
            for h in hits]


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])((?:\s*\[\d+(?:\s*,\s*\d+)*\])*)\s+(?=[^\s\[])")
_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Personal cues in an answer sentence (narrower than the planner's question rules: no generic words like "sports").
_PERSONAL_SENTENCE = re.compile(
    r"\b(?:child|children|kids?|wife|husband|son|sons|daughters?|grandchild(?:ren)?|grand(?:father|mother)|"
    r"mother|father|brother|sister|parents|marri(?:ed|age)|spouse|birthday|born|\d{1,3} years old|age \d{1,3}|"
    r"health|surgery|illness|diet|jok(?:e|es|ed|ing)|ping[- ]pong|first pitch|funeral|died|death)\b", re.I)


def drop_personal_sentences(answer: str, hits: Sequence[SearchHit], store: KnowledgeStore) -> str:
    """Finance reader answers keep personal content out. A sentence is removed when it carries a personal cue and
    every passage it cites is annotated `has_personal` or non-financial (a founder's family story told next to the
    company's figures, for example). Returns the refusal line when nothing is left."""
    if not answer or answer == REFUSAL_LINE:
        return answer

    def personal_source(n: int) -> bool:
        if not 1 <= n <= len(hits):
            return False
        ids = hits[n - 1].metadata.get("merged_ids") or [hits[n - 1].id]
        return any(store.has_personal(pid) or not store.is_financial(pid) for pid in ids)

    def personal(sentence: str) -> bool:
        if not _PERSONAL_SENTENCE.search(sentence):
            return False
        cited = [int(x) for m in _CITE_RE.findall(sentence) for x in m.split(",")]
        return bool(cited) and all(personal_source(n) for n in cited)

    out_lines = []
    for line in answer.split("\n"):
        parts = _SENTENCE_END_RE.split(line)
        sentences = [parts[i] + (parts[i + 1] if i + 1 < len(parts) else "") for i in range(0, len(parts), 2)]
        kept = [x for x in sentences if x.strip() and not personal(x)]
        if kept or not line.strip():
            out_lines.append(" ".join(x.strip() for x in kept))
    text = "\n".join(out_lines).strip()
    return text or REFUSAL_LINE


def format_reader_answer(text: str) -> str:
    """`format_answer_markdown` that keeps a number with thousands separators whole: its inline-list split turns
    "... is about 600,000. [2]" into "... is about 600" and a list item "000. [2]"."""
    return format_answer_markdown(_THOUSANDS_RE.sub(_COMMA_MASK, text)).replace(_COMMA_MASK, ",")


# claim validator options and neighbour expansion of the finance reader route (ask_flow._finalize_answer)
READER_VALIDATION: Dict[str, Any] = {"repair_citations": True, "gap_line": is_gap_line, "drop_contentless": True}
READER_EXPANSION: Dict[str, int] = {"after": READER_NEIGHBOR_AFTER, "before": READER_NEIGHBOR_BEFORE}


# --- reader v3 (FINANCE_ANSWER_V3) ---
def year_passage_block(hits: Sequence[SearchHit]) -> str:
    """The numbered passages exactly as `build_cited_prompt` writes them ([n] in served order), with a letter-year
    header before each run of passages from the same letter."""
    out: List[str] = []
    start = 0
    while start < len(hits):
        year = hits[start].metadata.get("year", "?")
        end = start + 1
        while end < len(hits) and str(hits[end].metadata.get("year", "?")) == str(year):
            end += 1
        span = f"passage [{start + 1}]" if end == start + 1 else f"passages [{start + 1}]-[{end}]"
        label = f"LETTER OF {year}" if str(year).strip() not in ("", "?", "None") else "LETTER YEAR UNKNOWN"
        out.append(f"{label} ({span})")
        out += [f"[{n}]{_passage_block([hit])[len('[1]'):]}" for n, hit in enumerate(hits[start:end], start + 1)]
        start = end
    return "\n\n".join(out)


def reader_prompt_v3(query: str, hits: Sequence[SearchHit], history: Optional[Sequence[Dict[str, str]]] = None,
                     premise_note: Optional[str] = None) -> str:
    """The v3 finance reader prompt: the cited prompt with its passages grouped under letter-year headers (the caller
    serves them in year order) and, at the end of the user block, the premise correction (when the plan has a
    `premise_note`) and V3_INSTRUCTIONS. The system prompt, the [n] passage lines, the question block and the
    "Answer:" cue are unchanged."""
    prompt = build_cited_prompt(query=query, hits=hits, history=history)
    block = f"{_PASSAGES_BEGIN}{_passage_block(hits)}\n\n"
    if not prompt.endswith(_ANSWER_SUFFIX) or block not in prompt:  # defensive, as in trend_prompt
        return prompt
    prompt = prompt.replace(block, f"{_PASSAGES_BEGIN}{year_passage_block(hits)}\n\n", 1)
    note = " ".join(str(premise_note or "").split())
    premise = f" {PREMISE_OPEN}{note}{PREMISE_CLOSE}" if note else ""
    return prompt[:-len(_ANSWER_SUFFIX)] + premise + _V3_TAIL


def is_v3_prompt(prompt: str, with_retry: bool = False) -> bool:
    """True for a prompt built by `reader_prompt_v3`; with `with_retry` also for its retry prompt."""
    return bool(prompt) and prompt.endswith((_V3_TAIL, _RETRY_TAIL) if with_retry else _V3_TAIL)


def prompt_premise(prompt: str) -> Optional[str]:
    """The premise note a v3 prompt or its retry prompt carries after the question block, else None."""
    for tail in (_V3_TAIL, _RETRY_TAIL):
        if prompt and prompt.endswith(tail):
            body = prompt[:-len(tail)]
            start = body.rfind(f" {PREMISE_OPEN}")
            if body.endswith(PREMISE_CLOSE) and start > body.rfind(_QUESTION_END):
                return body[start + 1 + len(PREMISE_OPEN):-len(PREMISE_CLOSE)] or None
    return None


def mark_topic_hits(plan: QueryPlan, store: KnowledgeStore, hits: Sequence[SearchHit]) -> List[SearchHit]:
    """The served hits; a hit with one of the plan's events, sectors, themes or companies among the annotation tags
    of its passages (its own id or a `merged_ids` one) is a copy with TOPIC_FLAG set. Ids, text and order stay."""
    wanted = {"events": set(plan.events), "sectors": set(plan.sectors), "themes": set(plan.themes),
              "companies": set(plan.companies)}

    def tagged(pid: str) -> bool:
        ann = store.annotation(pid) or {}
        return any(values.intersection(ann.get(name) or ()) for name, values in wanted.items())

    return [SearchHit(h.id, h.text, {**h.metadata, TOPIC_FLAG: True}, h.score)
            if any(tagged(pid) for pid in (h.id, *(h.metadata.get("merged_ids") or ()))) else h
            for h in hits]


# a sentence end: . ! or ?, closing quotes or brackets, citation markers, then space before the next sentence
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])[\"'”’)]*(?:\s*\[\d+(?:\s*,\s*\d+)*\])*(?=\s+[^\s\[])")
_ABBREVIATION_END_RE = re.compile(r"(?:^|[^A-Za-z])(?:[A-Z]|Mr|Mrs|Ms|Dr|St|Jr|Sr|Co|Inc|Corp|Ltd|No|vs)\.$")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+")
_YEAR_TOKEN_RE = re.compile(r"^(?:18|19|20)\d\d")
_PREMISE_STOP = {"the", "a", "an", "of", "in", "on", "at", "and", "or", "to", "was", "were", "is", "are", "it", "its",
                 "them", "they", "that", "this", "but", "from", "do", "does", "did", "not", "no", "s"}


def _sentence_spans(line: str) -> List[Tuple[int, int]]:
    """(start, end) of each sentence of a line, its closing quotes and markers included; "U.S." or "Mr." does not end
    one."""
    spans: List[Tuple[int, int]] = []
    start = 0
    for m in _SENTENCE_BREAK_RE.finditer(line):
        if not m.group() and _ABBREVIATION_END_RE.search(line[start:m.start()]):
            continue
        spans.append((start, m.end()))
        start = m.end()
        while start < len(line) and line[start].isspace():
            start += 1
    spans.append((start, len(line)))
    return [(a, b) for a, b in spans if line[a:b].strip()]


def _sentence_key(sentence: str) -> str:
    """A sentence as the dedupe compares it: no markers, list prefix, case or punctuation (quotes and dashes too)."""
    text = _LIST_PREFIX_RE.sub("", _CITE_RE.sub(" ", sentence)).lower()
    return " ".join(re.sub(r"[^\w\s]|_", " ", text).split())


def dedupe_sentences(text: str) -> Tuple[str, List[str]]:
    """(text, dropped): a sentence whose normalised text (`_sentence_key`) repeats an earlier one is dropped, the first
    kept; a sentence under DEDUPE_MIN_WORDS words always stays. A line left empty goes and the blank lines around it
    collapse; a text without a repeat comes back unchanged."""
    seen: set = set()
    dropped: List[str] = []
    lines: List[str] = []
    for line in str(text or "").split("\n"):
        prefix = _LIST_PREFIX_RE.match(line)
        head = prefix.group() if prefix else ""
        body = line[len(head):]
        kept, repeats = [], 0
        for a, b in _sentence_spans(body):
            sentence = body[a:b].strip()
            key = _sentence_key(sentence)
            if len(key.split()) >= DEDUPE_MIN_WORDS:
                if key in seen:
                    dropped.append(sentence)
                    repeats += 1
                    continue
                seen.add(key)
            kept.append(sentence)
        if not repeats:
            lines.append(line)
        elif kept:
            lines.append(head + " ".join(kept))
    if not dropped:
        return text, []
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(), dropped


def is_premise_line(sentence: str, premise: str) -> bool:
    """The sentence restates the premise note: it names one of the note's years or decades, at least half of its
    content words are the note's, and it has no money or percent figure."""
    words = [w for w in _sentence_key(sentence).split() if w not in _PREMISE_STOP]
    if not words or len(words) > PREMISE_LINE_MAX_WORDS or _GAP_FIGURE.search(sentence):
        return False
    note = set(_sentence_key(premise).split())
    shared = [w for w in words if w in note]
    return any(_YEAR_TOKEN_RE.match(w) for w in shared) and len(shared) * 2 >= len(words)


def strip_premise_line(answer: str, premise: Optional[str]) -> str:
    """The answer without its first sentence when that one is uncited and restates the premise (`is_premise_line`):
    the planner's note goes first instead (`with_premise`), so no premise wording of the model's own is kept."""
    if not premise or not answer:
        return answer
    lines = answer.split("\n")
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is None:
        return answer
    spans = _sentence_spans(lines[first])
    sentence = lines[first][spans[0][0]:spans[0][1]]
    if _CITE_RE.search(sentence) or not is_premise_line(sentence, premise):
        return answer
    lines[first] = lines[first][spans[1][0]:] if len(spans) > 1 else ""
    return "\n".join(lines).strip()


def with_premise(answer: str, premise: Optional[str]) -> str:
    """The premise note first, then the answer (a list after a blank line); the refusal line stays alone."""
    if not premise or not answer or answer == REFUSAL_LINE:
        return answer
    sep = "\n\n" if _LIST_PREFIX_RE.match(answer) else " "
    return f"{premise}{sep}{answer}"


def model_refused(raw_answer: str, premise: Optional[str] = None) -> bool:
    """The model's answer holds the refusal line, or is nothing but a premise sentence."""
    text = " ".join(_CITE_RE.sub(" ", str(raw_answer or "")).split())
    if _sentence_key(REFUSAL_LINE) in _sentence_key(text):
        return True
    return bool(premise and text) and not strip_premise_line(text, premise)


def partial_retry_prompt(prompt: str, raw_answer: str, hits: Sequence[SearchHit]) -> Optional[str]:
    """The prompt of the one partial retry, or None. Only a v3 prompt gets one, when the model refused
    (`model_refused`) while the prompt carries a premise note or a served hit carries the plan's topic tags
    (TOPIC_FLAG). The retry keeps the system prompt, the passages, the question and the premise correction, and asks
    for the covered part with RETRY_INSTRUCTIONS, which are shorter than V3_INSTRUCTIONS: the retry prompt fits the
    reader window wherever the first prompt did."""
    if not is_v3_prompt(prompt):
        return None
    premise = prompt_premise(prompt)
    if not model_refused(raw_answer, premise) or not (premise or any(h.metadata.get(TOPIC_FLAG) for h in hits)):
        return None
    return prompt[:-len(_V3_TAIL)] + _RETRY_TAIL


# --- scope ---
def scope_answer() -> FinanceAnswer:
    return FinanceAnswer(answer=SCOPE_REPLY, route="scope")


# --- numeric ---
def _value_text(row: Mapping[str, Any]) -> str:
    fmt = _UNIT_FORMATS.get(str(row.get("unit", "")))
    return fmt(row["value"]) if fmt and row.get("value") is not None else str(row.get("as_written", ""))


def plain_figure_lookup(plan: QueryPlan) -> bool:
    """No why/how/explain clause and no second sub-question: the question only asks for a figure."""
    text = plan.financial_query or plan.query
    if not text or _EXPLAIN_CLAUSE.search(text) or _SECOND_ASK.search(text):
        return False
    return len(split_question(text)) <= 1


def _named(name: str, text: str, low: str, aliases: Sequence[str] = ()) -> bool:
    """The name (any case) or one of its aliases (case kept, "Coke" but not "coke") is written in the text."""
    pattern = r"(?<![\w])%s(?:'s)?(?![\w])"
    return (bool(name) and re.search(pattern % re.escape(name.lower()), low) is not None) or any(
        len(a) >= 3 and re.search(pattern % re.escape(a), text) for a in aliases)


def _match_numbers(plan: QueryPlan, store: KnowledgeStore) -> List[Mapping[str, Any]]:
    """The best-matching (metric, entity) group of rows, by year; every group tied at the best score when several
    metrics match equally ("float": insurance float and cost of float)."""
    return [row for group in _match_groups(plan, store) for row in group][:MAX_NUMERIC_ROWS]


def metric_named(plan: QueryPlan, store: KnowledgeStore) -> bool:
    """One (metric, entity) group matches best and every token of its metric is in the question."""
    groups = _match_groups(plan, store, with_share=True)
    return len(groups) == 1 and groups[0][0][1] == 1.0


def _match_groups(plan: QueryPlan, store: KnowledgeStore, with_share: bool = False) -> List[List[Any]]:
    text = plan.financial_query or plan.query
    query_tokens = set(_tokens(text))
    low = text.lower()
    rows = store.numbers()
    aliases = store.company_aliases()
    entities = {str(r.get("entity", "")).strip() for r in rows}
    named = {e.lower() for e in entities if _named(e, text, low, aliases.get(e, ()))}
    companies = ({c.lower() for c in plan.companies} | {e for e in named if e in {a.lower() for a in aliases}}) - BERKSHIRE
    # another company named: only its rows; otherwise Berkshire's rows and named non-company entities (S&P 500)
    eligible = companies if companies else BERKSHIRE | {e for e in named if e not in {a.lower() for a in aliases}}
    scored: List[Tuple[float, Mapping[str, Any]]] = []
    for row in rows:
        metric_tokens = [t for t in _tokens(str(row["metric"]).replace("_", " ")) if t not in _METRIC_STOP]
        if not metric_tokens:
            continue
        share = sum(t in query_tokens for t in metric_tokens) / len(metric_tokens)
        entity = str(row.get("entity", "")).strip().lower()
        if entity not in eligible:  # a wrong-entity row is never used
            continue
        entity_hit = entity in named or entity in companies
        if share < 0.5 or (share < 1 and not entity_hit):
            continue
        year = row.get("year")
        if plan.years and year not in plan.years:
            continue
        if not plan.years and plan.year_range and not plan.year_range[0] <= int(year or 0) <= plan.year_range[1]:
            continue
        scored.append((share * 2 + entity_hit + (1 if plan.years else 0), share, row))
    if not scored:
        return []
    top = max(score for score, _, _ in scored)
    keys = list(dict.fromkeys((row["metric"], row.get("entity")) for score, _, row in scored if score == top))
    groups = []
    for key in keys:
        group = sorted(((row, share) for _, share, row in scored if (row["metric"], row.get("entity")) == key),
                       key=lambda item: item[0].get("year") or 0)
        groups.append([(row, share) if with_share else row for row, share in group])
    return groups


def figure_rows(plan: QueryPlan, store: KnowledgeStore, docs_by_id: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """The matching figure rows whose source passage can be served (same matching as the numeric answer)."""
    return [row for row in _match_numbers(plan, store) if row.get("source_id") in docs_by_id]


def figure_source_hits(rows: Sequence[Mapping[str, Any]], docs_by_id: Mapping[str, Any]) -> List[SearchHit]:
    """The rows' source passages, one hit each, in row order."""
    hits: Dict[str, SearchHit] = {}
    for row in rows:
        sid = row["source_id"]
        if sid not in hits and sid in docs_by_id:
            hits[sid] = _hit_from_doc(docs_by_id[sid], row.get("year"))
    return list(hits.values())


def _figure_line(row: Mapping[str, Any], marker: int) -> str:
    label = str(row["metric"]).replace("_", " ")
    entity = f"{row['entity']}, " if row.get("entity") else ""
    return (f"- {row.get('year')} ({entity}{label}): {_value_text(row)}; "
            f"as written: \"{row.get('as_written', '')}\" [{marker}]")


def figures_prompt(prompt: str, rows: Sequence[Mapping[str, Any]], hits: Sequence[SearchHit]) -> str:
    """The cited prompt plus the figure rows as extra context before the question; each row cites its served
    passage. Rows whose source is not served are left out; a prompt whose question block is unknown is unchanged."""
    index = {h.id: i for i, h in enumerate(hits, 1)}
    lines = [_figure_line(row, index[row["source_id"]]) for row in rows if row.get("source_id") in index]
    if not lines or _QUESTION_BLOCK not in prompt:
        return prompt
    block = f"{FIGURES_BEGIN}\n" + "\n".join(lines) + f"\n{FIGURES_END}\n\n"
    at = prompt.index(_QUESTION_BLOCK)
    return prompt[:at] + block + prompt[at:]


def numeric_answer(plan: QueryPlan, store: KnowledgeStore, docs_by_id: Mapping[str, Any]) -> Optional[FinanceAnswer]:
    """Answer from the numbers table, or None when no row matches or no source passage can be served."""
    rows = [row for row in _match_numbers(plan, store) if row.get("source_id") in docs_by_id]
    if not rows:
        return None
    hits: List[SearchHit] = []
    index: Dict[str, int] = {}
    lines: List[str] = []
    for row in rows:
        sid = row["source_id"]
        if sid not in index:
            hits.append(_hit_from_doc(docs_by_id[sid], row.get("year")))
            index[sid] = len(hits)
        lines.append(_figure_line(row, index[sid]))
    answer = "\n".join(lines)
    return FinanceAnswer(answer=answer, hits=hits, citations=parse_citations(answer, hits), route="numeric",
                         detail={"rows": len(rows)})


# --- digest ---
def digest_topic(plan: QueryPlan, store: KnowledgeStore) -> Optional[Tuple[str, Dict[str, Any]]]:
    """The first digest among the plan's events, sectors and themes (in that order)."""
    for topic in (*plan.events, *plan.sectors, *plan.themes):
        digest = store.digest(topic)
        if digest:
            return topic, digest
    return None


def digest_answer(
    plan: QueryPlan,
    store: KnowledgeStore,
    docs_by_id: Mapping[str, Any],
    reader_hits: Sequence[SearchHit] = (),
    gap_passages: int = DIGEST_GAP_PASSAGES,
) -> Optional[FinanceAnswer]:
    """Overview plus dated quotes citing their source passages; None when the plan has no usable digest."""
    found = digest_topic(plan, store)
    if found is None:
        return None
    topic, digest = found
    timeline = [e for e in digest.get("timeline") or [] if e.get("source_id") in docs_by_id and e.get("quote")]
    if not timeline:
        return None
    timeline = sorted(timeline, key=lambda e: int(e.get("year") or 0))
    hits: List[SearchHit] = []
    index: Dict[str, int] = {}
    quotes: List[str] = []
    for entry in timeline:
        sid = entry["source_id"]
        if sid not in index:
            hits.append(_hit_from_doc(docs_by_id[sid], entry.get("year")))
            index[sid] = len(hits)
        quotes.append(f"- {entry.get('year')}: \"{str(entry['quote']).strip()}\" [{index[sid]}]")
    answer = "\n\n".join(part for part in (str(digest.get("overview", "")).strip(), "\n".join(quotes)) if part)
    cited = len(hits)
    covered = {_year(h) for h in hits}
    gaps = [h for h in reader_hits if h.id not in index and _year(h) not in covered][:max(0, gap_passages)]
    hits.extend(gaps)
    return FinanceAnswer(answer=answer, hits=hits, citations=parse_citations(answer, hits[:cited]), route="digest",
                         detail={"topic": topic, "cited": cited, "gaps": len(gaps)})


# --- superlative ---
def _letter_of(pid: str, docs_by_id: Mapping[str, Any]) -> Optional[int]:
    doc = docs_by_id.get(pid)
    year = (getattr(doc, "metadata", None) or {}).get("year") if doc is not None else None
    if year is None and pid[:4].isdigit():
        year = pid[:4]
    try:
        return int(year)
    except (TypeError, ValueError):
        return None


def stance_letters(plan: QueryPlan, store: KnowledgeStore,
                   docs_by_id: Mapping[str, Any]) -> List[Tuple[str, List[int], List[str]]]:
    """[(target, sorted distinct letters, passage ids)] for the plan's superlative, most letters first.

    A target counts once per letter however many of that letter's passages carry the stance. Sector targets are
    taxonomy sector keys; company targets are every other stance target that is not a theme. The plan's year range,
    when it has one, limits the letters counted.
    """
    kind, _, polarity = plan.superlative.partition(":")
    sectors = set((store.taxonomy.get("sectors") or {}).keys())
    themes = set((store.taxonomy.get("themes") or {}).keys())
    letters: Dict[str, set] = {}
    passages: Dict[str, List[str]] = {}
    for ann in store.annotations():
        letter = _letter_of(ann["id"], docs_by_id)
        if letter is None or (plan.year_range and not plan.year_range[0] <= letter <= plan.year_range[1]):
            continue
        for stance in ann.get("stance") or ():
            target = str(stance.get("target", ""))
            if stance.get("polarity") != polarity or not target:
                continue
            if (kind == "sector") != (target in sectors) or target in themes:
                continue
            letters.setdefault(target, set()).add(letter)
            if ann["id"] not in passages.setdefault(target, []):
                passages[target].append(ann["id"])
    ranked = sorted(letters, key=lambda t: (-len(letters[t]), t))
    return [(t, sorted(letters[t]), passages[t]) for t in ranked]


def _label(target: str, kind: str) -> str:
    label = target.replace("_", " ") if kind == "sector" else target
    return label[:1].upper() + label[1:]


def _spread(items: Sequence[Any], n: int) -> List[Any]:
    """n items evenly spread over the sequence (all of them when there are at most n)."""
    if len(items) <= n:
        return list(items)
    step = (len(items) - 1) / max(1, n - 1)
    return [items[round(i * step)] for i in range(n)]


def _interleave(lists: Sequence[Sequence[str]]) -> List[str]:
    """The lists taken in turns (first items of each, then the second ones, ...), without repeats."""
    out: Dict[str, None] = {}
    for row in zip_longest(*lists):
        out.update((pid, None) for pid in row if pid is not None)
    return list(out)


def _source_evidence(plan: QueryPlan, top: Sequence[Tuple[str, List[int], List[str]]],
                     digest: Optional[Mapping[str, Any]], docs_by_id: Mapping[str, Any], served: set,
                     budget: int) -> List[str]:
    """Uncited evidence from the knowledge sources (finance-recall-75), [] when their settings are off: the winner's
    digest timeline passages (FINANCE_DIGEST_SOURCE) and the stance passages of the top targets (FINANCE_STANCE_SOURCE,
    the targets in turns, the winner first), each spread across letters, the two lists in turns. A letter not served
    yet comes first; at most `budget` ids, none of them already `served`."""
    letter = lambda pid: _letter_of(pid, docs_by_id)  # noqa: E731
    lists: List[List[str]] = []
    if finance_setting("FINANCE_DIGEST_SOURCE") and digest:
        lo, hi = plan.year_range or (0, 9999)
        timeline = sorted((e for e in digest.get("timeline") or [] if e.get("source_id") in docs_by_id
                           and lo <= int(e.get("year") or 0) <= hi), key=lambda e: int(e.get("year") or 0))
        lists.append(spread_letters([e["source_id"] for e in timeline], budget, letter))
    if finance_setting("FINANCE_STANCE_SOURCE"):
        lists.append(_interleave([spread_letters([p for p in ids if p in docs_by_id], budget, letter)
                                  for _, _, ids in top]))
    pool = [pid for pid in _interleave(lists) if pid not in served]
    letters = {letter(pid) for pid in served}
    picked: List[str] = []
    for pid in pool:  # one passage of each letter not served yet, then the rest in order
        if len(picked) < budget and letter(pid) not in letters:
            picked.append(pid)
            letters.add(letter(pid))
    picked += [pid for pid in pool if pid not in picked][:max(0, budget - len(picked))]
    return picked


def tagged_stance_ids(plan: QueryPlan, store: KnowledgeStore, docs_by_id: Mapping[str, Any], kind: str, winner: str,
                      polarity: str, skip: set, n: int) -> List[str]:
    """Up to n passages tagged with the winner that carry a stance of the asked polarity (FINANCE_SUPERLATIVE_TAGGED).

    A passage counts by its stance rows of the asked polarity: one on the winner itself (2 points), or, for a sector
    winner, one on a company of a passage tagged with that sector (1 point; other sectors and themes do not count). A
    passage with the opposite polarity on the winner itself is left out. The strongest passages come first within a
    letter, and the letters are taken in rounds spread across the years (`spread_letters`), so one letter does not fill
    the list. The plan's year range limits the letters; `skip` ids (already cited) are left out."""
    if n <= 0:
        return []
    opposite = "negative" if polarity == "positive" else "positive"
    sectors = set((store.taxonomy.get("sectors") or {}).keys())
    themes = set((store.taxonomy.get("themes") or {}).keys())
    scored: List[Tuple[int, str]] = []
    for ann in store.annotations():
        pid = ann["id"]
        letter = _letter_of(pid, docs_by_id)
        if pid not in docs_by_id or pid in skip or not store.is_financial(pid) or letter is None \
                or (plan.year_range and not plan.year_range[0] <= letter <= plan.year_range[1]):
            continue
        stances = ann.get("stance") or ()
        if any(s.get("target") == winner and s.get("polarity") == opposite for s in stances):
            continue
        points = 0
        for stance in stances:
            target = str(stance.get("target", ""))
            if stance.get("polarity") != polarity:
                continue
            if target == winner:
                points += 2
            elif kind == "sector" and target and target not in sectors and target not in themes \
                    and winner in (ann.get("sectors") or ()):
                points += 1
        if points:
            scored.append((points, pid))
    scored.sort(key=lambda row: -row[0])  # stable: annotation order within the same strength
    return spread_letters([pid for _, pid in scored], n, lambda pid: letter_of(pid, docs_by_id))


def _retriever_reranker() -> Any:
    """The cross-encoder of the app's retriever (`ask_flow._state`), None when there is none. ask_flow imports this
    module, so it is imported here, at call time."""
    try:
        from src.services import ask_flow
        return getattr(ask_flow._state.get("retriever"), "reranker", None)
    except Exception:  # noqa: BLE001 - no retriever (a unit test, an offline script) means no reranker
        return None


def _spread_ranked(ranked: Sequence[str], n: int, docs_by_id: Mapping[str, Any]) -> List[str]:
    """The first n of `ranked` taken in rounds: a round takes the next passage of every letter that has one left, in
    rank order, so no letter gets a second passage before every letter has a first. Unlike `spread_letters`, which
    spaces the letters of a round evenly over time, the rank decides which letters come first."""
    taken: Dict[Optional[int], int] = {}
    keyed: List[Tuple[int, int, str]] = []
    for rank, pid in enumerate(ranked):
        letter = _letter_of(pid, docs_by_id)
        keyed.append((taken.get(letter, 0), rank, pid))
        taken[letter] = taken.get(letter, 0) + 1
    return [pid for _, _, pid in sorted(keyed)[:max(0, n)]]


def reranked_stance_ids(plan: QueryPlan, store: KnowledgeStore, docs_by_id: Mapping[str, Any], reranker: Any, kind: str,
                        winner: str, polarity: str, skip: set, n: int) -> List[str]:
    """Up to n of the winner's tagged stance passages that the cross-encoder scores highest against the question
    (FINANCE_SUPERLATIVE_RERANK), best first.

    The candidates are the first SUPERLATIVE_RERANK_POOL ids of `tagged_stance_ids` (same pool: the asked polarity on
    the winner, `skip` ids left out); the question is the one `finance_search` reranks with. The best N are taken in
    rounds across the letters (`_spread_ranked`), so one letter does not fill the list."""
    if n <= 0:
        return []
    pool = tagged_stance_ids(plan, store, docs_by_id, kind, winner, polarity, skip, SUPERLATIVE_RERANK_POOL)
    if not pool:
        return []
    candidates = [_hit_from_doc(docs_by_id[pid]) for pid in pool]
    scored = reranker.rerank(plan.financial_query or plan.query, candidates, top_k=len(candidates))
    return _spread_ranked([h.id for h in scored], n, docs_by_id)


def superlative_answer(plan: QueryPlan, store: KnowledgeStore, docs_by_id: Mapping[str, Any],
                       max_hits: int = 16, max_new_tokens: int = 900, reranker: Any = None) -> Optional[FinanceAnswer]:
    """Top targets by distinct letters with a stance of the asked polarity, then quotes from the winner's digest
    timeline (or, without a digest, its stance quotes), citing the quoted passages; None without stance data.

    The served budget is `max_hits`, or FINANCE_K_CORPUS when that is larger; what is served beyond `max_hits` is cut
    to fit the reader window (`max_new_tokens` is the answer budget the window leaves out). FINANCE_SUPERLATIVE_RERANK
    uses `reranker`, else the retriever's (`_retriever_reranker`)."""
    if not plan.superlative:
        return None
    ranked = [r for r in stance_letters(plan, store, docs_by_id) if r[2]]
    if not ranked:
        return None
    kind, _, polarity = plan.superlative.partition(":")
    top = ranked[:SUPERLATIVE_TOP]
    winner, _, winner_ids = top[0]
    adverb = "positively" if polarity == "positive" else "negatively"
    lines = [f"Ranked by the number of letters in which Buffett speaks {adverb} about it (letters counted once each, "
             f"from the passage annotations):"]
    for rank, (target, years, _) in enumerate(top, 1):
        noun = "letter" if len(years) == 1 else "letters"
        lines.append(f"{rank}. {_label(target, kind)}: {len(years)} {noun} ({years[0]}–{years[-1]})")

    # quotes that carry the asked stance: the winner's digest timeline entries on its stance passages first, then
    # (for the letters they do not cover) the verbatim stance quotes of the annotations, one per letter
    stance_ids = set(winner_ids)
    digest = store.digest(f"{kind}:{winner}") or store.digest(winner)
    timeline = [e for e in (digest or {}).get("timeline") or []
                if e.get("quote") and e.get("source_id") in docs_by_id and e["source_id"] in stance_ids]
    quotes = _spread(sorted(timeline, key=lambda e: int(e.get("year") or 0)), SUPERLATIVE_QUOTES)
    covered = {_letter_of(e["source_id"], docs_by_id) for e in quotes}
    by_letter: Dict[int, Dict[str, Any]] = {}
    for pid in winner_ids:
        letter = _letter_of(pid, docs_by_id)
        if pid not in docs_by_id or letter in covered:
            continue
        for stance in (store.annotation(pid) or {}).get("stance") or ():
            if stance.get("target") == winner and stance.get("polarity") == polarity and stance.get("quote"):
                by_letter.setdefault(letter, {"year": letter, "quote": stance["quote"], "source_id": pid})
    quotes += _spread([by_letter[y] for y in sorted(by_letter)], max(0, SUPERLATIVE_QUOTES - len(quotes)))
    quotes = sorted(quotes, key=lambda e: int(e.get("year") or 0))

    hits: List[SearchHit] = []
    index: Dict[str, int] = {}
    quoted: List[str] = []
    for entry in quotes:
        sid = entry["source_id"]
        if sid not in index:
            hits.append(_hit_from_doc(docs_by_id[sid], entry.get("year")))
            index[sid] = len(hits)
        quoted.append(f"- {entry.get('year')}: \"{str(entry['quote']).strip()}\" [{index[sid]}]")
    if quoted:
        lines += ["", f"What he wrote about {_label(winner, kind).lower() if kind == 'sector' else winner}:", *quoted]
    answer = "\n".join(lines)
    cited = len(hits)
    limit = max(max_hits, finance_setting("FINANCE_K_CORPUS"))
    budget = max(0, limit - cited)
    relevant: List[str] = []
    rerank_note = ""
    if finance_setting("FINANCE_SUPERLATIVE_RERANK") > 0:
        reranker = reranker if reranker is not None else _retriever_reranker()
        if reranker is None:
            rerank_note = "skipped: no reranker"
        else:
            try:
                relevant = reranked_stance_ids(plan, store, docs_by_id, reranker, kind, winner, polarity, set(index),
                                               min(finance_setting("FINANCE_SUPERLATIVE_RERANK"), budget))
                rerank_note = f"{len(relevant)} served"
            except Exception as exc:  # noqa: BLE001 - a rerank failure never fails the answer
                rerank_note = f"skipped: {type(exc).__name__}"
    taken = set(index) | set(relevant)
    tagged = tagged_stance_ids(plan, store, docs_by_id, kind, winner, polarity, taken,
                               min(finance_setting("FINANCE_SUPERLATIVE_TAGGED"), budget - len(relevant)))
    extra = relevant + tagged + _source_evidence(plan, top, digest, docs_by_id, taken | set(tagged),
                                                 budget - len(relevant) - len(tagged))
    # uncited evidence for the count: the winner's other stance passages, one per letter, after the quoted ones
    by_letter_ids: Dict[int, str] = {}
    served_letters = {_year(h) for h in hits} | {_letter_of(pid, docs_by_id) for pid in extra}
    for pid in winner_ids:
        letter = _letter_of(pid, docs_by_id)
        if pid in docs_by_id and pid not in index and pid not in extra and letter not in served_letters:
            by_letter_ids.setdefault(letter, pid)
    extra += _spread([by_letter_ids[y] for y in sorted(by_letter_ids)], budget - len(extra))
    hits.extend(_hit_from_doc(docs_by_id[pid]) for pid in extra)
    if len(hits) > max(max_hits, cited):  # a budget beyond today's: the extra passages must fit the reader window
        window, per_passage = reader_window_chars(max_new_tokens), retrieval_setting("LLM_PASSAGE_MAX_CHARS")
        while len(hits) > max(max_hits, cited) and sum(min(len(h.text), per_passage) for h in hits) > window:
            hits.pop()
    detail = {"ranking": [(t, len(y)) for t, y, _ in top], "cited": cited, "evidence": len(hits) - cited}
    if rerank_note:
        detail["rerank"] = rerank_note
    return FinanceAnswer(answer=answer, hits=hits, citations=parse_citations(answer, hits[:cited]), route="superlative",
                         detail=detail)
