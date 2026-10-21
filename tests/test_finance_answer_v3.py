"""finance-answer-quality group 3: reader answer construction v3 (FINANCE_ANSWER_V3) with a fake LLM.

The year-grouped evidence pack and its instructions, the premise sentence, the sentence dedupe and the one partial
retry; then the finance route (prompt, token window, topic flags) and /ask + /ask/stream end to end.
"""
from __future__ import annotations

import re
from dataclasses import replace

import pytest

from src.evaluation.claim_validator import evidence_sentences
from src.generation import finance_answer as fa
from src.generation.prompt import REFUSAL_LINE, build_cited_prompt
from src.generation.providers.llama_provider import split_prompt
from src.knowledge import KnowledgeStore
from src.query.plan import QueryPlan
from src.services import ask_flow
from src.storage import SearchHit
from test_ask_flow_finance import TREND_Q, _ask, _stream, env  # noqa: F401  (env is a fixture)

NOTE = "The port strikes were in 1971 and 1976; the letters from the 1990s do not discuss them."
STRIKE = "Two port strikes in the 1970s cut our textile shipments sharply."
FLOAT = "Insurance float grew steadily over the decade. Our underwriting stayed disciplined through the cycle."
SEES = ("We bought See's for $25 million when its sales were $30 million. "
        "The capital then required to conduct the business was $8 million. "
        "See's later earned far more than the business ever needed in new capital.")
BOUGHT = "We bought See's for $25 million when its sales were $30 million."
QUESTION = "What did Buffett say about the port strikes of the 1990s?"
HEADER_RE = re.compile(r"LETTER OF \d{4} \(passages? \[\d+\](?:-\[\d+\])?\)\n\n")
CONVERTIBLE_Q = "How did Buffett's view of convertible preferred stocks change over time?"  # passes the stream's gate


def _hit(pid, text, year, **meta):
    return SearchHit(pid, text, {"year": year, "source_file": f"{year}.pdf", **meta}, 1.0)


def _hits():
    """Served in year order, two passages from one letter."""
    return [_hit("1978_p0001", STRIKE, 1978), _hit("1991_p0001", FLOAT, 1991), _hit("1991_p0002", SEES, 1991)]


@pytest.fixture
def prompt_env(monkeypatch):
    for name in ("PROMPT_VERSION", "LLM_MODEL_PATH", "ANSWER_REFUSAL_RETRY", "EVIDENCE_GATE_SOFT",
                 "ANSWER_NEIGHBOR_AFTER", "ANSWER_NEIGHBOR_BEFORE", "FINANCE_ANSWER_V3"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ask_flow, "_state", {"knowledge": KnowledgeStore()})


class _Replies:
    """A fake LLM that answers from a list, one reply per call (the last one repeats)."""
    provider_name = "remote"
    model = "fake"

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt, max_new_tokens=None):
        self.prompts.append(prompt)
        return self.replies[min(len(self.prompts), len(self.replies)) - 1]


class _Broken(_Replies):
    def generate(self, prompt, max_new_tokens=None):
        self.prompts.append(prompt)
        if len(self.prompts) > 1:
            raise RuntimeError("provider down")
        return REFUSAL_LINE


# --- prompt shape ---
def test_v3_prompt_groups_the_passages_by_letter_year_and_keeps_the_trained_blocks(prompt_env):
    hits = _hits()
    history = [{"role": "user", "content": "Tell me about Berkshire's textile mill."}]
    plain = build_cited_prompt(QUESTION, hits, history=history)
    v3 = fa.reader_prompt_v3(QUESTION, hits, history)
    assert split_prompt(v3)[0] == split_prompt(plain)[0]  # the system message is unchanged
    assert v3.endswith(f" {fa.V3_INSTRUCTIONS}\n\nAnswer:") and fa.READER_INSTRUCTIONS not in v3
    assert HEADER_RE.findall(v3) == ["LETTER OF 1978 (passage [1])\n\n", "LETTER OF 1991 (passages [2]-[3])\n\n"]
    # without the headers and the instructions it is the cited prompt: same [n] lines, question and history blocks
    assert HEADER_RE.sub("", v3).replace(" " + fa.V3_INSTRUCTIONS, "") == plain
    passages = v3[v3.index("BEGIN UNTRUSTED PASSAGES"):v3.index("END UNTRUSTED PASSAGES")]
    assert re.findall(r"^\[(\d+)\] \(year=(\d+)", passages, re.M) == [("1", "1978"), ("2", "1991"), ("3", "1991")]
    assert v3.index("END USER QUESTION") < v3.index(fa.V3_INSTRUCTIONS)
    assert fa.is_reader_prompt(v3) and fa.is_v3_prompt(v3) and fa.prompt_premise(v3) is None
    assert not fa.is_v3_prompt(fa.reader_prompt(QUESTION, hits)) and not fa.is_v3_prompt(plain)


def test_a_letter_seen_again_later_gets_a_new_header_and_an_unknown_year_its_own(prompt_env):
    hits = [_hit("1991_p0001", FLOAT, 1991), _hit("1978_p0001", STRIKE, 1978), _hit("1991_p0002", SEES, 1991),
            SearchHit("x_p0001", FLOAT, {}, 1.0)]
    block = fa.year_passage_block(hits)
    assert re.findall(r"^LETTER .*$", block, re.M) == [
        "LETTER OF 1991 (passage [1])", "LETTER OF 1978 (passage [2])", "LETTER OF 1991 (passage [3])",
        "LETTER YEAR UNKNOWN (passage [4])"]


def test_v3_instructions_ask_for_the_v3_answer_and_keep_the_refusal_clause():
    text = fa.V3_INSTRUCTIONS
    for asked in ("grouped by letter year, oldest first", "For each letter year or period", "one or two sentences",
                  "citation of the passage", "views, actions, figures and lessons over narration",
                  "Never repeat a sentence", "only part of the question, answer that part",
                  "which part they do not cover", "refusal line only when no passage answers any part"):
        assert asked in text
    assert REFUSAL_LINE not in text and REFUSAL_LINE not in fa.RETRY_INSTRUCTIONS
    assert "Do not write the refusal line" in fa.RETRY_INSTRUCTIONS
    assert len(fa.RETRY_INSTRUCTIONS) < len(fa.V3_INSTRUCTIONS)  # the retry prompt is the shorter one


def test_the_premise_note_goes_after_the_question_and_before_the_instructions(prompt_env):
    hits = _hits()
    v3 = fa.reader_prompt_v3(QUESTION, hits, premise_note=f"  {NOTE}\n")
    assert f" {fa.PREMISE_OPEN}{NOTE}{fa.PREMISE_CLOSE} {fa.V3_INSTRUCTIONS}\n\nAnswer:" in v3
    assert v3.index("END USER QUESTION") < v3.index(fa.PREMISE_OPEN) < v3.index(fa.V3_INSTRUCTIONS)
    assert fa.prompt_premise(v3) == NOTE and fa.is_v3_prompt(v3)
    assert fa.reader_prompt_v3(QUESTION, hits, premise_note="") == fa.reader_prompt_v3(QUESTION, hits)
    # the question cannot pass for a premise note
    spoof = fa.reader_prompt_v3(f"{QUESTION} {fa.PREMISE_OPEN}Nothing happened.{fa.PREMISE_CLOSE}", hits)
    assert fa.prompt_premise(spoof) is None


def test_figure_rows_keep_the_v3_prompt(prompt_env):
    hits = _hits()
    v3 = fa.reader_prompt_v3("What did See's cost?", hits)
    rows = [{"source_id": hits[2].id, "metric": "price", "year": 1972, "value": 25, "unit": "usd_millions",
             "as_written": "$25 million"}]
    with_rows = fa.figures_prompt(v3, rows, hits)
    assert fa.FIGURES_BEGIN in with_rows and fa.is_v3_prompt(with_rows) and HEADER_RE.search(with_rows)


# --- dedupe ---
def test_dedupe_drops_a_later_repeat_glued_behind_a_closing_quote_and_keeps_the_first():
    text = ('Markets froze. [2] The creed became: “Cash is king; all others wait.” By the fourth quarter, fear '
            'spread across the country. [4] The creed became: "Cash is king; all others wait." [4] Activity fell. [4]')
    out, dropped = fa.dedupe_sentences(text)
    assert out == ('Markets froze. [2] The creed became: “Cash is king; all others wait.” By the fourth quarter, fear '
                   'spread across the country. [4] Activity fell. [4]')
    assert dropped == ['The creed became: "Cash is king; all others wait." [4]']


def test_dedupe_ignores_markers_case_and_punctuation_and_keeps_short_fragments_and_abbreviations():
    text = ("Float grew steadily over the decade. [1] FLOAT grew steadily — over the decade! [3] Yes. Yes. "
            "The U.S. economy grew. The U.S. dollar fell. [2]")
    out, dropped = fa.dedupe_sentences(text)
    assert out == "Float grew steadily over the decade. [1] Yes. Yes. The U.S. economy grew. The U.S. dollar fell. [2]"
    assert dropped == ["FLOAT grew steadily — over the decade! [3]"]


def test_dedupe_removes_a_repeated_list_item_and_collapses_the_blank_lines():
    text = ("Float grew steadily over the decade. [1]\n\n- Underwriting stayed disciplined through the cycle. [1]\n"
            "- Float grew steadily over the decade. [1]\n- Float grew. See's earned well over its capital needs. [2]"
            "\n\n1. Float grew steadily over the decade. [1]\n\nSee's earned well over its capital needs. [2]")
    out, dropped = fa.dedupe_sentences(text)
    assert out == ("Float grew steadily over the decade. [1]\n\n- Underwriting stayed disciplined through the cycle. [1]\n"
                   "- Float grew. See's earned well over its capital needs. [2]")
    assert len(dropped) == 3
    assert fa.dedupe_sentences(out) == (out, [])  # idempotent


def test_dedupe_returns_a_text_without_repeats_unchanged():
    text = "  Float grew steadily. [1]\n\n\n- See's earned well. [2]  "
    out, dropped = fa.dedupe_sentences(text)
    assert out is text and dropped == []
    assert fa.dedupe_sentences("") == ("", [])


# --- premise sentence ---
@pytest.mark.parametrize("sentence,premise", [
    ("The port strikes happened in 1971 and 1976, not in the 1990s.", True),
    ("Port strikes took place in 1971 and 1976; the 1990s letters do not discuss them.", True),
    ("The port strikes were in the 1970s.", False),                        # no year or decade of the note
    ("In 1976 we shipped textiles to Europe and Asia through other ports and carriers.", False),  # mostly new words
    ("The port strikes of 1971 cost us $5 million.", False),               # a money figure is a claim
])
def test_is_premise_line(sentence, premise):
    assert fa.is_premise_line(sentence, NOTE) is premise


def test_strip_premise_line_takes_out_only_an_uncited_first_premise_sentence():
    model = "The port strikes happened in 1971 and 1976, not in the 1990s."
    assert fa.strip_premise_line(f"{model} {STRIKE} [1]", NOTE) == f"{STRIKE} [1]"
    assert fa.strip_premise_line(f"{model} [1] {STRIKE} [1]", NOTE) == f"{model} [1] {STRIKE} [1]"  # cited: a claim
    assert fa.strip_premise_line(f"{STRIKE} [1] {model}", NOTE) == f"{STRIKE} [1] {model}"  # not first
    assert fa.strip_premise_line(f"{model} {STRIKE} [1]", None) == f"{model} {STRIKE} [1]"
    assert fa.strip_premise_line(model, NOTE) == ""


def test_with_premise_puts_the_note_first_and_leaves_the_refusal_alone():
    assert fa.with_premise(f"{STRIKE} [1]", NOTE) == f"{NOTE} {STRIKE} [1]"
    assert fa.with_premise(f"- {STRIKE} [1]", NOTE) == f"{NOTE}\n\n- {STRIKE} [1]"
    assert fa.with_premise(REFUSAL_LINE, NOTE) == REFUSAL_LINE and fa.with_premise("x", None) == "x"


def test_finalize_puts_the_planner_note_first_in_place_of_the_model_premise_sentence(prompt_env):
    hits = _hits()
    prompt = fa.reader_prompt_v3(QUESTION, hits, premise_note=NOTE)
    raw = f"The port strikes happened in 1971 and 1976, not in the 1990s. {STRIKE} [1] {BOUGHT} [3]"
    trace = {}
    answer, citations = ask_flow._finalize_answer(None, prompt, hits, raw, 900, trace)
    assert answer.startswith(f"{NOTE} {STRIKE} [1] {BOUGHT} [3]") and "happened in 1971" not in answer
    assert answer.count(NOTE) == 1 and [c["passage_ids"] for c in citations][:2] == [[hits[0].id], [hits[2].id]]
    # the model left the premise out: the note still goes first
    answer, _ = ask_flow._finalize_answer(None, prompt, hits, f"{STRIKE} [1]", 900)
    assert answer == f"{NOTE} {STRIKE} [1]"
    # no premise note: no premise sentence
    answer, _ = ask_flow._finalize_answer(None, fa.reader_prompt_v3(QUESTION, hits), hits, f"{STRIKE} [1]", 900)
    assert answer == f"{STRIKE} [1]"


def test_finalize_drops_repeats_on_the_v3_prompt_only(prompt_env):
    hits = _hits()
    raw = f"{STRIKE} [1] {STRIKE} [1]"
    trace = {}
    v3, _ = ask_flow._finalize_answer(None, fa.reader_prompt_v3(QUESTION, hits), hits, raw, 900, trace)
    assert v3 == f"{STRIKE} [1]" and trace["validator"]["deduped"] == [f"{STRIKE} [1]"]
    v2, _ = ask_flow._finalize_answer(None, fa.reader_prompt(QUESTION, hits), hits, raw, 900)
    assert v2 == raw  # the v2 reader keeps today's behaviour


def test_finalize_drops_a_repeat_that_expansion_adds(prompt_env):
    creed = "The creed became: “Cash is king; all others wait.”"
    passage = (f"Credit markets froze across the country that autumn. {creed} By the fourth quarter, fear spread "
               "across the land. A freefall in business activity soon ensued.")
    hits = [_hit("2008_p0001", passage, 2008)]
    # the passage splitter keeps the quoted sentence and the next one together, so expansion after the first answer
    # sentence adds the quote again
    raw = f"Credit markets froze across the country that autumn. [1] {creed} [1]"
    v2, _ = ask_flow._finalize_answer(None, fa.reader_prompt(QUESTION, hits), hits, raw, 900)
    trace = {}
    v3, _ = ask_flow._finalize_answer(None, fa.reader_prompt_v3(QUESTION, hits), hits, raw, 900, trace)
    assert v2.count(creed) == 2 and v3.count(creed) == 1
    assert v3 == v2.replace(f" {creed} [1]", "", 1) and trace["validator"]["deduped"] == [f"{creed} [1]"]


# --- partial retry ---
def test_a_refusal_with_a_premise_note_is_retried_once_for_the_covered_part(prompt_env):
    hits = _hits()
    prompt = fa.reader_prompt_v3(QUESTION, hits, premise_note=NOTE)
    llm = _Replies(REFUSAL_LINE, f"{STRIKE} [1] The passages do not cover the 1990s.")
    trace = {"refusal_retry": False}
    answer, citations = ask_flow._finalize_answer(llm, prompt, hits, llm.generate(prompt), 900, trace)
    assert len(llm.prompts) == 2
    retry = llm.prompts[1]
    assert len(retry) < len(prompt)
    assert retry == prompt[:-len(f" {fa.V3_INSTRUCTIONS}\n\nAnswer:")] + f" {fa.RETRY_INSTRUCTIONS}\n\nAnswer:"
    assert fa.prompt_premise(retry) == NOTE and fa.is_reader_prompt(retry) and not fa.is_v3_prompt(retry)
    assert answer == f"{NOTE} {STRIKE} [1] The passages do not cover the 1990s."
    assert citations and citations[0]["passage_ids"] == [hits[0].id]
    assert trace["refusal_retry"] is True and trace["first_raw_answer"] == REFUSAL_LINE
    assert trace["retry_prompt_chars"] == len(retry) and trace["raw_answer"] == llm.replies[1]


@pytest.mark.parametrize("second", [
    REFUSAL_LINE,                                                       # refused again
    "Zebras graze peacefully across the open savanna every spring. [1]",  # nothing the validator keeps
])
def test_a_retry_that_does_not_validate_leaves_the_refusal(prompt_env, second):
    hits = _hits()
    prompt = fa.reader_prompt_v3(QUESTION, hits, premise_note=NOTE)
    llm = _Replies(REFUSAL_LINE, second)
    answer, citations = ask_flow._finalize_answer(llm, prompt, hits, llm.generate(prompt), 900)
    assert answer == REFUSAL_LINE and citations == [] and len(llm.prompts) == 2  # one retry, never a second


def test_a_premise_sentence_alone_counts_as_a_refusal(prompt_env):
    hits = _hits()
    prompt = fa.reader_prompt_v3(QUESTION, hits, premise_note=NOTE)
    llm = _Replies("The port strikes happened in 1971 and 1976, not in the 1990s.", f"{STRIKE} [1]")
    answer, _ = ask_flow._finalize_answer(llm, prompt, hits, llm.generate(prompt), 900)
    assert answer == f"{NOTE} {STRIKE} [1]" and len(llm.prompts) == 2


def test_served_passages_with_the_plan_topic_trigger_the_retry_and_nothing_else_does(prompt_env):
    hits = _hits()
    prompt = fa.reader_prompt_v3(QUESTION, hits)
    plain = _Replies(REFUSAL_LINE, f"{STRIKE} [1]")
    assert ask_flow._finalize_answer(plain, prompt, hits, plain.generate(prompt), 900) == (REFUSAL_LINE, [])
    assert len(plain.prompts) == 1  # no premise note, no topic passage: no retry
    flagged = [hits[0], replace(hits[1], metadata={**hits[1].metadata, fa.TOPIC_FLAG: True}), hits[2]]
    tagged = _Replies(REFUSAL_LINE, f"{STRIKE} [1]")
    answer, _ = ask_flow._finalize_answer(tagged, prompt, flagged, tagged.generate(prompt), 900)
    assert answer == f"{STRIKE} [1]" and len(tagged.prompts) == 2
    # the v2 reader prompt and the legacy prompt never get the partial retry
    for other in (fa.reader_prompt(QUESTION, flagged), build_cited_prompt(QUESTION, flagged)):
        llm = _Replies(REFUSAL_LINE, f"{STRIKE} [1]")
        assert ask_flow._finalize_answer(llm, other, flagged, llm.generate(other), 900)[0] == REFUSAL_LINE
        assert len(llm.prompts) == 1


def test_an_answer_the_validator_empties_is_not_retried(prompt_env):
    hits = _hits()
    prompt = fa.reader_prompt_v3(QUESTION, hits, premise_note=NOTE)
    llm = _Replies("Zebras graze peacefully across the open savanna every spring. [1]", f"{STRIKE} [1]")
    assert ask_flow._finalize_answer(llm, prompt, hits, llm.generate(prompt), 900)[0] == REFUSAL_LINE
    assert len(llm.prompts) == 1  # the model answered; only a model refusal is retried


def test_a_provider_error_on_the_retry_keeps_the_refusal(prompt_env):
    hits = _hits()
    prompt = fa.reader_prompt_v3(QUESTION, hits, premise_note=NOTE)
    llm = _Broken()
    assert ask_flow._finalize_answer(llm, prompt, hits, llm.generate(prompt), 900) == (REFUSAL_LINE, [])


def test_generate_answer_records_the_partial_retry_in_the_trace(prompt_env, monkeypatch):
    from src.generation.evidence_gate import EvidenceAssessment
    monkeypatch.setattr(ask_flow, "assess_evidence", lambda *a, **k: EvidenceAssessment(True, "ok", 0.9))
    hits = _hits()
    llm = _Replies(REFUSAL_LINE, f"{STRIKE} [1]")
    trace = {}
    answer, _ = ask_flow._generate_answer(llm, fa.reader_prompt_v3(QUESTION, hits, premise_note=NOTE), hits, 900,
                                          query=QUESTION, per_period=False, trace=trace)
    assert answer == f"{NOTE} {STRIKE} [1]" and len(llm.prompts) == 2
    assert trace["refusal_retry"] is True and "refusal_by" not in trace


def test_mark_topic_hits_flags_copies_of_the_tagged_passages_only():
    store = KnowledgeStore(annotations=[
        {"id": "1978_p0001", "events": ["port_strike"]}, {"id": "1991_p0003", "companies": ["See's"]},
        {"id": "1991_p0001", "themes": ["float"]}])
    hits = _hits()
    merged = replace(hits[2], metadata={**hits[2].metadata, "merged_ids": ["1991_p0002", "1991_p0003"]})
    served = [hits[0], hits[1], merged]
    plan = QueryPlan(query="q", events=("port_strike",), companies=("See's",))
    marked = fa.mark_topic_hits(plan, store, served)
    assert [h.metadata.get(fa.TOPIC_FLAG, False) for h in marked] == [True, False, True]
    assert [h.id for h in marked] == [h.id for h in served] and marked[1] is served[1]
    assert fa.TOPIC_FLAG not in served[0].metadata  # the served hits themselves are not changed
    assert not any(h.metadata.get(fa.TOPIC_FLAG) for h in fa.mark_topic_hits(QueryPlan(query="q"), store, served))


# --- the finance route and both endpoints ---
def test_the_finance_route_builds_the_v3_prompt_in_year_order_only_with_the_flag(env, monkeypatch):  # noqa: F811
    query = "What did Buffett say about bonds and inflation?"  # not a trend: v2 keeps the retrieval order
    off = ask_flow.finance_route(ask_flow.AskRequest(query=query))
    assert fa.is_reader_prompt(off.prompt) and not fa.is_v3_prompt(off.prompt) and not HEADER_RE.search(off.prompt)
    monkeypatch.setenv("FINANCE_ANSWER_V3", "1")
    for q in (query, TREND_Q):
        route = ask_flow.finance_route(ask_flow.AskRequest(query=q))
        years = [h.metadata["year"] for h in route.hits]
        assert route.route == "reader" and fa.is_v3_prompt(route.prompt) and years == sorted(years)
        headers = re.findall(r"^LETTER OF (\d{4})", route.prompt, re.M)
        assert [int(y) for y in headers] == sorted(set(years))
        assert fa.TREND_SENTENCE not in route.prompt and fa.READER_INSTRUCTIONS not in route.prompt
        assert route.debug["prompt_tokens"] <= route.debug["window_tokens"]


def test_the_plan_premise_note_reaches_the_v3_prompt(env, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FINANCE_ANSWER_V3", "1")
    planned = ask_flow._plan_with_history
    monkeypatch.setattr(ask_flow, "_plan_with_history", lambda *a: replace(planned(*a), premise_note=NOTE))
    route = ask_flow.finance_route(ask_flow.AskRequest(query=TREND_Q))
    assert fa.prompt_premise(route.prompt) == NOTE and route.debug["premise_note"] == NOTE


def test_the_v3_prompt_is_trimmed_to_the_token_window_at_any_budget(env, monkeypatch):  # noqa: F811
    for name in ("FINANCE_K_TREND", "FINANCE_K_SPECIFIC", "FINANCE_K_PERIOD", "FINANCE_K_CORPUS"):
        monkeypatch.setenv(name, "0")  # today's budget: the adaptive-budget trim does not run
    counted = []

    def count(text):
        counted.append(text)
        return len(text) // 4, "chars"
    monkeypatch.setattr(ask_flow, "count_tokens_and_source", count)
    monkeypatch.setenv("FINANCE_ANSWER_V3", "1")
    full = ask_flow.finance_route(ask_flow.AskRequest(query=TREND_Q))
    assert full.debug["prompt_tokens"] == len(full.prompt) // 4 and "window_trimmed_v3" not in full.debug
    monkeypatch.setenv("LLM_N_CTX", str(len(full.prompt) // 4 - 1 + 900 + 64))  # one token short of the full prompt
    trimmed = ask_flow.finance_route(ask_flow.AskRequest(query=TREND_Q, max_new_tokens=900))
    assert len(trimmed.hits) < len(full.hits) and trimmed.debug["window_trimmed_v3"] == len(full.hits) - len(trimmed.hits)
    assert len(trimmed.prompt) // 4 <= trimmed.debug["window_tokens"] == len(full.prompt) // 4 - 1
    assert {h.id for h in trimmed.hits} < {h.id for h in full.hits}
    monkeypatch.delenv("FINANCE_ANSWER_V3")
    counted.clear()
    off = ask_flow.finance_route(ask_flow.AskRequest(query=TREND_Q))
    assert len(off.hits) == len(full.hits) and counted == []  # the flag off: no token count at today's budget


def test_the_finance_route_flags_the_served_passages_that_carry_the_plan_topic(env, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FINANCE_ANSWER_V3", "1")
    route = ask_flow.finance_route(ask_flow.AskRequest(query=CONVERTIBLE_Q))
    store = ask_flow._state["knowledge"]
    assert route.plan.themes == ("preferred_convertible_investments",)
    flagged = {h.id for h in route.hits if h.metadata.get(fa.TOPIC_FLAG)}
    expected = {h.id for h in route.hits if any(
        "preferred_convertible_investments" in ((store.annotation(pid) or {}).get("themes") or ())
        for pid in (h.id, *(h.metadata.get("merged_ids") or ())))}
    assert flagged == expected and "1990_p0100" in flagged
    monkeypatch.delenv("FINANCE_ANSWER_V3")
    assert not any(h.metadata.get(fa.TOPIC_FLAG) for h in ask_flow.finance_route(
        ask_flow.AskRequest(query=CONVERTIBLE_Q)).hits)


def test_ask_and_stream_return_the_same_deduped_v3_answer(env, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FINANCE_ANSWER_V3", "1")
    hits = ask_flow.finance_route(ask_flow.AskRequest(query=CONVERTIBLE_Q)).hits
    sentence = next(s for s in evidence_sentences(hits[0].text) if len(s.split()) >= 8)
    env.llm.reply = f"{sentence} [1] {sentence} [1]"
    body = _ask(env, CONVERTIBLE_Q)
    events, _ = _stream(env, CONVERTIBLE_Q)
    assert body["answer"] == events["done"]["answer"] and body["answer"].count(sentence) == 1
    assert all(fa.is_v3_prompt(p) for p in env.llm.prompts)


def test_ask_and_stream_retry_a_refused_premise_question_once(env, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FINANCE_ANSWER_V3", "1")
    monkeypatch.setenv("FINANCE_DEBUG", "1")
    planned = ask_flow._plan_with_history
    monkeypatch.setattr(ask_flow, "_plan_with_history", lambda *a: replace(planned(*a), premise_note=NOTE))
    hits = ask_flow.finance_route(ask_flow.AskRequest(query=CONVERTIBLE_Q)).hits
    sentence = next(s for s in evidence_sentences(hits[0].text) if len(s.split()) >= 8)
    llm = _Replies(REFUSAL_LINE, f"{sentence} [1]", REFUSAL_LINE, f"{sentence} [1]")
    ask_flow._state["llm"] = llm
    body = _ask(env, CONVERTIBLE_Q)
    events, _ = _stream(env, CONVERTIBLE_Q)
    assert len(llm.prompts) == 4 and [fa.is_v3_prompt(p) for p in llm.prompts] == [True, False, True, False]
    assert body["answer"] == events["done"]["answer"] and body["answer"].startswith(f"{NOTE} {sentence} [1]")
    assert body["debug"]["refusal_retry"] is True and body["debug"]["first_raw_answer"] == REFUSAL_LINE
    assert "refusal_by" not in body["debug"]
