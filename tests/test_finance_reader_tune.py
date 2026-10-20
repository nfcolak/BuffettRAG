"""finance-reader-tune group 2: the finance reader prompt, the validator's reader options and the reader post-processing."""
from __future__ import annotations

import re

import pytest

from src.evaluation.claim_validator import evidence_sentences, validate_and_filter_answer
from src.generation import finance_answer as fa
from src.generation.evidence_gate import EvidenceAssessment
from src.generation.prompt import REFUSAL_LINE, SYSTEM_PROMPT, build_cited_prompt, format_answer_markdown
from src.generation.providers.llama_provider import split_prompt
from src.knowledge import KnowledgeStore
from src.services import ask_flow
from src.storage import SearchHit
from test_ask_flow_finance import TREND_Q, _ask, _stream, env  # noqa: F401  (env is a fixture)

FLOAT = "Insurance float grew steadily over the decade. Our underwriting stayed disciplined through the cycle."
SEES = ("We bought See's for $25 million when its sales were $30 million. "
        "The capital then required to conduct the business was $8 million. "
        "See's later earned far more than the business ever needed in new capital.")
BOUGHT = "We bought See's for $25 million when its sales were $30 million."


def _hits(*texts):
    return [SearchHit(f"199{i}_p000{i}", t, {"year": 1990 + i, "source_file": f"199{i}.pdf"}, 1.0)
            for i, t in enumerate(texts)]


@pytest.fixture
def prompt_env(monkeypatch):
    for name in ("PROMPT_VERSION", "LLM_MODEL_PATH", "ANSWER_REFUSAL_RETRY", "EVIDENCE_GATE_SOFT",
                 "ANSWER_NEIGHBOR_AFTER", "ANSWER_NEIGHBOR_BEFORE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ask_flow, "_state", {"knowledge": KnowledgeStore()})


# --- prompt ---
def test_reader_prompt_keeps_the_trained_format_and_only_appends_the_instructions(prompt_env):
    hits = _hits(FLOAT, SEES)
    plain = build_cited_prompt("What did See's cost?", hits)
    reader = fa.reader_prompt("What did See's cost?", hits)
    assert reader.startswith(SYSTEM_PROMPT) and reader.endswith("\n\nAnswer:")
    assert reader.replace(" " + fa.READER_INSTRUCTIONS, "") == plain
    assert split_prompt(reader)[0] == split_prompt(plain)[0] == SYSTEM_PROMPT  # the system message is unchanged
    assert reader.index("END USER QUESTION") < reader.index(fa.READER_INSTRUCTIONS)
    trend = fa.reader_prompt("How did it change?", hits, trend=True)
    assert trend.replace(" " + fa.READER_INSTRUCTIONS, "") == fa.trend_prompt("How did it change?", hits)
    assert trend.index(fa.TREND_SENTENCE) < trend.index(fa.READER_INSTRUCTIONS)


def test_reader_instructions_ask_for_a_fuller_answer_and_keep_the_refusal_clause():
    text = fa.READER_INSTRUCTIONS
    for asked in ("every passage", "reasons", "figures", "several sentences", "citation"):
        assert asked in text
    assert "only part of the question, answer that part" in text and "which part they do not cover" in text
    assert "refusal line only when no passage answers any part of the question" in text
    assert "only shares a topic word with the question does not answer it" in text
    assert REFUSAL_LINE not in text  # the system prompt defines the line; the instructions only refer to it


def test_is_reader_prompt_recognises_only_the_reader_tail(prompt_env):
    hits = _hits(FLOAT, SEES)
    reader = fa.reader_prompt("What did See's cost?", hits)
    assert fa.is_reader_prompt(reader)
    assert fa.is_reader_prompt(fa.figures_prompt(reader, [{"source_id": hits[1].id, "metric": "price", "year": 1972,
                                                          "value": 25, "unit": "usd_millions", "as_written": "$25"}],
                                                  hits))
    assert not fa.is_reader_prompt(build_cited_prompt("What did See's cost?", hits))
    assert not fa.is_reader_prompt(fa.trend_prompt("How did it change?", hits))
    assert not fa.is_reader_prompt(build_cited_prompt(f"Q? {fa.READER_INSTRUCTIONS}", hits))  # the question cannot
    assert not fa.is_reader_prompt("")


# --- validator reader options ---
def test_a_true_sentence_with_the_wrong_marker_is_repaired_to_the_passage_that_supports_it():
    hits = _hits(FLOAT, SEES)
    sentence = f"{BOUGHT} [1]"
    assert validate_and_filter_answer(sentence, hits).safe_answer == ""  # default: dropped as today
    result = validate_and_filter_answer(sentence, hits, **fa.READER_VALIDATION)
    assert result.safe_answer == f"{BOUGHT} [2]" and not result.blocked_claims
    assert [s["reason"] for s in result.sentences] == ["citation_repaired"]
    paraphrase = "See's was bought for $25 million when its sales were $30 million. [1]"
    assert validate_and_filter_answer(paraphrase, hits, **fa.READER_VALIDATION).safe_answer.endswith("[2]")


def test_an_uncited_sentence_is_repaired_only_when_it_is_a_passage_sentence():
    hits = _hits(FLOAT, SEES)
    assert validate_and_filter_answer(BOUGHT, hits, **fa.READER_VALIDATION).safe_answer == f"{BOUGHT} [2]"
    paraphrase = "See's was bought for $25 million when its sales were $30 million."
    assert validate_and_filter_answer(paraphrase, hits, **fa.READER_VALIDATION).safe_answer == ""


@pytest.mark.parametrize("sentence", [
    "We bought See's for $35 million when its sales were $30 million. [1]",  # a figure no passage states
    "We bought See's for $25 million when its sales were $90 million. [2]",
    "Zebras graze peacefully across the open savanna every spring. [2]",       # supported by no passage
    "We did not buy See's for $25 million when its sales were $30 million. [1]",  # polarity
])
def test_unsupported_numbers_and_unsupported_sentences_are_still_dropped(sentence):
    result = validate_and_filter_answer(sentence, _hits(FLOAT, SEES), **fa.READER_VALIDATION)
    assert result.safe_answer == "" and result.sentences[0]["reason"] == "unsupported_claim"


def test_a_gap_sentence_is_kept_beside_an_answer_without_markers():
    hits = _hits(FLOAT, SEES)
    answer = f"{BOUGHT} [2] The passages do not cover how the 2016 scandal changed his view. [1]"
    result = validate_and_filter_answer(answer, hits, **fa.READER_VALIDATION)
    assert result.safe_answer == f"{BOUGHT} [2] The passages do not cover how the 2016 scandal changed his view."
    assert [s["reason"] for s in result.sentences] == ["supported", "gap_line"]
    assert validate_and_filter_answer(answer, hits).safe_answer == f"{BOUGHT} [2]"  # default: dropped as today


def test_a_gap_sentence_alone_is_no_answer_and_the_refusal_line_beside_an_answer_is_dropped():
    hits = _hits(FLOAT, SEES)
    alone = validate_and_filter_answer(
        "Zebras graze peacefully across the open savanna every spring. [2]\nThe passages do not cover the rest.",
        hits, **fa.READER_VALIDATION)
    assert alone.safe_answer == "" and alone.sentences[-1]["reason"] == "gap_line_without_answer"
    beside = validate_and_filter_answer(f"{BOUGHT} [2]\n{REFUSAL_LINE}", hits, **fa.READER_VALIDATION)
    assert beside.safe_answer == f"{BOUGHT} [2]"


@pytest.mark.parametrize("text,gap", [
    ("The passages do not cover how the oil embargo hit the U.S. economy.", True),
    ("However, the retrieved passages do not say what Berkshire paid in 2024.", True),
    ("These letters don't mention robo-advisors.", True),
    ("There is no information about the 1997 Asian crisis in the passages.", True),
    (REFUSAL_LINE, False),
    ("The passages do not say why, but Berkshire sold half of its shares.", False),  # a second clause states a fact
    ("The passages do not say whether the stake cost $5 billion.", False),           # a money figure
    ("Berkshire did not pay a dividend.", False),
])
def test_is_gap_line(text, gap):
    assert fa.is_gap_line(text) is gap


def test_a_contentless_sentence_is_dropped_with_the_reader_options():
    hits = _hits(FLOAT, SEES)
    assert validate_and_filter_answer("000. [2]", hits).safe_answer == "000. [2]"  # today: kept, nothing to check
    result = validate_and_filter_answer("000. [2]", hits, **fa.READER_VALIDATION)
    assert result.safe_answer == "" and result.sentences[0]["reason"] == "no_content"


# --- formatting, finalize, retry ---
def test_reader_formatting_keeps_a_number_with_thousands_separators_whole():
    text = "We bought 80% of Borsheim's early in 1989. [1] Omaha's population is about 600,000. [2]"
    split = format_answer_markdown(text)  # the list split the reader formatting avoids
    assert "about 600\n" in split and "\n000. [2]" in split
    assert fa.format_reader_answer(text) == text
    listing = "Three lessons: 1. Price matters, 2. Moats matter, 3. Managers matter. [1]"
    assert fa.format_reader_answer(listing) == format_answer_markdown(listing)


def test_finalize_repairs_and_expands_a_reader_answer_and_leaves_the_legacy_prompt_unchanged(prompt_env):
    hits = _hits(FLOAT, SEES)
    raw = f"{BOUGHT} [1]"
    legacy, _ = ask_flow._finalize_answer(None, build_cited_prompt("What did See's cost?", hits), hits, raw, 900)
    assert legacy == REFUSAL_LINE
    trace = {}
    answer, citations = ask_flow._finalize_answer(None, fa.reader_prompt("What did See's cost?", hits), hits, raw,
                                                  900, trace)
    assert answer.startswith(f"{BOUGHT} [2] The capital then required to conduct the business was $8 million. [2]")
    assert answer.count("[2]") == 3 and "[1]" not in answer  # two following sentences of the same passage
    assert {c["passage_ids"][0] for c in citations} == {hits[1].id}
    assert trace["validator"]["repaired"] == [{"sentence": raw, "repaired": f"{BOUGHT} [2]"}]


def test_finalize_does_not_expand_inside_a_passage_with_personal_facts(prompt_env):
    hits = _hits(FLOAT, SEES)
    ask_flow._state["knowledge"] = KnowledgeStore(annotations=[{"id": hits[1].id, "has_personal": True}])
    answer, _ = ask_flow._finalize_answer(None, fa.reader_prompt("What did See's cost?", hits), hits,
                                          f"{BOUGHT} [2]", 900)
    assert answer == f"{BOUGHT} [2]"
    assert [h.text for h in fa.expansion_hits(hits, ask_flow._state["knowledge"])] == [FLOAT, ""]


def test_finalize_keeps_the_borsheims_population_sentence_whole(prompt_env):
    passage = ("It was a poor year for retailing, but sales at Borsheim's were up 18% early in 1990. "
               "Borsheim's could not do nearly that well if our customers came only from the Omaha metropolitan "
               "area, whose population is about 600,000.")
    hits = _hits(FLOAT, passage)
    raw = ("Sales at Borsheim's were up 18% early in 1990. [2] Borsheim's could not do nearly that well if our "
           "customers came only from the Omaha metropolitan area, whose population is about 600,000. [2]")
    legacy, _ = ask_flow._finalize_answer(None, build_cited_prompt("Q?", hits), hits, raw, 900)
    reader, _ = ask_flow._finalize_answer(None, fa.reader_prompt("Q?", hits), hits, raw, 900)
    assert "600,000" not in legacy
    assert reader.endswith("whose population is about 600,000. [2]")


class _Refuses:
    provider_name = "remote"
    model = "fake"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, max_new_tokens=None):
        self.prompts.append(prompt)
        return REFUSAL_LINE


def test_the_reader_route_has_no_refusal_retry_and_the_legacy_path_keeps_it(prompt_env, monkeypatch):
    monkeypatch.setattr(ask_flow, "assess_evidence", lambda *a, **k: EvidenceAssessment(False, "weak", 0.2))
    hits = _hits(FLOAT, SEES, FLOAT + " Again.", SEES + " Again.")
    legacy = _Refuses()
    ask_flow._generate_answer(legacy, build_cited_prompt("Q?", hits), hits, 900, query="Q?")
    assert len(legacy.prompts) == 2  # soft gate + refusal: one retry with the top 3 passages, as before
    reader = _Refuses()
    trace = {}
    answer, _ = ask_flow._generate_answer(reader, fa.reader_prompt("Q?", hits), hits, 900, query="Q?",
                                          per_period=False, trace=trace)
    assert answer == REFUSAL_LINE and len(reader.prompts) == 1 and trace["refusal_retry"] is False


# --- the finance route ---
def test_the_finance_route_builds_the_reader_prompt_for_trend_and_other_reader_questions(env):  # noqa: F811
    trend = ask_flow.finance_route(ask_flow.AskRequest(query=TREND_Q))
    assert trend.route == "reader" and trend.plan.kind == "trend" and fa.is_reader_prompt(trend.prompt)
    assert trend.prompt.count(fa.TREND_SENTENCE) == 1 and trend.prompt.count(fa.READER_INSTRUCTIONS) == 1
    other = ask_flow.finance_route(ask_flow.AskRequest(query="What did Buffett say about bonds and inflation?"))
    assert other.route == "reader" and other.plan.kind != "trend" and fa.is_reader_prompt(other.prompt)
    assert fa.TREND_SENTENCE not in other.prompt


def test_ask_and_stream_return_the_same_repaired_reader_answer(env):  # noqa: F811
    query = "How did Buffett's view of convertible preferred stocks change over time?"  # passes the stream's gate
    hits = ask_flow.finance_route(ask_flow.AskRequest(query=query)).hits
    sentence = next(s for s in evidence_sentences(hits[1].text) if len(s.split()) >= 8)
    assert sentence not in hits[0].text
    env.llm.reply = f"{sentence} [1]"
    body = _ask(env, query)
    events, _ = _stream(env, query)
    assert body["answer"] == events["done"]["answer"]
    assert body["answer"].startswith(f"{sentence} [2]") and not re.search(r"\[1\]", body["answer"])
