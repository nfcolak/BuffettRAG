"""Temporal comparison path fixes: passage cap, period-focused gate, join, detector."""
from __future__ import annotations

from types import SimpleNamespace

from src.generation.compare import (
    generate_comparison_answer, period_focused_query, prepare_answer_context,
)
from src.retrieval.context import build_doc_lookup
from src.retrieval.retriever import detect_temporal_comparison
from src.storage import SearchHit, StoredDoc


def _doc(name, year, prev=None, nxt=None):
    meta = {"year": year, "source_file": f"letter-{year}"}
    if prev:
        meta["previous_chunk_id"] = prev
    if nxt:
        meta["next_chunk_id"] = nxt
    return StoredDoc(name, f"Passage {name} about insurance float.", meta)


def test_cap_keeps_fourth_and_fifth_anchor_before_neighbours():
    # Five anchors per period, each with a neighbour on both sides: 15 passages per period.
    docs, anchors = [], []
    for year in (1991, 2003):
        for i in range(5):
            a = f"{year}_a{i}"
            docs += [_doc(a, year, f"{a}_p", f"{a}_n"), _doc(f"{a}_p", year), _doc(f"{a}_n", year)]
            anchors.append(SearchHit(a, f"Passage {a}", {"year": year, "source_file": f"letter-{year}"}, 1.0))
    llm = SimpleNamespace(provider_name="llama")
    served = prepare_answer_context(
        llm, anchors, build_doc_lookup(docs), "How did float differ in 1991 and 2003?",
        history=[], followup=False, neighbors=1, max_chars=9000, max_new_tokens=300, n_ctx=8192,
        max_passages=8, passage_max_chars=1800,
    )
    ids = [h.id for h in served]
    for year in (1991, 2003):
        for i in range(5):
            assert f"{year}_a{i}" in ids
        assert len([h for h in served if h.metadata["year"] == year]) <= 12
    first_year = [h.id for h in served if h.metadata["year"] == 1991]
    assert first_year[:5] == [f"1991_a{i}" for i in range(5)]


def test_period_focused_query_drops_other_periods_sub_ask():
    query = ("What was the margin in 1991 for the retail unit, "
             "and how did the freight unit's backlog compare in 2003?")
    periods = detect_temporal_comparison(query)
    focused = period_focused_query(query, periods[0], periods)
    assert "retail" in focused and "freight" not in focused and "2003" not in focused
    # A shared stem with trailing years is never emptied.
    stem = "Compare 2008 versus 2020"
    assert period_focused_query(stem, {"year": 2020}, detect_temporal_comparison(stem))


def _hv30_style(monkeypatch, soft):
    question = ("What market-share figures did NetJets report in the 2002 and 2003 letters, "
                "and what sales or aircraft-value measure accompanied each figure?")
    low = "NetJets held a 38 percent share of fractional ownership programs. [1]"  # overlap in [0.15, 0.30)
    high = ("NetJets reported market share figures and the sales or aircraft-value measure "
            "accompanied each figure in 2003.")
    context = [SearchHit("a", low.replace(" [1]", ""), {"year": 2002, "source_file": "l"}, 1.0),
               SearchHit("b", high, {"year": 2003, "source_file": "l"}, 1.0)]

    class Llm:
        calls = 0

        def generate(self, prompt, max_new_tokens=None):
            Llm.calls += 1
            return low if "(focus: 2002)" in prompt else high + " [1]"
    if soft:
        monkeypatch.setenv("EVIDENCE_GATE_SOFT", "1")
    else:
        monkeypatch.delenv("EVIDENCE_GATE_SOFT", raising=False)
    return question, context, Llm


def test_hv30_style_gate_blocks_by_default_and_soft_gate_calls_model(monkeypatch):
    question, context, Llm = _hv30_style(monkeypatch, soft=False)
    hard = generate_comparison_answer(Llm(), question, context, max_new_tokens=100)
    assert "In 2002: The retrieved passages for this period do not cover" in hard.answer
    question, context, Llm = _hv30_style(monkeypatch, soft=True)
    soft = generate_comparison_answer(Llm(), question, context, max_new_tokens=100)
    assert "In 2002: NetJets held a 38 percent share of fractional ownership programs." in soft.answer
    assert Llm.calls == 2


def test_soft_gate_still_refuses_when_model_refuses(monkeypatch):
    from src.generation.prompt import REFUSAL_LINE
    question, context, _ = _hv30_style(monkeypatch, soft=True)

    class Refuses:
        def generate(self, prompt, max_new_tokens=None):
            return REFUSAL_LINE
    result = generate_comparison_answer(Refuses(), question, context, max_new_tokens=100)
    assert "In 2002: The retrieved passages for this period do not cover" in result.answer


def test_join_keeps_quote_sentence_and_next_sentence_separate():
    early = ('We stayed because "textile jobs matter to the community" and we hoped for returns. '
             'In July we decided to close our textile operation.')
    context = [SearchHit("e", early, {"year": 1985, "source_file": "l"}, 1.0),
               SearchHit("o", "Other textile operation passage for the older year.", {"year": 1977, "source_file": "l"}, 1.0)]
    raw = ('We stayed because "textile jobs matter to the community" [1]\n\n'
           'In July we decided to close our textile operation. [1]')

    class Llm:
        def generate(self, prompt, max_new_tokens=None):
            return raw if "(focus: 1985)" in prompt else "Other textile operation passage for the older year. [1]"
    query = "Compare the textile operation in 1977 and 1985: what favored staying, and what shutdown followed?"
    answer = generate_comparison_answer(Llm(), query, context, max_new_tokens=100).answer
    paragraph = answer.split("\n\n")[1]
    lines = paragraph.splitlines()
    assert len(lines) == 2 and lines[0].endswith('community" [1]') and lines[1].startswith("In July")
    # Same break even when the model wrote both sentences on one line.
    one_line = ('We stayed because "textile jobs matter to the community" [1] '
                'In July we decided to close our textile operation. [1]')
    from src.generation.compare import _join_part
    assert _join_part(one_line).count("\n") == 1


def test_detector_ignores_event_year_without_cue_and_out_of_range_years():
    # No comparison cue and no "YYYY letter": two incidental years are not two periods.
    assert detect_temporal_comparison("Who took charge of Star Furniture in 1997 after the 1962 sale?") is None
    assert detect_temporal_comparison("How much did it pay in 1995, and what did he expect in 2011?") is None
    # Out-of-range year is dropped even with a cue.
    assert detect_temporal_comparison("Compare Berkshire's combined ratio in 1965 and 1985.") is None
    assert detect_temporal_comparison("How does 2014 compare with 2033?") is None
    # "YYYY letters" adjacency is enough without a cue; cues work too.
    assert detect_temporal_comparison("What did NetJets report in the 2002 and 2003 letters?") == [
        {"year": 2002}, {"year": 2003}]
    assert detect_temporal_comparison("How did GEICO change between 1986 and 2016?") == [
        {"year": 1986}, {"year": 2016}]


def test_detector_skips_followup_whose_history_pins_a_letter():
    history = [{"role": "user", "content": "What did Buffett say about Coca-Cola in the 2010 letter?"}]
    query = "Compare what it paid in 1995 versus what he expected in 2011?"
    assert detect_temporal_comparison(query) == [{"year": 1995}, {"year": 2011}]
    assert detect_temporal_comparison(query, history) is None
