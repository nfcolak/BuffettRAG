from __future__ import annotations

from types import SimpleNamespace

from src.generation.decompose import answer_by_parts, split_question
from src.generation.prompt import REFUSAL_LINE
from src.services import ask_flow


DEV_QUESTIONS = [
    ("What were Berkshire's total operating earnings and operating earnings per share in 1977?", ["What were Berkshire's total operating earnings and operating earnings per share in 1977?"]),
    ("In the 1979 letter, could a business earning 20% on capital still give its owners a negative real return, and why?", ["In the 1979 letter, could a business earning 20% on capital still give its owners a negative real return?", "In the 1979 letter, why could a business earning 20% on capital still give its owners a negative real return?"]),
    ("What warning did Buffett give in 1982 about paying too high a purchase price for an excellent company's stock?", ["What warning did Buffett give in 1982 about paying too high a purchase price for an excellent company's stock?"]),
    ("How large was Berkshire's USAir preferred-stock purchase, and how did Buffett characterize his analytical mistake in the 1994 letter?", ["How large was Berkshire's USAir preferred-stock purchase?", "How did Buffett characterize his analytical mistake in the 1994 letter?"]),
    ("How much did Berkshire own in foreign exchange contracts at yearend 2004, and across how many currencies?", ["How much did Berkshire own in foreign exchange contracts at yearend 2004?", "Across how many currencies did Berkshire own in foreign exchange contracts at yearend 2004?"]),
    ("What percentage of ISCAR did Berkshire buy in 2006, what did it pay, and who retained the rest?", ["What percentage of ISCAR did Berkshire buy in 2006?", "What did Berkshire pay in 2006?", "Who retained the rest of ISCAR?"]),
    ("What did Berkshire originally pay for its PetroChina holding, and how much did it receive when selling it in 2007?", ["What did Berkshire originally pay for its PetroChina holding?", "How much did Berkshire receive when selling it in 2007?"]),
    ("What price did MidAmerican pay for NV Energy in 2013, and approximately what share of Nevada's population did it supply with electricity?", ["What price did MidAmerican pay for NV Energy in 2013?", "Approximately what share of Nevada's population did MidAmerican supply with electricity in 2013?"]),
    ("What was the name of that dealership group, and how many automobile dealerships did it have?", ["What was the name of that dealership group?", "How many automobile dealerships did that dealership group have?"]),
    ("What did Berkshire estimate its losses from the three 2017 hurricanes to be before and after tax?", ["What did Berkshire estimate its losses from the three 2017 hurricanes to be before and after tax?"]),
    ("What did Buffett identify as Berkshire's top priority when deploying retained earnings?", ["What did Buffett identify as Berkshire's top priority when deploying retained earnings?"]),
    ("How did Buffett distinguish Charlie Munger's role as architect from his own role at Berkshire?", ["How did Buffett distinguish Charlie Munger's role as architect from his own role at Berkshire?"]),
    ("Compare Berkshire's stance toward its textile operation in 1977 and 1985: what employment consideration favored staying, and what shutdown decision followed?", ["Compare Berkshire's stance toward its textile operation in 1977 and 1985: what employment consideration favored staying?", "What decision did Berkshire make about its textile operation in 1985?"]),
    ("What exact quantum error-correction code and qubit architecture does Buffett recommend for building a quantum computer?", ["What exact quantum error-correction code and qubit architecture does Buffett recommend for building a quantum computer?"]),
]


def test_split_question_on_real_dev_questions():
    for query, expected in DEV_QUESTIONS:
        assert split_question(query) == expected, query


def test_answer_by_parts_keeps_order_and_drops_refusal():
    context = [SimpleNamespace(
        id="p1",
        text="Berkshire earned $10 million. Buffett paid $5 million. Berkshire held 3 shares.",
        metadata={"year": 2000, "source_file": "letter.txt"},
    )]

    class FakeLLM:
        def __init__(self):
            self.answers = iter([
                "Berkshire earned $10 million [1].",
                REFUSAL_LINE,
                "Buffett paid $5 million [1].",
            ])
            self.prompts = []

        def generate(self, prompt, max_new_tokens=None):
            self.prompts.append(prompt)
            return next(self.answers)

    llm = FakeLLM()
    answer, citations = answer_by_parts(
        llm, "(1) What happened first? (2) What happened second? (3) What happened third?", context, [], 80
    )
    # Three independent question parts; the middle refusal contributes nothing.
    assert len(llm.prompts) == 3
    assert answer == "Berkshire earned $10 million [1]. Buffett paid $5 million [1]."
    assert citations and all(citation["marker"] == "[1]" for citation in citations)


def test_flag_off_generates_once(monkeypatch):
    monkeypatch.setenv("ANSWER_DECOMPOSE", "0")
    monkeypatch.setattr(ask_flow, "assess_evidence", lambda *args, **kwargs: SimpleNamespace(sufficient=True))
    monkeypatch.setattr(ask_flow, "_finalize_answer", lambda *args, **kwargs: ("ok", []))

    class FakeLLM:
        provider_name = "fake"

        def __init__(self):
            self.calls = 0

        def generate(self, prompt, max_new_tokens=None):
            self.calls += 1
            return "raw answer"

    llm = FakeLLM()
    answer, _ = ask_flow._generate_answer(
        llm, "What happened first, and what happened next?", [], 80,
        query="What happened first, and what happened next?",
    )
    assert answer == "ok"
    assert llm.calls == 1


def _join(monkeypatch, raws):
    from src.generation import decompose

    monkeypatch.setattr(
        decompose, "validate_and_filter_answer", lambda ans, hits: SimpleNamespace(safe_answer=ans)
    )

    class FakeLLM:
        def __init__(self):
            self.answers = iter(raws)

        def generate(self, prompt, max_new_tokens=None):
            return next(self.answers)

    hit = SimpleNamespace(id="p1", text="x", metadata={"year": 2000, "source_file": "l.txt"})
    return decompose.answer_by_parts(
        FakeLLM(), "(1) What is first? (2) What is second?", [hit, hit], [], 80
    )[0]


def test_join_keeps_markers_on_their_sentences(monkeypatch):
    answer = _join(monkeypatch, [
        "Alpha holds many shares today. [1] Beta paid some dividends then. [1]",
        "Gamma bought another company later [2].",
    ])
    assert answer == (
        "Alpha holds many shares today. [1] Beta paid some dividends then. [1] "
        "Gamma bought another company later [2]."
    )


def test_join_dedupes_exact_and_near_duplicates(monkeypatch):
    answer = _join(monkeypatch, [
        "Alpha holds many shares of the big company today [1].",
        "Alpha holds many shares of the big company today [2]. "
        "Alpha holds many shares of the big company today now [1].",
    ])
    assert answer.count("Alpha holds") == 1


def test_bare_marker_never_becomes_own_sentence(monkeypatch):
    from src.generation.decompose import _sentences

    assert _sentences("Alpha holds many shares today. [1]") == ["Alpha holds many shares today. [1]"]
    assert _sentences("[1]") == []
