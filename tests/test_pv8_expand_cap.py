"""pv8 EXP-04: the model-answer cap and the expanded-answer cap are separate; the expanded cap can be raised."""
from src.generation.answer_expand import expand_answer
from src.storage import SearchHit

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def _passage(n):
    # 7 prose sentences per passage, the answer quotes the middle one (index 3)
    return " ".join(f"Passage {n} sentence {i} says that the {WORDS[(n + i) % 8]} business earned good money for owners." for i in range(7))


def _hits(parts=5):
    return [SearchHit(f"p{n}", _passage(n), {"year": 2000 + n}, 0.0) for n in range(parts)]


def _answer(parts=5):
    return "\n".join(f"Passage {n} sentence 3 says that the {WORDS[(n + 3) % 8]} business earned good money for owners. [{n + 1}]"
                     for n in range(parts))


def _lines(text):
    return [line for line in text.split("\n") if line.strip()]


def _sentences(line):
    return line.count("sentence ")


def test_default_cap_starves_later_parts():
    out = _lines(expand_answer(_answer(), _hits(), after=3, before=3))
    assert sum(_sentences(l) for l in out) <= 12
    assert _sentences(out[-1]) == 1  # the last part got nothing: this is the starvation


def test_raised_expanded_cap_expands_every_part_without_touching_the_model_cap(monkeypatch):
    monkeypatch.setenv("ANSWER_MAX_SENTENCES", "12")
    monkeypatch.setenv("ANSWER_EXPANDED_MAX_SENTENCES", "40")
    out = _lines(expand_answer(_answer(), _hits(), after=3, before=3))
    assert all(_sentences(l) == 7 for l in out)


def test_model_answer_cap_gates_expansion_before_it_starts(monkeypatch):
    monkeypatch.setenv("ANSWER_MAX_SENTENCES", "5")  # the 5-sentence model answer already fills the model cap
    monkeypatch.setenv("ANSWER_EXPANDED_MAX_SENTENCES", "40")
    answer = _answer()
    assert expand_answer(answer, _hits(), after=3, before=3) == answer


def test_default_expanded_cap_is_the_current_twelve(monkeypatch):
    monkeypatch.delenv("ANSWER_EXPANDED_MAX_SENTENCES", raising=False)
    monkeypatch.delenv("ANSWER_MAX_SENTENCES", raising=False)
    import config
    assert config.ANSWER_EXPANDED_MAX_SENTENCES == 12 == config.ANSWER_MAX_SENTENCES


def test_zero_expanded_cap_scales_with_the_answer(monkeypatch):
    monkeypatch.setenv("ANSWER_EXPANDED_MAX_SENTENCES", "0")
    out = _lines(expand_answer(_answer(), _hits(), after=3, before=3))
    assert sum(_sentences(l) for l in out) == 20  # max(12, 4 x 5 raw sentences)
