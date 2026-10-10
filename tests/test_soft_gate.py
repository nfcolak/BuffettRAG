from types import SimpleNamespace

import pytest

from src.generation.prompt import REFUSAL_LINE
from src.services import ask_flow


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, max_new_tokens=0):
        self.calls += 1
        return REFUSAL_LINE


def _run(monkeypatch, overlap, flag):
    if flag:
        monkeypatch.setenv("EVIDENCE_GATE_SOFT", "1")
    else:
        monkeypatch.delenv("EVIDENCE_GATE_SOFT", raising=False)
    monkeypatch.setattr(ask_flow, "is_grounded", lambda llm: False)
    monkeypatch.setattr(
        ask_flow, "assess_evidence",
        lambda *a, **k: SimpleNamespace(sufficient=overlap >= 0.30, best_overlap=overlap),
    )
    llm = FakeLLM()
    answer, _ = ask_flow._generate_answer(llm, "p", [], 64, query="What does Buffett say about floats?")
    return llm.calls, answer


@pytest.mark.parametrize("overlap,flag,calls", [(0.27, False, 0), (0.27, True, 1), (0.10, True, 0)])
def test_soft_gate(monkeypatch, overlap, flag, calls):
    n, answer = _run(monkeypatch, overlap, flag)
    assert n == calls
    if calls == 0:
        assert answer == REFUSAL_LINE
