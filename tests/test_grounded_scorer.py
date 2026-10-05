import sys
import threading
import types

import numpy as np

from src.generation.grounded import resources as res_mod
from src.generation.grounded.scorer import RerankerSentenceScorer
from src.generation.grounded.settings import GroundedSettings
from src.retrieval.reranker import CrossEncoderReranker
from src.storage import SearchHit


class StubModel:
    def __init__(self):
        self.activation_fn = "model-default"
        self.calls = []

    def predict(self, pairs, batch_size=32, show_progress_bar=False, **kw):
        self.calls.append(kw)
        logits = [(-4.0 if "irrelevant" in t else 3.0 - i) for i, (_, t) in enumerate(pairs)]
        return np.array(logits)


def make_reranker():
    r = CrossEncoderReranker.__new__(CrossEncoderReranker)
    r.model = StubModel()
    r.batch_size = 4
    return r


def test_score_pairs_identity_and_model_untouched(monkeypatch):
    seen = {}
    fake_torch = types.SimpleNamespace(nn=types.SimpleNamespace(Identity=lambda: seen.setdefault("id", object())))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    r = make_reranker()
    out = r.score_pairs("q", ["a", "irrelevant"])
    assert out == [3.0, -4.0]
    assert r.model.calls[0]["activation_fn"] is seen["id"]
    assert r.model.activation_fn == "model-default"


def test_irrelevant_probability_and_cache(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(nn=types.SimpleNamespace(Identity=object)))
    r = make_reranker()
    s = RerankerSentenceScorer(r)
    p = s.score("q", ["irrelevant"])
    assert 0 <= p[0] < 0.5
    s.score("q", ["irrelevant"])
    assert len(r.model.calls) == 1


def test_length_and_nonfinite_rejected():
    class Bad:
        def __init__(self, v):
            self.v = v

        def score_pairs(self, q, t):
            return self.v

    for v in ([0.0], [float("nan"), 0.0]):
        try:
            RerankerSentenceScorer(Bad(v)).score("q", ["a", "b"])
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_registry_single_instance_concurrent():
    res_mod.reset_resources()
    st = GroundedSettings()
    got = []
    barrier = threading.Barrier(4)

    def run():
        barrier.wait()
        got.append(res_mod.get_resources(st))

    ts = [threading.Thread(target=run) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len({id(x) for x in got}) == 1
    assert res_mod.rerank_lock() is got[0].locks["scorer"]


def test_get_composer_cached(monkeypatch):
    made = []

    class T:
        pass

    mod = types.ModuleType("src.generation.grounded.template")
    mod.TemplateComposer = lambda: made.append(1) or T()
    monkeypatch.setitem(sys.modules, "src.generation.grounded.template", mod)
    res_mod.reset_resources()
    r = res_mod.get_resources(GroundedSettings(COMPOSER="template"))
    assert r.get_composer("template") is r.get_composer("template")
    assert len(made) == 1
    assert r.describe()["reranker_loaded"] is False


def test_rerank_unchanged_on_stub():
    r = make_reranker()
    hits = [SearchHit(id=str(i), text=t, metadata={}, score=0.0) for i, t in enumerate(["a", "irrelevant", "c"])]
    out = r.rerank("q", hits, top_k=2)
    assert [h.id for h in out] == ["0", "2"]
    assert out[0].score == 1.0 and abs(out[1].score - 5 / 7) < 1e-9
    assert r.model.calls == [{}]
