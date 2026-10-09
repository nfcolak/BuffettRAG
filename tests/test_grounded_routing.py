"""Service routing for provider `grounded`, with a fake provider (no models, no engine)."""
from __future__ import annotations

import json
import sys
import threading
import types
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.generation.prompt import REFUSAL_LINE
from src.generation.providers.grounded_provider import GroundedProvider, ProviderUnavailable
from src.retrieval.retriever import RetrievalResult
from src.services import ask_flow, backend_app as backend
from src.storage import SearchHit

EARLY = "Insurance float was free when underwriting results broke even."
LATE = "Insurance float had a negative cost when underwriting produced a profit."
HITS = [SearchHit("early", EARLY, {"year": 2041, "source_file": "letter-2041"}, 1.0),
        SearchHit("late", LATE, {"year": 2053, "source_file": "letter-2053"}, 0.9)]
HISTORY = [{"role": "user", "content": "Discuss insurance float costs in 2041."},
           {"role": "assistant", "content": "It was free."}]
CIT = [{"index": 1, "id": "early", "years": [2041]}]


class FakeGrounded:
    provider_name = "grounded"
    model = "grounded:fake"

    def __init__(self, answer=None, citations=(), error=None):
        self.answer, self.citations, self.error = answer, list(citations), error
        self.calls = []

    def answer_grounded(self, query, context_hits, *, history=(), max_new_tokens=200):
        self.calls.append((query, [h.id for h in context_hits], list(history), max_new_tokens))
        if self.error:
            raise self.error
        return SimpleNamespace(answer=self.answer, citations=self.citations)

    def generate(self, prompt, max_new_tokens=None):  # pragma: no cover - must never be reached
        raise AssertionError("old generation path used for grounded")


class StubProvider:
    """Stand-in for an ordinary (non-grounded) provider; generation must not be reached."""
    provider_name = "llama"
    model = "stub"

    def generate(self, prompt, max_new_tokens=None):  # pragma: no cover
        raise AssertionError("generation must not be reached")


class Fixed:
    def __init__(self, hits):
        self.hits, self.queries, self.reranker = hits, [], object()

    def search(self, *args, **kwargs):
        self.queries.append(args[0] if args else kwargs["query"])
        return RetrievalResult(self.queries[-1], "hybrid", list(self.hits))


def run_routes(monkeypatch, llm, hits, payload):
    retriever = Fixed(hits)
    monkeypatch.setattr(ask_flow, "_state", {"retriever": retriever, "llm": llm, "docs_by_id": {}})
    monkeypatch.setattr(backend, "API_KEYS", ())
    client = TestClient(backend.app)
    payload = {"expand_query": False, **payload}
    ordinary = client.post("/ask", json=payload).json()
    stream = client.post("/ask/stream", json=payload)
    lines = [line for line in stream.text.splitlines() if line.startswith(("event: ", "data: "))]
    events = [(lines[i][7:], json.loads(lines[i + 1][6:])) for i in range(0, len(lines), 2)]
    return ordinary, events, retriever


CASES = {
    "answer": dict(answer=f"{EARLY} [1]", citations=CIT, hits=HITS, payload={"query": "insurance float cost?"}),
    "partial": dict(answer=f"In the 2041 letter: {EARLY} [1]\n\nThe retrieved passages do not cover 2053.",
                    citations=CIT, hits=HITS[:1], payload={"query": "insurance float cost?"}),
    "refuse": dict(answer=REFUSAL_LINE, citations=[], hits=HITS, payload={"query": "moon cheese?"}),
    "no-hit": dict(answer=REFUSAL_LINE, citations=[], hits=[], payload={"query": "insurance float cost?"}),
    "follow-up": dict(answer=f"{EARLY} [1]", citations=CIT, hits=HITS,
                      payload={"query": "What about that?", "history": HISTORY}),
    "comparison": dict(answer=f"In the 2041 letter: {EARLY} [1]\n\nIn the 2053 letter: {LATE} [2]",
                       citations=CIT, hits=HITS,
                       payload={"query": "How did insurance float costs differ in 2041 and 2053?"}),
}


@pytest.mark.parametrize("name", CASES)
def test_ask_and_stream_agree_for_grounded(monkeypatch, name):
    case = CASES[name]
    llm = FakeGrounded(case["answer"], case["citations"])
    ordinary, events, retriever = run_routes(monkeypatch, llm, case["hits"], case["payload"])
    stages = [data["stage"] for kind, data in events if kind == "status"]
    assert stages == ["retrieving", "generating", "validating"]
    assert events[-1][0] == "done"
    done = events[-1][1]
    assert done == {"answer": case["answer"], "citations": case["citations"]}
    assert ordinary["answer"] == done["answer"] and ordinary["citations"] == done["citations"]
    # The engine is called exactly once per route, with the ORIGINAL query, the client hits
    # and the history, even with no hits and for comparisons (no gate, no compare path).
    expected = (case["payload"]["query"], [h.id for h in case["hits"]],
                case["payload"].get("history", []), 900)
    assert llm.calls == [expected, expected]
    meta = dict(events)["meta"]
    assert [h["id"] for h in meta["hits"]] == [h["id"] for h in ordinary["hits"]]


def test_followup_retrieves_with_resolved_query_but_answers_original(monkeypatch):
    llm = FakeGrounded(f"{EARLY} [1]", CIT)
    _, _, retriever = run_routes(monkeypatch, llm, HITS, CASES["follow-up"]["payload"])
    assert "insurance" in retriever.queries[0] and retriever.queries[0] != "What about that?"
    assert llm.calls[0][0] == "What about that?"


def test_grounded_error_is_identical_on_both_routes(monkeypatch):
    llm = FakeGrounded(error=RuntimeError("secret detail"))
    ordinary, events, _ = run_routes(monkeypatch, llm, HITS, {"query": "insurance float cost?"})
    done = events[-1][1]
    assert ordinary["answer"] == done["answer"] == "[LLM unavailable: the embedded model failed to generate an answer]"
    assert ordinary["citations"] == done["citations"] == []
    assert "secret detail" not in done["answer"]


def test_old_providers_keep_no_hit_and_stream_behaviour(monkeypatch):
    ordinary, events, _ = run_routes(monkeypatch, StubProvider(), [], {"query": "insurance float cost?"})
    assert ordinary["answer"] is None and events[-1] == ("done", {"answer": None, "citations": []})
    assert [d["stage"] for k, d in events if k == "status"] == ["retrieving"]
    refused, events, _ = run_routes(monkeypatch, StubProvider(), HITS, {"query": "moon cheese recipes?"})
    assert refused["answer"] == REFUSAL_LINE and events[-1][1]["answer"] == REFUSAL_LINE


def test_pipeline_rejects_grounded():
    from src.pipeline import BuffettRAGPipeline

    retriever = Fixed(HITS)
    with pytest.raises(ValueError, match="grounded is served by the backend and the benchmark runner"):
        BuffettRAGPipeline(retriever, {}, FakeGrounded("x")).ask("insurance float cost?")
    assert retriever.queries == []


def test_attach_resources_only_for_grounded():
    attached = []
    llm = FakeGrounded("x")
    llm.attach_resources = lambda *, reranker: attached.append(reranker)
    retriever = Fixed(HITS)
    ask_flow.attach_grounded_resources(llm, retriever)
    ask_flow.attach_grounded_resources(StubProvider(), retriever)
    assert attached == [retriever.reranker]


def test_factory_builds_lazy_grounded_and_mlx(monkeypatch):
    from src.generation.providers import create_llm_provider

    provider = create_llm_provider("grounded")
    assert provider.provider_name == "grounded" and provider._engine is None and provider._resources is None
    fake = types.ModuleType("src.generation.providers.mlx_provider")
    fake.MlxProvider = lambda: SimpleNamespace(provider_name="mlx")
    monkeypatch.setitem(sys.modules, "src.generation.providers.mlx_provider", fake)
    assert create_llm_provider("mlx").provider_name == "mlx"
    with pytest.raises(ValueError):
        create_llm_provider("nonsense")


def test_provider_template_generate_unavailable_and_model_path_override():
    from src.generation.grounded.settings import GroundedSettings

    template = GroundedProvider(GroundedSettings(), composer_kind="template")
    with pytest.raises(ProviderUnavailable):
        template.generate("expand this")
    gguf = GroundedProvider(GroundedSettings(), composer_kind="llama", model_path="/tmp/x/model.gguf")
    assert str(gguf.settings.LLAMA_MODEL).endswith("/tmp/x/model.gguf")
    assert gguf.model == "grounded:llama:model.gguf"
    with pytest.raises(ValueError):
        GroundedProvider(GroundedSettings(), composer_kind="template", model_path="/tmp/x")


def test_provider_builds_one_engine_and_attaches_reranker(monkeypatch):
    from src.generation.grounded.settings import GroundedSettings

    built, attached = [], []

    class Resources:
        def attach_resources(self, *, reranker):
            attached.append(reranker)

    class Engine:
        def __init__(self, resources, settings, *, composer_kind=None):
            built.append((resources, composer_kind))

        def answer(self, query, hits, *, history=(), max_new_tokens=200):
            return SimpleNamespace(answer=f"{query}|{len(hits)}|{len(history)}|{max_new_tokens}", citations=[])

    shared = Resources()
    resources_mod = types.ModuleType("src.generation.grounded.resources")
    resources_mod.get_resources = lambda settings: shared
    engine_mod = types.ModuleType("src.generation.grounded.engine")
    engine_mod.GroundedEngine = Engine
    monkeypatch.setitem(sys.modules, "src.generation.grounded.resources", resources_mod)
    monkeypatch.setitem(sys.modules, "src.generation.grounded.engine", engine_mod)

    provider = GroundedProvider(GroundedSettings(), composer_kind="template")
    reranker = object()
    provider.attach_resources(reranker=reranker)
    outputs = []
    threads = [threading.Thread(target=lambda: outputs.append(
        provider.answer_grounded("q", HITS, history=HISTORY, max_new_tokens=50).answer)) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert outputs == ["q|2|2|50"] * 4
    assert built == [(shared, "template")] and attached == [reranker]


def test_config_grounded_settings_read_environment_over_defaults(monkeypatch):
    import config

    monkeypatch.setitem(config.GROUNDED_DEFAULTS, "GROUNDED_MAX_UNITS", "5")
    assert config.load_grounded_settings({}).MAX_UNITS == 5
    assert config.load_grounded_settings({"GROUNDED_MAX_UNITS": "7"}).MAX_UNITS == 7
    assert config.DEFAULT_LLM_PROVIDER != "grounded"
