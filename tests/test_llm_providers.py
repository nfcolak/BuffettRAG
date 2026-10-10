import json
from pathlib import Path
import importlib
import unittest
from unittest import mock

import pytest

import config
from src.generation.prompt import REFUSAL_LINE, build_cited_prompt
from src.generation.providers import LlamaCppProvider, UnavailableProvider, create_llm_provider
from src.generation.providers import factory
from src.storage import SearchHit


class LLMProviderTests(unittest.TestCase):
    def test_factory_returns_unavailable_provider_when_model_missing(self) -> None:
        with mock.patch.object(factory, "LLM_MODEL_PATH", config.BASE_DIR / "models" / "missing.gguf"):
            provider = create_llm_provider(provider="llama")
        self.assertIsInstance(provider, UnavailableProvider)
        self.assertEqual(provider.provider_name, "unavailable")
        with self.assertRaisesRegex(RuntimeError, "model file missing"):
            provider.generate("any prompt")

    def test_factory_returns_unavailable_provider_when_llama_cpp_missing(self) -> None:
        with mock.patch.object(factory, "LLM_MODEL_PATH", config.BASE_DIR / "config.py"), \
                mock.patch.object(factory.importlib.util, "find_spec", return_value=None):
            provider = create_llm_provider(provider="llama")
        self.assertIsInstance(provider, UnavailableProvider)
        with self.assertRaisesRegex(RuntimeError, "llama_cpp is not importable"):
            provider.generate("any prompt")

    def test_factory_rejects_removed_extractive_engine(self) -> None:
        for name in ("local", "extractive"):
            with self.assertRaisesRegex(ValueError, "extractive engine was removed"):
                create_llm_provider(provider=name)

    def test_default_model_path_is_7b_ft_r1(self) -> None:
        self.assertEqual(config.LLM_MODEL_PATH.name, "buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf")

    def test_factory_has_no_api_key_or_model_parameters(self) -> None:
        with self.assertRaises(TypeError):
            create_llm_provider(provider="llama", api_key="x")  # type: ignore[call-arg]

    def test_factory_rejects_external_and_unknown_providers(self) -> None:
        for name in ("openai", "anthropic", "openrouter", "unknown"):
            with self.assertRaisesRegex(ValueError, "Unsupported LLM provider"):
                create_llm_provider(provider=name)

    def test_external_provider_modules_are_gone(self) -> None:
        for name in ("openai_provider", "openrouter_provider", "anthropic_provider"):
            with self.assertRaises(ImportError):
                importlib.import_module(f"src.generation.providers.{name}")
        with self.assertRaises(ImportError):
            importlib.import_module("src.generation.openai_llm")

    def test_default_provider_is_llama(self) -> None:
        self.assertEqual(config.DEFAULT_LLM_PROVIDER, "llama")

@pytest.mark.skipif(not config.LLM_MODEL_PATH.is_file(), reason="GGUF model file not present")
def test_llama_provider_real_grounded_answer():
    hits = [
        SearchHit(id="a", score=1.0, metadata={"year": 2002, "source_file": "buffet_2002.txt"},
                  text="Derivatives are financial weapons of mass destruction, carrying dangers that, "
                       "while now latent, are potentially lethal."),
        SearchHit(id="b", score=0.9, metadata={"year": 1989, "source_file": "buffet_1989.txt"},
                  text="We made good progress on compounding rates this year."),
    ]
    provider = LlamaCppProvider()
    answer = provider.generate(build_cited_prompt("What did Buffett say about derivatives?", hits),
                               max_new_tokens=120)
    assert "[1]" in answer or REFUSAL_LINE in answer


def test_splitter_joins_numeric_fragments_and_preserves_honorifics():
    from src.generation.text_relevance import split_sentences
    text = "In 1998. 1999 the firm bought 2% of a supplier. Mr. Market is there to serve you, not to guide you."
    sentences = split_sentences(text)
    assert sentences == ["In 1998. 1999 the firm bought 2% of a supplier.",
                         "Mr. Market is there to serve you, not to guide you."]
    assert not any(s == "In 1998." or s[:1].isdigit() for s in sentences)
    from src.evaluation.claim_validator import validate_and_filter_answer
    hits = [SearchHit("p", text, {}, 1.0)]
    answer = "\n\n".join(s + " [1]" for s in sentences)
    assert validate_and_filter_answer(answer, hits).safe_answer == answer


def test_splitter_skips_salutations_headers_and_signatures():
    from src.generation.text_relevance import split_sentences
    text = ("To the Stockholders of Example Industries Inc.:\n\n"
            "Operating Earnings\n\nOperating earnings were $250 million this year.\n\n"
            "Warren E. Buffett\nChairman of the Board\nPage 12\nEXAMPLE INDUSTRIES\n")
    assert split_sentences(text) == ["Operating earnings were $250 million this year."]


def test_offtopic_incidental_keyword_refuses_before_generation(monkeypatch):
    from src.services.ask_flow import _generate_answer
    hits = [SearchHit("p", "We use computer models to estimate the cost of insurance operations.", {}, 1.0)]
    question = "How do I configure a firewall to protect a computer?"
    prompt = build_cited_prompt(question, hits)
    for name in ("llama", "mlx"):
        provider = mock.Mock(provider_name=name)
        answer, citations = _generate_answer(provider, prompt, hits, 100, query=question)
        assert answer == REFUSAL_LINE
        assert citations == []
        provider.generate.assert_not_called()


def test_relevance_uses_adjacent_evidence_not_unrelated_passages():
    from src.generation.evidence_gate import assess_evidence
    from src.generation.text_relevance import content_words, split_sentences

    question = "Describe orchard harvest equipment acreage irrigation staffing exports storage."
    passage = ("Orchard yields improved substantially. Harvest volumes exceeded expectations. "
               "Equipment upgrades reduced maintenance costs.")
    terms = set(content_words(question))
    assert all(len(terms & set(content_words(s))) / len(terms) < 0.30
               for s in split_sentences(passage))
    hit = SearchHit("p", passage, {}, 1.0)
    assert assess_evidence(question, [hit]).sufficient
    separate_hits = [SearchHit(str(i), s, {}, 1.0)
                     for i, s in enumerate(split_sentences(passage))]
    assert not assess_evidence(question, separate_hits).sufficient


def test_relevance_normalizes_inflection_and_percent_notation():
    from src.generation.evidence_gate import assess_evidence
    from src.services.ask_flow import _generate_answer

    question = "What percentage did the orchards own?"
    text = "The orchard owns 35% of a distributor."
    hits = [SearchHit("p", text, {}, 1.0)]
    assert assess_evidence(question, hits).sufficient
    class Quoting:
        provider_name, model = "llama", "stub"

        def generate(self, prompt, max_new_tokens=None):
            return text + " [1]"

    answer, citations = _generate_answer(Quoting(), build_cited_prompt(question, hits),
                                        hits, 100, query=question)
    assert answer == text + " [1]"
    assert citations


def test_citation_only_model_answer_becomes_refusal():
    from src.services.ask_flow import _generate_answer, is_citation_only
    hits = [SearchHit("p", "Derivatives are financial weapons of mass destruction.", {}, 1.0)]
    question = "What did Buffett say about derivatives?"
    prompt = build_cited_prompt(question, hits)
    assert is_citation_only(" [5]. ") and is_citation_only("[1] Yes [2].")
    assert not is_citation_only("Derivatives are dangerous. [1]")
    provider = mock.Mock(provider_name="llama")
    provider.generate.return_value = "[1]"
    answer, citations = _generate_answer(provider, prompt, hits, 100, query=question)
    assert answer == REFUSAL_LINE
    assert citations == []
    provider.generate.assert_called_once()


def test_missing_model_ask_returns_llm_unavailable_message(monkeypatch):
    from fastapi.testclient import TestClient
    from src.retrieval.retriever import RetrievalResult
    from src.services import ask_flow, backend_app as backend

    hit = SearchHit("p", "Derivatives are financial weapons of mass destruction.", {"year": 2002}, 1.0)

    class FixedRetriever:
        def search(self, **kwargs):
            return RetrievalResult(kwargs["query"], "hybrid", [hit])

    with mock.patch.object(factory, "LLM_MODEL_PATH", config.BASE_DIR / "models" / "missing.gguf"):
        provider = create_llm_provider(provider="llama")
    monkeypatch.setattr(ask_flow, "_state", {"retriever": FixedRetriever(), "docs_by_id": {}, "llm": provider})
    monkeypatch.setattr(backend, "API_KEYS", ())
    body = TestClient(backend.app).post(
        "/ask", json={"query": "What did Buffett say about derivatives?", "expand_query": False}).json()
    assert body["answer"] == ask_flow._llm_error_message(RuntimeError("x"))
    assert body["citations"] == []


if __name__ == "__main__":
    unittest.main()


_FT_V4_VALID = Path(
    "/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/data/ft_v4/valid.jsonl"
)


def test_split_prompt_keeps_answer_cue_and_training_system_prompt():
    from src.generation.prompt import SYSTEM_PROMPT
    from src.generation.providers.llama_provider import split_prompt

    hits = [
        SearchHit("a", "Derivatives are dangerous.", {"year": 2002, "source_file": "buffet_2002.txt"}, 1.0),
        SearchHit("b", "We compound steadily.", {"year": 1989, "source_file": "buffet_1989.txt"}, 0.9),
    ]
    system, user = split_prompt(build_cited_prompt("What about derivatives?", hits))
    assert user.endswith("\n\nAnswer:")
    assert user.startswith("BEGIN UNTRUSTED PASSAGES")
    if _FT_V4_VALID.exists():
        with _FT_V4_VALID.open() as fh:
            train_system = json.loads(fh.readline())["messages"][0]["content"]
        assert system == train_system
    assert system == SYSTEM_PROMPT
    assert system.endswith("outside the numbered passages.\n")
