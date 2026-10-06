import importlib
import unittest
from unittest import mock

import pytest

import config
from src.generation.prompt import REFUSAL_LINE, build_cited_prompt
from src.generation.providers import LlamaCppProvider, LocalProvider, create_llm_provider
from src.generation.providers import factory
from src.storage import SearchHit


def _build_prompt(question: str) -> str:
    return (
        "SYSTEM INSTRUCTIONS...\n\n"
        "BEGIN UNTRUSTED PASSAGES\n"
        "[1] (year=2008, source=buffet_2008.pdf)\n"
        "Derivatives are dangerous instruments that have dramatically increased "
        "the leverage and risks in our financial system.\n\n"
        "[2] (year=1989, source=buffet_1989.txt)\n"
        "We made good progress on compounding rates this year and Ike enjoyed "
        "his visit to the candy store on the 29th of March.\n\n"
        "END UNTRUSTED PASSAGES\n\n"
        "BEGIN USER QUESTION\n"
        f"Question: {question}\n\n"
        "END USER QUESTION\n\n"
        "Answer with inline citations [n] referring to the passages above.\n\nAnswer:"
    )


class LLMProviderTests(unittest.TestCase):
    def test_factory_falls_back_to_local_when_model_missing(self) -> None:
        with mock.patch.object(factory, "LLM_MODEL_PATH", config.BASE_DIR / "models" / "missing.gguf"):
            provider = create_llm_provider(provider="llama")
        self.assertIsInstance(provider, LocalProvider)

    def test_factory_falls_back_to_local_when_llama_cpp_missing(self) -> None:
        with mock.patch.object(factory, "LLM_MODEL_PATH", config.BASE_DIR / "config.py"), \
                mock.patch.object(factory.importlib.util, "find_spec", return_value=None):
            provider = create_llm_provider(provider="llama")
        self.assertIsInstance(provider, LocalProvider)

    def test_factory_creates_local_provider(self) -> None:
        provider = create_llm_provider(provider="local")
        self.assertIsInstance(provider, LocalProvider)
        self.assertEqual(provider.provider_name, "local")
        self.assertEqual(provider.model, "embedded-extractive-v1")

    def test_factory_has_no_api_key_or_model_parameters(self) -> None:
        with self.assertRaises(TypeError):
            create_llm_provider(provider="local", api_key="x")  # type: ignore[call-arg]

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

    def test_local_provider_extracts_cited_sentences(self) -> None:
        provider = LocalProvider()
        answer = provider.generate(_build_prompt("What did Buffett say about the danger of derivatives?"))
        self.assertIn("[1]", answer)
        self.assertIn("Derivatives are dangerous", answer)
        self.assertNotIn("candy store", answer)

    def test_local_provider_refuses_when_nothing_matches(self) -> None:
        provider = LocalProvider()
        answer = provider.generate(_build_prompt("What is the meaning of quantum entanglement?"))
        self.assertEqual(answer, REFUSAL_LINE)


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


def test_local_splitter_joins_numeric_fragments_and_preserves_honorifics():
    from src.generation.providers.local_provider import _split_sentences
    text = "In 1998. 1999 the firm bought 2% of a supplier. Mr. Market is there to serve you, not to guide you."
    sentences = _split_sentences(text)
    assert sentences == ["In 1998. 1999 the firm bought 2% of a supplier.",
                         "Mr. Market is there to serve you, not to guide you."]
    assert not any(s == "In 1998." or s[:1].isdigit() for s in sentences)
    from src.evaluation.claim_validator import validate_and_filter_answer
    hits = [SearchHit("p", text, {}, 1.0)]
    answer = "\n\n".join(s + " [1]" for s in sentences)
    assert validate_and_filter_answer(answer, hits).safe_answer == answer


def test_local_skips_salutations_headers_and_signatures():
    from src.generation.providers.local_provider import _split_sentences
    text = ("To the Stockholders of Example Industries Inc.:\n\n"
            "Operating Earnings\n\nOperating earnings were $250 million this year.\n\n"
            "Warren E. Buffett\nChairman of the Board\nPage 12\nEXAMPLE INDUSTRIES\n")
    assert _split_sentences(text) == ["Operating earnings were $250 million this year."]


def test_local_quantity_question_prefers_relevant_numeric_sentence():
    hits = [SearchHit("p", "The cost of float was an important issue for our insurance operations. "
                     "The cost of float was 8% of the funds held.", {}, 1.0)]
    prompt = build_cited_prompt("What percent was the cost of float?", hits)
    answer = LocalProvider().generate(prompt, max_new_tokens=18)
    assert answer == "The cost of float was 8% of the funds held. [1]"


def test_offtopic_incidental_keyword_refuses_before_both_engines(monkeypatch):
    from src.services.ask_flow import _generate_answer
    hits = [SearchHit("p", "We use computer models to estimate the cost of insurance operations.", {}, 1.0)]
    question = "How do I configure a firewall to protect a computer?"
    prompt = build_cited_prompt(question, hits)
    assert LocalProvider().generate(prompt) == REFUSAL_LINE
    for name in ("local", "llama"):
        provider = mock.Mock(provider_name=name)
        answer, citations = _generate_answer(provider, prompt, hits, 100, query=question)
        assert answer == REFUSAL_LINE
        assert citations == []
        provider.generate.assert_not_called()


def test_relevance_uses_adjacent_evidence_not_unrelated_passages():
    from src.generation.evidence_gate import assess_evidence
    from src.generation.providers.local_provider import _content_words, _split_sentences

    question = "Describe orchard harvest equipment acreage irrigation staffing exports storage."
    passage = ("Orchard yields improved substantially. Harvest volumes exceeded expectations. "
               "Equipment upgrades reduced maintenance costs.")
    terms = set(_content_words(question))
    assert all(len(terms & set(_content_words(s))) / len(terms) < 0.30
               for s in _split_sentences(passage))
    hit = SearchHit("p", passage, {}, 1.0)
    assert assess_evidence(question, [hit]).sufficient
    assert LocalProvider().generate(build_cited_prompt(question, [hit])) != REFUSAL_LINE
    separate_hits = [SearchHit(str(i), s, {}, 1.0)
                     for i, s in enumerate(_split_sentences(passage))]
    assert not assess_evidence(question, separate_hits).sufficient
    assert LocalProvider().generate(build_cited_prompt(question, separate_hits)) == REFUSAL_LINE


def test_relevance_normalizes_inflection_and_percent_notation():
    from src.generation.evidence_gate import assess_evidence
    from src.services.ask_flow import _generate_answer

    question = "What percentage did the orchards own?"
    text = "The orchard owns 35% of a distributor."
    hits = [SearchHit("p", text, {}, 1.0)]
    assert assess_evidence(question, hits).sufficient
    answer, citations = _generate_answer(LocalProvider(), build_cited_prompt(question, hits),
                                        hits, 100, query=question)
    assert answer == text + " [1]"
    assert citations


def test_local_leads_with_best_sentence_then_at_most_two_supporting():
    hits = [
        SearchHit("primary", "Orchard exports expanded storage capacity. "
                  "No shipments went to local customers. "
                  "The delivery schedule remained unchanged.", {}, 1.0),
        SearchHit("secondary", "Exports expanded storage capacity at the depot. "
                  "The loading crew used reinforced containers. "
                  "The warehouse opened before dawn.", {}, 0.9),
        SearchHit("high_overlap", "Orchard exports improve storage capacity. "
                  "The road network served distant farms. "
                  "The packaging crew worked overnight. "
                  "Freight vehicles followed the coast.", {}, 0.8),
        SearchHit("global_match", "Exports require storage capacity elsewhere.", {}, 0.7),
    ]
    question = "How did orchard exports improve storage capacity?"
    answer = LocalProvider().generate(build_cited_prompt(question, hits))
    lead, _, support = answer.partition("\n\n")
    assert lead == "Orchard exports improve storage capacity. [3]"
    assert support == ("Orchard exports expanded storage capacity. [1] "
                       "Exports expanded storage capacity at the depot. [2]")
    # Neighbours with no question term and no figure are filler.
    assert "road network" not in answer
    assert "No shipments" not in answer
    from src.evaluation.claim_validator import validate_and_filter_answer
    assert validate_and_filter_answer(answer, hits).safe_answer == answer


def test_local_supports_lead_with_following_figure():
    hits = [SearchHit("p", "The railroad acquisition closed in February. "
                      "The purchase price was $34 billion. "
                      "Weather in the region stayed mild.", {}, 1.0)]
    question = "When did the railroad acquisition close?"
    answer = LocalProvider().generate(build_cited_prompt(question, hits))
    assert answer == ("The railroad acquisition closed in February. [1]\n\n"
                      "The purchase price was $34 billion. [1]")


def test_citation_only_generation_falls_back_to_extractive_answer():
    from src.services.ask_flow import _generate_answer, is_citation_only
    hits = [SearchHit("p", "Derivatives are financial weapons of mass destruction.", {}, 1.0)]
    question = "What did Buffett say about derivatives?"
    prompt = build_cited_prompt(question, hits)
    assert is_citation_only(" [5]. ") and is_citation_only("[1] Yes [2].")
    assert not is_citation_only("Derivatives are dangerous. [1]")
    provider = mock.Mock(provider_name="llama")
    provider.generate.return_value = "[1]"
    answer, citations = _generate_answer(provider, prompt, hits, 100, query=question)
    assert answer == "Derivatives are financial weapons of mass destruction. [1]"
    assert answer == LocalProvider().generate(prompt, max_new_tokens=100)
    assert citations


if __name__ == "__main__":
    unittest.main()
