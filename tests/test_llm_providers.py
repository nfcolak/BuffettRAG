import importlib
import unittest
from unittest import mock

import pytest

import config
from src.generation.prompt import REFUSAL_LINE, build_cited_prompt
from src.generation.providers import LlamaCppProvider, LocalProvider, create_llm_provider
from src.generation.providers import factory
from src.vector_store import SearchHit


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


if __name__ == "__main__":
    unittest.main()
