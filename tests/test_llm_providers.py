import unittest

from src.generation.openai_llm import OpenAILLM
from src.generation.providers import (
    AnthropicProvider,
    LocalProvider,
    OpenAIProvider,
    OpenRouterProvider,
    create_llm_provider,
)
from src.generation.providers.local_provider import REFUSAL_LINE


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
    def test_factory_creates_openai_provider(self) -> None:
        provider = create_llm_provider(
            provider="openai",
            api_key="sk-test",
            model="gpt-test",
        )
        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(provider.provider_name, "openai")
        self.assertEqual(provider.model, "gpt-test")

    def test_legacy_openai_llm_import_still_works(self) -> None:
        provider = OpenAILLM(api_key="sk-test", model="gpt-test")
        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(provider.provider_name, "openai")

    def test_factory_creates_openrouter_provider(self) -> None:
        provider = create_llm_provider(
            provider="openrouter",
            api_key="sk-or-test",
            model="openrouter/free",
        )
        self.assertIsInstance(provider, OpenRouterProvider)
        self.assertEqual(provider.provider_name, "openrouter")
        self.assertEqual(provider.model, "openrouter/free")

    def test_factory_creates_anthropic_provider(self) -> None:
        provider = create_llm_provider(
            provider="anthropic",
            api_key="sk-ant-test",
            model="claude-test",
        )
        self.assertIsInstance(provider, AnthropicProvider)
        self.assertEqual(provider.provider_name, "anthropic")
        self.assertEqual(provider.model, "claude-test")

    def test_factory_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported LLM provider"):
            create_llm_provider(provider="unknown")

    def test_factory_creates_local_provider_and_ignores_api_key(self) -> None:
        provider = create_llm_provider(provider="local", api_key="unused-key")
        self.assertIsInstance(provider, LocalProvider)
        self.assertEqual(provider.provider_name, "local")
        self.assertEqual(provider.model, "embedded-extractive-v1")

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


if __name__ == "__main__":
    unittest.main()
