import unittest

from src.generation.openai_llm import OpenAILLM
from src.generation.providers import (
    AnthropicProvider,
    OpenAIProvider,
    OpenRouterProvider,
    create_llm_provider,
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


if __name__ == "__main__":
    unittest.main()
