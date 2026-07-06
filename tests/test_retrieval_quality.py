import unittest

from src.retrieval.query_expansion import expand_query, parse_expansion_keywords
from src.retrieval.retriever import deduplicate_hits
from src.vector_store import SearchHit


def _hit(hid: str, text: str, score: float = 1.0) -> SearchHit:
    return SearchHit(id=hid, text=text, metadata={"year": 2008}, score=score)


LONG_A = (
    "In 2004 our float cost us less than nothing, and I told you that we had a chance "
    "of no-cost float in 2005. But we had the mega-cat, and as a specialist in that "
    "coverage, Berkshire suffered hurricane losses of 3.4 billion dollars."
)
LONG_B = (
    "Charlie and I believe that derivatives are financial weapons of mass destruction, "
    "carrying dangers that, while now latent, are potentially lethal to the economy."
)


class DeduplicateHitsTests(unittest.TestCase):
    def test_exact_duplicates_are_dropped(self) -> None:
        hits = [_hit("2005_34", LONG_A), _hit("2005_35", LONG_A, score=0.9), _hit("2002_10", LONG_B)]
        kept = deduplicate_hits(hits)
        self.assertEqual([h.id for h in kept], ["2005_34", "2002_10"])

    def test_contained_overlap_window_is_dropped(self) -> None:
        overlap = LONG_A + " Nevertheless, our float was costless once again that year."
        hits = [_hit("a", overlap), _hit("b", LONG_A), _hit("c", LONG_B)]
        kept = deduplicate_hits(hits)
        self.assertEqual([h.id for h in kept], ["a", "c"])

    def test_distinct_passages_are_kept(self) -> None:
        hits = [_hit("a", LONG_A), _hit("b", LONG_B)]
        self.assertEqual(len(deduplicate_hits(hits)), 2)

    def test_higher_ranked_hit_wins(self) -> None:
        hits = [_hit("first", LONG_B, score=0.9), _hit("second", LONG_B, score=0.1)]
        kept = deduplicate_hits(hits)
        self.assertEqual([h.id for h in kept], ["first"])


class QueryExpansionTests(unittest.TestCase):
    def test_parses_comma_separated_keywords(self) -> None:
        raw = "ISCAR, Israel, OPEC, oil imports, Iscar Metalworking"
        self.assertEqual(
            parse_expansion_keywords(raw),
            ["ISCAR", "Israel", "OPEC", "oil imports", "Iscar Metalworking"],
        )

    def test_drops_refusal_and_junk(self) -> None:
        raw = "The shareholder letters do not contain information, ISCAR"
        self.assertEqual(parse_expansion_keywords(raw), ["ISCAR"])

    def test_expand_query_appends_keywords(self) -> None:
        class FakeLLM:
            provider_name = "openrouter"

            def generate(self, prompt, max_new_tokens=None):
                return "Israel, ISCAR, OPEC"

        expanded = expand_query("Middle East in letters?", FakeLLM())
        self.assertEqual(expanded, "Middle East in letters? Israel ISCAR OPEC")

    def test_expand_query_skips_local_provider(self) -> None:
        class FakeLocal:
            provider_name = "local"

            def generate(self, prompt, max_new_tokens=None):
                raise AssertionError("should not be called")

        self.assertIsNone(expand_query("anything", FakeLocal()))

    def test_expand_query_swallows_provider_errors(self) -> None:
        class Broken:
            provider_name = "openrouter"

            def generate(self, prompt, max_new_tokens=None):
                raise RuntimeError("boom")

        self.assertIsNone(expand_query("anything", Broken()))


if __name__ == "__main__":
    unittest.main()
