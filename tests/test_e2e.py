"""End-to-end smoke tests for BuffettRAG.

Designed to run quickly (no LLM) and catch breakage in the
ingest -> embed -> retrieve -> rerank chain.

Run:
    python -m tests.test_e2e
    pytest tests/test_e2e.py -q     (if pytest is installed)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHUNKS_FILE, CHUNKS_V2_FILE
from src.ingestion.chunker import merge_short_chunks, split_text
from src.ingestion.pdf_extractor import clean_extracted_text
from src.ingestion.pipeline_v2 import infer_section_title
from src.ingestion.topic_tagger import tag_topics
from src.retrieval.context import build_doc_lookup, expand_hits_with_neighbors
from src.retrieval.bm25 import BM25Retriever, tokenize
from src.retrieval.retriever import detect_year_filter, reciprocal_rank_fusion
from src.vector_store import SearchHit, StoredDoc, load_chunks_as_docs


class TestIngestion(unittest.TestCase):
    def test_clean_dehyphenates(self) -> None:
        text = "We made an invest-\nment in 2008."
        cleaned = clean_extracted_text(text)
        self.assertIn("investment", cleaned)
        self.assertNotIn("invest-\n", cleaned)

    def test_clean_drops_boilerplate(self) -> None:
        text = "BERKSHIRE HATHAWAY INC.\nReal sentence here.\n12\nReal sentence two."
        cleaned = clean_extracted_text(text)
        self.assertNotIn("BERKSHIRE HATHAWAY INC.", cleaned)
        self.assertIn("Real sentence here.", cleaned)
        self.assertIn("Real sentence two.", cleaned)

    def test_chunker_respects_paragraphs(self) -> None:
        text = ("Para one. " * 60) + "\n\n" + ("Para two. " * 60)
        chunks = split_text(text, chunk_size=400, chunk_overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 600 for c in chunks))  # incl. some overlap slack

    def test_chunker_merges_short_fragments(self) -> None:
        chunks = merge_short_chunks(
            ["Insurance Operations", "GEICO float gives Berkshire low-cost capital."],
            min_chunk_chars=160,
            target_chars=400,
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("Insurance Operations", chunks[0])
        self.assertIn("GEICO float", chunks[0])

    def test_infer_section_title(self) -> None:
        title = infer_section_title("Insurance Operations\n\nGEICO had a strong year.", "")
        self.assertEqual(title, "Insurance Operations")

    def test_topic_tagger_finds_inflation(self) -> None:
        text = (
            "Inflation continues to erode the real returns earned on capital. "
            "The Federal Reserve has raised rates again."
        )
        tags = tag_topics(text)
        self.assertIn("inflation", tags)


class TestPgWhereClause(unittest.TestCase):
    """Unit-test the pgvector WHERE-clause builder without needing a running DB."""

    def setUp(self) -> None:
        from src.vector_store import _pg_build_where_clause
        self._build = _pg_build_where_clause

    def test_empty(self) -> None:
        sql, params = self._build(None)
        self.assertEqual(sql, "")
        self.assertEqual(params, [])

    def test_equality(self) -> None:
        sql, params = self._build({"year": 2008})
        self.assertEqual(sql, "WHERE year = %s")
        self.assertEqual(params, [2008])

    def test_range(self) -> None:
        sql, params = self._build({"year": {"$gte": 2010, "$lte": 2015}})
        self.assertIn("year >= %s", sql)
        self.assertIn("year <= %s", sql)
        self.assertEqual(sorted(params), [2010, 2015])

    def test_in(self) -> None:
        sql, params = self._build({"year": {"$in": [2008, 2009]}})
        self.assertIn("year IN (%s,%s)", sql)
        self.assertEqual(params, [2008, 2009])

    def test_rejects_unknown_field(self) -> None:
        with self.assertRaises(ValueError):
            self._build({"year; DROP TABLE buffett_chunks; --": 2008})

    def test_rejects_unknown_operator(self) -> None:
        with self.assertRaises(ValueError):
            self._build({"year": {"$ne": 2008}})


class TestRetrievalUtilities(unittest.TestCase):
    def test_year_filter_single_year(self) -> None:
        f = detect_year_filter("What did Buffett write in 2008?")
        self.assertEqual(f, {"year": 2008})

    def test_year_filter_range(self) -> None:
        f = detect_year_filter("between 2010 and 2015 letters on inflation")
        self.assertEqual(f, {"year": {"$gte": 2010, "$lte": 2015}})

    def test_year_filter_decade_word(self) -> None:
        f = detect_year_filter("his views in the 1990s on technology")
        self.assertEqual(f, {"year": {"$gte": 1990, "$lte": 1999}})

    def test_year_filter_no_year(self) -> None:
        self.assertIsNone(detect_year_filter("intrinsic value and moats"))

    def test_rrf_fuses_two_rankings(self) -> None:
        a = [SearchHit(id=str(i), text="", metadata={}, score=0.0) for i in range(3)]
        b = [SearchHit(id=str(i), text="", metadata={}, score=0.0) for i in [2, 0, 4]]
        fused = reciprocal_rank_fusion([a, b], top_k=4)
        ids = [h.id for h in fused]
        # "0" is rank 1 in a and rank 2 in b -- best combined.
        self.assertEqual(ids[0], "0")

    def test_context_expansion_adds_neighbors(self) -> None:
        docs = [
            StoredDoc(
                id="0",
                text="before context",
                metadata={"next_chunk_id": "1"},
            ),
            StoredDoc(
                id="1",
                text="current context",
                metadata={"previous_chunk_id": "0", "next_chunk_id": "2"},
            ),
            StoredDoc(
                id="2",
                text="after context",
                metadata={"previous_chunk_id": "1"},
            ),
        ]
        hit = SearchHit(id="1", text="current context", metadata={}, score=1.0)
        expanded = expand_hits_with_neighbors(
            [hit],
            build_doc_lookup(docs),
            neighbors=1,
            max_chars=1000,
        )
        self.assertIn("before context", expanded[0].text)
        self.assertIn("current context", expanded[0].text)
        self.assertIn("after context", expanded[0].text)


class TestBM25(unittest.TestCase):
    def test_tokenize(self) -> None:
        tokens = tokenize("Buffett's 2008 GEICO float strategy!")
        self.assertEqual(tokens, ["buffett's", "2008", "geico", "float", "strategy"])

    def test_bm25_ranks_relevant_higher(self) -> None:
        docs = [
            StoredDoc(id="a", text="GEICO insurance float gives us low-cost capital", metadata={"year": 2010}),
            StoredDoc(id="b", text="See's Candies generates wonderful returns", metadata={"year": 1991}),
            StoredDoc(id="c", text="Insurance underwriting and float discipline", metadata={"year": 2014}),
        ]
        bm = BM25Retriever(docs)
        hits = bm.search("GEICO float", top_k=3)
        self.assertEqual(hits[0].id, "a")


class TestChunkStore(unittest.TestCase):
    """If chunks have been generated, sanity-check we can load them."""

    def test_chunks_file_loads(self) -> None:
        path = CHUNKS_V2_FILE if CHUNKS_V2_FILE.exists() else CHUNKS_FILE
        if not path.exists():
            self.skipTest("No chunks file generated yet.")
        docs = load_chunks_as_docs(path)
        self.assertGreater(len(docs), 100)
        # Spot check metadata.
        sample = docs[0]
        self.assertIsInstance(sample.id, str)
        self.assertIn("year", sample.metadata)


if __name__ == "__main__":
    unittest.main(verbosity=2)
