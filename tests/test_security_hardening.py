import unittest

from src.services.security import FixedWindowRateLimiter
from src.vector_store import _validate_pg_identifier


class SecurityHardeningTests(unittest.TestCase):
    def test_pg_identifier_rejects_sql_fragments(self) -> None:
        for value in ("bad-name", "1table", "chunks;DROP TABLE users"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _validate_pg_identifier(value, "PG_TABLE")

    def test_pg_identifier_accepts_simple_names(self) -> None:
        for value in ("buffett_chunks", "_private", "table_1"):
            with self.subTest(value=value):
                self.assertEqual(_validate_pg_identifier(value, "PG_TABLE"), value)

    def test_fixed_window_rate_limiter_blocks_after_limit(self) -> None:
        limiter = FixedWindowRateLimiter(max_requests=2, window_seconds=60)
        self.assertTrue(limiter.allow("client"))
        self.assertTrue(limiter.allow("client"))
        self.assertFalse(limiter.allow("client"))


if __name__ == "__main__":
    unittest.main()
