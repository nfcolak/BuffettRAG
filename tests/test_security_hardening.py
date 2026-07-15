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

    def test_fixed_window_rate_limiter_sweeps_expired_buckets(self) -> None:
        limiter = FixedWindowRateLimiter(max_requests=2, window_seconds=60)
        for i in range(100):
            limiter.allow(f"client-{i}", now=0.0)
        self.assertEqual(len(limiter._buckets), 100)
        # One request after the window expires must trigger a sweep that
        # drops all the stale buckets instead of accumulating forever.
        limiter.allow("fresh-client", now=61.0)
        self.assertEqual(set(limiter._buckets), {"fresh-client"})

    def test_fixed_window_rate_limiter_resets_after_window(self) -> None:
        limiter = FixedWindowRateLimiter(max_requests=1, window_seconds=60)
        self.assertTrue(limiter.allow("client", now=0.0))
        self.assertFalse(limiter.allow("client", now=1.0))
        self.assertTrue(limiter.allow("client", now=61.0))


if __name__ == "__main__":
    unittest.main()
