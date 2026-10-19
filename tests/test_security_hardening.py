import unittest

from src.services.security import FixedWindowRateLimiter


class SecurityHardeningTests(unittest.TestCase):
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
