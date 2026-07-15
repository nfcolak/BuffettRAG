"""Small security helpers shared by FastAPI services."""

from __future__ import annotations

import secrets
import time
from typing import Iterable

from fastapi import Request


class FixedWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max(1, max_requests)
        self.window_seconds = max(1, window_seconds)
        self._buckets: dict[str, tuple[int, float]] = {}
        self._next_sweep = 0.0

    def allow(self, key: str, now: float | None = None) -> bool:
        if now is None:
            now = time.monotonic()
        if now >= self._next_sweep:
            self._sweep(now)
        count, reset_at = self._buckets.get(key, (0, now + self.window_seconds))
        if now >= reset_at:
            count = 0
            reset_at = now + self.window_seconds
        count += 1
        self._buckets[key] = (count, reset_at)
        return count <= self.max_requests

    def _sweep(self, now: float) -> None:
        """Drop expired buckets so distinct client keys cannot grow memory forever."""
        self._buckets = {
            key: entry for key, entry in self._buckets.items() if entry[1] > now
        }
        self._next_sweep = now + self.window_seconds


def client_key(request: Request, trust_proxy_headers: bool = False) -> str:
    if trust_proxy_headers:
        forwarded_for = request.headers.get("x-forwarded-for", "")
        if forwarded_for:
            return forwarded_for.split(",", 1)[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def extract_api_key(request: Request) -> str:
    header_key = request.headers.get("x-api-key", "").strip()
    if header_key:
        return header_key

    auth = request.headers.get("authorization", "").strip()
    prefix = "Bearer "
    if auth.startswith(prefix):
        return auth[len(prefix) :].strip()
    return ""


def is_authorized(request: Request, allowed_keys: Iterable[str]) -> bool:
    keys = tuple(k for k in allowed_keys if k)
    if not keys:
        return True

    provided = extract_api_key(request)
    if not provided:
        return False
    return any(secrets.compare_digest(provided, expected) for expected in keys)
