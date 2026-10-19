"""Hugging Face Space entry point: built frontend at ``/``, backend under ``/api``.

Run:
    uvicorn src.services.space_app:app --host 0.0.0.0 --port 7860

Mounted sub-apps do not run their own lifespan, so the outer app runs
``backend_app.startup()`` once. The backend's security middleware (API_KEYS, rate
limit, body cap) keeps applying because it is part of the mounted backend app.
"""

from __future__ import annotations

import base64
import hmac
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services import backend_app

FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", str(PROJECT_ROOT / "frontend" / "dist")))
SITE_USER = "buffett"
_PUBLIC_PATHS = {"/api/health"}  # liveness probe; returns no data


def _password_gate(inner: ASGIApp) -> ASGIApp:
    """HTTP Basic auth over the whole site when SITE_PASSWORD is set (read per request); off otherwise.

    The browser shows its own login prompt and resends the credentials on every same-origin fetch,
    so the frontend needs no change.
    """

    async def gated(scope: Scope, receive: Receive, send: Send) -> None:
        password = os.environ.get("SITE_PASSWORD", "")
        if scope["type"] != "http" or not password or scope["path"] in _PUBLIC_PATHS:
            await inner(scope, receive, send)
            return
        header = dict(scope.get("headers") or []).get(b"authorization", b"").decode("latin-1")
        ok = False
        if header.startswith("Basic "):
            try:
                user, _, given = base64.b64decode(header[6:]).decode("utf-8").partition(":")
                ok = hmac.compare_digest(user, SITE_USER) and hmac.compare_digest(given, password)
            except (ValueError, UnicodeDecodeError):
                ok = False
        if ok:
            await inner(scope, receive, send)
            return
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"www-authenticate", b'Basic realm="BuffettRAG"'), (b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"Unauthorized"})

    return gated


def _strip_mount_prefix(inner: ASGIApp) -> ASGIApp:
    """Hand the backend its own paths (``/health``, not ``/api/health``).

    Starlette keeps the full path in ``scope["path"]`` under a mount, and the backend's
    security middleware matches its public endpoints on ``request.url.path``.
    """

    async def wrapped(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in {"http", "websocket"}:
            prefix = scope.get("root_path", "")
            path = scope["path"]
            if prefix and path.startswith(prefix):
                scope = dict(scope, path=path[len(prefix):] or "/", root_path="")
        await inner(scope, receive, send)

    return wrapped


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    await backend_app.startup()
    yield


class _PasswordGateMiddleware:
    """Starlette middleware wrapper around _password_gate (add_middleware passes ``app=``)."""

    def __init__(self, app: ASGIApp) -> None:
        self._gated = _password_gate(app)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._gated(scope, receive, send)


app = FastAPI(title="BuffettRAG Space", version="1.0.0", lifespan=_lifespan)
app.mount("/api", _strip_mount_prefix(backend_app.app))
# Mounted last: "/" matches everything the "/api" mount did not take.
app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True, check_dir=False), name="frontend")
# Outermost layer: the password gate covers the frontend and the API alike.
app.add_middleware(_PasswordGateMiddleware)
