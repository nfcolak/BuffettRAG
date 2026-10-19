"""space_app password gate: HTTP Basic auth over frontend and API when SITE_PASSWORD is set."""
from __future__ import annotations

import base64
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def space(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><body>buffett-ui</body></html>")
    monkeypatch.setenv("FRONTEND_DIST", str(dist))
    from src.services import backend_app, space_app

    async def fake_startup() -> None:
        return None

    monkeypatch.setattr(backend_app, "startup", fake_startup)
    yield importlib.reload(space_app)
    importlib.reload(space_app)


def _basic(user: str, password: str) -> dict:
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


def test_gate_off_without_password(space, monkeypatch) -> None:
    monkeypatch.delenv("SITE_PASSWORD", raising=False)
    with TestClient(space.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/stats").status_code == 200


def test_gate_requires_basic_auth(space, monkeypatch) -> None:
    monkeypatch.setenv("SITE_PASSWORD", "s3cret-pw")
    with TestClient(space.app) as client:
        root = client.get("/")
        assert root.status_code == 401
        assert root.headers["www-authenticate"].startswith("Basic")
        assert client.get("/api/stats").status_code == 401
        assert client.get("/", headers=_basic("buffett", "wrong")).status_code == 401
        assert client.get("/", headers=_basic("other", "s3cret-pw")).status_code == 401
        assert client.get("/", headers={"Authorization": "Basic !!notbase64"}).status_code == 401
        ok = client.get("/", headers=_basic("buffett", "s3cret-pw"))
        assert ok.status_code == 200 and "buffett-ui" in ok.text
        assert client.get("/api/stats", headers=_basic("buffett", "s3cret-pw")).status_code == 200


def test_health_stays_public(space, monkeypatch) -> None:
    monkeypatch.setenv("SITE_PASSWORD", "s3cret-pw")
    with TestClient(space.app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/ready").status_code == 401
