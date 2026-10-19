"""space_app: backend under /api (security kept), built frontend at /."""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def space(tmp_path, monkeypatch):
    """Reload space_app against a temp dist, with backend startup replaced by a fake."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>buffett-ui</body></html>")
    monkeypatch.setenv("FRONTEND_DIST", str(dist))

    from src.services import backend_app, space_app

    calls = []

    async def fake_startup() -> None:
        calls.append(1)

    monkeypatch.setattr(backend_app, "startup", fake_startup)
    module = importlib.reload(space_app)
    yield module, backend_app, calls
    importlib.reload(space_app)


def test_health_public_and_startup_runs_once(space) -> None:
    module, _backend, calls = space
    with TestClient(module.app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/health").json() == {"status": "alive"}
    assert calls == [1]


def test_stats_requires_key_when_api_keys_set(space, monkeypatch) -> None:
    module, backend, _calls = space
    monkeypatch.setattr(backend, "API_KEYS", ("secret-key",))
    with TestClient(module.app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/stats").status_code == 401
        assert client.get("/api/stats", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/api/stats", headers={"X-API-Key": "secret-key"}).status_code == 200


def test_ask_is_guarded_without_key(space, monkeypatch) -> None:
    module, backend, _calls = space
    monkeypatch.setattr(backend, "API_KEYS", ("secret-key",))
    with TestClient(module.app) as client:
        assert client.post("/api/ask", json={"question": "x"}).status_code == 401


def test_root_serves_index_and_unknown_asset_404(space) -> None:
    module, _backend, _calls = space
    with TestClient(module.app) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert "buffett-ui" in root.text
        assert client.get("/assets/x").status_code == 404
