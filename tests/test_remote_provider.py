"""RemoteProvider against a local stub HTTP server (stdlib, no mocks of requests)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from src.generation.prompt import system_prompt
from src.generation.providers import factory, remote_provider
from src.generation.providers.factory import create_llm_provider
from src.generation.providers.llama_provider import LlamaCppProvider
from src.generation.providers.remote_provider import RemoteProvider
from src.generation.providers.unavailable import UnavailableProvider

TOKEN = "sekret-token-123"
ANSWER = "Buffett wrote about float [1]."


class _Stub:
    """Local OpenAI-compatible server; ``script`` is a list of (status, body) consumed per request."""

    def __init__(self) -> None:
        self.requests: list = []
        self.script: list = []
        self.stream_pieces: tuple = ("Buffett ", "wrote ", "[1].")
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                status, override = stub.script.pop(0) if stub.script else (200, None)
                if status != 200:
                    payload = (override or f"error {status} echo {self.headers.get('Authorization')}").encode()
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                elif body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for piece in stub.stream_pieces:  # raw UTF-8 like llama-server (no \\u escapes, no charset)
                        chunk = {"choices": [{"delta": {"content": piece}}]}
                        self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    payload = json.dumps({"choices": [{"message": {"content": f" {ANSWER} "}}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub(monkeypatch):
    for name in ("PROMPT_VERSION", "REMOTE_LLM_MODEL", "REMOTE_LLM_TIMEOUT_S", "REMOTE_LLM_RETRIES", "LLM_MODEL_PATH"):
        monkeypatch.delenv(name, raising=False)
    server = _Stub()
    monkeypatch.setenv("REMOTE_LLM_URL", server.url)
    monkeypatch.setenv("REMOTE_LLM_TOKEN", TOKEN)
    monkeypatch.setattr(remote_provider, "_BACKOFF_S", 0.0)
    yield server
    server.close()


def _prompt() -> str:
    return (
        f"{system_prompt().strip()}\n\nBEGIN UNTRUSTED PASSAGES\n[1] Float is useful.\nEND UNTRUSTED PASSAGES\n"
        "Question: What is float?\nAnswer:"
    )


def test_messages_match_llama_provider(stub):
    llama = LlamaCppProvider.__new__(LlamaCppProvider)  # no model is loaded
    prompt = _prompt()
    assert RemoteProvider()._messages(prompt) == llama._messages(prompt)
    assert RemoteProvider()._messages("plain question")[0]["role"] == "user"


def test_generate_sends_openai_payload(stub, monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "0.1")
    provider = RemoteProvider()
    assert provider.generate(_prompt(), max_new_tokens=123) == ANSWER
    sent = stub.requests[0]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["auth"] == f"Bearer {TOKEN}"
    body = sent["body"]
    assert body["messages"] == provider._messages(_prompt())
    assert body["model"] == "buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf"
    assert body["max_tokens"] == 123 and body["seed"] == 42 and body["temperature"] == provider.temperature
    assert "stream" not in body


def test_url_with_v1_is_not_doubled(stub, monkeypatch):
    monkeypatch.setenv("REMOTE_LLM_URL", stub.url + "/v1/")
    assert RemoteProvider().generate(_prompt()) == ANSWER
    assert stub.requests[0]["path"] == "/v1/chat/completions"


def test_generate_stream_yields_deltas(stub):
    pieces = list(RemoteProvider().generate_stream(_prompt()))
    assert "".join(pieces) == "Buffett wrote [1]."
    assert stub.requests[0]["body"]["stream"] is True


def test_generate_stream_decodes_utf8_without_charset(stub):
    # text/event-stream without a charset: requests would decode it as ISO-8859-1 ("canât")
    stub.stream_pieces = ("We can’t ", "value Moody’s.")
    assert "".join(RemoteProvider().generate_stream(_prompt())) == "We can’t value Moody’s."


def test_retry_on_503_then_success(stub):
    stub.script = [(503, "cold"), (503, "cold")]
    assert RemoteProvider().generate(_prompt()) == ANSWER
    assert len(stub.requests) == 3


def test_retries_exhausted_raises_with_status(stub, monkeypatch):
    monkeypatch.setenv("REMOTE_LLM_RETRIES", "1")
    stub.script = [(503, "cold"), (503, "cold")]
    with pytest.raises(RuntimeError, match="503"):
        RemoteProvider().generate(_prompt())
    assert len(stub.requests) == 2


def test_no_retry_on_400_and_token_absent_from_error(stub):
    stub.script = [(400, f"bad request, auth was Bearer {TOKEN}")]
    with pytest.raises(RuntimeError) as err:
        RemoteProvider().generate(_prompt())
    assert "400" in str(err.value)
    assert TOKEN not in str(err.value)
    assert len(stub.requests) == 1


def test_timeout_setting_is_passed(stub, monkeypatch):
    monkeypatch.setenv("REMOTE_LLM_TIMEOUT_S", "42")
    seen = {}
    real_post = requests.post

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_post(*args, **kwargs)

    monkeypatch.setattr(remote_provider.requests, "post", spy)
    RemoteProvider().generate(_prompt())
    assert seen["timeout"] == 42.0


def test_connect_error_is_retried_then_raises(monkeypatch):
    monkeypatch.setenv("REMOTE_LLM_URL", "http://127.0.0.1:1")  # nothing listens
    monkeypatch.setenv("REMOTE_LLM_RETRIES", "1")
    monkeypatch.setenv("REMOTE_LLM_TOKEN", TOKEN)
    monkeypatch.delenv("PROMPT_VERSION", raising=False)
    monkeypatch.setattr(remote_provider, "_BACKOFF_S", 0.0)
    with pytest.raises(RuntimeError, match="unreachable after 2 attempts") as err:
        RemoteProvider().generate(_prompt())
    assert TOKEN not in str(err.value)


def test_factory_returns_remote_when_url_set(stub):
    provider = create_llm_provider("remote")
    assert isinstance(provider, RemoteProvider) and provider.provider_name == "remote"


def test_factory_unavailable_without_url(monkeypatch):
    monkeypatch.delenv("REMOTE_LLM_URL", raising=False)
    provider = factory.create_llm_provider("remote")
    assert isinstance(provider, UnavailableProvider)
    with pytest.raises(RuntimeError, match="REMOTE_LLM_URL not set"):
        provider.generate("anything")


def test_prompt_version_mismatch_raises(stub, monkeypatch):
    monkeypatch.setenv("PROMPT_VERSION", "v2")  # the default model is FT-r1 (v1)
    monkeypatch.delenv("ALLOW_PROMPT_MISMATCH", raising=False)
    with pytest.raises(ValueError, match="contradicts"):
        RemoteProvider()
