"""End-to-end smoke for the cloud serving shape, with no model anywhere.

Starts a fake OpenAI-compatible server on localhost, starts ``src.services.space_app``
(real retrieval, ``DEFAULT_LLM_PROVIDER=remote``) with uvicorn in a subprocess, then calls
``/api/ask`` and ``/api/ask/stream``. Asserts that the canned cited answer comes back with
citations mapped to retrieved passages, and that the fake received exactly the messages
``LlamaCppProvider._messages`` builds for the same prompt (training/serving parity).

Run:
    env -u PYTHONPATH HF_HUB_OFFLINE=1 .venv/bin/python scripts/deploy/smoke_cloud.py
Exit code 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.prompt import system_prompt
from src.generation.providers.llama_provider import LlamaCppProvider

QUESTION = "What does Buffett say about insurance float at Berkshire?"
TOKEN = "smoke-token-not-a-secret"
MODEL = "buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf"
READY_TIMEOUT_S = 600.0
PASSAGE_RE = re.compile(r"\[(\d+)\] \(year=[^)]*\)\n(.*?)(?=\n\n\[\d+\] \(year=|\n\nEND UNTRUSTED PASSAGES)", re.S)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def canned_chunks(user: str) -> List[str]:
    """Cited answer built from the first sentence of passages [1] and [2] (so claim validation keeps it)."""
    passages = {int(n): text for n, text in PASSAGE_RE.findall(user)}
    chunks = []
    for n in (1, 2):
        flat = re.sub(r"\s+", " ", passages.get(n, "")).strip()
        sentences = [s.strip() for s in SENTENCE_RE.split(flat) if len(s.strip()) > 40 and s.strip()[0].isupper()]
        if sentences:
            chunks.append(f"{sentences[0].rstrip('.')} [{n}]. ")
    return chunks or ["No passage text found [1]. "]


class FakeLLM(BaseHTTPRequestHandler):
    """Records every chat-completions request and answers with the canned text."""

    requests_seen: List[Dict[str, Any]] = []

    def log_message(self, *_args: Any) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        FakeLLM.requests_seen.append(
            {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
        )
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        pieces = canned_chunks(body["messages"][-1]["content"])
        FakeLLM.requests_seen[-1]["canned"] = "".join(pieces).strip()
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in pieces:
                chunk = {"choices": [{"delta": {"content": piece}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": "".join(pieces)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _expected_messages(received: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Rebuild the prompt from what the fake got and run the embedded provider's message builder."""
    user = received[-1]["content"]
    prompt = system_prompt().strip() + "\n\n" + user if received[0]["role"] == "system" else user
    return LlamaCppProvider._messages(None, prompt)  # type: ignore[arg-type]


def _parse_sse(text: str) -> Dict[str, Any]:
    """Return the data of the last ``done`` event (and the event names seen)."""
    events: List[str] = []
    done: Optional[Dict[str, Any]] = None
    for block in text.split("\n\n"):
        name, data = "", ""
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if name:
            events.append(name)
            if name == "done":
                done = json.loads(data)
    return {"events": events, "done": done}


def _check_answer(label: str, answer: Any, citations: Any, hits: List[Dict[str, Any]], failures: List[str]) -> None:
    """The answer keeps a sentence the fake produced, and every citation maps to a retrieved passage."""
    canned = " ".join(r.get("canned", "") for r in FakeLLM.requests_seen)
    sentences = [re.sub(r"\s*\[\d+\]\.?$", "", p).strip() for p in re.split(r"(?<=\]\.)\s+", canned) if p.strip()]
    if not isinstance(answer, str) or not any(s and s in answer for s in sentences):
        failures.append(f"{label}: canned text missing in answer {answer!r}")
    if not citations:
        failures.append(f"{label}: no citations returned")
        return
    sources = {h.get("source_file") for h in hits}
    for citation in citations:
        if not citation.get("sources") or not set(citation["sources"]) <= sources:
            failures.append(f"{label}: citation {citation} is not mapped to a retrieved passage")


def main() -> int:
    failures: List[str] = []
    fake_port, app_port = _free_port(), _free_port()
    fake = ThreadingHTTPServer(("127.0.0.1", fake_port), FakeLLM)
    threading.Thread(target=fake.serve_forever, daemon=True).start()

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(
        VECTOR_BACKEND="faiss",
        FAISS_INDEX_NAME="faiss_v3",
        DEFAULT_LLM_PROVIDER="remote",
        REMOTE_LLM_URL=f"http://127.0.0.1:{fake_port}",
        REMOTE_LLM_MODEL=MODEL,
        REMOTE_LLM_TOKEN=TOKEN,
        REMOTE_LLM_RETRIES="0",
        HF_HUB_OFFLINE="1",
        PUBLIC_DEMO_MODE="0",
        API_KEYS="",
    )
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.services.space_app:app",
         "--host", "127.0.0.1", "--port", str(app_port), "--log-level", "warning"],
        cwd=str(PROJECT_ROOT), env=env,
    )
    base = f"http://127.0.0.1:{app_port}"
    started = time.monotonic()
    try:
        while True:
            if server.poll() is not None:
                print(f"FAIL: space_app exited early with code {server.returncode}")
                return 1
            try:
                if requests.get(f"{base}/api/ready", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() - started > READY_TIMEOUT_S:
                print(f"FAIL: /api/ready not 200 within {READY_TIMEOUT_S:.0f}s")
                return 1
            time.sleep(1.0)
        startup_s = time.monotonic() - started
        print(f"startup: {startup_s:.1f}s to /api/ready")

        body = {"query": QUESTION}
        t0 = time.monotonic()
        ask = requests.post(f"{base}/api/ask", json=body, timeout=300)
        ask_s = time.monotonic() - t0
        if ask.status_code != 200:
            failures.append(f"/api/ask status {ask.status_code}: {ask.text[:200]}")
        else:
            data = ask.json()
            _check_answer("ask", data.get("answer"), data.get("citations"), data.get("hits", []), failures)
        print(f"ask: {ask_s:.2f}s")

        t0 = time.monotonic()
        stream = requests.post(f"{base}/api/ask/stream", json=body, timeout=300)
        stream_s = time.monotonic() - t0
        if stream.status_code != 200:
            failures.append(f"/api/ask/stream status {stream.status_code}: {stream.text[:200]}")
        else:
            parsed = _parse_sse(stream.text)
            meta = next(
                (json.loads(line[5:]) for block in stream.text.split("\n\n") if block.startswith("event: meta")
                 for line in block.splitlines() if line.startswith("data:")),
                {},
            )
            if parsed["done"] is None:
                failures.append(f"stream: no done event (events {parsed['events']})")
            else:
                _check_answer("stream", parsed["done"].get("answer"), parsed["done"].get("citations"), meta.get("hits", []), failures)
        print(f"ask/stream: {stream_s:.2f}s")

        seen = FakeLLM.requests_seen
        if not seen:
            failures.append("fake LLM received no request")
        for index, request in enumerate(seen):
            payload = request["body"]
            if request["auth"] != f"Bearer {TOKEN}":
                failures.append(f"request {index}: bearer header mismatch")
            if payload.get("messages") != _expected_messages(payload.get("messages", [{}])):
                failures.append(f"request {index}: messages differ from LlamaCppProvider._messages")
            if payload.get("seed") != 42 or payload.get("model") != MODEL:
                failures.append(f"request {index}: seed/model mismatch ({payload.get('seed')}, {payload.get('model')})")
        if not any(r["body"].get("stream") for r in seen) or not any(not r["body"].get("stream") for r in seen):
            failures.append("expected one streaming and one non-streaming request at the fake")
        print(f"fake LLM requests seen: {len(seen)}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        fake.shutdown()

    for line in failures:
        print(f"FAIL: {line}")
    print("SMOKE OK" if not failures else f"SMOKE FAILED ({len(failures)} problems)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
