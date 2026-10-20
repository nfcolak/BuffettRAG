"""finance-recall-75-r3 group 1: the finance reader window is trimmed by a token count.

`token_count.count_tokens` answers from the llama.cpp server's /tokenize, else the Qwen tokenizer.json, else
characters / 4; `ask_flow` trims the served passages with it when FINANCE_WINDOW_TOKENS is on and keeps 64 tokens free.
"""
from __future__ import annotations

import json
import math
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from config import FINANCE_DEFAULTS
from src.generation import token_count
from src.knowledge import KnowledgeStore
from src.services import ask_flow
from src.services.schemas import AskRequest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ask_flow_finance import FakeLLM, TREND_Q  # noqa: E402
from test_finance_budget import _hits_of, _long_docs  # noqa: E402
from test_finance_policy import FakeRetriever  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).parent / "fixtures" / "knowledge_mini"
SETTINGS = ("FINANCE_WINDOW_TOKENS", "REMOTE_LLM_URL", "REMOTE_LLM_TOKEN", "LLM_TOKENIZER_JSON", "LLM_N_CTX",
            "LLM_CONTEXT_PASSAGES", "LLM_CHARS_PER_TOKEN", "FINANCE_K_TREND", "FINANCE_PIPELINE", "HF_HUB_CACHE", "HF_HOME")
TOKEN = "sekret-token-123"
DENSE = "1,234 5,678 9,012 3,456 7,890 12.5% 1.2 $3,400 2019 2020 2021"  # Qwen splits digits: ~1 character per token


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RETRIEVAL_DEDUP", "0")
    token_count.reset()
    yield
    token_count.reset()


class FakeTokenizer:
    """`chars_per_token` characters per token; records the add_special_tokens flag of every call."""

    def __init__(self, chars_per_token: float = 2.0) -> None:
        self.chars_per_token = chars_per_token
        self.special = []

    def encode(self, text, add_special_tokens=True):
        self.special.append(add_special_tokens)
        return SimpleNamespace(ids=[0] * math.ceil(len(text) / self.chars_per_token))


def use_tokenizer(monkeypatch, tokenizer) -> FakeTokenizer:
    monkeypatch.setattr(token_count, "_tokenizer", lambda: tokenizer)
    return tokenizer


class TokenizeStub:
    """A local llama.cpp-style /tokenize endpoint: `script` is a list of (status, body) consumed per request."""

    def __init__(self, chars_per_token: int = 3) -> None:
        self.requests: list = []
        self.script: list = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                status, override = stub.script.pop(0) if stub.script else (200, None)
                payload = override if override is not None else json.dumps(
                    {"tokens": [0] * (len(body["content"]) // chars_per_token)})
                data = payload.encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    server = TokenizeStub()
    yield server
    server.close()


def tiny_tokenizer_file(path: Path) -> Path:
    """A real tokenizer.json (word level): three known words, everything else is one unknown token."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(WordLevel({"alpha": 0, "beta": 1, "[UNK]": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))
    return path


# --- the settings and the margin ---
def test_the_setting_key_exists_and_the_margin_is_64_tokens():
    assert "FINANCE_WINDOW_TOKENS" in FINANCE_DEFAULTS
    assert token_count.WINDOW_MARGIN_TOKENS == 64


def test_window_tokens_is_the_context_minus_the_answer_budget_and_the_margin(monkeypatch):
    assert token_count.reader_window_tokens(900) == 16384 - 900 - 64
    monkeypatch.setenv("LLM_N_CTX", "8192")
    assert token_count.reader_window_tokens(1000) == 8192 - 1000 - 64
    assert token_count.reader_window_tokens(20000) == 0  # never negative


# --- source 1: the llama.cpp server ---
def test_the_server_counts_when_the_setting_is_on(stub, monkeypatch):
    monkeypatch.setenv("FINANCE_WINDOW_TOKENS", "1")
    monkeypatch.setenv("REMOTE_LLM_URL", stub.url + "/v1/")  # /tokenize sits at the server root
    monkeypatch.setenv("REMOTE_LLM_TOKEN", TOKEN)
    text = "x" * 300
    assert token_count.count_tokens_and_source(text) == (100, "server")
    assert token_count.count_tokens(text) == 100
    request = stub.requests[0]
    assert request["path"] == "/tokenize" and request["auth"] == f"Bearer {TOKEN}"
    assert request["body"]["content"] == text and request["body"]["add_special"] is False


def test_the_server_is_not_asked_while_the_setting_is_off(stub, monkeypatch):
    monkeypatch.setenv("REMOTE_LLM_URL", stub.url)
    fake = use_tokenizer(monkeypatch, FakeTokenizer())
    assert token_count.count_tokens_and_source("abcd" * 10) == (20, "tokenizer")
    assert stub.requests == [] and fake.special == [False]


def test_the_server_is_not_asked_without_a_configured_server(monkeypatch):
    monkeypatch.setenv("FINANCE_WINDOW_TOKENS", "1")
    use_tokenizer(monkeypatch, FakeTokenizer())
    assert token_count.count_tokens_and_source("abcd" * 10) == (20, "tokenizer")


def test_server_counts_are_cached_by_text(stub, monkeypatch):
    monkeypatch.setenv("FINANCE_WINDOW_TOKENS", "1")
    monkeypatch.setenv("REMOTE_LLM_URL", stub.url)
    assert token_count.count_tokens("a" * 30) == token_count.count_tokens("a" * 30) == 10
    assert len(stub.requests) == 1
    assert token_count.count_tokens("b" * 60) == 20
    assert len(stub.requests) == 2


def test_a_failing_server_falls_through_and_is_not_asked_again_until_the_pause_ends(stub, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(token_count, "_clock", lambda: now[0])
    monkeypatch.setenv("FINANCE_WINDOW_TOKENS", "1")
    monkeypatch.setenv("REMOTE_LLM_URL", stub.url)
    use_tokenizer(monkeypatch, FakeTokenizer())
    stub.script = [(503, "Loading model")]
    assert token_count.count_tokens_and_source("a" * 40) == (20, "tokenizer")
    assert len(stub.requests) == 1
    for n in range(5):  # the trim loop counts again and again: a sick server costs one request, not five
        assert token_count.count_tokens_and_source("b" * (40 + n))[1] == "tokenizer"
    assert len(stub.requests) == 1
    now[0] += token_count.SERVER_PAUSE_S + 1
    assert token_count.count_tokens_and_source("c" * 60) == (20, "server")
    assert len(stub.requests) == 2


@pytest.mark.parametrize("body", ["not json", "{}", '{"tokens": "many"}', "[1, 2, 3]"])
def test_a_garbage_answer_falls_through(stub, monkeypatch, body):
    monkeypatch.setenv("FINANCE_WINDOW_TOKENS", "1")
    monkeypatch.setenv("REMOTE_LLM_URL", stub.url)
    use_tokenizer(monkeypatch, FakeTokenizer())
    stub.script = [(200, body)]
    assert token_count.count_tokens_and_source("a" * 40) == (20, "tokenizer")


def test_an_unreachable_server_falls_through(monkeypatch):
    monkeypatch.setenv("FINANCE_WINDOW_TOKENS", "1")
    monkeypatch.setenv("REMOTE_LLM_URL", "http://127.0.0.1:1")  # nothing listens
    use_tokenizer(monkeypatch, FakeTokenizer())
    assert token_count.count_tokens_and_source("a" * 40) == (20, "tokenizer")


def test_a_rejected_key_falls_through_without_leaking_it(stub, monkeypatch):
    monkeypatch.setenv("FINANCE_WINDOW_TOKENS", "1")
    monkeypatch.setenv("REMOTE_LLM_URL", stub.url)
    monkeypatch.setenv("REMOTE_LLM_TOKEN", TOKEN)
    use_tokenizer(monkeypatch, FakeTokenizer())
    stub.script = [(401, "unauthorized")]
    count, source = token_count.count_tokens_and_source("a" * 40)
    assert (count, source) == (20, "tokenizer") and TOKEN not in str((count, source))


# --- source 2: tokenizer.json ---
def test_tokenizer_json_from_the_environment_loads_once(tmp_path, monkeypatch):
    path = tiny_tokenizer_file(tmp_path / "tok.json")
    monkeypatch.setenv("LLM_TOKENIZER_JSON", str(path))
    loads = []
    real = token_count._load_tokenizer
    monkeypatch.setattr(token_count, "_load_tokenizer", lambda p: loads.append(p) or real(p))
    assert token_count.count_tokens_and_source("alpha beta alpha gamma") == (4, "tokenizer")
    path.unlink()  # the second count must not read the file again
    assert token_count.count_tokens("beta beta") == 2
    assert loads == [path]


def test_the_shipped_tokenizer_is_the_next_file_when_the_environment_path_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_TOKENIZER_JSON", str(tmp_path / "missing.json"))
    assert token_count.SHIPPED_TOKENIZER == ROOT / "data" / "tokenizer" / "qwen_tokenizer.json"
    assert token_count.count_tokens_and_source("hello world") == (2, "tokenizer")


def test_an_unreadable_tokenizer_file_is_skipped(tmp_path, monkeypatch):
    broken = tmp_path / "broken.json"
    broken.write_text("this is not a tokenizer")
    monkeypatch.setenv("LLM_TOKENIZER_JSON", str(broken))
    assert token_count.count_tokens_and_source("hello world") == (2, "tokenizer")  # the shipped file answered


def test_the_hf_cache_copy_is_the_last_tokenizer_file(tmp_path, monkeypatch):
    monkeypatch.setattr(token_count, "SHIPPED_TOKENIZER", tmp_path / "absent.json")
    snapshot = tmp_path / "hub" / token_count.HF_CACHE_REPO / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    tiny_tokenizer_file(snapshot / "tokenizer.json")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert token_count.count_tokens_and_source("alpha beta gamma") == (3, "tokenizer")


def test_the_shipped_tokenizer_counts_digits_one_by_one():
    count, source = token_count.count_tokens_and_source(DENSE)
    assert source == "tokenizer"
    assert count > 3 * math.ceil(len(DENSE) / 4)  # characters / 4 would say 16
    assert token_count.count_tokens("") == 0


# --- source 3: characters / 4 ---
def test_characters_over_four_when_nothing_else_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(token_count, "SHIPPED_TOKENIZER", tmp_path / "absent.json")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty"))
    assert token_count.count_tokens_and_source("a" * 41) == (11, "chars")  # rounded up, never under
    assert token_count.count_tokens_and_source("a" * 40) == (10, "chars")
    monkeypatch.setenv("LLM_CHARS_PER_TOKEN", "2")
    assert token_count.count_tokens("a" * 41) == 21
    assert token_count.count_tokens("") == 0


def test_the_fallback_order_server_then_tokenizer_then_characters(stub, tmp_path, monkeypatch):
    text = "a" * 120
    monkeypatch.setenv("FINANCE_WINDOW_TOKENS", "1")
    monkeypatch.setenv("REMOTE_LLM_URL", stub.url)
    monkeypatch.setenv("LLM_TOKENIZER_JSON", str(tiny_tokenizer_file(tmp_path / "tok.json")))
    assert token_count.count_tokens_and_source(text) == (40, "server")  # 3 characters per token
    stub.close()
    token_count.reset()  # a new server pause and cache; the dead server now fails
    assert token_count.count_tokens_and_source(text) == (1, "tokenizer")  # one unknown word
    monkeypatch.setattr(token_count, "SHIPPED_TOKENIZER", tmp_path / "absent.json")
    monkeypatch.setenv("LLM_TOKENIZER_JSON", str(tmp_path / "missing.json"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty"))
    token_count.reset()
    assert token_count.count_tokens_and_source(text) == (30, "chars")


# --- the trim in ask_flow ---
@pytest.fixture
def flow(monkeypatch):
    docs = _long_docs()
    retriever = FakeRetriever(_hits_of(docs))
    monkeypatch.setattr(ask_flow, "_state", {"retriever": retriever, "llm": FakeLLM(), "docs_by_id": docs,
                                              "knowledge": KnowledgeStore.load(FIX)})
    monkeypatch.setenv("FINANCE_PIPELINE", "1")
    monkeypatch.setenv("LLM_CONTEXT_PASSAGES", "4")
    monkeypatch.setenv("FINANCE_K_TREND", "12")
    return SimpleNamespace(docs=docs, retriever=retriever)


def _route(**kw):
    return ask_flow.finance_route(AskRequest(query=TREND_Q, **kw))


def test_the_trim_counts_tokens_and_keeps_the_margin_free(flow, monkeypatch):
    use_tokenizer(monkeypatch, FakeTokenizer(chars_per_token=2.0))
    monkeypatch.setenv("FINANCE_WINDOW_TOKENS", "1")
    full = _route()
    assert len(full.hits) == 12 and full.debug["window_trimmed"] == 0
    need = token_count.count_tokens(full.prompt)
    assert full.debug["token_source"] == "tokenizer" and full.debug["prompt_tokens"] == need
    assert "window_chars" not in full.debug

    monkeypatch.setenv("LLM_N_CTX", str(need + 900 + token_count.WINDOW_MARGIN_TOKENS))  # fits with the margin to spare
    exact = _route()
    assert len(exact.hits) == 12 and exact.debug["window_tokens"] == need and exact.debug["window_trimmed"] == 0

    monkeypatch.setenv("LLM_N_CTX", str(need + 900 + token_count.WINDOW_MARGIN_TOKENS - 1))  # one token short
    short = _route()
    assert 4 <= len(short.hits) < 12 and short.debug["window_trimmed"] == 12 - len(short.hits)
    assert short.debug["window_tokens"] == need - 1
    used = token_count.count_tokens(short.prompt)
    assert used == short.debug["prompt_tokens"] <= short.debug["window_tokens"]
    assert used + 900 + token_count.WINDOW_MARGIN_TOKENS <= config.llm_n_ctx()  # prompt + answer + margin fit the window
    kept = {h.id for h in short.hits}
    assert kept == {h.id for h in short.retrieved_hits[:len(short.hits)]}  # the lowest-ranked passages went first


def test_a_number_dense_prompt_that_the_character_window_keeps_is_trimmed_by_tokens(flow, monkeypatch):
    full = _route()  # the character path, a large window: all 12 passages
    assert len(full.hits) == 12
    monkeypatch.setenv("LLM_N_CTX", str(900 + len(full.prompt) // 4 + 1))  # just wide enough in characters
    assert len(_route().hits) == 12 and _route().debug["window_trimmed"] == 0
    use_tokenizer(monkeypatch, FakeTokenizer(chars_per_token=2.0))  # twice the characters / 4 estimate
    monkeypatch.setenv("FINANCE_WINDOW_TOKENS", "1")
    route = _route()
    assert 4 <= len(route.hits) < 12 and route.debug["window_trimmed"] == 12 - len(route.hits)
    assert route.debug["prompt_tokens"] + 900 + token_count.WINDOW_MARGIN_TOKENS <= config.llm_n_ctx()


def test_the_token_trim_never_cuts_below_todays_budget(flow, monkeypatch):
    use_tokenizer(monkeypatch, FakeTokenizer())
    monkeypatch.setenv("FINANCE_WINDOW_TOKENS", "1")
    monkeypatch.setenv("LLM_N_CTX", "1000")  # nothing fits
    route = _route()
    assert len(route.hits) == 4 and route.debug["window_trimmed"] == 8


def test_the_token_trim_uses_the_requests_answer_budget(flow, monkeypatch):
    use_tokenizer(monkeypatch, FakeTokenizer(chars_per_token=4.0))
    monkeypatch.setenv("FINANCE_WINDOW_TOKENS", "1")
    monkeypatch.setenv("LLM_N_CTX", "6000")
    small, big = _route(max_new_tokens=100), _route(max_new_tokens=2000)
    assert len(small.hits) > len(big.hits) >= 4
    assert small.debug["window_tokens"] == 6000 - 100 - 64 and big.debug["window_tokens"] == 6000 - 2000 - 64


def test_without_any_tokenizer_the_trim_counts_characters_over_four(flow, monkeypatch):
    use_tokenizer(monkeypatch, None)
    monkeypatch.setenv("FINANCE_WINDOW_TOKENS", "1")
    route = _route()
    assert route.debug["token_source"] == "chars" and route.debug["prompt_tokens"] == math.ceil(len(route.prompt) / 4)


def test_the_setting_off_keeps_the_character_trim_and_its_debug(flow, monkeypatch):
    fake = use_tokenizer(monkeypatch, FakeTokenizer())
    monkeypatch.setenv("LLM_N_CTX", "6000")
    route = _route()
    assert route.debug["window_chars"] == (6000 - 900) * 4 and len(route.prompt) <= route.debug["window_chars"]
    assert not {"window_tokens", "prompt_tokens", "token_source"} & set(route.debug)
    assert fake.special == []  # nothing was counted


def test_the_server_count_drives_the_trim(flow, stub, monkeypatch):
    monkeypatch.setenv("FINANCE_WINDOW_TOKENS", "1")
    monkeypatch.setenv("REMOTE_LLM_URL", stub.url)
    monkeypatch.setenv("REMOTE_LLM_TOKEN", TOKEN)
    full = _route()
    assert full.debug["token_source"] == "server" and full.debug["prompt_tokens"] == len(full.prompt) // 3
    monkeypatch.setenv("LLM_N_CTX", str(full.debug["prompt_tokens"] + 900 + 64 - 1))  # one token short
    route = _route()
    assert route.debug["token_source"] == "server" and len(route.hits) < 12
    assert route.debug["prompt_tokens"] <= route.debug["window_tokens"]
    assert all(r["auth"] == f"Bearer {TOKEN}" and r["path"] == "/tokenize" for r in stub.requests)


# --- the Modal image ---
def test_the_modal_image_ships_the_tokenizer_file_and_the_package_that_reads_it():
    app = (ROOT / "deploy" / "modal" / "app.py").read_text()
    assert 'TOKENIZER_FILE = "data/tokenizer/qwen_tokenizer.json"' in app
    assert re.search(r'if \(REPO / TOKENIZER_FILE\)\.is_file\(\):[^\n]*\n\s+image = image\.add_local_file\('
                     r'str\(REPO / TOKENIZER_FILE\), f"\{APP_DIR\}/\{TOKENIZER_FILE\}"\)', app)
    assert app.index("add_local_file(str(REPO / TOKENIZER_FILE)") > app.index("image = (")
    assert "_warn_missing_tokenizer()" in app.split("if modal.is_local():\n    _warn_missing_summary_index()", 1)[1]
    assert (ROOT / "data" / "tokenizer" / "qwen_tokenizer.json").is_file()
    assert token_count.SHIPPED_TOKENIZER == ROOT / "data" / "tokenizer" / "qwen_tokenizer.json"
    # the image installs requirements-cloud.txt, and sentence-transformers brings `tokenizers`
    assert "sentence-transformers" in (ROOT / "requirements-cloud.txt").read_text()
    import tokenizers  # noqa: F401
    # llama.cpp runs in the same container: the app reaches it at REMOTE_LLM_URL, so /tokenize is a source there
    assert "REMOTE_LLM_URL=f\"http://127.0.0.1:{LLM_PORT}\"" in app and "REMOTE_LLM_TOKEN=os.environ[\"LLM_TOKEN\"]" in app
