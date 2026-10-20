"""Token counting for the finance reader window (finance-recall-75-r3, group 1).

`count_tokens(text)` answers from the first source that works:

1. the llama.cpp server's `POST /tokenize`: exact for the model that reads the prompt. Used only when
   FINANCE_WINDOW_TOKENS is on and REMOTE_LLM_URL is set. Answers are cached by text hash and the call has a short
   timeout. A failure falls through to the next source and pauses this one for a few seconds: the trim loop counts the
   prompt once per dropped passage, so a dead server must not cost a timeout each time;
2. the Qwen `tokenizer.json` through `tokenizers`, loaded once: LLM_TOKENIZER_JSON, else
   data/tokenizer/qwen_tokenizer.json (shipped in the Modal image), else the local HF cache copy of Qwen/Qwen1.5-0.5B
   (the file round 2 measured with). Qwen2.5 shares its BPE vocabulary;
3. characters / LLM_CHARS_PER_TOKEN (4), the estimate the character window uses.

Qwen splits digits one by one (~3.1 characters per token on a number-dense prompt, about 1 on a bare list of
figures), so source 3 can overflow a window that sources 1 and 2 keep.
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import BASE_DIR, finance_setting, llm_n_ctx, remote_llm_token, remote_llm_url, retrieval_setting

SOURCE_SERVER, SOURCE_TOKENIZER, SOURCE_CHARS = "server", "tokenizer", "chars"

# Tokens kept free of the reader prompt, on top of the answer budget: the chat template wraps the prompt in about 20
# tokens (measured in round 2) and the tokenizer.json is not the served model's own file.
WINDOW_MARGIN_TOKENS = 64
SERVER_TIMEOUT_S = 2.0   # /tokenize of a 60 KB prompt takes milliseconds; anything slower is a sick server
SERVER_PAUSE_S = 15.0    # after a failed call the server source is skipped for this long
CACHE_ENTRIES = 512

SHIPPED_TOKENIZER = BASE_DIR / "data" / "tokenizer" / "qwen_tokenizer.json"
HF_CACHE_REPO = "models--Qwen--Qwen1.5-0.5B"

_lock = threading.Lock()
_server_cache: "OrderedDict[Tuple[str, str], int]" = OrderedDict()
_server_paused_until = 0.0
_tokenizers: Dict[str, Any] = {}  # LLM_TOKENIZER_JSON value -> loaded tokenizer, or None when nothing loaded
_clock = time.monotonic


def reset() -> None:
    """Forget the cached server counts, the server pause and the loaded tokenizers (tests, and a changed setting)."""
    global _server_paused_until
    with _lock:
        _server_cache.clear()
        _tokenizers.clear()
        _server_paused_until = 0.0


def reader_window_tokens(max_new_tokens: int) -> int:
    """Tokens the reader prompt may take: the window (`llm_n_ctx()`) minus the answer budget and the safety margin."""
    return max(0, llm_n_ctx() - max(0, max_new_tokens) - WINDOW_MARGIN_TOKENS)


# --- source 1: the llama.cpp server -------------------------------------------------------------------------------
def _server_endpoint() -> str:
    """`/tokenize` sits at the server root, not under the OpenAI `/v1` prefix REMOTE_LLM_URL may carry."""
    base = remote_llm_url().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-len("/v1")]
    return f"{base}/tokenize" if base else ""


def _server_count(text: str) -> Optional[int]:
    global _server_paused_until
    if not finance_setting("FINANCE_WINDOW_TOKENS"):
        return None
    url = _server_endpoint()
    if not url:
        return None
    key = (url, hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).hexdigest())
    with _lock:
        if key in _server_cache:
            _server_cache.move_to_end(key)
            return _server_cache[key]
        if _clock() < _server_paused_until:
            return None
    headers = {"Content-Type": "application/json"}
    token = remote_llm_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = requests.post(url, json={"content": text, "add_special": False}, headers=headers,
                                 timeout=SERVER_TIMEOUT_S)
        response.raise_for_status()
        tokens = response.json()["tokens"]
        if not isinstance(tokens, list):
            raise TypeError("tokens is not a list")
    except Exception:  # noqa: BLE001 - any failure (down, loading, auth, bad body) falls through to the next source
        with _lock:
            _server_paused_until = _clock() + SERVER_PAUSE_S
        return None
    with _lock:
        _server_cache[key] = len(tokens)
        while len(_server_cache) > CACHE_ENTRIES:
            _server_cache.popitem(last=False)
    return len(tokens)


# --- source 2: the Qwen tokenizer.json ----------------------------------------------------------------------------
def _hf_cache_file() -> Optional[Path]:
    root = os.environ.get("HF_HUB_CACHE") or str(Path(os.environ.get("HF_HOME") or "~/.cache/huggingface") / "hub")
    found = sorted(Path(root).expanduser().glob(f"{HF_CACHE_REPO}/snapshots/*/tokenizer.json"))
    return found[0] if found else None


def _tokenizer_files(explicit: str) -> List[Path]:
    files = [Path(explicit).expanduser()] if explicit else []
    files.append(SHIPPED_TOKENIZER)
    cached = _hf_cache_file()
    if cached is not None:
        files.append(cached)
    return [path for path in files if path.is_file()]


def _load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def _tokenizer():
    """The tokenizer, loaded once per LLM_TOKENIZER_JSON value; None when no file loads."""
    explicit = (os.environ.get("LLM_TOKENIZER_JSON") or "").strip()
    with _lock:
        if explicit not in _tokenizers:
            loaded = None
            for path in _tokenizer_files(explicit):
                try:
                    loaded = _load_tokenizer(path)
                    break
                except Exception:  # noqa: BLE001 - an unreadable file or a missing `tokenizers` package: try the next
                    continue
            _tokenizers[explicit] = loaded
        return _tokenizers[explicit]


def _tokenizer_count(text: str) -> Optional[int]:
    tokenizer = _tokenizer()
    if tokenizer is None:
        return None
    try:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)
    except Exception:  # noqa: BLE001 - fall through to the character estimate
        return None


# --- the public counter ---------------------------------------------------------------------------------------------
def count_tokens_and_source(text: str) -> Tuple[int, str]:
    """(tokens, source) of `text`: "server", "tokenizer" or "chars"."""
    count = _server_count(text) if text else None
    if count is not None:
        return count, SOURCE_SERVER
    count = _tokenizer_count(text) if text else None
    if count is not None:
        return count, SOURCE_TOKENIZER
    return math.ceil(len(text) / retrieval_setting("LLM_CHARS_PER_TOKEN")), SOURCE_CHARS


def count_tokens(text: str) -> int:
    """Tokens in `text`, from the first source that works (server, tokenizer.json, characters / 4)."""
    return count_tokens_and_source(text)[0]
