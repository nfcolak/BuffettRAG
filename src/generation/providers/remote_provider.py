"""Remote (OpenAI-compatible HTTP) generation provider, e.g. a llama.cpp server on Modal."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterator, Optional

import requests

import config
from config import LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE
from src.generation.prompt import check_prompt_matches_model
from src.generation.providers.llama_provider import split_prompt

_SEED = 42
_RETRY_STATUS = (502, 503, 504)
_BACKOFF_S = 5.0
_BODY_EXCERPT = 200


class RemoteProvider:
    provider_name = "remote"

    def __init__(self, temperature: float = LLM_TEMPERATURE) -> None:
        self.model = config.remote_llm_model()
        check_prompt_matches_model(self.model)  # PROMPT_VERSION must not contradict the model's training prompt
        self.temperature = temperature

    @property
    def endpoint(self) -> str:
        """Chat completions URL; ``/v1`` is appended to the base URL when missing."""
        base = config.remote_llm_url().rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        return base + "/chat/completions"

    def _messages(self, prompt: str):
        system, user = split_prompt(prompt)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return messages

    def _payload(self, prompt: str, max_new_tokens: Optional[int], stream: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(
            model=self.model,
            messages=self._messages(prompt),
            max_tokens=max_new_tokens or LLM_MAX_NEW_TOKENS,
            temperature=self.temperature,
            seed=_SEED,
        )
        if stream:
            payload["stream"] = True
        return payload

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = config.remote_llm_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _scrub(self, text: str) -> str:
        token = config.remote_llm_token()
        return text.replace(token, "***") if token else text

    def _post(self, payload: Dict[str, Any], stream: bool) -> requests.Response:
        """POST with retries on connect errors and 502/503/504 (cold start); other errors raise at once."""
        retries = max(0, config.remote_llm_retries())
        timeout = config.remote_llm_timeout_s()
        last = "no attempt"
        for attempt in range(retries + 1):
            if attempt:
                time.sleep(_BACKOFF_S)
            try:
                response = requests.post(
                    self.endpoint, json=payload, headers=self._headers(), timeout=timeout, stream=stream
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = f"connection error: {type(exc).__name__}"
                continue
            if response.status_code in _RETRY_STATUS and attempt < retries:
                last = f"HTTP {response.status_code}"
                response.close()
                continue
            if response.status_code >= 400:
                body = self._scrub(response.text[:_BODY_EXCERPT]).strip()
                response.close()
                raise RuntimeError(f"Remote LLM error: HTTP {response.status_code}: {body}")
            return response
        raise RuntimeError(f"Remote LLM unreachable after {retries + 1} attempts ({last})")

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        response = self._post(self._payload(prompt, max_new_tokens, stream=False), stream=False)
        try:
            data = response.json()
            return (data["choices"][0]["message"].get("content") or "").strip()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Remote LLM returned an unexpected response ({type(exc).__name__})") from exc
        finally:
            response.close()

    def generate_stream(self, prompt: str, max_new_tokens: Optional[int] = None) -> Iterator[str]:
        response = self._post(self._payload(prompt, max_new_tokens, stream=True), stream=True)
        try:
            for raw in response.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                except (ValueError, KeyError, IndexError, TypeError):
                    continue
                if delta:
                    yield delta
        finally:
            response.close()
