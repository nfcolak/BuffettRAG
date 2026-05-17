"""Streamlit session state and chat persistence helpers."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

import streamlit as st

from src.frontend.settings import HISTORY_FILE, MAX_PERSISTED_MESSAGES


def now_label() -> str:
    return time.strftime("%H:%M")


def new_message_id(role: str) -> str:
    return f"{role}_{uuid.uuid4().hex[:10]}"


def make_user_message(content: str) -> Dict[str, Any]:
    return {
        "id": new_message_id("user"),
        "role": "user",
        "content": content,
        "created_at": now_label(),
    }


def make_assistant_message(
    content: str,
    sources: List[Dict[str, Any]],
    meta: Dict[str, Any],
    citations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "id": new_message_id("assistant"),
        "role": "assistant",
        "content": content,
        "sources": sources,
        "meta": meta,
        "citations": citations,
        "created_at": now_label(),
    }


def load_persisted_messages() -> List[Dict[str, Any]]:
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return []


def persist_messages() -> None:
    try:
        msgs = st.session_state.get("messages", [])[-MAX_PERSISTED_MESSAGES:]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(msgs, f, ensure_ascii=False)
    except OSError:
        pass


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = load_persisted_messages()
    if "selected_message_id" not in st.session_state:
        st.session_state["selected_message_id"] = None
    if "pending_query" not in st.session_state:
        st.session_state["pending_query"] = None
    normalize_messages()


def clear_chat() -> None:
    st.session_state["messages"] = []
    st.session_state["selected_message_id"] = None
    st.session_state.pop("_max_visible", None)
    persist_messages()


def normalize_messages() -> None:
    for i, msg in enumerate(st.session_state.get("messages", [])):
        if "id" not in msg:
            msg["id"] = f"{msg.get('role', 'msg')}_{i}"
        if "created_at" not in msg:
            msg["created_at"] = ""
        if msg.get("role") == "assistant":
            msg.setdefault("sources", [])
            msg.setdefault("citations", [])
            raw_meta = msg.get("meta", {})
            if isinstance(raw_meta, str):
                msg["meta"] = {"details": raw_meta}
            else:
                msg.setdefault("meta", {})


def assistant_messages() -> List[Dict[str, Any]]:
    return [m for m in st.session_state.get("messages", []) if m.get("role") == "assistant"]


def get_selected_assistant_message() -> Optional[Dict[str, Any]]:
    selected_id = st.session_state.get("selected_message_id")
    assistant_msgs = assistant_messages()
    if not assistant_msgs:
        return None
    if selected_id:
        for msg in assistant_msgs:
            if msg["id"] == selected_id:
                return msg
    latest = assistant_msgs[-1]
    st.session_state["selected_message_id"] = latest["id"]
    return latest
