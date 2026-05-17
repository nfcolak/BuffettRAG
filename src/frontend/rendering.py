"""Rendering helpers for the Streamlit frontend."""

from __future__ import annotations

import html
from typing import Any, Dict, List

import streamlit as st

from src.frontend.assets import image_data_uri
from src.frontend.settings import ASSISTANT_AVATAR, USER_AVATAR
from src.frontend.state import get_selected_assistant_message


def format_meta_pills(meta: Dict[str, Any], citations: List[Dict[str, Any]]) -> str:
    items = {
        "strategy": meta.get("strategy", "unknown"),
        "reranked": str(meta.get("reranked", "unknown")),
        "filter": str(meta.get("used_filter") or "none"),
        "citations": str(len(citations or [])),
    }
    return "".join(
        f'<span class="message-status-pill">{html.escape(k)}: {html.escape(v)}</span>'
        for k, v in items.items()
    )


def render_message_status(msg: Dict[str, Any]) -> None:
    role = msg.get("role", "assistant")
    created_at = html.escape(msg.get("created_at", ""))

    if role == "user":
        st.markdown(
            f'<div class="message-status user">{created_at}</div>',
            unsafe_allow_html=True,
        )
        return

    meta = msg.get("meta", {})
    citations = msg.get("citations", [])
    pills_html = format_meta_pills(meta, citations) if isinstance(meta, dict) else ""

    st.markdown(
        f'<div class="message-status assistant">'
        f'<span class="message-status-time">{created_at}</span>'
        f'{pills_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_message(msg: Dict[str, Any], show_sources_button: bool = False) -> None:
    role = msg["role"]
    avatar = ASSISTANT_AVATAR if role == "assistant" else USER_AVATAR
    msg_id = msg["id"]
    is_selected = st.session_state.get("selected_message_id") == msg_id

    with st.chat_message(role, avatar=image_data_uri(avatar)):
        st.markdown(msg["content"])

        if role == "assistant" and show_sources_button:
            source_count = len(msg.get("sources", []))
            if source_count > 0:
                btn_label = (
                    f"📎 {source_count} sources (active)"
                    if is_selected
                    else f"📎 {source_count} sources"
                )
                if st.button(btn_label, key=f"src_btn_{msg_id}", type="secondary"):
                    st.session_state["selected_message_id"] = msg_id
                    st.rerun()

    render_message_status(msg)


def render_sources(hits: List[Dict[str, Any]]) -> None:
    if not hits:
        st.info("No source passages returned.")
        return

    st.markdown('<div class="sources-title">Retrieved Passages</div>', unsafe_allow_html=True)
    for i, passage in enumerate(hits, start=1):
        year = html.escape(str(passage.get("year") or "Unknown"))
        source_file = html.escape(str(passage.get("source_file") or "Unknown file"))
        try:
            score = float(passage.get("score", 0))
            score_label = f"{score:.3f}"
            score_width = max(0, min(100, score * 100))
        except (TypeError, ValueError):
            score_label = "n/a"
            score_width = 0

        topics = [t.strip() for t in str(passage.get("topics", "")).split(",") if t.strip()]
        topic_html = "".join(
            f'<span class="topic-chip">{html.escape(t)}</span>' for t in topics[:4]
        )
        if len(topics) > 4:
            topic_html += f'<span class="topic-chip">+{len(topics) - 4}</span>'

        text = " ".join(str(passage.get("text", "")).split())
        preview = html.escape(text[:320] + ("..." if len(text) > 320 else ""))
        full_text = html.escape(str(passage.get("text", "")))

        st.markdown(
            f'''<div class="source-card">
                <div class="source-card-rail"></div>
                <div class="source-head">
                    <span class="source-year">{year}</span>
                    <span class="source-rank">[{i}]</span>
                    <div class="source-head-spacer"></div>
                    <div class="source-score-mini">
                        <div class="score-bar">
                            <div class="score-bar-fill" style="width:{score_width:.0f}%"></div>
                        </div>
                        <span>{score_label}</span>
                    </div>
                </div>
                <div class="source-file" title="{source_file}">{source_file}</div>
                <div class="source-topics">{topic_html}</div>
                <p class="source-preview">{preview}</p>
                <details class="source-details">
                    <summary>
                        <span class="source-expand-icon">›</span>
                        View full passage
                    </summary>
                    <div class="source-full-text">{full_text}</div>
                </details>
            </div>''',
            unsafe_allow_html=True,
        )


def render_source_panel() -> None:
    selected = get_selected_assistant_message()
    st.markdown('<div class="source-panel-title">Sources</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-kicker">Click "📎 sources" on any answer to inspect its passages.</div>',
        unsafe_allow_html=True,
    )

    if selected is None:
        st.markdown(
            '<div class="source-panel-empty">'
            "Sources will appear here after the first answer."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    msg_idx = next(
        (
            i
            for i, msg in enumerate(st.session_state.get("messages", []))
            if msg.get("id") == selected["id"]
        ),
        None,
    )
    if msg_idx is not None and msg_idx > 0:
        parent_question = st.session_state["messages"][msg_idx - 1].get("content", "")
        if parent_question:
            question = html.escape(parent_question[:120])
            suffix = "..." if len(parent_question) > 120 else ""
            st.markdown(
                f'<div class="source-panel-question">"{question}{suffix}"</div>',
                unsafe_allow_html=True,
            )

    meta = selected.get("meta", {})
    citations = selected.get("citations", [])
    if isinstance(meta, dict):
        summary_items = {
            "strategy": meta.get("strategy", "unknown"),
            "reranked": meta.get("reranked", "unknown"),
            "filter": meta.get("used_filter") or "none",
            "citations": len(citations or []),
        }
    else:
        summary_items = {"details": meta or "n/a"}

    pills = "".join(
        f'<span class="summary-pill">{html.escape(str(k))}: {html.escape(str(v))}</span>'
        for k, v in summary_items.items()
    )
    st.markdown(f'<div class="answer-summary">{pills}</div>', unsafe_allow_html=True)

    with st.container(key="source_scroll"):
        render_sources(selected.get("sources", []))


def render_autoscroll_script(message_count: int) -> None:
    st.markdown(
        f"""<script>
        (function() {{
            const tag = "scroll-{message_count}";
            const findEl = () => {{
                for (const doc of [window.parent && window.parent.document, document]) {{
                    if (!doc) continue;
                    const el = doc.querySelector('.st-key-chat_messages');
                    if (el) return el;
                }}
                return null;
            }};
            const wireOnce = (el) => {{
                if (el.dataset.scrollWired === "1") return;
                el.dataset.scrollWired = "1";
                el.dataset.stickToBottom = "1";
                el.addEventListener('scroll', () => {{
                    el.dataset.stickToBottom =
                        el.scrollHeight - el.scrollTop - el.clientHeight < 60 ? "1" : "0";
                }}, {{ passive: true }});
            }};
            const run = () => {{
                const el = findEl();
                if (!el) return false;
                wireOnce(el);
                if (el.dataset.lastScroll === tag) return true;
                el.dataset.lastScroll = tag;
                if (el.dataset.stickToBottom !== "0") el.scrollTop = el.scrollHeight;
                return true;
            }};
            if (!run()) {{
                let tries = 0;
                const t = setInterval(
                    () => {{ if (run() || ++tries > 20) clearInterval(t); }}, 80
                );
            }}
        }})();
        </script>""",
        unsafe_allow_html=True,
    )
