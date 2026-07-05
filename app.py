"""Streamlit frontend.

Deployment: runs on the Nuvolos **Frontend app**.
This is a thin client. It does NOT load any models -- it just calls the
Backend service over HTTP. Backend URL is configured via BACKEND_URL env var
(defaults to http://localhost:8000 for local dev).

Run:
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import streamlit as st

from src.frontend import api, assets, state
from src.frontend.rendering import (
    render_autoscroll_script,
    render_message,
    render_source_panel,
)
from src.frontend.settings import (
    BACKEND_URL,
    DECADES,
    EXAMPLE_QUERIES,
    HEADER_LOGO,
    MAX_VISIBLE_MESSAGES,
    STYLE_FILE,
)


@dataclass(frozen=True)
class QueryOptions:
    strategy: str
    top_k: int
    rerank: bool
    use_llm: bool
    year_filter: Optional[int]
    decade_filter: Optional[int]


def render_page_chrome() -> None:
    st.markdown(
        f"<style>{assets.load_css(os.path.getmtime(STYLE_FILE))}</style>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'''<header class="app-topbar">
            <div class="topbar-brand">
                <div class="topbar-mark">
                    <img src="{assets.image_data_uri(HEADER_LOGO)}" alt="BuffettRAG">
                </div>
                <div class="topbar-brand-copy">
                    <div class="topbar-name">BuffettRAG</div>
                    <div class="topbar-sub">research console</div>
                </div>
            </div>
            <div class="topbar-divider"></div>
            <div class="topbar-corpus">
                <div class="topbar-label">Corpus</div>
                <div class="topbar-value">48 letters <span>/ 1977-2024</span></div>
            </div>
            <div class="topbar-spacer"></div>
            <div class="topbar-status">
                <span class="status-chip"><span class="status-dot"></span>backend <strong>configured</strong></span>
                <span class="status-chip">model <strong>openai</strong></span>
                <span class="status-chip">embed <strong>bge-small</strong></span>
            </div>
        </header>''',
        unsafe_allow_html=True,
    )


def render_left_rail(available_years: List[int]) -> QueryOptions:
    with st.container(key="left_rail"):
        st.markdown(
            '<div class="rail-section-title"><span>#</span> Retrieval</div>',
            unsafe_allow_html=True,
        )
        strategy = st.selectbox(
            "Strategy",
            ["hybrid", "vector", "metadata", "naive"],
            index=0,
            help="hybrid = vector + BM25 fused with RRF (recommended).",
        )
        top_k = st.slider("Top-k passages", 3, 15, 8)
        rerank = st.toggle("Rerank (cross-encoder)", value=True, help="bge-reranker-base")
        use_llm = st.toggle("Generate answer (LLM)", value=True, help="Off = retrieval only")

        st.markdown(
            '<div class="rail-section-title spaced"><span>#</span> Filters</div>',
            unsafe_allow_html=True,
        )
        year_mode = st.selectbox("Scope", ["All years", "Specific year", "Decade"], index=0)
        year_filter: Optional[int] = None
        decade_filter: Optional[int] = None
        if year_mode == "Specific year":
            year_filter = st.selectbox(
                "Year",
                available_years,
                index=len(available_years) - 1,
                help="Available shareholder letter years only.",
            )
        elif year_mode == "Decade":
            decade_filter = st.selectbox(
                "Decade",
                DECADES,
                index=len(DECADES) - 1,
                help="Filter by shareholder letter decade.",
            )

        st.markdown(
            '<div class="rail-section-title spaced"><span>#</span> Example prompts</div>',
            unsafe_allow_html=True,
        )
        for ex_idx, ex_query in enumerate(EXAMPLE_QUERIES):
            if st.button(ex_query, key=f"ex_{ex_idx}", use_container_width=True):
                st.session_state["pending_query"] = ex_query

        st.markdown(
            '<div class="backend-caption">Backend<br><code>{}</code></div>'.format(
                html.escape(BACKEND_URL)
            ),
            unsafe_allow_html=True,
        )
        if st.button("Clear chat", use_container_width=True):
            state.clear_chat()
            st.rerun()

    return QueryOptions(
        strategy=strategy,
        top_k=top_k,
        rerank=rerank,
        use_llm=use_llm,
        year_filter=year_filter,
        decade_filter=decade_filter,
    )


def render_chat_panel() -> Optional[str]:
    with st.container(border=True, key="chat_shell"):
        st.markdown(
            '<div class="chat-panel-head">'
            '<div><div class="chat-panel-title">Research chat</div>'
            '<div class="chat-panel-sub">retrieval -> rerank -> grounded answer</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.container(key="chat_messages"):
            all_messages = st.session_state["messages"]
            if not all_messages:
                st.markdown(
                    '<div class="chat-empty-state">'
                    "Start with a question about Buffett's letters, a specific year, "
                    "or how an idea changed over time."
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                cap = st.session_state.get("_max_visible", MAX_VISIBLE_MESSAGES)
                hidden_count = max(0, len(all_messages) - cap)
                visible_messages = all_messages[-cap:]

                if hidden_count > 0:
                    label = (
                        f"↑ Show {hidden_count} earlier "
                        f"message{'s' if hidden_count > 1 else ''}"
                    )
                    if st.button(label, key="load_earlier", use_container_width=True):
                        st.session_state["_max_visible"] = cap + MAX_VISIBLE_MESSAGES
                        st.rerun()

                for message in visible_messages:
                    render_message(message, show_sources_button=True)

        return st.chat_input(
            "Ask about Buffett's shareholder letters...",
            key="chat_input_widget",
        )


def render_dashboard() -> tuple[Optional[str], QueryOptions]:
    with st.container(key="dashboard_content"):
        left_col, chat_col, source_col = st.columns([0.23, 0.47, 0.30], gap="small")

        with left_col:
            options = render_left_rail(assets.load_available_years())

        with chat_col:
            prompt = render_chat_panel()

        with source_col:
            with st.container(border=True, key="source_panel"):
                render_source_panel()

        render_autoscroll_script(len(st.session_state.get("messages", [])))
        return prompt, options


def resolve_user_query(prompt: Optional[str]) -> Optional[str]:
    if prompt:
        st.session_state["pending_query"] = None
        return prompt.strip() or None
    if st.session_state.get("pending_query"):
        return st.session_state.pop("pending_query")
    return None


def fallback_answer(use_llm: bool, response: Dict[str, Any]) -> str:
    if use_llm and response.get("answer"):
        return response["answer"]
    return (
        "I found the most relevant passages for this query. "
        "Review the source panel for the supporting excerpts."
    )


def handle_query(user_query: str, options: QueryOptions) -> None:
    st.session_state["messages"].append(state.make_user_message(user_query))

    response = api.ask_backend(
        user_query=user_query,
        retrieval_strategy=options.strategy,
        retrieval_top_k=options.top_k,
        retrieval_rerank=options.rerank,
        generate_answer=options.use_llm,
        year=options.year_filter,
        decade=options.decade_filter,
    )
    if response is None:
        st.error("Backend did not respond. Is the server running?")
        st.stop()

    assistant_message = state.make_assistant_message(
        content=fallback_answer(options.use_llm, response),
        sources=response.get("hits", []),
        meta={
            "strategy": response.get("strategy"),
            "reranked": response.get("reranked"),
            "used_filter": response.get("used_filter"),
        },
        citations=response.get("citations", []),
    )
    st.session_state["messages"].append(assistant_message)
    st.session_state["selected_message_id"] = assistant_message["id"]
    state.persist_messages()
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="BuffettRAG", page_icon="📄", layout="wide")
    render_page_chrome()
    state.init_session_state()

    prompt, options = render_dashboard()
    user_query = resolve_user_query(prompt)
    if user_query:
        handle_query(user_query, options)


if __name__ == "__main__":
    main()
