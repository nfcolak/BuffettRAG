"""Backend HTTP helpers for the Streamlit frontend."""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests
import streamlit as st

from src.frontend.settings import BACKEND_API_KEY, BACKEND_URL


def build_where(year: Optional[int], decade: Optional[int]) -> Optional[Dict[str, Any]]:
    where: Dict[str, Any] = {}
    if year is not None:
        where["year"] = year
    if decade is not None:
        where["decade"] = decade
    return where or None


def backend_post(path: str, payload: dict, timeout: int = 180) -> Optional[dict]:
    headers = {"X-API-Key": BACKEND_API_KEY} if BACKEND_API_KEY else None
    try:
        response = requests.post(
            f"{BACKEND_URL}{path}",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def ask_backend(
    user_query: str,
    retrieval_strategy: str,
    retrieval_top_k: int,
    retrieval_rerank: bool,
    generate_answer: bool,
    year: Optional[int],
    decade: Optional[int],
) -> Optional[dict]:
    payload = {
        "query": user_query,
        "strategy": retrieval_strategy,
        "top_k": retrieval_top_k,
        "rerank": retrieval_rerank,
        "where": build_where(year, decade),
        "auto_year_filter": True,
    }

    if generate_answer:
        with st.spinner("Retrieving and generating answer..."):
            return backend_post("/ask", payload, timeout=180)

    with st.spinner("Retrieving..."):
        return backend_post("/search", payload, timeout=60)
