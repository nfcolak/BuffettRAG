"""Cached frontend asset and metadata loading."""

from __future__ import annotations

import base64
import json
import os
from typing import List

import streamlit as st

from src.frontend.settings import DEFAULT_YEARS, METADATA_FILE, METADATA_V2_FILE, STYLE_FILE


@st.cache_data
def image_data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@st.cache_data
def load_css(_mtime: float) -> str:
    with open(STYLE_FILE, "r", encoding="utf-8") as f:
        return f.read()


@st.cache_data
def load_available_years() -> List[int]:
    for path in (METADATA_V2_FILE, METADATA_FILE):
        try:
            with open(path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        year_stats = metadata.get("year_statistics", {})
        years = sorted(int(y) for y in year_stats.keys())
        if years:
            return years

        start, end = metadata.get("year_range", [None, None])
        if start is not None and end is not None:
            return list(range(int(start), int(end) + 1))

    return DEFAULT_YEARS


def make_rockets_html(count: int = 56) -> str:
    parts = []
    for i in range(count):
        style = (
            f"--start-x:{-18 + (i * 5) % 44}vw;"
            f"--start-y:{-24 + (i * 3) % 26}vh;"
            f"--travel-x:{112 + (i * 9) % 44}vw;"
            f"--travel-y:{112 + (i * 11) % 56}vh;"
            f"--wave:{18 + (i * 17) % 74}px;"
            f"--size:{1.05 + ((i * 9) % 28) / 20:.2f}rem;"
            f"--duration:{13 + (i * 7) % 16}s;"
            f"--delay:{-1 * ((i * 3.7) % 28):.1f}s;"
            f"--opacity:{0.26 + ((i * 5) % 20) / 100:.2f};"
        )
        parts.append(f'<span class="rocket" style="{style}">🚀</span>')
    return f'<div class="rocket-field">{"".join(parts)}</div>'
