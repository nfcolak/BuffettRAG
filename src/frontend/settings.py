"""Shared frontend settings and filesystem paths."""

from __future__ import annotations

import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")

METADATA_V2_FILE = os.path.join(PROCESSED_DATA_DIR, "metadata_v2.json")
METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, "metadata.json")
HISTORY_FILE = os.path.join(BASE_DIR, ".chat_history.json")

ASSISTANT_AVATAR = os.path.join(ASSETS_DIR, "chatbot_avatar.png")
USER_AVATAR = os.path.join(ASSETS_DIR, "user_avatar.png")
HEADER_LOGO = os.path.join(ASSETS_DIR, "header_logo.png")
STYLE_FILE = os.path.join(ASSETS_DIR, "style.css")

DEFAULT_YEARS = list(range(1977, 2025))
DECADES = [1970, 1980, 1990, 2000, 2010, 2020]
MAX_VISIBLE_MESSAGES = 25
MAX_PERSISTED_MESSAGES = 200

EXAMPLE_QUERIES = [
    "How did Buffett react to the 2008 financial crisis?",
    "What does Buffett look for when evaluating a company for acquisition?",
    "How has Buffett's view on technology stocks changed from the 1990s to the 2020s?",
    "What did Buffett say about derivatives?",
    "Buffett on inflation and purchasing power",
]
