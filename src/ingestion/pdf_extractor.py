"""PDF and text extraction with cleaning.

Primary backend: PyMuPDF (fitz) -- fast, robust on Berkshire letters.
Fallback:        pdfplumber       -- slower but sometimes recovers pages
                                     where fitz returns empty text.

Cleaning steps applied:
    - drop pages that are mostly whitespace
    - strip running headers / footers (e.g. "BERKSHIRE HATHAWAY INC." repeated)
    - de-hyphenate words split across linebreaks ("invest-\\nment" -> "investment")
    - collapse repeated blank lines but preserve paragraph boundaries
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:  # pragma: no cover
    _HAS_FITZ = False

try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:  # pragma: no cover
    _HAS_PDFPLUMBER = False


# -----------------------------------------------------------------------------
# Cleaning helpers
# -----------------------------------------------------------------------------

# Lines that look like running headers/footers in the BRK letters.
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*BERKSHIRE HATHAWAY INC\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,3}\s*$"),                          # bare page number
    re.compile(r"^\s*-\s*\d{1,3}\s*-\s*$"),                  # "- 12 -"
    re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
]


def _is_boilerplate(line: str) -> bool:
    return any(p.match(line) for p in _BOILERPLATE_PATTERNS)


def _dehyphenate(text: str) -> str:
    """Join words split across linebreaks: 'invest-\\nment' -> 'investment'."""
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def clean_extracted_text(text: str) -> str:
    """Normalize extracted text without destroying paragraph structure."""
    if not text:
        return ""

    text = _dehyphenate(text)

    # Filter out boilerplate lines.
    kept = [ln for ln in text.split("\n") if not _is_boilerplate(ln.strip())]
    text = "\n".join(kept)

    # Collapse runs of 3+ newlines into a paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Squeeze internal whitespace on each line but keep linebreaks.
    text = "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n"))
    # Strip leading/trailing whitespace.
    return text.strip()


# -----------------------------------------------------------------------------
# Extractors
# -----------------------------------------------------------------------------

def _extract_with_fitz(pdf_path: Path) -> str:
    pages: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            try:
                page_text = page.get_text("text") or ""
            except Exception as e:
                print(f"  [fitz] page {page_num + 1} of {pdf_path.name} failed: {e}")
                page_text = ""
            if page_text.strip():
                pages.append(page_text)
    return "\n\n".join(pages)


def _extract_with_pdfplumber(pdf_path: Path) -> str:
    pages: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception as e:
                print(f"  [pdfplumber] page {page_num + 1} of {pdf_path.name} failed: {e}")
                page_text = ""
            if page_text.strip():
                pages.append(page_text)
    return "\n\n".join(pages)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract and clean text from a PDF.

    Tries PyMuPDF first; if the result is suspiciously small (<200 chars) and
    pdfplumber is available, falls back to pdfplumber.
    """
    raw = ""
    if _HAS_FITZ:
        try:
            raw = _extract_with_fitz(pdf_path)
        except Exception as e:
            print(f"  [fitz] open failed for {pdf_path.name}: {e}")
            raw = ""

    if len(raw.strip()) < 200 and _HAS_PDFPLUMBER:
        print(f"  fitz output thin for {pdf_path.name}; falling back to pdfplumber")
        try:
            raw = _extract_with_pdfplumber(pdf_path)
        except Exception as e:
            print(f"  [pdfplumber] failed for {pdf_path.name}: {e}")

    return clean_extracted_text(raw)


def load_text_file(text_path: Path) -> str:
    """Load and clean a .txt file."""
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except UnicodeDecodeError:
        # Some early letters are latin-1.
        with open(text_path, "r", encoding="latin-1") as f:
            raw = f.read()
    except Exception as e:
        print(f"Error reading text file {text_path.name}: {e}")
        return ""
    return clean_extracted_text(raw)
