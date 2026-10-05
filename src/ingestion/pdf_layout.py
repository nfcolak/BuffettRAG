"""Layout-aware PDF extraction for the Berkshire letters (corpus v4).

Works on PyMuPDF "dict" blocks/lines/spans instead of flat page text:
    - drops repeated running headers/footers and bare page numbers
    - orders blocks by column (two-column pages: left column, then right)
    - joins wrapped lines inside a paragraph and de-hyphenates line-end splits
    - keeps paragraph breaks (one block = one paragraph, blank line between)
    - emits tables (dot-leader / numeric blocks) as their own paragraphs, row by
      row, prefixed with ``[TABLE]`` so they are never merged into prose
    - merges a paragraph that continues across a page break
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pymupdf

TABLE_MARKER = "[TABLE]"
_MARGIN = 0.085           # top/bottom band (fraction of page height) for headers/footers
_PAGE_NUM = re.compile(r"^\s*(?:[-–—]\s*)?\d{1,3}\s*(?:[-–—])?\s*$|^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.I)
_LEADERS = re.compile(r"(?:\s?\.){4,}\s*")
_SEP = re.compile(r"^[\s*•·]{3,}$")
_NUMTOK = re.compile(r"^[\$\(\)\-–—.,%\d]+$|^\(?\d[\d,.]*\)?%?$")
_JUNK_CHARS = str.maketrans({"\u00a0": " ", "\u00ad": "", "\uf0b7": "•", "Š": "-"})
_TERMINAL = tuple(".?!:;\"”’)")


@dataclass
class _Line:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass
class _Block:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    lines: List[_Line]
    kind: str = "prose"   # prose | table | sep

    @property
    def text(self) -> str:
        return " ".join(l.text for l in self.lines)


def _clean(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s.translate(_JUNK_CHARS)).strip()


def _page_blocks(page, pno: int) -> List[_Block]:
    out: List[_Block] = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        lines = []
        for ln in b["lines"]:
            t = _clean("".join(s["text"] for s in ln["spans"]))
            if t:
                x0, y0, x1, y1 = ln["bbox"]
                lines.append(_Line(x0, y0, x1, y1, t))
        if lines:
            x0, y0, x1, y1 = b["bbox"]
            out.append(_Block(pno, x0, y0, x1, y1, lines))
    return out


def _norm(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.lower())).strip()


def _repeated_margin_texts(pages: List[List[_Block]], heights: List[float]) -> set:
    counts: Counter = Counter()
    for blocks, h in zip(pages, heights):
        seen = set()
        for b in blocks:
            if b.y1 < h * _MARGIN * 1.6 or b.y0 > h * (1 - _MARGIN * 1.6):
                n = _norm(b.text)
                if n and len(n) < 90:
                    seen.add(n)
        counts.update(seen)
    need = max(3, int(0.25 * len(pages)))
    return {t for t, c in counts.items() if c >= need}


def _is_margin_noise(b: _Block, h: float, repeated: set) -> bool:
    in_margin = b.y1 < h * _MARGIN * 1.6 or b.y0 > h * (1 - _MARGIN * 1.6)
    if not in_margin:
        return False
    t = b.text.strip()
    return bool(_PAGE_NUM.match(t)) or _norm(t) in repeated


def _is_table(b: _Block) -> bool:
    text = b.text
    if _LEADERS.search(text):
        return True
    toks = text.split()
    if not toks:
        return False
    num = sum(1 for t in toks if _NUMTOK.match(t))
    if len(b.lines) >= 3 and num / len(toks) >= 0.5:
        return True
    avg = sum(len(l.text) for l in b.lines) / len(b.lines)
    if len(b.lines) >= 4 and avg < 22 and num / len(toks) >= 0.3:
        return True
    return False


def _tablish_line(l: _Line) -> bool:
    toks = l.text.split()
    if _LEADERS.search(l.text) or len(l.text) < 45:
        return True
    return sum(1 for t in toks if _NUMTOK.match(t)) / max(1, len(toks)) >= 0.4


def _split_mixed(b: _Block) -> List[_Block]:
    """Separate prose lines glued to a table in one PyMuPDF block."""
    if not _is_table(b):
        return [b]
    runs: List[List[_Line]] = []
    flags: List[bool] = []
    for l in b.lines:
        t = _tablish_line(l)
        if runs and flags[-1] == t:
            runs[-1].append(l)
        else:
            runs.append([l])
            flags.append(t)
    if len(runs) == 1:
        return [b]
    out = []
    for run, t in zip(runs, flags):
        nb = _Block(b.page, min(l.x0 for l in run), min(l.y0 for l in run),
                    max(l.x1 for l in run), max(l.y1 for l in run), run, "table" if t else "prose")
        out.append(nb)
    return out


def _is_short_label(b: _Block) -> bool:
    return len(b.text.split()) <= 8 and not b.text.rstrip().endswith(_TERMINAL)


def _order_page(blocks: List[_Block], width: float) -> List[_Block]:
    """Reading order: full-width blocks by y; two-column runs left-then-right."""
    mid = width / 2
    narrow = [b for b in blocks if (b.x1 - b.x0) < 0.52 * width]
    left = [b for b in narrow if b.x1 <= mid + 12]
    right = [b for b in narrow if b.x0 >= mid - 12]
    two_col = len(left) >= 3 and len(right) >= 3 and \
        sum(1 for b in left if len(b.text) > 120) >= 2 and sum(1 for b in right if len(b.text) > 120) >= 2
    if not two_col:
        return sorted(blocks, key=lambda b: (round(b.y0, 0), b.x0))
    lid, rid = {id(b) for b in left}, {id(b) for b in right}
    full = sorted([b for b in blocks if id(b) not in lid and id(b) not in rid], key=lambda b: b.y0)
    out: List[_Block] = []
    start = -1.0
    bounds = [f.y0 for f in full] + [1e9]
    # y-bands delimited by full-width blocks
    bi = 0
    for fb in full + [None]:
        limit = fb.y0 if fb else 1e9
        band = [b for b in left if start <= b.y0 < limit], [b for b in right if start <= b.y0 < limit]
        out.extend(sorted(band[0], key=lambda b: b.y0))
        out.extend(sorted(band[1], key=lambda b: b.y0))
        if fb:
            out.append(fb)
            start = fb.y0
        bi += 1
    return out


def _join_lines(lines: List[_Line]) -> str:
    out = ""
    for ln in lines:
        t = ln.text
        if not out:
            out = t
        elif re.search(r"[A-Za-z]{2}-$", out) and re.match(r"[a-z]", t):
            out = out[:-1] + t          # de-hyphenate "invest-" + "ment"
        else:
            out += " " + t
    return re.sub(r"\s+", " ", out).strip()


def _split_paragraphs(b: _Block) -> List[str]:
    """Some blocks hold several paragraphs (first-line indent after a short line)."""
    paras, cur = [], [b.lines[0]]
    base = min(l.x0 for l in b.lines)
    for prev, ln in zip(b.lines, b.lines[1:]):
        full = (prev.x1 - b.x0) > 0.8 * (b.x1 - b.x0)
        if ln.x0 - base > 8 and not full and prev.text.rstrip().endswith(_TERMINAL):
            paras.append(cur)
            cur = [ln]
        else:
            cur.append(ln)
    paras.append(cur)
    return [_join_lines(p) for p in paras]


def _render_table(blocks: List[_Block]) -> str:
    lines = [l for b in blocks for l in b.lines]
    lines.sort(key=lambda l: ((l.y0 + l.y1) / 2, l.x0))
    rows: List[List[_Line]] = []
    for l in lines:
        yc = (l.y0 + l.y1) / 2
        if rows and abs(yc - sum((r.y0 + r.y1) / 2 for r in rows[-1]) / len(rows[-1])) <= 3.5:
            rows[-1].append(l)
        else:
            rows.append([l])
    text_rows = []
    for r in rows:
        cells = [c for c in (_clean(_LEADERS.sub(" ", l.text)) for l in sorted(r, key=lambda l: l.x0)) if c]
        if cells:
            text_rows.append(" | ".join(cells))
    return TABLE_MARKER + "\n" + "\n".join(text_rows)


def extract_layout_paragraphs(pdf_path: Path) -> List[str]:
    with pymupdf.open(pdf_path) as doc:
        pages = [_page_blocks(p, i) for i, p in enumerate(doc)]
        heights = [p.rect.height for p in doc]
        widths = [p.rect.width for p in doc]
    repeated = _repeated_margin_texts(pages, heights)
    paras: List[str] = []
    page_start_flags: List[bool] = []   # True when paragraph is the first item of a page
    for pno, blocks in enumerate(pages):
        h = heights[pno]
        blocks = [b for b in blocks if not _is_margin_noise(b, h, repeated)]
        blocks = [nb for b in blocks for nb in _split_mixed(b)]
        blocks = _order_page(blocks, widths[pno])
        for b in blocks:
            if _SEP.match(b.text):
                b.kind = "sep"
            elif b.kind != "table" and _is_table(b):
                b.kind = "table"
            elif b.kind == "table" and not b.lines:
                b.kind = "prose"
        def label_before_table(k: int) -> bool:
            for j in range(k, min(len(blocks), k + 10)):
                if blocks[j].kind == "table":
                    return True
                if blocks[j].kind != "prose" or not _is_short_label(blocks[j]):
                    return False
            return False
        i, first = 0, True
        while i < len(blocks):
            b = blocks[i]
            if b.kind == "sep":
                paras.append("* * * * * * * * * * * *")
                page_start_flags.append(False)
                i += 1
                first = False
                continue
            if b.kind == "table" or (b.kind == "prose" and _is_short_label(b) and label_before_table(i + 1)):
                grp = [b]
                i += 1
                while i < len(blocks) and (blocks[i].kind == "table" or
                                           (blocks[i].kind == "prose" and _is_short_label(blocks[i])
                                            and label_before_table(i + 1))):
                    grp.append(blocks[i])
                    i += 1
                paras.append(_render_table(grp))
                page_start_flags.append(False)
                first = False
                continue
            for p in _split_paragraphs(b):
                paras.append(p)
                page_start_flags.append(first)
                first = False
            i += 1
    # merge paragraphs continued across a page break
    merged: List[str] = []
    for p, first in zip(paras, page_start_flags):
        if (first and merged and not merged[-1].startswith(TABLE_MARKER) and merged[-1] != "* * * * * * * * * * * *"
                and not merged[-1].rstrip().endswith(_TERMINAL) and p[:1].islower()):
            joiner = "" if re.search(r"[A-Za-z]{2}-$", merged[-1]) else " "
            merged[-1] = (merged[-1][:-1] if joiner == "" else merged[-1]) + joiner + p
        else:
            merged.append(p)
    return [m for m in merged if m.strip()]


def extract_layout_text(pdf_path: Path) -> str:
    return "\n\n".join(extract_layout_paragraphs(pdf_path)).strip() + "\n"
