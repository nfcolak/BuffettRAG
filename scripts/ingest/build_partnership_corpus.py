"""Build the Buffett Partnership letters (1957-1970) corpus extension, corpus v5_pl.

Steps (all deterministic):
  1. Extract the Ivey BPL PDF block by block (PyMuPDF), drop page numbers and letterhead,
     rebuild paragraphs and tables, and assign each letter to a year.
  2. Write data/raw/partnership/buffett_partnership_<YYYY>.txt.
  3. Chunk each year with the paragraph_v3 segmentation (src.ingestion.paragraph_index) and
     rename ids to bpl<YYYY>_p<NNNN>, adding "collection": "partnership".
  4. data/processed/chunks_v5_pl.jsonl = every v3 row byte-identical, then the new rows,
     plus chunks_v5_pl_manifest.json.

Year assignment: the PDF only prints "<YYYY> Letter" headings on the first six letters
(1957, 1958, 1959, 1960, the July 1961 letter headed "1960 Letter", and the January 1962 letter
headed "1961 Letter"); a heading names the year, as printed. Every later letter has no
heading, so it takes the year it reports on (LETTER_YEAR, keyed by the dateline).

Usage: python scripts/ingest/build_partnership_corpus.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402

from src.ingestion.paragraph_index import build_paragraph_records  # noqa: E402

DEFAULT_PDF = Path("/Users/necatifurkancolak/AI-Workplace/Artifacts/BuffettRAG/pv7/sources/ivey_bpl.pdf")
V3_FILE = ROOT / "data/processed/chunks_v3_paragraph.jsonl"
V5_FILE = ROOT / "data/processed/chunks_v5_pl.jsonl"
MANIFEST_FILE = ROOT / "data/processed/chunks_v5_pl_manifest.json"
RAW_DIR = ROOT / "data/raw/partnership"
CHUNK_SIZE, OVERLAP, MIN_CHARS = 800, 120, 160
EXTRACTOR = "pymupdf-blocks (scripts/ingest/build_partnership_corpus.py)"

# Letters without a printed "<YYYY> Letter" heading: dateline -> year the letter reports on.
LETTER_YEAR = {
    "July 6, 1962": 1962, "November 1, 1962": 1962, "January 18, 1963": 1962,
    "July 10, 1963": 1963, "November 6, 1963": 1963, "January 18, 1964": 1963,
    "July 8, 1964": 1964, "November 1964": 1964, "January 18, 1965": 1964,
    "July 9, 1965": 1965, "November 1, 1965": 1965, "January 20, 1966": 1965,
    "July 12, 1966": 1966, "November 1966": 1966, "January 25, 1967": 1966,
    "July 12, 1967": 1967, "October 9, 1967": 1967, "January 24, 1968": 1967,
    "July 11, 1968": 1968, "November 1968": 1968, "January 22, 1969": 1968,
    "May 29, 1969": 1969, "October 9, 1969": 1969, "December 5, 1969": 1969,
    "December 26, 1969": 1969, "February 25, 1970": 1970,
}

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_RE = re.compile(rf"^({_MONTHS})(?:\s+(\d{{1,2}})(?:st|nd|rd|th)?)?\s*[,.]?\s+(\d{{4}})\s*$")
HEADER_RE = re.compile(r"^(19\d\d) Letter\s*$")
LETTERHEAD_RES = [re.compile(p, re.I) for p in (
    r"^WARREN E\.? BUFFETT\s*$", r"^5202 Underwood Ave\..*$", r"^\d{3} KIEWIT PLAZA\s*$",
    r"^OMAHA( 31)?,? NEBRASKA( \d+)?\s*$", r"^TELEPHONE [\d-]+\s*$",
)]
PARTNERSHIP_RE = re.compile(r"^BUFFETT PARTNERSHIP[.,] LTD\.?\s*$", re.I)
ENUM_RE = re.compile(r"^\(?[0-9a-zA-Z]{1,2}[.)]$")
TERMINAL = tuple('.!?:;"”*')


def normalize_date(text: str) -> str | None:
    m = DATE_RE.match(text.strip())
    if not m:
        return None
    month, day, year = m.groups()
    return f"{month} {int(day)}, {year}" if day else f"{month} {year}"


def _clean(s: str) -> str:
    s = s.replace("\xa0", " ").replace("\u00ad", "").replace("\ufb01", "fi").replace("\ufb02", "fl")
    return re.sub(r"[ \t]+", " ", s).strip()


def _line_text(line) -> str:
    return "".join(sp["text"] for sp in line["spans"])


HYPHEN_JOINS: list[tuple[str, str, str]] = []  # (left tail, right head, decision) for review


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines. A hyphen at a line end is dropped when it only marks a broken word,
    kept when it belongs to a compound (Dow-Jones, work-out)."""
    out = lines[0]
    for nxt in lines[1:]:
        if re.search(r"[A-Za-z]-$", out):
            if nxt[:1].islower():
                split = _split_word(out, nxt)
                HYPHEN_JOINS.append((out[-14:], nxt[:14], "split" if split else "compound"))
                out = out[:-1] + nxt if split else out + nxt
            else:
                out = out + nxt
        else:
            out = out + " " + nxt
    return out


# Prefixes that make a real compound when a line breaks after the hyphen.
_COMPOUND_PREFIX = {"self", "non", "ex", "well", "long", "short", "year", "one", "two", "three", "four",
                    "five", "six", "ten", "half", "semi", "tax", "work", "sell", "cash", "mid", "high", "low",
                    "full", "all", "better", "poorer", "sub", "per", "twelve", "first", "second", "third",
                    "fifty", "pro", "rata", "above", "below", "of", "by", "important", "slow", "closed", "fall",
                    "over"}


def _split_word(left: str, right: str) -> bool:
    """True if `left-` + `right` is one word broken across lines (drop the hyphen)."""
    prefix = re.search(r"([A-Za-z]+)-$", left).group(1).lower()
    if prefix == "over" and right.startswith("all"):
        return True  # over-/all -> overall
    return prefix not in _COMPOUND_PREFIX


def extract_units(pdf: Path):
    """Yield (page_no, kind, payload) in reading order. kind: header(year) | lh | prose | table."""
    doc = pymupdf.open(pdf)
    removed_page_numbers = 0
    for pno, page in enumerate(doc, start=1):
        height = page.rect.height
        blocks = []
        for b in page.get_text("dict")["blocks"]:
            if b["type"]:
                continue
            lines = []
            for ln in b["lines"]:
                t = _clean(_line_text(ln))
                if t:
                    lines.append((t, ln["bbox"]))
            if not lines:
                continue
            blocks.append({"bbox": b["bbox"], "lines": lines})
        # page number: bare 1-3 digit block in the bottom margin
        kept = []
        for b in blocks:
            if len(b["lines"]) == 1 and re.fullmatch(r"\d{1,3}", b["lines"][0][0]) and b["bbox"][1] > height - 70:
                removed_page_numbers += 1
                continue
            kept.append(b)
        blocks = kept
        # table rows: >= 2 text lines on the same baseline row with a horizontal gap
        tabular = set()
        all_lines = [(i, ln) for i, b in enumerate(blocks) for ln in b["lines"]]
        for i, (bi, (t, bb)) in enumerate(all_lines):
            for bj, (t2, bb2) in all_lines[i + 1:]:
                if ENUM_RE.match(t) or ENUM_RE.match(t2):
                    continue  # "1." / "a." list markers sit beside their text: prose, not a table
                if abs(bb[1] - bb2[1]) < 3 and (bb2[0] - bb[2] > 8 or bb[0] - bb2[2] > 8):
                    tabular.add(bi)
                    tabular.add(bj)
        units = []
        for i, b in enumerate(blocks):
            y0, y1 = b["bbox"][1], b["bbox"][3]
            if i in tabular:
                units.append({"kind": "table", "y0": y0, "y1": y1, "lines": b["lines"]})
            else:
                units.append({"kind": "prose", "y0": y0, "y1": y1, "x1": b["bbox"][2], "lines": b["lines"]})
        # short label blocks touching a table (e.g. "Compounded", "Rate") belong to the table
        for u in units:
            if u["kind"] == "prose" and len(u["lines"]) <= 2 and all(len(t) < 40 for t, _ in u["lines"]):
                for v in units:
                    if v["kind"] == "table" and (0 <= u["y0"] - v["y1"] <= 4 or 0 <= v["y0"] - u["y1"] <= 4):
                        u["kind"] = "table"
                        break
        units.sort(key=lambda u: (u["y0"], u["lines"][0][1][0]))
        for u in units:
            if u["kind"] == "table":
                rows = defaultdict(list)
                for t, bb in sorted(u["lines"], key=lambda x: (x[1][1], x[1][0])):
                    for key in rows:
                        if abs(key - bb[1]) < 3:
                            rows[key].append((bb[0], t))
                            break
                    else:
                        rows[bb[1]].append((bb[0], t))
                text_rows = [" | ".join(t for _, t in sorted(rows[k])) for k in sorted(rows)]
                yield pno, "table", text_rows
                continue
            texts = []
            for t, _bb in u["lines"]:
                m = HEADER_RE.match(t)
                if m:
                    yield pno, "header", int(m.group(1))
                    continue
                if PARTNERSHIP_RE.match(t) or any(r.match(t) for r in LETTERHEAD_RES):
                    yield pno, "lh", t
                    continue
                if normalize_date(t) and len(u["lines"]) <= 6 and not texts:
                    yield pno, "prose", {"text": t, "wraps": False}  # dateline stays its own paragraph
                    continue
                texts.append(t)
            if texts:
                yield pno, "prose", {"text": _join_lines(texts), "wraps": u["x1"] > 520}
    yield 0, "removed_page_numbers", removed_page_numbers


def _mend_split_numbers(rows: list[str]) -> list[str]:
    """Balance-sheet cells wrap mid-number ("$4,07" / "7"): glue a lone digit run onto a truncated amount."""
    out: list[str] = []
    for row in rows:
        if out and re.fullmatch(r"\d{1,2}", row) and re.search(r"\$\d{1,3},\d{1,2}$|\$\d+\.\d$", out[-1]):
            out[-1] += row
        else:
            out.append(row)
    return out


def build_letters(pdf: Path):
    """Return [{'year','date','paragraphs'}] in document order, plus stats."""
    letters, cur = [], None
    prev_pending = ""  # enumerator waiting for its text
    removed = 0

    def start(year=None):
        nonlocal cur
        cur = {"year": year, "date": None, "paragraphs": [], "kinds": [], "_open": True, "page": None}
        letters.append(cur)

    for pno, kind, payload in extract_units(pdf):
        if kind == "removed_page_numbers":
            removed = payload
            continue
        if kind == "header":
            start(payload)
            continue
        if kind == "lh":
            if PARTNERSHIP_RE.match(payload) and (cur is None or cur["paragraphs"]):
                start(None)
            elif cur is None:
                start(None)
            continue
        if cur is None:
            start(None)
        if cur["page"] is None:
            cur["page"] = pno
        if kind == "table":
            if cur["paragraphs"] and cur["kinds"][-1] == "table":
                cur["paragraphs"][-1] += "\n" + "\n".join(payload)
            else:
                cur["paragraphs"].append("\n".join(payload))
                cur["kinds"].append("table")
            continue
        text = payload["text"]
        date = normalize_date(text)
        if date and cur["date"] is None and not cur["paragraphs"]:
            cur["date"] = date
        if prev_pending:
            text = prev_pending + " " + text
            prev_pending = ""
        if ENUM_RE.match(text):
            prev_pending = text
            continue
        last = cur["paragraphs"][-1] if cur["paragraphs"] else None
        if (last is not None and cur["kinds"][-1] == "prose" and cur.get("_wrap")
                and not last.endswith(TERMINAL) and (text[:1].islower() or cur["_wrap"] == "full")):
            cur["paragraphs"][-1] = _join_lines([last, text])
        else:
            cur["paragraphs"].append(text)
            cur["kinds"].append("prose")
        cur["_wrap"] = "full" if payload["wraps"] else "short"
    for let in letters:
        for i, kind in enumerate(let["kinds"]):
            if kind == "table":
                let["paragraphs"][i] = "\n".join(_mend_split_numbers(let["paragraphs"][i].split("\n")))
        if let["year"] is None:
            if let["date"] not in LETTER_YEAR:
                raise SystemExit(f"letter at page {let['page']} has no year: date={let['date']!r}")
            let["year"] = LETTER_YEAR[let["date"]]
    return letters, removed


def write_raw(letters, raw_dir: Path) -> dict[int, Path]:
    by_year: dict[int, list[str]] = defaultdict(list)
    for let in letters:
        by_year[let["year"]].extend(let["paragraphs"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for year in sorted(by_year):
        text = "\n\n".join(p.strip() for p in by_year[year] if p.strip()) + "\n"
        path = raw_dir / f"buffett_partnership_{year}.txt"
        path.write_text(text, encoding="utf-8")
        paths[year] = path
    return paths


def build_rows(paths: dict[int, Path]) -> list[dict]:
    rows: list[dict] = []
    for year, path in sorted(paths.items()):
        recs = build_paragraph_records(path, year=year, chunk_size=CHUNK_SIZE, overlap=OVERLAP,
                                       min_chunk_chars=MIN_CHARS)
        for idx, rec in enumerate(recs):
            rec["id"] = f"bpl{year}_p{idx:04d}"
            rec["previous_chunk_id"] = f"bpl{year}_p{idx - 1:04d}" if idx else None
            rec["next_chunk_id"] = f"bpl{year}_p{idx + 1:04d}" if idx + 1 < len(recs) else None
            rec["collection"] = "partnership"
        rows.extend(recs)
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    ap.add_argument("--v3", type=Path, default=V3_FILE)
    ap.add_argument("--out", type=Path, default=V5_FILE)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_FILE)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = ap.parse_args()

    letters, removed = build_letters(args.pdf)
    paths = write_raw(letters, args.raw_dir)
    new_rows = build_rows(paths)

    v3_bytes = args.v3.read_bytes()
    with args.out.open("wb") as handle:
        handle.write(v3_bytes)
        for row in new_rows:
            handle.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    v3_rows = v3_bytes.count(b"\n")

    per_year = {}
    for year, path in paths.items():
        yrows = [r for r in new_rows if r["year"] == year]
        per_year[str(year)] = {
            "passages": len(yrows),
            "words": len(path.read_text(encoding="utf-8").split()),
            "letters": sum(1 for let in letters if let["year"] == year),
            "source_file": path.name,
            "source_sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": 1,
        "corpus": "v5_pl",
        "source_pdf": {"path": str(args.pdf), "sha256": sha256_file(args.pdf), "pages": 152},
        "extractor": EXTRACTOR + f"; pymupdf {pymupdf.version[0]}",
        "year_assignment": "printed '<YYYY> Letter' heading where present (first six letters); "
                           "otherwise the year the letter reports on (LETTER_YEAR table)",
        "page_numbers_removed": removed,
        "chunk_config": {"size": CHUNK_SIZE, "overlap": OVERLAP, "min_chunk_chars": MIN_CHARS,
                         "chunker": "paragraph_v3"},
        "per_year": per_year,
        "v3_file": {"path": str(args.v3.relative_to(ROOT)) if args.v3.is_relative_to(ROOT) else str(args.v3),
                    "sha256": hashlib.sha256(v3_bytes).hexdigest(), "rows": v3_rows},
        "new_file": {"path": str(args.out.relative_to(ROOT)) if args.out.is_relative_to(ROOT) else str(args.out),
                     "sha256": sha256_file(args.out), "rows": v3_rows + len(new_rows)},
        "rows": {"v3": v3_rows, "partnership": len(new_rows), "total": v3_rows + len(new_rows)},
        "partnership_words": sum(v["words"] for v in per_year.values()),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": manifest["rows"], "per_year": {y: v["passages"] for y, v in per_year.items()}}))


if __name__ == "__main__":
    main()
