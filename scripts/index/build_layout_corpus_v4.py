"""Build the side-by-side layout-aware corpus v4 (active corpus v3 is untouched).

Writes data/raw_layout_v4/buffet_<year>.txt for every PDF year, the v4 chunk file
(same paragraph chunker/config as v3; non-PDF years identical to v3), a manifest,
the v3->v4 id map (token Jaccard), and the v3-vs-v4 audit/retrieval measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/index"))
from rebuild_paragraph_index import _map_old_to_new  # noqa: E402
from src.evaluation.gold_set import get_gold_queries  # noqa: E402
from src.evaluation.retrieval_metrics import aggregate_metrics, per_query_metrics  # noqa: E402
from src.ingestion.chunker import split_text  # noqa: E402
from src.ingestion.legacy_utils import get_all_letters_sorted  # noqa: E402
from src.ingestion.paragraph_index import build_paragraph_records  # noqa: E402
from src.ingestion.pdf_layout import TABLE_MARKER, extract_layout_text  # noqa: E402
from src.ingestion.topic_tagger import tag_topics  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.storage import load_chunks_as_docs  # noqa: E402

CONFIDENT = 0.5
SIZE, OVERLAP, MIN_CHARS = 800, 120, 160


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layout_records(pdf: Path, txt: Path, year: int) -> list[dict]:
    text = txt.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks = split_text("\n\n".join(paragraphs), SIZE, OVERLAP, MIN_CHARS)
    recs = []
    for idx, chunk in enumerate(chunks):
        n_par = max(1, len([p for p in re.split(r"\n\s*\n+", chunk) if p.strip()]))
        recs.append({"id": f"{year}_p{idx:04d}", "text": chunk, "year": year, "decade": (year // 10) * 10,
                     "source_file": pdf.name, "chunk_index": idx, "topics": ",".join(tag_topics(chunk)),
                     "provenance": {"source_sha256": sha(pdf), "extraction_method": "pymupdf_layout_v4",
                                    "layout_text_file": f"data/raw_layout_v4/{txt.name}",
                                    "layout_text_sha256": sha(txt), "paragraph_count": n_par,
                                    "chunker": "paragraph_v3"}})
    for idx, r in enumerate(recs):
        r["total_chunks"] = len(recs)
        r["previous_chunk_id"] = recs[idx - 1]["id"] if idx else None
        r["next_chunk_id"] = recs[idx + 1]["id"] if idx + 1 < len(recs) else None
    return recs


# ---------------------------------------------------------------- audit
_TERM = (".", "?", "!", ":", ";", '"', "”", "’", ")")
_HYPH = re.compile(r"[A-Za-z]-[ \t]*\n[ \t]*[a-z]|[a-z]{2}- [a-z]{2}")
_HEADER = re.compile(r"^\s*(Page \d+( of \d+)?|\d{4} Annual Report)\s*$", re.I)
_BARE = re.compile(r"^\s*(?:[-–—]\s*)?\d{1,3}\s*(?:[-–—])?\s*$")


_NUMLINE = re.compile(r"^[\s$()\-–.,%\d|]+$|\.{4,}|(?:\. ){4,}")


def _is_table_para(para: str, lines: list) -> bool:
    if para.startswith(TABLE_MARKER) or sum(1 for l in lines if " | " in l) * 2 >= len(lines) > 1:
        return True
    return False


def table_in_prose(para: str) -> bool:
    """Unmarked table text: most lines are numbers / dot leaders."""
    lines = [l for l in para.split("\n") if l.strip()]
    return len(lines) >= 4 and sum(1 for l in lines if _NUMLINE.search(l)) / len(lines) >= 0.5


def audit_chunk(text: str) -> tuple[int, int, int, int]:
    breaks = hyph = resid = tip = 0
    for para in re.split(r"\n\s*\n+", text):
        lines = [l for l in para.split("\n")]
        if _is_table_para(para, lines):
            continue
        tip += table_in_prose(para)
        for a, b in zip(lines, lines[1:]):
            if a.strip() and b.strip() and not a.rstrip().endswith(_TERM) and not _BARE.match(a) and not _BARE.match(b):
                breaks += 1
        hyph += len(_HYPH.findall(para))
        bare = sum(1 for l in lines if _BARE.match(l))
        resid += sum(1 for l in lines if _HEADER.match(l))
        if bare and (len(lines) == 1 or (len(lines) >= 2 and bare / len(lines) <= 0.34)):
            resid += bare
    return breaks, hyph, resid, tip


def audit(rows: list[dict]) -> dict:
    by_year: dict = {}
    for r in rows:
        by_year.setdefault(r["year"], []).append(r)
    out = {}
    for y, rs in sorted(by_year.items()):
        a = [audit_chunk(r["text"]) for r in rs]
        out[str(y)] = {"chunks": len(rs), "mean_chunk_chars": round(sum(len(r["text"]) for r in rs) / len(rs), 1),
                       "line_break_in_sentence": sum(x[0] for x in a), "hyphen_break": sum(x[1] for x in a),
                       "header_page_number_residue": sum(x[2] for x in a),
                       "table_paragraphs_unmarked_in_prose": sum(x[3] for x in a),
                       "table_paragraphs": sum(len(re.findall(r"(?m)^\[TABLE\]$", r["text"])) for r in rs)}
    return out


def total(per_year: dict, years) -> dict:
    ks = ["chunks", "line_break_in_sentence", "hyphen_break", "header_page_number_residue", "table_paragraphs", "table_paragraphs_unmarked_in_prose"]
    t = {k: sum(per_year[str(y)][k] for y in years) for k in ks}
    n = t["chunks"]
    t["mean_chunk_chars"] = round(sum(per_year[str(y)]["mean_chunk_chars"] * per_year[str(y)]["chunks"] for y in years) / n, 1)
    return t


# ---------------------------------------------------------------- retrieval
def gold_eval(rows: list[dict]) -> dict:
    from src.storage import StoredDoc
    docs = [StoredDoc(id=r["id"], text=r["text"], metadata={k: v for k, v in r.items() if k != "text"}) for r in rows]
    bm = BM25Retriever(docs)
    per = {g.qid: per_query_metrics(bm.search(g.query, top_k=10), g) for g in get_gold_queries()}
    return {k: round(v, 4) for k, v in aggregate_metrics(per).means.items()}, per, bm


def hn_eval(bm, cases, idmap=None) -> dict:
    rows, passed = [], 0
    for c in cases:
        rel = [idmap[i] if idmap else i for i in c["relevant_ids"]]
        neg = [idmap[i] if idmap else i for i in c["hard_negative_ids"]]
        ranked = [h.id for h in bm.search(c["query"], top_k=20)]
        rr = [ranked.index(i) + 1 for i in rel if i in ranked]
        nr = [ranked.index(i) + 1 for i in neg if i in ranked]
        ok = bool(rr) and (not nr or min(rr) < min(nr))
        passed += ok
        rows.append({"qid": c["qid"], "relevant_ids": rel, "hard_negative_ids": neg,
                     "best_relevant_rank": min(rr) if rr else None, "best_negative_rank": min(nr) if nr else None,
                     "passed": ok})
    return {"n_cases": len(cases), "passed": passed, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-out", type=Path, default=ROOT / "data/raw_layout_v4")
    ap.add_argument("--chunks", type=Path, default=ROOT / "data/processed/chunks_v4_layout.jsonl")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/processed/chunks_v4_layout_manifest.json")
    ap.add_argument("--idmap", type=Path, default=ROOT / "data/processed/chunk_id_map_v3_to_v4.jsonl")
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "data/evaluation/layout_v4")
    ap.add_argument("--raw-main", type=Path, default=None, help="dir holding raw PDFs/txt (default repo data/raw)")
    args = ap.parse_args()

    args.raw_out.mkdir(parents=True, exist_ok=True)
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    records, methods = [], {}
    for year, path, _ in get_all_letters_sorted():
        if path.suffix.lower() == ".pdf":
            txt = args.raw_out / f"buffet_{year}.txt"
            txt.write_text(extract_layout_text(path), encoding="utf-8")
            records.extend(layout_records(path, txt, year))
            methods[year] = "pymupdf_layout_v4"
        else:
            records.extend(build_paragraph_records(path, year=year, chunk_size=SIZE, overlap=OVERLAP,
                                                   min_chunk_chars=MIN_CHARS))
            methods[year] = "text"
    with args.chunks.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    v3_path = ROOT / "data/processed/chunks_v3_paragraph.jsonl"
    v3_docs = load_chunks_as_docs(v3_path)
    v3_rows = [{"id": d.id, "text": d.text, **d.metadata} for d in v3_docs]
    mapping = _map_old_to_new(v3_docs, records)
    for m in mapping:
        m["confident"] = m["best_token_jaccard"] >= CONFIDENT
    with args.idmap.open("w", encoding="utf-8") as fh:
        for m in mapping:
            fh.write(json.dumps(m, ensure_ascii=False, sort_keys=True) + "\n")

    pdf_years = [y for y, m in methods.items() if m != "text"]
    unconf = [m["old_id"] for m in mapping if not m["confident"]]
    unconf_pdf = [i for i in unconf if int(i[:4]) in pdf_years]
    manifest = {"schema_version": 1, "corpus": "paragraph_v4_layout", "source_files": len(methods),
                "chunks": len(records), "chunk_config": {"size": SIZE, "overlap": OVERLAP, "min_chunk_chars": MIN_CHARS},
                "corpus_sha256": sha(args.chunks), "mapping_sha256": sha(args.idmap),
                "extraction_method_by_year": {str(y): m for y, m in methods.items()},
                "layout_text_sha256_by_year": {str(y): sha(args.raw_out / f"buffet_{y}.txt") for y in pdf_years},
                "chunks_by_year": {str(y): sum(1 for r in records if r["year"] == y) for y in methods},
                "id_map": {"confident_threshold": CONFIDENT, "v3_ids": len(mapping),
                           "not_confident": len(unconf), "not_confident_pdf_years": len(unconf_pdf)},
                "vector_index": {"status": "not_built", "reason": "side-by-side corpus; no vector build"}}
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- audit
    a3, a4 = audit(v3_rows), audit(records)
    weak_a = [y for y in pdf_years if y <= 2003]
    weak_b = [y for y in pdf_years if y >= 2008]
    groups = {"pdf_years_all": pdf_years, "1998-2003": weak_a, "2008-2024": weak_b,
              "2004-2007": [y for y in pdf_years if 2004 <= y <= 2007]}
    audit_json = {"definitions": {
        "line_break_in_sentence": "single newline inside a (non-[TABLE]) paragraph where the previous line does not end in terminal punctuation",
        "hyphen_break": "word-hyphen at a line end or 'xx- yy' split",
        "header_page_number_residue": "'Page n'/'YYYY Annual Report' lines or bare 1-3 digit lines inside prose paragraphs",
        "table_paragraphs": "[TABLE]-marked paragraphs (v4 only)",
        "table_paragraphs_unmarked_in_prose": "paragraphs with >=4 lines, >=50% numeric/dot-leader lines, not marked as table"},
        "v3": {"per_year": a3, "totals": {g: total(a3, ys) for g, ys in groups.items()}},
        "v4": {"per_year": a4, "totals": {g: total(a4, ys) for g, ys in groups.items()}}}
    (args.eval_dir / "audit.json").write_text(json.dumps(audit_json, indent=2) + "\n", encoding="utf-8")

    # ---- retrieval
    m3, per3, bm3 = gold_eval(v3_rows)
    m4, per4, bm4 = gold_eval(records)
    cases = json.loads((ROOT / "data/evaluation/answer_quality_program/hard_negatives_v3.json").read_text())["cases"]
    idmap = {m["old_id"]: m["new_ids"][0] for m in mapping}
    h3 = hn_eval(bm3, cases)
    h4 = hn_eval(bm4, cases, idmap)
    for c, row in zip(cases, h4["rows"]):
        row["map_confident"] = all(next(m for m in mapping if m["old_id"] == i)["confident"]
                                   for i in c["relevant_ids"] + c["hard_negative_ids"])
    ret = {"gold_set": {"n_queries": len(per3), "v3": m3, "v4": m4,
                        "delta": {k: round(m4[k] - m3[k], 4) for k in m3}},
           "hard_negatives_v3": {"v3": {k: h3[k] for k in ("n_cases", "passed")} | {"rows": h3["rows"]},
                                 "v4_mapped": {k: h4[k] for k in ("n_cases", "passed")} | {"rows": h4["rows"]}},
           "per_query_gold": {"v3": per3, "v4": per4}}
    (args.eval_dir / "retrieval.json").write_text(json.dumps(ret, indent=2) + "\n", encoding="utf-8")

    keys = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "year_hit_rate@5"]
    t3, t4 = audit_json["v3"]["totals"]["pdf_years_all"], audit_json["v4"]["totals"]["pdf_years_all"]
    md = ["# Layout corpus v4 vs v3 (BM25, offline)", "",
          f"Gold set: {len(per3)} queries. Hard negatives: {len(cases)} cases (v4 ids mapped via v3->v4 map).", "",
          "| corpus | chunks | " + " | ".join(keys) + " | hard-neg passed |",
          "|---|---|" + "---|" * len(keys) + "---|",
          f"| v3 paragraph | {len(v3_rows)} | " + " | ".join(f"{m3[k]:.3f}" for k in keys) + f" | {h3['passed']}/{h3['n_cases']} |",
          f"| v4 layout | {len(records)} | " + " | ".join(f"{m4[k]:.3f}" for k in keys) + f" | {h4['passed']}/{h4['n_cases']} |", "",
          "## Audit, PDF years 1998-2024 (v3 -> v4)", "",
          "| metric | v3 | v4 |", "|---|---|---|"]
    for k in ["chunks", "mean_chunk_chars", "line_break_in_sentence", "hyphen_break", "header_page_number_residue", "table_paragraphs", "table_paragraphs_unmarked_in_prose"]:
        md.append(f"| {k} | {t3[k]} | {t4[k]} |")
    md += ["", f"v3 ids without a confident v4 match (token Jaccard < {CONFIDENT}): {len(unconf)} of {len(mapping)} "
           f"({len(unconf_pdf)} in PDF years)."]
    (args.eval_dir / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
