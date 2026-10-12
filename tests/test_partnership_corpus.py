"""Corpus v5_pl = v3 rows (byte-identical, same order) + Buffett Partnership letters 1957-1970."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data/processed/chunks_v3_paragraph.jsonl"
V5 = ROOT / "data/processed/chunks_v5_pl.jsonl"
MANIFEST = ROOT / "data/processed/chunks_v5_pl_manifest.json"


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_v5_pl_keeps_v3_prefix_byte_identical_and_ids_unique():
    v3, v5 = V3.read_bytes(), V5.read_bytes()
    assert v5.startswith(v3)
    rows = _rows(V5)
    assert len({r["id"] for r in rows}) == len(rows)
    v3_lines = v3.count(b"\n")
    assert [r.get("collection") for r in rows[:v3_lines]] == [None] * v3_lines
    assert all(r["collection"] == "partnership" for r in rows[v3_lines:])


def test_partnership_rows_follow_the_v3_schema_and_links_stay_within_a_year():
    v3_keys = set(_rows(V3)[0])
    rows = [r for r in _rows(V5) if r.get("collection") == "partnership"]
    assert rows and {y for y in (r["year"] for r in rows)} == set(range(1957, 1971))
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        assert set(r) == v3_keys | {"collection"}
        year = r["year"]
        assert re.fullmatch(rf"bpl{year}_p\d{{4}}", r["id"])
        assert r["source_file"] == f"buffett_partnership_{year}.txt"
        assert r["decade"] == year // 10 * 10
        assert r["id"] == f"bpl{year}_p{r['chunk_index']:04d}"
        for link in (r["previous_chunk_id"], r["next_chunk_id"]):
            assert link is None or by_id[link]["year"] == year
        assert (r["previous_chunk_id"] is None) == (r["chunk_index"] == 0)
        assert (r["next_chunk_id"] is None) == (r["chunk_index"] == r["total_chunks"] - 1)


def test_manifest_matches_files_and_counts():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = _rows(V5)
    new = [r for r in rows if r.get("collection") == "partnership"]
    assert manifest["new_file"]["sha256"] == hashlib.sha256(V5.read_bytes()).hexdigest()
    assert manifest["v3_file"]["sha256"] == hashlib.sha256(V3.read_bytes()).hexdigest()
    assert manifest["rows"] == {"v3": len(rows) - len(new), "partnership": len(new), "total": len(rows)}
    assert len(manifest["source_pdf"]["sha256"]) == 64 and manifest["extractor"]
    for year, info in manifest["per_year"].items():
        assert info["passages"] == sum(1 for r in new if r["year"] == int(year))
        raw = ROOT / "data/raw/partnership" / info["source_file"]
        assert info["words"] == len(raw.read_text(encoding="utf-8").split())
        assert info["source_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()


def test_partnership_raw_text_is_clean():
    for path in sorted((ROOT / "data/raw/partnership").glob("buffett_partnership_*.txt")):
        text = path.read_text(encoding="utf-8")
        assert "KIEWIT PLAZA" not in text and "Underwood Ave" not in text
        assert not re.search(r"(?m)^\d{1,3}$", text), "bare page number left in " + path.name
        assert not re.search(r"[a-z]-\n", text)
        assert not re.search(r"(?m)^\d{4} Letter$", text)


def test_serving_defaults_to_v5_pl_and_year_bounds_reach_1957():
    from config import CHUNKS_V3_FILE, CHUNKS_V5_PL_FILE, FAISS_DIR, START_YEAR
    from src.pipeline import PipelineConfig
    from src.retrieval.retriever import CORPUS_YEAR_MAX, CORPUS_YEAR_MIN, explicit_periods
    from src.services.ask_flow import _resolve_chunks_path

    assert CHUNKS_V5_PL_FILE.name == "chunks_v5_pl.jsonl" and CHUNKS_V3_FILE == CHUNKS_V5_PL_FILE
    assert FAISS_DIR.name == "faiss_v5_pl"
    assert PipelineConfig().chunks_file == CHUNKS_V5_PL_FILE
    assert _resolve_chunks_path() == CHUNKS_V5_PL_FILE
    assert (START_YEAR, CORPUS_YEAR_MIN, CORPUS_YEAR_MAX) == (1957, 1957, 2024)
    assert explicit_periods("What did the 1962 letter say?") == [{"year": 1962}]
    assert explicit_periods("the 1940s") == [] and explicit_periods("in 2030") == []
