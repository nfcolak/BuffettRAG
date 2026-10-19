"""finance-query-pipeline group 1: KnowledgeStore over the real-passage fixtures in tests/fixtures/knowledge_mini."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from src.knowledge import KnowledgeStore
from src.knowledge.store import default_dir

FIX = Path(__file__).parent / "fixtures" / "knowledge_mini"


def _norm(text: str) -> str:
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def _jsonl(name):
    return [json.loads(line) for line in (FIX / name).read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def store():
    return KnowledgeStore.load(FIX)


@pytest.fixture(scope="module")
def texts():
    return {row["id"]: row["text"] for row in _jsonl("passages.jsonl")}


def test_fixture_matches_schema(texts):
    tax = json.loads((FIX / "taxonomy.json").read_text())
    events = json.loads((FIX / "events.json").read_text())["events"]
    rows = _jsonl("annotations.jsonl")
    assert 30 <= len(rows) <= 50
    for row in rows:
        assert row["id"] in texts and row["scope"] in tax["scope"]
        assert set(row["sectors"]) <= set(tax["sectors"]) and set(row["themes"]) <= set(tax["themes"])
        assert set(row["events"]) <= set(events)
        if row["is_table"]:
            assert row["scope"] == "financial"
        if row["scope"] != "financial":
            assert not (row["sectors"] or row["themes"] or row["events"] or row["stance"]) and row["summary"] == ""
        for stance in row["stance"]:
            assert _norm(stance["quote"]) in _norm(texts[row["id"]])
    assert {r["scope"] for r in rows} == set(tax["scope"]) - {"boilerplate_other"}


def test_numbers_and_digests_are_grounded(texts):
    nums = _jsonl("numbers.jsonl")
    assert len(nums) == 10
    for row in nums:
        assert row["as_written"] in texts[row["source_id"]]
    for digest in _jsonl("digests.jsonl"):
        for point in digest["timeline"]:
            assert _norm(point["quote"]) in _norm(texts[point["source_id"]])


def test_annotation_and_scope_flags(store):
    assert len(store) == 30
    assert store.annotation("2008_p0011")["events"] == ["financial_crisis_2008"]
    assert store.annotation("0000_p0000") is None
    assert store.is_financial("2008_p0011") and not store.is_financial("2001_p0111")
    assert not store.is_financial("1984_p0029") and not store.is_financial("1985_p0013")
    assert store.has_personal("1983_p0020") and not store.has_personal("2008_p0011")
    assert store.is_table("1991_p0083") and store.is_financial("1991_p0083") and not store.is_table("2008_p0011")


def test_passages_with(store):
    assert set(store.passages_with(event="financial_crisis_2008")) == {
        "2008_p0011", "2008_p0067", "2008_p0093", "2010_p0119", "2010_p0142"}
    assert "1990_p0102" in store.passages_with(sector="airlines_aviation")
    assert "1991_p0083" in store.passages_with(sector="airlines_aviation")
    assert set(store.passages_with(company="USAir", sector="airlines_aviation")) >= {"1990_p0101", "1991_p0083"}
    assert store.passages_with(theme="inflation") == ["1994_p0004", "1981_p0042"]
    assert "1986_p0118" in store.passages_with(sector="oil_gas") and "1985_p0013" not in store.passages_with(sector="oil_gas")
    assert set(store.passages_with(sector=["oil_gas", "banking"])) >= {"1986_p0118", "1990_p0084"}
    assert store.passages_with() == [] and store.passages_with(theme="no_such_theme") == []


def test_numbers_lookup(store):
    assert len(store.numbers()) == 10
    row = store.numbers(metric="holding_cost", year=1991, entity="usair")[0]
    assert row["value"] == 358 and row["source_id"] == "1991_p0083"
    assert [n["metric"] for n in store.numbers(year=1987)] == [
        "gain_in_net_worth", "book_value_per_share_change_pct", "book_value_per_share"]
    assert store.numbers(metric="holding_cost", year="1977")[0]["entity"] == "Washington Post"
    assert store.numbers(metric="combined_ratio") == []


def test_digests_topics_events_companies(store):
    assert store.topics() == ["financial_crisis_2008", "airlines_aviation"]
    digest = store.digest("financial_crisis_2008")
    assert digest["kind"] == "event" and len(digest["timeline"]) == 4
    assert store.digest("sector:airlines_aviation")["stance_overall"] == "negative"
    assert store.digest("event:airlines_aviation") is None and store.digest("nothing") is None
    events = store.events()
    assert "credit crisis" in events["financial_crisis_2008"]["aliases"]
    assert events["financial_crisis_2008"]["event_years"] == [2007, 2008, 2009]
    assert "USAir" in store.company_names() and store.company_names() == sorted(store.company_names())


def test_coverage_and_partial_store(store):
    assert store.coverage(["2008_p0011", "2001_p0111", "1996_p0039", "2005_p0127"]) == 0.5
    assert store.coverage([]) == 0.0
    partial = KnowledgeStore(annotations=[a for a in _jsonl("annotations.jsonl") if a["id"].startswith("2008")])
    assert partial.coverage(["2008_p0011", "1996_p0039"]) == 0.5
    for pid in ("1996_p0039", "2005_p0127", "2011_p0079"):
        assert partial.annotation(pid) is None and partial.is_financial(pid) and not partial.has_personal(pid)
        assert not partial.is_table(pid)
    assert partial.numbers() == [] and partial.topics() == [] and partial.events() == {}


def test_empty_store_and_missing_dir(tmp_path):
    for empty in (KnowledgeStore.empty(), KnowledgeStore.load(tmp_path / "absent"), KnowledgeStore.load(tmp_path)):
        assert len(empty) == 0 and empty.annotation("2008_p0011") is None
        assert empty.is_financial("2008_p0011") and not empty.has_personal("2008_p0011")
        assert not empty.is_table("2008_p0011") and empty.coverage(["2008_p0011"]) == 0.0
        assert empty.passages_with(sector="oil_gas") == [] and empty.numbers(metric="holding_cost") == []
        assert empty.digest("financial_crisis_2008") is None and empty.topics() == []
        assert empty.events() == {} and empty.company_names() == []


def test_manifest_checked_when_present(tmp_path):
    shutil.copytree(FIX, tmp_path / "k")
    root = tmp_path / "k"
    assert len(KnowledgeStore.load(root)) == 30  # no MANIFEST: nothing to check
    files = {n: hashlib.sha256((root / n).read_bytes()).hexdigest()
             for n in ("annotations.jsonl", "numbers.jsonl", "digests.jsonl")}
    (root / "MANIFEST").write_text(json.dumps({"files": files}))
    assert len(KnowledgeStore.load(root)) == 30
    (root / "MANIFEST").write_text("".join(f"{sha}  {name}\n" for name, sha in files.items()))
    assert len(KnowledgeStore.load(root)) == 30
    with (root / "annotations.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="annotations.jsonl"):
        KnowledgeStore.load(root)
    (root / "MANIFEST").write_text(json.dumps({"gone.jsonl": "0" * 64}))
    with pytest.raises(ValueError, match="gone.jsonl"):
        KnowledgeStore.load(root)


def test_default_dir_reads_env_at_call_time(monkeypatch):
    monkeypatch.delenv("KNOWLEDGE_DIR", raising=False)
    assert default_dir().parts[-2:] == ("data", "knowledge")
    monkeypatch.setenv("KNOWLEDGE_DIR", str(FIX))
    assert default_dir() == FIX and len(KnowledgeStore.load()) == 30
