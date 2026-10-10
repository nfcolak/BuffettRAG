"""rescore_strict.py reproduces a stored strict total; merged hits carry the ids they contain."""
import importlib.util
import json
from pathlib import Path

from src.retrieval.context import build_doc_lookup, expand_hits_with_neighbors, fit_context_to_llm
from src.storage import SearchHit, StoredDoc

ROOT = Path(__file__).resolve().parents[1]

_TEXTS = {
    "2005_p0000": "Our insurance float grew again during the year and remained cheap.",
    "2005_p0001": "Berkshire earned 5 percent on equity capital invested in the subsidiary during 2005.",
    "2005_p0002": "Several managers retired during the period and were replaced from inside the company.",
}


def _load_script():
    spec = importlib.util.spec_from_file_location("rescore_strict", ROOT / "scripts/eval/rescore_strict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _chain():
    ids = list(_TEXTS)
    return [StoredDoc(i, _TEXTS[i], {"year": 2005, "source_file": "buffet_2005.txt",
                                     "previous_chunk_id": ids[n - 1] if n else None,
                                     "next_chunk_id": ids[n + 1] if n + 1 < len(ids) else None})
            for n, i in enumerate(ids)]


def test_merged_ids_populated_by_expand_hits_with_neighbors():
    docs = _chain()
    lookup = build_doc_lookup(docs)
    middle = docs[1]
    hit = SearchHit(middle.id, middle.text, dict(middle.metadata), 1.0)

    merged = expand_hits_with_neighbors([hit], lookup, neighbors=1, max_chars=2600)[0]
    assert merged.metadata["merged_ids"] == ["2005_p0001", "2005_p0000", "2005_p0002"]
    assert all(_TEXTS[i] in merged.text for i in merged.metadata["merged_ids"])

    # A budget that leaves no room for neighbours merges nothing but the anchor itself.
    tight = expand_hits_with_neighbors([hit], lookup, neighbors=1, max_chars=len(middle.text) + 1)[0]
    assert tight.metadata["merged_ids"] == ["2005_p0001"]
    assert tight.text == middle.text

    # The 1800-char style anchor truncation drops the ids it cuts off.
    fitted = fit_context_to_llm([merged], [hit], "question", max_new_tokens=10, n_ctx=8192,
                                max_passages=1, passage_max_chars=len(middle.text))[0]
    assert fitted.text == middle.text
    assert fitted.metadata["merged_ids"] == ["2005_p0001"]


def test_rescore_reproduces_stored_total(tmp_path):
    script = _load_script()
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text("\n".join(json.dumps({"id": d.id, "text": d.text, **d.metadata}) for d in _chain()) + "\n",
                      encoding="utf-8")
    claim = {"claim": "Berkshire earned 5 percent on equity capital.", "required_terms": ["5 percent", "equity"],
             "gold_passage_ids": ["2005_p0001"]}
    cases = {c["qid"]: c for c in (
        {"qid": "q1", "question_type": "fact_number", "answerable": True, "gold_passage_ids": ["2005_p0001"],
         "gold_claims": [claim]},
        {"qid": "q2", "question_type": "fact_number", "answerable": True, "gold_passage_ids": ["2005_p0001"],
         "gold_claims": [claim]})}
    cite = [{"marker": "[1]", "passage_ids": ["2005_p0001"]}]
    rows = [
        {"qid": "q1", "query": "What did Berkshire earn on equity?", "history": [], "status": "scored",
         "answer": "Berkshire earned 5 percent on equity capital invested in the subsidiary during 2005. [1]",
         "citations": cite, "passage_ids": ["2005_p0001"], "retrieved_passage_ids": ["2005_p0001"]},
        {"qid": "q2", "query": "What did Berkshire earn on equity?", "history": [], "status": "scored",
         "answer": "Several managers retired during the period. [1]",
         "citations": cite, "passage_ids": ["2005_p0001"], "retrieved_passage_ids": ["2005_p0001"]},
    ]
    run = {"summary": {"strict_supported_claims_met": 1, "strict_required_claims": 2},
           "answer_engine": {"provider": "llama", "resolved_provider": "llama", "model": "tiny-1.5b.gguf"}, "rows": rows}
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(run), encoding="utf-8")

    rebuilder = script.ContextRebuilder(corpus, "llama", 5)
    out = script.rescore(run, cases, script.load_scorer(None), rebuilder, "fixture", "arm")
    assert (out["strict_supported_claims_met"], out["strict_required_claims"]) == (1, 2)
    assert out["rows_with_id_mismatch"] == 0
    first, second = out["claims"]
    assert first["met"] and first["cited_ids"] == ["2005_p0001"]
    assert first["merged_ids_of_cited"] == {"2005_p0001": ["2005_p0001", "2005_p0000", "2005_p0002"]}
    assert first["supporting_sentence"].startswith("Berkshire earned 5 percent")
    assert not second["met"] and second["supporting_sentence"] is None
    assert script.verify(run_path, run, out, None)

    stale = dict(run, summary={"strict_supported_claims_met": 2, "strict_required_claims": 2})
    assert not script.verify(run_path, stale, out, None)
