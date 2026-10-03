"""Run curated answer benchmark without cloud models or secrets."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.answer_benchmark import evaluate_answer, validate_benchmark_case
from src.generation.prompt import build_cited_prompt
from src.generation.providers.local_provider import LocalProvider
from src.retrieval.bm25 import BM25Retriever
from src.storage import load_chunks_as_docs


def _load_cases(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        validate_benchmark_case(case)
    return payload["cases"]


def run(corpus: Path, cases_path: Path):
    corpus = corpus.resolve()
    cases_path = cases_path.resolve()
    docs = load_chunks_as_docs(corpus)
    bm25 = BM25Retriever(docs)
    answer_engine = LocalProvider()
    rows = []
    for case in _load_cases(cases_path):
        hits = bm25.search(case["query"], top_k=8)
        answer = answer_engine.generate(build_cited_prompt(case["query"], hits))
        score = evaluate_answer(answer, hits, case)
        rows.append({"qid": case["qid"], "answer": answer, "retrieved_ids": [hit.id for hit in hits], **score})
    return {"schema_version": 1, "mode": "offline_bm25_embedded_extractive_not_live_llm_quality",
            "corpus": str(corpus.relative_to(ROOT)),
            "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
            "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            "answer_engine": {"provider": answer_engine.provider_name, "model": answer_engine.model},
            "n_cases": len(rows),
            "accepted": sum(row["accepted"] for row in rows), "rows": rows}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/evaluation/answer_quality_program/answer_benchmark_v3.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.corpus, args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
