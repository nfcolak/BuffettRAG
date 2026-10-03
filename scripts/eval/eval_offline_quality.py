"""Reproducible OFFLINE diagnostics, not a cloud answer-quality benchmark.

Run with PYTHON_DOTENV_DISABLED=1 python3 scripts/eval/eval_offline_quality.py --output PATH
Uses the real corpus, existing 50-query gold set, BM25 and local extractive answers.
No embedding/reranker model downloads, database connections or provider calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.gold_set import get_gold_queries
from src.evaluation.retrieval_metrics import aggregate_metrics, per_query_metrics
from src.generation.prompt import build_cited_prompt
from src.generation.providers.local_provider import LocalProvider
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.context import build_doc_lookup, expand_hits_with_neighbors
from src.vector_store import load_chunks_as_docs


def evaluate():
    path = ROOT / 'data/processed/chunks_v2.jsonl'
    docs = load_chunks_as_docs(path)
    lookup = build_doc_lookup(docs)
    bm25 = BM25Retriever(docs)
    metrics, answers = {}, []
    over_budget = 0
    for gold in get_gold_queries():
        hits = bm25.search(gold.query, top_k=8)
        metrics[gold.qid] = per_query_metrics(hits, gold, ks=(1, 3, 5, 8))
        context = expand_hits_with_neighbors(hits, lookup, max_chars=2600)
        over_budget += sum(len(h.text) > 2600 for h in context)
        answer = LocalProvider().generate(build_cited_prompt(gold.query, context))
        answers.append({'qid': gold.qid, 'query': gold.query, 'answer': answer,
                        'retrieved_ids': [h.id for h in hits],
                        'context_passages': [asdict(h) for h in context]})
    return {'mode': 'offline_bm25_local_extractive_not_cloud_quality',
            'corpus_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'n_chunks': len(docs), 'n_queries': len(answers),
            'retrieval_keyword_proxy': asdict(aggregate_metrics(metrics)),
            'out_of_vocabulary_hits': len(bm25.search('xyzzyqzzzzz')),
            'context_passages_over_budget': over_budget,
            'answers': answers}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != 'answers'}, indent=2))
