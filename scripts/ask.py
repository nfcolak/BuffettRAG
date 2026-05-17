#!/usr/bin/env python3
"""One-shot CLI for asking BuffettRAG a question.

Examples:
    python scripts/ask.py "What did Buffett say about derivatives?"
    python scripts/ask.py --no-llm "BNSF acquisition"        # retrieval only
    python scripts/ask.py --strategy hybrid --top-k 5 "succession"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import BuffettRAGPipeline, PipelineConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="Question to ask")
    ap.add_argument("--strategy", default="hybrid",
                    choices=["naive", "metadata", "vector", "hybrid"])
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip LLM; just print retrieved passages")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--decade", type=int, default=None)
    ap.add_argument("--json", action="store_true",
                    help="Print full result as JSON")
    args = ap.parse_args()

    where = {}
    if args.year is not None:
        where["year"] = args.year
    if args.decade is not None:
        where["decade"] = args.decade

    cfg = PipelineConfig(use_llm=not args.no_llm, use_reranker=not args.no_rerank)
    pipeline = BuffettRAGPipeline.build(cfg)

    out = pipeline.ask(
        args.query,
        strategy=args.strategy,
        top_k=args.top_k,
        rerank=not args.no_rerank,
        where=where or None,
    )

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print("\n" + "=" * 72)
    print(f"Q: {args.query}")
    print("=" * 72)
    print(f"strategy={out['strategy']}  reranked={out['reranked']}  filter={out['used_filter']}")
    print()

    if out["answer"]:
        print("Answer:")
        print(out["answer"])
        print()
        if out["citations"]:
            print(f"({len(out['citations'])} citation markers found)")

    print("\nRetrieved passages:")
    print("-" * 72)
    for i, p in enumerate(out["passages"], start=1):
        print(f"[{i}] year={p['year']}  score={p['score']:.3f}  topics={p['topics']}")
        snippet = p["text"][:240].replace("\n", " ")
        print(f"    {snippet}{'...' if len(p['text']) > 240 else ''}")
        print()


if __name__ == "__main__":
    main()
