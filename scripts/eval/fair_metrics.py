"""Length-aware companions to the strict supported-claims count, with no model inference.

The strict count is recall-only: extra answer sentences cost nothing, so a 3-sentence dump ties a 1-sentence
selector. This script rescoring a run (same inputs and context rebuilding as rescore_strict.py) adds, per arm:

  strict                claims met by the whole answer (equals rescore_strict)
  met_at_k (k=1,2)      claims met when only the first k answer sentences are kept ("answer" mode: first k of the
                        whole answer, the trunc.py convention; "line" mode: first k of every line, i.e. of every
                        period paragraph of a comparison answer)
  sentences             answer sentences over the answerable cases (and mean per answer)
  precision             share of answer sentences that support >= 1 gold claim of their own case
                        (precision_non_refusal leaves the refusal line out of the denominator)
  sentences_per_met     answer sentences per met claim
  refusals              rows whose answer is the refusal line

  fair_metrics.py RUN.json [RUN2.json ...] [--cases CASES.json] [--ks 1 2] [--scorer-module M] [--out fair.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rescore_strict as rs  # noqa: E402  (also puts the repo root on sys.path)

from src.evaluation.claim_validator import evidence_sentences  # noqa: E402
from src.evaluation.supported_claims import _BULLET_RE, _answer_sentences  # noqa: E402
from src.generation.prompt import REFUSAL_LINE  # noqa: E402


def sentences_per_line(text: str) -> List[List[str]]:
    """Answer sentences grouped by line (a comparison answer has one line per period paragraph)."""
    groups = [[s.strip() for s in evidence_sentences(_BULLET_RE.sub("", line)) if s.strip()]
              for line in (text or "").splitlines()]
    return [g for g in groups if g]


def truncate(text: str, k: Optional[int], mode: str = "answer") -> str:
    """Keep the first k sentences of the answer ('answer') or of every line ('line'); k=None keeps all."""
    if k is None:
        return text
    if mode == "line":
        return "\n".join(s for group in sentences_per_line(text) for s in group[:k])
    return "\n".join(_answer_sentences(text)[:k])


def is_refusal(answer: str) -> bool:
    return REFUSAL_LINE in (answer or "")


def _met(result: Sequence[Dict[str, Any]]) -> int:
    return sum(bool(r["met"]) for r in result)


def fair_metrics(run: Dict[str, Any], cases: Dict[str, Dict[str, Any]], scorer, rebuilder: Optional[Any],
                 ks: Sequence[int] = (1, 2), arm: str = "") -> Dict[str, Any]:
    required = strict = sentences = supported = refusal_sentences = refusals = answers = 0
    met_at = {(mode, k): 0 for mode in ("answer", "line") for k in ks}
    for row in run["rows"]:
        if row.get("status", "scored") != "scored":
            continue
        answer = row.get("answer") or ""
        refusals += is_refusal(answer)
        case = cases[row["qid"]]
        hits, _source, _ok = rs.served_hits(row, rebuilder)
        citations = row.get("citations") or []
        full = scorer(case, answer, citations, hits)
        if not full:  # unanswerable case: no required claims, no precision to measure
            continue
        answers += 1
        required += len(full)
        strict += _met(full)
        for (mode, k) in met_at:
            met_at[(mode, k)] += _met(scorer(case, truncate(answer, k, mode), citations, hits))
        sents = _answer_sentences(answer)
        support = {s for res in full for s in res.get("supporting_sentences") or []}
        sentences += len(sents)
        supported += sum(s in support for s in sents)
        refusal_sentences += sum(is_refusal(s) for s in sents)
    non_refusal = sentences - refusal_sentences
    out: Dict[str, Any] = {
        "arm": arm, "answerable_answers": answers, "required_claims": required, "strict_claims_met": strict}
    for (mode, k), met in met_at.items():
        out[f"met_at_{k}" if mode == "answer" else f"met_at_{k}_per_line"] = met
    out.update({
        "answer_sentences": sentences, "sentences_per_answer": round(sentences / answers, 3) if answers else 0.0,
        "supporting_sentences": supported,
        "supported_sentence_precision": round(supported / sentences, 4) if sentences else 0.0,
        "supported_sentence_precision_non_refusal": round(supported / non_refusal, 4) if non_refusal else 0.0,
        "sentences_per_met_claim": round(sentences / strict, 3) if strict else None,
        "refusals": refusals})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+", help="run_live_benchmark output JSON(s); one arm each")
    ap.add_argument("--cases", type=Path, default=None, help="cases file (default: the run header's `cases`)")
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--scorer-module", default=None, help="module name or .py file exposing supported_claims_met")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--passages", type=int, default=None)
    ap.add_argument("--n-ctx", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write the per-arm metrics here as JSON")
    args = ap.parse_args()

    scorer = rs.load_scorer(args.scorer_module)
    arms = []
    for path in args.runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        cases = rs.load_cases(rs._resolve(run.get("cases", ""), args.cases))
        corpus = rs._resolve(run.get("corpus") or rs.DEFAULT_CORPUS, args.corpus)
        engine = run.get("answer_engine") or {}
        rebuilder = None
        if any(not r.get("context_snapshot") for r in run["rows"] if r.get("status", "scored") == "scored"):
            provider = engine.get("resolved_provider") or engine.get("provider") or "llama"
            passages = args.passages or rs.detect_passages(run, corpus, provider, args.n_ctx)
            rebuilder = rs.ContextRebuilder(corpus, provider, passages, args.n_ctx)
        arms.append(fair_metrics(run, cases, scorer, rebuilder, args.ks, path.stem))
    for arm in arms:
        print(json.dumps(arm, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(arms, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
