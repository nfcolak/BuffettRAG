"""Rescore a saved benchmark result with the NLI claim scorer; writes <result>.nli.json."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def rescore(result_path: Path, cases_path: Path | None = None, scorer=None) -> dict:
    from src.evaluation.nli_scorer import NliScorer

    result = json.loads(result_path.read_text(encoding="utf-8"))
    cases_path = cases_path or (ROOT / result["cases"])
    cases = {c["qid"]: c for c in json.loads(cases_path.read_text(encoding="utf-8"))["cases"]}
    scorer = scorer or NliScorer()
    rows = [scorer.score_row(r, cases[r["qid"]]) for r in result["rows"]]
    scored = [r for r in rows if not r["unscored"]]
    answerable = [r for r in scored if r["nli_claims"] or r["nli_claim_coverage"] is not None]
    claims = [c for r in answerable for c in r["nli_claims"]]
    both = sum(c["nli_met"] and c["lexical_met"] for c in claims)
    lex_only = sum(c["lexical_met"] and not c["nli_met"] for c in claims)
    nli_only = sum(c["nli_met"] and not c["lexical_met"] for c in claims)
    neither = sum(not c["nli_met"] and not c["lexical_met"] for c in claims)
    cited = sum(r["cited_sentences"] for r in answerable)
    supported = sum(r["supported_cited_sentences"] for r in answerable)
    summary = {
        "nli_accepted": sum(bool(r["nli_accepted"]) for r in scored),
        "scored_answers": len(scored),
        "lexical_accepted": result["summary"]["accepted"],
        "nli_required_claims_met": sum(c["nli_met"] for c in claims),
        "required_claims": len(claims),
        "lexical_required_claims_met": sum(c["lexical_met"] for c in claims),
        "citation_support_rate": supported / cited if cited else None,
        "cited_sentences": cited,
        "supported_cited_sentences": supported,
        "agreement": {"lexical_only": lex_only, "nli_only": nli_only, "both": both, "neither": neither},
    }
    out = {"source_result": str(result_path.name), "cases": result["cases"],
           "device": scorer.device, "rule": {"entail_threshold": 0.5, "window_tokens": 400},
           "summary": summary, "rows": rows}
    out_path = result_path.with_suffix(".nli.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    ap.add_argument("--cases", type=Path, default=None)
    args = ap.parse_args()
    print(json.dumps(rescore(args.result, args.cases), indent=2))


if __name__ == "__main__":
    main()
