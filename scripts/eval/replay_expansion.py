"""Re-run ONLY the post-processing of a stored run (validator + neighbour expansion) at other settings.

The model is never called: the stored `raw_answers` are fed, in order, to the real serving code
(ask_flow._generate_answer: evidence gate, citation check, claim validator, expansion, refusal retry,
per-period comparison path) through a scripted LLM, with the stored `context_snapshot` as the served
passages. The output is a run JSON in the same schema (rows re-scored when the cases file is found),
ready for the AI-judge export. `--verify` replays at the settings the run was made with and reports
how many stored final answers are reproduced exactly.

    replay_expansion.py RUN.json --verify
    replay_expansion.py RUN.json --after 3 --before 3 --max-sentences 12 --expanded-max 40 --output OUT.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTHON_DOTENV_DISABLED", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.answer_benchmark import evaluate_live_answer, is_unanswerable_case
from src.generation.prompt import REFUSAL_LINE
from src.storage import SearchHit

# Settings of the pv7 runs that carry no `settings` block (run names: *_p16_e3_*: 16 passages, expansion 3 after).
LEGACY_SETTINGS = {"ANSWER_NEIGHBOR_AFTER": 3, "ANSWER_NEIGHBOR_BEFORE": 0, "ANSWER_MAX_SENTENCES": 12,
                   "ANSWER_EXPANDED_MAX_SENTENCES": 12, "COMPARE_ROUTING_V2": 0, "ANSWER_REFUSAL_RETRY": "1"}
_ENV_KEYS = tuple(LEGACY_SETTINGS)


class _ScriptedLLM:
    """Serves the stored raw answers in call order; running dry means the replay diverged from the stored run."""

    provider_name = "replay"
    model = "replay"

    def __init__(self, raw_answers: List[str]):
        self.raw = list(raw_answers)
        self.calls = 0

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        self.calls += 1
        if not self.raw:
            raise RuntimeError("replay desync: more model calls than stored raw answers")
        return self.raw.pop(0)


def stored_settings(run: Dict[str, Any]) -> Dict[str, Any]:
    """Post-processing settings the run was made with (its `settings` block, else the pv7 defaults)."""
    block = run.get("settings") or {}
    return {key: block.get(key, LEGACY_SETTINGS[key]) for key in _ENV_KEYS}


def _hits(row: Dict[str, Any]) -> List[SearchHit]:
    """Served passages from the snapshot. It carries no source file: take it from the stored citations, else
    the corpus naming convention buffet_<year>.txt (only citation `sources` use it)."""
    known = {pid: src for c in row.get("citations") or [] for pid, src in zip(c.get("passage_ids", []), c.get("sources", []))}
    return [SearchHit(p["id"], p["text"], {"year": p.get("year"), "merged_ids": p.get("merged_ids"),
                                           "source_file": known.get(p["id"]) or (f"buffet_{p['year']}.txt" if p.get("year") else None)}, 0.0)
            for p in row.get("context_snapshot") or []]


def _apply_env(settings: Dict[str, Any]) -> None:
    for key in _ENV_KEYS:
        os.environ[key] = str(settings[key])


def replay_row(row: Dict[str, Any], settings: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """(new row, status): status is 'ok', 'copied' (no model answer to post-process) or 'desync'."""
    from src.services import ask_flow

    new = copy.deepcopy(row)
    hits = _hits(row)
    if row.get("status") != "scored" or not hits or row.get("answer") is None:
        return new, "copied"
    _apply_env(settings)
    llm = _ScriptedLLM(row.get("raw_answers") or [])
    history = row.get("history") or []
    extras = [turn["content"] for turn in history if turn.get("role") == "user"]
    answer, citations = ask_flow._generate_answer(llm, "", hits, 900, row["query"], extras, history)
    status = "ok" if not llm.raw and not answer.startswith("[LLM unavailable") else "desync"
    new["answer"], new["citations"] = answer, citations
    return new, status


def _rescore(row: Dict[str, Any], case: Optional[Dict[str, Any]]) -> None:
    if case is None or row.get("status") != "scored":
        return
    hits = [SearchHit(p["id"], p["text"], {"year": p.get("year")}, 0.0) for p in row["context_snapshot"]]
    score = evaluate_live_answer(row["answer"] or "", hits, case)
    row.update({"score": score, "required_claim_hits": score["claims"], "refusal_correct": score["refusal_correct"]})


def _summary(run: Dict[str, Any], rows: List[Dict[str, Any]], cases: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in rows if r.get("status") == "scored"]
    unanswerable = {q for q, c in cases.items() if is_unanswerable_case(c)}
    summary = dict(run.get("summary") or {})
    if not cases or any(r.get("score") is None for r in scored):
        return summary
    accepted = sum(r["score"]["accepted"] for r in scored)
    summary.update({
        "scored_answers": len(scored), "accepted": accepted, "rejected": len(scored) - accepted,
        "acceptance_rate": accepted / len(scored) if scored else None,
        "correct_refusals": sum(r["refusal_correct"] is True for r in scored),
        "unexpected_refusals": sum(r["qid"] not in unanswerable and r["answer"] == REFUSAL_LINE for r in scored),
        "required_claims": sum(len(r["required_claim_hits"]) for r in scored),
        "required_claims_met": sum(c["met"] for r in scored for c in r["required_claim_hits"]),
    })
    return summary


def _load_cases(run: Dict[str, Any], cases_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    path = cases_path or (ROOT / run["cases"] if run.get("cases") else None)
    if path is None or not Path(path).is_file():
        # the cases path in a run JSON is relative to the repo the run was made in
        alt = Path(os.environ.get("BUFFETT_MAIN_CHECKOUT", "/Users/necatifurkancolak/AI-Workplace/Projects/current/BuffettRAG")) / (run.get("cases") or "")
        path = alt if run.get("cases") and alt.is_file() else None
    if path is None:
        return {}
    return {c["qid"]: c for c in json.loads(Path(path).read_text(encoding="utf-8"))["cases"]}


def replay_run(run: Dict[str, Any], settings: Dict[str, Any], *, cases: Optional[Dict[str, Dict[str, Any]]] = None
               ) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    """(new run JSON, [(qid, status)]); `settings` are the post-processing settings to replay at."""
    from src.retrieval import retriever

    cases = cases or {}
    rows, statuses = [], []
    # A run over an older corpus (the Partnership letters, 1957-) routes comparisons by that corpus's years.
    years = [p["year"] for r in run["rows"] for p in r.get("context_snapshot") or [] if isinstance(p.get("year"), int)]
    saved_min = retriever.CORPUS_YEAR_MIN
    retriever.CORPUS_YEAR_MIN = min(saved_min, min(years, default=saved_min))
    try:
        for row in run["rows"]:
            new, status = replay_row(row, settings)
            if status == "ok":
                _rescore(new, cases.get(row["qid"]))
            rows.append(new)
            statuses.append((row["qid"], status))
    finally:
        retriever.CORPUS_YEAR_MIN = saved_min
    out = copy.deepcopy(run)
    out["rows"] = rows
    out["settings"] = {**(run.get("settings") or {}), **settings}
    out["summary"] = _summary(run, rows, cases)
    out["replay"] = {"statuses": {s: sum(1 for _, x in statuses if x == s) for s in ("ok", "copied", "desync")},
                     "settings": settings, "model_called": False}
    return out, statuses


def verify(run: Dict[str, Any]) -> Dict[str, Any]:
    """Replay at the stored settings and count the rows whose final answer and citations are reproduced."""
    settings = stored_settings(run)
    new, statuses = replay_run(run, settings)
    mismatches = [row["qid"] for row, old in zip(new["rows"], run["rows"])
                  if (row["answer"] or "").strip() != (old["answer"] or "").strip()
                  or row["citations"] != old["citations"]]
    return {"rows": len(run["rows"]), "matched": len(run["rows"]) - len(mismatches), "mismatched_qids": mismatches,
            "desync_qids": [q for q, s in statuses if s == "desync"], "settings": settings}


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="stored run JSON (rows need raw_answers and context_snapshot)")
    p.add_argument("--verify", action="store_true", help="replay at the stored settings and report N/N reproduced")
    p.add_argument("--after", type=int, help="ANSWER_NEIGHBOR_AFTER")
    p.add_argument("--before", type=int, help="ANSWER_NEIGHBOR_BEFORE")
    p.add_argument("--max-sentences", type=int, help="ANSWER_MAX_SENTENCES (model-answer cap)")
    p.add_argument("--expanded-max", type=int, help="ANSWER_EXPANDED_MAX_SENTENCES (expanded-answer cap)")
    p.add_argument("--cases", type=Path, default=None, help="cases file for re-scoring (default: the run's own)")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse()
    run = json.loads(args.run.read_text(encoding="utf-8"))
    if args.verify:
        report = verify(run)
        print(json.dumps({**report, "run": args.run.name, "result": f"{report['matched']}/{report['rows']}"}, indent=1))
        return 0 if report["matched"] == report["rows"] else 1
    if args.output is None:
        raise SystemExit("--output is required unless --verify")
    settings = stored_settings(run)
    for key, value in (("ANSWER_NEIGHBOR_AFTER", args.after), ("ANSWER_NEIGHBOR_BEFORE", args.before),
                       ("ANSWER_MAX_SENTENCES", args.max_sentences), ("ANSWER_EXPANDED_MAX_SENTENCES", args.expanded_max)):
        if value is not None:
            settings[key] = value
    if args.max_sentences is not None and args.expanded_max is None:
        settings["ANSWER_EXPANDED_MAX_SENTENCES"] = args.max_sentences  # one cap, as before the split
    new, statuses = replay_run(run, settings, cases=_load_cases(run, args.cases))
    new["replay"]["source"] = {"file": str(args.run), "sha256": hashlib.sha256(args.run.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **new["replay"]["statuses"]}))
    return 1 if new["replay"]["statuses"]["desync"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
