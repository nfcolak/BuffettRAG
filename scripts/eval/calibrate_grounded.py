"""Dev calibration of the grounded settings (plan section 6), template composer, no re-scoring.

Stages (one invocation each, all deterministic):
  scores   retrieve every dev case once (BM25, the runner's path through backend.ask with a capturing
           provider), score every candidate window once with the real cross-encoder and cache the
           probabilities in GroundedSettings.CACHE_DIR (git-ignored); print the score distribution.
  grid     replay selection / decision / template compose / verify for T_RELEVANT x T_SLOT x MAX_UNITS from
           the cache only (a missing score is an error, never re-scored); also run the extractive
           baseline (provider local, BM25); write grid.csv; apply the fixed objective; write chosen.json
           (or report BLOCKED-NO-FEASIBLE with the closest settings and write only grid.csv).

Dev sets: heldout_v1, heldout_v2 (both spent), answer_benchmark_v3 and the 8 dev-only adversarial cases.
heldout_v3 is never read.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import CHUNKS_V3_FILE  # noqa: E402
from src.evaluation.answer_benchmark import (  # noqa: E402
    evaluate_live_answer, is_unanswerable_case, validate_fixture_ids,
)
from src.evaluation.supported_claims import supported_claims_met  # noqa: E402
from src.generation.grounded.engine import GroundedEngine  # noqa: E402
from src.generation.grounded.settings import GroundedSettings  # noqa: E402
from src.generation.grounded.template import TemplateComposer  # noqa: E402
from src.generation.grounded.verify import DeterministicVerifier  # noqa: E402
from src.generation.prompt import REFUSAL_LINE  # noqa: E402
from src.retrieval.context import build_doc_lookup  # noqa: E402
from src.storage import SearchHit, load_chunks_as_docs  # noqa: E402

OUT_DIR = ROOT / "data/evaluation/grounded_dev_v1"
DEV_SETS = {
    "heldout_v1": ROOT / "data/evaluation/heldout_v1/answer_benchmark_heldout_v1.json",
    "heldout_v2": ROOT / "data/evaluation/heldout_v2/answer_benchmark_heldout_v2.json",
    "answer_benchmark_v3": ROOT / "data/evaluation/answer_quality_program/answer_benchmark_v3.json",
    "adversarial_dev": OUT_DIR / "adversarial_dev.json",
}
CACHE_NAME = "calibrate_scores_v1.json"
_MARKER_RE = re.compile(r"\[([^\[\]\n]*)\]")
_MARKER_BODY_RE = re.compile(r"\d+(?:\s*,\s*\d+)*")
SETTING_KEYS = ("T_RELEVANT", "T_SLOT", "MAX_UNITS")
ADVERSARIAL_SET = "adversarial_dev"


# --------------------------------------------------------------------------- helpers

def _load_runner():
    spec = importlib.util.spec_from_file_location("run_live_benchmark_for_calib", ROOT / "scripts/eval/run_live_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cases() -> List[Dict[str, Any]]:
    cases = []
    for name, path in DEV_SETS.items():
        for case in json.loads(path.read_text(encoding="utf-8"))["cases"]:
            case = dict(case)
            case["_set"] = name
            case.setdefault("question_type", "answerable")
            cases.append(case)
    qids = [c["qid"] for c in cases]
    if len(set(qids)) != len(qids):
        raise ValueError("duplicate qid across dev sets")
    return cases


class _Capture:
    """Stands in for the grounded provider so backend.ask builds the exact client context, then stops."""
    provider_name = "grounded"
    model = "capture"

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def answer_grounded(self, original_query, context_hits, *, history=(), max_new_tokens=200):
        self.calls.append({"query": original_query, "hits": list(context_hits),
                           "history": list(history), "max_new_tokens": max_new_tokens})
        return SimpleNamespace(answer="", citations=[])


def _backend_state(runner, docs, llm):
    from src.services import ask_flow, backend_app as backend
    ask_flow._state = {"retriever": runner.BM25OnlyRetriever(docs), "docs": docs,
                       "docs_by_id": build_doc_lookup(docs), "llm": llm}
    ask_flow.EXPOSE_DEBUG_STATUS = False
    return backend


def _request(backend, case):
    return backend.AskRequest(query=case["query"], history=case.get("history", []),
                              strategy="hybrid", rerank=False)


def retrieve_contexts(runner, docs, cases):
    """The grounded context for each case (retrieved once, BM25, same path as the runner)."""
    capture = _Capture()
    backend = _backend_state(runner, docs, capture)
    out = {}
    for case in cases:
        before = len(capture.calls)
        backend.ask(_request(backend, case))
        out[case["qid"]] = capture.calls[-1] if len(capture.calls) > before else \
            {"query": case["query"], "hits": [], "history": case.get("history", []), "max_new_tokens": 200}
    return out


def extractive_baseline(runner, docs, cases):
    """Provider local (extractive), BM25, through the same backend.ask path."""
    from src.generation.providers import create_llm_provider
    backend = _backend_state(runner, docs, create_llm_provider(provider="local"))
    out = {}
    for case in cases:
        response = backend.ask(_request(backend, case))
        hits = [SearchHit(h.id, h.text, {"year": h.year, "source_file": h.source_file}, h.score) for h in response.hits]
        out[case["qid"]] = {"answer": response.answer, "citations": response.citations, "hits": hits}
    return out


# --------------------------------------------------------------------------- score cache

def _key(query: str, text: str) -> str:
    return hashlib.sha256((query + "\x00" + text).encode("utf-8")).hexdigest()


class CachedScorer:
    """Serves cached probabilities; with a real scorer it scores only the missing pairs, without one a miss is fatal."""

    def __init__(self, cache: Dict[str, float], real: Any = None) -> None:
        self.cache, self.real, self.scored_new = cache, real, 0

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        keys = [_key(query, t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        if missing:
            if self.real is None:
                raise KeyError(f"{len(missing)} window scores missing from the cache (no re-scoring in replay)")
            probs = self.real.score(query, [texts[i] for i in missing])
            for i, p in zip(missing, probs):
                self.cache[keys[i]] = float(p)
            self.scored_new += len(missing)
        return [self.cache[k] for k in keys]


def cache_path(settings: GroundedSettings) -> Path:
    return Path(settings.CACHE_DIR) / CACHE_NAME


def load_cache(settings: GroundedSettings, corpus_sha: str) -> Dict[str, float]:
    path = cache_path(settings)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("corpus_sha256") == corpus_sha:
            return data["scores"]
    return {}


def save_cache(settings: GroundedSettings, corpus_sha: str, scores: Dict[str, float]) -> None:
    path = cache_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"corpus_sha256": corpus_sha, "scorer": "CrossEncoderReranker.score_pairs sigmoid",
                                "scores": scores}), encoding="utf-8")


def make_engine(settings: GroundedSettings, scorer: Any) -> GroundedEngine:
    resources = SimpleNamespace(scorer=scorer, get_composer=lambda kind: TemplateComposer(), locks={})
    return GroundedEngine(resources, settings, composer_kind="template")


# --------------------------------------------------------------------------- metrics

def invalid_markers(answer: Optional[str], hit_count: int) -> int:
    """Bracket markers that are malformed or point outside the client hit list."""
    bad = 0
    for inner in _MARKER_RE.findall(answer or ""):
        if not _MARKER_BODY_RE.fullmatch(inner.strip()):
            bad += 1
            continue
        bad += sum(not 1 <= int(n) <= hit_count for n in inner.split(","))
    return bad


def is_abstention(answer: Optional[str]) -> bool:
    """Refusal, all-notes (no cited sentence), empty or marker-only answer."""
    text = (answer or "").strip()
    if not text or text == REFUSAL_LINE or REFUSAL_LINE.casefold() in text.casefold():
        return True
    if not re.search(r"[A-Za-z0-9]", _MARKER_RE.sub("", text)):
        return True
    return not any(_MARKER_BODY_RE.fullmatch(inner.strip()) for inner in _MARKER_RE.findall(text))


class Scorer:
    """Per-case metrics with memoised R3 support checks (identical answers recur across settings)."""

    def __init__(self) -> None:
        self._memo: Dict[Any, Any] = {}

    def row(self, case, answer, citations, hits) -> Dict[str, Any]:
        unanswerable = is_unanswerable_case(case)
        row = {"qid": case["qid"], "set": case["_set"], "unanswerable": unanswerable,
               "invalid_markers": invalid_markers(answer, len(hits)), "abstained": is_abstention(answer),
               "required_claims": 0, "supported_met": 0, "correct_refusal": None}
        if unanswerable:
            row["correct_refusal"] = evaluate_live_answer(answer or "", hits, case)["refusal_correct"]
            return row
        memo_key = (case["qid"], answer)
        if memo_key not in self._memo:
            self._memo[memo_key] = supported_claims_met(case, answer or "", citations or [], hits)
        claims = self._memo[memo_key]
        row["required_claims"] = len(claims)
        row["supported_met"] = sum(c["met"] for c in claims)
        return row


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    answerable = [r for r in rows if not r["unanswerable"]]
    unans = [r for r in rows if r["unanswerable"]]
    pre = [r for r in unans if r["set"] != ADVERSARIAL_SET]
    adv = [r for r in unans if r["set"] == ADVERSARIAL_SET]
    out = {
        "preexisting_unanswerable": len(pre),
        "preexisting_refusals": sum(r["correct_refusal"] is True for r in pre),
        "adversarial_unanswerable": len(adv),
        "adversarial_refusals": sum(r["correct_refusal"] is True for r in adv),
        "supported_met": sum(r["supported_met"] for r in rows),
        "required_claims": sum(r["required_claims"] for r in rows),
        "invalid_markers": sum(r["invalid_markers"] for r in rows),
        "answerable": len(answerable),
        "abstain_answerable": sum(r["abstained"] for r in answerable),
        "unanswerable": len(unans),
        "correct_refusals": sum(r["correct_refusal"] is True for r in unans),
    }
    for name in DEV_SETS:
        part = [r for r in rows if r["set"] == name]
        out[f"supported_{name}"] = sum(r["supported_met"] for r in part)
        out[f"abstain_{name}"] = sum(r["abstained"] for r in part if not r["unanswerable"])
        out[f"refusals_{name}"] = sum(r["correct_refusal"] is True for r in part if r["unanswerable"])
    return out


def run_setting(settings, cases, contexts, replay_cache, scorer_cache) -> List[Dict[str, Any]]:
    engine = make_engine(settings, CachedScorer(replay_cache))
    rows = []
    for case in cases:
        ctx = contexts[case["qid"]]
        result = engine.answer(ctx["query"], ctx["hits"], history=ctx["history"], max_new_tokens=ctx["max_new_tokens"])
        rows.append(scorer_cache.row(case, result.answer, result.citations, ctx["hits"]))
    return rows


# --------------------------------------------------------------------------- stages

def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _prepare():
    runner = _load_runner()
    docs = load_chunks_as_docs(CHUNKS_V3_FILE)
    cases = load_cases()
    validate_fixture_ids(cases, docs)
    corpus_sha = hashlib.sha256(CHUNKS_V3_FILE.read_bytes()).hexdigest()
    return runner, docs, cases, corpus_sha


def stage_scores(args) -> int:
    runner, docs, cases, corpus_sha = _prepare()
    settings = GroundedSettings(COMPOSER="template")
    cache = load_cache(settings, corpus_sha)
    contexts = retrieve_contexts(runner, docs, cases)
    from src.generation.grounded.resources import get_resources
    real = get_resources(settings).scorer
    scorer = CachedScorer(cache, real)
    # Loosest thresholds: every candidate window is scored (windows do not depend on the grid knobs).
    loose = dataclasses.replace(settings, T_RELEVANT=0.0, T_SLOT=0.0, MAX_UNITS=8)
    engine = make_engine(loose, scorer)
    start = time.perf_counter()
    stats = {"max_score": {}, "unit_scores": {}}
    for case in cases:
        ctx = contexts[case["qid"]]
        result = engine.answer(ctx["query"], ctx["hits"], history=ctx["history"], max_new_tokens=ctx["max_new_tokens"])
        cands = (result.plan.trace or {}).get("candidates") or []
        scores = sorted((c["score"] for c in cands), reverse=True)
        stats["max_score"][case["qid"]] = scores[0] if scores else None
        stats["unit_scores"][case["qid"]] = scores
    save_cache(settings, corpus_sha, cache)
    print(f"cases={len(cases)} windows_cached={len(cache)} newly_scored={scorer.scored_new} "
          f"score_seconds={time.perf_counter() - start:.1f} cache={cache_path(settings)}")
    by_kind: Dict[str, List[float]] = {"answerable": [], "unanswerable": []}
    for case in cases:
        v = stats["max_score"][case["qid"]]
        by_kind["unanswerable" if is_unanswerable_case(case) else "answerable"].append(-1.0 if v is None else v)
    for kind, values in by_kind.items():
        values.sort()
        q = lambda p: values[min(len(values) - 1, int(p * len(values)))]  # noqa: E731
        print(f"max window score, {kind} (n={len(values)}): min={values[0]:.3f} p10={q(.1):.3f} p25={q(.25):.3f} "
              f"median={q(.5):.3f} p75={q(.75):.3f} p90={q(.9):.3f} max={values[-1]:.3f}")
    allscores = sorted(s for v in stats["unit_scores"].values() for s in v)
    if allscores:
        print("all window scores: " + " ".join(f"p{int(p*100)}={allscores[min(len(allscores)-1, int(p*len(allscores)))]:.3f}"
                                                for p in (.1, .25, .5, .75, .9, .95, .99)))
    (Path(args.run_dir) / "score_distribution.json").write_text(json.dumps(stats), encoding="utf-8")
    return 0


def _floats(text: str) -> List[float]:
    return [float(x) for x in text.split(",")]


def feasibility(agg: Dict[str, Any], ext_abstain: int) -> Dict[str, Any]:
    # plan section 6: correct refusals >= 5/6 of the PRE-EXISTING unanswerable dev cases; the dev-only
    # adversarial refusal cases are reported separately and are never part of the gate
    need = math.ceil(5 * agg["preexisting_unanswerable"] / 6)
    return {"zero_invalid": agg["invalid_markers"] == 0,
            "abstain_ok": agg["abstain_answerable"] <= ext_abstain,
            "refusals_ok": agg["preexisting_refusals"] >= need, "refusals_needed": need}


def stage_grid(args) -> int:
    runner, docs, cases, corpus_sha = _prepare()
    base = GroundedSettings(COMPOSER="template")
    cache = load_cache(base, corpus_sha)
    if not cache:
        raise SystemExit("score cache missing: run the scores stage first")
    contexts = retrieve_contexts(runner, docs, cases)
    scorer = Scorer()

    ext = extractive_baseline(runner, docs, cases)
    ext_rows = [scorer.row(c, ext[c["qid"]]["answer"], ext[c["qid"]]["citations"], ext[c["qid"]]["hits"]) for c in cases]
    ext_agg = aggregate(ext_rows)
    print("extractive baseline:", json.dumps(ext_agg))

    t_rel, t_slot, units = _floats(args.t_relevant), _floats(args.t_slot), [int(x) for x in args.max_units.split(",")]
    results = []
    start = time.perf_counter()
    for tr, ts, mu in itertools.product(t_rel, t_slot, units):
        settings = dataclasses.replace(base, T_RELEVANT=tr, T_SLOT=ts, MAX_UNITS=mu, NUMERIC_GUARD="bound")
        rows = run_setting(settings, cases, contexts, cache, scorer)
        agg = aggregate(rows)
        agg.update(T_RELEVANT=tr, T_SLOT=ts, MAX_UNITS=mu, NUMERIC_GUARD="bound")
        agg.update(feasibility(agg, ext_agg["abstain_answerable"]))
        agg["feasible"] = agg["zero_invalid"] and agg["abstain_ok"] and agg["refusals_ok"]
        results.append((agg, rows))
        print(f"T_REL={tr} T_SLOT={ts} UNITS={mu} supported={agg['supported_met']}/{agg['required_claims']} "
              f"invalid={agg['invalid_markers']} abstain={agg['abstain_answerable']}/{agg['answerable']} "
              f"refusals={agg['preexisting_refusals']}/{agg['preexisting_unanswerable']}"
              f" adv_refusals={agg['adversarial_refusals']}/{agg['adversarial_unanswerable']} feasible={agg['feasible']} "
              f"t={time.perf_counter() - start:.0f}s", flush=True)

    feasible = [(a, r) for a, r in results if a["feasible"]]
    key = lambda a: (-a["supported_met"], a["MAX_UNITS"], -a["T_RELEVANT"], -a["T_SLOT"])  # noqa: E731
    chosen = min(feasible, key=lambda ar: key(ar[0])) if feasible else None

    rows_out = [a for a, _ in results]
    if chosen is not None:
        for guard in ("verbatim",):
            settings = dataclasses.replace(base, T_RELEVANT=chosen[0]["T_RELEVANT"], T_SLOT=chosen[0]["T_SLOT"],
                                           MAX_UNITS=chosen[0]["MAX_UNITS"], NUMERIC_GUARD=guard)
            agg = aggregate(run_setting(settings, cases, contexts, cache, scorer))
            agg.update(T_RELEVANT=settings.T_RELEVANT, T_SLOT=settings.T_SLOT, MAX_UNITS=settings.MAX_UNITS,
                       NUMERIC_GUARD=guard)
            agg.update(feasibility(agg, ext_agg["abstain_answerable"]))
            agg["feasible"] = agg["zero_invalid"] and agg["abstain_ok"] and agg["refusals_ok"]
            rows_out.append(agg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["T_RELEVANT", "T_SLOT", "MAX_UNITS", "NUMERIC_GUARD", "supported_met", "required_claims",
               "invalid_markers", "answerable", "abstain_answerable", "unanswerable", "correct_refusals",
               "preexisting_unanswerable", "preexisting_refusals", "adversarial_unanswerable", "adversarial_refusals",
               "refusals_needed", "zero_invalid", "abstain_ok", "refusals_ok", "feasible"]
    columns += [c for name in DEV_SETS for c in (f"supported_{name}", f"abstain_{name}", f"refusals_{name}")]
    with (out_dir / "grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)

    dev_summary = {"sets": {n: sum(c["_set"] == n for c in cases) for n in DEV_SETS},
                   "answerable": ext_agg["answerable"], "unanswerable": ext_agg["unanswerable"],
                   "unanswerable_qids": sorted(c["qid"] for c in cases if is_unanswerable_case(c))}
    grid_axes = {"T_RELEVANT": t_rel, "T_SLOT": t_slot, "MAX_UNITS": units}
    if chosen is None:
        def violation(a):
            return (a["invalid_markers"] > 0, max(0, a["abstain_answerable"] - ext_agg["abstain_answerable"])
                    + max(0, a["refusals_needed"] - a["preexisting_refusals"]), -a["supported_met"])
        closest = sorted((a for a, _ in results), key=violation)[:5]
        print("BLOCKED-NO-FEASIBLE; closest settings:")
        for a in closest:
            print({k: a[k] for k in SETTING_KEYS + ("supported_met", "invalid_markers", "abstain_answerable",
                                                      "preexisting_refusals", "preexisting_unanswerable",
                                                      "adversarial_refusals", "adversarial_unanswerable")})
        (Path(args.run_dir) / "blocked.json").write_text(json.dumps(
            {"extractive": ext_agg, "closest": closest, "dev": dev_summary, "grid": grid_axes}), encoding="utf-8")
        return 2

    best, best_rows = chosen
    verbatim = rows_out[-1]
    per_case = [{k: r[k] for k in ("qid", "set", "supported_met", "required_claims", "abstained",
                                   "invalid_markers", "correct_refusal")} for r in best_rows]
    document = {
        "values": {k: best[k] for k in SETTING_KEYS},
        "composer": "template", "numeric_guard": "bound",
        "objective": {k: best[k] for k in ("supported_met", "required_claims", "invalid_markers", "answerable",
                                           "abstain_answerable", "unanswerable", "correct_refusals", "preexisting_unanswerable",
                                           "preexisting_refusals", "adversarial_unanswerable", "adversarial_refusals",
                                           "refusals_needed")},
        "constraints": {"zero_invalid_markers": best["zero_invalid"],
                        "abstentions_le_extractive": best["abstain_ok"],
                        "correct_refusals_ge_5_6": best["refusals_ok"],
                        "refusal_denominator": best["preexisting_unanswerable"],
                        "refusal_rule": "correct refusals >= ceil(5/6 * pre-existing unanswerable dev cases); "
                                        "adversarial refusal cases are reported, never gated",
                        "refusals_needed": best["refusals_needed"]},
        "tie_break": "most supported claims; then fewer MAX_UNITS; then higher T_RELEVANT; then higher T_SLOT",
        "extractive_baseline": ext_agg,
        "guard_comparison": {"bound": {k: best[k] for k in ("supported_met", "invalid_markers", "abstain_answerable",
                                                            "preexisting_refusals", "adversarial_refusals")},
                             "verbatim": {k: verbatim[k] for k in ("supported_met", "invalid_markers",
                                                                   "abstain_answerable", "preexisting_refusals",
                                                                   "adversarial_refusals")}},
        "feasible_settings": len(feasible), "grid_size": len(results), "grid": grid_axes,
        "dev": dev_summary,
        "per_case_at_chosen": per_case,
        "code_sha": args.code_sha or _git_sha(),
        "corpus_sha256": corpus_sha,
        "cases_sha256": {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in DEV_SETS.items()},
    }
    if args.extra_json:
        document.update(json.loads(Path(args.extra_json).read_text(encoding="utf-8")))
    (out_dir / "chosen.json").write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print("CHOSEN", json.dumps(document["values"]), json.dumps(document["objective"]))
    print("guard comparison", json.dumps(document["guard_comparison"]))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=["scores", "grid"])
    parser.add_argument("--run-dir", default=str(ROOT.parent / "_runs" / "u6b-fix"))
    parser.add_argument("--t-relevant", default="0.001,0.01,0.03,0.1,0.35")
    parser.add_argument("--t-slot", default="0.001,0.01,0.05,0.2")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--extra-json", default=None, help="merged into chosen.json (loss analysis)")
    parser.add_argument("--strip-engine-labels", action="store_true",
                        help="sensitivity only: the metric also strips the engine label 'In the YYYY letter:'")
    parser.add_argument("--max-units", default="4,6,8")
    parser.add_argument("--code-sha", default=None)
    args = parser.parse_args()
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    if args.strip_engine_labels:
        import src.evaluation.supported_claims as metric
        metric._LABEL_RE = re.compile(r"^\s*(?:In\s+(?:the\s+)?)?((?:19|20)\d{2})(?:\s+letters?)?\s*[:,]\s+", re.I)
    return stage_scores(args) if args.stage == "scores" else stage_grid(args)


if __name__ == "__main__":
    raise SystemExit(main())
