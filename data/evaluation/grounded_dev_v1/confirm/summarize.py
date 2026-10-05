"""Score the two confirm runs with the strict supported-claims metric (calibrate_grounded's Scorer, so the
numbers are comparable with chosen.json) and write summary.json. Run from the repo root with the project python.

    python data/evaluation/grounded_dev_v1/confirm/summarize.py [--load "<note on machine load>"]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.eval.calibrate_grounded import Scorer, aggregate, load_cases  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = {"grounded_mlx7b": HERE / "grounded_mlx7b.json", "grounded_llama_ftr2": HERE / "grounded_llama_ftr2.json"}
COLD_MS = 50.0  # a case that loaded anything (scorer / composer / NLI) for more than this is cold


def pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo), 1)


def summarize(path: Path, cases: dict) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text(encoding="utf-8"))
    scorer = Scorer()
    rows = []
    fallback_reasons, drops, fb_drops = collections.Counter(), collections.Counter(), collections.Counter()
    groups_total = groups_fallback = 0
    failures = collections.Counter()
    lat_all, lat_warm, lat_warm_compose, cold_cases = [], [], [], 0
    decisions = collections.Counter()
    for row in result["rows"]:
        case = cases[row["qid"]]
        snapshot = row.get("context_snapshot") or []
        rows.append(scorer.row(case, row.get("answer"), row.get("citations"), snapshot))
        for f in row.get("provider_failures") or []:
            failures[f"{f.get('stage')}:{f.get('error_type')}"] += 1
        call_cold, call_compose = 0.0, 0.0
        for call in row.get("grounded_calls") or []:
            decisions[call.get("decision")] += 1
            call_cold += sum((call.get("cold_load_ms") or {}).values())
            call_compose += (call.get("timings_ms") or {}).get("compose", 0.0) or 0.0
            for reason in (call.get("fallback_reasons") or {}).values():
                fallback_reasons[reason] += 1
            for group in ((call.get("trace") or {}).get("engine", {}).get("groups", {}) or {}).values():
                if (call.get("decision") == "answer"):
                    groups_total += 1
                groups_fallback += bool(group.get("fallback_reason"))
                for key, n in (group.get("dropped") or {}).items():
                    drops[key] += n
                for key, n in (group.get("fallback_dropped") or {}).items():
                    fb_drops[key] += n
        latency = row.get("latency_ms")
        if latency is None:
            continue
        lat_all.append(latency)
        if call_cold > COLD_MS:
            cold_cases += 1
        else:
            lat_warm.append(latency)
            if call_compose > 0:
                lat_warm_compose.append(latency)
    agg = aggregate(rows)
    unans_pre = [r for r in rows if r["unanswerable"] and r["set"] != "adversarial_dev"]
    return {
        "metrics": agg,
        "cases_run": len(rows),
        "answer_decisions": dict(decisions),
        "fallback": {"groups_composed": groups_total, "groups_fallback_to_template": groups_fallback,
                     "fallback_rate": round(groups_fallback / groups_total, 4) if groups_total else None,
                     "reasons": dict(fallback_reasons)},
        "verifier_drop_reasons_model_text": dict(drops),
        "verifier_drop_reasons_template_fallback": dict(fb_drops),
        "provider_failures": {"total": sum(failures.values()), "by_stage_type": dict(failures),
                              "summary_count": result["summary"]["provider_failures"]},
        "latency_ms": {"all_cases": {"n": len(lat_all), "p50": pct(lat_all, 0.5), "p95": pct(lat_all, 0.95)},
                       "warm_cases": {"n": len(lat_warm), "p50": pct(lat_warm, 0.5), "p95": pct(lat_warm, 0.95)},
                       "warm_cases_with_model_call": {"n": len(lat_warm_compose), "p50": pct(lat_warm_compose, 0.5),
                                                      "p95": pct(lat_warm_compose, 0.95)},
                       "cold_cases_excluded": cold_cases, "cold_rule": f"sum of cold_load_ms > {COLD_MS} ms"},
        "temperature": manifest["temperature"],
        "code_sha": manifest.get("code_sha"), "code_dirty": manifest.get("code_dirty"),
        "model": result["answer_engine"]["model"],
        "unexpected_refusals_summary": result["summary"].get("unexpected_refusals"),
        "preexisting_refusals_of": f"{agg['preexisting_refusals']}/{agg['preexisting_unanswerable']}",
        "_unans_pre": len(unans_pre),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", default="unknown")
    args = parser.parse_args()
    cases = {c["qid"]: c for c in load_cases()}
    chosen = json.loads((ROOT / "data/evaluation/grounded_dev_v1/chosen.json").read_text(encoding="utf-8"))
    keys = ("supported_met", "required_claims", "invalid_markers", "answerable", "abstain_answerable", "unanswerable",
            "preexisting_unanswerable", "preexisting_refusals", "adversarial_unanswerable", "adversarial_refusals")
    out = {
        "schema_version": 1, "retrieval": "bm25", "temperature": 0.0, "cases": 80,
        "metric": "strict supported claims (src.evaluation.supported_claims via calibrate_grounded.Scorer); same scorer as chosen.json",
        "settings": chosen["values"] | {"NUMERIC_GUARD": chosen["numeric_guard"]},
        "machine_load_note": args.load,
        "baselines_from_chosen_json": {
            "grounded_template": {k: chosen["objective"][k] for k in keys if k in chosen["objective"]},
            "extractive": {k: chosen["extractive_baseline"][k] for k in keys if k in chosen["extractive_baseline"]},
        },
    }
    for name, path in RUNS.items():
        if path.exists():
            part = summarize(path, cases)
            part.pop("_unans_pre")
            out[name] = part
    (HERE / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for name in RUNS:
        if name in out:
            m = out[name]["metrics"]
            print(name, {k: m[k] for k in keys}, out[name]["fallback"], out[name]["latency_ms"]["warm_cases"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
