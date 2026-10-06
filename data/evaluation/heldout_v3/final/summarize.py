"""U7 summary of the 7 arms on heldout_v3: writes summary.json and comparison.md next to this file.

Run from anywhere with the project python (after rescore_nli produced <arm>.nli.json for every arm):

    env -u PYTHONPATH PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 <project>/.venv/bin/python \
        data/evaluation/heldout_v3/final/summarize.py [--load "<note on machine load>"]

Strict supported claims use src.evaluation.supported_claims through calibrate_grounded.Scorer (same scorer as the
dev confirmation). Grounded arms are scored against the saved context_snapshot. Arms 1-4 (non-grounded) have no
snapshot, so their client context (the neighbour-expanded / fitted hit list that the [k] markers refer to) is rebuilt
by running the same retrieval + prepare_answer_context path with a stub provider of the arm's provider name (no
model call; EXPANSION_MODE is off so retrieval does not depend on the model). Every rebuilt id list is checked against
the saved row.passage_ids, and the method is validated against the saved snapshots of the grounded arms (ids and text).
Engineering screen only, not a statistical test.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.eval.calibrate_grounded import Scorer  # noqa: E402
from src.evaluation.answer_benchmark import is_unanswerable_case  # noqa: E402
from src.evaluation.citation_faithfulness import split_sentences  # noqa: E402
from src.generation.prompt import REFUSAL_LINE  # noqa: E402
from src.storage import SearchHit, load_chunks_as_docs  # noqa: E402

HERE = Path(__file__).resolve().parent
CASES_PATH = ROOT / "data/evaluation/heldout_v3/answer_benchmark_heldout_v3.json"
ARMS = [  # (arm number, name, label)
    (1, "local_extractive", "extractive"),
    (2, "llama_base", "llama base 1.5B"),
    (3, "llama_ftr2", "llama FT-r2 1.5B"),
    (4, "mlx7b_oldpath", "mlx 7B old path"),
    (5, "grounded_mlx7b", "grounded + mlx 7B"),
    (6, "grounded_template", "grounded + template"),
    (7, "grounded_llama_ftr2", "grounded + llama FT-r2"),
]
COLD_MS = 50.0          # a grounded case that loaded anything for more than this is cold
WARM_P95_LIMIT_MS = 8000.0
_NOTE_RE = re.compile(r"^\s*The retrieved passages do not cover\b")
_MARK_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


def norm(text: str) -> str:
    return " ".join(_MARK_RE.sub("", text or "").split())


def pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo), 1)


def lat_block(values):
    return {"n": len(values), "p50": pct(values, 0.5), "p95": pct(values, 0.95)}


def is_refusal_answer(answer) -> bool:
    text = (answer or "").strip()
    return bool(text) and REFUSAL_LINE.casefold() in text.casefold()


class _StubLLM:
    """No model: only provider_name matters to the context builder."""

    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name
        self.model = "stub"

    def generate(self, *_a, **_k):
        raise RuntimeError("stub")


_RUNNER = None


def rebuild_contexts(provider_name, cases, docs):
    """qid -> context hits exactly as /ask builds them for this provider (retrieval + prepare_answer_context)."""
    global _RUNNER
    import importlib.util
    from src.retrieval.context import build_doc_lookup
    from src.services import ask_flow, backend_app as backend
    if _RUNNER is None:
        spec = importlib.util.spec_from_file_location("run_live_benchmark_u7", ROOT / "scripts/eval/run_live_benchmark.py")
        _RUNNER = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_RUNNER)
    old = (ask_flow._state, ask_flow.EXPOSE_DEBUG_STATUS)
    ask_flow._state = {"retriever": _RUNNER.BM25OnlyRetriever(docs), "docs": docs,
                       "docs_by_id": build_doc_lookup(docs), "llm": _StubLLM(provider_name)}
    ask_flow.EXPOSE_DEBUG_STATUS = False
    out = {}
    try:
        for qid, case in cases.items():
            req = backend.AskRequest(query=case["query"], history=case.get("history", []), strategy="hybrid", rerank=False)
            context = ask_flow._prepare_ask(req)[4]
            out[qid] = [SearchHit(h.id, h.text, dict(h.metadata), h.score) for h in context]
    finally:
        ask_flow._state, ask_flow.EXPOSE_DEBUG_STATUS = old
    return out


def strict_rows_for_arm(result, cases, rebuilt, scorer):
    """Per-row scorer rows plus the per-claim strict detail."""
    out = {}
    for row in result["rows"]:
        case = dict(cases[row["qid"]])
        case["_set"] = "heldout_v3"
        if row.get("status") != "scored":
            out[row["qid"]] = {"scorer": None, "claims": [], "hits_source": "none"}
            continue
        snapshot = row.get("context_snapshot") or []
        if snapshot:
            hits, source = snapshot, "context_snapshot"
        else:
            hits, source = rebuilt[row["qid"]], "rebuilt_context"
            if [h.id for h in hits] != (row.get("passage_ids") or []):
                source = "rebuilt_context_ID_MISMATCH"
        srow = scorer.row(case, row.get("answer"), row.get("citations"), hits)
        claims = []
        if not is_unanswerable_case(case):
            from src.evaluation.supported_claims import supported_claims_met
            claims = supported_claims_met(case, row.get("answer") or "", row.get("citations") or [], hits)
        out[row["qid"]] = {"scorer": srow, "claims": claims, "hits_source": source}
    return out


def validate_rebuild(result, rebuilt):
    """Grounded arms: the rebuilt context must equal the saved snapshot (ids and text)."""
    checked = same = 0
    for row in result["rows"]:
        snap = row.get("context_snapshot") or []
        if not snap and row["qid"] not in rebuilt:
            continue
        checked += 1
        got = rebuilt[row["qid"]]
        same += [(h.id, h.text) for h in got] == [(s["id"], s["text"]) for s in snap]
    return {"rows_checked": checked, "rows_identical": same}


def summarize_arm(num, name, label, cases, rebuilt):
    path = HERE / f"{name}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    nli = json.loads((HERE / f"{name}.nli.json").read_text(encoding="utf-8"))
    manifest_path = HERE / f"{name}.json.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    scorer = Scorer()
    strict = strict_rows_for_arm(result, cases, rebuilt, scorer)
    nli_rows = {r["qid"]: r for r in nli["rows"]}
    grounded = result["answer_engine"]["provider"] == "grounded"

    per_qid, per_type = {}, collections.defaultdict(lambda: {"cases": 0, "supported_met": 0, "required_claims": 0})
    abstain_answerable = unexpected_refusals = correct_refusals = unanswerable = answerable = invalid = 0
    abstain_detail = []
    extra_total = content_total = 0
    for row in result["rows"]:
        qid, case = row["qid"], cases[row["qid"]]
        s = strict[qid]["scorer"]
        qtype = case.get("question_type", "answerable")
        met = sum(c["met"] for c in strict[qid]["claims"])
        req = len(strict[qid]["claims"])
        per_qid[qid] = {"type": qtype, "supported_met": met, "required_claims": req}
        bucket = per_type[qtype]
        bucket["cases"] += 1
        bucket["supported_met"] += met
        bucket["required_claims"] += req
        if s is None:
            continue
        invalid += s["invalid_markers"]
        if s["unanswerable"]:
            unanswerable += 1
            correct_refusals += s["correct_refusal"] is True
            continue
        answerable += 1
        if s["abstained"]:
            abstain_answerable += 1
            abstain_detail.append({"qid": qid, "refusal": is_refusal_answer(row.get("answer"))})
        if is_refusal_answer(row.get("answer")):
            unexpected_refusals += 1
        # unsupported extra sentences: content sentences that are neither a strict supporting sentence of a
        # required claim nor NLI-supported (cited and entailed >= 0.5 by a cited passage); notes/refusals skipped.
        supporting = {norm(x) for c in strict[qid]["claims"] for x in c["supporting_sentences"]}
        for sent in (nli_rows[qid].get("sentences") or []):
            text = sent["text"]
            if _NOTE_RE.match(text) or REFUSAL_LINE.casefold() in text.casefold():
                continue
            content_total += 1
            if norm(text) in supporting or sent["supported"]:
                continue
            extra_total += 1

    decisions = collections.Counter()
    fallback_reasons = collections.Counter()
    groups_total = groups_fallback = 0
    failures = collections.Counter()
    lat_all, lat_warm, lat_cold = [], [], []
    first_call_done = False
    for row in result["rows"]:
        for f in row.get("provider_failures") or []:
            failures[f"{f.get('stage')}:{f.get('error_type')}"] += 1
        call_cold = 0.0
        for call in row.get("grounded_calls") or []:
            decisions[call.get("decision")] += 1
            call_cold += sum((call.get("cold_load_ms") or {}).values())
            for reason in (call.get("fallback_reasons") or {}).values():
                fallback_reasons[reason] += 1
            for group in ((call.get("trace") or {}).get("engine", {}).get("groups", {}) or {}).values():
                if call.get("decision") == "answer":
                    groups_total += 1
                    groups_fallback += bool(group.get("fallback_reason"))
        latency = row.get("latency_ms")
        if latency is None:
            continue
        lat_all.append(latency)
        # cold = first case that called a model/composer in this process, or any grounded case that loaded
        # something (> COLD_MS); everything else is warm.
        made_call = bool(row.get("provider_calls")) or call_cold > 0
        cold = call_cold > COLD_MS or (made_call and not first_call_done and not grounded)
        if made_call:
            first_call_done = True
        (lat_cold if cold else lat_warm).append(latency)

    nli_sum = nli["summary"]
    rescore_strict = nli_sum.get("strict_supported_claims_met")
    supported_met = sum(v["supported_met"] for v in per_qid.values())
    required = sum(v["required_claims"] for v in per_qid.values())
    eff_settings = (manifest or {}).get("effective_settings") or {}
    out = {
        "arm": num, "name": name, "label": label,
        "provider": result["answer_engine"]["provider"], "composer": result["answer_engine"].get("composer"),
        "cases_run": result["summary"]["cases_run"], "scored_answers": result["summary"]["scored_answers"],
        "lexical_accepted": result["summary"]["accepted"],
        "nli_accepted": nli_sum["nli_accepted"],
        "strict_supported_claims_met": supported_met, "strict_required_claims": required,
        "strict_hits_source": sorted({v["hits_source"] for v in strict.values()}),
        "context_rebuild_validation": validate_rebuild(result, rebuilt) if grounded else None,
        "strict_from_rescore_nli": rescore_strict,
        "strict_matches_rescore_nli": (rescore_strict == supported_met) if rescore_strict is not None else None,
        "lexical_claims_met": nli_sum["lexical_required_claims_met"], "nli_claims_met": nli_sum["nli_required_claims_met"],
        "nli_required_claims": nli_sum["required_claims"],
        "invalid_markers": invalid,
        "unsupported_extra_sentences": extra_total, "content_sentences_answerable": content_total,
        "unanswerable_cases": unanswerable, "correct_refusals": correct_refusals,
        "unexpected_refusals": unexpected_refusals,
        "answerable_cases": answerable, "abstain_answerable_incl_notes": abstain_answerable,
        "abstain_answerable_qids": abstain_detail,
        "fallback": ({"groups_composed": groups_total, "groups_fallback_to_template": groups_fallback,
                      "fallback_rate": round(groups_fallback / groups_total, 4) if groups_total else None,
                      "reasons": dict(fallback_reasons), "answer_decisions": dict(decisions)} if grounded else None),
        "per_type": {k: dict(v) for k, v in sorted(per_type.items())},
        "provider_failures": result["summary"]["provider_failures"],
        "provider_error_events": result["summary"]["provider_error_events"],
        "provider_failures_by_stage_type": dict(failures),
        "latency_ms": {"all": lat_block(lat_all), "warm": lat_block(lat_warm), "cold": lat_block(lat_cold),
                       "setup_ms": result.get("setup_latency_ms")},
        "effective": {
            "model": result["answer_engine"].get("model"), "resolved_provider": result["answer_engine"].get("resolved_provider"),
            "model_path": ((manifest or {}).get("model") or {}).get("path"),
            "temperature_requested": result["answer_engine"].get("temperature"),
            "temperature_manifest": (manifest or {}).get("temperature"),
            "device": (manifest or {}).get("device"), "nli_device": nli.get("device"),
            "grounded_settings": ({k: eff_settings.get(k) for k in eff_settings
                                   if k in ("t_relevant", "t_slot", "max_units", "numeric_guard", "composer",
                                            "T_RELEVANT", "T_SLOT", "MAX_UNITS", "NUMERIC_GUARD")} or eff_settings)
            if grounded else None,
            "code_sha": (manifest or {}).get("code_sha"), "code_dirty": (manifest or {}).get("code_dirty"),
            "resumes": (manifest or {}).get("resumes"),
        },
    }
    return out, per_qid


def screening(arms, per_type_names):
    a1, a4, a5 = arms[1], arms[4], arms[5]
    conds = []

    def add(name, ok, detail):
        conds.append({"condition": name, "pass": bool(ok), "detail": detail})

    s1, s4, s5 = (a["strict_supported_claims_met"] for a in (a1, a4, a5))
    add("supported claims met: arm5 >= arm1 + 3 AND arm5 >= arm4", s5 >= s1 + 3 and s5 >= s4,
        f"arm5={s5}, arm1={s1} (need >= {s1 + 3}), arm4={s4} (need >= {s4})")
    genuine = a5["unanswerable_cases"]
    add("correct refusals = all genuine refusal cases", a5["correct_refusals"] == genuine and genuine > 0,
        f"arm5 correct refusals={a5['correct_refusals']}/{genuine}")
    add("abstentions on answerable <= arm1", a5["abstain_answerable_incl_notes"] <= a1["abstain_answerable_incl_notes"],
        f"arm5={a5['abstain_answerable_incl_notes']}, arm1={a1['abstain_answerable_incl_notes']}")
    add("zero invalid markers", a5["invalid_markers"] == 0, f"arm5={a5['invalid_markers']}")
    add("zero provider failures", a5["provider_failures"] == 0 and a5["provider_error_events"] == 0,
        f"arm5 failed cases={a5['provider_failures']}, error events={a5['provider_error_events']}")
    worse = []
    for t in per_type_names:
        v5 = a5["per_type"].get(t, {}).get("supported_met", 0)
        v1 = a1["per_type"].get(t, {}).get("supported_met", 0)
        if v5 < v1 - 1:
            worse.append(f"{t}: arm5={v5} < arm1={v1} - 1")
    add("no type with fewer supported claims than arm1 by more than 1", not worse,
        "; ".join(worse) if worse else "per type arm5 vs arm1: " + ", ".join(
            f"{t} {a5['per_type'].get(t, {}).get('supported_met', 0)} vs {a1['per_type'].get(t, {}).get('supported_met', 0)}"
            for t in per_type_names))
    p95 = a5["latency_ms"]["warm"]["p95"]
    add(f"warm p95 <= {WARM_P95_LIMIT_MS / 1000:.0f} s", p95 is not None and p95 <= WARM_P95_LIMIT_MS,
        f"arm5 warm p95={p95} ms (n warm={a5['latency_ms']['warm']['n']})")
    verdict = "candidate default" if all(c["pass"] for c in conds) else "not a candidate"
    return {"rule": "plan section 7, applied literally to arm 5; engineering screen, not a statistical test "
                    "(40 blind cases, per-type differences of 1-2 claims are noise)",
            "conditions": conds, "verdict": verdict}


def paired(per_qid, left, right, cases):
    rows = []
    for qid in (c["qid"] for c in cases.values()):
        a, b = per_qid[left][qid], per_qid[right][qid]
        if a["required_claims"] == 0 and b["required_claims"] == 0:
            continue
        rows.append({"qid": qid, "type": a["type"], "required": a["required_claims"],
                     f"arm{left}": a["supported_met"], f"arm{right}": b["supported_met"],
                     "delta": a["supported_met"] - b["supported_met"]})
    return rows


def render_md(out) -> str:
    arms = out["arms"]
    L = []
    L.append("# heldout_v3 blind measurement (U7): 7 arms, each run once\n")
    L.append("Frozen blind set `data/evaluation/heldout_v3/answer_benchmark_heldout_v3.json` "
             f"(40 cases, sha256 `{out['cases_sha256']}`), BM25, temperature 0, `--save-raw`, one arm after another, fresh process each. "
             "Frozen grounded defaults (T_RELEVANT 0.01, T_SLOT 0.001, MAX_UNITS 6, NUMERIC_GUARD bound); no code or setting changed.\n")
    L.append("Engineering screen, not a statistical test: 40 cases, one run per arm; per-type differences of 1-2 claims are noise. "
             "Human adjudication of arms 1, 4, 5 is required before any default change; the user decides.\n")
    L.append(f"Machine load: {out['machine_load_note']}\n")
    L.append("## Screening verdict for arm 5 (grounded + mlx 7B)\n")
    for c in out["screening"]["conditions"]:
        L.append(f"- {'PASS' if c['pass'] else 'FAIL'}: {c['condition']} -- {c['detail']}")
    L.append(f"\nVerdict: **{out['screening']['verdict']}**\n")
    L.append("## Per-arm results\n")
    hdr = ["arm", "name", "lexical acc.", "NLI acc.", "strict supported met/req", "NLI claims met", "invalid markers",
           "unsupp. extra sent.", "correct refusals", "unexpected refusals", "abstain answerable (incl. notes)",
           "provider failures", "warm p50 ms", "warm p95 ms", "cold n / max ms"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for n in range(1, 8):
        a = arms[str(n)]
        lat = a["latency_ms"]
        cold_max = max([v for v in [lat["cold"]["p95"]] if v is not None], default=None)
        L.append("| " + " | ".join(str(x) for x in [
            n, a["name"], f"{a['lexical_accepted']}/{a['scored_answers']}", f"{a['nli_accepted']}/{a['scored_answers']}",
            f"{a['strict_supported_claims_met']}/{a['strict_required_claims']}",
            f"{a['nli_claims_met']}/{a['nli_required_claims']}", a["invalid_markers"],
            a["unsupported_extra_sentences"], f"{a['correct_refusals']}/{a['unanswerable_cases']}",
            a["unexpected_refusals"], f"{a['abstain_answerable_incl_notes']}/{a['answerable_cases']}",
            a["provider_failures"], lat["warm"]["p50"], lat["warm"]["p95"],
            f"{lat['cold']['n']} / {cold_max}"]) + " |")
    L.append("")
    L.append("Definitions. Strict supported claim: the gold claim is met by an answer sentence that matches lexically and whose "
             "cited gold hit passes the deterministic verifier (`src.evaluation.supported_claims`, same scorer as the dev confirmation). "
             "Grounded arms use the saved `context_snapshot`; arms 1-4 have none, so their client context is rebuilt "
             "by re-running the retrieval and context-building path with a stub provider (no model call; ids checked against `passage_ids`, method validated against the saved snapshots of the grounded arms). Unsupported extra sentence: an answer sentence on an answerable case that is not a coverage note or "
             "refusal, not a strict supporting sentence of a required claim, and not NLI-supported (cited and entailed >= 0.5). "
             "Abstention on answerable: refusal, all-notes, empty or marker-only answer. Unexpected refusal: refusal line on an answerable case. "
             "Correct refusals are counted on the genuine unanswerable cases only (R5: a comparison with one side absent is answerable and "
             "is not scored as a refusal). Warm = every case except those that loaded a model/scorer "
             f"(grounded: summed cold_load_ms > {COLD_MS:.0f} ms; other arms: the first case that called the model).\n")
    L.append("## Strict supported claims per type (met / required)\n")
    types = out["types"]
    L.append("| arm | " + " | ".join(types) + " |")
    L.append("|" + "---|" * (len(types) + 1))
    for n in range(1, 8):
        a = arms[str(n)]
        L.append(f"| {n} {a['name']} | " + " | ".join(
            f"{a['per_type'].get(t, {}).get('supported_met', 0)}/{a['per_type'].get(t, {}).get('required_claims', 0)}"
            for t in types) + " |")
    L.append("")
    L.append("## Grounded arms: fallback to template\n")
    for n in (5, 6, 7):
        a = arms[str(n)]
        fb = a["fallback"]
        L.append(f"- arm {n} {a['name']}: groups composed {fb['groups_composed']}, fallback to template "
                 f"{fb['groups_fallback_to_template']} (rate {fb['fallback_rate']}); reasons {fb['reasons']}; decisions {fb['answer_decisions']}")
    L.append("")
    L.append("## Abstentions on answerable cases (incl. coverage-note-only answers)\n")
    for n in range(1, 8):
        a = arms[str(n)]
        L.append(f"- arm {n} {a['name']}: {a['abstain_answerable_incl_notes']} "
                 f"({', '.join(x['qid'] + (' refusal' if x['refusal'] else ' no cited sentence') for x in a['abstain_answerable_qids']) or 'none'})")
    L.append("")
    L.append("## Latency (ms)\n")
    L.append("| arm | all n/p50/p95 | warm n/p50/p95 | cold n/p50/p95 | setup |")
    L.append("|---|---|---|---|---|")
    for n in range(1, 8):
        a = arms[str(n)]
        lat = a["latency_ms"]
        f = lambda b: f"{b['n']}/{b['p50']}/{b['p95']}"
        L.append(f"| {n} {a['name']} | {f(lat['all'])} | {f(lat['warm'])} | {f(lat['cold'])} | {lat['setup_ms']} |")
    L.append("")
    L.append(f"Machine load per arm start (from progress.txt): {'; '.join(out['load_per_arm_start'])}. The machine is shared with other agents; "
             "latency is indicative only.\n")
    L.append("## Effective model / device / temperature (from the manifests)\n")
    L.append("| arm | provider/composer | model path | temperature (requested; asserted at provider) | device | code sha (dirty) |")
    L.append("|---|---|---|---|---|---|")
    for n in range(1, 8):
        a = arms[str(n)]
        e = a["effective"]
        t = e["temperature_manifest"] or {}
        dev = e["device"] or {}
        L.append(f"| {n} {a['name']} | {a['provider']}/{a['composer']} | {e['model_path'] or e['model']} | "
                 f"{t.get('requested', e['temperature_requested'])}; {t.get('asserted_at_provider')} | "
                 f"{dev.get('machine')}, torch_mps={dev.get('torch_mps')}; NLI on {e['nli_device']} | "
                 f"{(e['code_sha'] or '')[:10]} ({e['code_dirty']}) |")
    L.append("")
    for n in range(1, 8):
        a = arms[str(n)]
        if a["effective"]["resumes"]:
            L.append(f"- arm {n} was resumed: {a['effective']['resumes']}")
        v = a["context_rebuild_validation"]
        if v:
            L.append(f"- arm {n}: context rebuild check against the saved snapshot: {v['rows_identical']}/{v['rows_checked']} rows identical (ids and text)")
        if any("MISMATCH" in x for x in a["strict_hits_source"]):
            L.append(f"- arm {n}: WARNING some rebuilt id lists differ from the saved passage_ids ({a['strict_hits_source']})")
        if a["strict_matches_rescore_nli"] is False:
            L.append(f"- arm {n}: strict count {a['strict_supported_claims_met']} differs from rescore_nli's column {a['strict_from_rescore_nli']}")
    L.append("")
    for left, right, title in ((5, 1, "arm 5 vs arm 1 (extractive)"), (5, 4, "arm 5 vs arm 4 (mlx 7B old path)")):
        L.append(f"## Paired strict supported claims per case: {title}\n")
        L.append(f"| qid | type | required | arm{left} | arm{right} | delta |")
        L.append("|---|---|---|---|---|---|")
        rows = out["paired"][f"arm{left}_vs_arm{right}"]
        for r in rows:
            L.append(f"| {r['qid']} | {r['type']} | {r['required']} | {r[f'arm{left}']} | {r[f'arm{right}']} | {r['delta']:+d} |")
        wins = sum(r["delta"] > 0 for r in rows)
        losses = sum(r["delta"] < 0 for r in rows)
        L.append(f"\nCases with a different count: arm{left} better in {wins}, worse in {losses}, equal in {len(rows) - wins - losses}.\n")
    return "\n".join(L) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", default=None)
    args = parser.parse_args()
    cases_list = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    cases = {c["qid"]: c for c in cases_list}
    docs = load_chunks_as_docs(ROOT / "data/processed/chunks_v3_paragraph.jsonl")
    arms, per_qid = {}, {}
    rebuilt_by_provider = {}
    for num, name, label in ARMS:
        prov = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))["answer_engine"]["provider"]
        stub = "llama" if prov == "llama" else "mlx" if prov in ("mlx", "grounded") else "local"
        if stub not in rebuilt_by_provider:
            rebuilt_by_provider[stub] = rebuild_contexts(stub, cases, docs)
        arms[num], per_qid[num] = summarize_arm(num, name, label, cases, rebuilt_by_provider[stub])
    types = [t for t in dict.fromkeys(c.get("question_type", "answerable") for c in cases_list)]
    answerable_types = [t for t in types if t != "unanswerable"]
    progress = (HERE / "progress.txt").read_text(encoding="utf-8").splitlines()
    loads = [f"{l.split()[1]} {l.split('load=')[1]}" for l in progress if l.startswith("START ") and "load=" in l]
    try:
        uptime = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    except Exception:
        uptime = "unknown"
    out = {
        "schema_version": 1, "set": "heldout_v3", "retrieval": "bm25", "temperature": 0.0,
        "cases": len(cases_list), "cases_sha256": json.loads((HERE / "local_extractive.json").read_text())["cases_sha256"],
        "metric": "strict supported claims (src.evaluation.supported_claims via calibrate_grounded.Scorer)",
        "machine_load_note": args.load or f"shared machine; uptime at summary time: {uptime}",
        "load_per_arm_start": loads,
        "types": types,
        "arms": {str(n): arms[n] for n in range(1, 8)},
        "screening": screening(arms, answerable_types),
        "paired": {"arm5_vs_arm1": paired(per_qid, 5, 1, cases), "arm5_vs_arm4": paired(per_qid, 5, 4, cases)},
        "per_case_strict": {str(n): per_qid[n] for n in per_qid},
    }
    (HERE / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "comparison.md").write_text(render_md(out), encoding="utf-8")
    for c in out["screening"]["conditions"]:
        print("PASS" if c["pass"] else "FAIL", c["condition"], "--", c["detail"])
    print("VERDICT:", out["screening"]["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
