"""Cloud dev-set CPU check: score the 4 arms and write summary.json and comparison.md next to this file.

Run after run_arms.sh, from the repository root:

    PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 .venv/bin/python data/evaluation/cloud_dev_check_v1/summarize.py

NLI: scripts/eval/rescore_nli.py writes <arm>.nli.json (model in models/nli-deberta-v3-base).
Strict supported claims: src.evaluation.supported_claims through calibrate_grounded.Scorer, against the client
context rebuilt by re-running retrieval + prepare_answer_context with a stub provider of the arm's provider name
(no model call), checked against each row's saved passage_ids; same method as heldout_v3/final/summarize.py.
Dev cases only. Engineering read-out for the owner, not a statistical test; no default is changed on it.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from scripts.eval.calibrate_grounded import Scorer  # noqa: E402
from scripts.eval.rescore_nli import rescore  # noqa: E402
from src.evaluation.answer_benchmark import is_unanswerable_case  # noqa: E402
from src.evaluation.supported_claims import supported_claims_met  # noqa: E402
from src.generation.prompt import REFUSAL_LINE, format_answer_markdown, strip_chat_artifacts  # noqa: E402
from src.services.ask_flow import is_citation_only  # noqa: E402
from src.storage import SearchHit, load_chunks_as_docs  # noqa: E402

CASES_PATH = ROOT / "data/evaluation/grounded_dev_v1/confirm/dev_cases.json"
ARMS = [  # (name, label, code)
    ("extractive_main", "extractive, main code (old)", "main"),
    ("extractive_lead", "extractive, task A (lead + <=2 supporting)", "branch"),
    ("llama_ftr2_guard", "llama FT-r2 1.5B, task B guard", "branch"),
    ("llama_base", "llama base Qwen2.5-1.5B", "branch"),
]


def pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo), 1)


def is_refusal(answer) -> bool:
    return REFUSAL_LINE.casefold() in (answer or "").casefold()


class _StubLLM:
    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name
        self.model = "stub"

    def generate(self, *_a, **_k):
        raise RuntimeError("stub")


def rebuild_contexts(provider_name, cases, docs):
    import importlib.util
    from src.retrieval.context import build_doc_lookup
    from src.services import ask_flow, backend_app as backend
    spec = importlib.util.spec_from_file_location("run_live_benchmark_cdc", ROOT / "scripts/eval/run_live_benchmark.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    old = (ask_flow._state, ask_flow.EXPOSE_DEBUG_STATUS)
    ask_flow._state = {"retriever": runner.BM25OnlyRetriever(docs), "docs": docs,
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


def summarize_arm(name, label, code, cases, docs, contexts):
    result = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
    nli = rescore(HERE / f"{name}.json", CASES_PATH)
    provider = result["answer_engine"]["provider"]
    if provider not in contexts:
        contexts[provider] = rebuild_contexts(provider, cases, docs)
    rebuilt = contexts[provider]
    scorer = Scorer()
    met = required = id_mismatch = correct_refusals = unanswerable = unexpected = abstain = answerable = 0
    citation_only, citation_only_qids, latencies = 0, [], []
    for row in result["rows"]:
        case = dict(cases[row["qid"]])
        latencies += [row["latency_ms"]] if row.get("latency_ms") is not None else []
        raws = row.get("raw_answers") or []
        if provider != "local" and any(is_citation_only(format_answer_markdown(strip_chat_artifacts(r))) for r in raws):
            citation_only += 1
            citation_only_qids.append(row["qid"])
        if row.get("status") != "scored":
            continue
        hits = rebuilt[row["qid"]]
        id_mismatch += [h.id for h in hits] != (row.get("passage_ids") or [])
        s = scorer.row(case, row.get("answer"), row.get("citations"), hits)
        if s["unanswerable"]:
            unanswerable += 1
            correct_refusals += s["correct_refusal"] is True
            continue
        answerable += 1
        abstain += s["abstained"]
        unexpected += is_refusal(row.get("answer"))
        claims = supported_claims_met(case, row.get("answer") or "", row.get("citations") or [], hits)
        met += sum(c["met"] for c in claims)
        required += len(claims)
    return {
        "name": name, "label": label, "code": code, "provider": provider,
        "model": result["answer_engine"].get("model"),
        "cases_run": result["summary"]["cases_run"], "scored_answers": result["summary"]["scored_answers"],
        "lexical_accepted": result["summary"]["accepted"], "nli_accepted": nli["nli_accepted"],
        "nli_claims_met": nli["nli_required_claims_met"], "nli_required_claims": nli["required_claims"],
        "strict_supported_claims_met": met, "strict_required_claims": required,
        "context_id_mismatches": id_mismatch,
        "unanswerable_cases": unanswerable, "correct_refusals": correct_refusals,
        "answerable_cases": answerable, "unexpected_refusals": unexpected, "abstain_answerable": abstain,
        "citation_only_caught_by_guard": citation_only if provider != "local" else None,
        "citation_only_qids": citation_only_qids,
        "provider_failures": result["summary"]["provider_failures"],
        "latency_ms": {"n": len(latencies), "p50": pct(latencies, 0.5), "p95": pct(latencies, 0.95),
                       "mean": result["summary"]["mean_latency_ms"]},
    }


def render_md(out) -> str:
    L = ["# Cloud dev-set CPU check (v1)\n",
         f"Dev cases only: `data/evaluation/grounded_dev_v1/confirm/dev_cases.json` ({out['n_cases']} cases, "
         f"sha256 `{out['cases_sha256'][:16]}...`; the union of the four dev sets heldout_v1, heldout_v2, "
         "answer_benchmark_v3 and adversarial_dev that the file marks `dev_only`). No frozen held-out set "
         "(heldout_v2/ or heldout_v3/ folders) was read, run or scored.\n",
         f"BM25 retrieval, temperature 0, CPU only (`LLM_GPU_LAYERS=0`, {out['cpu_note']}), one arm after another, "
         f"fresh process each, `--save-raw`. {out['scope_note']}\n",
         "Engineering read-out for the owner, not a statistical test; no default setting was changed on these numbers.\n",
         "| arm | strict supported claims met/required | NLI accepted | lexical accepted | correct refusals | "
         "unexpected refusals | citation-only answers caught by guard | provider failures | p50 latency ms | p95 latency ms |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for i, a in enumerate(out["arms"], 1):
        caught = "n/a" if a["citation_only_caught_by_guard"] is None else a["citation_only_caught_by_guard"]
        L.append("| " + " | ".join(str(x) for x in [
            f"{i} {a['label']}", f"{a['strict_supported_claims_met']}/{a['strict_required_claims']}",
            f"{a['nli_accepted']}/{a['scored_answers']}", f"{a['lexical_accepted']}/{a['scored_answers']}",
            f"{a['correct_refusals']}/{a['unanswerable_cases']}", f"{a['unexpected_refusals']}/{a['answerable_cases']}",
            caught, a["provider_failures"], a["latency_ms"]["p50"], a["latency_ms"]["p95"]]) + " |")
    L.append("")
    L.append("Definitions. Strict supported claim: a gold claim met by an answer sentence that matches lexically and "
             "whose cited gold hit passes the deterministic verifier (`src.evaluation.supported_claims`), scored against "
             "the client context rebuilt with a stub provider (row passage ids checked: "
             + ", ".join(f"arm {i} {a['context_id_mismatches']} mismatches" for i, a in enumerate(out["arms"], 1))
             + "). NLI accepted: `scripts/eval/rescore_nli.py` with `models/nli-deberta-v3-base` (entailment >= 0.5). "
             "Correct refusals are over the unanswerable cases; unexpected refusals are refusal-line answers on answerable "
             "cases. Citation-only: a raw model answer with fewer than three words once `[n]` markers and punctuation "
             "are removed (`ask_flow.is_citation_only`); the guard replaces it with the extractive answer. Latency is "
             "per case end to end on this CPU, including the first (cold) case.\n")
    L.append("Citation-only qids: " + "; ".join(
        f"arm {i}: {', '.join(a['citation_only_qids']) or 'none'}" for i, a in enumerate(out["arms"], 1)
        if a["citation_only_caught_by_guard"] is not None) + "\n")
    return "\n".join(L)


def main() -> None:
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope-note", default="All cases run for every arm.")
    args = ap.parse_args()
    cases_doc = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = {c["qid"]: c for c in cases_doc["cases"]}
    from config import CHUNKS_V3_FILE
    docs = load_chunks_as_docs(CHUNKS_V3_FILE)
    contexts: dict = {}
    arms = [summarize_arm(name, label, code, cases, docs, contexts) for name, label, code in ARMS]
    out = {"n_cases": len(cases), "cases": str(CASES_PATH.relative_to(ROOT)),
           "cases_sha256": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
           "cpu_note": f"{os.cpu_count()} vCPU", "scope_note": args.scope_note, "arms": arms}
    (HERE / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "comparison.md").write_text(render_md(out), encoding="utf-8")
    print(render_md(out))


if __name__ == "__main__":
    main()
