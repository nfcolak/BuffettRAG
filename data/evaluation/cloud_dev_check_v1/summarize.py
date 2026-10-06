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
ARMS = [  # (name, label, code); the informational arm gets strict scoring only
    ("extractive_main", "extractive, main code (old)", "main"),
    ("extractive_lead", "extractive, task A (lead + <=2 supporting)", "branch"),
    ("extractive_lead_unstemmed", "info: task A with the original unstemmed ranking", "branch, one line changed"),
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


_NLI = None
_NLI_BATCH = 16


def nli_scorer():
    """NliScorer that batches premise windows: same numbers, bounded memory, far fewer forward calls.

    The stock scorer pads every window of one premise into a single batch (this exhausted the machine's memory on
    long premises) and scores each 1-3 sentence window of a cited block in its own call (hours per arm on CPU).
    Here every (premise window, hypothesis) pair of a support check runs in batches of _NLI_BATCH; the result per
    premise is unchanged: max P(entailment) over its windows where entailment is the argmax label, cached per
    (premise, hypothesis) exactly as before.
    """
    global _NLI
    if _NLI is None:
        from src.evaluation.citation_faithfulness import split_sentences
        from src.evaluation.nli_scorer import NliScorer

        class BatchedNliScorer(NliScorer):
            def _entail_many(self, premises, hypothesis):
                todo = [p for p in dict.fromkeys(premises)
                        if (p, hypothesis) not in self._cache]
                pairs = []
                for p in todo:
                    if p.strip() and hypothesis.strip():
                        pairs += [(p, w) for w in self._windows(p, hypothesis)]
                best = {p: 0.0 for p in todo}
                torch = self._torch
                for start in range(0, len(pairs), _NLI_BATCH):
                    part = pairs[start:start + _NLI_BATCH]
                    enc = self.tok([w for _, w in part], [hypothesis] * len(part), truncation=True, max_length=512,
                                   padding=True, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu()
                    for (p, _), row in zip(part, probs):
                        if int(row.argmax()) == self.entail_idx:
                            best[p] = max(best[p], float(row[self.entail_idx]))
                for p, value in best.items():
                    self._cache[(p, hypothesis)] = value
                return max((self._cache[(p, hypothesis)] for p in premises), default=0.0)

            def entail_prob(self, premise, hypothesis):
                return self._entail_many([premise], hypothesis)

            def support_prob(self, block, sentence):
                if not block.strip() or not sentence.strip():
                    return 0.0
                if self._norm(sentence) and self._norm(sentence) in self._norm(block):
                    return 1.0
                bs = [x for x in split_sentences(block) if x.strip()]
                premises = [block] + [" ".join(bs[i:i + n]) for n in (1, 2, 3)
                                      for i in range(max(0, len(bs) - n + 1))]
                return self._entail_many(premises, sentence)

            def score_row(self, row, case):
                out = super().score_row(row, case)
                self.rows_done = getattr(self, "rows_done", 0) + 1
                if self.rows_done % 10 == 0:
                    print(f"  nli rows scored: {self.rows_done}", flush=True)
                return out

        _NLI = BatchedNliScorer()
    return _NLI


def rescore_first(name, max_cases):
    """NLI-rescore the first max_cases rows of an arm; writes <arm>.nli.json next to this file."""
    import shutil
    import tempfile
    result = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
    result["rows"] = result["rows"][:max_cases]
    result["summary"]["accepted"] = sum(bool((r.get("score") or {}).get("accepted")) for r in result["rows"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        summary = rescore(path, CASES_PATH, scorer=nli_scorer())
        shutil.move(str(path.with_suffix(".nli.json")), HERE / f"{name}.nli.json")
    return summary


def summarize_arm(name, label, code, cases, docs, contexts, max_cases, with_nli):
    result = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
    print(f"scoring {name}", flush=True)
    nli = rescore_first(name, max_cases) if with_nli else None
    first = {row["qid"] for row in result["rows"][:max_cases]}
    provider = result["answer_engine"]["provider"]
    if provider not in contexts:
        contexts[provider] = rebuild_contexts(provider, cases, docs)
    rebuilt = contexts[provider]
    scorer = Scorer()
    met = required = id_mismatch = correct_refusals = unanswerable = unexpected = abstain = answerable = 0
    citation_only, citation_only_qids, latencies = 0, [], []
    all_met = all_required = 0
    for row in result["rows"]:
        case = dict(cases[row["qid"]])
        if row.get("status") == "scored" and not is_unanswerable_case(case):
            claims = supported_claims_met(case, row.get("answer") or "", row.get("citations") or [],
                                          rebuilt[row["qid"]])
            all_met += sum(c["met"] for c in claims)
            all_required += len(claims)
        if row["qid"] not in first:
            continue
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
        "cases_run": result["summary"]["cases_run"], "cases_scored_here": len(first),
        "scored_answers": sum(r.get("status") == "scored" for r in result["rows"][:max_cases]),
        "lexical_accepted": sum(bool((r.get("score") or {}).get("accepted")) for r in result["rows"][:max_cases]),
        "nli_accepted": nli["nli_accepted"] if nli else None,
        "nli_claims_met": nli["nli_required_claims_met"] if nli else None,
        "nli_required_claims": nli["required_claims"] if nli else None,
        "strict_supported_claims_met": met, "strict_required_claims": required,
        "strict_all_cases_met": all_met, "strict_all_cases_required": all_required,
        "context_id_mismatches": id_mismatch,
        "unanswerable_cases": unanswerable, "correct_refusals": correct_refusals,
        "answerable_cases": answerable, "unexpected_refusals": unexpected, "abstain_answerable": abstain,
        "citation_only_caught_by_guard": citation_only if provider != "local" else None,
        "citation_only_qids": citation_only_qids,
        "provider_failures": result["summary"]["provider_failures"],
        "latency_ms": {"n": len(latencies), "p50": pct(latencies, 0.5), "p95": pct(latencies, 0.95),
                       "mean": round(sum(latencies) / len(latencies), 1) if latencies else None},
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
         f"Columns are over the first {out['max_cases']} cases of every arm except the last strict column, which "
         "covers all 80 cases.\n",
         f"| arm | strict supported claims met/required (first {out['max_cases']}) | NLI accepted | lexical accepted | "
         "correct refusals | unexpected refusals | citation-only answers caught by guard | provider failures | "
         "p50 latency ms | p95 latency ms | strict met/required (all 80) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, a in enumerate(out["arms"], 1):
        caught = "n/a" if a["citation_only_caught_by_guard"] is None else a["citation_only_caught_by_guard"]
        L.append("| " + " | ".join(str(x) for x in [
            f"{i} {a['label']}", f"{a['strict_supported_claims_met']}/{a['strict_required_claims']}",
            "not scored" if a["nli_accepted"] is None else f"{a['nli_accepted']}/{a['scored_answers']}",
            f"{a['lexical_accepted']}/{a['scored_answers']}",
            f"{a['correct_refusals']}/{a['unanswerable_cases']}", f"{a['unexpected_refusals']}/{a['answerable_cases']}",
            caught, a["provider_failures"], a["latency_ms"]["p50"], a["latency_ms"]["p95"],
            f"{a['strict_all_cases_met']}/{a['strict_all_cases_required']}"]) + " |")
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
    ap.add_argument("--max-cases", type=int, default=80, help="score only the first N cases of every arm")
    args = ap.parse_args()
    cases_doc = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = {c["qid"]: c for c in cases_doc["cases"]}
    from config import CHUNKS_V3_FILE
    docs = load_chunks_as_docs(CHUNKS_V3_FILE)
    contexts: dict = {}
    arms = [summarize_arm(name, label, code, cases, docs, contexts, args.max_cases,
                          with_nli=not name.endswith("_unstemmed")) for name, label, code in ARMS]
    out = {"n_cases": len(cases), "cases": str(CASES_PATH.relative_to(ROOT)),
           "cases_sha256": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
           "cpu_note": f"{os.cpu_count()} vCPU", "scope_note": args.scope_note,
           "max_cases": args.max_cases, "arms": arms}
    (HERE / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "comparison.md").write_text(render_md(out), encoding="utf-8")
    print(render_md(out))


if __name__ == "__main__":
    main()
