"""Rebuild the FT-r4 pilot-gate dev slice from heldout_v6 only (PG-04, PG-05, PG-07, PG-08).

Recipe (premortem pilot-gate unit, minus the devpl questions that are gone):
  deep     16 answerable questions with >= 2 claims whose gold passage sits at prompt rank >= 4 in the FT-r3 served
           prompt, balanced over question types (the deep-rank recall that ft_v7 step 3a should lift)
  anchors   4 questions whose claims are all at rank 1-3 and that FT-r3 and FT-r1 both got right on >= 2 claims
  refusals  4 unanswerable questions
Outputs (default /Users/necatifurkancolak/AI-Workplace/Artifacts/BuffettRAG/pv8/slice/):
  slice.json          qids, groups, claims (required_terms, gold passage ids, rank in the new served prompt)
  slice_prompts.jsonl one record per model call: id, qid, messages exactly as served (prompt v3, current serving
                      settings, real CPU retrieval); a comparison question has one record per period call

    build_slice_prompts.py [--out DIR] [--select-only]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTHON_DOTENV_DISABLED", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VECTOR_BACKEND", "faiss")  # the local FAISS index; pgvector is the server default
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MAIN = Path(os.environ.get("BUFFETT_MAIN_CHECKOUT", "/Users/necatifurkancolak/AI-Workplace/Projects/current/BuffettRAG"))
ARTIFACTS = Path("/Users/necatifurkancolak/AI-Workplace/Artifacts/BuffettRAG")
CASES = [MAIN / f"data/evaluation/heldout_v6/answer_benchmark_heldout_v6_{s}.json" for s in "ab"]
FTR3 = [ARTIFACTS / f"pv7/runs/ftr3/ftr3_p16_e3_hv6{s}.json" for s in "ab"]
JUDGE = {"ftr3": ARTIFACTS / "pv7/judge/ftr3_v6/merged/claims_final.csv",
         "ftr1": ARTIFACTS / "pv7/judge/ftr1_v6_p16/merged/claims_final.csv"}
PROMPT_VERSION = "v3"
N_DEEP, N_ANCHORS, N_REFUSALS = 16, 4, 4
DEEP_RANK = 4


def load_cases(paths=CASES) -> Dict[str, Dict[str, Any]]:
    return {c["qid"]: c for p in paths for c in json.loads(Path(p).read_text(encoding="utf-8"))["cases"]}


def load_rows(paths=FTR3) -> Dict[str, Dict[str, Any]]:
    return {r["qid"]: r for p in paths for r in json.loads(Path(p).read_text(encoding="utf-8"))["rows"]}


def load_verdicts(path: Path) -> Dict[tuple, str]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return {(r["qid"], r["claim"]): r["ai_final"] for r in csv.DictReader(fh)}


def claim_ranks(case: Dict[str, Any], passage_ids: List[str]) -> List[Optional[int]]:
    """Best 1-based prompt rank of each claim's gold passages (None: not in the prompt)."""
    out = []
    for claim in case["gold_claims"]:
        ranks = [passage_ids.index(g) + 1 for g in claim["gold_passage_ids"] if g in passage_ids]
        out.append(min(ranks) if ranks else None)
    return out


def select_slice(cases, rows, verdicts_ftr3, verdicts_ftr1) -> Dict[str, List[str]]:
    """deep / anchors / refusals qid lists; ties broken by qid so the slice is reproducible."""
    deep_pool, anchor_pool = collections.defaultdict(list), []
    for qid in sorted(cases):
        case = cases[qid]
        if not case.get("answerable", True) or qid not in rows:
            continue
        ranks = claim_ranks(case, rows[qid]["passage_ids"])
        n_deep = sum(1 for r in ranks if r is not None and r >= DEEP_RANK)
        if n_deep >= 2:
            deep_pool[case["question_type"]].append((-n_deep, -len(ranks), qid))
        if ranks and all(r is not None and r < DEEP_RANK for r in ranks):
            met = sum(verdicts_ftr3.get((qid, c["claim"])) == "met" and verdicts_ftr1.get((qid, c["claim"])) == "met"
                      for c in case["gold_claims"])
            if met >= 2:
                anchor_pool.append((-len(ranks), qid))
    for pool in deep_pool.values():
        pool.sort()
    deep: List[str] = []
    while len(deep) < N_DEEP and any(deep_pool.values()):
        for kind in sorted(deep_pool):
            if deep_pool[kind] and len(deep) < N_DEEP:
                deep.append(deep_pool[kind].pop(0)[2])
    anchors = [qid for _, qid in sorted(anchor_pool)[:N_ANCHORS]]
    refusals = [q for q in sorted(cases) if not cases[q].get("answerable", True)][:N_REFUSALS]
    return {"deep": deep, "anchors": anchors, "refusals": refusals}


class _RecordingLLM:
    """Stands in for the model: records every prompt the serving code builds, answers with a non-refusal stub."""

    provider_name = "llama"
    model = "slice-recorder"

    def __init__(self):
        self.prompts: List[str] = []

    def generate(self, prompt, max_new_tokens=None):
        self.prompts.append(prompt)
        return "Recorded prompt only. [1]"


def _messages(prompt: str) -> List[Dict[str, str]]:
    from src.generation.providers.llama_provider import LlamaCppProvider
    return LlamaCppProvider._messages(None, prompt)  # the exact system/user split the model receives


_BLOCK_RE = re.compile(r"^\[(\d+)\] \(year=[^)]*\)\n(.*?)(?=\n\[\d+\] \(year=|\n\nEND UNTRUSTED PASSAGES)", re.M | re.S)


def prompt_passage_ids(user: str, context) -> List[str]:
    """Passage ids behind the [n] blocks of a served user message, in prompt order (matched by text)."""
    by_text = {h.text.strip(): h.id for h in context}
    ids = [by_text.get(text.strip()) for _, text in _BLOCK_RE.findall(user)]
    if not ids or None in ids:
        raise ValueError("a served passage block does not match any context passage")
    return ids


def serve_prompts(cases: Dict[str, Dict[str, Any]], qids: List[str]) -> Dict[str, Dict[str, Any]]:
    """qid -> {"calls": [{"id", "messages", "passage_ids", ...}], "retrieved", "passage_ids"} via the real serving path."""
    os.environ["PROMPT_VERSION"] = PROMPT_VERSION
    os.environ.pop("LLM_MODEL_PATH", None)
    import config
    from src.generation.compare import comparison_periods, generate_comparison_answer
    from src.pipeline import BuffettRAGPipeline, PipelineConfig
    from src.retrieval.context import build_doc_lookup
    from src.services import ask_flow
    from src.services.schemas import AskRequest
    from src.storage import load_chunks_as_docs

    corpus = config.CHUNKS_V3_FILE
    pipeline = BuffettRAGPipeline.build(PipelineConfig(chunks_file=corpus, use_llm=False))
    docs = load_chunks_as_docs(corpus)
    llm = _RecordingLLM()
    ask_flow._state = {"retriever": pipeline.retriever, "docs": docs, "docs_by_id": build_doc_lookup(docs), "llm": llm}
    out = {}
    for qid in qids:
        case = cases[qid]
        history = case.get("history", [])
        req = AskRequest(query=case["query"], history=history, strategy="hybrid", rerank=True)
        _, hits, _, _, context, prompt, expanded = ask_flow._prepare_ask(req)
        history_dicts = [dict(turn) for turn in history]
        calls = []
        if comparison_periods(case["query"], history_dicts):
            llm.prompts.clear()
            generate_comparison_answer(llm, case["query"], context, history=history_dicts, max_new_tokens=req.max_new_tokens,
                                       extra_queries=ask_flow._extra_queries(req, expanded))
            for prompt_text in llm.prompts:
                focus = re.findall(r"\(focus: ([^)]+)\)", prompt_text)
                calls.append({"id": f"{qid}#{focus[-1] if focus else len(calls)}", "prompt": prompt_text})
        elif context:
            calls.append({"id": qid, "prompt": prompt})
        for call in calls:
            call["messages"] = _messages(call.pop("prompt"))
            call["passage_ids"] = prompt_passage_ids(call["messages"][-1]["content"], context)
        out[qid] = {"calls": calls, "context_ids": [h.id for h in context], "retrieved_ids": [h.id for h in hits]}
    return out


def _token_counter():
    """Qwen2.5 chat-template token counter (the 1.5B tokenizer shares the 7B vocabulary), or None offline."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    except Exception:
        return None
    return lambda messages: len(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


def _serving_settings() -> Dict[str, Any]:
    import subprocess
    import config
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    return {"LLM_CONTEXT_PASSAGES": config.llm_context_passages(), "n_ctx": config.llm_n_ctx(),
            "COMPARE_ROUTING_V2": config.retrieval_setting("COMPARE_ROUTING_V2"),
            "ANSWER_CONTEXT_NEIGHBORS": config.retrieval_setting("ANSWER_CONTEXT_NEIGHBORS"),
            "LLM_PASSAGE_ORDER": (os.environ.get("LLM_PASSAGE_ORDER") or "natural"),
            "corpus": str(config.CHUNKS_V3_FILE.name), "faiss_index_name": config.FAISS_DIR.name, "git_commit": commit}


def build(out_dir: Path, select_only: bool = False) -> Dict[str, Any]:
    cases, rows = load_cases(), load_rows()
    groups = select_slice(cases, rows, load_verdicts(JUDGE["ftr3"]), load_verdicts(JUDGE["ftr1"]))
    qids = [q for g in ("deep", "anchors", "refusals") for q in groups[g]]
    group_of = {q: g for g in groups for q in groups[g]}
    served = {} if select_only else serve_prompts(cases, qids)
    count_tokens = None if select_only else _token_counter()
    questions = {}
    for qid in qids:
        case = cases[qid]
        ids = served[qid]["context_ids"] if qid in served else rows[qid]["passage_ids"]
        questions[qid] = {
            "group": group_of[qid], "question_type": case["question_type"], "query": case["query"],
            "history": case.get("history", []),
            "recorded_ftr3_passage_ids": rows[qid]["passage_ids"], "served_passage_ids": ids,
            "claims": [{"claim": c["claim"], "required_terms": c.get("required_terms", []),
                        "gold_passage_ids": c["gold_passage_ids"], "rank": r}
                       for c, r in zip(case.get("gold_claims", []), claim_ranks(case, ids) if case.get("gold_claims") else [])]}
    prompts = []
    for qid in qids:
        for call in (served.get(qid) or {}).get("calls", []):
            period = call["id"].partition("#")[2]
            prompts.append({**call, "qid": qid, "group": group_of[qid], "period": period or None, "prompt_version": PROMPT_VERSION,
                            "n_passages": len(call["passage_ids"]),
                            **({"n_tokens": count_tokens(call["messages"])} if count_tokens else {})})
    slice_doc = {
        "source": "heldout_v6 only (devpl is gone); FT-r3 pv7 runs for ranks, pv7 judge verdicts for anchors",
        "prompt_version": PROMPT_VERSION, "serving_settings": _serving_settings(), "groups": groups,
        "counts": {"questions": len(qids), **{g: len(v) for g, v in groups.items()},
                   "claims": sum(len(q["claims"]) for q in questions.values()),
                   "deep_claims": sum(1 for q in questions.values() if q["group"] == "deep"
                                      for c in q["claims"] if c["rank"] and c["rank"] >= DEEP_RANK),
                   "prompts": len(prompts)},
        "same_passages_as_recorded": sorted(q for q in qids if q in served and served[q]["context_ids"] == rows[q]["passage_ids"]),
        "questions": questions}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "slice.json").write_text(json.dumps(slice_doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if not select_only:
        with (out_dir / "slice_prompts.jsonl").open("w", encoding="utf-8") as fh:
            for record in prompts:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return slice_doc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "pv8/slice")
    parser.add_argument("--select-only", action="store_true", help="write slice.json from the recorded runs, no retrieval")
    args = parser.parse_args()
    doc = build(args.out, args.select_only)
    print(json.dumps({"out": str(args.out), **doc["counts"], "same_passages_as_recorded": len(doc["same_passages_as_recorded"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
