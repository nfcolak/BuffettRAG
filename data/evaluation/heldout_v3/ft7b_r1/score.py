"""Score the heldout_v3 7B arms (8 = base, 9 = FT-r1) with the U7 scorer and the pre-registered screen.

Run: env -u PYTHONPATH PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 <repo>/.venv/bin/python score.py
Writes results/{llama7b_base,llama7b_ft}.nli.json, results/summary.json, results/comparison.md,
judge-input/{claims_input,unanswerable_input}.csv and judge-key.json (blinded system labels, not for judges).
Arm 1 (extractive) is re-scored from its saved U7 files (heldout_v3/final) as the reference; it is not re-run.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

R = Path("/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG")
D = Path(__file__).resolve().parent
RES = D / "results"
FINAL = R / "data/evaluation/heldout_v3/final"
sys.path.insert(0, str(R))


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load_mod("u7_summarize", FINAL / "summarize.py")
C = load_mod("cloud_summarize", R / "data/evaluation/cloud_dev_check_v1/summarize.py")
from scripts.eval.rescore_nli import rescore  # noqa: E402
from src.evaluation.answer_benchmark import is_unanswerable_case  # noqa: E402
from src.storage import load_chunks_as_docs  # noqa: E402

ARMS = [(1, "local_extractive", "extractive (U7 reference, not re-run)", FINAL),
        (8, "llama7b_base", "llama 7B base (Kaggle q4_k_m)", RES),
        (9, "llama7b_ft", "llama 7B FT-r1 (Kaggle q4_k_m)", RES)]


def main():
    cases_list = json.loads(S.CASES_PATH.read_text(encoding="utf-8"))["cases"]
    cases = {c["qid"]: c for c in cases_list}
    for _, name, _, here in ARMS[1:]:
        if not (here / f"{name}.nli.json").exists():
            print("NLI rescoring", name, flush=True)
            rescore(here / f"{name}.json", S.CASES_PATH, scorer=C.nli_scorer())
    docs = load_chunks_as_docs(R / "data/processed/chunks_v3_paragraph.jsonl")
    rebuilt, arms, per_qid, strict_detail = {}, {}, {}, {}
    for num, name, label, here in ARMS:
        result = json.loads((here / f"{name}.json").read_text(encoding="utf-8"))
        prov = result["answer_engine"]["provider"]
        stub = "llama" if prov == "llama" else "local"
        if stub not in rebuilt:
            rebuilt[stub] = S.rebuild_contexts(stub, cases, docs)
        S.HERE = here
        arms[num], per_qid[num] = S.summarize_arm(num, name, label, cases, rebuilt[stub])
        strict_detail[num] = S.strict_rows_for_arm(result, cases, rebuilt[stub], S.Scorer())
        arms[num]["source_dir"] = str(here.relative_to(R)) if here == FINAL else "data/evaluation/heldout_v3/ft7b_r1"
    types = list(dict.fromkeys(c.get("question_type", "answerable") for c in cases_list))
    ans_types = [t for t in types if t != "unanswerable"]

    a1, a8, a9 = arms[1], arms[8], arms[9]
    s1, s8, s9 = (a["strict_supported_claims_met"] for a in (a1, a8, a9))
    conds = []
    add = lambda n, ok, d: conds.append({"condition": n, "pass": bool(ok), "detail": d})
    add("strict supported claims: arm9 >= arm1 + 3 AND arm9 >= arm8", s9 >= s1 + 3 and s9 >= s8,
        f"arm9={s9}, arm1={s1} (need >= {s1 + 3}), arm8={s8}")
    g = a9["unanswerable_cases"]
    add("correct refusals = all genuine unanswerable cases", a9["correct_refusals"] == g and g > 0,
        f"arm9 correct refusals={a9['correct_refusals']}/{g}")
    add("abstentions on answerable <= arm1", a9["abstain_answerable_incl_notes"] <= a1["abstain_answerable_incl_notes"],
        f"arm9={a9['abstain_answerable_incl_notes']}, arm1={a1['abstain_answerable_incl_notes']}")
    add("zero invalid markers", a9["invalid_markers"] == 0, f"arm9={a9['invalid_markers']}")
    add("zero provider failures", a9["provider_failures"] == 0 and a9["provider_error_events"] == 0,
        f"arm9 failed cases={a9['provider_failures']}, error events={a9['provider_error_events']}")
    pt = lambda a, t: a["per_type"].get(t, {}).get("supported_met", 0)
    worse = [f"{t}: arm9={pt(a9, t)} < arm1={pt(a1, t)} - 1" for t in ans_types if pt(a9, t) < pt(a1, t) - 1]
    add("no question type with fewer strict claims than arm1 by more than 1", not worse,
        "; ".join(worse) or "per type arm9 vs arm1: " + ", ".join(f"{t} {pt(a9, t)} vs {pt(a1, t)}" for t in ans_types))
    verdict = "candidate default" if all(c["pass"] for c in conds) else "not a candidate"
    progress = (RES / "progress.txt").read_text(encoding="utf-8").splitlines()
    out = {
        "schema_version": 1, "set": "heldout_v3", "retrieval": "bm25", "temperature": 0.0, "cases": len(cases_list),
        "cases_sha256": json.loads((RES / "llama7b_ft.json").read_text())["cases_sha256"],
        "preregistration": "Artifacts/BuffettRAG/heldout7b/PREREGISTRATION.md (written before the run; copied here)",
        "device": "local Mac, llama.cpp Metal (LLM_GPU_LAYERS=-1); latency not comparable with U7 arms",
        "progress": progress, "types": types,
        "arms": {str(n): arms[n] for n in (1, 8, 9)},
        "screening": {"rule": "pre-registered: U7 plan section 7 applied to arm 9, comparator arm 4 replaced by arm 8; "
                              "engineering screen, not a statistical test", "conditions": conds, "verdict": verdict},
        "paired": {"arm9_vs_arm1": S.paired(per_qid, 9, 1, cases), "arm9_vs_arm8": S.paired(per_qid, 9, 8, cases)},
        "per_case_strict": {str(n): per_qid[n] for n in per_qid},
    }
    (RES / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (RES / "comparison.md").write_text(render(out, ans_types), encoding="utf-8")
    for c in conds:
        print("PASS" if c["pass"] else "FAIL", c["condition"], "--", c["detail"])
    print("VERDICT:", verdict)
    judge_inputs(cases, cases_list, strict_detail)


def render(out, ans_types):
    A = out["arms"]
    L = ["# heldout_v3 blind measurement: 7B base (arm 8) and Kaggle 7B FT-r1 (arm 9), each run once\n",
         f"Frozen blind set `data/evaluation/heldout_v3/answer_benchmark_heldout_v3.json` (40 cases, sha256 `{out['cases_sha256']}`), "
         "BM25, temperature 0, `--save-raw`, llama provider, fresh process per arm, no code or setting change. "
         "Custody: `scripts/eval/check_heldout_v3.py` OK (FT answer passages of data/ft_v3 train/valid excluded). "
         "Screen written before the run (PREREGISTRATION.md). Arm 1 is the U7 extractive result re-scored from `../final/`, not re-run.\n",
         "Engineering screen, not a statistical test: 40 cases, one run per arm; per-type differences of 1-2 claims are noise. "
         f"{out['device']}.\n",
         "## Screening verdict for arm 9 (7B FT-r1)\n"]
    L += [f"- {'PASS' if c['pass'] else 'FAIL'}: {c['condition']} -- {c['detail']}" for c in out["screening"]["conditions"]]
    L.append(f"\nVerdict: **{out['screening']['verdict']}** (no default changed; the user decides)\n")
    L.append("## Per-arm results\n")
    L.append("| arm | name | lexical acc. | NLI acc. | strict supported met/req | NLI claims met | invalid markers | "
             "unsupp. extra sent. | correct refusals | unexpected refusals | abstain answerable | provider failures | warm p50 ms | warm p95 ms |")
    L.append("|" + "---|" * 14)
    for n in ("1", "8", "9"):
        a = A[n]
        w = a["latency_ms"]["warm"]
        L.append(f"| {n} | {a['name']} | {a['lexical_accepted']}/{a['scored_answers']} | {a['nli_accepted']}/{a['scored_answers']} | "
                 f"{a['strict_supported_claims_met']}/{a['strict_required_claims']} | {a['nli_claims_met']}/{a['nli_required_claims']} | "
                 f"{a['invalid_markers']} | {a.get('unsupported_extra_sentences')} | {a['correct_refusals']}/{a['unanswerable_cases']} | "
                 f"{a['unexpected_refusals']} | {a['abstain_answerable_incl_notes']}/{a['answerable_cases']} | {a['provider_failures']} | "
                 f"{w.get('p50')} | {w.get('p95')} |")
    L.append("\n## Strict supported claims per type (met / required)\n")
    L.append("| arm | " + " | ".join(ans_types) + " |")
    L.append("|" + "---|" * (len(ans_types) + 1))
    for n in ("1", "8", "9"):
        pt = A[n]["per_type"]
        L.append(f"| {n} {A[n]['name']} | " + " | ".join(
            f"{pt.get(t, {}).get('supported_met', 0)}/{pt.get(t, {}).get('required_claims', 0)}" for t in ans_types) + " |")
    for key in ("arm9_vs_arm1", "arm9_vs_arm8"):
        rows = out["paired"][key]
        wins, losses = sum(r["delta"] > 0 for r in rows), sum(r["delta"] < 0 for r in rows)
        L.append(f"\nPaired {key}: arm 9 better in {wins} cases, worse in {losses}, equal in {len(rows) - wins - losses}.")
    return "\n".join(L) + "\n"


def judge_inputs(cases, cases_list, strict_detail):
    """Blinded judge sheets: systems relabelled P/Q at random, rows shuffled; no automatic-check columns."""
    ji = D / "judge-input"
    ji.mkdir(exist_ok=True)
    rng = random.Random(20261009)
    names = {8: "llama7b_base", 9: "llama7b_ft"}
    labels = ["system P", "system Q"]
    rng.shuffle(labels)
    key = {"system_labels": {labels[0]: names[8], labels[1]: names[9]}, "claims": {}, "unanswerable": {}}
    claim_rows, un_rows = [], []
    for num, label in ((8, labels[0]), (9, labels[1])):
        result = json.loads((RES / f"{names[num]}.json").read_text(encoding="utf-8"))
        for row in result["rows"]:
            case = cases[row["qid"]]
            if is_unanswerable_case(case):
                un_rows.append((label, num, row["qid"], case["query"], row.get("answer") or "", None))
                continue
            for c in strict_detail[num][row["qid"]]["claims"]:
                claim_rows.append((label, num, row["qid"], case.get("question_type"), case["query"], row.get("answer") or "",
                                   c["claim"], bool(c["met"])))
    rng.shuffle(claim_rows)
    rng.shuffle(un_rows)
    with open(ji / "claims_input.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "system", "qid", "type", "question", "answer", "required claim"])
        for i, (label, num, qid, t, q, a, claim, met) in enumerate(claim_rows, 1):
            rid = f"c{i:03d}"
            w.writerow([rid, label, qid, t, q, a, claim])
            key["claims"][rid] = {"arm": num, "qid": qid, "claim": claim, "strict_met": met}
    with open(ji / "unanswerable_input.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "system", "qid", "question", "answer"])
        for i, (label, num, qid, q, a, _) in enumerate(un_rows, 1):
            rid = f"u{i:03d}"
            w.writerow([rid, label, qid, q, a])
            sc = strict_detail[num][qid]["scorer"] or {}
            key["unanswerable"][rid] = {"arm": num, "qid": qid, "auto_correct_refusal": sc.get("correct_refusal")}
    (D / "judge-key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"judge inputs: {len(claim_rows)} claim rows, {len(un_rows)} unanswerable rows")


if __name__ == "__main__":
    main()
