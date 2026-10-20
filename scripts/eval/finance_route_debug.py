"""Finance route diagnostics without an LLM (finance-dev-fix-1, task 1.1).

Runs `ask_flow.finance_route` (plan, route decision, finance_search, served context) for every finance_v1 DEV question
on corpus v3 + FAISS with the local bge embedder and reranker, and a fake LLM that never generates. The fake LLM is a
"remote" provider, as in the cloud app, so the served list is the one the deployed /ask returns (no llama window fit).

  python scripts/eval/finance_route_debug.py [--set dev|heldout_v6] [--loss] [--cache PATH] [--compare RESPONSES.jsonl]
                                             [--json OUT] [--ids]

Per question: qid, group, plan (scope, kind, years, events, sectors, companies), route, served ids (with --ids) and
gold served / total. It ends with the gold-served share per group (mean over the questions that have gold passages,
as scripts/eval/score_finance.py computes it from a run) and the route counts. --compare adds the gold served of a
stored run (its served_ids) per question and group. Only the dev half and heldout_v6 are read; the test half is never
opened.

--set heldout_v6 merges answer_benchmark_heldout_v6_{a,b}.json (second check, already spent) and groups by question_type.
--loss adds, per group, the first stage that lost each missing gold passage (finance-recall-75, task 1.1):
  not_in_pool    not among the hybrid candidates of any ranking (nor injected by a candidate source)
  scope_filtered annotated non-financial (or a table on a non-numeric plan)
  rerank_cut     in the pool but below the reranker window
  budget_cut     reranked, but below the served budget k
  letter_cap     reranked inside the budget, removed by the per-letter cap / diversity pass
  other          anything else (deduplicated, cut by the reader context, or a route that does not search)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTHON_DOTENV_DISABLED", "1")
os.environ.setdefault("VECTOR_BACKEND", "faiss")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["FINANCE_PIPELINE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEV_FILE = ROOT / "data/evaluation/finance_v1/dev.json"
HELDOUT_FILES = [ROOT / f"data/evaluation/heldout_v6/answer_benchmark_heldout_v6_{part}.json" for part in "ab"]
GROUP_NAMES = {1: "crises/macro", 2: "views", 3: "figures", 4: "personal", 5: "whole collection"}
LOSS_STAGES = ("not_in_pool", "scope_filtered", "rerank_cut", "budget_cut", "letter_cap", "other")


def load_cases(name: str) -> List[Dict[str, Any]]:
    """dev: the finance_v1 dev half (group 1-5). heldout_v6: the a and b files merged, grouped by question_type."""
    if name == "dev":
        return json.loads(DEV_FILE.read_text(encoding="utf-8"))["cases"]
    cases: List[Dict[str, Any]] = []
    for path in HELDOUT_FILES:
        cases += json.loads(path.read_text(encoding="utf-8"))["cases"]
    for case in cases:
        case["group"] = case["question_type"]
    return cases


def group_label(group: Any) -> str:
    return f"{group} {GROUP_NAMES[group]}" if group in GROUP_NAMES else str(group)


def classify_loss(pid: str, debug: Dict[str, Any], served: List[str], retrieved: List[str]) -> str:
    """The first stage that lost gold passage `pid` (it is not in `served`), from finance_search's stage debug."""
    if "stage_pool" not in debug:  # a route without a search (scope, superlative, numeric)
        return "other"
    if pid not in debug.get("stage_pool", ()) and pid not in debug.get("stage_window", ()):
        return "not_in_pool"
    if pid in debug.get("dropped_not_financial", ()):
        return "scope_filtered"
    if pid not in debug.get("stage_window", ()):
        return "rerank_cut"
    ranked = debug.get("stage_ranked", [])
    if pid not in ranked or pid in retrieved:  # deduplicated, or selected but cut after the search (reader context)
        return "other"
    budget = int(debug.get("k", 0)) - len(debug.get("pinned", ()))
    return "letter_cap" if ranked.index(pid) < budget else "budget_cut"


class FakeLLM:
    """The cloud provider's name (no llama window fit); generation is never part of the route."""
    provider_name = "remote"
    model = "fake"

    def generate(self, prompt, max_new_tokens=None):  # pragma: no cover - guard
        raise RuntimeError("finance_route_debug never generates")

    def generate_stream(self, prompt, max_new_tokens=None):  # pragma: no cover - guard
        raise RuntimeError("finance_route_debug never generates")


def build(cache_path: Optional[Path]):
    """ask_flow with the v3 corpus, FAISS, bge embedder and reranker (cached on disk when a cache path is given)."""
    from config import EMBEDDING_DEVICE, EMBEDDING_MODEL_PRIMARY
    from served_context_recall import CachedEmbedder, CachedReranker, _Cache
    from src.retrieval.context import build_doc_lookup
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.retriever import Retriever
    from src.services import ask_flow
    from src.storage import load_chunks_as_docs
    from src.storage.embeddings import BGEEmbedder
    from src.storage.faiss_store import FaissStore

    docs = load_chunks_as_docs(ROOT / "data/processed/chunks_v3_paragraph.jsonl")
    embedder = BGEEmbedder(model_name=EMBEDDING_MODEL_PRIMARY, device=EMBEDDING_DEVICE)
    store = FaissStore(persist_dir=ROOT / "data/indices/faiss_v3", dim=embedder.dimension)
    if len(store) != len(docs):
        raise RuntimeError(f"faiss_v3 has {len(store)} vectors for {len(docs)} chunks")
    reranker = CrossEncoderReranker(device=EMBEDDING_DEVICE)
    cache = _Cache(cache_path) if cache_path else None
    if cache is not None:
        embedder, reranker = CachedEmbedder(embedder, cache), CachedReranker(reranker, cache)
    retriever = Retriever(vector_store=store, embedder=embedder, docs=docs, reranker=reranker)
    ask_flow._state = {"retriever": retriever, "docs": docs, "docs_by_id": build_doc_lookup(docs), "llm": FakeLLM()}
    return ask_flow, cache


def route_one(ask_flow, case: Dict[str, Any]) -> Dict[str, Any]:
    from src.services.schemas import AskRequest

    route = ask_flow.finance_route(AskRequest(query=case["query"], history=case.get("history") or []))
    plan = route.plan
    served = [h.id for h in route.hits]
    gold = list(case.get("gold_passage_ids") or [])
    got = [g for g in gold if g in set(served)]
    retrieved = [h.id for h in route.retrieved_hits]
    losses = {g: classify_loss(g, route.debug, served, retrieved) for g in gold if g not in set(served)}
    return {
        "loss": losses,
        "qid": case["qid"], "group": case["group"], "type": case["question_type"],
        "scope": plan.scope, "scope_source": plan.scope_source, "kind": plan.kind,
        "year_range": list(plan.year_range) if plan.year_range else None, "year_hard": plan.year_hard,
        "years": list(plan.years), "events": list(plan.events), "sectors": list(plan.sectors),
        "companies": list(plan.companies), "route": ask_flow.finance_answer_mode(route),
        "served_ids": served, "retrieved_ids": retrieved,
        "gold": len(gold), "gold_served": len(got), "gold_missing": [g for g in gold if g not in set(served)],
    }


def share(rows: List[Dict[str, Any]], key: str = "gold_served") -> Optional[float]:
    vals = [r[key] / r["gold"] for r in rows if r["gold"] and r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def group_tag(group: Any) -> str:
    return f"g{group}" if group in GROUP_NAMES else str(group)[:12].ljust(12)


def _pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def _plan_text(r: Dict[str, Any]) -> str:
    years = f"{r['year_range'][0]}-{r['year_range'][1]}" if r["year_range"] else "-"
    parts = [f"{r['scope']}/{r['scope_source']}", r["kind"], years + ("!" if r["year_hard"] else "")]
    for key in ("events", "sectors", "companies"):
        if r[key]:
            parts.append(f"{key[:3]}={','.join(r[key])[:60]}")
    return " ".join(parts)


def loss_table(rows: List[Dict[str, Any]], groups: List[Any]) -> List[str]:
    """Missing gold passages per group and first losing stage."""
    lines = [f"{'group':24s} {'gold':>5s} {'missing':>8s} " + " ".join(f"{s:>14s}" for s in LOSS_STAGES)]
    for label, rs in [("ALL", rows)] + [(group_label(g), [r for r in rows if r["group"] == g]) for g in groups]:
        counts = Counter(stage for r in rs for stage in r["loss"].values())
        lines.append(f"{label:24s} {sum(r['gold'] for r in rs):5d} {sum(counts.values()):8d} "
                     + " ".join(f"{counts.get(s, 0):14d}" for s in LOSS_STAGES))
    return lines


def report(rows: List[Dict[str, Any]], show_ids: bool, compared: bool, set_name: str = "dev",
           loss: bool = False) -> str:
    lines = []
    for r in rows:
        stored = f" stored {r['stored_served']}/{r['gold']}" if compared and r.get("stored_served") is not None else ""
        lines.append(f"{r['qid']:10s} {group_tag(r['group'])} {r['route']:15s} served={len(r['served_ids']):2d} "
                     f"gold {r['gold_served']}/{r['gold']}{stored} | {_plan_text(r)}")
        if show_ids:
            lines.append(f"    served: {' '.join(r['served_ids'])}")
            if r["gold_missing"]:
                lines.append(f"    missing gold: {' '.join(r['gold_missing'])}")
        if loss and r["loss"]:
            lines.append("    lost: " + " ".join(f"{g}={stage}" for g, stage in r["loss"].items()))
    lines.append("")
    head = f"{'group':24s} {'n':>3s} {'gold served':>12s}" + (f" {'stored':>8s}" if compared else "")
    lines.append(head)
    group_ids = list(GROUP_NAMES) if set_name == "dev" else sorted({r["group"] for r in rows})
    groups = [(f"ALL ({set_name})", rows)] + [(group_label(g), [r for r in rows if r["group"] == g]) for g in group_ids]
    for label, rs in groups:
        line = f"{label:24s} {len(rs):3d} {_pct(share(rs)):>12s}"
        if compared:
            line += f" {_pct(share(rs, 'stored_served')):>8s}"
        lines.append(line)
    lines.append("")
    lines.append("routes: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(r["route"] for r in rows).items())))
    by_group = defaultdict(Counter)
    for r in rows:
        by_group[r["group"]][r["route"]] += 1
    for g in sorted(by_group):
        lines.append(f"  {g}: " + ", ".join(f"{k}={v}" for k, v in sorted(by_group[g].items())))
    if loss:
        lines += ["", f"first stage that lost each missing gold passage ({set_name}):", *loss_table(rows, group_ids)]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", choices=("dev", "heldout_v6"), default="dev", dest="set_name",
                    help="question set: the finance_v1 dev half, or heldout_v6 (a + b merged, grouped by question_type)")
    ap.add_argument("--loss", action="store_true", help="classify the stage that lost every missing gold passage")
    ap.add_argument("--cache", type=Path, help="pickle cache of query embeddings and reranker scores")
    ap.add_argument("--compare", type=Path, help="stored run responses.jsonl (served_ids) to compare per question")
    ap.add_argument("--json", type=Path, dest="json_out", help="write the per-question rows here")
    ap.add_argument("--ids", action="store_true", help="print the served ids and the missing gold ids")
    ap.add_argument("--only", nargs="*", default=None, help="qids to run (default: every dev question)")
    args = ap.parse_args(argv)

    cases = load_cases(args.set_name)
    if args.only:
        cases = [c for c in cases if c["qid"] in set(args.only)]
    ask_flow, cache = build(args.cache)
    stored: Dict[str, List[str]] = {}
    if args.compare:
        for line in args.compare.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                stored[rec["qid"]] = rec.get("served_ids") or []
    rows = []
    for case in cases:
        row = route_one(ask_flow, case)
        if args.compare:
            served = stored.get(case["qid"])
            row["stored_served"] = None if served is None else len(set(served) & set(case.get("gold_passage_ids") or []))
        rows.append(row)
    if cache is not None:
        cache.save()
    print(report(rows, args.ids, bool(args.compare), args.set_name, args.loss))
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
