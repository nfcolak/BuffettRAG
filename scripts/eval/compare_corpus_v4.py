"""Corpus v4 (layout) vs v3 (paragraph): retrieval comparison for the fix-v6 switch decision.

Three steps (PREREGISTRATION.md section 1):
  build-map  v3 label ids -> v4 ids   -> v3_to_v4_label_map.json
  retrieve   run one corpus (child process; FAISS_INDEX_NAME picks the index) -> raw JSON
  report     metrics, paired bootstrap, rule lines, verdict -> results.json + comparison.md

Serving retrieval = Retriever.search exactly as ask_flow._do_search calls it for
run_live_benchmark.py --retrieval hybrid (strategy hybrid, top_k 8, fetch_k 30, rerank on,
auto year filter, follow-up query resolution, no query expansion). BM25 = the benchmark's
BM25 mode (production fusion/year handling, BM25 candidate source, no reranker).
Gold-50 relevance is text based (year + keyword), so it needs no id mapping.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

os.environ.setdefault("PYTHON_DOTENV_DISABLED", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MAIN = Path(os.getenv("BUFFETTRAG_MAIN", "/Users/necatifurkancolak/AI-Workplace/Projects/current/BuffettRAG"))
OUT_DIR = ROOT / "data/evaluation/retrieval_compare_v4"
TOP_K, FETCH_K = 8, 30
CORPUS_FILES = {"v3": "data/processed/chunks_v3_paragraph.jsonl", "v4": "data/processed/chunks_v4_layout.jsonl"}
INDEX_NAMES = {"v3": "faiss_v3", "v4": "faiss_v4"}
SETS = ("gold50", "dev", "hard_neg")

# Map thresholds: a v4 chunk "holds the same text" as a v3 chunk when >= 50% of its tokens occur in the
# v3 chunk (containment) and it covers >= 30% of the v3 tokens (coverage); dev evidence chunks need
# only 25% coverage but must contain the claim's required terms.
MAP_JACCARD = 0.5
CONT_MIN, COV_MIN, DEV_COV_MIN = 0.5, 0.3, 0.25

# Decisions made by reading both texts where the rules above give nothing or the wrong chunk.
MANUAL: Dict[Any, Dict[str, Any]] = {
    ("dev", "2013_p0046"): {"v4": ["2013_p0043", "2013_p0044"],
                            "note": "v3 paragraph (BHSI + Peter Eastwood) is split: BHSI in 0043, 'led by Peter Eastwood' in 0044"},
    ("hard_neg", "2010_p0030"): {"v4": ["2010_p0020"],
                                 "note": "float '$66 billion' sentence is in 0020; 0019 holds only the section heading/intro"},
    ("hard_neg", "2016_p0041"): {"v4": ["2016_p0040"],
                                 "note": "float table (1970-2016) is the whole of v4 0040"},
    ("hard_neg", "2001_p0061"): {"v4": ["2001_p0063", "2001_p0064"],
                                 "note": "'$4.25 billion in float / $221 million' in 0063, policy-count disappointment in 0064"},
}


def _p(rel: str) -> Path:
    return ROOT / rel if (ROOT / rel).exists() else MAIN / rel


def _jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


_TOK = re.compile(r"[a-z0-9']+")


def _tok(text: str) -> set:
    return set(_TOK.findall(text.lower()))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def load_cases():
    dev = json.loads(_p("data/evaluation/grounded_dev_v1/confirm/dev_cases.json").read_text(encoding="utf-8"))["cases"]
    dev = [c for c in dev if c.get("question_type") != "unanswerable" and c.get("gold_passage_ids")]
    hn = json.loads(_p("data/evaluation/hard_negatives_v4/hard_negatives_v4.json").read_text(encoding="utf-8"))["cases"]
    return dev, hn


# ----------------------------------------------------------------------------- build-map
def build_map(out: Path) -> None:
    v3 = {r["id"]: r["text"] for r in _jsonl(_p(CORPUS_FILES["v3"]))}
    v4 = {r["id"]: r["text"] for r in _jsonl(_p(CORPUS_FILES["v4"]))}
    jmap = {r["old_id"]: r for r in _jsonl(_p("data/processed/chunk_id_map_v3_to_v4.jsonl"))}
    t3, t4 = {k: _tok(t) for k, t in v3.items()}, {k: _tok(t) for k, t in v4.items()}
    by_year: Dict[str, List[str]] = {}
    for k in v4:
        by_year.setdefault(k[:4], []).append(k)
    dev, hn = load_cases()

    def cont_cov(pid, k):
        inter = len(t3[pid] & t4[k])
        return inter / max(1, len(t4[k])), inter / max(1, len(t3[pid]))

    def overlap(pid):
        scored = []
        for k in by_year[pid[:4]]:
            cont, cov = cont_cov(pid, k)
            if cont >= CONT_MIN and cov >= COV_MIN:
                scored.append((cont * cov, k))
        return [k for _, k in sorted(scored, reverse=True)]

    terms: Dict[str, List[List[str]]] = {}
    for c in dev:
        for pid in c["gold_passage_ids"]:
            lists = [g["required_terms"] for g in c.get("gold_claims", []) if pid in g["gold_passage_ids"]]
            terms.setdefault(pid, []).extend(lists or [g["required_terms"] for g in c.get("gold_claims", [])])
    ids_by_set = {"dev": sorted({p for c in dev for p in c["gold_passage_ids"]}),
                  "hard_neg": sorted({p for c in hn for p in c["relevant_ids"] + c["hard_negative_ids"]})}

    entries, unresolved = [], []
    for label_set, ids in ids_by_set.items():
        for pid in ids:
            j = jmap[pid]
            top1, jac = j["new_ids"][0], j["best_token_jaccard"]
            manual = MANUAL.get((label_set, pid))
            if manual:
                v4ids, method, note = manual["v4"], "manual", manual["note"]
            else:
                if label_set == "dev":
                    holders = []
                    for k in by_year[pid[:4]]:
                        if cont_cov(pid, k)[1] >= DEV_COV_MIN and any(
                                all(_norm(x) in _norm(v4[k]) for x in tl) for tl in terms[pid]):
                            holders.append(k)
                    holders.sort(key=lambda k: -cont_cov(pid, k)[1])
                    v4ids, how = holders, "v4 chunk(s) holding the claim's required terms"
                else:
                    v4ids, how = overlap(pid), "v4 chunk(s) with >=50% of their tokens in the v3 text covering >=30% of it"
                if jac >= MAP_JACCARD and top1 not in v4ids:
                    v4ids = ([top1] if label_set == "hard_neg" else []) + v4ids
                if not v4ids and jac >= MAP_JACCARD:
                    v4ids, how = [top1], "map top-1 (no term-holding chunk found)"
                if jac >= MAP_JACCARD and v4ids == [top1]:
                    method, note = "map", f"id map Jaccard {jac:.2f}; top-1 chunk confirmed by text ({how})"
                else:
                    method = "manual"
                    note = f"id map Jaccard {jac:.2f}, top-1 {top1} {'not ' if top1 not in v4ids else ''}among holders; {how}"
            if not v4ids:
                unresolved.append({"v3_id": pid, "label_set": label_set})
            literal = [top1] if jac >= MAP_JACCARD else v4ids
            entries.append({"v3_id": pid, "label_set": label_set, "v4_ids": v4ids, "method": method,
                            "jaccard": round(jac, 4), "note": note, "literal_map_v4_ids": literal})
    payload = {"schema_version": 1,
               "rule": "map top-1 where Jaccard>=0.5 and confirmed by text; otherwise v4 chunk(s) holding the same text "
                       "(any hit counts); literal_map_v4_ids = sensitivity variant (top-1 where Jaccard>=0.5)",
               "gold50": "text-based relevance (target year + keyword); no ids, no mapping needed",
               "unresolved": unresolved, "entries": entries}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n_map = sum(e["method"] == "map" for e in entries)
    print(f"entries {len(entries)} (map {n_map}, manual {len(entries) - n_map}); unresolved {len(unresolved)} -> {out}")


# ----------------------------------------------------------------------------- retrieve
def retrieve(corpus: str, raw_out: Path) -> None:
    import config
    from src.evaluation.gold_set import get_gold_queries
    from src.pipeline import BuffettRAGPipeline, PipelineConfig
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.query_expansion import build_followup_retrieval_query
    from src.retrieval.retriever import Retriever
    from src.storage import load_chunks_as_docs

    if config.FAISS_DIR.name != INDEX_NAMES[corpus]:
        raise SystemExit(f"FAISS_INDEX_NAME must be {INDEX_NAMES[corpus]} (got {config.FAISS_DIR.name})")
    chunks = _p(CORPUS_FILES[corpus])
    docs = load_chunks_as_docs(chunks)

    class BM25Only(Retriever):  # same as run_live_benchmark.BM25OnlyRetriever
        def __init__(self, d):
            super().__init__(vector_store=None, embedder=None, docs=d, reranker=None)

        def _hybrid_for_filter(self, query, fetch_k, where):
            return self.bm25.search(query, top_k=fetch_k, where=where)

        def _vector_search(self, *_a, **_k):
            raise RuntimeError("vector retrieval disabled")

    pipeline = BuffettRAGPipeline.build(PipelineConfig(chunks_file=chunks, use_llm=False))
    hybrid, bm25_only, plain = pipeline.retriever, BM25Only(docs), BM25Retriever(docs)
    dev, hn = load_cases()
    queries = ([("gold50", g.qid, g.query, []) for g in get_gold_queries()]
               + [("dev", c["qid"], c["query"], c.get("history", [])) for c in dev]
               + [("hard_neg", c["qid"], c["query"], []) for c in hn])

    def run(retriever, query, history, rerank):
        res = retriever.search(query=build_followup_retrieval_query(query, history), strategy="hybrid",
                               top_k=TOP_K, fetch_k=FETCH_K, rerank=rerank, where=None, auto_year_filter=True,
                               retrieval_query=None)
        return {"reranked": res.reranked, "hits": [{"id": h.id, "year": h.metadata.get("year"), "text": h.text}
                                                   for h in res.hits]}

    rows = []
    for i, (label_set, qid, query, history) in enumerate(queries, 1):
        row = {"set": label_set, "qid": qid, "hybrid": run(hybrid, query, history, True),
               "bm25": run(bm25_only, query, history, False)}
        if label_set == "hard_neg":  # exact eval_hard_negatives.py path: plain BM25, top 20, no filters
            row["bm25_plain20"] = [h.id for h in plain.search(query, top_k=20)]
        rows.append(row)
        print(f"[{corpus}] {i}/{len(queries)} {qid}", flush=True)
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw_out.write_text(json.dumps({"corpus": corpus, "faiss_dir": str(config.FAISS_DIR.resolve()),
                                   "chunks": str(chunks), "top_k": TOP_K, "fetch_k": FETCH_K, "rows": rows},
                                  ensure_ascii=False) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- report
def _bootstrap(diff: np.ndarray, n: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def report(raw_dir: Path, out_dir: Path) -> None:
    from src.evaluation.gold_set import get_gold_queries
    from src.evaluation.hard_negatives import score_hard_negative_case
    from src.evaluation.retrieval_metrics import _is_relevant
    from src.storage import SearchHit

    label_map = json.loads((out_dir / "v3_to_v4_label_map.json").read_text(encoding="utf-8"))
    raw = {c: json.loads((raw_dir / f"raw_{c}.json").read_text(encoding="utf-8")) for c in ("v3", "v4")}
    gold = {g.qid: g for g in get_gold_queries()}
    dev, hn = load_cases()
    dev_by, hn_by = {c["qid"]: c for c in dev}, {c["qid"]: c for c in hn}
    unresolved = {(u["label_set"], u["v3_id"]) for u in label_map["unresolved"]}
    lm = {(e["label_set"], e["v3_id"]): e for e in label_map["entries"]}

    dropped = set()
    for c in dev:
        if any(("dev", p) in unresolved for p in c["gold_passage_ids"]):
            dropped.add(("dev", c["qid"]))
    for c in hn:
        if any(("hard_neg", p) in unresolved for p in c["relevant_ids"]):
            dropped.add(("hard_neg", c["qid"]))

    def labels(corpus, label_set, qid, key):
        """(relevant ids, negative ids) in the ids of the given corpus."""
        if label_set == "dev":
            pids = dev_by[qid]["gold_passage_ids"]
            neg = []
            rel = pids if corpus == "v3" else [i for p in pids for i in lm[("dev", p)][key]]
        else:
            c = hn_by[qid]
            if corpus == "v3":
                rel, neg = list(c["relevant_ids"]), list(c["hard_negative_ids"])
            else:
                rel = [i for p in c["relevant_ids"] for i in lm[("hard_neg", p)][key]]
                neg = [i for p in c["hard_negative_ids"] for i in lm[("hard_neg", p)][key]]
        rel_set = set(rel)
        return rel_set, sorted(set(neg) - rel_set)

    def rel_flags(corpus, row, hits, key):
        if row["set"] == "gold50":
            return [_is_relevant(SearchHit(h["id"], h["text"], {"year": h["year"]}, 0.0), gold[row["qid"]]) for h in hits]
        rel, _ = labels(corpus, row["set"], row["qid"], key)
        return [h["id"] in rel for h in hits]

    def evaluate(key):
        out: Dict[str, Dict[str, Any]] = {}
        for corpus in ("v3", "v4"):
            for mode in ("hybrid", "bm25"):
                per_q = {}
                hn_rows = {}
                for row in raw[corpus]["rows"]:
                    if (row["set"], row["qid"]) in dropped:
                        continue
                    hits = row[mode]["hits"]
                    flags = rel_flags(corpus, row, hits, key)
                    rank = next((i for i, f in enumerate(flags, 1) if f), None)
                    per_q[(row["set"], row["qid"])] = {
                        "r1": float(any(flags[:1])), "r5": float(any(flags[:5])), "r8": float(any(flags[:8])),
                        "rr": 1.0 / rank if rank else 0.0, "ids": [h["id"] for h in hits]}
                    if row["set"] == "hard_neg":
                        rel, neg = labels(corpus, "hard_neg", row["qid"], key)
                        case = {"qid": row["qid"], "relevant_ids": sorted(rel), "hard_negative_ids": neg}
                        hn_rows[row["qid"]] = {
                            **score_hard_negative_case(case, [h["id"] for h in hits]),
                            **({"plain20": score_hard_negative_case(case, row["bm25_plain20"])} if mode == "bm25" else {})}
                out[f"{corpus}/{mode}"] = {"per_q": per_q, "hn": hn_rows}
        return out

    def agg(per_q, sets):
        sel = [v for (s, _), v in per_q.items() if s in sets]
        if not sel:
            return None
        return {"n": len(sel), **{m: round(float(np.mean([v[m] for v in sel])), 4) for m in ("r1", "r5", "r8", "rr")}}

    results: Dict[str, Any] = {"schema_version": 1, "top_k": TOP_K, "fetch_k": FETCH_K,
                               "dropped_queries": sorted(f"{s}:{q}" for s, q in dropped),
                               "unresolved_label_ids": label_map["unresolved"], "variants": {}}
    for variant, key in (("primary", "v4_ids"), ("literal_map", "literal_map_v4_ids")):
        ev = evaluate(key)
        var: Dict[str, Any] = {"metrics": {}, "hard_neg_passed": {}, "bootstrap": {}}
        for cfg, data in ev.items():
            var["metrics"][cfg] = {name: agg(data["per_q"], sets) for name, sets in
                                   (("gold50", {"gold50"}), ("dev", {"dev"}), ("hard_neg", {"hard_neg"}),
                                    ("pooled", set(SETS)))}
            passed = sum(r["passed"] for r in data["hn"].values())
            var["hard_neg_passed"][cfg] = {"passed": passed, "n": len(data["hn"]),
                                           **({"passed_plain_bm25_top20": sum(r["plain20"]["passed"] for r in data["hn"].values())}
                                              if cfg.endswith("bm25") else {})}
            var.setdefault("per_query", {})[cfg] = {f"{s}:{q}": {"ids": v["ids"], "r8": v["r8"], "rr": v["rr"]}
                                                    for (s, q), v in data["per_q"].items()}
        for mode in ("hybrid", "bm25"):
            a, b = ev[f"v3/{mode}"]["per_q"], ev[f"v4/{mode}"]["per_q"]
            keys = sorted(a)
            d8 = np.array([b[k]["r8"] - a[k]["r8"] for k in keys])
            drr = np.array([b[k]["rr"] - a[k]["rr"] for k in keys])
            var["bootstrap"][mode] = {
                "n_queries": len(keys), "resamples": 10000, "seed": 0,
                "recall@8_diff": round(float(d8.mean()), 4), "recall@8_diff_ci95": [round(x, 4) for x in _bootstrap(d8)],
                "mrr_diff": round(float(drr.mean()), 4), "mrr_diff_ci95": [round(x, 4) for x in _bootstrap(drr)],
                "queries_v4_better_r8": int((d8 > 0).sum()), "queries_v3_better_r8": int((d8 < 0).sum())}
        m3, m4 = var["metrics"]["v3/hybrid"]["pooled"], var["metrics"]["v4/hybrid"]["pooled"]
        bs = var["bootstrap"]["hybrid"]
        hn3, hn4 = (var["hard_neg_passed"][f"{c}/hybrid"]["passed"] for c in ("v3", "v4"))
        rules = [
            {"rule": "recall@8 v4 - v3 >= +0.03 and paired-bootstrap 95% CI lower bound > 0",
             "met": bool(m4["r8"] - m3["r8"] >= 0.03 - 1e-12 and bs["recall@8_diff_ci95"][0] > 0),
             "detail": f"diff {m4['r8'] - m3['r8']:+.4f}, CI {bs['recall@8_diff_ci95']}"},
            {"rule": "MRR v4 >= v3", "met": bool(m4["rr"] >= m3["rr"]),
             "detail": f"v3 {m3['rr']:.4f}, v4 {m4['rr']:.4f}"},
            {"rule": "hard_negatives_v4 passed v4 >= v3", "met": bool(hn4 >= hn3), "detail": f"v3 {hn3}, v4 {hn4}"}]
        var["rules"] = rules
        var["verdict"] = "SWITCH" if all(r["met"] for r in rules) else "STAY"
        results["variants"][variant] = var
    results["verdict"] = results["variants"]["primary"]["verdict"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "comparison.md").write_text(render(results, label_map), encoding="utf-8")
    print(results["verdict"], json.dumps(results["variants"]["primary"]["metrics"]["v3/hybrid"]["pooled"]),
          json.dumps(results["variants"]["primary"]["metrics"]["v4/hybrid"]["pooled"]))


def render(res: Dict[str, Any], label_map: Dict[str, Any]) -> str:
    L: List[str] = ["# Corpus v4 (layout) vs v3 (paragraph): retrieval comparison", "",
                    "Serving retrieval as in `run_live_benchmark.py --retrieval hybrid` (FAISS bge-base-en-v1.5 + BM25 + RRF, "
                    f"bge-reranker-v2-m3, year filter, top {res['top_k']}, fetch_k {res['fetch_k']}); BM25 = the benchmark's "
                    "BM25 mode. Labels: gold-50 (year + keyword, text based), dev (gold passages of the 71 answerable "
                    "grounded_dev_v1 cases), hard_negatives_v4 (40 cases; relevant ids). Pooled = all three sets. "
                    "MRR is over the returned top 8.", ""]
    ent = label_map["entries"]
    n_map = sum(e["method"] == "map" for e in ent)
    L += [f"Label map: {len(ent)} (set, v3 id) entries, {n_map} method=map, {len(ent) - n_map} method=manual; "
          f"unresolved {len(res['unresolved_label_ids'])}; dropped queries {res['dropped_queries'] or 'none'}.", ""]
    for variant, title in (("primary", "Primary (text-verified label map)"),
                           ("literal_map", "Sensitivity: literal id map (top-1 where Jaccard >= 0.5)")):
        var = res["variants"][variant]
        L += [f"## {title}", "", "| mode | set | n | corpus | recall@1 | recall@5 | recall@8 | MRR |", "|---|---|---|---|---|---|---|---|"]
        for mode in ("hybrid", "bm25"):
            for name in ("gold50", "dev", "hard_neg", "pooled"):
                for corpus in ("v3", "v4"):
                    m = var["metrics"][f"{corpus}/{mode}"][name]
                    L.append(f"| {mode} | {name} | {m['n']} | {corpus} | {m['r1']:.3f} | {m['r5']:.3f} | {m['r8']:.3f} | {m['rr']:.3f} |")
        L += ["", "hard_negatives_v4 passed (existing criterion: best relevant rank < best hard-negative rank, within the returned list):", ""]
        for cfg, v in var["hard_neg_passed"].items():
            extra = f"; plain BM25 top-20 (eval_hard_negatives.py path): {v['passed_plain_bm25_top20']}/{v['n']}" if "passed_plain_bm25_top20" in v else ""
            L.append(f"- {cfg}: {v['passed']}/{v['n']}{extra}")
        L += ["", "Paired per-query bootstrap (10,000 resamples, seed 0), v4 - v3:", ""]
        for mode, b in var["bootstrap"].items():
            L.append(f"- {mode} (n={b['n_queries']}): recall@8 {b['recall@8_diff']:+.4f} CI95 {b['recall@8_diff_ci95']}; "
                     f"MRR {b['mrr_diff']:+.4f} CI95 {b['mrr_diff_ci95']}; queries recall@8 v4 better {b['queries_v4_better_r8']}, v3 better {b['queries_v3_better_r8']}")
        L += ["", f"Rules (PREREGISTRATION.md section 1), hybrid, pooled — {variant}:", ""]
        for r in var["rules"]:
            L.append(f"- {r['rule']}: {'met' if r['met'] else 'not met'} ({r['detail']})")
        L += ["", f"Verdict under {variant}: {var['verdict']}", ""]
    L += [f"VERDICT: {res['verdict']}", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-map")
    r = sub.add_parser("retrieve")
    r.add_argument("--corpus", choices=("v3", "v4"), required=True)
    r.add_argument("--raw-dir", type=Path, required=True)
    rep = sub.add_parser("report")
    rep.add_argument("--raw-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "build-map":
        build_map(OUT_DIR / "v3_to_v4_label_map.json")
    elif args.cmd == "retrieve":
        if os.getenv("FAISS_INDEX_NAME") != INDEX_NAMES[args.corpus]:
            raise SystemExit(f"run with FAISS_INDEX_NAME={INDEX_NAMES[args.corpus]}")
        retrieve(args.corpus, args.raw_dir / f"raw_{args.corpus}.json")
    else:
        report(args.raw_dir, OUT_DIR)


if __name__ == "__main__":
    main()
