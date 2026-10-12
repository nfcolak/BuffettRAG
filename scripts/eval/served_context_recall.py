"""Served-context recall: how often the gold passage reaches the 7B prompt.

Runs the exact /ask path up to prompt assembly (ask_flow._prepare_ask: retrieval, rerank, period handling,
neighbour expansion, caps, fit to the llama window) with a stub LLM that is never called, on corpus v3 + FAISS,
for every answerable case of the spent dev pool, and records the passage ids placed in the prompt.

  run     python scripts/eval/served_context_recall.py run --out DIR --config NAME[:KEY=VAL,KEY=VAL] ...
  report  python scripts/eval/served_context_recall.py report --out DIR --final NAME

Metric (primary): claim-level served recall = share of gold claims whose claim ``gold_passage_ids`` intersect the
served list (the ids of the passages of the prompt: what the /ask response lists as ``hits`` and the strict scorer
sees). Secondary: the same over every chunk id merged into a served passage (neighbour text inside the prompt).

Cross-encoder scores and query embeddings are cached on disk (keyed by query + passage text) so that candidates that
change only fusion, caps or selection do not pay for the same model call twice. The cached cost is added back to the
latency of every case that hits the cache (measured mean cost per pair / per query), so p50 stays comparable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTHON_DOTENV_DISABLED", "1")
os.environ.setdefault("VECTOR_BACKEND", "faiss")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MAIN = Path("/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG")  # git-ignored data lives here
EVAL = ROOT / "data" / "evaluation"
CASE_FILES = {
    "heldout_v3": "heldout_v3/answer_benchmark_heldout_v3.json",
    "heldout_v4": "heldout_v4/answer_benchmark_heldout_v4.json",
    "heldout_v5a": "heldout_v5/answer_benchmark_heldout_v5_a.json",
    "heldout_v5b": "heldout_v5/answer_benchmark_heldout_v5_b.json",
    "heldout_v6a": "heldout_v6/answer_benchmark_heldout_v6_a.json",
    "heldout_v6b": "heldout_v6/answer_benchmark_heldout_v6_b.json",
    "dev_cases": "grounded_dev_v1/confirm/dev_cases.json",
}
HARD_NEGATIVES = "hard_negatives_v4/hard_negatives_v4.json"
STORED_V6 = Path("/Users/necatifurkancolak/AI-Workplace/Artifacts/BuffettRAG/fix-v6/runs")
TOKENIZER_DIR = MAIN / "models/ft/7b_r1/adapter"  # Qwen2.5-7B tokenizer of the served FT model
MIN_ANSWER_TOKENS = 768
BASELINE_ENV = {"LLM_CONTEXT_PASSAGES": "8", "RETRIEVAL_MULTIPART": "0"}  # the pre-v7 serving defaults (run_arm.sh)


# ----------------------------------------------------------------------------- cases
def load_pool() -> List[Dict[str, Any]]:
    pool = []
    for name, rel in CASE_FILES.items():
        for case in json.loads((EVAL / rel).read_text(encoding="utf-8"))["cases"]:
            claims = [list(c.get("gold_passage_ids") or []) for c in case.get("gold_claims") or []]
            claims = [c for c in claims if c]
            if case.get("question_type") == "unanswerable" or not claims:
                continue
            pool.append({"set": name, "qid": case["qid"], "type": case.get("question_type", "answerable"),
                         "query": case["query"], "history": case.get("history") or [], "claims": claims})
    return pool


def load_hv6_unanswerable() -> List[Dict[str, Any]]:
    """heldout_v6 refusal cases: not part of the recall pool (no gold), served only for the determinism check."""
    out = []
    for name in ("heldout_v6a", "heldout_v6b"):
        for case in json.loads((EVAL / CASE_FILES[name]).read_text(encoding="utf-8"))["cases"]:
            if case.get("question_type") == "unanswerable":
                out.append({"qid": case["qid"], "query": case["query"], "history": case.get("history") or []})
    return out


def load_hard_negatives() -> List[Dict[str, Any]]:
    return json.loads((EVAL / HARD_NEGATIVES).read_text(encoding="utf-8"))["cases"]


# ----------------------------------------------------------------------------- caching model wrappers
class _Cache:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = {"embed": {}, "pair": {}, "cost": {"embed_s": 0.0, "embed_n": 0,
                                                                          "pair_s": 0.0, "pair_n": 0}}
        if path.exists():
            self.data = pickle.loads(path.read_bytes())
        self.hits_embed = 0
        self.hits_pairs = 0

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(self.data))
        tmp.replace(self.path)

    def mean_embed_s(self) -> float:
        cost = self.data["cost"]
        return cost["embed_s"] / cost["embed_n"] if cost["embed_n"] else 0.0

    def mean_pair_s(self) -> float:
        cost = self.data["cost"]
        return cost["pair_s"] / cost["pair_n"] if cost["pair_n"] else 0.0


class CachedEmbedder:
    def __init__(self, base, cache: _Cache):
        self.base, self.cache = base, cache

    def embed_query(self, text: str):
        store = self.cache.data["embed"]
        if text in store:
            self.cache.hits_embed += 1
            return store[text]
        start = time.perf_counter()
        vec = self.base.embed_query(text)
        cost = self.cache.data["cost"]
        cost["embed_s"] += time.perf_counter() - start
        cost["embed_n"] += 1
        store[text] = vec
        return vec

    def __getattr__(self, name):
        return getattr(self.base, name)


class CachedReranker:
    """Same ordering and min-max normalisation as CrossEncoderReranker.rerank, raw scores cached per pair."""

    def __init__(self, base, cache: _Cache):
        self.base, self.cache = base, cache

    @staticmethod
    def _key(query: str, text: str) -> str:
        return hashlib.md5((query + "\x00" + text).encode("utf-8")).hexdigest()

    def rerank(self, query, candidates, top_k: int = 8):
        import numpy as np
        from src.storage import SearchHit

        if not candidates:
            return []
        store = self.cache.data["pair"]
        keys = [self._key(query, c.text) for c in candidates]
        missing = [i for i, k in enumerate(keys) if k not in store]
        self.cache.hits_pairs += len(keys) - len(missing)
        if missing:
            start = time.perf_counter()
            raw_new = np.asarray(self.base.model.predict(
                [(query, candidates[i].text) for i in missing], batch_size=self.base.batch_size,
                show_progress_bar=False), dtype=float)
            cost = self.cache.data["cost"]
            cost["pair_s"] += time.perf_counter() - start
            cost["pair_n"] += len(missing)
            for i, value in zip(missing, raw_new):
                store[keys[i]] = float(value)
        raw = np.asarray([store[k] for k in keys], dtype=float)
        spread = float(raw.max() - raw.min())
        normalized = (raw - raw.min()) / spread if spread > 0 else np.ones_like(raw)
        out = []
        for idx in np.argsort(-raw)[:top_k]:
            c = candidates[int(idx)]
            out.append(SearchHit(id=c.id, text=c.text, metadata=c.metadata, score=float(normalized[int(idx)])))
        return out

    def __getattr__(self, name):
        return getattr(self.base, name)


# ----------------------------------------------------------------------------- serving path
class _StubLLM:
    """provider_name 'llama' selects the fitted-window path; generation is never needed (expansion is off)."""
    provider_name = "llama"
    model = "stub"

    def generate(self, prompt, max_new_tokens=None):  # pragma: no cover - guard
        raise RuntimeError("served_context_recall never generates")


def build_environment(cache: _Cache):
    from config import EMBEDDING_DEVICE, EMBEDDING_MODEL_PRIMARY
    from src.retrieval.context import build_doc_lookup
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.retriever import Retriever
    from src.services import ask_flow
    from src.storage import load_chunks_as_docs
    from src.storage.embeddings import BGEEmbedder
    from src.storage.faiss_store import FaissStore

    docs = load_chunks_as_docs(MAIN / "data/processed/chunks_v3_paragraph.jsonl")
    embedder = BGEEmbedder(model_name=EMBEDDING_MODEL_PRIMARY, device=EMBEDDING_DEVICE)
    store = FaissStore(persist_dir=MAIN / "data/indices/faiss_v3", dim=embedder.dimension)
    if len(store) != len(docs):
        raise RuntimeError(f"faiss_v3 has {len(store)} vectors for {len(docs)} chunks")
    reranker = CrossEncoderReranker(device=EMBEDDING_DEVICE)
    retriever = Retriever(vector_store=store, embedder=CachedEmbedder(embedder, cache), docs=docs,
                          reranker=CachedReranker(reranker, cache))
    ask_flow._state = {"retriever": retriever, "docs": docs, "docs_by_id": build_doc_lookup(docs),
                       "llm": _StubLLM()}
    return ask_flow


class _Env:
    """Apply env overrides for one configuration; restore afterwards."""

    def __init__(self, overrides: Dict[str, str]):
        self.overrides, self.saved = overrides, {}

    def __enter__(self):
        for key, value in self.overrides.items():
            self.saved[key] = os.environ.get(key)
            os.environ[key] = str(value)

    def __exit__(self, *_):
        for key, old in self.saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)


def prompt_tokens(tok, prompt: str) -> int:
    """Tokens of the chat-templated request exactly as the llama provider builds it (system + user + generation prompt)."""
    from src.generation.providers.llama_provider import split_prompt

    system, user = split_prompt(prompt)
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def serve_one(ask_flow, cache: _Cache, tok, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    req = ask_flow.AskRequest(query=query, history=history, strategy="hybrid", rerank=True)
    cache.hits_embed = cache.hits_pairs = 0
    start = time.perf_counter()
    _, hits, used_filter, reranked, context_hits, prompt, _ = ask_flow._prepare_ask(req)
    wall = time.perf_counter() - start
    effective = wall + cache.hits_embed * cache.mean_embed_s() + cache.hits_pairs * cache.mean_pair_s()
    return {
        "passage_ids": [h.id for h in context_hits],
        "merged_ids": sorted({i for h in context_hits for i in (h.metadata.get("merged_ids") or [h.id])}),
        "retrieved_ids": [h.id for h in hits],
        "prompt_chars": len(prompt),
        "prompt_tokens": prompt_tokens(tok, prompt) if prompt else 0,
        "wall_ms": round(wall * 1000, 2),
        "latency_ms": round(effective * 1000, 2),
        "used_filter": used_filter,
    }


def _scope_predicate(overrides: Dict[str, str]):
    """(query, history) -> can this configuration change the result of that case?

    Settings that act on a subset of questions (follow-ups, multi-part, temporal comparisons) leave every other
    case byte-identical to the baseline, so those baseline rows are reused instead of re-served.
    """
    from src.generation.decompose import split_question
    from src.retrieval.query_expansion import build_followup_retrieval_query
    from src.retrieval.retriever import detect_temporal_comparison

    scopes = []
    for key in overrides:
        if overrides[key] == BASELINE_ENV.get(key):
            continue
        if key.startswith("RETRIEVAL_FOLLOWUP"):
            scopes.append(lambda q, h: bool(h))
        elif key == "RETRIEVAL_MULTIPART":
            scopes.append(lambda q, h: len(split_question(build_followup_retrieval_query(q, h))) >= 2)
        elif key == "RETRIEVAL_PERIOD_RESERVE":
            scopes.append(lambda q, h: bool(detect_temporal_comparison(build_followup_retrieval_query(q, h))))
        else:
            return lambda q, h: True
    if not scopes:
        return lambda q, h: True
    return lambda q, h: any(scope(q, h) for scope in scopes)


def run_config(name: str, overrides: Dict[str, str], ask_flow, cache: _Cache, tok, pool, hard, out: Path,
               limit: Optional[int]) -> None:
    rows, hn_rows = [], []
    base_path = out / "served_baseline.json"
    base = json.loads(base_path.read_text(encoding="utf-8")) if name != "baseline" and base_path.exists() else None
    base_rows = {r["qid"]: r for r in base["rows"]} if base else {}
    base_hn = {r["qid"]: r for r in base["hard_negatives"]} if base else {}
    in_scope = _scope_predicate(overrides) if base else (lambda q, h: True)
    reused = 0
    with _Env(overrides):
        for i, case in enumerate(pool[:limit]):
            if case["qid"] in base_rows and not in_scope(case["query"], case["history"]):
                rows.append(base_rows[case["qid"]])
                reused += 1
                continue
            row = serve_one(ask_flow, cache, tok, case["query"], case["history"])
            rows.append({"set": case["set"], "qid": case["qid"], "type": case["type"], "claims": case["claims"], **row})
            if (i + 1) % 50 == 0:
                cache.save()
                print(f"[{name}] {i + 1}/{len(pool)}", flush=True)
        for case in hard[:limit]:
            if case["qid"] in base_hn and not in_scope(case["query"], []):
                hn_rows.append(base_hn[case["qid"]])
                continue
            res = serve_one(ask_flow, cache, tok, case["query"], [])
            hn_rows.append({"qid": case["qid"], "relevant_ids": case["relevant_ids"],
                            "hard_negative_ids": case["hard_negative_ids"], "ranked_ids": res["retrieved_ids"],
                            "latency_ms": res["latency_ms"]})
    print(f"[{name}] baseline rows reused for {reused} out-of-scope cases", flush=True)
    cache.save()
    (out / f"served_{name}.json").write_text(json.dumps(
        {"name": name, "overrides": overrides, "rows": rows, "hard_negatives": hn_rows}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"[{name}] done: {len(rows)} cases, {len(hn_rows)} hard negatives", flush=True)


def parse_config(text: str):
    name, _, rest = text.partition(":")
    overrides = dict(BASELINE_ENV)
    for item in filter(None, rest.split(",")):
        key, _, value = item.partition("=")
        overrides[key.strip()] = value.strip()
    return name.strip(), overrides


# ----------------------------------------------------------------------------- scoring / report
def claim_hits(row: Dict[str, Any], field: str = "passage_ids") -> List[bool]:
    served = set(row[field])
    return [bool(served.intersection(claim)) for claim in row["claims"]]


def hn_pass(row: Dict[str, Any]) -> bool:
    ranked, relevant, negatives = row["ranked_ids"], set(row["relevant_ids"]), set(row["hard_negative_ids"])
    best_rel = next((i for i, p in enumerate(ranked) if p in relevant), None)
    best_neg = next((i for i, p in enumerate(ranked) if p in negatives), None)
    return best_rel is not None and (best_neg is None or best_rel < best_neg)


def summarize(result: Dict[str, Any]) -> Dict[str, Any]:
    rows = result["rows"]

    def agg(sub):
        hits = [h for r in sub for h in claim_hits(r)]
        merged = [h for r in sub for h in claim_hits(r, "merged_ids")]
        return {"cases": len(sub), "claims": len(hits), "claims_served": sum(hits),
                "claim_recall": sum(hits) / len(hits) if hits else None,
                "claim_recall_merged": sum(merged) / len(merged) if merged else None,
                "case_any_gold": sum(any(claim_hits(r)) for r in sub) / len(sub) if sub else None}

    sets = sorted({r["set"] for r in rows})
    types = sorted({r["type"] for r in rows})
    lat = [r["latency_ms"] for r in rows]
    toks = [r["prompt_tokens"] for r in rows]
    hn = result.get("hard_negatives") or []
    return {
        "pooled": agg(rows),
        "per_set": {s: agg([r for r in rows if r["set"] == s]) for s in sets},
        "per_type": {t: agg([r for r in rows if r["type"] == t]) for t in types},
        "latency_p50_ms": statistics.median(lat) if lat else None,
        "wall_p50_ms": statistics.median(r["wall_ms"] for r in rows) if rows else None,
        "prompt_tokens_max": max(toks) if toks else None,
        "prompt_tokens_p50": statistics.median(toks) if toks else None,
        "passages_p50": statistics.median(len(r["passage_ids"]) for r in rows) if rows else None,
        "n_ctx_ok": all(8192 - t >= MIN_ANSWER_TOKENS for t in toks),
        "hard_negatives_passed": sum(hn_pass(r) for r in hn),
        "hard_negatives_n": len(hn),
    }


def paired_bootstrap(base_rows, cand_rows, field="passage_ids", draws: int = 10000, seed: int = 0):
    import numpy as np

    base = np.array([h for r in base_rows for h in claim_hits(r, field)], dtype=float)
    cand = np.array([h for r in cand_rows for h in claim_hits(r, field)], dtype=float)
    if base.shape != cand.shape or not len(base):
        raise ValueError("candidate and baseline claim lists differ")
    diff = cand - base
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(draws, len(diff)))
    means = diff[idx].mean(axis=1)
    return {"delta": float(diff.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def determinism_check(base_result: Dict[str, Any], extra_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    stored: Dict[str, List[str]] = {}
    for part in "ab":
        path = STORED_V6 / f"ftr1soft_hv6{part}.json"
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            stored[row["qid"]] = row["passage_ids"]
    checked = exact = same_set = 0
    mismatches = []
    for row in [*base_result["rows"], *(extra_rows or [])]:
        if row["qid"] not in stored:
            continue
        checked += 1
        if row["passage_ids"] == stored[row["qid"]]:
            exact += 1
        elif set(row["passage_ids"]) == set(stored[row["qid"]]):
            same_set += 1
        else:
            mismatches.append({"qid": row["qid"], "now": row["passage_ids"], "stored": stored[row["qid"]]})
    same = exact + same_set
    return {"checked": checked, "exact_order": exact, "same_set_other_order": same_set,
            "same_ids_rate": same / checked if checked else None, "mismatches": mismatches}


def report(out: Path, final: str, base_name: str = "baseline", notes: str = "") -> None:
    results = {p.stem[len("served_"):]: json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(out.glob("served_*.json"))}
    base = results[base_name]
    summary = {n: summarize(r) for n, r in results.items()}
    extra_path = out / "determinism_extra.json"
    determinism = determinism_check(base, json.loads(extra_path.read_text(encoding="utf-8")) if extra_path.exists() else None)
    comps: Dict[str, Any] = {}
    for name, res in results.items():
        if name == base_name:
            continue
        comp = paired_bootstrap(base["rows"], res["rows"])
        comp["merged"] = paired_bootstrap(base["rows"], res["rows"], "merged_ids")
        b, c = summary[base_name], summary[name]
        type_drops = {t: c["per_type"][t]["claim_recall"] - b["per_type"][t]["claim_recall"] for t in b["per_type"]}
        comp["type_deltas"] = type_drops
        comp["gate"] = {
            "ci_lower_gt_0": comp["ci95"][0] > 0,
            "hard_negatives_not_lower": c["hard_negatives_passed"] >= b["hard_negatives_passed"],
            "no_type_drop_gt_0.02": all(d >= -0.02 for d in type_drops.values()),
            "prompt_fits_n_ctx": c["n_ctx_ok"],
        }
        comp["gate"]["all"] = all(comp["gate"].values())
        comps[name] = comp
    payload = {"baseline": base_name, "final": final, "notes": notes, "overrides": {n: r["overrides"] for n, r in results.items()},
               "summary": summary, "vs_baseline": comps, "determinism_hv6": {
                   k: v for k, v in determinism.items()}, "bootstrap": {"draws": 10000, "seed": 0}}
    (out / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "comparison.md").write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {out / 'results.json'} and comparison.md")


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def render_markdown(p: Dict[str, Any]) -> str:
    s, comps, base = p["summary"], p["vs_baseline"], p["baseline"]
    names = list(s)
    lines = ["# Retrieval dev v7: served-context recall", "",
             "Claim-level served recall = share of gold claims whose `gold_passage_ids` hit the ids of the passages "
             "placed in the 7B prompt (the `hits` list of /ask). Pool: answerable cases of heldout_v3..v6 and "
             "dev_cases (all spent). Secondary column: ids merged into a served passage as neighbour text.", ""]
    d = p["determinism_hv6"]
    lines += ["## Determinism (baseline vs stored heldout_v6 `passage_ids`)", "",
              f"{d['same_ids_rate'] and round(100 * d['same_ids_rate'], 1)}% of {d['checked']} cases have the same "
              f"served ids ({d['exact_order']} identical order, {d['same_set_other_order']} same ids other order); "
              f"{len(d['mismatches'])} differ.", ""]
    for m in d["mismatches"][:10]:
        lines.append(f"- {m['qid']}: now {m['now']} vs stored {m['stored']}")
    lines += ["", "## Pooled", "",
              "| config | claims served | claim recall % | delta pp | CI95 pp (paired bootstrap) | merged-ids recall % | "
              "case any-gold % | HN pass | p50 ms | max tokens | gate |", "|" + "---|" * 11]
    for n in names:
        a = s[n]["pooled"]
        c = comps.get(n)
        delta = "" if c is None else f"{100 * c['delta']:+.2f}"
        ci = "" if c is None else f"[{100 * c['ci95'][0]:+.2f}, {100 * c['ci95'][1]:+.2f}]"
        gate = "" if c is None else ("PASS" if c["gate"]["all"] else "fail: " + ",".join(k for k, v in c["gate"].items() if not v and k != "all"))
        lines.append(f"| {n} | {a['claims_served']}/{a['claims']} | {_pct(a['claim_recall'])} | {delta} | {ci} | "
                     f"{_pct(a['claim_recall_merged'])} | {_pct(a['case_any_gold'])} | "
                     f"{s[n]['hard_negatives_passed']}/{s[n]['hard_negatives_n']} | {s[n]['latency_p50_ms']:.0f} | "
                     f"{s[n]['prompt_tokens_max']} | {gate} |")
    lines += ["", "## Per set (claim recall %)", "", "| config | " + " | ".join(s[base]["per_set"]) + " |",
              "|" + "---|" * (len(s[base]["per_set"]) + 1)]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(_pct(s[n]["per_set"][k]["claim_recall"]) for k in s[base]["per_set"]) + " |")
    lines += ["", "## Per question type (claim recall %, claims in brackets)", "",
              "| config | " + " | ".join(f"{k} [{s[base]['per_type'][k]['claims']}]" for k in s[base]["per_type"]) + " |",
              "|" + "---|" * (len(s[base]["per_type"]) + 1)]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(_pct(s[n]["per_type"][k]["claim_recall"]) for k in s[base]["per_type"]) + " |")
    lines += ["", "## Latency and prompt size", "",
              "Latency = wall time of retrieval + rerank + context assembly plus the measured cost of cached model "
              "calls (machine shared with other jobs: indicative only). Tokens: Qwen2.5 tokenizer of the served FT "
              "model on the chat-templated request; n_ctx 8192, 768 tokens reserved for the answer.", "",
              "| config | p50 ms | wall p50 ms | passages p50 | tokens p50 | tokens max | fits n_ctx-768 |", "|" + "---|" * 7]
    for n in names:
        a = s[n]
        lines.append(f"| {n} | {a['latency_p50_ms']:.0f} | {a['wall_p50_ms']:.0f} | {a['passages_p50']} | "
                     f"{a['prompt_tokens_p50']} | {a['prompt_tokens_max']} | {a['n_ctx_ok']} |")
    lines += ["", "## Overrides", ""] + [f"- {n}: {p['overrides'][n]}" for n in names]
    if p.get("notes"):
        lines += ["", "## Caveats and candidates not measured", "", p["notes"].strip()]
    final = p["final"]
    changed = {k: v for k, v in p["overrides"].get(final, {}).items() if BASELINE_ENV.get(k) != v}
    verdict = comps.get(final, {}).get("gate", {}).get("all", False) if final != base else False
    lines += ["", "## Rule and decision", "",
              "A candidate becomes the default only when its pooled claim-level served-recall gain has a paired "
              "per-claim bootstrap (10,000 draws, seed 0) CI95 lower bound > 0, hard_negatives_v4 passes are not "
              "lower, no question type drops by more than 0.02, and the longest prompt leaves >= 768 tokens of the "
              "8192-token window. Each setting is an environment variable read at call time "
              "(`config.retrieval_setting`, `config.llm_context_passages`).", "",
              f"Final configuration: `{final}` {p['overrides'].get(final, {})}.", "",
              "Note: scripts that set `LLM_CONTEXT_PASSAGES=8` explicitly (Artifacts `run_arm.sh`) override the new "
              "default and must be changed to the new value to measure it.", ""]
    if verdict and changed:
        lines.append("DEFAULT: CHANGED " + " ".join(f"{k}={v}" for k, v in sorted(changed.items())))
    else:
        lines.append("DEFAULT: UNCHANGED")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--config", action="append", help="NAME[:KEY=VAL,KEY=VAL]")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--determinism-extra", action="store_true",
                   help="also serve the heldout_v6 unanswerable cases under the baseline env (determinism check)")
    rp = sub.add_parser("report")
    rp.add_argument("--out", type=Path, required=True)
    rp.add_argument("--final", required=True)
    rp.add_argument("--notes", type=Path, default=None, help="text file inserted as 'Caveats' in comparison.md")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "report":
        report(args.out, args.final, notes=args.notes.read_text(encoding="utf-8") if args.notes else "")
        return 0
    cache = _Cache(args.out / "model_cache.pkl")
    ask_flow = build_environment(cache)
    tok = _tokenizer()
    pool, hard = load_pool(), load_hard_negatives()
    print(f"pool: {len(pool)} answerable cases, {sum(len(c['claims']) for c in pool)} claims; "
          f"{len(hard)} hard negatives", flush=True)
    if args.determinism_extra:
        with _Env(BASELINE_ENV):
            extra = [{"qid": c["qid"], "passage_ids": serve_one(ask_flow, cache, tok, c["query"], c["history"])["passage_ids"]}
                     for c in load_hv6_unanswerable()]
        (args.out / "determinism_extra.json").write_text(json.dumps(extra) + "\n", encoding="utf-8")
        print(f"determinism extra: {len(extra)} unanswerable hv6 cases", flush=True)
    for text in args.config or []:
        name, overrides = parse_config(text)
        run_config(name, overrides, ask_flow, cache, tok, pool, hard, args.out, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
