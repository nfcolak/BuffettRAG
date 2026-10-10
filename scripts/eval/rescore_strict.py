"""Strict (supported-claims) re-scoring of a run_live_benchmark output, with no model inference.

Input: a run JSON plus its cases file. Rows that carry a `context_snapshot` are scored against exactly that
text. For older rows (llama/local/mlx runs written before snapshots existed) the served context is rebuilt
with the code the run used: retrieved chunks -> src.generation.compare.prepare_answer_context (neighbour
merge via src.retrieval.context.expand_hits_with_neighbors, then src.retrieval.context.fit_context_to_llm
for the llama provider: passage cap and 1800-char anchor truncation). The rebuilt id list is checked against
the row's saved passage_ids. `src.evaluation.supported_claims.supported_claims_met` (or the `supported_claims_met`
of --scorer-module) then scores each case.

  rescore_strict.py RUN.json [--cases CASES.json] [--scorer-module MODULE_OR_FILE.py] [--out claims.json] [--verify]

--verify compares the recomputed strict total with the one stored next to the run and prints
`REPRO OK|FAIL <file> <met>/<required>` (or `REPRO NA` when no stored total exists).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTHON_DOTENV_DISABLED", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MAIN_CHECKOUT = Path(os.environ.get("BUFFETTRAG_MAIN", "/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG"))
ARTIFACTS = Path("/Users/necatifurkancolak/AI-Workplace/Artifacts/BuffettRAG")
DEFAULT_CORPUS = "data/processed/chunks_v3_paragraph.jsonl"
MAX_NEW_TOKENS = 900  # AskRequest default, which run_live_benchmark uses

# Dev runs whose strict total was stored in a differently shaped summary: file name -> (summary file, key).
KNOWN_SUMMARIES = {
    "llama7b_ft_p8_valfix.json": (ARTIFACTS / "ft-r2/results/dev_summary.json", "FT-r1"),
    "llama7b_ftr2_p8_dev.json": (ARTIFACTS / "ft-r2/results/dev_summary.json", "FT-r2"),
}


def _resolve(rel: str, explicit: Optional[Path] = None) -> Path:
    """An explicit path, else `rel` in this worktree, else in the main checkout (git-ignored data lives there)."""
    if explicit is not None:
        return Path(explicit)
    p = Path(rel)
    if p.is_absolute():
        return p
    return ROOT / p if (ROOT / p).exists() else MAIN_CHECKOUT / p


def load_scorer(spec: Optional[str]):
    if not spec:
        from src.evaluation.supported_claims import supported_claims_met
        return supported_claims_met
    if spec.endswith(".py") or os.sep in spec:
        module_spec = importlib.util.spec_from_file_location("rescore_alt_scorer", spec)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(spec)
    return module.supported_claims_met


def load_cases(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data if isinstance(data, list) else data["cases"]
    return {c["qid"]: c for c in cases}


def default_passages(model: Optional[str]) -> int:
    """config._DEFAULT_CONTEXT_PASSAGES: 8 when the model file name says 7b, else 5."""
    return 8 if "7b" in (model or "").lower() else 5


# --------------------------------------------------------------------------- served context

def snapshot_hits(snapshot: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"id": s["id"], "text": s.get("text", ""), "year": s.get("year"),
             "metadata": {"year": s.get("year"), "merged_ids": s.get("merged_ids") or [s["id"]]}} for s in snapshot]


class ContextRebuilder:
    """Served-context rebuild from retrieved chunk ids (no retrieval, no model)."""

    def __init__(self, corpus: Path, provider: str, passages: int, n_ctx: Optional[int] = None):
        from config import LLM_N_CTX, ANSWER_CONTEXT_MAX_CHARS, ANSWER_CONTEXT_NEIGHBORS, LLM_PASSAGE_MAX_CHARS
        from src.retrieval.context import build_doc_lookup
        from src.storage import load_chunks_as_docs

        self.docs_by_id = build_doc_lookup(load_chunks_as_docs(corpus))
        self.llm = SimpleNamespace(provider_name=provider)
        self.kw = dict(neighbors=ANSWER_CONTEXT_NEIGHBORS, max_chars=ANSWER_CONTEXT_MAX_CHARS,
                       max_new_tokens=MAX_NEW_TOKENS, n_ctx=n_ctx or LLM_N_CTX,
                       max_passages=passages, passage_max_chars=LLM_PASSAGE_MAX_CHARS)

    def rebuild(self, row: Dict[str, Any]) -> List[Any]:
        from src.generation.compare import prepare_answer_context
        from src.retrieval.query_expansion import build_followup_retrieval_query
        from src.storage import SearchHit

        hits = []
        for pid in row.get("retrieved_passage_ids") or row.get("passage_ids") or []:
            doc = self.docs_by_id[pid]
            hits.append(SearchHit(pid, doc.text, dict(doc.metadata), 0.0))
        history = list(row.get("history") or [])
        if not hits:
            return []
        return prepare_answer_context(
            self.llm, hits, self.docs_by_id, row["query"], history=history,
            followup=build_followup_retrieval_query(row["query"], history) != row["query"], **self.kw)


def detect_passages(run: Dict[str, Any], corpus: Path, provider: str, n_ctx: Optional[int]) -> int:
    """LLM_CONTEXT_PASSAGES is not recorded in the run header: take the candidate whose rebuilt ids match the
    saved passage_ids on the most rows (ties: the config rule on the model name)."""
    rows = [r for r in run["rows"] if r.get("status", "scored") == "scored" and not r.get("context_snapshot")]
    favoured = default_passages((run.get("answer_engine") or {}).get("model"))
    best = None
    for n in (favoured, *[c for c in (5, 8) if c != favoured]):
        rebuilder = ContextRebuilder(corpus, provider, n, n_ctx)
        bad = sum([h.id for h in rebuilder.rebuild(r)] != r.get("passage_ids") for r in rows)
        if best is None or bad < best[0]:
            best = (bad, n)
        if bad == 0:
            break
    return best[1]


def served_hits(row: Dict[str, Any], rebuilder: Optional[ContextRebuilder]) -> Tuple[List[Any], str, bool]:
    """(hits, source, ids_match). Snapshot when the row has one, else the rebuilt context."""
    snap = row.get("context_snapshot")
    if snap:
        return snapshot_hits(snap), "context_snapshot", [h["id"] for h in snap] == row.get("passage_ids")
    assert rebuilder is not None
    hits = rebuilder.rebuild(row)
    return hits, "rebuilt_context", [h.id for h in hits] == row.get("passage_ids")


# --------------------------------------------------------------------------- scoring

from src.evaluation.supported_claims import _answer_sentences  # noqa: E402

_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _cited_indexes(sentence: str) -> List[int]:
    return sorted({int(n) - 1 for raw in _CITE_RE.findall(sentence) for n in raw.split(",") if n.strip().isdigit()})


def _hit_id(hit: Any) -> str:
    return hit["id"] if isinstance(hit, dict) else hit.id


def _hit_merged(hit: Any) -> List[str]:
    meta = hit.get("metadata") if isinstance(hit, dict) else hit.metadata
    return list((meta or {}).get("merged_ids") or [_hit_id(hit)])


def rescore(run: Dict[str, Any], cases: Dict[str, Dict[str, Any]], scorer, rebuilder: Optional[ContextRebuilder],
            set_name: str, arm: str) -> Dict[str, Any]:
    claims_out: List[Dict[str, Any]] = []
    met = required = id_mismatch = 0
    sources = set()
    per_qid: Dict[str, Dict[str, int]] = {}
    for row in run["rows"]:
        if row.get("status", "scored") != "scored":
            continue
        case = cases[row["qid"]]
        hits, source, ids_ok = served_hits(row, rebuilder)
        sources.add(source)
        id_mismatch += not ids_ok
        result = scorer(case, row.get("answer") or "", row.get("citations") or [], hits)
        if not result:  # unanswerable case: no required claims
            continue
        per_qid[row["qid"]] = {"supported_met": sum(bool(r["met"]) for r in result), "required_claims": len(result)}
        for index, res in enumerate(result):
            support = list(res.get("supporting_sentences") or [])
            # Cited ids: those of the supporting sentence when the claim is met, else of the whole answer.
            texts = support or _answer_sentences(row.get("answer") or "")
            cited, seen = [], set()
            for sentence in texts:
                for i in _cited_indexes(sentence):
                    if 0 <= i < len(hits) and _hit_id(hits[i]) not in seen:
                        seen.add(_hit_id(hits[i]))
                        cited.append(hits[i])
            claims_out.append({
                "set": set_name, "arm": arm, "qid": row["qid"], "claim_index": index, "claim": res["claim"],
                "met": bool(res["met"]), "cited_ids": [_hit_id(h) for h in cited],
                "merged_ids_of_cited": {_hit_id(h): _hit_merged(h) for h in cited},
                "supporting_sentence": support[0] if support else None})
            met += bool(res["met"])
            required += 1
    return {"set": set_name, "arm": arm, "strict_supported_claims_met": met, "strict_required_claims": required,
            "context_sources": sorted(sources), "rows_with_id_mismatch": id_mismatch,
            "per_case": per_qid, "claims": claims_out}


# --------------------------------------------------------------------------- stored totals

def stored_total(path: Path, run: Dict[str, Any], expected: Optional[str]) -> Optional[Dict[str, Any]]:
    """{'met','required','source','per_case'} or None when nothing is stored for this run."""
    if expected:
        m, r = expected.split("/")
        return {"met": int(m), "required": int(r), "source": "--expected", "per_case": None}
    summary = run.get("summary") or {}
    if summary.get("strict_supported_claims_met") is not None:
        return {"met": summary["strict_supported_claims_met"], "required": summary.get("strict_required_claims"),
                "source": "run summary", "per_case": None}
    nli = path.with_name(path.name[:-5] + ".nli.json")
    if nli.exists():
        s = json.loads(nli.read_text(encoding="utf-8")).get("summary") or {}
        if s.get("strict_supported_claims_met") is not None:
            return {"met": s["strict_supported_claims_met"], "required": s.get("strict_required_claims"),
                    "source": nli.name, "per_case": None}
    sibling = path.with_name("summary.json")
    if sibling.exists():
        data = json.loads(sibling.read_text(encoding="utf-8"))
        for num, arm in (data.get("arms") or {}).items():
            if isinstance(arm, dict) and arm.get("name") == path.stem and arm.get("strict_supported_claims_met") is not None:
                per_case = {q: v for q, v in ((data.get("per_case_strict") or {}).get(str(num)) or {}).items()}
                return {"met": arm["strict_supported_claims_met"], "required": arm.get("strict_required_claims"),
                        "source": f"{sibling.name} arms[{num}]", "per_case": per_case or None}
    known = KNOWN_SUMMARIES.get(path.name)
    if known and known[0].exists():
        entry = json.loads(known[0].read_text(encoding="utf-8")).get(known[1]) or {}
        if entry.get("strict") is not None:
            return {"met": entry["strict"], "required": entry.get("required"),
                    "source": f"{known[0].name}[{known[1]}]", "per_case": None}
    return None


def verify(path: Path, run: Dict[str, Any], out: Dict[str, Any], expected: Optional[str]) -> bool:
    met, required = out["strict_supported_claims_met"], out["strict_required_claims"]
    stored = stored_total(path, run, expected)
    if stored is None:
        print(f"REPRO NA {path} {met}/{required} (no stored strict total)")
        return True
    if (stored["met"], stored["required"]) == (met, required):
        print(f"REPRO OK {path} {met}/{required}")
        return True
    print(f"REPRO FAIL {path} recomputed {met}/{required} stored {stored['met']}/{stored['required']} ({stored['source']})")
    if stored["per_case"]:
        for qid, now in out["per_case"].items():
            old = stored["per_case"].get(qid)
            if old and old.get("supported_met") != now["supported_met"]:
                print(f"  differs: {qid} stored {old['supported_met']}/{old.get('required_claims')} "
                      f"recomputed {now['supported_met']}/{now['required_claims']}")
                for c in out["claims"]:
                    if c["qid"] == qid:
                        print(f"    claim {c['claim_index']} met={c['met']}: {c['claim'][:110]}")
    else:
        print("  stored per-claim data unavailable; recomputed met claims:")
        for c in out["claims"]:
            if c["met"]:
                print(f"    {c['qid']}#{c['claim_index']}: {c['claim'][:110]}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="run_live_benchmark output JSON")
    ap.add_argument("--cases", type=Path, default=None, help="cases file (default: the run header's `cases`)")
    ap.add_argument("--corpus", type=Path, default=None, help=f"chunk corpus (default {DEFAULT_CORPUS} from the main checkout)")
    ap.add_argument("--scorer-module", default=None, help="module name or .py file exposing supported_claims_met")
    ap.add_argument("--passages", type=int, default=None, help="LLM_CONTEXT_PASSAGES of the run (default: 5 or 8, whichever reproduces the saved passage_ids)")
    ap.add_argument("--n-ctx", type=int, default=None, help="LLM_N_CTX of the run (default: config)")
    ap.add_argument("--set", dest="set_name", default=None, help="label for the `set` column (default: cases file stem)")
    ap.add_argument("--arm", default=None, help="label for the `arm` column (default: run file stem)")
    ap.add_argument("--out", type=Path, default=None, help="write per-claim rows and totals here")
    ap.add_argument("--verify", action="store_true", help="compare with the stored strict total; print REPRO OK/FAIL/NA")
    ap.add_argument("--expected", default=None, help="MET/REQUIRED to verify against instead of a discovered stored total")
    args = ap.parse_args()

    run = json.loads(args.run.read_text(encoding="utf-8"))
    cases_path = _resolve(run.get("cases", ""), args.cases)
    corpus = _resolve(run.get("corpus") or DEFAULT_CORPUS, args.corpus)
    cases = load_cases(cases_path)
    engine = run.get("answer_engine") or {}
    needs_rebuild = any(not r.get("context_snapshot") for r in run["rows"] if r.get("status", "scored") == "scored")
    rebuilder = None
    if needs_rebuild:
        provider = engine.get("resolved_provider") or engine.get("provider") or "llama"
        passages = args.passages or detect_passages(run, corpus, provider, args.n_ctx)
        rebuilder = ContextRebuilder(corpus, provider, passages, args.n_ctx)
    out = rescore(run, cases, load_scorer(args.scorer_module), rebuilder,
                  args.set_name or cases_path.stem, args.arm or args.run.stem)
    out.update({"run": str(args.run), "cases": str(cases_path), "scorer": args.scorer_module or "src.evaluation.supported_claims"})
    if out["rows_with_id_mismatch"]:
        print(f"warning: {out['rows_with_id_mismatch']} rows whose served ids differ from the saved passage_ids", file=sys.stderr)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.verify:
        return 0 if verify(args.run, run, out, args.expected) else 1
    print(f"{args.run} strict {out['strict_supported_claims_met']}/{out['strict_required_claims']} "
          f"(context: {', '.join(out['context_sources'])}; id mismatches: {out['rows_with_id_mismatch']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
