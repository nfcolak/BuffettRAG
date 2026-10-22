"""Score raw answers on the pilot-gate dev slice with cheap proxies: no model, no judge (PG-05).

Per claim: `terms` = every required term appears in the raw answer text; `cite_gold` = the answer cites a gold
passage. Citation position: the share of all [n] citations that point at passage [1] of their prompt, plus a
rank histogram (rank 1 / 2-3 / 4+). Claims are split by where the gold passage sits in the served prompt:
deep (rank >= 4), shallow (rank 1-3, deep-group questions), anchor (anchor questions), absent (not in the prompt).

    score_slice.py GEN.json [--slice-dir DIR]
GEN.json is {prompt id or qid: raw model answer}; a comparison question has one prompt id per period call
("qid#2023", see slice_prompts.jsonl). A key that is a bare qid is scored against the question's served context.
    score_slice.py --run RUN.json     scores the raw_answers of a stored run (qid -> joined raw answers)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("BUFFETTRAG_ARTIFACTS", REPO.parents[2] / "Artifacts" / "BuffettRAG"))
DEFAULT_DIR = ARTIFACTS / "pv8" / "slice"
REFUSAL_LINE = "The retrieved passages do not provide enough evidence to answer this question."
_REFUSAL = re.compile(r"(cannot|can't|can not|do not|don't|doesn't|does not|not).{0,60}(find|contain|support|answer|mention|provide|cover)"
                      r"|insufficient|no information", re.I)
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _norm(text: str) -> str:
    return " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip() + " "


def _citations(text: str) -> List[int]:
    return [int(n) for group in _CITE.findall(text) for n in group.split(",")]


def load_slice(slice_dir: Path) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    doc = json.loads((Path(slice_dir) / "slice.json").read_text(encoding="utf-8"))
    prompts: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    path = Path(slice_dir) / "slice_prompts.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                prompts[record["qid"]].append(record)
    return doc, prompts


def score(gen: Dict[str, str], doc: Dict[str, Any], prompts: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Counters for the slice; `gen` keys are prompt ids or bare qids."""
    claims = collections.Counter()
    cites = collections.Counter()
    missing, refusals_ok, refusals_n = [], 0, 0
    for qid, question in doc["questions"].items():
        group = question["group"]
        calls = prompts.get(qid) or [{"id": qid, "passage_ids": question["served_passage_ids"]}]
        texts = []  # (text, passage ids of its prompt)
        for call in calls:
            if call["id"] in gen:
                texts.append((gen[call["id"]], call["passage_ids"]))
        if not texts and qid in gen:
            texts.append((gen[qid], calls[0]["passage_ids"]))
        if not texts:
            missing.append(qid)
            continue
        joined = " ".join(text for text, _ in texts)
        if group == "refusals":
            refusals_n += 1
            refusals_ok += REFUSAL_LINE in joined or bool(_REFUSAL.search(joined))
            continue
        cited_ids = set()
        for text, ids in texts:
            for number in _citations(text):
                if not 1 <= number <= len(ids):
                    continue
                cited_ids.add(ids[number - 1])
                bucket = "rank1" if number == 1 else "rank2_3" if number <= 3 else "rank4plus"
                for key in (group, "all"):
                    cites[f"{key}_{bucket}"] += 1
                    cites[f"{key}_total"] += 1
        normalized = _norm(joined)
        for claim in question["claims"]:
            rank = claim["rank"]
            kind = "anchor" if group == "anchors" else "absent" if rank is None else "deep" if rank >= 4 else "shallow"
            terms = [_norm(t) for t in claim["required_terms"]]
            hit = bool(terms) and all(t in normalized for t in terms)
            gold = any(g in cited_ids for g in claim["gold_passage_ids"])
            claims[f"{kind}_n"] += 1
            claims[f"{kind}_terms"] += hit
            claims[f"{kind}_cite_gold"] += gold
    out: Dict[str, Any] = dict(claims)
    out.update(cites)
    for key in ("deep", "anchors", "all"):
        total = cites[f"{key}_total"]
        out[f"{key}_cite_1_share"] = round(cites[f"{key}_rank1"] / total, 3) if total else None
    out.update({"refusals_ok": refusals_ok, "refusals_n": refusals_n, "questions_missing": missing})
    return out


def _run_gen(run: Dict[str, Any]) -> Dict[str, str]:
    return {r["qid"]: " ".join(r["raw_answers"]) for r in run["rows"] if r.get("raw_answers")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gen", nargs="?", type=Path, help="{prompt id or qid: raw answer}.json")
    parser.add_argument("--run", type=Path, nargs="+", help="stored run JSON(s): score their raw_answers by qid")
    parser.add_argument("--slice-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    doc, prompts = load_slice(args.slice_dir)
    if args.run:
        gen: Dict[str, str] = {}
        for path in args.run:
            gen.update(_run_gen(json.loads(path.read_text(encoding="utf-8"))))
    elif args.gen:
        gen = json.loads(args.gen.read_text(encoding="utf-8"))
    else:
        parser.error("give GEN.json or --run RUN.json")
    print(json.dumps(score(gen, doc, prompts), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
