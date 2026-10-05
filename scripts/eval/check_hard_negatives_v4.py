"""Validate data/evaluation/hard_negatives_v4/hard_negatives_v4.json against corpus v3."""
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SET = ROOT / "data/evaluation/hard_negatives_v4/hard_negatives_v4.json"
CHUNKS = ROOT / "data/processed/chunks_v3_paragraph.jsonl"
PRIOR = [ROOT / "data/evaluation/answer_quality_program" / f"hard_negatives_v{v}.json" for v in (2, 3)]


def main() -> int:
    raw = SET.read_bytes()
    data = json.loads(raw)
    cases = data["cases"]
    errs = []
    ids = {json.loads(line)["id"] for line in CHUNKS.open(encoding="utf-8")}
    if len(cases) != 40:
        errs.append(f"expected 40 cases, got {len(cases)}")
    qids = [c["qid"] for c in cases]
    queries = [c["query"] for c in cases]
    if len(set(qids)) != len(qids):
        errs.append("duplicate qids")
    if len(set(queries)) != len(queries):
        errs.append("duplicate queries")
    prior_q, prior_id = set(), set()
    for p in PRIOR:
        for c in json.load(p.open(encoding="utf-8"))["cases"]:
            prior_q.add(c["query"])
            prior_id.add(c["qid"])
    if prior_q & set(queries) or prior_id & set(qids):
        errs.append("qid/query reused from v2/v3")
    for c in cases:
        if not c["relevant_ids"] or not c["hard_negative_ids"]:
            errs.append(f"{c['qid']}: empty id list")
        for i in c["relevant_ids"] + c["hard_negative_ids"]:
            if i not in ids:
                errs.append(f"{c['qid']}: unknown id {i}")
        if set(c["relevant_ids"]) & set(c["hard_negative_ids"]):
            errs.append(f"{c['qid']}: relevant/negative overlap")
        if not c.get("rationale") or not c.get("type"):
            errs.append(f"{c['qid']}: missing type/rationale")
    types = collections.Counter(c["type"] for c in cases)
    if len(types) != 6 or min(types.values()) < 6:
        errs.append(f"type counts {dict(types)}")
    if errs:
        print("FAIL", *errs, sep="\n")
        return 1
    print(f"OK n={len(cases)} sha256={hashlib.sha256(raw).hexdigest()}")
    print(dict(types), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
