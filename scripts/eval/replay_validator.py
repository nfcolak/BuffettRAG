#!/usr/bin/env python3
"""Replay the serving claim validator over stored answers: baseline vs working tree.

Input: run JSON(s) whose rows carry `raw_answers` (list of model answers) and `context_snapshot`
(served passages in citation order: {id, text, ...}). Every sentence of every raw answer is validated
alone (as round5.evaluate_answer does), once with the baseline validator (git ref, default 487ee15)
and once with the validator in the working tree. No LLM, no retrieval.

  env -u PYTHONPATH .venv/bin/python scripts/eval/replay_validator.py RUN.json [RUN2.json ...] --out replay.md

Prints per file: sentences, old kept, new kept, newly kept, newly dropped. --out is a markdown file with the
per-cause table and every newly kept / newly dropped sentence with its cited passage id and text; a
JSON twin (same stem, .json, or --json-out) carries the same data.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation import claim_validator as new  # noqa: E402
from src.storage import SearchHit  # noqa: E402

VALIDATOR_PATH = "src/evaluation/claim_validator.py"
# fix name -> (module attribute, neutral replacement): switching one off shows which fix a sentence needs
FIXES = {
    "stem": ("_stem", lambda token: token.lower()),
    "evidence_resplit": ("_extra_units", lambda sentence: []),
    "context_year": ("_hit_year", lambda hit: None),
    "table_scale": ("_TABLE_SCALE_RE", re.compile(r"(?!x)x")),
}


def load_baseline(ref: str):
    source = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:{VALIDATOR_PATH}"],
                            capture_output=True, text=True, check=True).stdout
    module = types.ModuleType("claim_validator_baseline")
    label = f"{ref}:{VALIDATOR_PATH}"
    module.__dict__["__file__"] = label
    sys.modules[module.__name__] = module
    exec(compile(source, label, "exec"), module.__dict__)
    return module


def sentences_of(module, answer: str) -> list[str]:
    return [s.strip() for line in answer.split("\n") if line.strip() for s in module._split_original(line)]


def kept(module, sentence: str, hits) -> bool:
    result = module.validate_and_filter_answer(sentence, hits)
    return bool(result.safe_answer.strip()) and not result.blocked_claims


def cited(module, sentence: str, hits) -> list[int]:
    numbers = [int(n) - 1 for marker in module._CITATION_RE.findall(sentence) for n in re.split(r"\s*,\s*", marker) if n.strip().isdigit()]
    return [i for i in dict.fromkeys(numbers) if 0 <= i < len(hits)]


def failed_checks(module, claim: str, passage: str) -> set[str]:
    """Checks the best-covering evidence unit of `passage` fails for `claim` (baseline logic)."""
    normalized = module._normalize(claim)
    tokens = module._TOKEN_RE.findall(normalized)
    content = {t.lower() for t in tokens if t.lower() not in module._STOPWORDS and t.lower() not in module._IGNORE}
    entities = {t.lower() for t in tokens[1:] if t[:1].isupper() and t.lower() not in module._STOPWORDS
                and t.lower() not in module._IGNORE and not t[:1].isdigit()}
    needed = 1.0 if len(content) <= 3 else module._MIN_COVERAGE
    best, best_cov = set(), -1.0
    for unit in module._evidence_units(module._normalize(passage)):
        evidence = {t.lower() for t in module._TOKEN_RE.findall(unit)}
        coverage = len(content & evidence) / len(content) if content else 0.0
        fails = set()
        if coverage < needed:
            fails.add("coverage")
        claim_q, unit_q = module._quantities(claim), module._quantities(unit)
        if not claim_q.issubset(unit_q):
            only_years = all(u == "number" and 1800 <= v <= 2100 for v, u in claim_q - unit_q)
            fails.add("quantity_year" if only_years else "quantity")
        if bool(module._PREDICATE_NEGATION_RE.search(normalized)) != bool(module._PREDICATE_NEGATION_RE.search(module._normalize(unit))):
            fails.add("polarity")
        if not entities.issubset(evidence):
            fails.add("entity")
        if coverage > best_cov or (coverage == best_cov and len(fails) < len(best)):
            best, best_cov = fails, coverage
    return best


def old_cause(old, sentence: str, hits) -> tuple[str, list[str]]:
    """(cause label, failing claims) for a sentence the baseline drops."""
    valid = cited(old, sentence, hits)
    if not valid:
        return "no_valid_citation", []
    body = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", sentence)
    claims = old.split_claims(body)
    if len(claims) > 1 and len(old._quantities(body)) > 1:
        claims.append(old._CITATION_RE.sub("", body).strip().rstrip(".!?"))
    failing = [c for c in claims if not any(old._deterministic_agreement(c, hits[i].text)[0] for i in valid)]
    if not failing:
        return "unknown", []
    labels: set[str] = set()
    for claim in failing:
        per_passage = [failed_checks(old, claim, hits[i].text) for i in valid]
        labels |= min(per_passage, key=lambda f: (len(f), sorted(f)))
    others = [j for j in range(len(hits)) if j not in valid]
    if all(any(old._deterministic_agreement(c, hits[j].text)[0] for j in others) for c in failing):
        return "citation_wrong_passage", failing
    return "+".join(sorted(labels)) or "unknown", failing


def fix_needed(sentence: str, hits) -> list[str]:
    needed = []
    for name, (attribute, neutral) in FIXES.items():
        original = getattr(new, attribute)
        setattr(new, attribute, neutral)
        try:
            if not kept(new, sentence, hits):
                needed.append(name)
        finally:
            setattr(new, attribute, original)
    return needed


def replay_file(path: Path, old) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    report = {"file": str(path), "sentences": 0, "old_kept": 0, "new_kept": 0, "uncited": 0,
              "newly_kept": [], "newly_dropped": [], "still_dropped": []}
    for row in data["rows"]:
        hits = [SearchHit(p["id"], p["text"], {"year": p.get("year"), "merged_ids": p.get("merged_ids")}, 1.0)
                for p in row.get("context_snapshot") or []]
        for number, answer in enumerate(row.get("raw_answers") or []):
            for sentence in sentences_of(new, answer):
                report["sentences"] += 1
                valid = cited(new, sentence, hits)
                was, now = kept(old, sentence, hits), kept(new, sentence, hits)
                report["old_kept"] += was
                report["new_kept"] += now
                if not valid:
                    report["uncited"] += 1
                entry = {"qid": row.get("qid"), "answer_index": number, "sentence": sentence,
                         "cited": [{"index": i + 1, "id": hits[i].id, "text": hits[i].text} for i in valid]}
                if now and not was:
                    cause, failing = old_cause(old, sentence, hits)
                    report["newly_kept"].append({**entry, "old_cause": cause, "failing_claims": failing, "fix_needed": fix_needed(sentence, hits)})
                elif was and not now:
                    report["newly_dropped"].append(entry)
                elif not was:
                    cause, failing = old_cause(old, sentence, hits)
                    report["still_dropped"].append({**entry, "old_cause": cause, "failing_claims": failing})
    return report


def render(reports: list[dict], baseline: str) -> str:
    lines = [f"# Validator replay: baseline {baseline} vs working tree", ""]
    lines += ["| file | sentences | uncited | old kept | new kept | newly kept | newly dropped |", "|---|---|---|---|---|---|---|"]
    totals = Counter()
    for r in reports:
        lines.append(f"| {Path(r['file']).name} | {r['sentences']} | {r['uncited']} | {r['old_kept']} | {r['new_kept']} | "
                     f"{len(r['newly_kept'])} | {len(r['newly_dropped'])} |")
        totals.update(sentences=r["sentences"], uncited=r["uncited"], old_kept=r["old_kept"], new_kept=r["new_kept"],
                      newly_kept=len(r["newly_kept"]), newly_dropped=len(r["newly_dropped"]))
    lines.append(f"| TOTAL | {totals['sentences']} | {totals['uncited']} | {totals['old_kept']} | {totals['new_kept']} | "
                 f"{totals['newly_kept']} | {totals['newly_dropped']} |")
    old_dropped = totals["sentences"] - totals["old_kept"]
    lines += ["", f"Dropped before: {old_dropped}; kept now among those: {totals['newly_kept']}; "
                  f"newly dropped (kept before, dropped now): {totals['newly_dropped']}.", ""]
    causes: dict[str, Counter] = {}
    for r in reports:
        for key in ("newly_kept", "still_dropped"):
            for item in r[key]:
                causes.setdefault(item["old_cause"], Counter())[key] += 1
    lines += ["## Per-cause table (sentences the baseline dropped)", "",
              "Cause = failed check(s) of the best cited evidence unit in the baseline; `citation_wrong_passage` = the cited passage "
              "fails but another served passage passes the baseline check (mis-numbered citation, or text repeated by overlapping chunks); "
              "`no_valid_citation` = no marker in range (refusals, uncited sentences).", "",
              "| baseline cause | dropped before | kept now | still dropped |", "|---|---|---|---|"]
    for cause, c in sorted(causes.items(), key=lambda kv: -sum(kv[1].values())):
        lines.append(f"| {cause} | {c['newly_kept'] + c['still_dropped']} | {c['newly_kept']} | {c['still_dropped']} |")
    fixes = Counter(f for r in reports for item in r["newly_kept"] for f in item["fix_needed"])
    lines += ["", "## Newly kept by fix (a sentence counts for every fix whose removal drops it again)", "",
              "| fix | sentences |", "|---|---|"] + [f"| {name} | {fixes.get(name, 0)} |" for name in FIXES]
    lines += ["", "## Newly kept sentences", ""]
    for r in reports:
        for item in r["newly_kept"]:
            lines += [f"### {Path(r['file']).name} / {item['qid']} (answer {item['answer_index']})", "",
                      f"- sentence: {item['sentence']}",
                      f"- baseline cause: {item['old_cause']}; fix needed: {', '.join(item['fix_needed']) or 'none single'}"]
            for passage in item["cited"]:
                lines += [f"- cited [{passage['index']}] passage id: {passage['id']}", "", "  > " + passage["text"].replace("\n", "\n  > "), ""]
    lines += ["## Newly dropped sentences", ""]
    dropped = [(r, item) for r in reports for item in r["newly_dropped"]]
    if not dropped:
        lines += ["none", ""]
    for r, item in dropped:
        lines += [f"### {Path(r['file']).name} / {item['qid']}", "", f"- sentence: {item['sentence']}"]
        for passage in item["cited"]:
            lines += [f"- cited [{passage['index']}] passage id: {passage['id']}", "", "  > " + passage["text"].replace("\n", "\n  > "), ""]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="run JSON(s) with rows[].raw_answers and rows[].context_snapshot")
    parser.add_argument("--out", type=Path, required=True, help="markdown report path")
    parser.add_argument("--json-out", type=Path, help="JSON twin (default: --out with .json)")
    parser.add_argument("--baseline-ref", default="487ee15", help="git ref holding the baseline validator")
    args = parser.parse_args(argv)
    old = load_baseline(args.baseline_ref)
    reports = [replay_file(path, old) for path in args.runs]
    for r in reports:
        print(f"{Path(r['file']).name}: sentences={r['sentences']} old_kept={r['old_kept']} new_kept={r['new_kept']} "
              f"newly_kept={len(r['newly_kept'])} newly_dropped={len(r['newly_dropped'])}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(reports, args.baseline_ref), encoding="utf-8")
    (args.json_out or args.out.with_suffix(".json")).write_text(json.dumps({"baseline": args.baseline_ref, "reports": reports}, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 1 if any(r["newly_dropped"] for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
