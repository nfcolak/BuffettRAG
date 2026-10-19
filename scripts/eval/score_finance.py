"""Score a finance_v1 run: python scripts/eval/score_finance.py <runs dir>/<name>/responses.jsonl [--set dev] [--cases]

Loads only the case half named by --set (default dev). --set test refuses without --i-am-the-main-agent-final-run.
Per case (answer text normalised: lowercase, quotes/dashes unified, whitespace collapsed; numbers compared with and without commas):
  answerable / partial / mixed_scope / corpus_wide : term_cov = share of all required terms found (lenient); claim_cov = share of gold claims whose required_terms ALL occur in the answer
  partial      : also states_missing = the answer says something is not covered (refusal-style wording)
  unanswerable : refused = exact refusal sentence or refusal wording, and no claim-like content
  out_of_scope : scope_ok = answer is the scope reply or a refusal (counts the current system's plain refusal as 'refused, not scoped')
  mixed_scope  : leak = any reject.forbidden_terms in the answer
Retrieval: gold_recall_served = share of gold passages among the passages served to the model; same for retrieved.
Prints the aggregate for the half and per question group; --cases prints the cases.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finance_eval import FINAL_RUN_FLAG, load_cases  # noqa: E402

REFUSE = "the retrieved passages do not provide enough evidence to answer this question."
SCOPE = "i only answer questions about buffett's financial views and berkshire's financial results."
REFUSAL_WORDS = re.compile(r"(do not|does not|don't|doesn't|not)\s.{0,40}(provide|contain|cover|mention|discuss|include|address)|"
                           r"no (information|evidence|mention)|not enough evidence|insufficient", re.I)
GROUP_NAMES = {1: "crises/macro", 2: "views", 3: "figures", 4: "personal", 5: "whole collection"}


def norm(s):
    s = (s or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def has(ans, term):
    t = norm(term)
    if t in ans:
        return True
    if re.search(r"\d", t):                       # 1,234 vs 1234
        return t.replace(",", "") in ans.replace(",", "")
    return False


def score_rows(responses: list[dict], cases: dict[str, dict], half: str) -> list[dict]:
    rows = []
    for r in responses:
        c = cases[r["qid"]]
        ans = norm(r.get("answer"))
        qt = c["question_type"]
        claims = c.get("gold_claims") or []
        hit = [all(has(ans, t) for t in cl.get("required_terms") or []) for cl in claims]
        gold = set(c.get("gold_passage_ids") or [])
        terms = [t for cl in claims for t in cl.get("required_terms") or []]
        rows.append({
            "qid": r["qid"], "half": half, "group": c["group"], "type": qt, "seconds": r.get("seconds"),
            "claim_cov": (sum(hit) / len(hit)) if hit else None,
            "term_cov": (sum(has(ans, t) for t in terms) / len(terms)) if terms else None,
            "recall_served": (len(gold & set(r.get("served_ids") or [])) / len(gold)) if gold else None,
            "recall_retrieved": (len(gold & set(r.get("retrieved_ids") or [])) / len(gold)) if gold else None,
            "refused": REFUSE in ans or bool(REFUSAL_WORDS.search(ans)) or not ans,
            "scoped": SCOPE in ans,
            "leak": any(has(ans, t) for t in ((c.get("reject") or {}).get("forbidden_terms") or [])) if qt == "mixed_scope" else None,
            "error": r.get("error")})
    return rows


def agg_line(rs: list[dict], label: str) -> str:
    ans_rs = [x for x in rs if x["claim_cov"] is not None]
    rec = [x["recall_served"] for x in rs if x["recall_served"] is not None]
    unans = [x for x in rs if x["type"] == "unanswerable"]
    oos = [x for x in rs if x["type"] == "out_of_scope"]
    part = [x for x in rs if x["type"] == "partial"]
    mixed = [x for x in rs if x["type"] == "mixed_scope"]
    f = lambda v: f"{100 * sum(v) / len(v):.0f}%" if v else "-"  # noqa: E731
    return (f"{label:28s} n={len(rs):2d} | claim coverage {f([x['claim_cov'] for x in ans_rs])} (n={len(ans_rs)}), "
            f"terms {f([x['term_cov'] for x in ans_rs])} "
            f"| gold passages served {f(rec)} | unanswerable refused {sum(x['refused'] for x in unans)}/{len(unans)} "
            f"| partial states gap {sum(x['refused'] for x in part)}/{len(part)} "
            f"| personal: scope reply {sum(x['scoped'] for x in oos)}/{len(oos)}, refused {sum(x['refused'] for x in oos)}/{len(oos)}, "
            f"answered {sum(not x['refused'] and not x['scoped'] for x in oos)}/{len(oos)} "
            f"| mixed leaks {sum(bool(x['leak']) for x in mixed)}/{len(mixed)} "
            f"| errors {sum(bool(x['error']) for x in rs)} | sec {sum(x['seconds'] or 0 for x in rs) / max(1, len(rs)):.1f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("responses")
    ap.add_argument("--set", default="dev", choices=["dev", "test"], dest="set_name")
    ap.add_argument("--cases", action="store_true")
    ap.add_argument(FINAL_RUN_FLAG, action="store_true", dest="final_run")
    args = ap.parse_args(argv)
    cases = {c["qid"]: c for c in load_cases(args.set_name, args.final_run)}
    responses = [json.loads(l) for l in Path(args.responses).read_text().splitlines() if l.strip()]
    rows = score_rows([r for r in responses if r["qid"] in cases], cases, args.set_name)
    print(agg_line(rows, f"ALL ({args.set_name})"))
    print()
    for g, name in GROUP_NAMES.items():
        print(agg_line([x for x in rows if x["group"] == g], f"group {g} {name}"))
    if args.cases:
        print(f"\n{args.set_name.upper()} cases:")
        for x in rows:
            print(f"  {x['qid']:10s} {x['type']:18s} cov={x['claim_cov'] if x['claim_cov'] is None else round(x['claim_cov'], 2)} "
                  f"served={x['recall_served'] if x['recall_served'] is None else round(x['recall_served'], 2)} "
                  f"refused={x['refused']} leak={x['leak']}")
    Path(args.responses).with_name("scores.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
