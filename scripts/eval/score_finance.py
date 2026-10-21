"""Score a finance_v1 run: python scripts/eval/score_finance.py <runs dir>/<name>/responses.jsonl [--set dev] [--cases]

Loads only the case half named by --set (default dev). --set test refuses without --i-am-the-main-agent-final-run.
Per case (answer text normalised: lowercase, quotes/dashes unified, whitespace collapsed; numbers compared with and without commas):
  answerable / partial / mixed_scope / corpus_wide : term_cov = share of all required terms found (lenient); claim_cov = share of gold claims whose required_terms ALL occur in the answer
  partial      : also states_missing = the answer says something is not covered (refusal-style wording)
  unanswerable : refused = exact refusal sentence or refusal wording, and no claim-like content
  out_of_scope : scope_ok = answer is the scope reply or a refusal (counts the current system's plain refusal as 'refused, not scoped')
  mixed_scope  : leak = any reject.forbidden_terms in the answer
Retrieval: gold_recall_served = share of gold passages among the passages served to the model; same for retrieved.
Answer quality (offline, from the response alone; printed as "quality" lines after the aggregate lines):
  year_cov         : distinct letter years in the answer's citations that are also gold letter years / gold letter years
  dup_rate         : repeated sentences / sentences, after dropping citation markers and normalising case, whitespace, punctuation
                     (sentences = how many sentences it was measured on; fragments under 3 words are not counted)
  refused_with_gold: the answer is exactly the refusal line while at least one gold passage was served
  unused_gold      : gold passages served but not cited (count; gold_served is the count it is measured against)
  partial_ok       : partial cases only - the answer states a gap AND contains at least one gold required term
Prints the aggregate for the half and per question group, then the quality lines; --cases prints the cases.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finance_eval import CASE_FILES, FINAL_RUN_FLAG, load_cases  # noqa: E402

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


MARKER = re.compile(r"\[\d+(?:\s*[,;-]\s*\d+)*\]")
BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d{1,2}[.)])\s+")
ABBREV_END = re.compile(r"(?:\b[a-z]|\b(?:mr|mrs|ms|dr|jr|sr|st|inc|co|corp|ltd|no|vs|etc|approx))\.$", re.I)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"\u201d\u2019')\]])\s+")    # a closing quote or bracket may follow the stop
MIN_SENTENCE_WORDS = 3                                # shorter fragments ("Yes.", "(Inc.)") are not counted as sentences


def strip_markers(text) -> str:
    """Answer text without [n] citation markers and list bullets, whitespace collapsed (case and punctuation kept)."""
    lines = [BULLET.sub("", MARKER.sub("", line)) for line in (text or "").splitlines()]
    return re.sub(r"[ \t]+([,.;:!?])", r"\1", re.sub(r"[ \t]+", " ", "\n".join(lines))).strip()


def sentences(text) -> list[str]:
    """Sentences of an answer (paragraph breaks, and ., !, ? plus an optional closing quote or bracket, then whitespace; abbreviations such as 'Mr.' and 'U.S.' do not split)."""
    out = []
    for para in strip_markers(text).split("\n"):
        cur = ""
        for piece in SENTENCE_END.split(para):
            cur = f"{cur} {piece}".strip() if cur else piece.strip()
            if cur and not ABBREV_END.search(cur):
                out.append(cur)
                cur = ""
        if cur:
            out.append(cur)
    return out


def sentence_key(sentence: str) -> str:
    return " ".join(re.findall(r"[\w$%]+", norm(sentence)))


def sentence_keys(text) -> list[str]:
    return [k for k in (sentence_key(x) for x in sentences(text)) if len(k.split()) >= MIN_SENTENCE_WORDS]


def dup_rate(text) -> float:
    """Share of the answer's sentences that repeat an earlier sentence (0.0 when it has none)."""
    keys = sentence_keys(text)
    return (len(keys) - len(set(keys))) / len(keys) if keys else 0.0


def letter_year(passage_id) -> int | None:
    m = re.match(r"(\d{4})_", str(passage_id or ""))
    return int(m.group(1)) if m else None


def gold_years(case: dict) -> set:
    ids = case.get("gold_passage_ids") or []
    years = {y for y in map(letter_year, ids) if y is not None}
    return years or (set(case.get("years") or []) if ids else set())


def cited_ids_years(citations) -> tuple[set, set]:
    ids, years = set(), set()
    for c in citations or []:
        ids.update(c.get("passage_ids") or [])
        years.update(y for y in c.get("years") or [] if isinstance(y, int))
        years.update(y for y in map(letter_year, c.get("passage_ids") or []) if y is not None)
    return ids, years


def states_gap(ans: str) -> bool:
    return REFUSE in ans or bool(REFUSAL_WORDS.search(ans))


def quality_fields(r: dict, c: dict, ans: str) -> dict:
    """The answer-quality fields of one row (`ans` is the normalised answer)."""
    gold = set(c.get("gold_passage_ids") or [])
    served = gold & set(r.get("served_ids") or [])
    cited_ids, cited_years = cited_ids_years(r.get("citations"))
    gy = gold_years(c)
    terms = [t for cl in c.get("gold_claims") or [] for t in cl.get("required_terms") or []]
    is_refusal = bool(ans) and norm(strip_markers(ans)) == REFUSE
    return {
        "year_cov": (len(cited_years & gy) / len(gy)) if gy else None,
        "gold_years": len(gy),
        "dup_rate": dup_rate(r.get("answer")),
        "sentences": len(sentence_keys(r.get("answer"))),
        "refused_with_gold": is_refusal and bool(served),
        "gold_served": len(served) if gold else None,
        "unused_gold": len(served - cited_ids) if gold else None,
        "partial_ok": (states_gap(ans) and any(has(ans, t) for t in terms)) if c["question_type"] == "partial" else None}


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
            "recall_merged": (len(gold & set(r.get("served_ids_merged") or [])) / len(gold))
                             if gold and r.get("served_ids_merged") is not None else None,
            "recall_retrieved": (len(gold & set(r.get("retrieved_ids") or [])) / len(gold)) if gold else None,
            "refused": REFUSE in ans or bool(REFUSAL_WORDS.search(ans)) or not ans,
            "scoped": SCOPE in ans,
            "leak": any(has(ans, t) for t in ((c.get("reject") or {}).get("forbidden_terms") or [])) if qt == "mixed_scope" else None,
            "error": r.get("error"),
            **quality_fields(r, c, ans)})
    return rows


def agg_line(rs: list[dict], label: str) -> str:
    ans_rs = [x for x in rs if x["claim_cov"] is not None]
    rec = [x["recall_served"] for x in rs if x["recall_served"] is not None]
    merged = [x["recall_merged"] for x in rs if x.get("recall_merged") is not None]
    unans = [x for x in rs if x["type"] == "unanswerable"]
    oos = [x for x in rs if x["type"] == "out_of_scope"]
    part = [x for x in rs if x["type"] == "partial"]
    mixed = [x for x in rs if x["type"] == "mixed_scope"]
    f = lambda v: f"{100 * sum(v) / len(v):.0f}%" if v else "-"  # noqa: E731
    return (f"{label:28s} n={len(rs):2d} | claim coverage {f([x['claim_cov'] for x in ans_rs])} (n={len(ans_rs)}), "
            f"terms {f([x['term_cov'] for x in ans_rs])} "
            f"| gold passages served {f(rec)}" + (f" (merged {f(merged)})" if merged else "") + " "
            f"| unanswerable refused {sum(x['refused'] for x in unans)}/{len(unans)} "
            f"| partial states gap {sum(x['refused'] for x in part)}/{len(part)} "
            f"| personal: scope reply {sum(x['scoped'] for x in oos)}/{len(oos)}, refused {sum(x['refused'] for x in oos)}/{len(oos)}, "
            f"answered {sum(not x['refused'] and not x['scoped'] for x in oos)}/{len(oos)} "
            f"| mixed leaks {sum(bool(x['leak']) for x in mixed)}/{len(mixed)} "
            f"| errors {sum(bool(x['error']) for x in rs)} | sec {sum(x['seconds'] or 0 for x in rs) / max(1, len(rs)):.1f}")


def quality_line(rs: list[dict], label: str) -> str:
    """Aggregate of the answer-quality fields (see the module docstring); the denominators are printed with every number."""
    f = lambda v: f"{100 * sum(v) / len(v):.0f}%" if v else "-"  # noqa: E731
    cov = [x["year_cov"] for x in rs if x["year_cov"] is not None]
    multi = [x["year_cov"] for x in rs if x["year_cov"] is not None and x["gold_years"] >= 2]
    dup = [x["dup_rate"] for x in rs if x["sentences"] >= 2]          # a refusal, a scope reply or one sentence cannot repeat
    with_gold = [x for x in rs if x["gold_served"]]
    served = sum(x["gold_served"] for x in with_gold)
    unused = sum(x["unused_gold"] for x in with_gold)
    part = [x for x in rs if x["partial_ok"] is not None]
    return (f"{label:28s} n={len(rs):2d} | year coverage {f(cov)} (n={len(cov)}), multi-year {f(multi)} (n={len(multi)}) "
            f"| dup sentences {100 * sum(dup) / len(dup) if dup else 0:.1f}% of multi-sentence answers (n={len(dup)}), {sum(d > 0 for d in dup)} repeat "
            f"| refused with gold served {sum(x['refused_with_gold'] for x in with_gold)}/{len(with_gold)} "
            f"| gold served but not cited {unused}/{served} ({f([unused / served]) if served else '-'}) "
            f"| partial gap+content {sum(x['partial_ok'] for x in part)}/{len(part)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("responses")
    ap.add_argument("--set", default="dev", choices=sorted(CASE_FILES), dest="set_name")
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
    print("\nquality (year coverage, repeated sentences, refusals with gold, unused gold, partial gap+content):")
    print(quality_line(rows, f"ALL ({args.set_name})"))
    for g, name in GROUP_NAMES.items():
        print(quality_line([x for x in rows if x["group"] == g], f"group {g} {name}"))
    if args.cases:
        print(f"\n{args.set_name.upper()} cases:")
        for x in rows:
            print(f"  {x['qid']:10s} {x['type']:18s} cov={x['claim_cov'] if x['claim_cov'] is None else round(x['claim_cov'], 2)} "
                  f"served={x['recall_served'] if x['recall_served'] is None else round(x['recall_served'], 2)} "
                  f"refused={x['refused']} leak={x['leak']}")
    Path(args.responses).with_name("scores.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
