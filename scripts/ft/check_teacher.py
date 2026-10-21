"""FT-r4 finance teacher check (ft-r4-finance group 1): validates the teacher's JSON lines of one batch.

  PY scripts/ft/check_teacher.py --batch batches/batch_000.jsonl out/teacher_batch_000.jsonl

One JSON line per job: {"id": "<job id>", ...the object the job's task asks for...}, or {"id": ..., "skip": "<why>"}
for a job no sentence of its passages can answer. Every line is checked against its job (the batch file is
self-contained, no corpus is read):
  - schema: the keys and types of the job kind's JSON shape;
  - sentence ids: every id exists in the job's passages, under the letter year it is listed with, is a FINANCIAL
    passage's sentence (a mixed_scope job's PERSONAL sentences go to "personal_sentences" only) and is not marked
    "(not quotable)" (the job's `unquotable` numbers: build_ft_finance.Pools.unquotable);
  - sentences: at least 5 words, a real sentence (capital start, end mark, under 70 words, no '* *'), understandable
    alone (a This/It/However... opener needs the sentence before it chosen too), at most 4 per letter and 12 in all;
  - the question: one question of 20-400 characters, no talk of "the passage", no six words copied from the passages,
    at most 70% of its content words in the passages, not routed away from the reader by the finance planner (a
    digest, superlative, plain figure lookup or per-period comparison question is answered without the reader);
  - the kind's rules (`kind_problem`).
It prints OK / SKIP / REJECT per job, with a fix hint for every REJECT, and exits 1 when a job of the batch has no
valid line. `validate(job, obj)` is the same check `build_ft_finance.py assemble` runs before it serves a question.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_ft_v6 as V6  # noqa: E402  (has_copy, bad_sentence, term_present, PASSAGE_TALK, DEPENDENT_START)
import common  # noqa: E402
import ft_v7_checks as C  # noqa: E402

KINDS = ('crisis_period', 'view_over_time', 'single_year', 'figure', 'premise_partial', 'partial', 'refusal', 'mixed_scope')
ANSWER_KINDS = tuple(k for k in KINDS if k != 'refusal')
QUESTION_OVERLAP_MAX = .70          # tasks.md 1.1 (ft_v7 used 0.60 for its single-letter questions)
MAX_PER_LETTER = 4
MAX_SENTENCES = 12
MIN_WORDS = 5
MIN_LETTERS = {'crisis_period': 2, 'view_over_time': 3}
MIN_DECADES = {'view_over_time': 2}
_YEAR_TOKEN = re.compile(r'^(?:19|20)\d{2}$')
_NUMBER = re.compile(r'\d[\d,.]*')
HINTS = {
    'json_invalid': 'write one JSON object per line, in the shape the task asks for, nothing else on the line',
    'unknown_job': 'the "id" must be the id of a job of this batch',
    'bad_question': 'one question of 20-400 characters that ends with ?',
    'question_refers_to_passage': 'never mention the passage, the text, an excerpt or what the letter says',
    'question_copies_passage': 'the question copies six or more consecutive words of a passage: rewrite it in your own words',
    'question_overlaps_passage': 'the question reuses too many words of the passages: ask it the way an investor who has not read them would',
    'comparison_question': 'do not ask to compare two named periods (no compare / versus / differ / better than)',
    'digest_route_question': 'drop "over the years", "always", "never", "most", "overall", "across the letters": '
                             'such a collection-wide question is answered from a digest, not from the passages',
    'superlative_question': 'do not ask which sector or company he praised or criticised most',
    'numeric_lookup_question': 'a bare figure lookup is answered from the numbers table: also ask why, or what he said about the figure',
    'personal_question': 'the question reads as personal (family, health, the meeting): ask about the business or investing side',
    'premise_not_fired': 'name the topic in the words of the task and the wrong period exactly as given, so the system notices the wrong period',
    'bad_answer': '"answer" is a list of {"year": <letter year>, "sentences": ["<sentence id>", ...]}',
    'unknown_sentence_id': 'use only sentence ids printed in this job (passage id, #, sentence number)',
    'sentence_wrong_letter': 'list every sentence id under the letter year of its passage',
    'personal_sentence_in_answer': 'answer only with FINANCIAL passages; personal sentences go to "personal_sentences"',
    'sentence_not_quotable': 'a sentence marked (not quotable) cannot be picked: pick another one',
    'duplicate_sentence': 'use every sentence id once',
    'no_sentences': 'pick at least one answer sentence',
    'too_many_sentences': f'at most {MAX_PER_LETTER} sentences per letter and {MAX_SENTENCES} in all',
    'sentence_too_short': f'pick complete sentences of at least {MIN_WORDS} words',
    'bad_sentence': 'pick normal sentences: capital start, end mark, under 70 words, no row of asterisks, no mailing address',
    'dependent_start': 'a sentence that starts with This, That, It, They, However, But, And... needs the sentence before it chosen too, or pick another',
    'too_few_letters': 'answer from more letters (crisis_period: at least 2, view_over_time: at least 3)',
    'too_few_decades': 'view_over_time: the answer letters must span at least two decades',
    'year_missing': 'single_year: the question must name the letter year literally',
    'no_figure': 'figure: at least one chosen sentence must state a figure (an amount, a percentage, a count)',
    'period_missing': 'premise_partial: the question must name the wrong period exactly as the task gives it',
    'bad_gap': 'partial: "gap" = {"ask": "<noun phrase, 6-100 chars>", "missing": ["<1-2 capitalised names or numbers '
               'that are in your question and in NO Berkshire letter (not a bare year; e.g. a company Berkshire never discussed)>"]}',
    'refusal_has_answer': 'refusal: "answer" must be []',
    'bad_missing': 'refusal: "missing" lists 1-3 capitalised names or numbers that are in your question and in NO Berkshire letter (not a bare year)',
    'bad_present': 'refusal (near): "present" lists 1-3 capitalised names or numbers that are in your question AND in the passages',
    'refusal_cue': 'refusal: no year after 2024 and no stock, share or crypto price in the question',
    'bad_parts': 'mixed_scope: give "financial_part" and "personal_part", each a short phrase',
    'bad_personal_sentences': 'mixed_scope: "personal_sentences" lists 1-4 ids of PERSONAL passage sentences',
}


# ------------------------------------------------------------------ job helpers
def sentence_table(job):
    """sid -> {sid, passage_id, year, n, text, role} for every sentence the job shows the teacher."""
    table = {}
    for passage in job['passages']:
        for n, text in passage['sentences']:
            sid = f"{passage['id']}#{n}"
            table[sid] = {'sid': sid, 'passage_id': passage['id'], 'year': int(passage['year']), 'n': int(n), 'text': text,
                          'role': passage['role'], 'quotable': int(n) not in set(passage.get('unquotable') or ())}
    return table


def passage_texts(job, roles=('financial',)):
    return [' '.join(text for _, text in p['sentences']) for p in job['passages'] if p['role'] in roles]


def has_figure(text):
    """A number that is not a bare letter year: an amount, a percentage, a count."""
    return any(not _YEAR_TOKEN.match(token.strip('.,')) for token in _NUMBER.findall(text))


def chrono_key(item):
    return item['year'], item['passage_id'], item['n']


# ------------------------------------------------------------------ the finance planner's routing (no retrieval)
@lru_cache(maxsize=1)
def planner_store():
    from src.knowledge import KnowledgeStore
    return KnowledgeStore.load()


def route_problem(question, kind, store=None):
    """The planner routes the question away from the reader, or a premise_partial question raises no premise note.

    The planner runs on the rules alone (the live route adds a retrieval scope vote, which can still make a question
    personal: assemble reports that as `route_scope`)."""
    from src.generation import finance_answer
    from src.generation.compare import comparison_periods
    from src.query.plan import COMPARISON, CORPUS_WIDE, NUMERIC, PERSONAL, plan_query
    store = planner_store() if store is None else store
    plan = plan_query(question, store, premise_check=True)
    if plan.scope == PERSONAL:
        return 'personal_question'
    if plan.superlative:
        return 'superlative_question'
    if plan.kind == COMPARISON and comparison_periods(plan.financial_query or question):
        return 'comparison_question'
    if plan.kind == CORPUS_WIDE and finance_answer.digest_topic(plan, store):
        return 'digest_route_question'
    if plan.kind == NUMERIC and finance_answer.plain_figure_lookup(plan) and finance_answer.metric_named(plan, store):
        return 'numeric_lookup_question'
    if kind == 'premise_partial' and not plan.premise_note:
        return 'premise_not_fired'
    return None


# ------------------------------------------------------------------ checks
def question_problem(question, texts, all_texts):
    if not isinstance(question, str) or not 20 <= len(question.strip()) <= 400 or not question.strip().endswith('?'):
        return 'bad_question'
    if V6.PASSAGE_TALK.search(question):
        return 'question_refers_to_passage'
    if V6.has_copy(question, all_texts):
        return 'question_copies_passage'
    if C.question_overlap(question, texts) > QUESTION_OVERLAP_MAX:
        return 'question_overlaps_passage'
    return None


def answer_problem(answer, job, table):
    """-> (reason or None, chosen sentences in chronological order)."""
    if not isinstance(answer, list) or not all(isinstance(a, dict) for a in answer):
        return 'bad_answer', None
    chosen, seen, per_letter = [], set(), Counter()
    letters = {int(y) for y in job['years']}
    for item in answer:
        year, sids = item.get('year'), item.get('sentences')
        if isinstance(year, str) and year.strip().isdigit():
            year = int(year)
        if not isinstance(year, int) or not isinstance(sids, list) or not all(isinstance(s, str) for s in sids):
            return 'bad_answer', None
        if year not in letters:
            return 'sentence_wrong_letter', None
        for sid in sids:
            sid = sid.strip()
            entry = table.get(sid)
            if entry is None:
                return 'unknown_sentence_id', None
            if entry['year'] != year:
                return 'sentence_wrong_letter', None
            if entry['role'] != 'financial':
                return 'personal_sentence_in_answer', None
            if not entry['quotable']:
                return 'sentence_not_quotable', None
            if sid in seen:
                return 'duplicate_sentence', None
            seen.add(sid)
            per_letter[year] += 1
            chosen.append(entry)
    if not chosen:
        return 'no_sentences', None
    if len(chosen) > MAX_SENTENCES or max(per_letter.values()) > MAX_PER_LETTER:
        return 'too_many_sentences', None
    for entry in chosen:
        if len(entry['text'].split()) < MIN_WORDS:
            return 'sentence_too_short', None
        if V6.bad_sentence(entry['text']) or MAILING.search(entry['text']):
            return 'bad_sentence', None
        if C.DEPENDENT_START.match(entry['text']) and f"{entry['passage_id']}#{entry['n'] - 1}" not in seen:
            return 'dependent_start', None
    return None, sorted(chosen, key=chrono_key)


def distinctive_terms(values, question, texts, *, present):
    """1-3 distinctive terms (C.distinctive_missing) that are in the question and in (present) or absent from the texts."""
    if not isinstance(values, list) or not 1 <= len(values) <= 3 or not all(isinstance(v, str) and v.strip() for v in values):
        return None
    terms = [v.strip() for v in values]
    if not all(C.distinctive_missing(t) and V6.term_present(t, [question]) for t in terms):
        return None
    if any(V6.term_present(t, texts) != present for t in terms):
        return None
    if not present:
        # Pilot lesson: a bare year as the missing term is useless; a question naming a year retrieves that letter, so
        # the year is always served and the row drops (refusal_term_served 5/5, partial_gap_term_served 3/3). The
        # teacher sees only the seed passages, so an absent term must be absent from EVERY letter, not just the seeds.
        if any(BARE_YEAR.match(t) for t in terms) or any(corpus_has(t) for t in terms):
            return None
    return terms


# Pilot lesson: '(Write to: Corporate Secretary, Salomon Inc, Seven World Trade Center, New York, NY 10048)' was quoted.
MAILING = re.compile(r"\bWrite to:|P\.\s?O\.\s?Box|\b[A-Z]{2} \d{5}\b")
BARE_YEAR = re.compile(r"^\s*(?:fiscal\s+)?(?:19|20)\d\ds?\s*$", re.I)
_CORPUS = []


def corpus_has(term):
    """True when the term occurs in any letter passage (chunks_v3_paragraph.jsonl); no corpus file -> False."""
    if not _CORPUS:
        path = Path(__file__).resolve().parents[2] / 'data' / 'processed' / 'chunks_v3_paragraph.jsonl'
        if path.exists():
            _CORPUS.extend(json.loads(line)['text'] for line in path.open() if line.strip())
        else:
            _CORPUS.append('')
    return V6.term_present(term, _CORPUS)


def kind_problem(job, obj, chosen, question):
    """The job kind's rules -> (reason or None, extra fields of the validated plan)."""
    kind, texts = job['kind'], passage_texts(job, ('financial', 'personal'))
    letters = sorted({c['year'] for c in chosen or ()})
    if len(letters) < MIN_LETTERS.get(kind, 1):
        return 'too_few_letters', None
    if len({y // 10 for y in letters}) < MIN_DECADES.get(kind, 1):
        return 'too_few_decades', None
    if kind == 'single_year' and not any(str(y) in question for y in job['years']):
        return 'year_missing', None
    if kind == 'figure' and not any(has_figure(c['text']) for c in chosen):
        return 'no_figure', None
    if kind == 'premise_partial' and not any(v.lower() in question.lower() for v in job['period_variants']):
        return 'period_missing', None
    if kind == 'partial':
        gap = obj.get('gap')
        if not isinstance(gap, dict) or not isinstance(gap.get('ask'), str):
            return 'bad_gap', None
        ask = C.norm(gap['ask']).rstrip('.?! ')
        missing = distinctive_terms(gap.get('missing'), question, texts, present=False)
        if not 6 <= len(ask) <= 100 or missing is None or len(missing) > 2:
            return 'bad_gap', None
        return None, {'gap': {'ask': ask, 'missing': missing}}
    if kind == 'mixed_scope':
        parts = [obj.get('financial_part'), obj.get('personal_part')]
        if not all(isinstance(p, str) and 3 <= len(p.strip()) <= 200 for p in parts):
            return 'bad_parts', None
        table = sentence_table(job)
        personal = obj.get('personal_sentences')
        if not isinstance(personal, list) or not 1 <= len(personal) <= 4 or not all(
                isinstance(s, str) and table.get(s.strip(), {}).get('role') == 'personal' for s in personal):
            return 'bad_personal_sentences', None
        return None, {'financial_part': parts[0].strip(), 'personal_part': parts[1].strip(),
                      'personal_sids': [s.strip() for s in personal]}
    return None, {}


def refusal_problem(job, obj, question):
    texts = passage_texts(job, ('financial', 'personal'))
    if obj.get('answer') not in (None, []):
        return 'refusal_has_answer', None
    if C.FUTURE_YEAR.search(question) or C.PRICE_TALK.search(question):
        return 'refusal_cue', None
    missing = distinctive_terms(obj.get('missing'), question, texts, present=False)
    if missing is None:
        return 'bad_missing', None
    present = []
    if job.get('variant') == 'near':
        present = distinctive_terms(obj.get('present'), question, texts, present=True)
        if present is None:
            return 'bad_present', None
    return None, {'missing': missing, 'present': present}


def validate(job, obj, *, store=None, route=True):
    """Teacher JSON of one job -> (reason or None, plan): {question, chosen: [sentence entries, chronological], ...}.

    `route=False` skips the planner check (tests without the knowledge layer)."""
    if not isinstance(obj, dict):
        return 'json_invalid', None
    if job['kind'] not in KINDS:
        raise ValueError(f"unknown job kind {job['kind']!r}; known: {list(KINDS)}")
    question = obj.get('question')
    reason = question_problem(question, passage_texts(job), passage_texts(job, ('financial', 'personal')))
    if reason:
        return reason, None
    question = C.norm(question)
    if job['kind'] == 'refusal':
        reason, extra = refusal_problem(job, obj, question)
        chosen = []
    else:
        reason, chosen = answer_problem(obj.get('answer'), job, sentence_table(job))
        extra = None
        if not reason:
            reason, extra = kind_problem(job, obj, chosen, question)
    if reason:
        return reason, None
    if route:
        reason = route_problem(question, job['kind'], store)
        if reason:
            return reason, None
    return None, {'question': question, 'chosen': chosen, **extra}


# ------------------------------------------------------------------ CLI
def read_lines(path):
    """[(line number, parsed object or None)] of a teacher file; blank lines are skipped."""
    out = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append((number, json.loads(line)))
        except json.JSONDecodeError:
            out.append((number, None))
    return out


def check_file(jobs, lines, *, store=None, route=True):
    """-> (per job id: (status, reason), problems with no job). status OK | SKIP | REJECT | MISSING; a later line wins."""
    by_id = {j['id']: j for j in jobs}
    result, stray = {}, []
    for number, obj in lines:
        if not isinstance(obj, dict) or not isinstance(obj.get('id'), str):
            stray.append((number, 'json_invalid'))
            continue
        job = by_id.get(obj['id'])
        if job is None:
            stray.append((number, 'unknown_job'))
            continue
        if isinstance(obj.get('skip'), str) and obj['skip'].strip():
            result[job['id']] = ('SKIP', obj['skip'].strip())
            continue
        reason, _ = validate(job, obj, store=store, route=route)
        result[job['id']] = ('OK', None) if reason is None else ('REJECT', reason)
    for job in jobs:
        result.setdefault(job['id'], ('MISSING', None))
    return result, stray


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--batch', required=True, help='the batch file the jobs came from (batches/batch_NNN.jsonl)')
    parser.add_argument('teacher', help='the teacher lines for that batch (out/teacher_batch_NNN.jsonl)')
    args = parser.parse_args(argv)
    jobs = common.read_jsonl(Path(args.batch))
    result, stray = check_file(jobs, read_lines(args.teacher))
    for number, reason in stray:
        print(f'line {number}: REJECT {reason}: {HINTS[reason]}')
    for job in jobs:
        status, reason = result[job['id']]
        if status == 'REJECT':
            print(f"{job['id']}: REJECT {reason}: {HINTS.get(reason, 'follow the task')}")
        elif status == 'MISSING':
            print(f"{job['id']}: MISSING: write one line for this job")
        else:
            print(f"{job['id']}: {status}" + (f' ({reason})' if reason else ''))
    counts = Counter(status for status, _ in result.values())
    bad = counts['REJECT'] + counts['MISSING'] + len(stray)
    print(f"{'OK' if not bad else 'FAIL'} jobs={len(jobs)} ok={counts['OK']} skip={counts['SKIP']} reject={counts['REJECT']} "
          f"missing={counts['MISSING']} bad_lines={len(stray)}")
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
