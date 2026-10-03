"""Seeded, resumable local-MLX question synthesis; never contacts a provider."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import random
import re
import shutil
from common import OUT, SEED, V1_DIR, Teacher, append, atomic_json, inputs, near_frozen, parse_questions, read_jsonl, sha, token_set


def unavailable_templates():
    coins = ['Bitcoin', 'Ethereum', 'Solana', 'Dogecoin', 'XRP', 'Cardano', 'Avalanche', 'Polkadot', 'Litecoin', 'Chainlink']
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October']
    result = [f'What was the exact closing price in US dollars of {coin} on the last trading day of {month} 2025?' for coin in coins for month in months]
    people = ['Satya Nadella', 'Sundar Pichai', 'Tim Cook', 'Jensen Huang', 'Lisa Su', 'Andy Jassy', 'Mary Barra', 'Jamie Dimon', 'Brian Chesky', 'Reed Hastings']
    topics = ['a confidential 2025 board discussion of artificial intelligence', 'a private 2025 conversation about succession planning', 'an undisclosed 2026 strategy for employee compensation', 'a secret 2025 meeting about product launch timing', 'a private 2026 discussion of cybersecurity spending']
    result += [f'What did {person} privately say during {topic}?' for person in people for topic in topics]
    firms = ['Apple', 'Microsoft', 'Nvidia', 'Amazon', 'Alphabet', 'Tesla', 'Meta', 'Netflix', 'Adobe', 'Salesforce']
    metrics = ['audited revenue', 'reported operating profit', 'dividend per share', 'cash flow from operations', 'year-end workforce headcount']
    result += [f'What was {firm}\'s exact {metric} for fiscal year 2025?' for firm in firms for metric in metrics]
    assert len(result) == 200
    return result


def make_plan(docs, identity):
    path = OUT / 'question_plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if plan['identity'] != identity:
            raise RuntimeError('Input identity changed; refusing to mix resumable data.')
        return plan
    rng = random.Random(SEED)
    by_year = defaultdict(list)
    for doc in docs:
        # Natural prose, not tables, legal appendices or salutations alone.
        if len(doc.text) >= 200 and sum(c.isalpha() for c in doc.text) / len(doc.text) > .55:
            by_year[int(doc.metadata['year'])].append(doc.id)
    years = sorted(by_year)
    for year in years:
        rng.shuffle(by_year[year])
    rng.shuffle(years)
    sampled = []
    for round_index in range(19):
        for year in years:
            if len(sampled) < 900:
                sampled.append(by_year[year][round_index])
    assert len(sampled) == 900 and len({ident[:4] for ident in sampled}) == 48
    templates = unavailable_templates()
    rng.shuffle(templates)
    jobs = []
    for index, ident in enumerate(sampled):
        jobs.append({'id': f'passage:{ident}', 'source_id': ident, 'type': 'answerable'})
        if index % 4 == 3 and templates:
            number = 200 - len(templates)
            jobs.append({'id': f'unanswerable:{number:03d}', 'template': templates.pop(), 'type': 'unanswerable'})
    plan = {'identity': identity, 'jobs': jobs}
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(path, plan)
    return plan


def generate_pending(teacher, limit=None):
    docs, excluded, frozen, valid_years, identity = inputs()
    lookup = {doc.id: doc for doc in docs}
    plan = make_plan(docs, identity)
    existing = read_jsonl(OUT / 'questions.jsonl')
    known = {tuple(sorted(token_set(row['question']))) for row in existing}
    done = {row['id'] for row in read_jsonl(OUT / 'question_attempts.jsonl')}
    generated = 0
    for job in plan['jobs']:
        if job['id'] in done:
            continue
        if limit is not None and generated >= limit:
            break
        if job['type'] == 'answerable':
            doc = lookup[job['source_id']]
            prompt = (f'Write exactly TWO distinct, natural user questions directly answerable from this passage of Berkshire\'s {doc.metadata["year"]} letter. '
                      'Each question must ask for one specific fact or idea stated explicitly in the passage. '
                      'Include concrete distinctive company names, topic words and the source year so lexical retrieval can find the passage. '
                      'Do not mention passage IDs or ask for calculations, opinions not stated, or surrounding text. '
                      'Avoid vague pronouns. Return ONLY JSON: {"questions":["First question?","Second question?"]}.\nPASSAGE:\n' + doc.text[:1800])
        else:
            prompt = ('Paraphrase this question as one natural user question. Preserve every name, date, private/confidential qualifier and requested fact. '
                      'Do not answer it. Return ONLY JSON: {"questions":["Question?"]}.\n' + job['template'])
        raw, metrics = teacher.generate([{'role': 'system', 'content': 'You write factual questions and return valid JSON only.'}, {'role': 'user', 'content': prompt}], job['id'], max_tokens=180)
        parsed = parse_questions(raw)
        kept = []
        for index, question in enumerate(parsed[:2 if job['type'] == 'answerable' else 1]):
            if job['type'] == 'answerable':
                year = str(lookup[job['source_id']].metadata['year'])
                if year not in question:
                    question = f'In the {year} letter, ' + question[:1].lower() + question[1:]
            fingerprint = tuple(sorted(token_set(question)))
            if fingerprint in known or near_frozen(question, frozen):
                continue
            row = {'id': f'{job["id"]}:q{index}', 'job_id': job['id'], 'question': question, 'type': job['type'], 'source_id': job.get('source_id'), 'source_year': int(lookup[job['source_id']].metadata['year']) if job.get('source_id') else None}
            row['split'] = ('valid' if row['source_year'] in valid_years else 'train') if row['source_year'] else ('valid' if int(sha(f'{SEED}:{row["id"]}'.encode())[:8], 16) % 10 == 0 else 'train')
            append(OUT / 'questions.jsonl', row)
            kept.append(row)
            known.add(fingerprint)
        append(OUT / 'question_attempts.jsonl', {'id': job['id'], 'kept': len(kept), 'raw': raw, **metrics})
        generated += 1
        print(f'question_job={job["id"]} kept={len(kept)} tok/s={metrics["generation_tps"]:.2f}', flush=True)
        yield kept


# ---------------------------------------------------------------- style v2 (round 2)
# Additive: the v1 questions/refusals are carried into data/ft_v2 untouched; new question kinds are tagged style=v2.
TOPIC_WORDS = ['float', 'goodwill', 'inflation', 'derivatives', 'dividends', 'repurchases', 'reinsurance', 'pension', 'junk bonds',
               'stock options', 'acquisitions', 'depreciation', 'leverage', 'insurance float', 'retained earnings', 'capital allocation']
NOT_TOPICS = {'january', 'february', 'march', 'april', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'berkshire',
              'buffett', 'charlie', 'munger', 'class', 'table', 'page', 'letter', 'year', 'hathaway', 'warren', 'following', 'there', 'these', 'those'}
CYCLE = ['new'] * 7 + ['passage'] * 4 + ['unanswerable'] * 2   # job mix per 13 slots; v1 single-fact questions are not re-answered


def init_v2():
    """Seed data/ft_v2 once: frozen snapshot, v1 questions, v1 refusal examples (kept for refusal share), and the v2 job plan."""
    if (OUT / 'question_plan.json').exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(V1_DIR / 'frozen_heldout_v1.json', OUT / 'frozen_heldout_v1.json')
    v1_questions = read_jsonl(V1_DIR / 'questions.jsonl')
    for row in v1_questions:
        append(OUT / 'questions.jsonl', {**row, 'style': 'v1q'})
    for row in read_jsonl(V1_DIR / 'examples.jsonl'):
        if row['type'] != 'answerable':   # v1 refusals only; single-sentence v1 positives are NOT mixed in
            append(OUT / 'examples.jsonl', {**row, 'style': 'v1_refusal'})
    docs, excluded, frozen, valid_years, identity = inputs()
    v1_plan = json.loads((V1_DIR / 'question_plan.json').read_text())
    assert v1_plan['identity'] == identity
    v1_done = {row['id'] for row in read_jsonl(V1_DIR / 'question_attempts.jsonl')}
    rng = random.Random(SEED + 2)
    prose = [d for d in docs if len(d.text) >= 200 and sum(c.isalpha() for c in d.text) / len(d.text) > .55]
    v1_sources = {job['source_id'] for job in v1_plan['jobs'] if job.get('source_id')}
    # temporal comparison jobs: two passages, different years, sharing a topic term
    lower = defaultdict(int)
    for d in prose:
        for word in re.findall(r'\b[a-z]{4,}\b', d.text):
            lower[word] += 1
    docs_for = defaultdict(list)
    for d in prose:
        text = d.text
        terms = {w for w in re.findall(r'\b[A-Z][A-Za-z]{3,}(?:-[A-Z][a-z]+)?\b', text) if lower[w.lower()] < 3 and w.lower() not in NOT_TOPICS}
        terms |= {w for w in TOPIC_WORDS if w in text.lower()}
        for term in terms:
            docs_for[term].append(d)
    usable = []
    for term, ds in docs_for.items():
        years = {int(d.metadata['year']) for d in ds}
        if 12 <= len(ds) <= (600 if term in TOPIC_WORDS else 150) and len(years) >= 6:
            usable.append(term)
    usable.sort()
    rng.shuffle(usable)
    # curated topic words (float, goodwill, ...) alternate with distinctive proper nouns for better comparisons
    curated = [t for t in usable if t in TOPIC_WORDS]
    nouns = [t for t in usable if t not in TOPIC_WORDS]
    usable = [t for pair in zip(curated * 10, nouns) for t in pair] + nouns
    temporal, uses = [], defaultdict(int)
    def pick_pair(term, want_valid):
        by_year = defaultdict(list)
        for d in docs_for[term]:
            year = int(d.metadata['year'])
            if (year in valid_years) == want_valid:
                by_year[year].append(d)
        keys = sorted(by_year)
        pairs = [(a, b) for a in keys for b in keys if b - a >= 3]
        if not pairs:
            return None
        a, b = rng.choice(pairs)
        best = lambda ds: max(ds, key=lambda d: (d.text.lower().count(term.lower()), rng.random()))
        return best(by_year[a]), best(by_year[b])
    index = 0
    while len(temporal) < 240 and index < 20 * len(usable):
        term = usable[index % len(usable)]
        index += 1
        if uses[term] >= (8 if term in TOPIC_WORDS else 2):
            continue
        pair = pick_pair(term, want_valid=len(temporal) % 10 == 9)
        if pair is None:
            continue
        uses[term] += 1
        temporal.append({'id': f'temporal:{len(temporal):03d}', 'kind': 'temporal', 'type': 'answerable', 'topic': term, 'source_id': pair[0].id, 'source_ids': [pair[0].id, pair[1].id]})
    # multi-part and follow-up jobs from passages the v1 plan never sampled
    by_year = defaultdict(list)
    for d in prose:
        if d.id not in v1_sources:
            by_year[int(d.metadata['year'])].append(d.id)
    years = sorted(by_year)
    for year in years:
        rng.shuffle(by_year[year])
    single = []
    for round_index in range(8):
        for year in years:
            if round_index < len(by_year[year]):
                single.append(by_year[year][round_index])
    rng.shuffle(single)
    new = list(temporal)
    for number in range(270):
        kind = 'multipart' if number % 27 < 15 else 'followup'
        new.append({'id': f'{kind}:{number:03d}', 'kind': kind, 'type': 'answerable', 'source_id': single[number]})
    assert sum(job['kind'] == 'multipart' for job in new) == 150 and sum(job['kind'] == 'followup' for job in new) == 120
    rng.shuffle(new)
    lists = {
        'new': new,
        'passage': [{**job, 'kind': 'passage'} for job in v1_plan['jobs'] if job['type'] == 'answerable' and job['id'] not in v1_done and job['source_id'] not in {d['source_id'] for d in new if d.get('source_id')}],
        'unanswerable': [{**job, 'kind': 'unanswerable'} for job in v1_plan['jobs'] if job['type'] == 'unanswerable' and job['id'] not in v1_done],
    }
    jobs, cursor = [], {key: 0 for key in lists}
    while any(cursor[key] < len(lists[key]) for key in lists):
        for key in CYCLE:
            if cursor[key] < len(lists[key]):
                jobs.append(lists[key][cursor[key]])
                cursor[key] += 1
    atomic_json(OUT / 'question_plan.json', {'identity': identity, 'style': 'v2', 'counts': {key: len(value) for key, value in lists.items()}, 'jobs': jobs})
    print(f'v2 plan: {len(jobs)} jobs, counts={ {key: len(value) for key, value in lists.items()} }', flush=True)


def _row(job, index, question, lookup, valid_years, **extra):
    year = int(lookup[job['source_id']].metadata['year']) if job.get('source_id') else None
    row = {'id': f'{job["id"]}:q{index}', 'job_id': job['id'], 'question': question, 'type': job['type'], 'source_id': job.get('source_id'), 'source_year': year, **extra}
    if job.get('source_ids'):
        row['source_ids'] = job['source_ids']
    row['split'] = ('valid' if year in valid_years else 'train') if year else ('valid' if int(sha(f'{SEED}:{row["id"]}'.encode())[:8], 16) % 10 == 0 else 'train')
    return row


def _json_object(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == '{':
            try:
                obj, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return {}


def _clean(value, limit=350):
    value = re.sub(r'\s+', ' ', value).strip() if isinstance(value, str) else ''
    return value if 15 <= len(value) <= limit else ''


def generate_pending_v2(teacher, limit=None):
    docs, excluded, frozen, valid_years, identity = inputs()
    lookup = {doc.id: doc for doc in docs}
    plan = json.loads((OUT / 'question_plan.json').read_text())
    assert plan['identity'] == identity, 'Input identity changed; refusing to mix resumable data.'
    existing = read_jsonl(OUT / 'questions.jsonl')
    known = {tuple(sorted(token_set(row['question'] + ' ' + ' '.join(t['content'] for t in row.get('history', []))))) for row in existing}
    done = {row['id'] for row in read_jsonl(OUT / 'question_attempts.jsonl')}
    generated = 0
    for job in plan['jobs']:
        if limit is not None and generated >= limit:
            break
        kind = job['kind']
        if job['id'] in done:
            continue
        system = 'You write factual questions and return valid JSON only.'
        if kind == 'passage' or kind == 'unanswerable':
            doc = lookup.get(job.get('source_id'))
            if kind == 'passage':
                prompt = ('Write exactly TWO distinct, natural user questions about this passage of Berkshire\'s ' + str(doc.metadata['year']) + ' letter. '
                          'Each question must ask for an explanation, description or overview of ONE idea the passage develops over several sentences (for example how/why/what are the main points of ...), '
                          'so a good answer needs two or three separate statements from the passage, not a single number or name. '
                          'Include distinctive company names or topic words and the year ' + str(doc.metadata['year']) + ' so lexical retrieval can find the passage. '
                          'Do not mention passage IDs or ask for calculations, opinions not stated, or surrounding text. '
                          'Avoid vague pronouns. Return ONLY JSON: {"questions":["First question?","Second question?"]}.\nPASSAGE:\n' + doc.text[:1800])
            else:
                prompt = ('Paraphrase this question as one natural user question. Preserve every name, date, private/confidential qualifier and requested fact. '
                          'Do not answer it. Return ONLY JSON: {"questions":["Question?"]}.\n' + job['template'])
            tokens = 180
        elif kind == 'temporal':
            a, b = (lookup[i] for i in job['source_ids'])
            ya, yb = a.metadata['year'], b.metadata['year']
            topic = job['topic']
            prompt = ('Write ONE natural user question that compares two letters: how Berkshire/Buffett\'s treatment of a specific subject differs between the ' + str(ya) + ' letter and the ' + str(yb) + ' letter. '
                      'Each passage below discusses something related to "' + topic + '". Name the concrete subject, company, instrument or metric that BOTH passages touch (use words that literally appear in the passages), '
                      'and put both years (' + str(ya) + ' and ' + str(yb) + ') in the question, e.g. "Compare how the ' + str(ya) + ' and ' + str(yb) + ' letters describe <subject>: what <fact> in ' + str(ya) + ' versus <fact> in ' + str(yb) + '?". '
                      'Each year\'s side must be answerable from that year\'s passage; no calculations. If the two passages do not discuss a common subject, return {"questions":[]}. '
                      'Return ONLY JSON: {"questions":["Question?"]}.\nPASSAGE FROM ' + str(ya) + ':\n' + a.text[:1200] + '\n\nPASSAGE FROM ' + str(yb) + ':\n' + b.text[:1200])
            tokens = 200
        elif kind == 'multipart':
            doc = lookup[job['source_id']]
            year = doc.metadata['year']
            prompt = (f'Write ONE natural user question about this passage of Berkshire\'s {year} letter that has TWO parts joined by "and": '
                      'the first part asks what/which/how much/who (a fact stated in the passage) and the second asks why or how (an explanation stated in the passage). '
                      f'Include distinctive names or topic words and the year {year}. Both parts must be answerable from the passage; no calculations. '
                      'Return ONLY JSON: {"questions":["Question?"]}.\nPASSAGE:\n' + doc.text[:1800])
            tokens = 200
        else:  # followup
            doc = lookup[job['source_id']]
            year = doc.metadata['year']
            prompt = (f'From this passage of Berkshire\'s {year} letter, write a short conversation opener topic and a follow-up question. '
                      '"topic" is a short noun phrase naming the specific subject of the passage (for example "the automobile dealership acquisition"). '
                      '"followup" is a natural question about one or two specific facts in the passage that refers to the topic only with a pronoun or "that" '
                      '(it, that, they, this) and never names it, so it cannot be understood without the topic. It must be answerable from the passage. '
                      'Return ONLY JSON: {"topic":"...","followup":"...?"}.\nPASSAGE:\n' + doc.text[:1800])
            tokens = 200
        raw, metrics = teacher.generate([{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}], job['id'], max_tokens=tokens)
        kept = []
        candidates = []   # (question, extra)
        if kind == 'followup':
            obj = _json_object(raw)
            topic, followup = _clean(obj.get('topic'), 120), _clean(obj.get('followup'))
            if topic and followup.endswith('?'):
                year = lookup[job['source_id']].metadata['year']
                topic = topic.rstrip('.').strip()
                history = [{'role': 'user', 'content': f"Let's discuss {topic} described in Berkshire's {year} letter."},
                           {'role': 'assistant', 'content': f'Which aspect of {topic} would you like to examine?'}]
                candidates.append((followup, {'history': history, 'kind': kind, 'style': 'v2'}))
        else:
            for question in parse_questions(raw)[:2 if kind == 'passage' else 1]:
                extra = {'kind': kind, 'style': 'v2'}
                if kind in ('passage', 'multipart'):
                    year = str(lookup[job['source_id']].metadata['year'])
                    if year not in question:
                        question = f'In the {year} letter, ' + question[:1].lower() + question[1:]
                if kind == 'temporal':
                    years = [str(lookup[i].metadata['year']) for i in job['source_ids']]
                    if not all(y in question for y in years):
                        continue
                if kind == 'passage':
                    extra = {'kind': 'single', 'style': 'v2'}
                if kind == 'unanswerable':
                    extra = {'kind': 'unanswerable', 'style': 'v2'}
                candidates.append((question, extra))
        for index, (question, extra) in enumerate(candidates):
            history_text = ' '.join(t['content'] for t in extra.get('history', []))
            fingerprint = tuple(sorted(token_set(question + ' ' + history_text)))
            if fingerprint in known or near_frozen(question, frozen):
                continue
            row = _row(job, index, question, lookup, valid_years, **extra)
            append(OUT / 'questions.jsonl', row)
            kept.append(row)
            known.add(fingerprint)
        append(OUT / 'question_attempts.jsonl', {'id': job['id'], 'kept': len(kept), 'raw': raw, **metrics})
        done.add(job['id'])
        generated += 1
        print(f'question_job={job["id"]} kept={len(kept)} tok/s={metrics["generation_tps"]:.2f}', flush=True)
        yield kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int, help='Maximum new source/template jobs this invocation')
    parser.add_argument('--style', choices=('v1', 'v2'), default='v2')
    args = parser.parse_args()
    if args.style == 'v2':
        init_v2()
        for _ in generate_pending_v2(Teacher(), args.limit):
            pass
        print(f'questions_total={len(read_jsonl(OUT / "questions.jsonl"))}', flush=True)
        return
    if (OUT / 'questions.jsonl').exists() and not args.resume:
        parser.error('Output exists; use --resume (no destructive overwrite).')
    for _ in generate_pending(Teacher(), args.limit):
        pass
    print(f'questions_total={len(read_jsonl(OUT / "questions.jsonl"))}', flush=True)


if __name__ == '__main__':
    main()
