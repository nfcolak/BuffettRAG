"""Seeded, resumable local-MLX question synthesis; never contacts a provider."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import random
from common import OUT, SEED, Teacher, append, atomic_json, inputs, near_frozen, parse_questions, read_jsonl, sha, token_set


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int, help='Maximum new source/template jobs this invocation')
    args = parser.parse_args()
    if (OUT / 'questions.jsonl').exists() and not args.resume:
        parser.error('Output exists; use --resume (no destructive overwrite).')
    for _ in generate_pending(Teacher(), args.limit):
        pass
    print(f'questions_total={len(read_jsonl(OUT / "questions.jsonl"))}', flush=True)


if __name__ == '__main__':
    main()
