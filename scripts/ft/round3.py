"""Round-3 evidence-obligation generation and leakage-safe, immutable v2 reuse."""
from __future__ import annotations

from collections import Counter, defaultdict
import fcntl
import json
import os
import random
import re
import time

import common
import make_examples as examples
from make_questions import TOPIC_WORDS, _json_object, unavailable_templates
from src.evaluation.citation_faithfulness import split_sentences
from src.generation.prompt import REFUSAL_LINE, SYSTEM_PROMPT, build_cited_prompt, parse_citations
from src.generation.evidence_gate import assess_evidence
from src.retrieval.context import expand_hits_with_neighbors
from src.storage import SearchHit

STYLE_V3 = (
    '\nAnswer every supported part of the question, including every requested period. '
    'Select the sentences that directly answer, not nearby background or repeated filler. '
    'Preserve the supporting passage\'s exact numbers, units, names, key nouns and negation. '
    'Copy each supplied evidence sentence in full with its own passage citation; do not merge '
    'different passages or invent calculations, motives or trends. One sentence is enough for '
    'one fact; use as many distinct sentences as the supported parts require, at most six. '
    'Resolve follow-up pronouns from conversation history. Refuse only if no passage supports '
    'any requested part, using the exact refusal line alone. Do not repeat sentences.'
)
REVISION = 2
KINDS = ('single', 'multipart', 'followup', 'temporal', 'unanswerable')
NUMBER = re.compile(r'(?<![\w])[$€£]?\d[\d,]*(?:\.\d+)?%?(?:\s+(?:million|billion|trillion))?', re.I)


def fingerprint(row):
    return tuple(sorted(common.token_set(common.question_text(row))))


def chat_hash(row):
    return common.sha(json.dumps(row['messages'], sort_keys=True, ensure_ascii=False).encode())


def frozen_jsonl(path):
    """Immutable prior-round inputs: malformed rows fail, never repair in place."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reuse_quality_reason(row, lookup):
    """Never mutate a legacy target; omit it if today's serving contract rejects it."""
    anchors = [SearchHit(id=i, text=lookup[i].text, metadata=lookup[i].metadata, score=0.) for i in row['passage_ids']]
    local = {i: d for i, d in lookup.items() if i != row['source_id']} if row['type'] == 'distractor' else lookup
    hits = expand_hits_with_neighbors(anchors, local, neighbors=1, max_chars=1800)
    messages = examples.chat_messages(row['question'], hits, row.get('history'))
    if row['messages'][:-1] != messages:
        return 'serving_prompt_drift'
    answer = row['messages'][-1]['content']
    if row['type'] == 'answerable':
        years = [int(lookup[i].metadata['year']) for i in row['source_ids']] if row.get('kind') == 'temporal' else None
        return examples.check_answer(answer, hits, row['source_id'], years)
    if answer != REFUSAL_LINE:
        return 'non_contract_refusal'
    if row['type'] == 'distractor' and assess_evidence(row['question'], hits).sufficient:
        return 'distractor_evidence_sufficient'
    return None


def reused_rows(excluded, frozen):
    """Read exactly the delivered v2 splits, not discarded/undelivered examples."""
    directory = common.ROOT / 'data/ft_v2'
    lookup = {d.id: d for d in common.load_chunks_as_docs(common.CORPUS) if d.id not in excluded}
    metadata = frozen_jsonl(directory / 'split_metadata.jsonl')
    all_rows = {row['id']: row for row in frozen_jsonl(directory / 'examples.jsonl')}
    hashes, questions, result = set(), set(), []
    rejected = Counter()
    candidates = Counter()
    for split in ('train', 'valid'):
        rows = [row for row in metadata if row['split'] == split]
        delivered = frozen_jsonl(directory / f'{split}.jsonl')
        if not delivered or len(delivered) != len(rows):
            raise RuntimeError(f'Incomplete v2 provenance for {split}')
        for meta, chat in zip(rows, delivered):
            row = all_rows[meta['id']]
            assert chat == {'messages': row['messages']}, 'v2 delivered chat/provenance mismatch'
            assert all(row.get(k) == v for k, v in meta.items()), 'v2 metadata/provenance mismatch'
            candidates[split] += 1
            reasons = common.leakage_reasons(row, excluded, frozen)
            key, qkey = chat_hash(row), fingerprint(row)
            if key in hashes or qkey in questions:
                reasons.append('duplicate_chat_or_question')
            if not reasons:
                quality = reuse_quality_reason(row, lookup)
                if quality:
                    reasons.append('legacy_quality_' + quality)
            if reasons:
                rejected.update(reasons)
                continue
            result.append({**row, 'origin': 'ft_v2', 'original_id': row['id'], 'id': 'reuse:v2:' + row['id']})
            hashes.add(key)
            questions.add(qkey)
    identity = {name: common.sha((directory / name).read_bytes()) for name in
                ('train.jsonl', 'valid.jsonl', 'split_metadata.jsonl', 'examples.jsonl')}
    return result, {'candidate_counts': dict(candidates), 'kept_counts': dict(Counter(r['split'] for r in result)),
                    'excluded_or_duplicate_reason_counts': dict(rejected), 'input_sha256': identity}


def init_plan():
    docs, excluded, frozen, valid_years, identity = common.inputs()
    path = common.OUT / 'question_plan.json'
    reused, reuse = reused_rows(excluded, frozen)
    if path.exists():
        plan = json.loads(path.read_text())
        assert plan['identity'] == identity, 'Round-3 input identity changed; refusing resume'
        assert plan['reuse']['input_sha256'] == reuse['input_sha256'], 'v2 reuse inputs changed; refusing resume'
        return plan
    for row in reused:
        common.append(common.OUT / 'examples.jsonl', row)
    rng = random.Random(common.SEED + 3)
    old_sources = {r.get('source_id') for r in reused} | {i for r in reused for i in r.get('source_ids', [])}
    prose = [d for d in docs if d.id not in old_sources and len(d.text) >= 250
             and sum(c.isalpha() for c in d.text) / len(d.text) > .60]
    by_split = {s: [d for d in prose if ('valid' if int(d.metadata['year']) in valid_years else 'train') == s]
                for s in ('train', 'valid')}
    for ds in by_split.values():
        rng.shuffle(ds)
    topics = defaultdict(list)
    for doc in prose:
        for topic in TOPIC_WORDS:
            if topic in doc.text[:1600].lower():
                topics[topic].append(doc)
    templates = unavailable_templates()
    rng.shuffle(templates)
    jobs, cursors = [], Counter()
    for cycle in range(300):
        split = 'valid' if cycle % 10 == 9 else 'train'
        for kind in KINDS:
            job: dict = {'id': f'v3:{kind}:{cycle:03d}', 'kind': kind, 'type': 'answerable', 'split': split}
            if kind == 'unanswerable':
                if not templates:
                    continue
                job.update(type='unanswerable', template=templates.pop(), source_id=None)
            elif kind == 'temporal':
                viable = []
                for topic, ds in sorted(topics.items()):
                    ds = [d for d in ds if (int(d.metadata['year']) in valid_years) == (split == 'valid')]
                    if len({d.metadata['year'] for d in ds}) > 1:
                        viable.append((topic, ds))
                if not viable:
                    continue
                topic, ds = rng.choice(viable)
                a = rng.choice(ds)
                b = rng.choice([d for d in ds if d.metadata['year'] != a.metadata['year']])
                job.update(topic=topic, source_id=a.id, source_ids=[a.id, b.id])
            else:
                index = cursors[split]
                if index >= len(by_split[split]):
                    continue
                doc = by_split[split][index]
                cursors[split] += 1
                job['source_id'] = doc.id
            jobs.append(job)
    plan = {'identity': identity, 'style': 'v3', 'prompt_revision': REVISION, 'reuse': reuse,
            'counts': dict(Counter(j['kind'] for j in jobs)), 'jobs': jobs}
    common.atomic_json(path, plan)
    print(f'v3 plan: {len(jobs)} jobs; reused={len(reused)}; pilot={common.PILOT}', flush=True)
    return plan


def source_evidence(job, lookup):
    evidence = []
    for ident in job.get('source_ids') or [job['source_id']]:
        doc = lookup[ident]
        hit = SearchHit(id=ident, text=doc.text[:1800], metadata=doc.metadata, score=0.)
        sentences = split_sentences(re.sub(r'\s+', ' ', doc.text[:1600]))
        candidates = []
        for sentence in sentences:
            if not 45 <= len(sentence) <= 550 or sentence[-1] not in '.?!\u201d"':
                continue
            if not sentence[0].isupper() or re.match(r'^(?:This|That|These|Those|It|They|He|She|Such|But he|The remaining)\b', sentence):
                continue
            if sum(c.isdigit() or c == '.' for c in sentence) / len(sentence) > .25:
                continue
            if examples.check_answer(sentence + ' [1]', [hit], ident) is not None:
                continue
            topic_match = job.get('topic', '').lower() in sentence.lower()
            candidates.append((not topic_match, -len(NUMBER.findall(sentence)), len(sentence), sentence))
        candidates.sort()
        wanted = 2 if job['kind'] == 'multipart' else 1
        if len(candidates) < wanted or (job['kind'] == 'temporal' and (not candidates or candidates[0][0])):
            return []
        for _, _, _, sentence in candidates[:wanted]:
            evidence.append({'id': f'E{len(evidence) + 1}', 'source_id': ident,
                             'year': int(doc.metadata['year']), 'sentence': sentence})
    return evidence


def synthesize(teacher, job, lookup, frozen, known):
    evidence = source_evidence(job, lookup) if job['type'] == 'answerable' else []
    if job['type'] == 'answerable' and not evidence:
        return None, {'id': job['id'], 'kind': job['kind'], 'kept': 0, 'reason': 'no_complete_clean_source_sentence', 'seconds': 0.}
    if job['kind'] == 'unanswerable':
        prompt = ('Paraphrase this unavailable-evidence question without removing any date, name or qualifier. '
                  'Return ONLY JSON: {"question":"...?"}.\n' + job['template'])
    else:
        instructions = {
            'single': 'Ask ONE specific fact or explanation answered completely by E1.',
            'multipart': 'Ask TWO distinct parts joined by and: part one answered by E1, part two by E2.',
            'temporal': 'Ask a two-period comparison with BOTH source years: what does each letter state about the shared subject? Each side must use its evidence; never request a calculated change.',
            'followup': 'Write a concrete noun-phrase topic and a follow-up question referring to it only by it/they/that. The question must ask the specific fact E1 states and be intelligible with topic history.',
        }[job['kind']]
        prompt = (instructions + ' Use only these source sentences, not outside facts. Preserve concrete names and key topic words. '
                  'Include the source year(s) in the question, except follow-ups use history instead. '
                  'For each evidence item, supply one short question part that it answers. '
                  'Return ONLY JSON: {"question":"...?","topic":"noun phrase for followup only",'
                  '"parts":[{"ask":"specific requested fact","evidence_id":"E1"}]}. '
                  'Include every evidence_id exactly once. Ask genuine interrogative questions beginning What, Which, How, Why, Who or When; '
                  'never paste an answer statement and add a question mark or reveal the requested answer in the question. '
                  'Follow-ups must use a subject pronoun, not the relative-clause word that in a copied statement. '
                  '\nSOURCE SENTENCES:\n' + json.dumps(evidence, ensure_ascii=False))
    raw, metrics = teacher.generate([{'role': 'system', 'content': 'Write grounded user questions as valid JSON only.'},
                                     {'role': 'user', 'content': prompt}], job['id'] + ':question', max_tokens=360, temperature=.1)
    return build_question(job, evidence, raw, metrics, frozen, known)


def build_question(job, evidence, raw, metrics, frozen, known):
    """Strict checks on a teacher-written question (shared with the v4 retry path); (row, record)."""
    record = {'id': job['id'], 'kind': job['kind'], 'kept': 0, 'raw': raw, **metrics}
    obj = _json_object(raw)
    question = obj.get('question', '')
    if not isinstance(question, str) or not 20 <= len(question) <= 600 or not question.endswith('?'):
        record['reason'] = 'invalid_question'
        return None, record
    history = []
    if not re.search(r'\b(what|which|how|why|who|when|where)\b', question, re.I):
        record['reason'] = 'not_an_interrogative_question'
        return None, record
    if job['kind'] == 'followup':
        topic = obj.get('topic')
        if not isinstance(topic, str) or not 2 <= len(topic) <= 150 or not re.search(r'\b(it|they|that|this|their)\b', question, re.I):
            record['reason'] = 'invalid_followup_topic_or_pronoun'
            return None, record
        history = [{'role': 'user', 'content': f"Let's discuss {topic} in Berkshire's {evidence[0]['year']} letter."},
                   {'role': 'assistant', 'content': f'Which aspect of {topic} would you like to examine?'}]
    elif evidence and not all(str(e['year']) in question for e in evidence):
        if job['kind'] == 'temporal':
            record['reason'] = 'source_year_missing'
            return None, record
        question = f"In the {evidence[0]['year']} letter, " + question[:1].lower() + question[1:]
    parts = obj.get('parts', [])
    if evidence:
        if not isinstance(parts, list) or any(not isinstance(p, dict) or not isinstance(p.get('ask'), str) or not p['ask'].strip() for p in parts):
            record['reason'] = 'invalid_question_parts'
            return None, record
        if sorted(p.get('evidence_id', '') for p in parts) != sorted(e['id'] for e in evidence):
            record['reason'] = 'incomplete_question_parts'
            return None, record
    row = {k: v for k, v in job.items() if k not in {'template', 'topic'}}
    row.update(question=question, style='v3', origin='v3_new', evidence=evidence, parts=parts,
               source_year=evidence[0]['year'] if evidence else None)
    if history:
        row['history'] = history
    if fingerprint(row) in known or common.leakage_reasons(row, set(), frozen):
        record['reason'] = 'duplicate_or_frozen_question'
        return None, record
    record['kept'] = 1
    return row, record


def coverage_reason(answer, hits, obligations):
    sentences = [s for line in answer.splitlines() for s in examples._split_original(line)]
    plain = [re.sub(r'\[\d+(?:\s*,\s*\d+)*\]', '', s).strip() for s in sentences]
    if len(set(plain)) != len(plain):
        return 'repeated_sentence'
    for evidence in obligations:
        terms = common.content_terms(evidence['sentence'])
        numbers = set(NUMBER.findall(evidence['sentence']))
        covered = False
        for sentence in sentences:
            cited = {i for c in parse_citations(sentence, hits) for i in c['passage_ids']}
            if evidence['source_id'] in cited and terms <= common.content_terms(sentence) and numbers <= set(NUMBER.findall(sentence)):
                covered = True
                break
        if not covered:
            return 'requested_part_missing_or_source_wording_changed'
    return None


class BuilderV3(examples.Builder):
    def __init__(self, teacher, target):
        super().__init__(teacher, style='v3', target=target)
        _, _, self.frozen, _, _ = common.inputs()
        self.known = {fingerprint(row) for row in self.rows}
        self.hashes = {chat_hash(row) for row in self.rows}

    def process(self, question):
        ident = question['id']
        attempt_id = f'{ident}:v3p{REVISION}'
        if ident in self.accepted or attempt_id in self.done:
            return False
        hits = self.retrieve(question)
        messages = examples.chat_messages(question['question'], hits, question.get('history'))
        generation = [dict(m) for m in messages]
        generation[0]['content'] += STYLE_V3
        obligations = question.get('evidence', [])
        if obligations:
            mapped = []
            for evidence in obligations:
                indices = [i + 1 for i, h in enumerate(hits) if h.id == evidence['source_id'] and evidence['sentence'] in re.sub(r'\s+', ' ', h.text)]
                if not indices:
                    record = {'attempt_id': attempt_id, 'id': ident, 'type': question['type'], 'kind': question['kind'], 'kept': False,
                              'reason': 'source_evidence_not_in_context', 'seconds': 0., 'tokens': 0}
                    common.append(common.OUT / 'example_attempts.jsonl', record)
                    self.attempts.append(record)
                    self.done.add(attempt_id)
                    return True
                part = next(p['ask'] for p in question['parts'] if p['evidence_id'] == evidence['id'])
                mapped.append(f"Requested part: {part}\nDirect supporting sentence: {evidence['sentence']} [{indices[0]}]")
            generation[1]['content'] += ('\n\nComplete coverage checklist (answer every part from its cited source):\n' + '\n'.join(mapped)
                                         + '\n\nOUTPUT CONTRACT: Return ONLY the Direct supporting sentence lines, copied verbatim '
                                           'with their given citations. No requested-part labels, no framing, no added years, '
                                           'no paraphrasing. Include each distinct supporting sentence once.')
        output, metrics = self.teacher.generate(generation, attempt_id, max_tokens=600, temperature=.1)
        reason = None
        if metrics['finish_reason'] in ('length', 'budget'):
            reason = 'truncated_teacher_output'
        elif question['type'] == 'answerable':
            years = [e['year'] for e in obligations] if question['kind'] == 'temporal' else None
            reason = examples.check_answer(output, hits, question['source_id'], years)
            if reason is None:
                reason = coverage_reason(output, hits, obligations)
            if reason is None and not 1 <= examples.answer_stats(output, hits)['sentences'] <= 6:
                reason = 'sentence_count_out_of_range'
        elif output != REFUSAL_LINE:
            reason = 'unsupported_teacher_did_not_refuse'
        stats = examples.answer_stats(output, hits) if output else {}
        row = {**question, 'messages': messages + [{'role': 'assistant', 'content': output}]}
        if reason is None and (fingerprint(row) in self.known or chat_hash(row) in self.hashes):
            reason = 'duplicate_chat_or_question'
        record = {'attempt_id': attempt_id, 'id': ident, 'type': question['type'], 'kind': question['kind'],
                  'kept': reason is None, 'reason': reason, 'teacher_output': output, **stats, **metrics}
        if reason is None:
            self.keep(question, hits, messages, output, metrics)
            self.known.add(fingerprint(row))
            self.hashes.add(chat_hash(row))
        common.append(common.OUT / 'example_attempts.jsonl', record)
        self.attempts.append(record)
        self.done.add(attempt_id)
        print(f'example={ident} kind={question["kind"]} kept={reason is None} reason={reason} new_kept={self.new_kept}', flush=True)
        return True

    @property
    def new_kept(self):
        return sum(row.get('origin') == 'v3_new' for row in self.rows)

    def materialize(self, *, final=False):
        """Do not downsample immutable v2 reuse to impose a refusal ratio."""
        selected, hashes, seen = [], set(), set()
        for row in self.rows:
            key, qkey = chat_hash(row), fingerprint(row)
            if key in hashes or qkey in seen:
                continue
            selected.append(row)
            hashes.add(key)
            seen.add(qkey)
        for split in ('train', 'valid'):
            rows = sorted([r for r in selected if r['split'] == split], key=lambda r: common.sha(f'{common.SEED}:v3:{r["id"]}'.encode()))
            temporary = common.OUT / f'{split}.jsonl.tmp'
            temporary.write_text(''.join(json.dumps({'messages': r['messages']}, ensure_ascii=False) + '\n' for r in rows))
            temporary.replace(common.OUT / f'{split}.jsonl')
        # Metadata order MUST match each delivered split's order.
        selected.sort(key=lambda r: (r['split'], common.sha(f'{common.SEED}:v3:{r["id"]}'.encode())))
        temporary = common.OUT / 'split_metadata.jsonl.tmp'
        temporary.write_text(''.join(json.dumps({k: v for k, v in r.items() if k not in {'messages', 'teacher_metrics'}}, ensure_ascii=False) + '\n' for r in selected))
        temporary.replace(common.OUT / 'split_metadata.jsonl')
        return selected

    def finish(self):
        selected = self.materialize(final=True)
        plan = json.loads((common.OUT / 'question_plan.json').read_text())
        manifest = {**plan['identity'], 'style': 'v3', 'pilot': common.PILOT, 'prompt_revision': REVISION,
                    'target_new_examples': self.target, 'target_met': self.new_kept >= self.target,
                    'mix': {origin: {split: sum(r.get('origin') == origin and r['split'] == split for r in selected)
                                     for split in ('train', 'valid')} for origin in ('ft_v2', 'v3_new')},
                    'counts_per_split': dict(Counter(r['split'] for r in selected)),
                    'counts_per_split_type': {s: dict(Counter(r['type'] for r in selected if r['split'] == s)) for s in ('train', 'valid')},
                    'kinds_kept_new': dict(Counter(r['kind'] for r in selected if r.get('origin') == 'v3_new')),
                    'reuse': plan['reuse'], 'dedup': 'normalized question+user history and exact chat SHA256; original v2 split preserved',
                    'heldout_v2_exclusion': 'SKIPPED_PILOT_NOT_TRAINABLE' if common.PILOT else 'required_and_applied_to_new_and_all_reused_examples',
                    'settings': {'temperature': .1, 'max_tokens': 600, 'coverage': 'all source-derived question obligations, exact numeric units and key content terms, per-sentence source citation'},
                    'rejection_reasons': dict(Counter(r['reason'] for r in self.attempts if r.get('reason')))}
        common.atomic_json(common.OUT / 'manifest.json', manifest)
        print('MIX ' + json.dumps(manifest['mix']), flush=True)


def run(args, parser):
    pilot = args.pilot is not None or os.environ.get('PILOT') == '1'
    if pilot and args.split_only:
        parser.error('Pilot is generation-only; no production split-only or training stage')
    if args.max_minutes is not None and args.max_minutes <= 0:
        parser.error('--max-minutes must be positive')
    common.configure_round3(pilot)
    examples.OUT = common.OUT
    common.OUT.mkdir(parents=True, exist_ok=True)
    lock = (common.OUT / 'generation.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = init_plan()
    target = args.target or 600
    if args.split_only:
        builder = BuilderV3(None, target)
        builder.finish()
        import check_leakage
        check_leakage.OUT = common.OUT
        check_leakage.check()
        total = sum(len(common.read_jsonl(common.OUT / f'{s}.jsonl')) for s in ('train', 'valid'))
        assert len(common.read_jsonl(common.OUT / 'valid.jsonl')) >= .05 * total
        return
    previous_seconds = sum(r.get('seconds', 0.) for name in ('example_attempts.jsonl', 'question_attempts.jsonl')
                           for r in common.read_jsonl(common.OUT / name))
    minutes = min(args.max_minutes or 7., max(0., (450. - previous_seconds) / 60)) if pilot else args.max_minutes
    started = time.monotonic()
    deadline = started + minutes * 60 if minutes is not None else None
    teacher = common.Teacher(deadline=deadline)
    builder = BuilderV3(teacher, target)
    initial_attempts = len(builder.attempts)
    initial_questions = len(common.read_jsonl(common.OUT / 'question_attempts.jsonl'))
    done = {r['id'] for r in common.read_jsonl(common.OUT / 'question_attempts.jsonl')}
    processed = 0

    def stop():
        return ((deadline is not None and time.monotonic() >= deadline) or builder.new_kept >= target
                or (args.pilot is not None and processed >= args.pilot))

    # Resume previously synthesized questions before spending teacher time on more.
    for question in common.read_jsonl(common.OUT / 'questions.jsonl'):
        if stop():
            break
        if builder.process(question):
            processed += 1
    known = {fingerprint(r) for r in builder.rows + common.read_jsonl(common.OUT / 'questions.jsonl')}
    for job in plan['jobs']:
        if stop():
            break
        if job['id'] in done:
            continue
        question, record = synthesize(teacher, job, builder.lookup, builder.frozen, known)
        common.append(common.OUT / 'question_attempts.jsonl', record)
        if question:
            common.append(common.OUT / 'questions.jsonl', question)
            known.add(fingerprint(question))
            if not stop() and builder.process(question):
                processed += 1
        print(f'question_job={job["id"]} kind={job["kind"]} kept={record["kept"]} reason={record.get("reason")}', flush=True)
    builder.finish()
    attempts = builder.attempts if pilot else builder.attempts[initial_attempts:]
    question_jobs = common.read_jsonl(common.OUT / 'question_attempts.jsonl')
    if not pilot:
        question_jobs = question_jobs[initial_questions:]
    teacher_seconds = sum(r.get('seconds', 0.) for r in attempts + question_jobs)
    kept = sum(r['kept'] for r in attempts)
    per_kind = {}
    for kind in KINDS:
        answers = [r for r in attempts if r['kind'] == kind]
        jobs = [r for r in question_jobs if r['kind'] == kind]
        n = sum(r['kept'] for r in answers)
        per_kind[kind] = {'kept': n, 'answer_attempts': len(answers), 'keep_rate': n / len(answers) if answers else None,
                          'question_jobs': len(jobs), 'questions_kept': sum(r['kept'] for r in jobs)}
    summary = {'pilot': pilot, 'new_kept': kept, 'answer_attempts': len(attempts), 'keep_rate': kept / max(1, len(attempts)),
               'teacher_seconds': teacher_seconds, 'wall_seconds': time.monotonic() - started,
               'seconds_per_kept_example': teacher_seconds / kept if kept else None,
               'estimated_teacher_minutes_for_600_new': teacher_seconds / kept * 600 / 60 if kept else None,
               'by_kind': per_kind, 'rejection_reasons': dict(Counter(r['reason'] for r in attempts if r.get('reason'))),
               'question_rejection_reasons': dict(Counter(r['reason'] for r in question_jobs if r.get('reason'))),
               'target_new_examples': target, 'total_new_kept': builder.new_kept, 'output': str(common.OUT)}
    common.atomic_json(common.OUT / ('pilot_v3.json' if pilot else 'generation_summary.json'), summary)
    print(('PILOT ' if pilot else 'RUN ') + json.dumps(summary), flush=True)
