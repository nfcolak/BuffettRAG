"""Style v4 (FT-r2): complete multi-part answers on serving-identical 8-passage prompts.

Usage:
  PY scripts/ft/round4.py --dry-run N                   # no teacher; prompt lengths per kind
  PY scripts/ft/round4.py --pilot N [--max-minutes M]   # throwaway output (models/ft/round4_pilot)
  PY scripts/ft/round4.py [--target T] [--mix k=v,..] [--max-minutes M]   # resumable, writes data/ft_v4
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time

import common
import make_examples as examples
import round3
from make_questions import TOPIC_WORDS, _json_object, unavailable_templates
from config import (
    ANSWER_CONTEXT_MAX_CHARS, ANSWER_CONTEXT_NEIGHBORS, LLM_CONTEXT_PASSAGES, LLM_N_CTX,
    LLM_PASSAGE_MAX_CHARS, RETRIEVAL_FETCH_K,
)
from src.evaluation.claim_validator import _split_original, validate_and_filter_answer
from src.generation.compare import comparison_periods, period_label, prepare_answer_context
from src.generation.evidence_gate import assess_evidence
from src.generation.prompt import REFUSAL_LINE, parse_citations
from src.retrieval.bm25 import BM25Retriever, _meta_matches
from src.retrieval.query_expansion import build_followup_retrieval_query
from src.retrieval.retriever import (
    deduplicate_hits, detect_temporal_comparison, detect_year_filter, reciprocal_rank_fusion,
    reserve_period_hits,
)
from src.services.schemas import AskRequest
from src.storage import SearchHit

REVISION = 1
HELDOUT_V3 = 'data/evaluation/heldout_v3/answer_benchmark_heldout_v3.json'
HELDOUT_V4_CHECKER = common.ROOT / 'scripts/eval/check_heldout_v4.py'
DEFAULT_TEACHER = '/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/models/teacher-qwen2.5-14b-mlx4'
PILOT_DIR = 'models/ft/round4_pilot'
FINAL_DIR = 'data/ft_v4'
SERVING_PASSAGES = 8          # 7B serving context (config._DEFAULT_CONTEXT_PASSAGES)
KINDS = ('single', 'multipart', 'followup', 'temporal', 'unanswerable', 'distractor')
CATEGORIES = ('multipart', 'temporal', 'followup', 'single', 'refusal')
DEFAULT_MIX = {'multipart': 300, 'temporal': 200, 'followup': 120, 'single': 250, 'refusal': 130}
MAX_ANSWER_SENTENCES = 6

STYLE_V4 = (
    '\nANSWER EVERY PART of the question and never drop one. Write each part in its own sentence(s); each sentence '
    'quotes or only lightly trims ONE sentence of ONE passage (keep its numbers, names and negation) and ends with '
    'that passage\'s citation [n]. Never merge two passage sentences into one sentence. If the question ends with '
    '"(focus: X)", answer only for that period. If a part is not covered by any passage, write exactly one sentence '
    '"The passages do not state <that part>." with no citation instead of leaving the part out. Copy the refusal line '
    'alone, without citation, only when NO part of the question is covered. Do not infer, calculate or compare beyond '
    'what each sentence states, and do not start a sentence with Additionally, Furthermore, Moreover or Therefore.'
)
BANNED_OPENER = re.compile(r'(?m)(?:^|[.!?\]] )(?:Additionally|Furthermore|Moreover|Therefore),')
ABSENT = re.compile(r'^The passages do not state\s+(?P<part>.+?)\.?\s*$')
BULLET = re.compile(r'^\s*(?:[-*]|\d+\.)\s+')
CITATION = re.compile(r'\[\d+(?:\s*,\s*\d+)*\]')
NUMBER = re.compile(r'\d[\d,]*(?:\.\d+)?')


# ------------------------------------------------------------------ serving-identical prompts
class LlamaStub:
    """What prepare_answer_context needs to apply the 7B fit (provider_name == 'llama')."""
    provider_name = 'llama'


def serving_settings():
    if LLM_CONTEXT_PASSAGES != SERVING_PASSAGES:
        raise RuntimeError(f'LLM_CONTEXT_PASSAGES={LLM_CONTEXT_PASSAGES}, expected {SERVING_PASSAGES} (7B serving); '
                           'run with LLM_MODEL_PATH and LLM_CONTEXT_PASSAGES unset')
    return {'neighbors': ANSWER_CONTEXT_NEIGHBORS, 'max_chars': ANSWER_CONTEXT_MAX_CHARS,
            'max_new_tokens': AskRequest(query='x').max_new_tokens, 'n_ctx': LLM_N_CTX,
            'max_passages': LLM_CONTEXT_PASSAGES, 'passage_max_chars': LLM_PASSAGE_MAX_CHARS}


def retrieve(bm25, query, history=None, *, top_k=8, fetch_k=None, drop_ids=()):
    """BM25 anchors the way Retriever.search(strategy='bm25') returns them for /ask (top_k=8)."""
    fetch_k = fetch_k or RETRIEVAL_FETCH_K
    resolved = build_followup_retrieval_query(query, list(history or []))
    drop = set(drop_ids)

    def search(q, where):
        found = bm25.search(q, top_k=fetch_k + len(drop), where=where)
        return [h for h in found if h.id not in drop][:fetch_k]

    periods = detect_temporal_comparison(resolved)
    if periods:
        rankings = [search(resolved, period) for period in periods]
        candidates = reciprocal_rank_fusion(rankings, top_k=sum(len(r) for r in rankings))
        buckets = [deduplicate_hits([h for h in candidates if _meta_matches(h.metadata, p)]) for p in periods]
        eligible = {h.id for bucket in buckets for h in bucket}
        return reserve_period_hits([h for h in candidates if h.id in eligible], periods, top_k)
    where = detect_year_filter(resolved)
    hits = search(resolved, where)
    if where and not hits:
        hits = search(resolved, None)
    return deduplicate_hits(hits)[:top_k]


def with_sources(anchors, needed, lookup):
    """If retrieval missed a source passage, it replaces the lowest-ranked anchor (as in v2/v3)."""
    anchors, needed = list(anchors), list(dict.fromkeys(needed))
    injected = False
    for ident in needed:
        if ident in {hit.id for hit in anchors}:
            continue
        doc = lookup[ident]
        fresh = SearchHit(id=doc.id, text=doc.text, metadata=doc.metadata, score=0.)
        injected = True
        for index in range(len(anchors) - 1, -1, -1):
            if anchors[index].id not in needed:
                anchors[index] = fresh
                break
        else:
            anchors.append(fresh)
    return anchors, injected


def serving_prompts(query, anchors, lookup, history=None):
    """Every model prompt /ask would send for `query` given retrieved `anchors`.

    One prompt for an ordinary question; one per period (focus suffix, only that
    period's passages) for a temporal comparison, exactly as generate_comparison_answer.
    """
    history = list(history or [])
    settings = serving_settings()
    followup = build_followup_retrieval_query(query, history) != query
    context = prepare_answer_context(LlamaStub(), anchors, lookup, query, history=history, followup=followup,
                                     **settings)
    extras = [turn['content'] for turn in history if turn['role'] == 'user']
    periods = comparison_periods(query)
    if not periods:
        return [{'period': None, 'query': query, 'hits': context, 'messages': examples.chat_messages(query, context, history),
                 'gate': assess_evidence(query, context, extra_queries=extras).sufficient}]
    prompts = []
    for period in periods:
        label = period_label(period)
        subset = [hit for hit in context if _meta_matches(hit.metadata, period)]
        focus = f'{query} (focus: {label})'
        prompts.append({'period': label, 'query': focus, 'hits': subset, 'messages': examples.chat_messages(focus, subset, history),
                        'gate': assess_evidence(query, subset, extra_queries=extras).sufficient})
    return prompts


def qkey(row):
    """Question identity; a temporal question yields one row per period."""
    return round3.fingerprint(row), row.get('period')


def category(kind):
    return 'refusal' if kind in ('unanswerable', 'distractor') else kind


# ------------------------------------------------------------------ exclusions
def heldout_v4_excluded_ids(checker=None):
    """Passage ids the blind v4 set excludes. Refuses to run without its checker."""
    checker = Path(checker) if checker is not None else HELDOUT_V4_CHECKER
    if not checker.is_file():
        raise RuntimeError(f'heldout_v4 checker missing ({checker.name}); refusing to generate without the blind exclusions')
    result = subprocess.run([sys.executable, str(checker), '--excluded-passage-ids'], cwd=common.ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'heldout_v4 checker failed (exit {result.returncode}); refusing to generate')
    try:
        ids = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError('heldout_v4 checker did not print a JSON list; refusing to generate') from error
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise RuntimeError('heldout_v4 checker output is not a list of passage ids; refusing to generate')
    return sorted(set(ids))


def heldout_v3_passage_ids():
    ids = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {'gold_passage_ids', 'relevant_ids'}:
                    ids.update(value)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    blob = (common.ROOT / HELDOUT_V3).read_bytes()
    walk(json.loads(blob))
    return ids, common.sha(blob)


def with_neighbours(ids, full):
    result = set(ids)
    for ident in ids:
        if ident not in full:
            raise ValueError(f'Frozen passage not in corpus ({len(ids)} ids checked)')
        for key in ('previous_chunk_id', 'next_chunk_id'):
            neighbour = full[ident].metadata.get(key)
            if neighbour:
                result.add(neighbour)
    return result


def load_inputs(*, dry_run=False, checker=None, teacher_path=DEFAULT_TEACHER, pilot=False):
    """v3 exclusions + heldout_v3 + heldout_v4. Pilots keep full custody (no shortcuts)."""
    checker_path = Path(checker) if checker is not None else HELDOUT_V4_CHECKER
    v4_ids, v4_sha = [], None
    if dry_run and not checker_path.is_file():
        print('dry-run: heldout_v4 checker not present; v4 exclusions NOT applied (not trainable)', flush=True)
    else:
        v4_ids = heldout_v4_excluded_ids(checker_path)     # refuses (before touching any output) if the checker is missing
        v4_sha = common.sha(json.dumps(v4_ids).encode())
    common.ROUND3, common.PILOT = True, False
    docs, excluded, frozen, valid_years, identity = common.inputs()
    full = {d.id: d for d in common.load_chunks_as_docs(common.CORPUS)}
    v3_ids, v3_sha = heldout_v3_passage_ids()
    excluded = set(excluded) | with_neighbours(v3_ids, full) | with_neighbours(v4_ids, full)
    docs = [d for d in docs if d.id not in excluded]
    identity = {**identity, 'style': 'v4', 'pilot': bool(pilot), 'teacher_id': str(teacher_path),
                'excluded_passage_count': len(excluded), 'heldout_v4_excluded_sha256': v4_sha,
                'frozen_set_hashes': {**identity['frozen_set_hashes'], HELDOUT_V3: v3_sha}}
    return docs, excluded, frozen, valid_years, identity


# ------------------------------------------------------------------ plan
def parse_mix(text):
    mix = dict(DEFAULT_MIX)
    for part in filter(None, (text or '').split(',')):
        key, _, value = part.partition('=')
        if key.strip() not in mix or not value.strip().isdigit():
            raise ValueError(f'bad --mix entry {part!r}; keys: {", ".join(DEFAULT_MIX)}')
        mix[key.strip()] = int(value)
    return mix


def scale_mix(mix, target):
    if not target:
        return dict(mix)
    total = sum(mix.values())
    return {key: max(1, round(value * target / total)) for key, value in mix.items()}


def has_topic_sentence(doc, topic, cache):
    """True if round3.source_evidence finds a clean sentence of `doc` that contains `topic` (the temporal rule)."""
    key = (topic, doc.id)
    if key not in cache:
        cache[key] = bool(round3.source_evidence({'kind': 'temporal', 'topic': topic, 'source_id': doc.id}, {doc.id: doc}))
    return cache[key]


def clean_topic_pair(topic, docs, rng, cache):
    """Two docs of different years that each have a topic-matching clean sentence, or None.

    Temporal jobs used to pair any two docs mentioning the topic in their first 1600 chars; half of them
    then failed source_evidence because the mention was not in a complete, clean sentence.
    """
    docs = list(docs)
    rng.shuffle(docs)
    first = None
    for doc in docs:
        if not has_topic_sentence(doc, topic, cache):
            continue
        if first is None:
            first = doc
        elif doc.metadata['year'] != first.metadata['year']:
            return first, doc
    return None


def build_plan(docs, valid_years, identity, quotas, oversample):
    rng = random.Random(common.SEED + 4)
    prose = [d for d in docs if len(d.text) >= 250 and sum(c.isalpha() for c in d.text) / len(d.text) > .60]
    by_split = {s: [d for d in prose if ('valid' if int(d.metadata['year']) in valid_years else 'train') == s] for s in ('train', 'valid')}
    for ds in by_split.values():
        rng.shuffle(ds)
    topics = defaultdict(list)
    for doc in prose:
        for topic in TOPIC_WORDS:
            if topic in doc.text[:1600].lower():
                topics[topic].append(doc)
    templates = unavailable_templates()
    rng.shuffle(templates)
    used, cursors, clean_cache = set(), Counter(), {}

    def next_doc(split):
        while cursors[split] < len(by_split[split]):
            doc = by_split[split][cursors[split]]
            cursors[split] += 1
            if doc.id not in used:
                used.add(doc.id)
                return doc
        return None

    sizes = {'single': quotas['single'], 'multipart': quotas['multipart'], 'followup': quotas['followup'],
             'temporal': math.ceil(quotas['temporal'] / 2), 'unanswerable': math.ceil(quotas['refusal'] / 2),
             'distractor': math.ceil(quotas['refusal'] / 2)}
    lists = {}
    for kind in sorted(KINDS, key=lambda k: k != 'temporal'):   # temporal needs the scarcest sources: allocate first
        wanted = math.ceil(sizes[kind] * oversample)
        jobs = []
        for index in range(wanted):
            split = 'valid' if index % 10 == 9 else 'train'
            job: dict = {'id': f'v4:{kind}:{index:04d}', 'kind': kind, 'split': split,
                         'type': 'unanswerable' if kind == 'unanswerable' else 'answerable'}
            if kind == 'unanswerable':
                if not templates:
                    break
                job.update(template=templates.pop(), source_id=None)
            elif kind == 'temporal':
                viable = []
                for topic, ds in sorted(topics.items()):
                    ds = [d for d in ds if d.id not in used and (int(d.metadata['year']) in valid_years) == (split == 'valid')]
                    if len({d.metadata['year'] for d in ds}) > 1:
                        viable.append((topic, ds))
                pair = None
                while viable and pair is None:
                    topic, ds = viable.pop(rng.randrange(len(viable)))
                    pair = clean_topic_pair(topic, ds, rng, clean_cache)
                if pair is None:
                    break
                a, b = pair
                used.update((a.id, b.id))
                job.update(topic=topic, source_id=a.id, source_ids=[a.id, b.id])
            else:
                doc = next_doc(split)
                if doc is None:
                    break
                job['source_id'] = doc.id
                if kind == 'multipart':
                    job['variant'] = 'partial' if index % 4 == 3 else 'full'
                if kind == 'distractor':
                    job['type'] = 'distractor'
            jobs.append(job)
        lists[kind] = jobs
    keyed = sorted(((i + .5) / len(jobs), KINDS.index(kind), job) for kind, jobs in lists.items() for i, job in enumerate(jobs))
    return {'identity': identity, 'style': 'v4', 'prompt_revision': REVISION, 'quotas_at_plan': quotas, 'oversample': oversample,
            'counts': {kind: len(jobs) for kind, jobs in lists.items()}, 'jobs': [job for _, _, job in keyed]}


# ------------------------------------------------------------------ question synthesis
QUESTION_SYSTEM = 'Write grounded user questions as valid JSON only.'
QUESTION_RULES = {
    'followup': (
        'Write the user\'s NEXT chat message in a conversation. The conversation so far is only "Let\'s discuss <topic>.", '
        'so the message must refer to the topic ONLY with the word it, they, that, this or their and must NOT name the topic. '
        '"topic" is a short noun phrase for the subject of E1. The question asks the one specific fact E1 states.\n'
        'EXAMPLE sources: [{"id":"E1","year":1994,"sentence":"The bakery sold 4,000 loaves a day in 1994."}]\n'
        'EXAMPLE answer: {"question":"How many loaves did it sell each day?","topic":"the bakery\'s daily sales",'
        '"parts":[{"ask":"loaves sold per day","evidence_id":"E1"}]}\n'
        'Wrong (standalone, no it/they/that): "How many loaves did the bakery sell each day?"'),
    'temporal': (
        'Write ONE question that compares two letters. It must name BOTH source years literally and ask what each letter '
        'states about the shared subject. Never ask for a calculated change. Give one part per evidence item (E1 and E2).\n'
        'EXAMPLE sources: [{"id":"E1","year":1991,"sentence":"Retiree health costs are not funded in advance."},'
        '{"id":"E2","year":2003,"sentence":"Retiree health costs are now recorded as a liability."}]\n'
        'EXAMPLE answer: {"question":"How did the 1991 and 2003 letters describe retiree health costs?","topic":"",'
        '"parts":[{"ask":"how the 1991 letter treats retiree health costs","evidence_id":"E1"},'
        '{"ask":"how the 2003 letter treats retiree health costs","evidence_id":"E2"}]}\n'
        'Wrong (names only one year): "What does the 1991 letter say about retiree health costs compared to later?"'),
}
RETRYABLE = {'invalid_question', 'not_an_interrogative_question', 'invalid_followup_topic_or_pronoun', 'source_year_missing',
             'invalid_question_parts', 'incomplete_question_parts'}


def question_prompt_v4(job, evidence):
    return (QUESTION_RULES[job['kind']] + '\nUse only these source sentences, not outside facts. Preserve concrete names and key topic words. '
            'Ask a genuine interrogative question beginning What, Which, How, Why, Who or When; never paste an answer statement '
            'and add a question mark. Return ONLY JSON: {"question":"...?","topic":"...","parts":[{"ask":"...","evidence_id":"E1"}]} '
            'with every evidence_id exactly once.\nSOURCE SENTENCES:\n' + json.dumps(evidence, ensure_ascii=False))


def retry_message(reason, job, evidence):
    years = ' and '.join(str(e['year']) for e in evidence)
    ids = ', '.join(e['id'] for e in evidence)
    rules = {
        'invalid_followup_topic_or_pronoun': 'a follow-up needs a non-empty "topic" noun phrase and a question that contains it, they, that, this or their '
                                             'instead of naming the topic',
        'source_year_missing': f'the question must contain every source year literally: {years}',
        'invalid_question_parts': 'every entry of "parts" needs a non-empty "ask" string',
        'incomplete_question_parts': f'"parts" must hold exactly one entry for each of {ids}',
    }
    rule = rules.get(reason, '"question" must be one interrogative sentence of 20-600 characters that starts with What, Which, How, Why, Who or When '
                             'and ends with ?')
    return f'REJECTED ({reason}): {rule}. Rewrite the JSON so it obeys this rule; return ONLY the JSON object.'


def synthesize_v4_question(teacher, job, lookup, frozen, known):
    """Follow-up / temporal questions: instruction plus worked example, and one retry naming the broken rule."""
    evidence = round3.source_evidence(job, lookup)
    if not evidence:
        return None, {'id': job['id'], 'kind': job['kind'], 'kept': 0, 'reason': 'no_complete_clean_source_sentence', 'seconds': 0.}
    messages = [{'role': 'system', 'content': QUESTION_SYSTEM}, {'role': 'user', 'content': question_prompt_v4(job, evidence)}]
    raw, metrics = teacher.generate(messages, job['id'] + ':question', max_tokens=360, temperature=.1)
    row, record = round3.build_question(job, evidence, raw, metrics, frozen, known)
    if metrics.get('finish_reason') == 'budget' or row or record.get('reason') not in RETRYABLE:
        return row, record
    messages += [{'role': 'assistant', 'content': raw}, {'role': 'user', 'content': retry_message(record['reason'], job, evidence)}]
    raw2, metrics2 = teacher.generate(messages, job['id'] + ':question:retry', max_tokens=360, temperature=.1)
    if metrics2.get('finish_reason') == 'budget':
        return None, {**record, **metrics2}
    row, retried = round3.build_question(job, evidence, raw2, metrics2, frozen, known)
    retried.update(retry=True, first_reason=record['reason'], first_raw=raw,
                   seconds=metrics.get('seconds', 0.) + metrics2.get('seconds', 0.), tokens=metrics.get('tokens', 0) + metrics2.get('tokens', 0))
    return row, retried


def synthesize(teacher, job, lookup, frozen, known):
    """Round-3 question synthesis (evidence-obligation parts) relabelled for v4."""
    if job['kind'] == 'multipart' and job.get('variant') == 'partial':
        return synthesize_partial(teacher, job, lookup, frozen, known)
    if job['kind'] in QUESTION_RULES:
        row, record = synthesize_v4_question(teacher, job, lookup, frozen, known)
        if row:
            row.update(style='v4', origin='v4_new')
        return row, record
    shadow = {**job, 'kind': 'single', 'type': 'answerable'} if job['kind'] == 'distractor' else job
    row, record = round3.synthesize(teacher, shadow, lookup, frozen, known)
    record['kind'] = job['kind']
    if row:
        row.update(kind=job['kind'], type=job['type'], style='v4', origin='v4_new')
    return row, record


def synthesize_partial(teacher, job, lookup, frozen, known):
    """Two-part question: one part stated by E1, one detail the evidence does not state."""
    evidence = round3.source_evidence({**job, 'kind': 'single'}, lookup)
    if not evidence:
        return None, {'id': job['id'], 'kind': 'multipart', 'kept': 0, 'reason': 'no_complete_clean_source_sentence', 'seconds': 0.}
    prompt = ('Write ONE question with TWO parts joined by and. Part one is answered by E1. Part two asks for one specific detail '
              'on the same subject that E1 does NOT state and that cannot be inferred from it (a figure, name, date or reason it leaves out). '
              'Use only the source sentence, no outside facts. Include the source year in the question. '
              'Ask a genuine interrogative question beginning What, Which, How, Why, Who or When. '
              'Return ONLY JSON: {"question":"...?","parts":[{"ask":"specific requested fact","evidence_id":"E1"}],'
              '"missing":"short noun phrase (3-10 words) naming the unanswered detail"}.\nSOURCE SENTENCES:\n'
              + json.dumps(evidence, ensure_ascii=False))
    raw, metrics = teacher.generate([{'role': 'system', 'content': 'Write grounded user questions as valid JSON only.'},
                                     {'role': 'user', 'content': prompt}], job['id'] + ':question', max_tokens=360, temperature=.1)
    record = {'id': job['id'], 'kind': 'multipart', 'kept': 0, 'raw': raw, **metrics}
    obj = _json_object(raw)
    question, missing, parts = obj.get('question', ''), obj.get('missing', ''), obj.get('parts', [])
    year = evidence[0]['year']
    if not isinstance(question, str) or not 20 <= len(question) <= 600 or not question.endswith('?') \
            or not re.search(r'\b(what|which|how|why|who|when|where)\b', question, re.I):
        record['reason'] = 'invalid_question'
        return None, record
    if not isinstance(missing, str) or not 8 <= len(missing.strip()) <= 90:
        record['reason'] = 'invalid_missing_part'
        return None, record
    missing = re.sub(r'\s+', ' ', missing).strip().rstrip('.?')
    terms = common.content_terms(missing)
    if not terms or len(terms & common.content_terms(evidence[0]['sentence'])) / len(terms) >= .6:
        record['reason'] = 'missing_part_in_evidence'
        return None, record
    if not isinstance(parts, list) or [p.get('evidence_id') for p in parts if isinstance(p, dict)] != ['E1'] \
            or not isinstance(parts[0].get('ask'), str) or not parts[0]['ask'].strip():
        record['reason'] = 'invalid_question_parts'
        return None, record
    if str(year) not in question:
        question = f'In the {year} letter, ' + question[:1].lower() + question[1:]
    row = {k: v for k, v in job.items() if k not in {'template', 'topic'}}
    row.update(question=question, style='v4', origin='v4_new', evidence=evidence, parts=parts[:1], source_year=year,
               missing_parts=[{'id': 'M1', 'ask': missing}])
    if round3.fingerprint(row) in known or common.leakage_reasons(row, set(), frozen):
        record['reason'] = 'duplicate_or_frozen_question'
        return None, record
    record['kept'] = 1
    return row, record


# ------------------------------------------------------------------ completeness filter
def _numbers(text):
    return {m.group().replace(',', '') for m in NUMBER.finditer(text)}


def split_answer(answer):
    """(cited_lines, absent_sentences): `The passages do not state ...` sentences never go through the validator."""
    cited_lines, absent = [], []
    for line in answer.splitlines():
        kept = []
        for sentence in _split_original(line):
            match = ABSENT.match(BULLET.sub('', sentence.strip()))
            if match and not CITATION.search(sentence):
                absent.append(match.group('part'))
            else:
                kept.append(sentence.strip())
        if kept:
            cited_lines.append(' '.join(kept))
    return cited_lines, absent


def completeness_reason(answer, hits, covered, missing=(), question_text=''):
    """None if every question obligation is answered in a validator-kept cited sentence or named as not stated."""
    if not answer or answer == REFUSAL_LINE:
        return 'answerable_refusal'
    cited_lines, absent = split_answer(answer)
    cited_text = '\n'.join(cited_lines)
    sentences = [s for line in cited_lines for s in _split_original(line)]
    if not sentences:
        return 'no_cited_sentence'
    if BANNED_OPENER.search(answer):
        return 'banned_filler'
    for sentence in sentences:
        citations = parse_citations(sentence, hits)
        if not citations or not all(c['passage_indices'] and not c['invalid_numbers'] for c in citations):
            return 'missing_or_invalid_sentence_citation'
    result = validate_and_filter_answer(cited_text, hits)
    kept = [s for line in result.safe_answer.splitlines() for s in _split_original(line)]
    if result.blocked_claims or len(kept) != len(sentences):
        return 'validator_dropped_sentence'
    if len(sentences) + len(absent) > MAX_ANSWER_SENTENCES:
        return 'too_many_sentences'
    reason = round3.coverage_reason(cited_text, hits, covered)
    if reason:
        return reason
    if len(absent) > len(missing):
        return 'unrequested_absent_statement'
    for part in missing:
        terms = common.content_terms(part['ask'])
        if not any(terms and len(terms & common.content_terms(a)) / len(terms) >= .8 for a in absent):
            return 'missing_part_not_named'
        for hit in hits:
            for sentence in round3.split_sentences(re.sub(r'\s+', ' ', hit.text)):
                if terms and len(terms & common.content_terms(sentence)) / len(terms) >= .8:
                    return 'missing_part_found_in_context'
    allowed = _numbers(' '.join(h.text for h in hits) + ' ' + question_text)
    if not _numbers(CITATION.sub('', answer)) <= allowed:
        return 'invented_number'
    return None


# ------------------------------------------------------------------ builder
def checklist_messages(messages, covered, missing):
    generation = [dict(m) for m in messages]
    generation[0]['content'] += STYLE_V4
    if covered or missing:
        lines = []
        for part, sentence, index in covered:
            lines.append(f'Requested part: {part}\nDirect supporting sentence: {sentence} [{index}]')
        for item in missing:
            lines.append(f"Requested part: {item['ask']}\nNot stated in any passage. Write exactly: The passages do not state {item['ask']}.")
        generation[1]['content'] += ('\n\nPART CHECKLIST (answer every part, in this order):\n' + '\n'.join(lines)
                                     + '\n\nOUTPUT CONTRACT: Return ONLY one line per checklist part, in order. For a stated part copy the '
                                       'Direct supporting sentence verbatim with its given citation. For a part that is not stated write the '
                                       'given sentence with no citation. No labels, framing, added years or paraphrasing.')
    return generation


class Builder4:
    def __init__(self, teacher, quotas):
        self.teacher, self.quotas = teacher, quotas
        self.docs, self.excluded, self.frozen, self.valid_years, self.identity = LOADED
        self.lookup = {d.id: d for d in self.docs}
        self.bm25 = {s: BM25Retriever([d for d in self.docs if (int(d.metadata['year']) in self.valid_years) == (s == 'valid')])
                     for s in ('train', 'valid')}
        self.rows = common.read_jsonl(common.OUT / 'examples.jsonl')
        self.accepted = {row['id'] for row in self.rows}
        self.attempts = common.read_jsonl(common.OUT / 'example_attempts.jsonl')
        self.done = {row['attempt_id'] for row in self.attempts}
        self.known = {qkey(row) for row in self.rows}
        self.hashes = {round3.chat_hash(row) for row in self.rows}
        self.materialize()

    def kept_per_category(self):
        return Counter(category(row['kind']) for row in self.rows)

    def quota_met(self, cat):
        return self.kept_per_category()[cat] >= self.quotas[cat]

    def all_met(self):
        return all(self.quota_met(cat) for cat in CATEGORIES)

    def context_ids(self, hits, lookup):
        ids = set()
        for hit in hits:
            ids.add(hit.id)
            doc = lookup.get(hit.id)
            for field in ('previous_chunk_id', 'next_chunk_id'):
                if doc and doc.metadata.get(field) in lookup:
                    ids.add(doc.metadata[field])
        return sorted(ids)

    def prompts_for(self, question):
        distractor = question['type'] == 'distractor'
        lookup = {k: v for k, v in self.lookup.items() if k != question['source_id']} if distractor else self.lookup
        history = question.get('history')
        anchors = retrieve(self.bm25[question['split']], question['question'], history,
                           drop_ids=[question['source_id']] if distractor else ())
        natural = True
        if question['type'] == 'answerable':
            anchors, injected = with_sources(anchors, question.get('source_ids') or [question['source_id']], lookup)
            natural = not injected
        prompts = serving_prompts(question['question'], anchors, lookup, history)
        for prompt in prompts:
            prompt['natural'] = natural
        return prompts, lookup

    def log(self, record):
        common.append(common.OUT / 'example_attempts.jsonl', record)
        self.attempts.append(record)
        self.done.add(record['attempt_id'])

    def process(self, question):
        """Answer one question (one prompt, or one per period); returns the number of teacher calls spent."""
        spent = 0
        prompts, lookup = self.prompts_for(question)
        kind = question['kind']
        if kind == 'temporal' and {p['period'] for p in prompts} != {str(e['year']) for e in question['evidence']}:
            self.log({'attempt_id': f"{question['id']}:v4p{REVISION}", 'id': question['id'], 'type': question['type'], 'kind': kind,
                      'kept': False, 'reason': 'periods_mismatch', 'seconds': 0.})
            return 0
        for prompt in prompts:
            row_id = question['id'] if prompt['period'] is None else f"{question['id']}:{prompt['period']}"
            attempt_id = f'{row_id}:v4p{REVISION}'
            if row_id in self.accepted or attempt_id in self.done:
                continue
            if self.quota_met(category(kind)):
                break
            hits, answerable = prompt['hits'], question['type'] == 'answerable'
            base = {'attempt_id': attempt_id, 'id': row_id, 'type': question['type'], 'kind': kind, 'period': prompt['period'],
                    'passages': len(hits), 'prompt_chars': sum(len(m['content']) for m in prompt['messages']),
                    'gate_sufficient': prompt['gate'], 'natural_retrieval': prompt['natural']}
            covered, missing, reason = [], list(question.get('missing_parts', [])), None
            obligations = [e for e in question.get('evidence', []) if prompt['period'] is None or str(e['year']) == prompt['period']]
            if answerable and not prompt['gate']:
                reason = 'evidence_gate_insufficient'
            for evidence in obligations if answerable and reason is None else ():
                indices = [i + 1 for i, h in enumerate(hits) if h.id == evidence['source_id']
                           and evidence['sentence'] in re.sub(r'\s+', ' ', h.text)]
                if not indices:
                    reason = 'source_evidence_not_in_context'
                    break
                ask = next(p['ask'] for p in question['parts'] if p['evidence_id'] == evidence['id'])
                covered.append((ask, evidence['sentence'], indices[0]))
            if reason:
                self.log({**base, 'kept': False, 'reason': reason, 'seconds': 0., 'tokens': 0})
                continue
            generation = checklist_messages(prompt['messages'], covered, missing if answerable else [])
            output, metrics = self.teacher.generate(generation, attempt_id, max_tokens=600, temperature=.1)
            if metrics['finish_reason'] == 'budget':
                return spent
            spent += 1
            if metrics['finish_reason'] == 'length':
                reason = 'truncated_teacher_output'
            elif answerable:
                reason = completeness_reason(output, hits, [e for e in obligations], missing,
                                             question['question'] + ' ' + ' '.join(t['content'] for t in question.get('history', [])))
            elif output != REFUSAL_LINE:
                reason = 'refusal_expected_teacher_answered'
            row = {**question, 'id': row_id, 'period': prompt['period'], 'serving_query': prompt['query'],
                   'passage_ids': [h.id for h in hits], 'context_passage_ids': self.context_ids(hits, lookup),
                   'passages': len(hits), 'prompt_chars': base['prompt_chars'], 'gate_sufficient': prompt['gate'],
                   'natural_retrieval': prompt['natural'], 'messages': prompt['messages'] + [{'role': 'assistant', 'content': output}]}
            if reason is None:
                reason = ','.join(common.leakage_reasons(row, self.excluded, self.frozen)) or None
            if reason is None and (qkey(row) in self.known or round3.chat_hash(row) in self.hashes):
                reason = 'duplicate_chat_or_question'
            stats = examples.answer_stats(output, hits) if output else {}
            self.log({**base, 'kept': reason is None, 'reason': reason, 'teacher_output': output, **stats, **metrics})
            if reason is None:
                row['teacher_metrics'] = metrics
                common.append(common.OUT / 'examples.jsonl', row)
                self.rows.append(row)
                self.accepted.add(row_id)
                self.known.add(qkey(row))
                self.hashes.add(round3.chat_hash(row))
                if len(self.rows) % 5 == 0:
                    self.materialize()
            print(f'example={row_id} kind={kind} kept={reason is None} reason={reason} kept_total={len(self.rows)}', flush=True)
        return spent

    def materialize(self):
        selected, hashes, seen = [], set(), set()
        for row in self.rows:
            key, question_key = round3.chat_hash(row), qkey(row)
            if key in hashes or question_key in seen:
                continue
            selected.append(row)
            hashes.add(key)
            seen.add(question_key)
        order = lambda r: common.sha(f'{common.SEED}:v4:{r["id"]}'.encode())
        for split in ('train', 'valid'):
            rows = sorted([r for r in selected if r['split'] == split], key=order)
            temporary = common.OUT / f'{split}.jsonl.tmp'
            temporary.write_text(''.join(json.dumps({'messages': r['messages']}, ensure_ascii=False) + '\n' for r in rows))
            temporary.replace(common.OUT / f'{split}.jsonl')
        selected.sort(key=lambda r: (r['split'], order(r)))   # metadata order == delivered split order
        temporary = common.OUT / 'split_metadata.jsonl.tmp'
        temporary.write_text(''.join(json.dumps({k: v for k, v in r.items() if k not in {'messages', 'teacher_metrics'}}, ensure_ascii=False) + '\n' for r in selected))
        temporary.replace(common.OUT / 'split_metadata.jsonl')
        return selected

    def finish(self, pilot):
        selected = self.materialize()
        kept = Counter(category(r['kind']) for r in selected)
        chars = sorted(r['prompt_chars'] for r in selected)
        manifest = {**self.identity, 'target_new_examples': sum(self.quotas.values()), 'quotas': self.quotas, 'kept_per_category': dict(kept),
                    'target_met': all(kept[c] >= self.quotas[c] for c in CATEGORIES),
                    'counts_per_split': dict(Counter(r['split'] for r in selected)),
                    'counts_per_split_kind': {s: dict(Counter(r['kind'] for r in selected if r['split'] == s)) for s in ('train', 'valid')},
                    'refusal_share': sum(r['type'] != 'answerable' for r in selected) / max(1, len(selected)),
                    'refusals_gate_sufficient': sum(r['type'] != 'answerable' and r.get('gate_sufficient', False) for r in selected),
                    'natural_retrieval_share': sum(bool(r.get('natural_retrieval')) for r in selected) / max(1, len(selected)),
                    'passages_per_prompt': dict(Counter(r['passages'] for r in selected)),
                    'prompt_chars_max': chars[-1] if chars else None, 'prompt_chars_p95': chars[int(.95 * (len(chars) - 1))] if chars else None,
                    'settings': {**serving_settings(), 'temperature': .1, 'teacher_max_tokens': 600,
                                 'split_strategy': 'seeded source-year grouping (~10% valid years); retrieval stays inside each split',
                                 'prompt_revision': REVISION, 'retrieval': 'bm25 top 8 (serving-style year/period handling)'},
                    'heldout_exclusions': 'dev, heldout_v1, heldout_v2, heldout_v3 (ids), heldout_v4 (checker) with +-1 neighbours',
                    'rejection_reasons': dict(Counter(r['reason'] for r in self.attempts if r.get('reason')))}
        common.atomic_json(common.OUT / 'manifest.json', manifest)
        print('MIX ' + json.dumps(manifest['kept_per_category']), flush=True)
        return manifest


LOADED = None


# ------------------------------------------------------------------ dry run
def token_counter(teacher_path):
    """Qwen tokenizer from the teacher dir if present, else chars/4."""
    try:
        if Path(teacher_path).is_dir():
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(teacher_path), local_files_only=True)
            return 'qwen-tokenizer', lambda messages: len(tok(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))['input_ids'])
    except Exception as error:   # noqa: BLE001 - any tokenizer problem falls back to the estimate
        print(f'dry-run: tokenizer unavailable ({type(error).__name__}); using chars/4', flush=True)
    return 'chars/4', lambda messages: sum(len(m['content']) for m in messages) // 4


def proxy_question(job, lookup):
    """Dry-run stand-in for the teacher-written question (same year/keyword shape)."""
    evidence = round3.source_evidence({**job, 'kind': 'single'} if job['kind'] in ('distractor', 'multipart') else job, lookup) \
        if job['type'] != 'unanswerable' else []
    if job['type'] == 'unanswerable':
        return {**job, 'question': job['template']}
    if not evidence:
        return None
    words = lambda e, n: ', '.join(sorted(common.content_terms(e['sentence']), key=len, reverse=True)[:n])
    row = {k: v for k, v in job.items() if k not in {'template', 'topic'}}
    year = evidence[0]['year']
    if job['kind'] == 'temporal':
        question = f"How do the {evidence[0]['year']} and {evidence[1]['year']} letters describe {job['topic']}, including {words(evidence[0], 2)}?"
    elif job['kind'] == 'multipart':
        question = f"In the {year} letter, what does Berkshire say about {words(evidence[0], 2)} and why, regarding {words(evidence[0], 3)}?"
    elif job['kind'] == 'followup':
        row['history'] = [{'role': 'user', 'content': f"Let's discuss {words(evidence[0], 2)} in Berkshire's {year} letter."},
                          {'role': 'assistant', 'content': 'Which aspect of it would you like to examine?'}]
        question = 'What did Buffett say about it and how did they handle that?'
    else:
        question = f"In the {year} letter, what does Berkshire say about {words(evidence[0], 3)}?"
    row.update(question=question, evidence=evidence)
    return row


def quantile(values, q):
    values = sorted(values)
    return values[int(q * (len(values) - 1))] if values else None


def dry_run(count, args):
    common.OUT = Path(tempfile.mkdtemp(prefix='ftv4_dry_'))
    teacher_path = os.environ.get('FT_TEACHER', DEFAULT_TEACHER)
    global LOADED
    LOADED = load_inputs(dry_run=True, teacher_path=teacher_path)
    docs, _, _, valid_years, identity = LOADED
    quotas = scale_mix(parse_mix(args.mix), args.target)
    plan = build_plan(docs, valid_years, identity, quotas, args.oversample)
    by_kind = defaultdict(list)
    for job in plan['jobs']:
        by_kind[job['kind']].append(job)
    chosen = []
    share = {kind: len(jobs) for kind, jobs in by_kind.items()}
    for kind, jobs in by_kind.items():
        take = max(1, round(count * share[kind] / sum(share.values())))
        chosen += random.Random(common.SEED).sample(jobs, min(take, len(jobs)))
    name, tokens = token_counter(teacher_path)
    builder = Builder4(None, quotas)
    stats = defaultdict(lambda: {'tokens': [], 'passages': [], 'skipped': 0, 'gate': 0, 'natural': 0})
    for job in chosen:
        question = proxy_question(job, builder.lookup)
        if question is None:
            stats[job['kind']]['skipped'] += 1
            continue
        question['source_id'] = job.get('source_id')
        prompts, _ = builder.prompts_for(question)
        for prompt in prompts:
            entry = stats[job['kind']]
            entry['tokens'].append(tokens(prompt['messages']))
            entry['passages'].append(len(prompt['hits']))
            entry['gate'] += bool(prompt['gate'])
            entry['natural'] += bool(prompt['natural'])
    print(f'dry-run: {len(chosen)} sampled jobs, token counter={name}, proxy questions (no teacher); prompt tokens exclude the answer')
    print(f'{"kind":13} {"prompts":>7} {"skipped":>7} {"tok_max":>8} {"tok_p95":>8} {"tok_mean":>8} {"passages(min/mean/max)":>24} {"gate_ok":>8} {"natural":>8}')
    for kind in KINDS:
        entry = stats.get(kind)
        if not entry:
            continue
        n = len(entry['tokens'])
        if not n:
            print(f'{kind:13} {0:>7} {entry["skipped"]:>7}')
            continue
        passages = entry['passages']
        print(f'{kind:13} {n:>7} {entry["skipped"]:>7} {max(entry["tokens"]):>8} {quantile(entry["tokens"], .95):>8} '
              f'{sum(entry["tokens"]) // n:>8} {min(passages):>8}/{sum(passages) / n:.1f}/{max(passages):<2}      {entry["gate"]:>5}/{n:<2} {entry["natural"]:>5}/{n:<2}')
    everything = [t for e in stats.values() for t in e['tokens']]
    print(f'ALL: prompts={len(everything)} max={max(everything)} p95={quantile(everything, .95)} (add up to ~350 answer tokens for the training max length)')


# ------------------------------------------------------------------ run
def run(args, parser):
    global LOADED
    pilot = args.pilot is not None or os.environ.get('PILOT') == '1'
    if args.max_minutes is not None and args.max_minutes <= 0:
        parser.error('--max-minutes must be positive')
    if pilot and args.split_only:
        parser.error('Pilot is generation-only')
    teacher_path = os.environ.get('FT_TEACHER', DEFAULT_TEACHER)
    common.OUT = common.ROOT / (PILOT_DIR if pilot else FINAL_DIR)
    LOADED = load_inputs(teacher_path=teacher_path, pilot=pilot)   # raises without the heldout_v4 checker
    examples.OUT = common.OUT
    common.OUT.mkdir(parents=True, exist_ok=True)
    lock = (common.OUT / 'generation.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    docs, _, frozen, valid_years, identity = LOADED
    quotas = scale_mix(parse_mix(args.mix), args.target)
    path = common.OUT / 'question_plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        assert plan['identity'] == identity, 'Round-4 input identity changed; refusing resume'
    else:
        plan = build_plan(docs, valid_years, identity, quotas, args.oversample)
        common.atomic_json(path, plan)
        print(f'v4 plan: {plan["counts"]} pilot={pilot}', flush=True)
    if args.split_only:
        Builder4(None, quotas).finish(pilot)
        return
    started = time.monotonic()
    deadline = started + args.max_minutes * 60 if args.max_minutes else None
    teacher = common.Teacher(deadline=deadline, path=teacher_path)
    builder = Builder4(teacher, quotas)
    initial = len(builder.attempts)
    processed = 0

    def stop():
        return ((deadline is not None and time.monotonic() >= deadline) or builder.all_met()
                or (args.pilot is not None and processed >= args.pilot))

    for question in common.read_jsonl(common.OUT / 'questions.jsonl'):     # resume synthesized-but-unanswered questions
        if stop():
            break
        processed += builder.process(question)
    done = {r['id'] for r in common.read_jsonl(common.OUT / 'question_attempts.jsonl')}
    known = {round3.fingerprint(r) for r in builder.rows + common.read_jsonl(common.OUT / 'questions.jsonl')}
    for job in plan['jobs']:
        if stop():
            break
        if job['id'] in done or builder.quota_met(category(job['kind'])):
            continue
        question, record = synthesize(teacher, job, builder.lookup, frozen, known)
        if record.get('finish_reason') == 'budget':
            break
        common.append(common.OUT / 'question_attempts.jsonl', record)
        if question:
            common.append(common.OUT / 'questions.jsonl', question)
            known.add(round3.fingerprint(question))
            processed += builder.process(question)
        print(f'question_job={job["id"]} kind={job["kind"]} kept={record["kept"]} reason={record.get("reason")}', flush=True)
    manifest = builder.finish(pilot)
    attempts = builder.attempts[initial:] if not pilot else builder.attempts
    answered = [a for a in attempts if a.get('teacher_output') is not None]
    seconds = sum(a.get('seconds', 0.) for a in attempts)
    summary = {'pilot': pilot, 'new_kept': sum(a['kept'] for a in attempts), 'teacher_answers': len(answered),
               'keep_rate': sum(a['kept'] for a in answered) / max(1, len(answered)), 'teacher_seconds': seconds,
               'wall_seconds': time.monotonic() - started, 'kept_per_category': manifest['kept_per_category'],
               'rejection_reasons': dict(Counter(a['reason'] for a in attempts if a.get('reason'))), 'output': str(common.OUT)}
    common.atomic_json(common.OUT / ('pilot_v4.json' if pilot else 'generation_summary.json'), summary)
    print(('PILOT ' if pilot else 'RUN ') + json.dumps(summary), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', type=int, metavar='N', help='no teacher/heldout_v4: print prompt token lengths per kind for N sampled jobs')
    parser.add_argument('--pilot', type=int, nargs='?', const=50, metavar='N', help='answer at most N questions into a throwaway dir (models/ft/round4_pilot)')
    parser.add_argument('--max-minutes', type=float, help='stop teacher work after this wall-clock time')
    parser.add_argument('--target', type=int, help='total new kept rows; scales the mix proportionally')
    parser.add_argument('--mix', help='per-category targets, e.g. multipart=300,temporal=200,followup=120,single=250,refusal=130')
    parser.add_argument('--oversample', type=float, default=4., help='plan size multiple of each quota (plan is fixed once written)')
    parser.add_argument('--split-only', action='store_true', help='rewrite train/valid/metadata/manifest from existing examples; no generation')
    args = parser.parse_args(argv)
    if args.dry_run is not None:
        return dry_run(args.dry_run, args)
    return run(args, parser)


if __name__ == '__main__':
    main()
