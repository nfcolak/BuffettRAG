"""ft_v5 generator: the teacher answers the exact serving prompt; nothing else is shown to it.

Usage:
  PY scripts/ft/round5.py --dry-run N                   # no teacher; prompt token stats per kind (Qwen tokenizer)
  PY scripts/ft/round5.py --pilot N [--max-minutes M]   # throwaway output (models/ft/round5_pilot)
  PY scripts/ft/round5.py [--target T] [--mix k=v,..] [--max-minutes M] [--split-only]   # resumable, writes data/ft_v5
Run with VECTOR_BACKEND=faiss HF_HUB_OFFLINE=1 PYTHON_DOTENV_DISABLED=1 (and LLM_MODEL_PATH / LLM_CONTEXT_PASSAGES unset).

What differs from round4 (each a measured defect):
  answers    the teacher sees only the serving messages (+ one short style suffix): no gold sentence, no part checklist.
             Kept only if the serving validator keeps >= 85% of the sentences and every question part is covered.
  questions  the writer sees the source passage and must not copy it (4-gram / token-Jaccard / "the passage says" checks).
  retrieval  hybrid + reranker + neighbour merge + fit_context_to_llm through ask_flow._prepare_ask; the source passage is
             never injected. Rows whose source is not served are natural_miss=true and capped at 10% of the kept rows.
  temporal   one serving call per period like compare.py around a shared number-bearing metric; a gated period drops the pair.
  followup   the history assistant turn is a real kept answer to a first question.
  gaps       "not stated" sentences only for a part whose source passage is not among the served passages.
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
import round3
import round4
from make_questions import TOPIC_WORDS, _json_object, unavailable_templates
from src.evaluation.claim_validator import _split_original, evidence_sentences, validate_and_filter_answer
from src.generation.compare import comparison_periods, period_label
from src.generation.evidence_gate import assess_evidence
from src.generation.prompt import REFUSAL_LINE, build_cited_prompt, format_answer_markdown, parse_citations, strip_chat_artifacts
from src.generation.providers.llama_provider import split_prompt
from src.retrieval.bm25 import _meta_matches
from src.services import ask_flow
from src.services.schemas import AskRequest

REVISION = 1
SCRIPTS_EVAL = common.ROOT / 'scripts/eval'
HELDOUT_V4_CHECKER = SCRIPTS_EVAL / 'check_heldout_v4.py'
HELDOUT_V5_CHECKER = SCRIPTS_EVAL / 'check_heldout_v5.py'
DEV_CASES = 'data/evaluation/grounded_dev_v1/confirm/dev_cases.json'
DEFAULT_TEACHER = round4.DEFAULT_TEACHER
PILOT_DIR = 'models/ft/round5_pilot'
FINAL_DIR = 'data/ft_v5'
KINDS = ('single', 'multipart', 'followup', 'temporal', 'refusal')
DEFAULT_MIX = {'multipart': 300, 'temporal': 150, 'followup': 120, 'single': 300, 'refusal': 100}
KEEP_RATE = .85
NATURAL_MISS_SHARE = .10
MAX_ANSWER_TOKENS = 600

STYLE_V5 = ('\nAnswer every part of the question. Write one fact per sentence; each sentence must restate ONE passage sentence closely, '
            "reusing that sentence's own words and numbers. Shorten it if needed; do not merge two passage sentences into one. "
            'End each sentence with exactly one citation [n], never multiple citations such as [1,3]. '
            'If a part is truly not in the passages, say so in one short sentence.')
CITATION = re.compile(r'\[\d+(?:\s*,\s*\d+)*\]')
BULLET = re.compile(r'^\s*(?:[-*]|\d+\.)\s+')
GAP = re.compile(r"\b(?:do(?:es)?\s+not|don't|doesn't|never)\s+(?:state|say|provide|mention|cover|address|specify|include|give|describe|report|"
                 r"indicate|contain|discuss)\b|\b(?:is|are|was|were)\s+not\s+(?:stated|mentioned|provided|covered|addressed|specified|given|"
                 r"reported|discussed)\b|\bno\s+(?:information|mention|details?|figures?|data)\b", re.I)
PASSAGE_TALK = re.compile(r"\b(?:passages?|excerpts?|paragraphs?|the text|the document)\b|\b(?:the|this)\s+(?:\d{4}\s+)?letter\s+"
                          r"(?:says|said|states|stated|mentions|mentioned|notes|noted|claims|writes|wrote)\b|\baccording to (?:the|this) "
                          r"(?:\d{4} )?(?:letter|text)\b", re.I)
PRONOUN = re.compile(r'\b(?:it|its|they|their|that|this|those|these)\b', re.I)
DIGIT = re.compile(r'\d')
WORDS = re.compile(r'[a-z0-9]+')


# ------------------------------------------------------------------ exclusions
def checker_ids(checker, name):
    """Sorted passage ids a held-out checker excludes. Any failure refuses to run."""
    checker = Path(checker)
    if not checker.is_file():
        raise RuntimeError(f'{name} checker missing ({checker.name}); refusing to generate without the blind exclusions')
    result = subprocess.run([sys.executable, str(checker), '--excluded-passage-ids'], cwd=common.ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'{name} checker failed (exit {result.returncode}); refusing to generate')
    try:
        ids = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f'{name} checker did not print a JSON list; refusing to generate') from error
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        raise RuntimeError(f'{name} checker output is not a non-empty list of passage ids; refusing to generate')
    return sorted(set(ids))


def gold_ids(path):
    """Every gold_passage_ids / relevant_ids value in a JSON file (ids only, never question text). Returns (ids, sha256)."""
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
    blob = Path(path).read_bytes()
    walk(json.loads(blob))
    return ids, common.sha(blob)


def exclusion_union(full, *, base=(), v4=None, v5=None, v3=None, dev=None):
    """base + heldout_v3 + heldout_v4 + heldout_v5 + dev_cases ids, each with its +-1 neighbours. `full`: id -> doc (whole corpus).

    v3 and dev ids are real gold passages and must exist in the corpus. The v4/v5 checkers already add +-1 neighbours
    arithmetically, so a few of their ids can fall past a letter's last chunk: those are kept as ids but have no neighbours.
    """
    excluded = set(base)
    for ids in (v3, dev):
        excluded |= round4.with_neighbours(set(ids or ()), full)
    for ids in (v4, v5):
        ids = set(ids or ())
        excluded |= ids | round4.with_neighbours({i for i in ids if i in full}, full)
    return excluded


def load_inputs(out, *, teacher_path=DEFAULT_TEACHER, pilot=False, v4_checker=None, v5_checker=None):
    """Checkers run first: a failing checker refuses before any output is touched."""
    v4_ids = checker_ids(v4_checker or HELDOUT_V4_CHECKER, 'heldout_v4')
    v5_ids = checker_ids(v5_checker or HELDOUT_V5_CHECKER, 'heldout_v5')
    common.OUT, common.ROUND3, common.PILOT = Path(out), True, False
    docs, excluded, frozen, valid_years, identity = common.inputs()
    full = {d.id: d for d in common.load_chunks_as_docs(common.CORPUS)}
    v3_ids, v3_sha = round4.heldout_v3_passage_ids()
    dev_ids, dev_sha = gold_ids(common.ROOT / DEV_CASES)
    excluded = exclusion_union(full, base=excluded, v3=v3_ids, v4=v4_ids, v5=v5_ids, dev=dev_ids)
    docs = [d for d in docs if d.id not in excluded]
    identity = {**identity, 'style': 'v5', 'pilot': bool(pilot), 'teacher_id': str(teacher_path), 'excluded_passage_count': len(excluded),
                'heldout_v4_excluded_sha256': common.sha(json.dumps(v4_ids).encode()),
                'heldout_v5_excluded_sha256': common.sha(json.dumps(v5_ids).encode()),
                'frozen_set_hashes': {**identity['frozen_set_hashes'], round4.HELDOUT_V3: v3_sha, DEV_CASES: dev_sha}}
    return docs, excluded, frozen, valid_years, identity, full


# ------------------------------------------------------------------ serving path
def messages_from(prompt):
    system, user = split_prompt(prompt)
    return ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': user}]


def served_ids(hits):
    """Ids of every chunk whose text is in the served passages (anchor + merged neighbours, minus truncated ones)."""
    ids = set()
    for hit in hits:
        ids.add(hit.id)
        ids.update(hit.metadata.get('merged_ids') or ())
    return ids


def context_ids(hits, full):
    ids = set(served_ids(hits))
    for hit in hits:
        doc = full.get(hit.id)
        for field in ('previous_chunk_id', 'next_chunk_id'):
            if doc is not None and doc.metadata.get(field) in full:
                ids.add(doc.metadata[field])
    return ids


class Serving:
    """What /ask does for one question: Retriever.search (hybrid, rerank, top_k 8) -> neighbour merge -> fit_context_to_llm.

    The prompt comes out of ask_flow._prepare_ask itself; temporal questions are split per period exactly like
    generate_comparison_answer. Nothing here knows the source passage.
    """

    def __init__(self, retriever, docs_by_id):
        self.retriever, self.docs_by_id = retriever, docs_by_id

    def prompts(self, query, history=None):
        history = [dict(turn) for turn in (history or [])]
        request = AskRequest(query=query, history=history, expand_query=False)   # type: ignore[arg-type]
        previous = ask_flow._state
        ask_flow._state = {'llm': round4.LlamaStub(), 'retriever': self.retriever, 'docs_by_id': self.docs_by_id}
        try:
            _, hits, _, reranked, context, prompt, expanded = ask_flow._prepare_ask(request)
        finally:
            ask_flow._state = previous
        extras = ask_flow._extra_queries(request, expanded)
        base = {'retrieved_ids': [h.id for h in hits], 'reranked': reranked}
        periods = comparison_periods(request.query)
        if not periods:
            gate = bool(context) and assess_evidence(request.query, context, extra_queries=extras).sufficient
            return [{**base, 'period': None, 'query': request.query, 'hits': context, 'gate': gate,
                     'messages': messages_from(prompt) if context else []}]
        prompts = []
        for period in periods:
            label = period_label(period)
            subset = [hit for hit in context if _meta_matches(hit.metadata, period)]
            focus = f'{request.query} (focus: {label})'
            gate = bool(subset) and assess_evidence(request.query, subset, extra_queries=extras).sufficient
            prompts.append({**base, 'period': label, 'query': focus, 'hits': subset, 'gate': gate,
                            'messages': messages_from(build_cited_prompt(focus, subset, history=history)) if subset else []})
        return prompts


def load_serving():
    """The /ask retriever stack, built the way backend_app.startup builds it (faiss index must already exist)."""
    from config import EMBEDDING_DEVICE, EMBEDDING_MODEL_PRIMARY, VECTOR_BACKEND
    if VECTOR_BACKEND != 'faiss' or os.environ.get('HF_HUB_OFFLINE') != '1':
        raise RuntimeError('serving retrieval needs VECTOR_BACKEND=faiss and HF_HUB_OFFLINE=1')
    from src.retrieval.context import build_doc_lookup
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.retriever import Retriever
    from src.storage import get_vector_store, load_chunks_as_docs
    from src.storage.embeddings import BGEEmbedder
    from src.storage.index_manifest import ensure_index_identity
    path = ask_flow._resolve_chunks_path()
    docs = load_chunks_as_docs(path)
    embedder = BGEEmbedder(model_name=EMBEDDING_MODEL_PRIMARY, device=EMBEDDING_DEVICE)
    store = get_vector_store(backend=VECTOR_BACKEND, dim=embedder.dimension)
    if len(store) == 0:
        raise RuntimeError('faiss index is empty; this script never builds an index')
    ensure_index_identity(store, backend=VECTOR_BACKEND, corpus=path, docs=docs, model_name=embedder.model_name, dimension=embedder.dimension)
    return Serving(Retriever(vector_store=store, embedder=embedder, docs=docs, reranker=CrossEncoderReranker()), build_doc_lookup(docs))


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


def has_metric(doc, topic, cache):
    """True if `doc` has a sentence with the metric word AND a number (a number-bearing fact)."""
    key = (topic, doc.id)
    if key not in cache:
        cache[key] = any(topic in s.lower() and DIGIT.search(s) for s in evidence_sentences(doc.text))
    return cache[key]


def metric_pair(topic, docs, rng, cache):
    docs = list(docs)
    rng.shuffle(docs)
    first = None
    for doc in docs:
        if not has_metric(doc, topic, cache):
            continue
        if first is None:
            first = doc
        elif doc.metadata['year'] != first.metadata['year']:
            return first, doc
    return None


def build_plan(docs, valid_years, identity, quotas, oversample):
    rng = random.Random(common.SEED + 5)
    prose = [d for d in docs if len(d.text) >= 250 and sum(c.isalpha() for c in d.text) / len(d.text) > .60]
    by_split = {s: [d for d in prose if ('valid' if int(d.metadata['year']) in valid_years else 'train') == s] for s in ('train', 'valid')}
    for ds in by_split.values():
        rng.shuffle(ds)
    topics = defaultdict(list)
    for doc in prose:
        for topic in TOPIC_WORDS:
            if topic in doc.text.lower():
                topics[topic].append(doc)
    templates = unavailable_templates()
    rng.shuffle(templates)
    used, cursors, cache = set(), Counter(), {}

    def next_doc(split):
        while cursors[split] < len(by_split[split]):
            doc = by_split[split][cursors[split]]
            cursors[split] += 1
            if doc.id not in used:
                used.add(doc.id)
                return doc
        return None

    sizes = {'single': quotas['single'], 'multipart': quotas['multipart'], 'followup': quotas['followup'],
             'temporal': math.ceil(quotas['temporal'] / 2), 'refusal': quotas['refusal']}
    lists = {}
    for kind in sorted(KINDS, key=lambda k: k != 'temporal'):   # temporal needs the scarcest sources: allocate first
        jobs = []
        for index in range(math.ceil(sizes[kind] * oversample)):
            split = 'valid' if index % 10 == 9 else 'train'
            job: dict = {'id': f'v5:{kind}:{index:04d}', 'kind': kind, 'split': split, 'type': 'unanswerable' if kind == 'refusal' else 'answerable'}
            if kind == 'refusal':
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
                    pair = metric_pair(topic, ds, rng, cache)
                if pair is None:
                    break
                used.update(d.id for d in pair)
                job.update(topic=topic, source_id=pair[0].id, source_ids=[d.id for d in pair])
            else:
                doc = next_doc(split)
                if doc is None:
                    break
                job['source_id'] = doc.id
            jobs.append(job)
        lists[kind] = jobs
    keyed = sorted(((i + .5) / len(jobs), KINDS.index(kind), job) for kind, jobs in lists.items() for i, job in enumerate(jobs))
    return {'identity': identity, 'style': 'v5', 'prompt_revision': REVISION, 'quotas_at_plan': quotas, 'oversample': oversample,
            'counts': {kind: len(jobs) for kind, jobs in lists.items()}, 'jobs': [job for _, _, job in keyed]}


# ------------------------------------------------------------------ questions
QUESTION_SYSTEM = 'Write natural user questions as valid JSON only.'
QUESTION_RULES = {
    'single': 'Imagine a reader who has not seen the passage and wants to learn one specific fact it contains. Write ONE question for that fact.',
    'multipart': 'Imagine a reader who wants to learn two different things the passage tells. Write ONE question with TWO distinct parts '
                 '(joined naturally, e.g. with "and"), each answered by the passage.',
    'temporal': 'Two letters from different years each report a number about the same subject. Write ONE question that names BOTH '
                'years literally and asks what each letter reports about the subject. Never ask for a calculated change. '
                'Give exactly one part per year.',
    'followup': 'A reader just received the first answer below and now asks a SECOND, different question about the same subject whose answer is '
                'also in the passage but is NOT already stated in the first answer. The question must refer to the subject only with it, they, '
                'that, this or their, and must not name the subject.',
}
PARTS = {'single': 1, 'multipart': 2, 'temporal': 2, 'followup': 1}
QUESTION_RETRY = {
    'question_copies_passage': 'your question reused four or more consecutive words of the passage; rewrite it in entirely different words',
    'question_close_to_passage_sentence': 'your question is too similar to one passage sentence; rewrite it in your own words',
    'question_refers_to_passage': 'never mention the passage, the text or what the letter says; ask as a normal user would',
    'source_year_missing': 'the question must contain every source year literally',
    'invalid_key_terms': 'every part needs 1-3 key_terms copied from the passage',
    'invalid_question_parts': 'parts must have exactly the requested number of entries, each with ask, key_terms' ,
    'invalid_followup_pronoun': 'the question must refer to the subject with it, they, that, this or their',
}


def words(text):
    return WORDS.findall(text.lower())


def grams(tokens, n=4):
    return set(zip(*(tokens[i:] for i in range(n)))) if len(tokens) >= n else set()


def copy_reason(question, texts):
    """Rule 2: a question must be the user's own words, not the passage's."""
    if PASSAGE_TALK.search(question):
        return 'question_refers_to_passage'
    tokens = words(question)
    asked = grams(tokens)
    qset = set(tokens)
    for text in texts:
        if asked & grams(words(text)):
            return 'question_copies_passage'
        for sentence in evidence_sentences(text):
            sset = set(words(sentence))
            if sset and len(qset & sset) / len(qset | sset) > .35:
                return 'question_close_to_passage_sentence'
    return None


def question_prompt(kind, sources, first=None):
    lines = [QUESTION_RULES[kind],
             'Write the question in your own words: never copy four or more consecutive words from the passage, never mention "the passage", '
             '"the text" or what "the letter says". Ask a genuine interrogative question beginning What, Which, How, Why, Who or When. '
             'Use only the passage, no outside facts.']
    if kind in ('single', 'multipart'):
        lines.append(f'Mention the letter year ({sources[0]["year"]}) naturally in the question.')
    year_field = ', "year": 1995' if kind == 'temporal' else ''
    lines.append('For each thing the question asks give "ask" (a short noun phrase for what is requested) and "key_terms" (1-3 single words '
                 'that the passage itself uses for it). Return ONLY JSON: {"question":"...?","parts":[{"ask":"...","key_terms":["..."]' + year_field + '}]}')
    if first:
        lines.append(f'FIRST QUESTION: {first["question"]}\nFIRST ANSWER: {first["answer"]}')
    for source in sources:
        lines.append(f'PASSAGE (year {source["year"]}):\n{source["text"]}')
    return '\n\n'.join(lines)


def build_question(job, kind, sources, raw, metrics, frozen, known, history=None, first=None):
    """Strict checks on a passage-based question written by the teacher; returns (row, record)."""
    record = {'id': job['id'], 'kind': job['kind'], 'kept': 0, 'raw': raw, **metrics}
    obj = _json_object(raw)
    question = obj.get('question', '')
    if not isinstance(question, str) or not 20 <= len(question) <= 600 or not question.endswith('?'):
        record['reason'] = 'invalid_question'
        return None, record
    if not re.search(r'\b(what|which|how|why|who|when|where)\b', question, re.I):
        record['reason'] = 'not_an_interrogative_question'
        return None, record
    years = [str(s['year']) for s in sources]
    if kind == 'temporal' and not all(y in question for y in years):
        record['reason'] = 'source_year_missing'
        return None, record
    if kind in ('single', 'multipart') and years[0] not in question:
        question = f'In the {years[0]} letter, ' + question[:1].lower() + question[1:]
    if kind == 'followup' and not PRONOUN.search(question):
        record['reason'] = 'invalid_followup_pronoun'
        return None, record
    reason = copy_reason(question, [s['text'] for s in sources])
    if reason:
        record['reason'] = reason
        return None, record
    parts = obj.get('parts')
    if not isinstance(parts, list) or len(parts) != PARTS[kind] or not all(isinstance(p, dict) and isinstance(p.get('ask'), str) and p['ask'].strip() for p in parts):
        record['reason'] = 'invalid_question_parts'
        return None, record
    clean_parts, seen = [], set()
    by_year = {str(s['year']): s for s in sources}
    if kind == 'temporal' and {str(p.get('year')) for p in parts} != set(by_year):
        record['reason'] = 'invalid_question_parts'     # exactly one part per source year
        return None, record
    for index, part in enumerate(parts, 1):
        key_terms = part.get('key_terms')
        if not isinstance(key_terms, list) or not 1 <= len(key_terms) <= 3 or not all(isinstance(t, str) and t.strip() for t in key_terms):
            record['reason'] = 'invalid_key_terms'
            return None, record
        source = by_year[str(part.get('year'))] if kind == 'temporal' else sources[0]
        terms = common.content_terms(' '.join(key_terms))
        if not terms or not terms <= common.content_terms(source['text']) or (kind == 'multipart' and frozenset(terms) in seen):
            record['reason'] = 'invalid_key_terms'
            return None, record
        seen.add(frozenset(terms))
        clean_parts.append({'id': f'P{index}', 'ask': part['ask'].strip(), 'key_terms': [t.strip() for t in key_terms],
                            'source_id': source['id'], 'year': int(source['year']), 'needs_number': kind == 'temporal'})
    if first and common.content_terms(' '.join(t for p in clean_parts for t in p['key_terms'])) <= common.content_terms(first['answer']):
        record['reason'] = 'followup_repeats_first_answer'
        return None, record
    row = {k: v for k, v in job.items() if k not in {'template', 'topic'}}
    row.update(question=question, style='v5', origin='v5_new', parts=clean_parts, source_year=int(sources[0]['year']))
    if kind == 'temporal':
        row['topic'] = job.get('topic')
    if history:
        row['history'] = history
        row['first_question'] = first['question']
    if round3.fingerprint(row) in known or common.leakage_reasons(row, set(), frozen):
        record['reason'] = 'duplicate_or_frozen_question'
        return None, record
    record['kept'] = 1
    return row, record


# ------------------------------------------------------------------ answer evaluation
def sentence_table(text):
    """Per line: [(sentence, is_gap)]; a gap sentence is uncited and says something is not stated."""
    table = []
    for line in text.split('\n'):
        items = []
        for sentence in _split_original(line):
            body = BULLET.sub('', sentence.strip())
            items.append((sentence.strip(), not CITATION.search(body) and bool(GAP.search(body))))
        table.append(items)
    return table


def plain(sentence):
    return re.sub(r'\s+', ' ', CITATION.sub('', BULLET.sub('', sentence))).strip()


def evaluate_answer(raw, hits, parts, served, *, answerable=True):
    """Serving-validator filter + part coverage. Returns (reason or None, target, stats).

    The validator is run sentence by sentence with the serving defaults (it judges every sentence independently, so this
    equals one call on the whole answer). Gap sentences are not claims, so they are outside the 85% denominator.
    """
    clean = format_answer_markdown(strip_chat_artifacts(raw or ''))
    stats = {'sentences': 0, 'kept_sentences': 0, 'gap_sentences': 0, 'keep_rate': None}
    if not answerable:
        return (None if clean == REFUSAL_LINE else 'refusal_expected_teacher_answered'), clean, stats
    if not clean:
        return 'empty_answer', '', stats
    missing = [p for p in parts if p['source_id'] not in served]
    if clean == REFUSAL_LINE:
        return (None if parts and len(missing) == len(parts) else 'answerable_refusal'), clean, stats
    if REFUSAL_LINE in clean:
        return 'refusal_mixed_with_answer', clean, stats
    if round4.BANNED_OPENER.search(clean):
        return 'banned_filler', clean, stats
    table = sentence_table(clean)
    flat = [item for items in table for item in items]
    normal = [s for s, gap in flat if not gap]
    gaps = [s for s, gap in flat if gap]
    stats.update(sentences=len(normal), gap_sentences=len(gaps))
    if not normal and not gaps:
        return 'empty_answer', '', stats
    if not normal:
        return 'no_cited_sentence', clean, stats
    ok = {}
    for sentence in normal:
        citations = parse_citations(sentence, hits)
        valid = bool(citations) and all(c['passage_indices'] and not c['invalid_numbers'] for c in citations)
        verdict = validate_and_filter_answer(sentence, hits) if valid else None
        ok[sentence] = bool(verdict and verdict.safe_answer and not verdict.blocked_claims)
    kept = [s for s in normal if ok[s]]
    stats.update(kept_sentences=len(kept), keep_rate=len(kept) / len(normal))
    if len(kept) / len(normal) < KEEP_RATE:
        return 'validator_dropped_over_15pct', clean, stats
    if len(normal) + len(gaps) > 2 * max(1, len(parts)) + 1:
        return 'too_long', clean, stats
    if len({plain(s) for s in normal}) != len(normal):
        return 'repeated_sentence', clean, stats
    rows = []
    for items in table:
        line = ' '.join(s for s, gap in items if gap or ok.get(s))
        rows.append(line)
    target = re.sub(r'\n{3,}', '\n\n', '\n'.join(rows)).strip()
    for gap in gaps:
        gap_terms = common.content_terms(gap)
        matched = [p for p in parts if (lambda t: t and len(t & gap_terms) / len(t) >= .5)(common.content_terms(' '.join(p['key_terms'] + [p['ask']])))]
        if not matched:
            return 'unmatched_gap_sentence', clean, stats
        if all(p['source_id'] in served for p in matched):
            return 'gap_for_served_part', clean, stats
    for part in parts:
        terms = common.content_terms(' '.join(part['key_terms']))
        covered = any(terms <= common.content_terms(plain(s)) and (not part.get('needs_number') or DIGIT.search(plain(s))) for s in kept)
        if not covered and part['source_id'] not in served:
            gap_terms_needed = terms
            covered = any(len(gap_terms_needed & common.content_terms(g)) / len(gap_terms_needed) >= .5 for g in gaps) if gap_terms_needed else False
        if not covered:
            return 'part_not_covered', clean, stats
    return None, target, stats


ANSWER_RETRY = {
    'answerable_refusal': 'the passages are relevant to the question; answer the parts they support instead of refusing',
    'refusal_mixed_with_answer': 'either answer from the passages or give the refusal line alone, never both',
    'banned_filler': 'do not start sentences with Additionally, Furthermore, Moreover or Therefore',
    'no_cited_sentence': 'every sentence needs a [n] citation',
    'validator_dropped_over_15pct': 'more than 15% of your sentences were not supported by the passage they cite; each sentence must restate a single passage sentence in that passage\'s own words, one [n] each',
    'part_not_covered': 'every part of the question needs its own cited sentence that uses the passages\' own key words',
    'gap_for_served_part': 'a "not stated" sentence is only allowed for a part the passages really do not contain',
    'unmatched_gap_sentence': 'a "not stated" sentence must name the part of the question it refers to',
    'repeated_sentence': 'do not repeat a sentence',
    'too_long': 'write 1-2 sentences per part',
    'empty_answer': 'write the answer',
}


def retry_message(reason):
    return f'REJECTED ({reason}): {ANSWER_RETRY.get(reason, "follow the answer rules")}. Write the complete answer again; output only the answer.'


# ------------------------------------------------------------------ generator
def category(kind):
    return kind


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


class Generator:
    def __init__(self, teacher, serving, loaded, quotas, out):
        self.teacher, self.serving, self.quotas, self.out = teacher, serving, quotas, Path(out)
        self.docs, self.excluded, self.frozen, self.valid_years, self.identity, self.full = loaded
        self.lookup = {d.id: d for d in self.docs}
        self.rows = common.read_jsonl(self.out / 'examples.jsonl')
        self.accepted = {row['id'] for row in self.rows}
        self.attempts = common.read_jsonl(self.out / 'example_attempts.jsonl')
        self.done = {row['attempt_id'] for row in self.attempts}
        self.known = {round4.qkey(row) for row in self.rows}
        self.hashes = {round3.chat_hash(row) for row in self.rows}
        if self.rows:
            self.materialize()

    # -- bookkeeping
    def kept_per_category(self):
        return Counter(category(row['kind']) for row in self.rows)

    def quota_met(self, cat):
        return self.kept_per_category()[cat] >= self.quotas[cat]

    def all_met(self):
        return all(self.quota_met(cat) for cat in KINDS)

    def natural_miss_allowed(self):
        misses = sum(bool(row.get('natural_miss')) for row in self.rows)
        return (misses + 1) / (len(self.rows) + 1) <= NATURAL_MISS_SHARE

    def log(self, record):
        append(self.out / 'example_attempts.jsonl', record)
        self.attempts.append(record)
        self.done.add(record['attempt_id'])

    # -- questions
    def sources(self, ids):
        return [{'id': i, 'year': int(self.lookup[i].metadata['year']), 'text': self.lookup[i].text[:1800]} for i in ids]

    def write(self, job, kind, sources, *, history=None, first=None):
        """One teacher call to write the question and, if it is rejected, one retry naming the broken rule."""
        messages = [{'role': 'system', 'content': QUESTION_SYSTEM}, {'role': 'user', 'content': question_prompt(kind, sources, first)}]
        key = job['id'] + (':q2' if first else ':question')
        raw, metrics = self.teacher.generate(messages, key, max_tokens=420, temperature=.1)
        if metrics.get('finish_reason') == 'budget':
            return None, {'id': job['id'], 'kind': job['kind'], 'kept': 0, **metrics}
        row, record = build_question(job, kind, sources, raw, metrics, self.frozen, self.known, history, first)
        if row or record.get('reason') == 'duplicate_or_frozen_question':
            return row, record
        rule = QUESTION_RETRY.get(record['reason'], 'the question must be one interrogative sentence of 20-600 characters ending with ? and '
                                                      'the JSON must follow the requested shape')
        messages += [{'role': 'assistant', 'content': raw}, {'role': 'user', 'content': f'REJECTED ({record["reason"]}): {rule}. Return ONLY the corrected JSON object.'}]
        raw2, metrics2 = self.teacher.generate(messages, key + ':retry', max_tokens=420, temperature=.1)
        if metrics2.get('finish_reason') == 'budget':
            return None, {**record, **metrics2}
        row, second = build_question(job, kind, sources, raw2, metrics2, self.frozen, self.known, history, first)
        second.update(retry=True, first_reason=record['reason'], first_raw=raw)
        return row, second

    def synthesize(self, job):
        if job['kind'] == 'refusal':
            raw, metrics = self.teacher.generate(
                [{'role': 'system', 'content': QUESTION_SYSTEM},
                 {'role': 'user', 'content': 'Paraphrase this unavailable-evidence question without removing any date, name or qualifier. '
                                             'Return ONLY JSON: {"question":"...?"}.\n' + job['template']}], job['id'] + ':question', max_tokens=200, temperature=.1)
            if metrics.get('finish_reason') == 'budget':
                return None, {'id': job['id'], 'kind': 'refusal', 'kept': 0, **metrics}
            row, record = round3.build_question(job, [], raw, metrics, self.frozen, self.known)
            if row:
                row.update(style='v5', origin='v5_new', parts=[])
            return row, record
        sources = self.sources(job.get('source_ids') or [job['source_id']])
        if job['kind'] != 'followup':
            row, record = self.write(job, job['kind'], sources)
            return row, record
        # follow-up: a real first question, answered and kept through the same filter, becomes the history
        first_job = {**job, 'id': job['id'] + ':first'}
        first_q, record = self.write(first_job, 'single', sources)
        if not first_q:
            return None, {**record, 'id': job['id'], 'reason': 'first_question_' + str(record.get('reason'))}
        first_q['kind'] = 'single'
        outcome = self.first_turn(first_q)
        if outcome.get('status') == 'budget':
            return None, {'id': job['id'], 'kind': 'followup', 'kept': 0, 'finish_reason': 'budget'}
        record = {'id': job['id'], 'kind': 'followup', 'kept': 0, 'first_turn': {k: outcome.get(k) for k in ('reason', 'target', 'natural_miss')}}
        if outcome['reason']:
            record['reason'] = 'first_turn_' + outcome['reason']
            return None, record
        answer = outcome['target']
        history = [{'role': 'user', 'content': first_q['question']}, {'role': 'assistant', 'content': answer}]
        if not fact_bearing(answer):
            record['reason'] = 'first_turn_not_fact_bearing'
            return None, record
        row, second = self.write(job, 'followup', sources, history=history, first={'question': first_q['question'], 'answer': answer})
        second['first_turn'] = record['first_turn']
        return row, second

    def first_turn(self, question):
        prompts = self.serving.prompts(question['question'])
        prompt = prompts[0]
        outcome = self.attempt(question, prompt, question['id'])
        if outcome['status'] == 'done' and not outcome['reason'] and outcome['natural_miss']:
            outcome['reason'] = 'source_not_served'
        return outcome

    # -- answers
    def generation_messages(self, messages, answerable):
        generation = [dict(m) for m in messages]
        if answerable:
            generation[0]['content'] += STYLE_V5
        return generation

    def attempt(self, question, prompt, key):
        """Teacher answer(s) for one serving prompt -> outcome dict. No file is written here."""
        answerable = question['type'] == 'answerable'
        hits = prompt['hits']
        parts = [p for p in question.get('parts', []) if prompt['period'] is None or str(p['year']) == prompt['period']]
        served = served_ids(hits)
        out = {'status': 'done', 'reason': None, 'target': None, 'calls': 0, 'natural_miss': answerable and any(p['source_id'] not in served for p in parts),
               'served': sorted(served), 'tries': []}
        years = {int(h.metadata['year']) for h in hits}
        if not hits or not prompt['messages']:
            out['reason'] = 'no_context'
        elif context_ids(hits, self.full) & self.excluded:
            out['reason'] = 'frozen_passage_or_neighbor'
        elif any((y in self.valid_years) != (question['split'] == 'valid') for y in years):
            out['reason'] = 'cross_split_context'
        elif answerable and not prompt['gate']:
            out['reason'] = 'evidence_gate_insufficient'
        if out['reason']:
            return out
        generation = self.generation_messages(prompt['messages'], answerable)
        for attempt in range(2 if answerable else 1):
            raw, metrics = self.teacher.generate(generation, f'{key}:v5p{REVISION}:{attempt}', max_tokens=MAX_ANSWER_TOKENS, temperature=.1)
            if metrics['finish_reason'] == 'budget':
                out['status'] = 'budget'
                return out
            out['calls'] += 1
            if metrics['finish_reason'] == 'length':
                reason, target, stats = 'truncated_teacher_output', raw, {}
            else:
                reason, target, stats = evaluate_answer(raw, hits, parts, served, answerable=answerable)
            out['tries'].append({'raw': raw, 'reason': reason, **stats, **metrics})
            out.update(reason=reason, target=target, metrics=metrics, stats=stats)
            if reason is None or reason == 'truncated_teacher_output':
                break
            generation = generation + [{'role': 'assistant', 'content': raw}, {'role': 'user', 'content': retry_message(reason)}]
        return out

    def process(self, question):
        """Answer one question (one prompt, or one per period); returns the number of teacher calls spent."""
        spent, kind = 0, question['kind']
        try:
            prompts = self.serving.prompts(question['question'], question.get('history'))
        except ValueError as error:   # pydantic ValidationError is a ValueError
            self.log({'attempt_id': f"{question['id']}:v5p{REVISION}", 'id': question['id'], 'type': question['type'], 'kind': kind,
                      'kept': False, 'reason': 'invalid_serving_request', 'detail': str(error)[:200], 'seconds': 0.})
            return 0
        if kind == 'temporal':
            years = {str(p['year']) for p in question['parts']}
            if {p['period'] for p in prompts} != years or not all(p['gate'] and p['hits'] for p in prompts):
                self.log({'attempt_id': f"{question['id']}:v5p{REVISION}", 'id': question['id'], 'type': question['type'], 'kind': kind,
                          'kept': False, 'reason': 'period_gated_or_missing', 'seconds': 0.})
                return 0
        for prompt in prompts:
            row_id = question['id'] if prompt['period'] is None else f"{question['id']}:{prompt['period']}"
            attempt_id = f'{row_id}:v5p{REVISION}'
            if row_id in self.accepted or attempt_id in self.done:
                continue
            if self.quota_met(category(kind)):
                break
            hits = prompt['hits']
            base = {'attempt_id': attempt_id, 'id': row_id, 'type': question['type'], 'kind': kind, 'period': prompt['period'],
                    'passages': len(hits), 'prompt_chars': sum(len(m['content']) for m in prompt['messages']), 'gate_sufficient': prompt['gate']}
            parts = [p for p in question.get('parts', []) if prompt['period'] is None or str(p['year']) == prompt['period']]
            if question['type'] == 'answerable' and any(p['source_id'] not in served_ids(hits) for p in parts) and not self.natural_miss_allowed():
                self.log({**base, 'kept': False, 'reason': 'natural_miss_cap', 'seconds': 0., 'tokens': 0})
                continue
            outcome = self.attempt(question, prompt, row_id)
            if outcome['status'] == 'budget':
                return spent
            spent += outcome['calls']
            reason, target = outcome['reason'], outcome['target']
            row = {**question, 'id': row_id, 'period': prompt['period'], 'serving_query': prompt['query'],
                   'passage_ids': [h.id for h in hits], 'context_passage_ids': sorted(context_ids(hits, self.full)),
                   'served_ids': outcome['served'], 'passages': len(hits), 'prompt_chars': base['prompt_chars'], 'gate_sufficient': prompt['gate'],
                   'natural_miss': bool(outcome['natural_miss']), 'source_in_top8': all(p['source_id'] in prompt['retrieved_ids'] for p in parts),
                   'messages': prompt['messages'] + [{'role': 'assistant', 'content': target or ''}]}
            if reason is None:
                reason = ','.join(common.leakage_reasons(row, self.excluded, self.frozen)) or None
            if reason is None and (round4.qkey(row) in self.known or round3.chat_hash(row) in self.hashes):
                reason = 'duplicate_chat_or_question'
            metrics = outcome.get('metrics', {})
            self.log({**base, 'kept': reason is None, 'reason': reason, 'natural_miss': row['natural_miss'], 'retried': len(outcome['tries']) > 1,
                      'tries': outcome['tries'], 'seconds': sum(t.get('seconds', 0.) for t in outcome['tries']),
                      'tokens': sum(t.get('tokens', 0) for t in outcome['tries']), **outcome.get('stats', {}), 'finish_reason': metrics.get('finish_reason')})
            if reason is None:
                row['teacher_metrics'] = metrics
                append(self.out / 'examples.jsonl', row)
                self.rows.append(row)
                self.accepted.add(row_id)
                self.known.add(round4.qkey(row))
                self.hashes.add(round3.chat_hash(row))
                if len(self.rows) % 5 == 0:
                    self.materialize()
            print(f'example={row_id} kind={kind} kept={reason is None} reason={reason} natural_miss={row["natural_miss"]} kept_total={len(self.rows)}', flush=True)
        return spent

    # -- outputs
    def materialize(self):
        selected, hashes, seen = [], set(), set()
        for row in self.rows:
            key, question_key = round3.chat_hash(row), round4.qkey(row)
            if key in hashes or question_key in seen:
                continue
            selected.append(row)
            hashes.add(key)
            seen.add(question_key)
        order = lambda r: common.sha(f'{common.SEED}:v5:{r["id"]}'.encode())
        self.out.mkdir(parents=True, exist_ok=True)
        for split in ('train', 'valid'):
            rows = sorted([r for r in selected if r['split'] == split], key=order)
            temporary = self.out / f'{split}.jsonl.tmp'
            temporary.write_text(''.join(json.dumps({'messages': r['messages']}, ensure_ascii=False) + '\n' for r in rows))
            temporary.replace(self.out / f'{split}.jsonl')
        selected.sort(key=lambda r: (r['split'], order(r)))   # metadata order == delivered split order
        temporary = self.out / 'split_metadata.jsonl.tmp'
        temporary.write_text(''.join(json.dumps({k: v for k, v in r.items() if k not in {'messages', 'teacher_metrics'}}, ensure_ascii=False) + '\n' for r in selected))
        temporary.replace(self.out / 'split_metadata.jsonl')
        return selected

    def finish(self, pilot):
        selected = self.materialize()
        kept = Counter(category(r['kind']) for r in selected)
        chars = sorted(r['prompt_chars'] for r in selected)
        manifest = {**self.identity, 'target_new_examples': sum(self.quotas.values()), 'quotas': self.quotas, 'kept_per_category': dict(kept),
                    'target_met': all(kept[c] >= self.quotas[c] for c in KINDS),
                    'counts_per_split': dict(Counter(r['split'] for r in selected)),
                    'counts_per_split_kind': {s: dict(Counter(r['kind'] for r in selected if r['split'] == s)) for s in ('train', 'valid')},
                    'valid_years': sorted(self.valid_years),
                    'natural_miss_share': sum(bool(r.get('natural_miss')) for r in selected) / max(1, len(selected)),
                    'refusal_share': sum(r['type'] != 'answerable' for r in selected) / max(1, len(selected)),
                    'refusals_gate_sufficient': sum(r['type'] != 'answerable' and r.get('gate_sufficient', False) for r in selected),
                    'source_in_top8_share': sum(bool(r.get('source_in_top8')) for r in selected if r['type'] == 'answerable')
                    / max(1, sum(r['type'] == 'answerable' for r in selected)),
                    'passages_per_prompt': dict(Counter(r['passages'] for r in selected)),
                    'prompt_chars_max': chars[-1] if chars else None, 'prompt_chars_p95': chars[int(.95 * (len(chars) - 1))] if chars else None,
                    'settings': {**round4.serving_settings(), 'temperature': .1, 'teacher_max_tokens': MAX_ANSWER_TOKENS, 'prompt_revision': REVISION,
                                 'keep_rate_min': KEEP_RATE, 'natural_miss_cap': NATURAL_MISS_SHARE,
                                 'retrieval': 'ask_flow._prepare_ask: hybrid + bge-reranker-v2-m3, top_k 8, neighbour merge, fit_context_to_llm, faiss',
                                 'split_strategy': 'seeded source-year grouping (~10% valid years); rows whose served context crosses the split are rejected'},
                    'heldout_exclusions': 'dev, heldout_v1..v3 (ids), heldout_v4 + heldout_v5 (checkers), grounded_dev_v1/confirm dev_cases; all with +-1 neighbours',
                    'rejection_reasons': dict(Counter(r['reason'] for r in self.attempts if r.get('reason')))}
        common.atomic_json(self.out / 'manifest.json', manifest)
        print('MIX ' + json.dumps(manifest['kept_per_category']), flush=True)
        return manifest


def fact_bearing(answer):
    """A usable first-turn answer: cited, no refusal or gap sentence, and either a number or a real statement."""
    if not answer or answer == REFUSAL_LINE or not CITATION.search(answer):
        return False
    flat = [item for items in sentence_table(answer) for item in items]
    return not any(gap for _, gap in flat) and any(DIGIT.search(plain(s)) or len(plain(s).split()) >= 8 for s, _ in flat)


# ------------------------------------------------------------------ dry run
def proxy_question(job, lookup):
    """Dry-run stand-in for the teacher-written question (same year/keyword shape; never used for training)."""
    if job['kind'] == 'refusal':
        return {**job, 'question': job['template']}
    docs = [lookup[i] for i in (job.get('source_ids') or [job['source_id']])]
    top = lambda doc, n: ', '.join(sorted(common.content_terms(doc.text[:600]), key=len, reverse=True)[:n])
    row = {k: v for k, v in job.items() if k not in {'template', 'topic'}}
    year = docs[0].metadata['year']
    if job['kind'] == 'temporal':
        question = f"How do the {docs[0].metadata['year']} and {docs[1].metadata['year']} letters describe {job['topic']}?"
    elif job['kind'] == 'multipart':
        question = f"In the {year} letter, what does Berkshire say about {top(docs[0], 2)} and why, regarding {top(docs[0], 3)}?"
    elif job['kind'] == 'followup':
        row['history'] = [{'role': 'user', 'content': f"What did Berkshire say about {top(docs[0], 2)} in {year}?"},
                          {'role': 'assistant', 'content': re.sub(r'\s+', ' ', docs[0].text[:300]).strip() + ' [1]'}]
        question = 'What did Buffett say about it and how did they handle that?'
    else:
        question = f"In the {year} letter, what does Berkshire say about {top(docs[0], 3)}?"
    row['question'] = question
    return row


def dry_run(count, args, *, serving=None, loaded=None, out=None):
    out = Path(out) if out else Path(tempfile.mkdtemp(prefix='ftv5_dry_'))
    teacher_path = os.environ.get('FT_TEACHER', DEFAULT_TEACHER)
    loaded = loaded or load_inputs(out, teacher_path=teacher_path)
    docs, excluded, _, valid_years, identity, _ = loaded
    print(f'dry-run: {len(docs)} usable source passages, {len(excluded)} excluded passage ids (union of v1-v5 + dev_cases, +-1 neighbours)')
    quotas = scale_mix(parse_mix(args.mix), args.target)
    plan = build_plan(docs, valid_years, identity, quotas, args.oversample)
    by_kind = defaultdict(list)
    for job in plan['jobs']:
        by_kind[job['kind']].append(job)
    share = {kind: len(jobs) for kind, jobs in by_kind.items()}
    chosen = []
    for kind, jobs in by_kind.items():
        take = max(1, round(count * share[kind] / sum(share.values())))
        chosen += random.Random(common.SEED).sample(jobs, min(take, len(jobs)))
    name, tokens = round4.token_counter(teacher_path)
    serving = serving or load_serving()
    lookup = {d.id: d for d in docs}
    stats = defaultdict(lambda: {'tokens': [], 'passages': [], 'skipped': 0, 'gate': 0, 'served': 0})
    for job in chosen:
        question = proxy_question(job, lookup)
        for prompt in serving.prompts(question['question'], question.get('history')):
            entry = stats[job['kind']]
            if not prompt['messages']:
                entry['skipped'] += 1
                continue
            entry['tokens'].append(tokens(prompt['messages']))
            entry['passages'].append(len(prompt['hits']))
            entry['gate'] += bool(prompt['gate'])
            if job['kind'] != 'refusal':
                ids = served_ids(prompt['hits'])
                wanted = job['source_ids'] if job['kind'] == 'temporal' else [job['source_id']]
                entry['served'] += any(i in ids for i in wanted) if job['kind'] == 'temporal' else wanted[0] in ids
    print(f'dry-run: {len(chosen)} sampled jobs, token counter={name}, proxy questions (no teacher); prompt tokens exclude the answer')
    print(f'{"kind":10} {"prompts":>7} {"skipped":>7} {"tok_p50":>8} {"tok_p95":>8} {"tok_max":>8} {"passages(min/mean/max)":>24} {"gate_ok":>8} {"source_served":>14}')
    for kind in KINDS:
        entry = stats.get(kind)
        if not entry:
            continue
        n = len(entry['tokens'])
        if not n:
            print(f'{kind:10} {0:>7} {entry["skipped"]:>7}')
            continue
        passages = entry['passages']
        served = 'n/a' if kind == 'refusal' else f'{entry["served"]:>5}/{n:<2}'
        print(f'{kind:10} {n:>7} {entry["skipped"]:>7} {round4.quantile(entry["tokens"], .5):>8} {round4.quantile(entry["tokens"], .95):>8} {max(entry["tokens"]):>8} '
              f'{min(passages):>8}/{sum(passages) / n:.1f}/{max(passages):<2}      {entry["gate"]:>5}/{n:<2} {served:>14}')
    everything = [t for e in stats.values() for t in e['tokens']]
    if everything:
        print(f'ALL: prompts={len(everything)} p50={round4.quantile(everything, .5)} p95={round4.quantile(everything, .95)} max={max(everything)} '
              f'(add up to ~{MAX_ANSWER_TOKENS} answer tokens for MAX_SEQ)')
    return stats


# ------------------------------------------------------------------ run
def run(args, parser):
    pilot = args.pilot is not None or os.environ.get('PILOT') == '1'
    if args.max_minutes is not None and args.max_minutes <= 0:
        parser.error('--max-minutes must be positive')
    if pilot and args.split_only:
        parser.error('Pilot is generation-only')
    teacher_path = os.environ.get('FT_TEACHER', DEFAULT_TEACHER)
    out = common.ROOT / (PILOT_DIR if pilot else FINAL_DIR)
    loaded = load_inputs(out, teacher_path=teacher_path, pilot=pilot)   # raises if a held-out checker fails
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'generation.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    docs, _, frozen, valid_years, identity, _ = loaded
    quotas = scale_mix(parse_mix(args.mix), args.target)
    path = out / 'question_plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        assert plan['identity'] == identity, 'Round-5 input identity changed; refusing resume'
    else:
        plan = build_plan(docs, valid_years, identity, quotas, args.oversample)
        common.atomic_json(path, plan)
        print(f'v5 plan: {plan["counts"]} pilot={pilot}', flush=True)
    if args.split_only:
        Generator(None, None, loaded, quotas, out).finish(pilot)
        return
    started = time.monotonic()
    deadline = started + args.max_minutes * 60 if args.max_minutes else None
    teacher = common.Teacher(deadline=deadline, path=teacher_path)
    generator = Generator(teacher, load_serving(), loaded, quotas, out)
    initial = len(generator.attempts)
    processed = 0

    def stop():
        return ((deadline is not None and time.monotonic() >= deadline) or generator.all_met()
                or (args.pilot is not None and processed >= args.pilot))

    for question in common.read_jsonl(out / 'questions.jsonl'):     # resume synthesized-but-unanswered questions
        if stop():
            break
        processed += generator.process(question)
    done = {r['id'] for r in common.read_jsonl(out / 'question_attempts.jsonl')}
    generator.known |= {round3.fingerprint(r) for r in common.read_jsonl(out / 'questions.jsonl')}
    for job in plan['jobs']:
        if stop():
            break
        if job['id'] in done or generator.quota_met(category(job['kind'])):
            continue
        question, record = generator.synthesize(job)
        if record.get('finish_reason') == 'budget':
            break
        append(out / 'question_attempts.jsonl', record)
        if question:
            append(out / 'questions.jsonl', question)
            generator.known.add(round3.fingerprint(question))
            processed += generator.process(question)
        print(f'question_job={job["id"]} kind={job["kind"]} kept={record["kept"]} reason={record.get("reason")}', flush=True)
    manifest = generator.finish(pilot)
    attempts = generator.attempts[initial:] if not pilot else generator.attempts
    answered = [a for a in attempts if a.get('tries')]
    summary = {'pilot': pilot, 'new_kept': sum(a['kept'] for a in attempts), 'teacher_answers': len(answered),
               'keep_rate': sum(a['kept'] for a in answered) / max(1, len(answered)), 'teacher_seconds': sum(a.get('seconds', 0.) for a in attempts),
               'wall_seconds': time.monotonic() - started, 'kept_per_category': manifest['kept_per_category'],
               'counts_per_split_kind': manifest['counts_per_split_kind'], 'natural_miss_share': manifest['natural_miss_share'],
               'rejection_reasons': dict(Counter(a['reason'] for a in attempts if a.get('reason'))), 'output': str(out)}
    common.atomic_json(out / 'generation_summary.json', summary)
    print(('PILOT ' if pilot else 'RUN ') + json.dumps(summary), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', type=int, metavar='N', help='no teacher: print prompt token stats per kind for N sampled jobs (serving retrieval, proxy questions)')
    parser.add_argument('--pilot', type=int, nargs='?', const=50, metavar='N', help='answer at most N questions into a throwaway dir (models/ft/round5_pilot)')
    parser.add_argument('--max-minutes', type=float, help='stop teacher work after this wall-clock time')
    parser.add_argument('--target', type=int, help='total new kept rows; scales the mix proportionally')
    parser.add_argument('--mix', help='per-kind targets, e.g. multipart=300,temporal=150,followup=120,single=300,refusal=100')
    parser.add_argument('--oversample', type=float, default=4., help='plan size multiple of each quota (plan is fixed once written)')
    parser.add_argument('--split-only', action='store_true', help='rewrite train/valid/metadata/manifest from existing examples; no generation')
    args = parser.parse_args(argv)
    if args.dry_run is not None:
        return dry_run(args.dry_run, args)
    return run(args, parser)


if __name__ == '__main__':
    main()
