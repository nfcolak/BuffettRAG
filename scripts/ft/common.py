"""Shared offline generation, immutable input identity, and leakage protection."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['PYTHON_DOTENV_DISABLED'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from src.vector_store import load_chunks_as_docs
from src.retrieval.bm25 import _SEARCH_STOPWORDS, _search_tokens

SEED = 13
CORPUS = ROOT / 'data/processed/chunks_v3_paragraph.jsonl'
OUT = ROOT / 'data/ft'
TEACHER = Path('/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/models/teacher-qwen2.5-7b-mlx4')
DEV = 'data/evaluation/answer_quality_program/answer_benchmark_v3.json'
HELDOUT = 'agent/fix-eval:data/evaluation/heldout_v1/answer_benchmark_heldout_v1.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    if not path.exists():
        return []
    result = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            # Recover only a torn final write, never silently skip interior damage.
            path.write_text('\n'.join(lines[:i]) + ('\n' if i else ''))
    return result


def append(path, row):
    OUT.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def token_set(text):
    return set(re.findall(r'[a-z0-9]+', text.casefold()))


def content_terms(text):
    return set(_search_tokens(text)) - _SEARCH_STOPWORDS - {'hathaway', 'warren', 'shareholders', 'question', 'according'}


def near_frozen(question, frozen_questions):
    terms = token_set(question)
    return any(len(terms & other) / max(1, len(terms | other)) >= .5 for other in frozen_questions)


def inputs():
    docs = load_chunks_as_docs(CORPUS)
    lookup = {doc.id: doc for doc in docs}
    snapshot = OUT / 'frozen_heldout_v1.json'
    if snapshot.exists():
        heldout_blob = snapshot.read_bytes()
    else:
        result = subprocess.run(['git', 'show', HELDOUT], cwd=ROOT, capture_output=True)
        if result.returncode:
            # The integrator can delete its branch after merging it into main.
            result = subprocess.run(['git', 'show', 'main:' + HELDOUT.split(':', 1)[1]], cwd=ROOT, capture_output=True, check=True)
        heldout_blob = result.stdout
        plan_path = OUT / 'question_plan.json'
        if plan_path.exists():
            expected = json.loads(plan_path.read_text())['identity']['frozen_set_hashes'][HELDOUT]
            if sha(heldout_blob) != expected:
                raise RuntimeError('Merged held-out bytes differ from the originally frozen branch.')
        OUT.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(heldout_blob)
    blobs = {DEV: (ROOT / DEV).read_bytes(), HELDOUT: heldout_blob}
    excluded, questions = set(), []
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {'gold_passage_ids', 'relevant_ids'}:
                    excluded.update(value)
                if key in {'query', 'question'} and isinstance(value, str):
                    questions.append(token_set(value))
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    for blob in blobs.values():
        walk(json.loads(blob))
    gold = set(excluded)
    for ident in gold:
        if ident not in lookup:
            raise ValueError(f'Frozen passage not in corpus: {ident}')
        for key in ('previous_chunk_id', 'next_chunk_id'):
            neighbor = lookup[ident].metadata.get(key)
            if neighbor:
                excluded.add(neighbor)
    years = sorted({int(doc.metadata['year']) for doc in docs})
    assert len(years) == 48, years
    valid_years = set(random.Random(SEED).sample(years, 5))
    safe = [doc for doc in docs if doc.id not in excluded]
    identity = {'corpus_sha256': sha(CORPUS.read_bytes()), 'frozen_set_hashes': {key: sha(blob) for key, blob in blobs.items()}, 'seed': SEED, 'teacher_id': str(TEACHER), 'teacher_license': 'Apache-2.0', 'excluded_passage_count': len(excluded), 'validation_years': sorted(valid_years)}
    return safe, excluded, questions, valid_years, identity


def split_for(year=None, key=''):
    if year is not None:
        years = sorted({int(doc.metadata['year']) for doc in load_chunks_as_docs(CORPUS)})
        return 'valid' if int(year) in set(random.Random(SEED).sample(years, 5)) else 'train'
    return 'valid' if int(sha(f'{SEED}:{key}'.encode())[:8], 16) % 10 == 0 else 'train'


class Teacher:
    def __init__(self):
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        if not TEACHER.is_dir():
            raise RuntimeError(f'Teacher path missing: {TEACHER}')
        self.model, self.tokenizer = load(str(TEACHER), trust_remote_code=False)
        print(f'teacher_loaded={TEACHER}', flush=True)

    def generate(self, messages, key, max_tokens=350, temperature=.2):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        self.mx.random.seed(int(sha(f'{SEED}:{key}'.encode())[:8], 16))
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        started = time.monotonic()
        text, last, refusal_stopped = '', None, False
        from src.generation.prompt import REFUSAL_LINE
        for response in stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temperature)):
            text += response.text
            last = response
            # A contract-defined stop sequence, not post-hoc target replacement:
            # the teacher has actually generated the entire exact refusal line.
            # Stop before it can append forbidden citations or an explanation.
            if text.strip() == REFUSAL_LINE:
                refusal_stopped = True
                break
        self.mx.clear_cache()
        return text.strip(), {'tokens': last.generation_tokens if last else 0, 'seconds': time.monotonic() - started, 'generation_tps': last.generation_tps if last else 0., 'finish_reason': 'refusal_stop_sequence' if refusal_stopped else (last.finish_reason if last else None)}


def parse_questions(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in '[{':
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        values = obj.get('questions', []) if isinstance(obj, dict) else obj
        if isinstance(values, list):
            clean = []
            for value in values:
                if isinstance(value, dict):
                    value = value.get('question')
                if isinstance(value, str):
                    value = re.sub(r'\s+', ' ', value).strip()
                    if 20 <= len(value) <= 350 and value.endswith('?'):
                        clean.append(value)
            if clean:
                return list(dict.fromkeys(clean))[:2]
    return []
