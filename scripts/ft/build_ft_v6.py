"""ft_v6 generator: completeness training data (many-obligation questions, verbatim multi-sentence targets) for a retrained 7B.

Three stages (VECTOR_BACKEND=faiss HF_HUB_OFFLINE=1 PYTHON_DOTENV_DISABLED=1, LLM_MODEL_PATH / LLM_CONTEXT_PASSAGES unset):
  PY scripts/ft/build_ft_v6.py plan --n 2600 --seed 7 --out data/ft_v6/jobs.jsonl         # no LLM: sample jobs + the exact teacher messages
  (teacher: Kaggle kernel reads jobs.jsonl, writes teacher_out.jsonl -- see the pv7/kaggle-teacher kit)
  PY scripts/ft/build_ft_v6.py assemble --jobs data/ft_v6/jobs.jsonl --teacher teacher_out.jsonl   # no LLM: rows, checks, splits, manifest
  PY scripts/ft/build_ft_v6.py teach-local --pilot 20 [--kinds multipart,refusal] [--max-minutes M]   # today's MLX path (small smoke runs)
  PY scripts/ft/build_ft_v6.py teach-local [--target 1760] [--max-minutes M]                          # resumable local run -> data/ft_v6/{train,valid}.jsonl
  PY scripts/ft/build_ft_v6.py verify data/ft_v6/pilot.jsonl                                           # offline re-check of every row (old `--verify` still works)

Design (differs from round5, whose teacher WROTE the answer and lost 60-75% of its rows to validator/paraphrase rejects):
  * The 14B teacher only WRITES THE QUESTION and says which passage sentences answer each part (3-5 parts; 1-2 adjacent sentences each).
  * The target is built deterministically: those sentences verbatim, one part after another, each sentence ending with its [n],
    numbered against the SERVING context (real hybrid retrieval, 8 passages, prompt version PROMPT_VERSION). A source passage the retrieval did
    not serve is put into a random slot replacing a served passage; this is recorded in `injected`.
  * Jobs come from Berkshire passages (corpus v3) that are not gold/relevant (+-1 neighbour) in any spent evaluation set; partnership
    passages (bpl*) are never used as sources and a row whose SERVED context holds one is rejected (counted in the manifest).
    Spent-set passages may still appear in the served context as distractors (only sources are excluded; --strict-context rejects them).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import glob
import gzip
import json
import os
from pathlib import Path
import random
import re
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402  (sets offline env flags, puts the repo root on sys.path)
import ft_v7_checks as C  # noqa: E402
import round4  # noqa: E402
import round5  # noqa: E402
from src.evaluation.claim_validator import _split_original, evidence_sentences, validate_and_filter_answer  # noqa: E402
from src.generation.prompt import REFUSAL_LINE, build_cited_prompt, parse_citations, system_prompt  # noqa: E402
from src.retrieval.context import truncate_around_anchor  # noqa: E402
from src.storage import SearchHit  # noqa: E402

ROOT = common.ROOT
PROMPT_VERSION = 'v3'                # ft_v7 / FT-r4 (Berkshire-only corpus); ft_v6 rows carry 'v2' in their prompt_version field
OUT_DIR = 'data/ft_v7'
MIN_REFUSAL_SHARE = 0.12             # ft_v6 had 121/717 (17%); repo assemble once kept 19 (2%) because of a too-strict term check
DEFAULT_TEACHER = round4.DEFAULT_TEACHER
MAX_SEQ = 6144                       # colab/train_7b_lora.ipynb (8-passage rows)
DEFAULT_PASSAGES = 8                 # served context size of the default (no --passages-mix) rows
MAX_SEQ_MIX = 12288                  # row length cap with --passages-mix (colab/train_7b_lora_r3.ipynb cap)
KINDS = ('multipart', 'single', 'follow_up', 'temporal', 'refusal')
SHARE = {'multipart': .55, 'single': .10, 'follow_up': .10, 'temporal': .10, 'refusal': .15}
PARTS = {'multipart': (3, 5), 'single': (2, 3), 'follow_up': (2, 3), 'temporal': (3, 5)}
RUN_LEN = {'multipart': (2, 3), 'single': (1, 2), 'follow_up': (1, 2), 'temporal': (2, 3), 'refusal': (1, 1)}
MIN_SENTENCES = {'multipart': 7, 'single': 3, 'follow_up': 5, 'temporal': 7}
BERKSHIRE_ID = re.compile(r'^(?:19|20)\d{2}_p\d{4}$')
ID_TEXT = re.compile(r'\b(?:19|20)\d{2}_p\d{4}\b')
OLD_ID_TEXT = re.compile(r'\b(?:19|20)\d{2}_\d+\b')
DEPENDENT_START = C.DEPENDENT_START      # ft_v6 openers + the connectives of target-audit a12.py (Again, Clearly, Therefore, Consequently, ...)
QUESTION_OVERLAP_MAX = C.QUESTION_OVERLAP_MAX
MIN_OBLIGATION_WORDS = C.MIN_OBLIGATION_WORDS
PASSAGE_TALK = round5.PASSAGE_TALK
PRONOUN = round5.PRONOUN
CITATION_END = re.compile(r'\s*\[(\d+(?:\s*,\s*\d+)*)\]\s*$')
PASSAGE_HEADER = re.compile(r'(?m)^\[(\d+)\] \(year=([^,\n]*), source=([^)\n]*)\)\n')
WORD = re.compile(r'[a-z0-9]+')
MAX_TEACHER_TOKENS = 700
TEMPERATURE = .1
KAGGLE_TEACHER = 'Qwen/Qwen2.5-32B-Instruct-AWQ'
EXPECT = {'multipart': 'plan', 'temporal': 'plan', 'single': 'plan', 'follow_up': 'follow_up', 'refusal': 'refusal'}   # teacher JSON shape per kind
PARTNERSHIP_REASONS = ('partnership_in_context', 'partnership_passage')
_PLAN_EXAMPLE = (
    'WORKED EXAMPLE (other passages). Sentences: P1.S1 "Berkshire repurchased 120,000 shares during 1995." P1.S2 "The buyback was funded from retained earnings." '
    'P1.S3 "This explains why float grew." P1.S4 "Float reached $3.4 billion." P1.S5 "GEICO added 400,000 policyholders." '
    'Good: {"question": "In 1995, how many shares did Berkshire repurchase, how big did float get and what happened at GEICO?", "parts": '
    '[{"ask": "shares repurchased", "passage": 1, "sentences": [1]}, {"ask": "float", "passage": 1, "sentences": [4]}, '
    '{"ask": "GEICO policyholders", "passage": 1, "sentences": [5]}]}. Bad: a part with sentences [3] (it starts with "This" and needs S2), '
    'and fewer than the required number of parts. Count your parts before you answer: they MUST number at least the minimum asked for.\n')
WORKED_EXAMPLES: dict = {'multipart': _PLAN_EXAMPLE, 'temporal': _PLAN_EXAMPLE}   # added after the first pilot (multipart/temporal keep rate 0/3)


def norm(text):
    return re.sub(r'\s+', ' ', text).strip()


def is_partnership(ident_or_source):
    """A Buffett Partnership passage id (bpl<YYYY>_p<NNNN>) or its source file (buffett_partnership_<YYYY>.txt)."""
    text = str(ident_or_source or '')
    return text.startswith('bpl') or 'partnership' in text.lower()


def served_has_partnership(hits):
    return any(is_partnership(i) for i in round5.served_ids(hits)) or any(
        str(h.metadata.get('collection', '')) == 'partnership' or is_partnership(h.metadata.get('source_file')) for h in hits)


def words(text):
    return WORD.findall(text.lower())


def parse_passages_mix(text):
    """'8,12,16' -> [8, 12, 16] (a repeated size is weighted); None/'' -> [DEFAULT_PASSAGES]."""
    if not text:
        return [DEFAULT_PASSAGES]
    try:
        mix = [int(part) for part in str(text).split(',') if part.strip()]
    except ValueError:
        raise ValueError(f'bad --passages-mix {text!r}; expected integers like 8,12,16') from None
    if not mix or any(not 1 <= k <= 32 for k in mix):
        raise ValueError(f'bad --passages-mix {text!r}; sizes must be 1..32')
    return mix


def pick_k(job_id, mix):
    """Served context size of one row: a deterministic hash of the job id picks from `mix`."""
    mix = list(mix)
    return mix[int(common.sha(f'{common.SEED}:k:{job_id}'.encode())[:8], 16) % len(mix)]


@contextmanager
def serving_passages(k):
    """Serve with LLM_CONTEXT_PASSAGES=k (ask_flow reads it at call time), restoring the caller's value afterwards."""
    old = os.environ.get('LLM_CONTEXT_PASSAGES')
    os.environ['LLM_CONTEXT_PASSAGES'] = str(k)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop('LLM_CONTEXT_PASSAGES', None)
        else:
            os.environ['LLM_CONTEXT_PASSAGES'] = old


# ------------------------------------------------------------------ exclusions
def _text_ids(text, old_map):
    ids = set(ID_TEXT.findall(text))
    for old in set(OLD_ID_TEXT.findall(text)):
        ids.update(old_map.get(old, ()))
    return ids


def old_id_map():
    mapping = {}
    for line in (ROOT / 'data/processed/chunk_id_map_v2_to_v3.jsonl').read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            mapping[row['old_id']] = set(row['new_ids'])
    return mapping


def gold_set_ids(docs):
    """gold_set.py names years and keywords, not passages: exclude every passage of a target year that holds the answer keywords."""
    from src.evaluation.gold_set import GOLD_QUERIES
    ids = set()
    for query in GOLD_QUERIES:
        for doc in docs:
            if int(doc.metadata['year']) not in query.target_years:
                continue
            low = doc.text.lower()
            if all(k.lower() in low for k in query.answer_keywords) or any(' ' in k and k.lower() in low for k in query.answer_keywords):
                ids.add(doc.id)
    return ids


def exclusion_sets(full, *, with_checkers=True):
    """(excluded ids incl. +-1 neighbours, per-source counts). `full`: id -> doc of the whole corpus."""
    old = old_id_map()
    evaluation = ROOT / 'data/evaluation'
    files = {'heldout_v1': [evaluation / 'heldout_v1/answer_benchmark_heldout_v1.json'],
             **{f'heldout_v{n}': sorted(Path(p) for p in glob.glob(str(evaluation / f'heldout_v{n}/answer_benchmark_heldout_v{n}*.json')))
                for n in range(2, 7)},
             'dev_cases': [evaluation / 'grounded_dev_v1/confirm/dev_cases.json'],
             'answer_benchmark_v3': [evaluation / 'answer_quality_program/answer_benchmark_v3.json'],
             'hard_negatives_v2': [evaluation / 'answer_quality_program/hard_negatives_v2.json'],
             'hard_negatives_v3': [evaluation / 'answer_quality_program/hard_negatives_v3.json'],
             'hard_negatives_v4': [evaluation / 'hard_negatives_v4/hard_negatives_v4.json']}
    sets = {}
    for name, paths in files.items():
        if not paths or not all(p.is_file() for p in paths):
            raise RuntimeError(f'{name}: evaluation file missing ({[str(p) for p in paths]}); refusing to generate without the exclusions')
        sets[name] = set().union(*(_text_ids(p.read_text(), old) for p in paths))
    sets['gold_set_py'] = gold_set_ids(list(full.values()))
    if with_checkers:
        for n in (4, 5, 6):
            sets[f'heldout_v{n}_checker'] = set(round5.checker_ids(ROOT / f'scripts/eval/check_heldout_v{n}.py', f'heldout_v{n}'))
    counts = {name: len(ids) for name, ids in sets.items()}
    union = set().union(*sets.values())
    # checker ids add +-1 arithmetically, so a few fall past a letter's last chunk: kept as ids, no neighbours
    counts['ids_not_in_corpus'] = sum(1 for i in union if BERKSHIRE_ID.match(i) and i not in full)
    excluded = set(union)
    for ident in union:
        doc = full.get(ident)
        for key in ('previous_chunk_id', 'next_chunk_id'):
            if doc is not None and doc.metadata.get(key):
                excluded.add(doc.metadata[key])
    counts.update(union_ids=len(union), with_neighbours=len(excluded))
    return excluded, counts


# ------------------------------------------------------------------ plan
def sentences_of(doc, cap=None):
    """Numbered source sentences of a passage as the 7B will see them (first `cap` chars; a cut last sentence is dropped)."""
    cap = cap or round4.LLM_PASSAGE_MAX_CHARS
    sentences = evidence_sentences(doc.text[:cap])
    if len(doc.text) > cap and sentences:
        sentences = sentences[:-1]
    return sentences


def eligible(doc, excluded):
    if not BERKSHIRE_ID.match(doc.id) or doc.id in excluded or len(doc.text) < 250:
        return False
    if sum(c.isalpha() for c in doc.text) / len(doc.text) <= .60:
        return False
    return len([s for s in sentences_of(doc) if len(s.split()) >= 5]) >= 2


def build_runs(docs_by_id, excluded, valid_years, kind, split, count, used, rng):
    """`count` runs of 1-3 consecutive eligible passages of one letter for (kind, split); `used` is shared across kinds."""
    pool = sorted(d.id for d in docs_by_id.values() if eligible(d, excluded)
                  and (int(d.metadata['year']) in valid_years) == (split == 'valid'))
    rng.shuffle(pool)
    lo, hi = RUN_LEN[kind]
    runs, starts = [], set()
    for overlap in (False, True):          # first pass: disjoint runs; if the pool runs dry, runs may share neighbours (distinct start)
        for start in pool:
            if len(runs) >= count:
                break
            if start in starts or (not overlap and start in used):
                continue
            want, run = rng.randint(lo, hi), [start]
            for step in ('next_chunk_id', 'previous_chunk_id'):
                cursor = start
                while len(run) < want:
                    neighbour = docs_by_id[cursor].metadata.get(step)
                    if not neighbour or neighbour not in docs_by_id or (not overlap and neighbour in used) or neighbour in run \
                            or not eligible(docs_by_id[neighbour], excluded):
                        break
                    run = run + [neighbour] if step == 'next_chunk_id' else [neighbour] + run
                    cursor = neighbour
            if kind != 'refusal' and sum(len([s for s in sentences_of(docs_by_id[i]) if len(s.split()) >= 5]) for i in run) < MIN_SENTENCES[kind]:
                continue
            used.update(run)
            starts.add(start)
            runs.append(run)
    return runs


def quotas_for(target):
    """{(kind, split): n}; valid is 1/11 of every kind (160 of 1760)."""
    out = {}
    for kind in KINDS:
        total = round(target * SHARE[kind])
        valid = round(total / 11)
        out[(kind, 'valid')], out[(kind, 'train')] = valid, total - valid
    return out


def build_plan(docs_by_id, excluded, valid_years, quotas, oversample, identity, rng_seed):
    rng, jobs = random.Random(rng_seed), []
    templates = round5.unavailable_templates()
    rng.shuffle(templates)
    for kind in ('temporal', 'multipart', 'follow_up', 'single', 'refusal'):    # scarcest sources first
        for split in ('train', 'valid'):
            count = int(quotas[(kind, split)] * oversample) + 1
            runs = build_runs(docs_by_id, excluded, valid_years, kind, split, count, set(), rng)   # passages are reused across kinds, not within one
            for index, run in enumerate(runs):
                job = {'id': f'v6:{kind}:{split}:{index:04d}', 'kind': kind, 'split': split, 'source_ids': run}
                if kind == 'refusal':
                    job['variant'] = 'template' if index % 2 == 0 and templates else 'adjacent'
                    if job['variant'] == 'template':
                        job['template'] = templates.pop()
                jobs.append(job)
    return {'identity': identity, 'oversample': oversample, 'quotas': {f'{k}:{s}': n for (k, s), n in quotas.items()},
            'counts': dict(Counter(f"{j['kind']}:{j['split']}" for j in jobs)), 'jobs': jobs}


# ------------------------------------------------------------------ teacher prompts
SYSTEM = 'You write test questions for a reading-comprehension dataset about Warren Buffett\'s letters. Return valid JSON only.'


def passages_block(docs):
    lines = []
    for p, doc in enumerate(docs, 1):
        lines.append(f'PASSAGE {p} (letter year {doc.metadata["year"]}):')
        lines += [f'P{p}.S{i}: {s}' for i, s in enumerate(sentences_of(doc), 1)]
        lines.append('')
    return '\n'.join(lines)


RULES = ('- Ask as a normal reader would: never mention "the passage", "the text", "the excerpt" or what "the letter says"; use no outside facts; '
         'do not copy six or more consecutive words from the passages.\n'
         '- Every part is about a DIFFERENT fact and is answered by one sentence, or by two ADJACENT sentences when one fact needs both.\n'
         '- Choose sentences that are understandable alone. If a sentence starts with This, That, It, They, These, Such, However, But or And, '
         'use the previous sentence as well (two adjacent sentences) or choose another sentence.\n'
         '- "sentences" lists the sentence numbers (S-numbers) of that passage: one number, or two adjacent numbers like [4, 5].\n')
STYLE = ('- Write the question the way a curious reader would phrase it, in your own words: vary the opening (How, Why, What did, In the letter for the '
         'year, Buffett...), and never reuse the passages\' phrases; a natural question shares only its key nouns with the passages.\n')
RULES += STYLE
SHAPE = '{"question": "...?", "parts": [{"ask": "short phrase", "passage": 1, "sentences": [3]}, {"ask": "short phrase", "passage": 2, "sentences": [1, 2]}]}'


def teacher_prompt(job, docs, history_first=None):
    kind, year = job['kind'], docs[0].metadata['year']
    lo, hi = PARTS.get(kind, (0, 0))
    head = f'Below are {len(docs)} consecutive passage(s) from the {year} Berkshire Hathaway shareholder letter. Sentences are numbered Pn.Sm.\n\n{passages_block(docs)}'
    if kind == 'multipart':
        task = (f'Write ONE natural question with {lo}-{hi} distinct parts (joined naturally: "what ..., how ... and why ..."), each part answered by the passages.\n'
                f'{RULES}- The question may mention the letter year ({year}) if natural.\nReturn ONLY JSON: {SHAPE}')
    elif kind == 'temporal':
        task = (f'Write ONE natural question about the {year} letter with {lo}-{hi} distinct parts, each answered by the passages. '
                f'The question MUST contain the year {year} literally (e.g. "In {year}, what ... and how ...?") and must not ask for a comparison with another year.\n'
                f'{RULES}Return ONLY JSON: {SHAPE}')
    elif kind == 'single':
        task = (f'Write ONE natural question about ONE subject (a single ask, no "and"-joined sub-questions) whose complete answer needs {lo}-{hi} separate facts '
                f'that the passages state, e.g. "What did Berkshire say about X?" or "What happened with X?". List those facts as the parts.\n'
                f'{RULES}- The question may mention the letter year ({year}).\nReturn ONLY JSON: {SHAPE}')
    elif kind == 'follow_up':
        task = (f'A reader first asked a question that one fact answers; now the reader asks a FOLLOW-UP question about the same subject with {lo}-{hi} further parts, '
                'answered by DIFFERENT sentences than the first question. The follow-up must refer to the subject only with it, they, that, this, their or those, and must not name the subject.\n'
                f'{RULES}Return ONLY JSON: {{"first_question": "...?", "first_part": {{"passage": 1, "sentences": [2]}}, "question": "...?", "parts": [{{"ask": "short phrase", "passage": 1, "sentences": [4]}}]}}')
    else:
        raise ValueError(kind)
    example = WORKED_EXAMPLES.get(kind, '')
    return f'{head}\nTASK: {task}\n{example}'


def refusal_prompt(docs):
    passage = passages_block(docs[:1])
    return (f'Below is a passage from a Berkshire Hathaway shareholder letter.\n\n{passage}\nTASK: Write ONE natural question on the same subject that a reader might ask but '
            'that this passage CANNOT answer, because it asks for a specific detail the passage does not give (another year\'s figure, a named person or company the passage does not mention, '
            'a number or date that is absent). The question must contain the missing detail itself as a distinctive word or number (a year, a number, or a capitalized name).\n'
            'Return ONLY JSON: {"question": "...?", "missing": ["distinctive word or number that is in your question and NOT in the passage"]}')


RETRY = {
    'json_invalid': 'return only the JSON object in the requested shape',
    'bad_question': 'the question must be one question of 20-600 characters ending with ?',
    'parts_count': 'give the requested number of parts',
    'bad_part': 'every part needs ask, passage and sentences with existing numbers',
    'sentence_index': 'use only sentence numbers that exist in that passage',
    'non_adjacent': 'a part uses one sentence, or two sentences with consecutive numbers',
    'duplicate_sentence': 'every part must use different sentences',
    'dependent_start': 'the first chosen sentence of a part must be understandable alone: add the previous sentence or choose another',
    'sentence_too_short': 'choose complete sentences of at least five words',
    'obligation_too_short': 'each part needs at least eight words of answer: add the adjacent sentence or choose another',
    'question_overlaps_passage': 'your question reuses too many words of the passages; ask it the way a curious reader who has not read them would',
    'bad_sentence': 'choose normal sentences (under 70 words, no row of asterisks)',
    'year_missing': 'the question must contain the letter year literally',
    'question_copies_passage': 'your question copied six or more consecutive words of a passage; rewrite it in your own words',
    'question_refers_to_passage': 'never mention the passage, the text or what the letter says',
    'followup_pronoun': 'the follow-up question must refer to its subject with it, they, that, this, their or those',
    'followup_same_sentences': 'the follow-up parts must use different sentences than the first question',
}


def parse_json(raw):
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw or ''):
        if char == '{':
            try:
                obj, _ = decoder.raw_decode(raw[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def has_copy(question, texts, n=6):
    asked = set(zip(*(words(question)[i:] for i in range(n))))
    return any(asked & set(zip(*(words(t)[i:] for i in range(n)))) for t in texts)


def check_question(question, kind, docs, overlap=True):
    if not isinstance(question, str) or not 20 <= len(question) <= 600 or not question.strip().endswith('?'):
        return 'bad_question'
    if PASSAGE_TALK.search(question):
        return 'question_refers_to_passage'
    if has_copy(question, [d.text for d in docs]):
        return 'question_copies_passage'
    if overlap and C.question_overlap(question, [d.text for d in docs]) > QUESTION_OVERLAP_MAX:
        return 'question_overlaps_passage'      # eval questions share 0.50-0.57 of their words with the gold passage, ft_v6 questions 0.75
    if kind == 'temporal' and str(docs[0].metadata['year']) not in question:
        return 'year_missing'
    return None


def check_parts(parts, kind, docs, sentence_lists, forbidden=(), parts_range=None):
    """-> (reason or None, obligations). An obligation: {ask, passage(1-based), sentence_numbers, sentences, source_id}.

    `parts_range` is the job's own [lo, hi] (an ft_v7 multipart job asks for EXACTLY 4 or 5 parts; ft_v6 jobs carry [3, 5])."""
    lo, hi = parts_range or PARTS[kind]
    if not isinstance(parts, list) or not lo <= len(parts) <= hi:
        return 'parts_count', None
    seen, out = set(forbidden), []
    for part in parts:
        if not isinstance(part, dict) or not isinstance(part.get('ask'), str):
            return 'bad_part', None
        passage, numbers = part.get('passage'), part.get('sentences')
        if not isinstance(passage, int) or not 1 <= passage <= len(docs) or not isinstance(numbers, list) or not 1 <= len(numbers) <= 2 \
                or not all(isinstance(n, int) for n in numbers):
            return 'bad_part', None
        if not all(1 <= n <= len(sentence_lists[passage - 1]) for n in numbers):
            return 'sentence_index', None
        if len(numbers) == 2 and numbers[1] != numbers[0] + 1:
            return 'non_adjacent', None
        texts = [sentence_lists[passage - 1][n - 1] for n in numbers]
        if any(len(t.split()) < 5 for t in texts):
            return 'sentence_too_short', None
        if any(bad_sentence(t) for t in texts):
            return 'bad_sentence', None
        if DEPENDENT_START.match(texts[0]) and len(numbers) == 1 and numbers[0] > 1:
            # repair instead of reject: a sentence that leans on the one before it is stated together with that sentence
            earlier = sentence_lists[passage - 1][numbers[0] - 2]
            if (passage, numbers[0] - 1) not in seen and len(earlier.split()) >= 5 and not DEPENDENT_START.match(earlier) \
                    and not bad_sentence(earlier):
                numbers, texts = [numbers[0] - 1, numbers[0]], [earlier] + texts
        if DEPENDENT_START.match(texts[0]):
            return 'dependent_start', None
        if sum(len(t.split()) for t in texts) < MIN_OBLIGATION_WORDS:
            return 'obligation_too_short', None
        keys = {(passage, n) for n in numbers}
        if keys & seen:
            return 'duplicate_sentence', None
        seen |= keys
        out.append({'ask': part['ask'].strip(), 'passage': passage, 'sentence_numbers': numbers, 'sentences': texts, 'source_id': docs[passage - 1].id})
    return None, out


def validate_plan(obj, kind, docs, parts_range=None):
    """Teacher JSON -> (reason or None, plan dict). `parts_range`: the job's own [lo, hi] number of parts (default PARTS[kind])."""
    if obj is None:
        return 'json_invalid', None
    sentence_lists = [sentences_of(d) for d in docs]
    reason = check_question(obj.get('question'), kind, docs)
    if reason:
        return reason, None
    history = None
    forbidden = ()
    if kind == 'follow_up':
        first = obj.get('first_part')
        if not isinstance(first, dict):
            return 'bad_part', None
        first_check = check_question(obj.get('first_question'), 'single', docs, overlap=False)   # one sentence answers it: it echoes that sentence
        if first_check:
            return first_check, None
        passage, numbers = first.get('passage'), first.get('sentences')
        if not isinstance(passage, int) or not 1 <= passage <= len(docs) or not isinstance(numbers, list) or not 1 <= len(numbers) <= 2 \
                or not all(isinstance(n, int) and 1 <= n <= len(sentence_lists[passage - 1]) for n in numbers) \
                or (len(numbers) == 2 and numbers[1] != numbers[0] + 1):
            return 'bad_part', None
        texts = [sentence_lists[passage - 1][n - 1] for n in numbers]
        if any(len(t.split()) < 5 for t in texts) or DEPENDENT_START.match(texts[0]):
            return 'dependent_start', None
        if any(bad_sentence(t) for t in texts):
            return 'bad_sentence', None
        if not PRONOUN.search(obj['question']):
            return 'followup_pronoun', None
        forbidden = {(passage, n) for n in numbers}
        history = {'first_question': obj['first_question'].strip(), 'sentences': texts}
    reason, obligations = check_parts(obj.get('parts'), kind, docs, sentence_lists, forbidden, parts_range)
    if reason:
        return ('followup_same_sentences' if reason == 'duplicate_sentence' and kind == 'follow_up' else reason), None
    return None, {'question': obj['question'].strip(), 'obligations': obligations, 'history': history}


def distinctive_terms(question):
    """Capitalised names and numbers in a template question (the details its passages must not contain)."""
    terms = set(re.findall(r'\b\d{4}\b', question))
    tokens = re.findall(r"[A-Za-z0-9'’.]+", question)
    for index, token in enumerate(tokens):
        token = re.sub(r"['’]s$", '', token).strip('.')
        if index > 0 and token[:1].isupper() and token.lower() not in {'the', 'what', 'which', 'how', 'who', 'when', 'why', 'us', 'u.s'}:
            terms.add(token)
    return terms


MONTHS_DAYS = {'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november',
               'december', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}


def refusal_terms(question):
    """distinctive_terms without month/day names and bare years, for the refusal check on served text.

    A 2024 letter says '2025' and 'March' without answering 'the closing price of Ethereum in March 2025'; with the full
    term set 98 of 105 template refusals were rejected. Entity names stay and still reject a row whose served text holds them.
    (Ported from the out-of-repo pv7/ftv6-assemble/shard_assemble.py that actually built ft_v6.)"""
    return {t for t in distinctive_terms(question) if t.lower() not in MONTHS_DAYS and not re.fullmatch(r'\d{4}', t)}


def bad_sentence(text):
    """Layout debris, not a fact: very long run-ons, '* * *' separators, a start that is not a capital/digit/quote, a cut-off end."""
    return len(text.split()) > 70 or '* *' in text or not re.match(r'[A-Z0-9"“‘(]', text) or not C.ends_like_a_sentence(text)


def term_present(term, texts):
    pattern = re.compile(r'(?<![A-Za-z0-9])' + re.escape(term) + r'(?![A-Za-z0-9])', re.I)
    return any(pattern.search(t) for t in texts)


# ------------------------------------------------------------------ rows
def chat_messages(question, hits, history=None):
    prompt = build_cited_prompt(question, hits, history)
    system = system_prompt(PROMPT_VERSION)
    prefix = system + '\n\n'
    assert prompt.startswith(prefix), f'PROMPT_VERSION is not {PROMPT_VERSION} in this process'
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt[len(prefix):]}]


def locate(sentence, hit_texts):
    """1-based slot [n] of a target sentence: the lowest-numbered served passage of which it is a COMPLETE sentence.

    This is the citation rule the 7B learns (ft_v7). A passage that only contains the sentence as a substring, or as half of
    a longer sentence, does not count (SP-01); several passages hold it because neighbour merging makes them overlap, and
    then the lowest one is the label (TARGET-06, TPC-03), checked again by check_row."""
    return C.lowest_holder(sentence, hit_texts)


def place_sources(hits, obligations, docs_by_id, rng):
    """Make every cited sentence present in a served passage. Returns (hits, injected, error)."""
    hits, injected = list(hits), []
    needed_docs = list(dict.fromkeys(o['source_id'] for o in obligations))
    texts = lambda: [norm(h.text) for h in hits]    # noqa: E731
    for ident in needed_docs:
        sentences = [s for o in obligations if o['source_id'] == ident for s in o['sentences']]
        if all(locate(s, texts()) for s in sentences):
            continue
        protected = {slot for other in needed_docs if other != ident for o in obligations if o['source_id'] == other
                     for s in o['sentences'] for slot in [locate(s, texts())] if slot}
        free = [i for i in range(len(hits)) if (i + 1) not in protected]
        doc = docs_by_id[ident]
        fresh = SearchHit(id=doc.id, text=truncate_around_anchor(doc.text, '', round4.LLM_PASSAGE_MAX_CHARS), metadata=dict(doc.metadata), score=0.)
        if free:
            slot = rng.choice(free)
            hits[slot] = fresh
        else:
            slot = len(hits)
            hits.append(fresh)
        injected.append({'passage_id': ident, 'slot': slot + 1})
        if not all(locate(s, texts()) for s in sentences):
            return hits, injected, 'sentence_not_servable'
    return hits, injected, None


def build_target(obligations, hits):
    """One line per obligation: its sentences verbatim, each followed by the [n] of the served passage that holds it."""
    texts = [norm(h.text) for h in hits]
    lines, slots = [], []
    for ob in obligations:
        parts, used = [], []
        for sentence in ob['sentences']:
            slot = locate(sentence, texts)
            if slot is None:
                return None, None
            parts.append(f'{norm(sentence)} [{slot}]')
            used.append(slot)
        lines.append(' '.join(parts))
        slots.append(used)
    return '\n'.join(lines), slots


def parse_prompt_passages(user):
    """slot -> (year, source, text) from the serving user message."""
    body = user.split('BEGIN UNTRUSTED PASSAGES\n', 1)[-1].split('\n\nEND UNTRUSTED PASSAGES', 1)[0]
    matches = list(PASSAGE_HEADER.finditer(body))
    out = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() - 2 if index + 1 < len(matches) else len(body)
        out[int(match.group(1))] = (match.group(2), match.group(3), norm(body[match.end():end]))
    return out


def check_row(row, *, excluded=None, count_tokens=None):
    """Offline checks of one stored row (the same function runs at generation and under --verify). -> list of problems."""
    problems = []
    messages = row.get('messages') or []
    if [m.get('role') for m in messages] != ['system', 'user', 'assistant']:
        return ['messages_shape']
    system, user, target = (m['content'] for m in messages)
    if system != system_prompt(row.get('prompt_version') or PROMPT_VERSION):
        problems.append('system_prompt_mismatch')
    passages = parse_prompt_passages(user)
    if not 1 <= len(passages) <= max(DEFAULT_PASSAGES, row.get('k') or 0) or sorted(passages) != list(range(1, len(passages) + 1)):
        return problems + ['passage_slots']
    if any(is_partnership(i) for i in (row.get('served_ids') or []) + (row.get('source_ids') or []) + (row.get('passage_ids') or [])
           + (row.get('context_passage_ids') or [])) or any(is_partnership(src) for _, src, _ in passages.values()):
        problems.append('partnership_passage')
    if excluded and set(row.get('source_ids') or []) & set(excluded):
        problems.append('source_excluded')
    if row.get('type') == 'unanswerable':
        if target != REFUSAL_LINE:
            problems.append('refusal_target_not_exact')
        terms = row.get('absent_terms') or []
        if not terms or any(term_present(t, [p[2] for p in passages.values()]) for t in terms):
            problems.append('refusal_terms_not_absent')
        return problems
    lines = target.split('\n')
    obligations = row.get('obligations') or []
    if any(bad_sentence(t) for ob in obligations for t in ob['sentences']):
        problems.append('bad_sentence')
    if any(DEPENDENT_START.match(ob['sentences'][0]) for ob in obligations if ob['sentences']):
        problems.append('dependent_start')
    if any(sum(len(t.split()) for t in ob['sentences']) < MIN_OBLIGATION_WORDS for ob in obligations):
        problems.append('obligation_too_short')
    if not 1 <= len(obligations) <= 5 or len(lines) != len(obligations):
        return problems + ['obligation_count']
    texts = [passages[i][2] for i in sorted(passages)]
    for line, ob in zip(lines, obligations):
        rest = line
        for sentence in ob['sentences']:
            want = norm(sentence)
            if not rest.startswith(want):
                problems.append('obligation_not_in_target')
                break
            rest = rest[len(want):]
            match = re.match(r'\s*\[(\d+)\]', rest)
            if not match:
                problems.append('sentence_without_marker')
                break
            slot = int(match.group(1))
            if slot not in passages or want not in passages[slot][2]:
                problems.append('sentence_not_in_cited_passage')
                break
            if not C.is_aligned(want, passages[slot][2]):
                problems.append('target_not_sentence_aligned')      # a fragment of a longer served sentence (SP-01)
                break
            if C.lowest_holder(want, texts) != slot:
                problems.append('citation_not_lowest_holder')        # the learned rule: [n] = lowest passage holding the sentence
                break
            if not C.locatable(want, passages[slot][2]):
                problems.append('sentence_not_locatable')            # answer_expand could not find it (EXP-03)
                break
            rest = rest[match.end():].lstrip()
        else:
            if rest.strip():
                problems.append('target_extra_text')
    hits = [SearchHit(id=f'slot{i}', text=passages[i][2], metadata={'year': passages[i][0]}, score=0.) for i in sorted(passages)]
    citations = parse_citations(target, hits)
    if not citations or any(c['invalid_numbers'] for c in citations):
        problems.append('invalid_citation')
    if C.serving_changes(target, [(passages[i][0], passages[i][2]) for i in sorted(passages)]):
        problems.append('serving_validator_drops_sentence')         # /ask would not show the target as written (POSTPROCESS-01/02)
    if count_tokens is not None:
        if count_tokens(messages[:2]) + count_tokens([messages[2]]) + 8 > row.get('max_seq', MAX_SEQ):
            problems.append('too_long_for_max_seq')
    return problems


# ------------------------------------------------------------------ generator
def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


class Generator:
    def __init__(self, teacher, serving, full, excluded, out, count_tokens, max_seq=MAX_SEQ, strict_context=False, passages_mix=None):
        self.teacher, self.serving, self.full, self.excluded, self.out = teacher, serving, full, excluded, Path(out)
        self.count_tokens, self.max_seq, self.strict_context = count_tokens, max_seq, strict_context
        self.mix = parse_passages_mix(passages_mix) if not isinstance(passages_mix, list) else list(passages_mix)
        self.rows = common.read_jsonl(self.out / 'examples.jsonl')
        self.attempts = common.read_jsonl(self.out / 'example_attempts.jsonl')
        self.done = {a['id'] for a in self.attempts}
        self.known = {norm(r['question']).lower() for r in self.rows}

    def attempt(self, job):
        """One job -> attempt record (and an examples row when kept). Never raises for data problems."""
        started = time.monotonic()
        record = {'id': job['id'], 'kind': job['kind'], 'split': job['split'], 'k': pick_k(job['id'], self.mix), 'kept': False, 'reason': None,
                  'teacher_calls': 0}
        row = None
        try:
            row, reason = self._attempt(job, record)
        except Exception as error:   # noqa: BLE001 - a bad job must not stop a resumable run
            if 'METAL' in str(error):
                raise            # out of GPU memory is not a property of the job: stop, the run resumes later
            reason = f'error_{type(error).__name__}'
            record['detail'] = str(error)[:300]
        if reason == 'budget':
            return None
        record.update(reason=reason, kept=reason is None, seconds=round(time.monotonic() - started, 2))
        if row is not None and reason is None:
            if row['obligations']:
                record.update(sentences=sum(len(o['sentences']) for o in row['obligations']), obligations=len(row['obligations']))
            append(self.out / 'examples.jsonl', row)
            self.rows.append(row)
            self.known.add(norm(row['question']).lower())
        append(self.out / 'example_attempts.jsonl', record)
        self.attempts.append(record)
        self.done.add(job['id'])
        return record

    # -- teacher
    def ask_teacher(self, job, prompt, record, validate):
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
        for attempt in range(2):
            raw, metrics = self.teacher.generate(messages, f'{job["id"]}:v6:{attempt}', max_tokens=MAX_TEACHER_TOKENS, temperature=TEMPERATURE)
            if metrics.get('finish_reason') == 'budget':
                return None, 'budget'
            record['teacher_calls'] += 1
            record['teacher_tokens'] = record.get('teacher_tokens', 0) + metrics.get('tokens', 0)
            reason, value = validate(parse_json(raw))
            if reason is None:
                return value, None
            record.setdefault('first_reasons', []).append(reason)
            messages = messages + [{'role': 'assistant', 'content': raw},
                                   {'role': 'user', 'content': f'REJECTED ({reason}): {RETRY.get(reason, "follow the rules")}. Return ONLY the corrected JSON object.'}]
        return None, reason

    def _attempt(self, job, record):
        docs = [self.full[i] for i in job['source_ids']]
        kind = job['kind']
        if kind == 'refusal':
            return self._refusal(job, docs, record)
        plan, reason = self.ask_teacher(job, teacher_prompt(job, docs), record, lambda obj: validate_plan(obj, kind, docs))
        if reason:
            return None, reason
        if norm(plan['question']).lower() in self.known:
            return None, 'duplicate_question'
        history = None
        if plan['history']:
            first = plan['history']
            history = [{'role': 'user', 'content': first['first_question']},
                       {'role': 'assistant', 'content': ' '.join(f'{norm(s)} [1]' for s in first['sentences'])}]
        return self._finish(job, docs, plan, history, record)

    def _refusal(self, job, docs, record):
        if job['variant'] == 'template':
            question, terms = job['template'], sorted(refusal_terms(job['template']))
        else:
            def validate(obj):
                if obj is None:
                    return 'json_invalid', None
                question, missing = obj.get('question'), obj.get('missing')
                if not isinstance(question, str) or not 20 <= len(question) <= 400 or not question.strip().endswith('?'):
                    return 'bad_question', None
                if not isinstance(missing, list) or not 1 <= len(missing) <= 3 or not all(isinstance(m, str) and m.strip() for m in missing):
                    return 'bad_part', None
                if not all(term_present(m, [question]) for m in missing):
                    return 'bad_part', None
                if PASSAGE_TALK.search(question):
                    return 'question_refers_to_passage', None
                return None, (question.strip(), [m.strip() for m in missing])
            value, reason = self.ask_teacher(job, refusal_prompt(docs), record, validate)
            if reason:
                return None, reason
            question, terms = value
        if norm(question).lower() in self.known:
            return None, 'duplicate_question'
        with serving_passages(record['k']):
            prompts = self.serving.prompts(question)
        if len(prompts) != 1 or not prompts[0]['hits']:
            return None, 'no_context'
        hits = prompts[0]['hits']
        if served_has_partnership(hits):      # partnership passages are evaluation-only: never in a training prompt (also with --strict-context off)
            return None, 'partnership_in_context'
        if any(term_present(t, [h.text for h in hits]) for t in terms):
            return None, 'refusal_terms_present_in_served'
        messages = chat_messages(question, hits) + [{'role': 'assistant', 'content': REFUSAL_LINE}]
        row = self.row(job, docs, question, [], hits, messages, None, [], prompts[0], {'absent_terms': terms, 'k': record['k']})
        return self.finalize(row)

    def _finish(self, job, docs, plan, history, record):
        question = plan['question']
        with serving_passages(record['k']):
            prompts = self.serving.prompts(question, history)
        if len(prompts) != 1 or prompts[0]['period'] is not None:
            return None, 'comparison_path_question'
        prompt = prompts[0]
        if not prompt['hits']:
            return None, 'no_context'
        if served_has_partnership(prompt['hits']):
            return None, 'partnership_in_context'
        if self.strict_context and round5.context_ids(prompt['hits'], self.full) & self.excluded:
            return None, 'excluded_passage_in_context'
        rng = random.Random(common.sha(f'{common.SEED}:{job["id"]}'.encode())[:12])
        hits, injected, error = place_sources(prompt['hits'], plan['obligations'], self.full, rng)
        if error:
            return None, error
        target, slots = build_target(plan['obligations'], hits)
        if target is None:
            return None, 'sentence_not_servable'
        messages = chat_messages(question, hits, history) + [{'role': 'assistant', 'content': target}]
        row = self.row(job, docs, question, plan['obligations'], hits, messages, history, injected, prompt, {'slots': slots, 'k': record['k']})
        return self.finalize(row)

    def row(self, job, docs, question, obligations, hits, messages, history, injected, prompt, extra):
        served = round5.served_ids(hits)
        answerable = bool(obligations)
        row = {'id': job['id'], 'kind': job['kind'], 'split': job['split'], 'type': 'answerable' if answerable else 'unanswerable',
               'source_id': job['source_ids'][0], 'source_ids': job['source_ids'], 'variant': job.get('variant', 'v6'), 'question': question,
               'style': 'v6', 'origin': 'v6_new', 'prompt_version': PROMPT_VERSION,
               'parts': [{'id': f'P{i}', 'ask': o['ask'], 'source_id': o['source_id'], 'sentence_numbers': o['sentence_numbers'],
                          'year': int(self.full[o['source_id']].metadata['year'])} for i, o in enumerate(obligations, 1)],
               'obligations': [{k: o[k] for k in ('ask', 'source_id', 'sentence_numbers', 'sentences')} for o in obligations],
               'source_year': int(docs[0].metadata['year']), 'period': None, 'serving_query': question,
               'passage_ids': [h.id for h in hits], 'context_passage_ids': sorted(round5.context_ids(hits, self.full)),
               'served_ids': sorted(served), 'passages': len(hits), 'prompt_chars': sum(len(m['content']) for m in messages[:2]),
               'gate_sufficient': bool(prompt.get('gate')), 'injected': injected, 'natural_miss': bool(injected),
               'source_in_top8': all(o['source_id'] in prompt['retrieved_ids'] for o in obligations) if obligations else None,
               'max_seq': self.max_seq, 'messages': messages, **extra}
        if history:
            row['history'] = history
        return row

    def finalize(self, row):
        problems = check_row(row, excluded=self.excluded, count_tokens=self.count_tokens)
        if row['type'] == 'answerable':
            row['sentences_in_target'] = sum(len(o['sentences']) for o in row['obligations'])
        return (row, None) if not problems else (None, problems[0])


# ------------------------------------------------------------------ outputs
def materialize(out, rows):
    out = Path(out)
    order = lambda r: common.sha(f'{common.SEED}:v6:{r["id"]}'.encode())   # noqa: E731
    out.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'valid'):
        chosen = sorted((r for r in rows if r['split'] == split), key=order)
        tmp = out / f'{split}.jsonl.tmp'
        tmp.write_text(''.join(json.dumps({'messages': r['messages']}, ensure_ascii=False) + '\n' for r in chosen))
        tmp.replace(out / f'{split}.jsonl')
    ordered = sorted(rows, key=lambda r: (r['split'], order(r)))
    tmp = out / 'split_metadata.jsonl.tmp'
    tmp.write_text(''.join(json.dumps({k: v for k, v in r.items() if k != 'messages'}, ensure_ascii=False) + '\n' for r in ordered))
    tmp.replace(out / 'split_metadata.jsonl')


def summarize(attempts, rows, seconds):
    by_kind = {}
    for kind in KINDS:
        mine = [a for a in attempts if a['kind'] == kind]
        kept = [a for a in mine if a['kept']]
        kept_rows = [r for r in rows if r['kind'] == kind]
        sentences = [r['sentences_in_target'] for r in kept_rows if r['type'] == 'answerable']
        by_kind[kind] = {'attempted': len(mine), 'kept': len(kept), 'keep_rate': round(len(kept) / len(mine), 3) if mine else None,
                         'reasons': dict(Counter(a['reason'] for a in mine if a['reason'])),
                         'sentences_per_target': round(sum(sentences) / len(sentences), 2) if sentences else None,
                         'teacher_seconds': round(sum(a.get('seconds', 0) for a in mine), 1)}
    kept_total = sum(v['kept'] for v in by_kind.values())
    return by_kind, kept_total


def k_histogram(rows):
    return {str(k): n for k, n in sorted(Counter(r.get('k') or DEFAULT_PASSAGES for r in rows).items())}


def manifest_for(out, rows, attempts, counts, identity, seconds, target, max_seq=MAX_SEQ, mix=None):
    by_kind, kept_total = summarize(attempts, rows, seconds)
    return {**identity, 'prompt_version': PROMPT_VERSION, 'max_seq': max_seq, 'passages_mix': list(mix or [DEFAULT_PASSAGES]), 'k_histogram': k_histogram(rows),
            'target_rows': target, 'exclusion_counts': counts,
            'row_counts_per_kind': {k: sum(r['kind'] == k for r in rows) for k in KINDS},
            'counts_per_split_kind': {s: dict(Counter(r['kind'] for r in rows if r['split'] == s)) for s in ('train', 'valid')},
            'by_kind': by_kind, 'natural_miss_share': round(sum(r['natural_miss'] for r in rows) / max(1, len(rows)), 3),
            'rejection_reasons': dict(Counter(a['reason'] for a in attempts if a['reason'])),
            'wall_seconds': round(seconds, 1), 'rows_per_minute': round(60 * kept_total / seconds, 3) if seconds else None}


# ------------------------------------------------------------------ jobs file: plan (no LLM) and assemble (no LLM)
def open_text(path, mode='r'):
    return gzip.open(path, mode + 't', encoding='utf-8') if str(path).endswith('.gz') else open(path, mode, encoding='utf-8')


def read_jsonl_file(path):
    with open_text(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_file(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('tmp-' + path.name)
    with open_text(tmp, 'w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def read_id_file(path):
    """Extra passage ids to exclude (e.g. spent sets that are not on this branch): a JSON list or one id per line."""
    if not path:
        return set()
    text = Path(path).read_text()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return {str(i) for i in data}
    except json.JSONDecodeError:
        pass
    return {line.strip() for line in text.splitlines() if line.strip() and not line.startswith('#')}


def teacher_messages(job, docs):
    """The exact chat the teacher receives (None for a template refusal, which needs no teacher)."""
    if job['kind'] == 'refusal':
        if job.get('variant') == 'template':
            return None
        prompt = refusal_prompt(docs)
    else:
        prompt = teacher_prompt(job, docs)
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]


def job_record(job, full, identity, count_tokens=None):
    """One self-contained line of jobs.jsonl: the sampled job plus everything the Kaggle teacher and the shape check need."""
    docs = [full[i] for i in job['source_ids']]
    messages = teacher_messages(job, docs)
    lo, hi = PARTS.get(job['kind'], (0, 0))
    seed = int(common.sha(f'{common.SEED}:{job["id"]}:v6:0'.encode())[:8], 16)
    return {**job, 'year': int(docs[0].metadata['year']), 'expect': EXPECT[job['kind']], 'needs_teacher': messages is not None,
            'n_passages': len(docs), 'sentence_counts': [len(sentences_of(d)) for d in docs], 'parts_range': [lo, hi] if lo else None,
            'sampling': {'temperature': TEMPERATURE, 'max_tokens': MAX_TEACHER_TOKENS, 'seed': seed},
            'messages': messages, 'prompt_tokens': count_tokens(messages) if messages and count_tokens else None, 'identity': identity}


def order_jobs(jobs):
    """Interleave kinds and splits proportionally so that a partial teacher run still has the intended mix."""
    groups = Counter((j['kind'], j['split']) for j in jobs)
    position = {}
    keyed = []
    for job in jobs:
        group = (job['kind'], job['split'])
        index = position[group] = position.get(group, -1) + 1
        keyed.append(((index + .5) / groups[group], KINDS.index(job['kind']), job['split'], job))
    return [item[3] for item in sorted(keyed, key=lambda item: item[:3])]


def plan_jobs(n, seed, exclude_ids=()):
    """n jobs in total, kind mix and splits as quotas_for(n); sampling is build_plan with rng seed `seed` (exclusions as always)."""
    full, excluded, counts, valid_years, identity = load_inputs(KAGGLE_TEACHER)
    extra = set(exclude_ids)
    excluded = excluded | extra
    identity = {'corpus_sha256': identity['corpus_sha256'], 'plan_seed': seed, 'n': n, 'valid_years': identity['valid_years'],
                'exclusion_with_neighbours': len(excluded), 'extra_exclude_ids': len(extra)}
    plan = build_plan(full, excluded, valid_years, quotas_for(n), 1.0, identity, seed)
    _, count_tokens = round4.token_counter(DEFAULT_TEACHER)      # the Qwen2.5 tokenizer is shared by 14B and 32B
    return [job_record(job, full, identity, count_tokens) for job in order_jobs(plan['jobs'])], counts


def smoke_jobs(jobs, per_kind=6):
    chosen = []
    for kind in KINDS:
        chosen += [j for j in jobs if j['kind'] == kind and j['needs_teacher']][:per_kind]
    return chosen


def plan_command(args):
    jobs, counts = plan_jobs(args.n, args.seed, read_id_file(args.exclude_ids))
    write_jsonl_file(args.out, jobs)
    by_kind = Counter(j['kind'] for j in jobs)
    tokens = [j['prompt_tokens'] for j in jobs if j['prompt_tokens']]
    print(f'jobs={len(jobs)} per_kind={dict(by_kind)} teacher_jobs={sum(j["needs_teacher"] for j in jobs)} '
          f'longest_prompt_tokens={max(tokens)} mean_prompt_tokens={sum(tokens) // len(tokens)} excluded_with_neighbours={counts["with_neighbours"]}')
    print(f'sha256={common.sha(Path(args.out).read_bytes())} bytes={Path(args.out).stat().st_size}')
    if args.smoke_out:
        smoke = smoke_jobs(jobs, args.smoke_per_kind)
        write_jsonl_file(args.smoke_out, smoke)
        print(f'smoke jobs={len(smoke)} per_kind={dict(Counter(j["kind"] for j in smoke))}')


def read_teacher_output(path):
    """id -> record; a later line for the same id wins (a resumed or re-run kernel), a record with a parsed answer wins over a failed one."""
    records = {}
    for rec in read_jsonl_file(path):
        old = records.get(rec['id'])
        if old is None or rec.get('ok') or not old.get('ok'):
            records[rec['id']] = rec
    return records


class RecordedGenerator(Generator):
    """Generator whose 'teacher' is a recorded Kaggle output: same validation, no retry (the kernel already retried shape errors)."""

    def __init__(self, recorded, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        self.recorded = recorded

    def ask_teacher(self, job, prompt, record, validate):
        rec = self.recorded.get(job['id'])
        if rec is None:
            return None, 'no_teacher_output'
        record['teacher_calls'] = rec.get('calls', 1)
        record['teacher_tokens'] = rec.get('tokens', 0)
        reason, value = validate(parse_json(rec.get('raw')))
        if reason is None:
            return value, None
        record.setdefault('first_reasons', []).append(reason)
        return None, reason


def assemble_command(args):
    if (Path(args.out_dir) / 'train.jsonl').exists() and not args.force:   # data/ft_v6 is what FT-r3 was trained on: never overwrite
        raise SystemExit(f'{args.out_dir}/train.jsonl exists; pick a new --out-dir or pass --force')
    os.environ['PROMPT_VERSION'] = PROMPT_VERSION
    mix = parse_passages_mix(args.passages_mix)       # every row serves with its own LLM_CONTEXT_PASSAGES (no flag: 8, as before)
    max_seq = args.max_seq or (MAX_SEQ_MIX if args.passages_mix else MAX_SEQ)
    jobs_path, teacher_path = Path(args.jobs), Path(args.teacher)
    jobs, recorded = read_jsonl_file(jobs_path), read_teacher_output(teacher_path)
    full, excluded, counts, valid_years, identity = load_inputs(KAGGLE_TEACHER)
    extra = read_id_file(args.exclude_ids)
    excluded = excluded | extra
    counts.update(extra_exclude_ids=len(extra), with_neighbours=len(excluded))
    wrong = {j['identity']['corpus_sha256'] for j in jobs if j.get('identity')} - {identity['corpus_sha256']}
    if wrong:
        raise SystemExit('jobs were planned on a different corpus than data/processed/chunks_v3_paragraph.jsonl; re-run plan')
    identity = {**identity, 'jobs_sha256': common.sha(jobs_path.read_bytes()), 'teacher_output_sha256': common.sha(teacher_path.read_bytes()),
                'plan': jobs[0]['identity'] if jobs else None}
    out = Path(args.out_dir)
    work = out / 'assemble_run'
    work.mkdir(parents=True, exist_ok=True)
    for name in ('examples.jsonl', 'example_attempts.jsonl'):        # assemble is a pure function of its inputs: no resume state
        (work / name).unlink(missing_ok=True)
    _, count_tokens = round4.token_counter(DEFAULT_TEACHER)
    generator = RecordedGenerator(recorded, round5.load_serving(), full, excluded, work, count_tokens, max_seq, args.strict_context, mix)
    quotas = quotas_for(args.target) if args.target else None
    started = time.monotonic()
    for index, job in enumerate(jobs, 1):
        key = (job['kind'], job['split'])
        if job['kind'] not in KINDS:
            continue
        if quotas is not None and sum(1 for a in generator.attempts if a['kept'] and (a['kind'], a['split']) == key) >= quotas[key]:
            continue
        generator.attempt(job)
        if index % 100 == 0:
            print(f'assembled {index}/{len(jobs)} kept={len(generator.rows)}', flush=True)
    materialize(out, generator.rows)
    seconds = time.monotonic() - started
    manifest = manifest_for(out, generator.rows, generator.attempts, counts, identity, seconds, args.target or len(jobs), max_seq, mix)
    reasons = manifest['rejection_reasons']
    manifest.update(jobs=len(jobs), jobs_per_kind=dict(Counter(j['kind'] for j in jobs)), teacher_records=len(recorded),
                    teacher_jobs_without_output=sum(1 for j in jobs if j.get('needs_teacher') and j['id'] not in recorded),
                    partnership_rejected=sum(reasons.get(r, 0) for r in PARTNERSHIP_REASONS),
                    spent_set_distractor_rows=sum(1 for r in generator.rows if set(r['context_passage_ids']) & excluded),
                    strict_context=bool(args.strict_context))
    common.atomic_json(out / 'manifest.json', manifest)
    kept = manifest['row_counts_per_kind']
    print('kept rows per kind: ' + ' '.join(f'{k}={kept[k]}' for k in KINDS) + f' total={sum(kept.values())} of jobs={len(jobs)}')
    print(f'rejected: {reasons}')
    print(f'partnership_rejected={manifest["partnership_rejected"]} spent_set_distractor_rows={manifest["spent_set_distractor_rows"]} -> {out}/{{train,valid}}.jsonl, manifest.json')
    check_refusal_share(generator.rows, args.min_refusal_share)


def check_over_length(attempts, allow):
    """Rows the token cap dropped are rows the 7B never trains on, and they are the longest ones (TOKEN-02, TRAIN-08, PD-03): print
    them and exit non-zero unless --allow-drops. -> the dropped ids."""
    dropped = [a for a in attempts if a.get('reason') == 'too_long_for_max_seq']
    for a in dropped:
        print(f"over-length drop {a['id']} kind={a['kind']} k={a.get('k')}")
    if dropped and not allow:
        raise SystemExit(f'{len(dropped)} over-length drop(s); raise --max-seq (cap {MAX_SEQ_MIX}) or pass --allow-drops')
    return [a['id'] for a in dropped]


def check_refusal_share(rows, minimum):
    """Fail loudly when refusals collapse (a too-strict term check once cut them from 17% to 2% without any error)."""
    share = sum(r['kind'] == 'refusal' for r in rows) / max(len(rows), 1)
    print(f'refusal share {share:.3f} (minimum {minimum})')
    if share < minimum:
        raise SystemExit(f'refusal share {share:.3f} is below {minimum}; check the refusal term filter before training')


# ------------------------------------------------------------------ run / verify
def verify(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    full = {d.id: d for d in common.load_chunks_as_docs(common.CORPUS)}
    excluded, _ = exclusion_sets(full)
    _, counter = round4.token_counter(DEFAULT_TEACHER)
    bad = []
    for row in rows:
        problems = check_row(row, excluded=excluded, count_tokens=counter)
        if row.get('kind') not in KINDS:
            problems.append('unknown_kind')
        if problems:
            bad.append((row.get('id'), problems))
    if bad:
        for ident, problems in bad[:20]:
            print(f'BAD {ident}: {problems}')
        raise SystemExit(f'FAIL bad_rows={len(bad)} of {len(rows)}')
    print(f'OK rows={len(rows)}')


def load_inputs(teacher_path):
    docs = common.load_chunks_as_docs(common.CORPUS)
    full = {d.id: d for d in docs}
    excluded, counts = exclusion_sets(full)
    years = sorted({int(d.metadata['year']) for d in docs if BERKSHIRE_ID.match(d.id)})
    valid_years = set(random.Random(common.SEED).sample(years, 5))
    identity = {'corpus_sha256': common.sha(common.CORPUS.read_bytes()), 'teacher': str(teacher_path), 'teacher_license': 'Apache-2.0 (open weights)',
                'seed': common.SEED, 'valid_years': sorted(valid_years)}
    return full, excluded, counts, valid_years, identity


def run(args, parser):
    os.environ['PROMPT_VERSION'] = PROMPT_VERSION
    if round4.LLM_CONTEXT_PASSAGES != 8:
        parser.error('LLM_CONTEXT_PASSAGES must be 8 (unset LLM_MODEL_PATH / LLM_CONTEXT_PASSAGES)')
    pilot = args.pilot is not None
    teacher_path = os.environ.get('FT_TEACHER', DEFAULT_TEACHER)
    out = ROOT / OUT_DIR
    work = out / 'pilot_run' if pilot else out
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / 'generation.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    full, excluded, counts, valid_years, identity = load_inputs(teacher_path)
    kinds = [k for k in (args.kinds.split(',') if args.kinds else KINDS) if k in KINDS]
    target = args.target
    quotas = quotas_for(target)
    if pilot:
        quotas = {key: (args.pilot if key[1] == 'train' else 0) for key in quotas}
    plan_path = work / 'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        assert plan['identity'] == identity, 'input identity changed; refusing to resume'
    else:
        plan = build_plan(full, excluded, valid_years, quotas, 1.0 if pilot else args.oversample, identity, common.SEED + 6)
        common.atomic_json(plan_path, plan)
        print(f'plan {plan["counts"]} excluded={counts["with_neighbours"]}', flush=True)
    started = time.monotonic()
    deadline = started + args.max_minutes * 60 if args.max_minutes else None
    teacher = common.Teacher(deadline=deadline, path=teacher_path)
    _, count_tokens = round4.token_counter(teacher_path)
    generator = Generator(teacher, round5.load_serving(), full, excluded, work, count_tokens, MAX_SEQ, args.strict_context)
    pending = defaultdict(list)
    for job in plan['jobs']:
        if job['id'] not in generator.done and job['kind'] in kinds:
            pending[(job['kind'], job['split'])].append(job)

    def kept(key):
        return sum(1 for a in generator.attempts if a['kept'] and (a['kind'], a['split']) == key)

    def attempted(key):
        return sum(1 for a in generator.attempts if (a['kind'], a['split']) == key)

    while True:
        if deadline and time.monotonic() >= deadline:
            break
        open_keys = [k for k in pending if pending[k] and quotas[k] > 0 and ((attempted(k) < quotas[k]) if pilot else (kept(k) < quotas[k]))]
        if not open_keys:
            break
        key = min(open_keys, key=lambda k: ((attempted(k) if pilot else kept(k)) / quotas[k], KINDS.index(k[0])))
        job = pending[key].pop(0)
        record = generator.attempt(job)
        if record is None:
            break
        print(f'{job["id"]} kept={record["kept"]} reason={record["reason"]} s={record["seconds"]}', flush=True)
        if len(generator.rows) % 10 == 0:
            materialize(work, generator.rows)
    materialize(work, generator.rows)
    seconds = sum(a.get('seconds', 0) for a in generator.attempts)
    manifest = manifest_for(work, generator.rows, generator.attempts, counts, identity, seconds, target)
    if pilot:
        (out / 'pilot.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in generator.rows))
        manifest.update(pilot_per_kind=args.pilot, eta=eta(manifest, quotas_for(target)))
        common.atomic_json(out / 'pilot_manifest.json', manifest)
    else:
        common.atomic_json(out / 'manifest.json', manifest)
    print('MANIFEST ' + json.dumps({k: manifest[k] for k in ('row_counts_per_kind', 'rows_per_minute', 'rejection_reasons')}), flush=True)


def eta(manifest, quotas):
    """Hours to keep the full quota at each kind's measured seconds-per-kept-row."""
    total = 0.
    detail = {}
    for kind, stats in manifest['by_kind'].items():
        if not stats['kept']:
            detail[kind] = None
            continue
        need = sum(n for (k, _), n in quotas.items() if k == kind)
        seconds = stats['teacher_seconds'] / stats['kept'] * need
        detail[kind] = round(seconds / 3600, 2)
        total += seconds
    return {'hours_by_kind': detail, 'hours_total': round(total / 3600, 2), 'rows': sum(quotas.values())}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = ('plan', 'teach-local', 'assemble', 'verify')
    if argv and argv[0] not in commands + ('-h', '--help'):
        argv = ['teach-local'] + argv                  # the pre-split CLI (--pilot / --target / --verify ...) is teach-local
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('plan', help='sample jobs + exact teacher messages -> jobs.jsonl (no LLM)')
    p.add_argument('--n', type=int, required=True, help='total number of jobs (kind mix as SHARE, 1/11 of each kind valid)')
    p.add_argument('--seed', type=int, required=True, help='rng seed of the job sampling')
    p.add_argument('--out', required=True, help='jobs file (.jsonl or .jsonl.gz)')
    p.add_argument('--smoke-out', help='also write a small smoke file (first --smoke-per-kind teacher jobs of each kind)')
    p.add_argument('--smoke-per-kind', type=int, default=6)
    p.add_argument('--exclude-ids', help='file of extra passage ids (JSON list or one per line) that must not be sources')

    t = sub.add_parser('teach-local', help="today's MLX teacher path (pilot / small runs)")
    t.add_argument('--pilot', type=int, metavar='N', help='attempt N jobs per kind into data/ft_v6/pilot_run (+ pilot.jsonl, pilot_manifest.json)')
    t.add_argument('--kinds', help='comma-separated subset of kinds (pilot re-runs)')
    t.add_argument('--target', type=int, default=1760, help='total rows (default 1600 train + 160 valid)')
    t.add_argument('--oversample', type=float, default=2.5, help='plan size multiple of each quota')
    t.add_argument('--max-minutes', type=float, help='stop teacher work after this wall-clock time')
    t.add_argument('--strict-context', action='store_true', help='reject rows whose served context holds an excluded (spent-set) passage')
    t.add_argument('--verify', metavar='JSONL', help='re-check every row of a generated jsonl; prints OK rows=<n>')

    a = sub.add_parser('assemble', help='teacher output + jobs -> rows, checks, splits, manifest (no LLM)')
    a.add_argument('--jobs', required=True)
    a.add_argument('--teacher', required=True, help='teacher_out.jsonl written by the Kaggle kernel')
    a.add_argument('--out-dir', default=OUT_DIR, help=f'writes train.jsonl, valid.jsonl, split_metadata.jsonl, manifest.json here (default {OUT_DIR})')
    a.add_argument('--target', type=int, help='cap kept rows per (kind, split) at the quotas of this total (default: keep all)')
    a.add_argument('--exclude-ids', help='file of extra passage ids that must not be sources (spent sets not on this branch)')
    a.add_argument('--strict-context', action='store_true', help='also reject rows whose served context holds an excluded (spent-set) passage')
    a.add_argument('--passages-mix', help='served context sizes, e.g. 8,12,16: each row picks one by a hash of its job id (default: 8 for every row)')
    a.add_argument('--force', action='store_true', help='overwrite an existing --out-dir')
    a.add_argument('--min-refusal-share', type=float, default=MIN_REFUSAL_SHARE,
                   help=f'exit non-zero when refusal rows are below this share of kept rows (default {MIN_REFUSAL_SHARE}; 0 = off)')
    a.add_argument('--max-seq', type=int, help=f'reject rows longer than this many tokens (default {MAX_SEQ}; {MAX_SEQ_MIX} with --passages-mix)')

    v = sub.add_parser('verify', help='offline re-check of every row of a generated jsonl')
    v.add_argument('path')

    args = parser.parse_args(argv)
    os.environ['PROMPT_VERSION'] = PROMPT_VERSION
    if args.command == 'plan':
        return plan_command(args)
    if args.command == 'assemble':
        return assemble_command(args)
    if args.command == 'verify':
        return verify(args.path)
    if args.verify:
        return verify(args.verify)
    return run(args, parser)


if __name__ == '__main__':
    main()
