"""FT-r4 finance training-data builder (openspec change ft-r4-finance, group 1): rows for the Gemma 4 12B reader.

  PY=".venv/bin/python"  (env -u PYTHONPATH; the serving flags below are set by the script itself)
  PY scripts/ft/build_ft_finance.py plan --n 900 --seed 41 --out DIR/jobs.jsonl          # no model: jobs + exact teacher messages
  PY scripts/ft/build_ft_finance.py export-batches --jobs DIR/jobs.jsonl --size 50 --out DIR/batches/   # + TEACHER.md
  (teacher: Claude panes follow batches/TEACHER.md, write DIR/out/teacher_batch_NNN.jsonl, run scripts/ft/check_teacher.py)
  PY scripts/ft/build_ft_finance.py fake-teacher --jobs DIR/jobs.jsonl --out DIR/fake_teacher.jsonl   # deterministic picks
  PY scripts/ft/build_ft_finance.py assemble --jobs DIR/jobs.jsonl --teacher DIR/out --out DIR/rows   # retrieval models, no LLM
  PY scripts/ft/build_ft_finance.py verify DIR/rows/rows.jsonl [--static]                     # re-serve + re-check every row

The rule (PREREGISTRATION.md): every row is the chat the finance reader route sends for its question, and its target
is built, not written: the premise sentence when the plan has a `premise_note`, then the teacher's chosen letter
sentences verbatim, oldest letter first, each ending with its [n] (ft_v7_checks.lowest_holder: the lowest-numbered
served passage that holds the whole sentence). One line per letter year; a partial row ends with one gap line
(ft_v7_checks.gap_line), a refusal row is the refusal line.

Serving replay (`Router`): `ask_flow.finance_route` itself, offline (corpus v3, FAISS, bge embedder + reranker, a provider
that never generates and is named "remote" like the Modal app's, so no llama window fit), with FINANCE_PIPELINE=1,
FINANCE_ANSWER_V3=1, FINANCE_PREMISE_CHECK=1, the Modal app's VALIDATOR_SPLIT_V2=1 / VALIDATOR_VERBATIM_KEEP=1,
PROMPT_VERSION=v2 (the Gemma reader's system prompt, tasks.md 3.1) and every other FINANCE_* / retrieval setting at its
config.py default (an override in the environment stops the script). The reader window is counted with the Gemma 4 12B
tokenizer.json (LLM_TOKENIZER_JSON): at serving the llama.cpp server's /tokenize counts it, for the Gemma GGUF in Gemma
tokens. What the replay changes, and records per row:
  - mask (`mask_context_ids.json`, the sealed sets still to be measured): finance_search sees those passages as
    non-financial (`MaskedStore`), so it drops them before the rerank and the next candidates fill their places, and the
    neighbour merge / figure pins cannot reach them (they are left out of the doc lookup). `masked.pool` = masked ids
    among the search candidates, `masked.neighbours` = masked chunks a served hit would have merged;
  - injection: a chosen source sentence the route did not serve -> the route runs again with that passage first among
    the finance_search hits (inside the budget, so the window trim keeps it); the v3 compose puts it in its letter-year
    slot. `injected` = [{passage_id, slot}].

Checks (`check_row`, run by assemble and again by verify): the user turn is the route's (verify re-serves), no masked id
in the context, the target rebuilds from the served passages ([n] = lowest holder, chronological, premise first, gap
last), no target sentence comes from an `exclude_ids.json` passage, the finance post-processing keeps every sentence
(`ask_flow._finalize_once`: validator, dedupe, premise strip / re-add, personal-sentence drop), prompt + target <= 15,500
Gemma tokens, question overlap <= 0.70, no forbidden personal sentence in a mixed row, the row's letters on its side of
the letter-year train/valid split. Jobs and teacher answers are checked first by check_teacher.py (`validate`).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_ft_v6 as V6  # noqa: E402  (bad_sentence, term_present, valid_years_of, read/write jsonl)
import check_teacher as CT  # noqa: E402
import common  # noqa: E402  (offline env flags; the repo root on sys.path)
import ft_v7_checks as C  # noqa: E402
from src.evaluation.claim_validator import evidence_sentences  # noqa: E402
from src.generation import finance_answer  # noqa: E402
from src.generation.prompt import REFUSAL_LINE, system_prompt  # noqa: E402
from src.query import plan as P  # noqa: E402
from src.storage import SearchHit  # noqa: E402

ROOT = common.ROOT
ART = Path.home() / 'AI-Workplace/Artifacts/BuffettRAG/ftr4-finance'
EXCLUDE_FILE = ART / 'exclude_ids.json'
MASK_FILE = ART / 'mask_context_ids.json'
GEMMA_TOKENIZER = ROOT / 'data/tokenizer/gemma4_12b_it/tokenizer.json'
GEMMA_REVISION = '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'
CACHE_NAME = 'model_cache.pkl'
PROMPT_VERSION = 'v2'               # tasks.md 3.1: the Gemma reader's system prompt stays the v2 text
MAX_TOKENS = 15500                  # tasks.md 1.1: Gemma tokens of prompt + target
QUESTION_OVERLAP_MAX = CT.QUESTION_OVERLAP_MAX
KINDS = CT.KINDS
ANSWER_KINDS = CT.ANSWER_KINDS
# Share of the JOBS per kind (tasks.md gives the kinds, not the shares). Groups the dev gate scores: crises (1), views
# (2), figures (3), refusals 4/4, scope 3/3, mixed leaks 0/2 -- refusal + mixed_scope + partial kinds are a fifth+.
SHARE = {'crisis_period': .16, 'view_over_time': .20, 'single_year': .16, 'figure': .12, 'premise_partial': .08,
         'partial': .08, 'refusal': .12, 'mixed_scope': .08}
VALID_SHARE = .10
PLAN_ORDER = ('premise_partial', 'view_over_time', 'crisis_period', 'mixed_scope', 'figure', 'single_year', 'partial', 'refusal')
SERVING_FLAGS = {'FINANCE_PIPELINE': '1', 'FINANCE_ANSWER_V3': '1', 'FINANCE_PREMISE_CHECK': '1',
                 'VALIDATOR_SPLIT_V2': '1', 'VALIDATOR_VERBATIM_KEEP': '1'}     # the last two: deploy/modal/app.py serve()
PINNED_ENV = ('LLM_CONTEXT_PASSAGES', 'LLM_N_CTX', 'LLM_PASSAGE_ORDER', 'LLM_MODEL_PATH', 'EVIDENCE_GATE_SOFT',
              'ANSWER_MAX_SENTENCES', 'ANSWER_EXPANDED_MAX_SENTENCES', 'KNOWLEDGE_DIR', 'FAISS_INDEX_NAME')
OIL = 'oil_shocks'                  # the planner's user-phrase event without a store id (plan._OIL_SHOCKS)
SECTOR_LABELS = {
    'financial_services_other': 'financial services', 'utilities_energy': 'utilities and energy', 'oil_gas': 'oil and gas',
    'airlines_aviation': 'airlines', 'newspapers_publishing': 'newspapers', 'tv_broadcasting_media': 'television and media',
    'retail': 'retailing', 'manufacturing_industrial': 'manufacturing', 'housing_construction': 'housing and construction',
    'textiles_apparel_shoes': 'textiles and shoes', 'commodities_gold': 'gold and commodities',
    'japan_trading_houses': 'the Japanese trading houses', 'transportation_other': 'trucking and logistics',
    'healthcare_pharma': 'healthcare', 'steel_metals_mining': 'steel and mining'}
THEME_LABELS = {
    'bonds_fixed_income': 'bonds', 'preferred_convertible_investments': 'preferred stocks and convertibles',
    'arbitrage_workouts': 'merger arbitrage', 'market_valuation': 'stock market valuations',
    'market_timing_forecasts': 'market forecasts and timing', 'cash_liquidity': 'cash and liquidity', 'debt_leverage': 'debt and leverage',
    'buybacks': 'share buybacks', 'dividends_capital_allocation': 'dividends and capital allocation', 'acquisitions_deals': 'acquisitions',
    'intrinsic_value_book_value': 'intrinsic value and book value', 'accounting_reporting': 'accounting',
    'pensions_retirement_accounting': 'pension accounting', 'float_underwriting': 'insurance float and underwriting',
    'management_governance': 'management and corporate governance', 'succession_culture': "succession and Berkshire's culture",
    'shareholder_programs': "Berkshire's shareholder programs", 'index_funds_fees': 'index funds and fees',
    'currency_trade_deficit': 'the dollar and the trade deficit', 'government_fiscal_policy': 'government deficits',
    'regulation_politics': 'regulation', 'moats_competitive_advantage': 'moats and competitive advantage',
    'mistakes_lessons': 'mistakes and lessons', 'berkshire_performance': "Berkshire's performance",
    'stock_issuance_splits': 'share issuance and splits', 'climate_esg': 'climate change',
    'american_economy_outlook': "America's economic outlook"}
FIGURE_METRIC_SKIP = {'other'}
TOKEN_EDGES = (0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 15000, 15500, 16384)
TARGET_EDGES = (0, 100, 200, 400, 600, 900, 1200)


def norm(text):
    return C.norm(text)


def read_ids(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return set(data['ids'] if isinstance(data, dict) else data)


def file_sha(path):
    path = Path(path)
    return common.sha(path.read_bytes()) if path.is_file() else None


def year_of(pid, full=None):
    doc = (full or {}).get(pid)
    return int(doc.metadata['year']) if doc is not None else int(str(pid)[:4])


# ------------------------------------------------------------------ plan: the knowledge layer's view of the corpus
class Pools:
    """Eligible seed passages and their knowledge-layer tags, with use counts that spread the plan over the corpus.

    Eligible (a source of target sentences): a Berkshire passage that is financial (`KnowledgeStore.is_financial`), not
    a table, not in exclude_ids.json, with at least two usable sentences. A mixed_scope job's PERSONAL passage is a
    has_personal or non-financial passage that is not excluded either (it is shown to the teacher, never quoted)."""

    def __init__(self, full, store, exclude, valid_years, mask=()):
        self.full, self.store, self.exclude, self.valid_years = full, store, set(exclude), set(valid_years)
        self.visible = {pid: doc for pid, doc in full.items() if pid not in set(mask)}   # what the neighbour merge can reach
        self.used, self.topic_used, self.seen = Counter(), Counter(), set()
        self._sentences, self._probes = {}, {}
        self.eligible = [pid for pid in sorted(full) if self.usable(pid) and store.is_financial(pid)]
        self.personal = [pid for pid in sorted(full) if self.usable(pid)
                         and (store.has_personal(pid) or not store.is_financial(pid))]
        self.tags = defaultdict(list)                 # (type, key) -> eligible ids
        oil = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(p) for p in P._OIL_SHOCKS['letter_phrases']) + r')(?:e?s)?(?!\w)', re.I)
        for pid in self.eligible:
            ann = self.ann(pid)
            for name, kind in (('events', 'event'), ('sectors', 'sector'), ('themes', 'theme'), ('companies', 'company')):
                for key in ann.get(name) or ():
                    self.tags[(kind, key)].append(pid)
            if oil.search(str(ann.get('summary') or '')):
                self.tags[('event', OIL)].append(pid)
        self.eligible_set = set(self.eligible)
        self.digest = defaultdict(set)                # (type, key) -> eligible ids its digest timeline quotes (the key passages)
        for topic in store.topics():
            row = store.digest(topic) or {}
            kind = row.get('kind')
            for point in row.get('timeline') or ():
                if isinstance(point, dict) and point.get('source_id') in self.eligible_set:
                    self.digest[(kind, topic)].add(point['source_id'])
        self.figures = defaultdict(set)               # eligible id -> metrics of numbers.jsonl rows it is a source of
        for row in store.numbers():
            for sid in {row.get('source_id'), *(row.get('source_ids') or ())}:
                if sid in self.eligible_set:
                    self.figures[sid].add(str(row['metric']))

    # passages
    def ann(self, pid):
        return self.store.annotation(pid) or {}

    def sentences(self, pid):
        if pid not in self._sentences:
            self._sentences[pid] = [norm(s) for s in evidence_sentences(self.full[pid].text) if norm(s)]
        return self._sentences[pid]

    def good(self, pid):
        return [(n, s) for n, s in enumerate(self.sentences(pid), 1) if len(s.split()) >= CT.MIN_WORDS and not V6.bad_sentence(s)]

    def unquotable(self, pid):
        """Sentence numbers of a passage a target cannot use. Either the sentence is not a whole sentence of the passage
        as served (alone, or merged with its neighbours the way `expand_hits_with_neighbors` serves it: a chunk that
        follows a sentence ending in '.)' loses its first sentence to it). Or it carries a personal cue next to personal
        facts (the passage or a merged neighbour has_personal / non-financial), which `drop_personal_sentences` removes
        from a served answer."""
        from config import ANSWER_CONTEXT_MAX_CHARS, retrieval_setting
        from src.retrieval.context import expand_hits_with_neighbors
        doc = self.full[pid]
        merged = expand_hits_with_neighbors([SearchHit(pid, doc.text, dict(doc.metadata), 1.)], self.visible,
                                            neighbors=retrieval_setting('ANSWER_CONTEXT_NEIGHBORS'), max_chars=ANSWER_CONTEXT_MAX_CHARS)[0]
        ids = merged.metadata.get('merged_ids') or [pid]
        personal = any(self.store.has_personal(i) or not self.store.is_financial(i) for i in ids)
        return [n for n, s in enumerate(self.sentences(pid), 1)
                if not (C.is_aligned(s, doc.text) and C.is_aligned(s, merged.text))
                or (personal and finance_answer._PERSONAL_SENTENCE.search(s))]

    def usable(self, pid):
        doc = self.full[pid]
        if not V6.BERKSHIRE_ID.match(pid) or pid in self.exclude or self.store.is_table(pid) or len(doc.text) < 250:
            return False
        return sum(c.isalpha() for c in doc.text) / len(doc.text) > .60 and len(self.good(pid)) >= 2

    def year(self, pid):
        return year_of(pid, self.full)

    def side(self, pid):
        return 'valid' if self.year(pid) in self.valid_years else 'train'

    def on(self, pids, side):
        return [p for p in pids if self.side(p) == side]

    def letters(self, pids, side):
        out = defaultdict(list)
        for pid in self.on(pids, side):
            out[self.year(pid)].append(pid)
        return out

    def stance(self, pid, target=None):
        return any(target is None or s.get('target') == target for s in self.ann(pid).get('stance') or ())

    def next_eligible(self, pid):
        nxt = self.full[pid].metadata.get('next_chunk_id')
        return nxt if nxt in self.eligible_set and self.year(nxt) == self.year(pid) else None

    def pick(self, pids, rng, prefer=None):
        """The least used of `pids` (ties: random), preferring those `prefer` accepts."""
        pool = list(pids)
        rng.shuffle(pool)
        return min(pool, key=lambda p: (not prefer(p) if prefer else False, self.used[p]))

    def key_passage(self, topic, target=None):
        """The preference of a topic's seeds: its digest-timeline passages and passages with a stance (on `target` when given)."""
        quoted = self.digest.get(topic, set())
        return lambda p: p in quoted or self.stance(p, target)

    def pick_years(self, letters, n, topic, rng):
        """n letter years of `letters` (year -> ids), those holding a digest-timeline passage of the topic first."""
        quoted = self.digest.get(topic, set())
        years = sorted(letters)
        rng.shuffle(years)
        years.sort(key=lambda y: not quoted & set(letters[y]))
        return sorted(years[:n])

    def pick_topic(self, keys, rng):
        pool = list(keys)
        rng.shuffle(pool)
        return min(pool, key=lambda k: self.topic_used[k])

    # topics
    def label(self, kind, key):
        if kind == 'event':
            return P._OIL_SHOCKS['label'] if key == OIL else P.EVENT_NAMES.get(key) or 'the ' + key.replace('_', ' ')
        if kind == 'sector':
            return SECTOR_LABELS.get(key, key.replace('_', ' '))
        return key if kind == 'company' else THEME_LABELS.get(key, key.replace('_', ' '))

    def topic(self, kind, key):
        out = {'type': kind, 'key': key, 'label': self.label(kind, key)}
        if kind == 'event':
            spans = P._OIL_SHOCKS['real_years'] if key == OIL else P._runs(self.store.events().get(key, {}).get('event_years', []))
            out['event_years'] = [list(s) for s in spans]
        return out

    def topic_of(self, pid):
        """The passage's own subject: its first stance target, else its first company, sector or theme tag."""
        ann = self.ann(pid)
        for s in ann.get('stance') or ():
            target = s.get('target')
            if target in (ann.get('sectors') or ()):
                return self.topic('sector', target)
            if target in (ann.get('themes') or ()):
                return self.topic('theme', target)
            if target:
                return self.topic('company', target)
        for name, kind in (('companies', 'company'), ('sectors', 'sector'), ('themes', 'theme')):
            if ann.get(name):
                return self.topic(kind, ann[name][0])
        return {'type': 'general', 'key': '', 'label': "Berkshire's business"}

    def phrases(self, kind, key):
        """Words the planner recognises for a topic, most natural first (the premise probe tries them in order)."""
        if kind == 'event' and key == OIL:
            return ['the oil crisis', 'the oil shocks', 'the energy crisis']
        if kind == 'event':
            words = [P.EVENT_NAMES.get(key, '')] + [a for a, spec in P.USER_EVENT_ALIASES.items() if spec.get('event') == key]
            words += list(self.store.events().get(key, {}).get('aliases', ()))[:3]
            words = [w if w.startswith('the ') else f'the {w}' for w in words if w]
        elif kind == 'sector':
            words = [SECTOR_LABELS.get(key, '')] + list(P.SECTOR_TERMS.get(key, ()))
        else:
            words = [THEME_LABELS.get(key, key.replace('_', ' '))] + list(P.THEME_TERMS.get(key, ()))
        return list(dict.fromkeys(w for w in words if w))

    def probes(self, kind, key):
        """(topic phrase, decade, premise note) for which the planner (FINANCE_PREMISE_CHECK) corrects the asked period."""
        if (kind, key) not in self._probes:
            out = []
            for phrase in self.phrases(kind, key)[:4]:
                for decade in range(P.FIRST_LETTER_YEAR // 10 * 10, P.LAST_LETTER_YEAR // 10 * 10 + 10, 10):
                    plan = P.plan_query(f'What did Buffett say about {phrase} in the {decade}s?', self.store, premise_check=True)
                    matched = {'event': plan.events, 'sector': plan.sectors, 'theme': plan.themes}[kind]
                    if plan.premise_note and (key in matched or (key == OIL and 'oil' in plan.premise_note.lower())):
                        out.append((phrase, decade, plan.premise_note))
                if out:
                    break
            self._probes[(kind, key)] = out
        return self._probes[(kind, key)]

    # seeds per kind: {source_ids, personal_ids, topic, ...} or None
    def seed(self, kind, side, rng, index):
        for _ in range(25):
            got = getattr(self, f'seed_{kind}')(side, rng, index)
            if got is None:
                return None
            key = (kind, tuple(got['source_ids']), tuple(got.get('personal_ids', ())), got.get('period_phrase'))
            if key not in self.seen:
                self.seen.add(key)
                return got
        return None

    def take(self, seed):
        self.used.update(seed['source_ids'] + seed.get('personal_ids', []))
        topic = seed['topic']
        self.topic_used[(topic['type'], topic['key'])] += 1

    def seed_crisis_period(self, side, rng, index):
        events = {key: self.letters(pids, side) for (kind, key), pids in self.tags.items() if kind == 'event'}
        events = {k: v for k, v in events.items() if len(v) >= 2}
        if not events:
            return None
        key = self.pick_topic([('event', k) for k in events], rng)[1]
        letters, prefer = events[key], self.key_passage(('event', key))
        ids = []
        for year in self.pick_years(letters, rng.randint(2, min(4, len(letters))), ('event', key), rng):
            first = self.pick(letters[year], rng, prefer=prefer)
            ids.append(first)
            rest = [p for p in letters[year] if p != first]
            if rest and rng.random() < .3:
                ids.append(self.pick(rest, rng, prefer=prefer))
        return {'source_ids': sorted(ids), 'topic': self.topic('event', key)}

    def seed_view_over_time(self, side, rng, index):
        topics = {}
        for (kind, key), pids in self.tags.items():
            if kind in ('sector', 'theme', 'company'):
                letters = self.letters(pids, side)
                if len(letters) >= 3 and len({y // 10 for y in letters}) >= 2:
                    topics[(kind, key)] = letters
        if not topics:
            return None
        kind, key = self.pick_topic(list(topics), rng)
        letters = topics[(kind, key)]
        n = rng.randint(3, min(5, len(letters)))
        by_decade = defaultdict(dict)
        for year, pids in letters.items():
            by_decade[year // 10][year] = pids
        years = [self.pick_years(by_decade[d], 1, (kind, key), rng)[0] for d in rng.sample(sorted(by_decade), min(n, len(by_decade)))]
        rest = self.pick_years({y: p for y, p in letters.items() if y not in years}, n - len(years), (kind, key), rng)
        prefer = self.key_passage((kind, key), key)
        ids = [self.pick(letters[y], rng, prefer=prefer) for y in sorted(years + rest)]
        return {'source_ids': sorted(ids), 'topic': self.topic(kind, key)}

    def _one_letter(self, pool, side, rng, more=.35):
        pool = self.on(pool, side)
        if not pool:
            return None
        first = self.pick(pool, rng)
        ids = [first]
        nxt = self.next_eligible(first)
        if nxt and rng.random() < more:
            ids.append(nxt)
        return ids

    def seed_single_year(self, side, rng, index):
        ids = self._one_letter([p for p in self.eligible if self.stance(p)] or self.eligible, side, rng)
        return ids and {'source_ids': ids, 'topic': self.topic_of(ids[0])}

    def seed_figure(self, side, rng, index):
        ids = self._one_letter(sorted(self.figures), side, rng, more=.25)
        if not ids:
            return None
        metrics = sorted(set().union(*(self.figures.get(p, set()) for p in ids)) - FIGURE_METRIC_SKIP)
        return {'source_ids': ids, 'topic': {'type': 'figure', 'key': ','.join(metrics),
                                             'label': ', '.join(m.replace('_', ' ') for m in metrics[:4]) or 'the reported figures'}}

    def seed_partial(self, side, rng, index):
        ids = self._one_letter(self.eligible, side, rng, more=.4)
        return ids and {'source_ids': ids, 'topic': self.topic_of(ids[0])}

    def seed_refusal(self, side, rng, index):
        ids = self._one_letter(self.eligible, side, rng, more=0)
        return ids and {'source_ids': ids, 'topic': self.topic_of(ids[0]), 'variant': 'near' if index % 2 == 0 else 'off'}

    def seed_premise_partial(self, side, rng, index):
        topics = [k for k in self.tags if k[0] in ('event', 'sector', 'theme') and self.on(self.tags[k], side)]
        rng.shuffle(topics)
        for kind, key in sorted(topics, key=lambda k: self.topic_used[k])[:60]:
            probes = self.probes(kind, key)
            if not probes:
                continue
            phrase, decade, note = rng.choice(probes)
            letters = self.letters(self.tags[(kind, key)], side)
            years = sorted(rng.sample(sorted(letters), rng.randint(1, min(3, len(letters)))))
            short = f"the {str(decade)[2:]}s"
            return {'source_ids': sorted(self.pick(letters[y], rng, prefer=self.stance) for y in years),
                    'topic': self.topic(kind, key), 'topic_phrase': phrase, 'period_phrase': f'the {decade}s',
                    'period_variants': [f'the {decade}s', f"the '{str(decade)[2:]}s", short, f'{decade}s'], 'probe_note': note}
        return None

    def seed_mixed_scope(self, side, rng, index):
        personal = self.on(self.personal, side)
        if not personal:
            return None
        person = self.pick(personal, rng)
        companies = set(self.ann(person).get('companies') or ())
        same = [p for p in self.on(self.eligible, side) if self.year(p) == self.year(person) and p != person]
        if not same:
            pool = self.on(self.eligible, side)
            if not pool:
                return None
            nearest = min(abs(self.year(p) - self.year(person)) for p in pool)
            same = [p for p in pool if abs(self.year(p) - self.year(person)) == nearest]
        fin = self.pick(same, rng, prefer=lambda p: bool(companies & set(self.ann(p).get('companies') or ())))
        return {'source_ids': [fin], 'personal_ids': [person], 'topic': self.topic_of(fin)}


def allocate(n, shares):
    """Largest-remainder counts per kind that sum to n."""
    raw = {k: n * shares[k] for k in KINDS}
    counts = {k: int(v) for k, v in raw.items()}
    for k in sorted(KINDS, key=lambda k: (-(raw[k] - counts[k]), KINDS.index(k)))[:n - sum(counts.values())]:
        counts[k] += 1
    return counts


def parse_shares(text):
    """'crisis_period=.2,refusal=.1,...' over SHARE (the given kinds replace their shares; the rest are rescaled)."""
    if not text:
        return dict(SHARE)
    given = {}
    for part in filter(None, (p.strip() for p in text.split(','))):
        key, _, value = part.partition('=')
        if key.strip() not in KINDS:
            raise ValueError(f'bad --shares entry {part!r}; kinds: {", ".join(KINDS)}')
        given[key.strip()] = float(value)
    rest = 1 - sum(given.values())
    other = sum(v for k, v in SHARE.items() if k not in given)
    if rest < 0 or (rest > 0 and not other):
        raise ValueError('--shares must sum to at most 1')
    return {k: given.get(k, SHARE[k] * rest / other if other else 0.) for k in KINDS}


# ------------------------------------------------------------------ teacher messages
TEACHER_SYSTEM = ("You build training data for a question-answering system over Warren Buffett's Berkshire Hathaway "
                  "shareholder letters. You write one investor question and pick the letter sentences that answer it. "
                  "Reply with one JSON object only.")
STYLE = ('QUESTION STYLE\n'
         '- Write ONE natural question in English, the way an investor asks it, in your own words. Examples of the style: '
         '"Did Buffett say anything about the oil crisis in the 80s?", "Give me Buffett\'s insights related to the 2008 crisis", '
         '"In 1999, does Buffett suggest any sector with potential to grow substantially?" End it with a question mark.\n'
         '- Never mention "the passage", "the text", "the excerpt" or what "the letter says"; use no outside facts; do not copy '
         'six or more consecutive words from the passages; share only the key nouns with them.\n'
         '- Do not compare two named periods (no "compare", "versus", "differ", "better than") and do not use collection-wide '
         'words ("over the years", "over the decades", "always", "never", "most", "overall", "across the letters"): such '
         'questions are answered by other routes.\n')
ANSWER_RULES = ('ANSWER SENTENCES\n'
                '- Pick, per letter, the sentences that answer the question, by id ("<passage id>#<number>"). They are copied '
                'verbatim into the answer, oldest letter first, so pick what answers the question and nothing else.\n'
                "- Prefer sentences that state Buffett's views, actions, figures, reasons or lessons over narration of events.\n"
                '- Every sentence must be understandable alone: if it starts with This, That, It, They, These, Such, However, '
                'But, And or a similar word that leans on the sentence before, also pick that previous sentence, or pick another.\n'
                f'- Complete sentences of at least {CT.MIN_WORDS} words; at most {CT.MAX_PER_LETTER} per letter and '
                f'{CT.MAX_SENTENCES} in all. Never pick a sentence marked "(not quotable)": the system cannot quote it '
                'whole, or it carries a personal detail the answer leaves out.\n')
SHAPES = {
    'answer': '{"question": "...?", "answer": [{"year": 1987, "sentences": ["1987_p0012#3", "1987_p0012#4"]}, '
              '{"year": 1990, "sentences": ["1990_p0040#2"]}]}',
    'partial': '{"question": "...?", "answer": [{"year": 1987, "sentences": ["1987_p0012#3"]}], '
               '"gap": {"ask": "the price Berkshire paid for X", "missing": ["X"]}}',
    'near': '{"question": "...?", "answer": [], "present": ["Name that is in the passages"], '
            '"missing": ["Name or number that is not in the passages"]}',
    'off': '{"question": "...?", "answer": [], "missing": ["Name that is not in the passages"]}',
    'mixed_scope': '{"question": "...?", "financial_part": "...", "personal_part": "...", '
                   '"answer": [{"year": 1990, "sentences": ["1990_p0040#2"]}], "personal_sentences": ["1990_p0041#3"]}',
}


def years_text(years):
    years = sorted(set(years))
    return years[0] if len(years) == 1 else ', '.join(map(str, years[:-1])) + f' and {years[-1]}'


def spans_text(spans):
    return ' and '.join(P._span_text(a, b) for a, b in spans) if spans else 'the period of these letters'


def passages_text(job):
    lines, year = [], None
    for p in job['passages']:
        if p['year'] != year:
            year = p['year']
            lines += ['', f'LETTER OF {year}']
        role = f" ({p['role'].upper()})" if job['kind'] == 'mixed_scope' else ''
        lines.append(f"PASSAGE {p['id']}{role}")
        blocked = set(p.get('unquotable') or ())
        lines += [f"{p['id']}#{n}{' (not quotable)' if n in blocked else ''}: {text}" for n, text in p['sentences']]
    return '\n'.join(lines).strip()


def task_text(job):
    kind, topic, years = job['kind'], job['topic'], job['years']
    label, when, year = topic['label'], years_text(years), years[0]
    if kind == 'crisis_period':
        return (f'TASK (crisis_period): these passages from the {when} letters discuss {label} (it happened in '
                f'{spans_text(topic.get("event_years"))}). Write a question about what Buffett said about this event: its causes, how '
                'it hit Berkshire or the economy, what he did, or what he learned. Name the event in everyday words (for example '
                '"the 2008 crisis" or "the oil crisis"); you may add a period, but only one these letters belong to. Answer from at '
                'least 2 of the letters, from every letter that answers the question.')
    if kind == 'view_over_time':
        return (f'TASK (view_over_time): these passages from the {when} letters discuss {label}. Write a question about how '
                f'Buffett\'s view of {label} developed over time (for example "How did Buffett\'s thinking on {label} evolve?" or '
                f'"How did Buffett\'s stance on {label} change over time?"), without naming single years. Answer from at least 3 '
                'letters spanning at least 2 decades.')
    if kind == 'single_year':
        return (f'TASK (single_year): these passages come from the {year} letter and touch on {label}. Write a question about a '
                f'view Buffett held or a fact he reported in {year}; the question must contain the year {year} (for example '
                f'"In {year}, ..."). Answer with 1-{CT.MAX_PER_LETTER} sentences of this letter.')
    if kind == 'figure':
        return (f'TASK (figure): these passages come from the {year} letter and state figures ({label}). Write a question that '
                'asks for the exact figures and what Buffett said about them (a reason, a comparison or a consequence), naming '
                f'the year or the business, for example "What did the float amount to in {year}, and why did Buffett value it?". '
                f'Not a bare lookup of one number. Answer with 1-{CT.MAX_PER_LETTER} sentences; at least one must state the figures.')
    if kind == 'premise_partial':
        phrase, period = job['topic_phrase'], job['period_phrase']
        return (f'TASK (premise_partial): these passages from the {when} letters discuss {label}. Write a question that asks what '
                f'Buffett said about {phrase} in {period}, a period whose letters do not discuss it. Use the words "{phrase}" (or a '
                f'close form of them) and "{period}" literally, for example "Did Buffett say anything about {phrase} in {period}?". '
                'The system adds a sentence that corrects the period; you only pick the sentences of these letters that tell what '
                f'Buffett said about {label}, from every letter that does.')
    if kind == 'partial':
        return (f'TASK (partial): these passages come from the {year} letter. Write ONE question with two parts joined naturally. '
                'One part is answered by the passages. The other part asks for a specific detail the passages do NOT give (a '
                'figure of another year, a named person or company they do not mention, a number they do not state); that detail '
                'must appear in your question as a distinctive capitalised name, year or number that does NOT occur in the '
                f'passages. Answer the covered part with 1-{CT.MAX_PER_LETTER} sentences. "gap.ask" is a short noun phrase that '
                'completes "The retrieved passages do not cover ___." (for example "the price Berkshire paid for Fechheimer"); '
                '"gap.missing" lists the 1-2 distinctive words or numbers of your question that are not in the passages.')
    if kind == 'refusal' and job.get('variant') == 'near':
        return ('TASK (refusal): write ONE plausible finance question about a company, person, business or event that IS '
                'discussed in these passages, but that asks for a specific detail the passages do NOT give: a figure, date, name '
                'or reason that is not stated, or something that did not happen. Stay inside the period of the letters (no year '
                'after 2024) and never ask for a stock, share or crypto price. The question must contain, as distinctive words '
                'or numbers (a capitalised name, a year or a number), both the subject that IS in the passages ("present") and '
                'the detail that is NOT ("missing"). Nothing in the passages answers it, so "answer" is [].')
    if kind == 'refusal':
        return ('TASK (refusal): write ONE plausible finance question in the same business area as these passages about a '
                "specific company, person, product or deal that these passages do not mention and that Buffett's letters most "
                'likely never discuss. Stay inside the period of the letters (no year after 2024) and never ask for a stock, '
                'share or crypto price. "missing" lists the distinctive capitalised names or numbers of your question that are '
                'not in the passages; "answer" is [].')
    return (f'TASK (mixed_scope): the FINANCIAL passages come from the business part of the {when} letter(s); the PERSONAL passage '
            'tells a personal story (family, health, age, hobbies, a joke, a life story). Write ONE question with two parts joined '
            'naturally: a financial part answered by the FINANCIAL passages and a personal part about the PERSONAL passage that '
            'is not about money. Pick answer sentences ONLY for the financial part, ONLY from FINANCIAL passages. List in '
            '"personal_sentences" the ids (1-4) of the PERSONAL sentences that would answer the personal part: they are never '
            'used in the answer. "financial_part" and "personal_part" name the two parts in a few words.')


def teacher_messages(job):
    kind = job['kind']
    shape = SHAPES.get(job.get('variant')) if kind == 'refusal' else SHAPES.get(kind, SHAPES['answer'])
    user = (f"Below are passages from Berkshire Hathaway shareholder letters. Every sentence has an id: the passage id, '#', "
            f"the sentence number.\n\n{passages_text(job)}\n\n{task_text(job)}\n\n{STYLE}\n"
            + (ANSWER_RULES + '\n' if kind in ANSWER_KINDS else '')
            + f'Reply with JSON only, exactly this shape (with the ids printed above, not the ones of this example):\n{shape}')
    return [{'role': 'system', 'content': TEACHER_SYSTEM}, {'role': 'user', 'content': user}]


def job_record(pools, kind, split, index, seed, identity):
    passages = []
    for role, ids in (('financial', seed['source_ids']), ('personal', seed.get('personal_ids', []))):
        for pid in ids:
            passages.append({'id': pid, 'year': pools.year(pid), 'role': role,
                             'sentences': [[n, s] for n, s in enumerate(pools.sentences(pid), 1)],
                             'unquotable': pools.unquotable(pid) if role == 'financial' else []})
    passages.sort(key=lambda p: (p['year'], p['role'] != 'financial', p['id']))
    job = {'id': f'fr4:{kind}:{split}:{index:04d}', 'kind': kind, 'split': split, 'variant': seed.get('variant'),
           'topic': seed['topic'], 'source_ids': seed['source_ids'], 'personal_ids': seed.get('personal_ids', []),
           'years': sorted({pools.year(p) for p in seed['source_ids']}), 'passages': passages, 'identity': identity}
    for key in ('topic_phrase', 'period_phrase', 'period_variants', 'probe_note'):
        if key in seed:
            job[key] = seed[key]
    job['messages'] = teacher_messages(job)
    return job


def order_jobs(jobs):
    """Kinds interleaved in proportion, so that any prefix of the file (a batch, a partial teacher run) has the mix."""
    size = Counter(j['kind'] for j in jobs)
    seen = Counter()
    keyed = []
    for job in jobs:
        keyed.append(((seen[job['kind']] + .5) / size[job['kind']], KINDS.index(job['kind']), job['id'], job))
        seen[job['kind']] += 1
    return [k[-1] for k in sorted(keyed, key=lambda k: k[:3])]


def build_jobs(full, store, exclude, valid_years, n, seed, shares=None, identity=None, mask=()):
    """-> (jobs, plan report). About VALID_SHARE of every kind's jobs come from valid letter years; a kind whose valid
    seeds run out falls back to train (`split_fallbacks`), one whose seeds run out altogether is short (`shortfalls`)."""
    shares = shares or SHARE
    pools = Pools(full, store, exclude, valid_years, mask)
    rng = random.Random(seed)
    counts = allocate(n, shares)
    valid = allocate(round(n * VALID_SHARE), {k: counts[k] / max(1, n) for k in KINDS})
    jobs, fallbacks, shortfalls = [], Counter(), Counter()
    index = Counter()
    for kind in PLAN_ORDER:
        for split, count in (('valid', min(valid[kind], counts[kind])), ('train', counts[kind] - min(valid[kind], counts[kind]))):
            for _ in range(count):
                got, side = pools.seed(kind, split, rng, index[kind]), split
                if got is None and split == 'valid':
                    got, side = pools.seed(kind, 'train', rng, index[kind]), 'train'
                    fallbacks[kind] += got is not None
                if got is None:
                    shortfalls[kind] += 1
                    continue
                pools.take(got)
                jobs.append(job_record(pools, kind, side, index[(kind, side)], got, identity or {}))
                index[(kind, side)] += 1
                index[kind] += 1
    report = {'requested': counts, 'requested_valid': valid, 'jobs': dict(Counter(j['kind'] for j in jobs)),
              'per_split': {s: dict(Counter(j['kind'] for j in jobs if j['split'] == s)) for s in ('train', 'valid')},
              'split_fallbacks': dict(fallbacks), 'shortfalls': dict(shortfalls), 'eligible_passages': len(pools.eligible),
              'personal_passages': len(pools.personal), 'reused_passages': sum(1 for v in pools.used.values() if v > 1)}
    return order_jobs(jobs), report


def load_corpus():
    return {d.id: d for d in common.load_chunks_as_docs(common.CORPUS)}


def load_store():
    from src.knowledge import KnowledgeStore
    return KnowledgeStore.load()


def plan_identity(seed, n, valid_years, shares):
    return {'corpus_sha256': file_sha(common.CORPUS), 'knowledge_manifest_sha256': file_sha(ROOT / 'data/knowledge/MANIFEST'),
            'exclude_sha256': file_sha(EXCLUDE_FILE), 'mask_sha256': file_sha(MASK_FILE), 'plan_seed': seed, 'n': n,
            'valid_years': sorted(valid_years), 'shares': shares}


def plan_command(args):
    full, store = load_corpus(), load_store()
    exclude = read_ids(args.exclude_file)
    valid_years = V6.valid_years_of(full.values())
    shares = parse_shares(args.shares)
    identity = plan_identity(args.seed, args.n, valid_years, shares)
    jobs, report = build_jobs(full, store, exclude, valid_years, args.n, args.seed, shares, identity, read_ids(args.mask_file))
    out = Path(args.out)
    V6.write_jsonl_file(out, jobs)
    sizes = sorted(len(j['messages'][1]['content']) for j in jobs)
    report.update(identity=identity, jobs_sha256=file_sha(out), teacher_message_chars={'median': sizes[len(sizes) // 2], 'max': sizes[-1]}
                  if sizes else {})
    common.atomic_json(out.with_name('plan_manifest.json'), report)
    print(f"jobs={len(jobs)} per_kind={report['jobs']} valid={report['per_split']['valid']} fallbacks={report['split_fallbacks']} "
          f"shortfalls={report['shortfalls']} eligible={report['eligible_passages']} -> {out}")


# ------------------------------------------------------------------ export batches for the Claude teacher panes
TEACHER_MD = """# FT-r4 finance teacher: instructions for one Claude pane

You are the teacher of the FT-r4 finance training data. You write a question and pick answer sentences; you never write
the answer text yourself (the builder copies your chosen sentences verbatim).

Files (absolute paths):
- batches: `{batches}/batch_NNN.jsonl` (one job per line; {n_batches} batches, at most {size} jobs each)
- your output: `{out}/teacher_batch_NNN.jsonl` (the same NNN as the batch)
- the checker: `{check}`

For your batch NNN:
1. Read `{batches}/batch_NNN.jsonl`. Every line is one job. For each job, `messages[1].content` is the task: the
   passages with sentence ids, the task for the job's kind, the style and answer rules and the JSON shape.
   `messages[0].content` is your role. Follow the task text exactly; it is the same text for every teacher.
2. Write `{out}/teacher_batch_NNN.jsonl`: one line per job, the JSON object the task asks for plus the job id, e.g.
   `{{"id": "fr4:crisis_period:train:0003", "question": "...?", "answer": [{{"year": 2008, "sentences": ["2008_p0011#2"]}}]}}`
   No prose, no code fences, no blank lines between objects. Keep the order of the batch.
   Only when no sentence of a job's passages can answer any sensible question of its kind, write
   `{{"id": "<job id>", "skip": "<one short reason>"}}`.
3. Run the checker and fix every REJECT or MISSING line (rewrite that job's line in the file), then run it again until
   it ends with `OK`:
   `env -u PYTHONPATH {python} {check} --batch {batches}/batch_NNN.jsonl {out}/teacher_batch_NNN.jsonl`

Rules:
- Use only the passages of the job; no outside facts, no other files of the repository, never an evaluation set.
- Do not edit the batch files. Do not run anything else than the checker.
- The checker also runs the finance planner on your question: a question it would answer from a digest, the numbers
  table, a superlative ranking or a per-period comparison is rejected, and a premise_partial question must make the
  planner correct its period. Rephrase as the hint says.
"""


def export_batches(jobs_path, size, out_dir):
    jobs = V6.read_jsonl_file(jobs_path)
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir = out_dir.parent / 'out'
    teacher_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for number, start in enumerate(range(0, len(jobs), size)):
        name = f'batch_{number:03d}'
        part = jobs[start:start + size]
        V6.write_jsonl_file(out_dir / f'{name}.jsonl', part)
        index[name] = {'jobs': [j['id'] for j in part], 'kinds': dict(Counter(j['kind'] for j in part)),
                       'teacher_file': str(teacher_dir / f'teacher_{name}.jsonl')}
    common.atomic_json(out_dir / 'index.json', {'jobs_file': str(Path(jobs_path).resolve()), 'jobs_sha256': file_sha(jobs_path),
                                                'size': size, 'batches': index})
    (out_dir / 'TEACHER.md').write_text(TEACHER_MD.format(batches=out_dir.resolve(), out=teacher_dir.resolve(), n_batches=len(index),
                                                          size=size, check=HERE / 'check_teacher.py', python=ROOT / '.venv/bin/python'),
                                        encoding='utf-8')
    return index


# ------------------------------------------------------------------ fake teacher (deterministic picks; pipeline smoke and tests)
FAKE_ENTITY = 'Quorvant Industries'          # a name no letter mentions (the refusal / gap detail)
FAKE_BUYER = 'Zephyrine Holdings'


def _good_sentences(passage, *, figure=False):
    out, blocked = [], set(passage.get('unquotable') or ())
    for n, text in passage['sentences']:
        if len(text.split()) >= 8 and not V6.bad_sentence(text) and not C.DEPENDENT_START.match(text) \
                and n not in blocked and (not figure or CT.has_figure(text)):
            out.append(f"{passage['id']}#{n}")
    return out


def _present_name(job):
    """A capitalised name of the passages the refusal check accepts as distinctive (for a 'near' refusal)."""
    for p in job['passages']:
        for _, text in p['sentences']:
            for name in re.findall(r"(?<=[a-z,;] )([A-Z][a-zA-Z]{3,}(?: [A-Z][a-zA-Z]{2,})?)", text):
                if C.distinctive_missing(name) and name.lower() not in ('buffett', 'charlie', 'berkshire'):
                    return name
    return None


def fake_questions(job):
    """Template questions of a job, tried in order (the first one check_teacher accepts is used)."""
    kind, label, year = job['kind'], job['topic'].get('label') or "Berkshire's business", job['years'][0]
    if kind == 'crisis_period':
        return [f'Did Buffett say anything about {label}?', f'What lessons did Buffett draw from {label}?',
                f'How did Buffett see {label} unfold for Berkshire?']
    if kind == 'view_over_time':
        return [f"How did Buffett's thinking about {label} evolve over time?", f"How did Buffett's stance on {label} shift as time went on?"]
    if kind == 'single_year':
        return [f'In {year}, what did Buffett say about {label}?', f'In {year}, what was Buffett telling shareholders about {label}?',
                f'In {year}, did Buffett share any notable view on {label}?', f'Back in {year}, what point did Buffett make on {label}?']
    if kind == 'figure':
        return [f'In {year}, which figures did Buffett report on {label}, and why did they matter to him?',
                f'What numbers did Buffett give for {label} in {year}, and what was his take on them?']
    if kind == 'premise_partial':
        phrase, period = job['topic_phrase'], job['period_phrase']
        return [f'Did Buffett say anything about {phrase} in {period}?', f"What were Buffett's thoughts on {phrase} in {period}?"]
    if kind == 'partial':
        return [f'In {year}, what did Buffett say about {label}, and how much did {FAKE_BUYER} pay for its stake?',
                f'What was Buffett telling shareholders in {year} about {label}, and what did {FAKE_BUYER} pay for its stake?']
    if kind == 'refusal':
        present = _present_name(job) if job.get('variant') == 'near' else None
        if present:
            return [f'What did {present} pay when it bought {FAKE_ENTITY}?', f'How much did {present} earn from its deal with {FAKE_ENTITY}?']
        return [f'What did Buffett say about {FAKE_ENTITY} and its {label} business?', f'Did Buffett ever invest in {FAKE_ENTITY}?']
    return [f'What did Buffett say about {label} in {year}, and how old was his mother at the time?',
            f'What did Buffett report about {label} in {year}, and what personal story did he tell about his wife?']


def fake_answer(job, store=None):
    """A deterministic teacher line for one job: a template question and the first usable sentences of every letter."""
    kind = job['kind']
    financial = [p for p in job['passages'] if p['role'] == 'financial']
    per = {'single_year': 2, 'figure': 2}.get(kind, 1)
    answer = []
    for y in job['years']:
        ids = [sid for p in financial if p['year'] == y for sid in _good_sentences(p, figure=kind == 'figure')]
        if kind == 'figure' and not ids:
            ids = [sid for p in financial if p['year'] == y for sid in _good_sentences(p)]
        if ids:
            answer.append({'year': y, 'sentences': ids[:per]})
    out = {'id': job['id'], 'answer': answer}
    if kind == 'partial':
        out['gap'] = {'ask': f'what {FAKE_BUYER} paid for its stake', 'missing': [FAKE_BUYER]}
    elif kind == 'refusal':
        present = _present_name(job) if job.get('variant') == 'near' else None
        out.update(answer=[], missing=[FAKE_ENTITY], **({'present': [present]} if present else {}))
    elif kind == 'mixed_scope':
        personal = [p for p in job['passages'] if p['role'] == 'personal']
        out.update(financial_part=job['topic'].get('label') or 'the business', personal_part='a family story',
                   personal_sentences=[f"{personal[0]['id']}#{personal[0]['sentences'][0][0]}"] if personal else [])
    questions = fake_questions(job)
    for question in questions:
        if CT.validate(job, {**out, 'question': question}, store=store)[0] is None:
            return {**out, 'question': question}
    return {**out, 'question': questions[0]}


def fake_teacher_command(args):
    jobs = V6.read_jsonl_file(args.jobs)
    store = load_store()
    V6.write_jsonl_file(Path(args.out), [fake_answer(j, store) for j in jobs])
    print(f'fake teacher lines={len(jobs)} -> {args.out}')


# ------------------------------------------------------------------ serving replay
class NeverLLM:
    """The cloud provider's name ("remote": no llama window fit, as the Modal app serves); never generates."""
    provider_name = 'remote'
    model = 'never'

    def generate(self, prompt, max_new_tokens=None):  # pragma: no cover - guard
        raise RuntimeError('build_ft_finance never generates')

    def generate_stream(self, prompt, max_new_tokens=None):  # pragma: no cover - guard
        raise RuntimeError('build_ft_finance never generates')


class MaskedStore:
    """The knowledge store as finance_search sees it for a training row: masked passages are not financial, so the search
    drops them before the rerank and the next candidates take their places. Every other lookup is the store's."""

    def __init__(self, store, mask):
        self._store, self._mask = store, frozenset(mask)

    def is_financial(self, pid):
        return pid not in self._mask and self._store.is_financial(pid)

    def __len__(self):
        return len(self._store)

    def __getattr__(self, name):
        return getattr(self._store, name)


@dataclass
class Served:
    route: str
    plan: Any
    query: str
    prompt: str
    system: str
    user: str
    hits: List[SearchHit]
    retrieved_ids: List[str]
    per_period: bool
    debug: Dict[str, Any]
    masked_pool: List[str] = field(default_factory=list)
    masked_neighbours: List[str] = field(default_factory=list)


class Router:
    """`ask_flow.finance_route` for one question, with the mask and the optional injection (module docstring)."""

    def __init__(self, retriever, full, store, mask, llm=None, cache=None):
        self.full, self.store, self.mask, self.cache = full, store, frozenset(mask), cache
        self.masked_store = MaskedStore(store, mask)      # one instance: knowledge_source memoises per store object
        visible = {pid: doc for pid, doc in full.items() if pid not in self.mask}
        self.state = {'retriever': retriever, 'docs': list(visible.values()), 'docs_by_id': visible, 'llm': llm or NeverLLM(),
                      'knowledge': store}

    @classmethod
    def load(cls, full, store, mask, cache_path=None):
        """The /ask retriever stack of backend startup (corpus v3, faiss_v3, bge embedder, bge reranker), on this machine's
        device; embeddings and reranker scores are cached on disk (scripts/eval/served_context_recall.py)."""
        sys.path.insert(0, str(ROOT / 'scripts/eval'))
        from config import EMBEDDING_DEVICE, EMBEDDING_MODEL_PRIMARY
        from served_context_recall import CachedEmbedder, CachedReranker, _Cache
        from src.retrieval.reranker import CrossEncoderReranker
        from src.retrieval.retriever import Retriever
        from src.storage.embeddings import BGEEmbedder
        from src.storage.faiss_store import FaissStore
        from src.storage.index_manifest import ensure_index_identity
        docs = list(full.values())
        embedder = BGEEmbedder(model_name=EMBEDDING_MODEL_PRIMARY, device=EMBEDDING_DEVICE)
        vectors = FaissStore(persist_dir=ROOT / 'data/indices/faiss_v3', dim=embedder.dimension)
        ensure_index_identity(vectors, backend='faiss', corpus=common.CORPUS, docs=docs, model_name=embedder.model_name,
                              dimension=embedder.dimension)
        reranker = CrossEncoderReranker(device=EMBEDDING_DEVICE)
        cache = _Cache(Path(cache_path)) if cache_path else None
        if cache is not None:
            embedder, reranker = CachedEmbedder(embedder, cache), CachedReranker(reranker, cache)
        return cls(Retriever(vector_store=vectors, embedder=embedder, docs=docs, reranker=reranker), full, store, mask, cache=cache)

    def save_cache(self):
        if self.cache is not None:
            self.cache.save()

    @contextmanager
    def active(self, search=None):
        from src.services import ask_flow
        previous, real = ask_flow._state, ask_flow.finance_search
        ask_flow._state = self.state
        if search is not None:
            ask_flow.finance_search = search
        try:
            yield ask_flow
        finally:
            ask_flow._state, ask_flow.finance_search = previous, real

    def serve(self, question, inject=()):
        from src.generation.providers.llama_provider import split_prompt
        from src.retrieval import finance_policy
        from src.services.schemas import AskRequest
        record = {}
        front_docs = [self.full[pid] for pid in inject]

        def search(plan, retriever, store, k=None, pinned=()):
            hits, debug = finance_policy.finance_search(plan, retriever, self.masked_store, k=k, pinned=pinned)
            record['pool'] = sorted(set(debug.get('stage_pool') or ()) & self.mask)
            if front_docs:
                by_id = {h.id: h for h in hits}
                front = [by_id.get(d.id) or SearchHit(d.id, d.text, dict(d.metadata), 1.0) for d in front_docs]
                ids = {h.id for h in front}
                hits = (front + [h for h in hits if h.id not in ids])[:max(len(hits), len(front))]
            return hits, debug

        with self.active(search) as ask_flow:
            route = ask_flow.finance_route(AskRequest(query=question, history=[], expand_query=False))
        system, user = split_prompt(route.prompt) if route.prompt else ('', '')
        neighbours = sorted({doc.metadata.get(key) for h in route.hits for doc in [self.full.get(h.id)] if doc is not None
                             for key in ('previous_chunk_id', 'next_chunk_id') if doc.metadata.get(key) in self.mask})
        return Served(route.route, route.plan, route.query, route.prompt, system, user, list(route.hits),
                      [h.id for h in route.retrieved_hits], bool(route.per_period), dict(route.debug), record.get('pool', []),
                      neighbours)

    def postprocess(self, prompt, hits, target):
        return replay_postprocess(prompt, hits, target, self.store, self)


def replay_postprocess(prompt, hits, target, store, router=None):
    """`ask_flow._finalize_once` on the target as if the model had written it -> (served answer, trace)."""
    trace: Dict[str, Any] = {}
    if router is not None:
        with router.active() as ask_flow:
            answer, _ = ask_flow._finalize_once(prompt, hits, target, trace)
        return answer, trace
    from src.services import ask_flow
    previous = ask_flow._state
    ask_flow._state = {**previous, 'knowledge': store}
    try:
        answer, _ = ask_flow._finalize_once(prompt, hits, target, trace)
    finally:
        ask_flow._state = previous
    return answer, trace


def route_reason(served):
    if served.route != 'reader':
        return f'route_{served.route}'
    if served.per_period:
        return 'route_comparison'
    if not served.hits or not served.prompt:
        return 'no_context'
    return None


# ------------------------------------------------------------------ tokens
class GemmaCounter:
    """Gemma 4 12B tokens (local tokenizer.json, revision GEMMA_REVISION) of a chat rendered in the Gemma turn format.

    The local tokenizer files carry no chat template, so the turns are rendered as <bos><|turn>role\\n...<turn|>\\n with
    the model turn last; the markers are single special tokens, so the count is the content tokens plus a few."""

    def __init__(self, path=GEMMA_TOKENIZER):
        from tokenizers import Tokenizer
        self.path = Path(path)
        self.tok = Tokenizer.from_file(str(self.path))

    def count(self, text):
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def prompt(self, system, user):
        return self.count(f'<bos><|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n')

    def target(self, text):
        return self.count(f'{text}<turn|>\n')


# ------------------------------------------------------------------ rows
_PASSAGE_HEAD = re.compile(r'(?m)^\[(\d+)\] \(year=([^,\n]*), source=([^)\n]*)\)\n')
_LETTER_HEAD = re.compile(r'\n\n(?:LETTER OF \S+|LETTER YEAR UNKNOWN) \(passages? \[\d+\](?:-\[\d+\])?\)\s*$')


def parse_user_passages(user):
    """[(year, source, text)] by slot from a v3 reader user turn (letter-year headers taken off)."""
    body = user.split('BEGIN UNTRUSTED PASSAGES\n', 1)[-1].split('\n\nEND UNTRUSTED PASSAGES', 1)[0]
    heads = list(_PASSAGE_HEAD.finditer(body))
    out = []
    for i, m in enumerate(heads):
        text = body[m.end():heads[i + 1].start() if i + 1 < len(heads) else len(body)]
        out.append((m.group(2), m.group(3), _LETTER_HEAD.sub('', text).strip()))
    return out


def static_hits(row):
    """The served hits rebuilt from a stored row: text from its user turn, ids / merged ids / topic flag from `served`."""
    hits = []
    for (year, source, text), slot in zip(parse_user_passages(row['messages'][1]['content']), row['served']):
        metadata = {'year': int(year) if str(year).isdigit() else year, 'source_file': source, 'merged_ids': slot['merged_ids']}
        if slot.get('topic'):
            metadata[finance_answer.TOPIC_FLAG] = True
        hits.append(SearchHit(slot['id'], text, metadata, 0.))
    return hits


def served_ids(hits):
    return list(dict.fromkeys(pid for h in hits for pid in (h.id, *(h.metadata.get('merged_ids') or ()))))


def build_target(sentences, premise=None, gap=None):
    """One line per letter year (oldest first): the sentences verbatim, each ending with ' [n]'; the premise note opens the
    first line (with_premise joins it with a space), a partial row ends with its gap line."""
    lines, year = [], None
    for s in sorted(sentences, key=CT.chrono_key):
        part = f"{norm(s['text'])} [{s['slot']}]"
        if s['year'] == year:
            lines[-1] += ' ' + part
        else:
            lines.append(part)
            year = s['year']
    if premise and lines:
        lines[0] = f'{norm(premise)} {lines[0]}'
    if gap:
        lines.append(C.gap_line(gap['ask']))
    return '\n'.join(lines)


@dataclass
class Context:
    full: Dict[str, Any]
    store: Any
    exclude: set
    mask: frozenset
    valid_years: set
    counter: Any
    max_tokens: int = MAX_TOKENS
    prompt_version: str = PROMPT_VERSION


def row_for(job, plan, served, injected, ctx):
    """The training row of one validated teacher answer on its served context -> (row, reason)."""
    texts = [h.text for h in served.hits]
    sentences = []
    for c in plan['chosen']:
        slot = C.lowest_holder(c['text'], texts)
        if slot is None:
            return None, 'sentence_not_servable'
        sentences.append({'sid': c['sid'], 'passage_id': c['passage_id'], 'year': c['year'], 'n': c['n'], 'text': c['text'],
                          'slot': slot})
    note = served.plan.premise_note
    target = REFUSAL_LINE if job['kind'] == 'refusal' else build_target(sentences, note, plan.get('gap'))
    forbidden = []
    if job['kind'] == 'mixed_scope':
        personal = {s for p in job['passages'] if p['role'] == 'personal' for _, s in p['sentences']}
        forbidden = sorted(personal | {CT.sentence_table(job)[sid]['text'] for sid in plan.get('personal_sids', ())})
    messages = [{'role': 'system', 'content': served.system}, {'role': 'user', 'content': served.user},
                {'role': 'assistant', 'content': target}]
    tokens = {'prompt': ctx.counter.prompt(served.system, served.user), 'target': ctx.counter.target(target)}
    tokens['total'] = tokens['prompt'] + tokens['target']
    from src.generation.evidence_gate import assess_evidence
    gate = assess_evidence(served.query, served.hits)
    soft = os.getenv('EVIDENCE_GATE_SOFT') != '0' and .15 <= gate.best_overlap < .30
    p = served.plan
    row = {'id': job['id'], 'kind': job['kind'], 'split': job['split'], 'variant': job.get('variant'), 'question': plan['question'],
           'served_query': served.query, 'prompt_version': ctx.prompt_version, 'route': served.route,
           'plan': {'scope': p.scope, 'kind': p.kind, 'events': list(p.events), 'sectors': list(p.sectors),
                    'companies': list(p.companies), 'themes': list(p.themes), 'year_range': list(p.year_range or ()) or None,
                    'premise_note': note},
           'topic': job['topic'], 'seed_ids': job['source_ids'], 'personal_ids': job.get('personal_ids', []),
           'source_ids': sorted({s['passage_id'] for s in sentences}), 'sentences': sentences, 'forbidden': forbidden,
           'gap': plan.get('gap'), 'absent_terms': plan.get('missing') or (plan.get('gap') or {}).get('missing') or [],
           'present_terms': plan.get('present') or [],
           'served': [{'id': h.id, 'merged_ids': list(h.metadata.get('merged_ids') or [h.id]),
                       'topic': bool(h.metadata.get(finance_answer.TOPIC_FLAG))} for h in served.hits],
           'served_ids': served_ids(served.hits), 'retrieved_ids': served.retrieved_ids,
           'masked': {'pool': served.masked_pool, 'neighbours': served.masked_neighbours}, 'injected': injected,
           'gate': {'sufficient': bool(gate.sufficient), 'soft': bool(soft), 'overlap': round(gate.best_overlap, 3)},
           'retry_would_fire': job['kind'] == 'refusal' and finance_answer.partial_retry_prompt(served.prompt, REFUSAL_LINE, served.hits) is not None,
           'window_tokens': served.debug.get('prompt_tokens'), 'token_source': served.debug.get('token_source'),
           'tokens': tokens, 'messages': messages}
    return row, None


def holder_chunks(text, hit, full):
    """Corpus chunks merged into `hit` that hold `text` as one of their own sentences."""
    return [pid for pid in (hit.metadata.get('merged_ids') or [hit.id]) if pid in full and C.is_aligned(text, full[pid].text)]


def postprocess_problems(row, prompt, hits, store, router=None):
    target = row['messages'][2]['content']
    answer, trace = (router.postprocess(prompt, hits, target) if router is not None
                     else replay_postprocess(prompt, hits, target, store))
    if row['kind'] == 'refusal':
        return [] if answer == REFUSAL_LINE else ['postprocess_changes_refusal']
    if answer == REFUSAL_LINE:
        return ['postprocess_refuses']
    out = []
    validator = trace.get('validator') or {}
    if validator.get('dropped'):
        out.append('postprocess_drops_sentence')
    if validator.get('deduped'):
        out.append('postprocess_dedupes')
    if validator.get('repaired'):
        out.append('postprocess_repairs_citation')
    flat = norm(answer)
    if any(f"{norm(s['text'])} [{s['slot']}]" not in flat for s in row['sentences']):
        out.append('postprocess_loses_sentence')
    note = row['plan'].get('premise_note')
    if note and not flat.startswith(norm(note)):
        out.append('premise_not_first')
    if row.get('gap') and C.gap_line(row['gap']['ask']) not in flat:
        out.append('postprocess_drops_gap_line')
    return out


def check_row(row, hits, ctx, prompt=None, router=None):
    """Offline checks of one row against its served hits (assemble and verify). -> list of problems ([] = keep)."""
    messages = row.get('messages') or []
    if [m.get('role') for m in messages] != ['system', 'user', 'assistant']:
        return ['messages_shape']
    system, user, target = (m['content'] for m in messages)
    problems = []
    if system != system_prompt(row.get('prompt_version') or ctx.prompt_version):
        problems.append('system_prompt_mismatch')
    prompt = prompt or f'{system}\n\n{user}'
    if not finance_answer.is_v3_prompt(prompt):
        problems.append('not_v3_reader_prompt')
    if len(parse_user_passages(user)) != len(hits) or not hits:
        return problems + ['passage_slots']
    texts = [h.text for h in hits]
    if set(served_ids(hits)) & ctx.mask:
        problems.append('masked_in_context')
    years = {s['year'] for s in row['sentences']}
    if any((y in ctx.valid_years) != (row['split'] == 'valid') for y in years):
        problems.append('split_straddle')
    seed_texts = [ctx.full[pid].text for pid in row['seed_ids'] if pid in ctx.full]
    if C.question_overlap(row['question'], seed_texts) > QUESTION_OVERLAP_MAX:
        problems.append('question_overlap')
    if row['kind'] == 'refusal':
        if target != REFUSAL_LINE:
            problems.append('refusal_target_not_exact')
        if not row['absent_terms'] or any(V6.term_present(t, texts) for t in row['absent_terms']):
            problems.append('refusal_term_served')
        if row.get('present_terms') and not any(V6.term_present(t, texts) for t in row['present_terms']):
            problems.append('refusal_topic_not_served')
    else:
        slots = [C.lowest_holder(s['text'], texts) for s in row['sentences']]
        if not row['sentences'] or any(slot is None for slot in slots):
            problems.append('sentence_not_servable')
        elif slots != [s['slot'] for s in row['sentences']]:
            problems.append('citation_not_lowest_holder')
        elif target != build_target(row['sentences'], row['plan'].get('premise_note'), row.get('gap')):
            problems.append('target_mismatch')
        if [CT.chrono_key(s) for s in row['sentences']] != sorted(CT.chrono_key(s) for s in row['sentences']):
            problems.append('not_chronological')
        for s, slot in zip(row['sentences'], slots):
            holders = holder_chunks(s['text'], hits[slot - 1], ctx.full) if slot else []
            if s['passage_id'] in ctx.exclude or (holders and all(h in ctx.exclude for h in holders)):
                problems.append('sentence_from_excluded')
                break
        if row['kind'] == 'premise_partial' and not row['plan'].get('premise_note'):
            problems.append('premise_not_fired')
        if row.get('gap') and any(V6.term_present(t, texts) for t in row['gap']['missing']):
            problems.append('partial_gap_term_served')
        if row['kind'] == 'mixed_scope':
            personal = any(norm(f) in norm(target) for f in row.get('forbidden') or ())
            for s, slot in zip(row['sentences'], slots):
                if slot and finance_answer._PERSONAL_SENTENCE.search(s['text']) and any(
                        ctx.store.has_personal(pid) or not ctx.store.is_financial(pid) for pid in holder_chunks(s['text'], hits[slot - 1], ctx.full)):
                    personal = True
            if personal:
                problems.append('personal_sentence')
    if 'passage_slots' not in problems and 'sentence_not_servable' not in problems:
        problems += postprocess_problems(row, prompt, hits, ctx.store, router)
    tokens = {'prompt': ctx.counter.prompt(system, user), 'target': ctx.counter.target(target)}
    tokens['total'] = tokens['prompt'] + tokens['target']
    if row.get('tokens') and row['tokens'] != tokens:
        problems.append('token_count_mismatch')
    if tokens['total'] > ctx.max_tokens:
        problems.append('too_long')
    return problems


def attempt(job, record, router, ctx):
    """One job + its teacher line -> (row or None, attempt record)."""
    started = time.monotonic()
    info = {'id': job['id'], 'kind': job['kind'], 'split': job['split'], 'kept': False, 'reason': None, 'problems': []}

    def done(reason, row=None, problems=()):
        info.update(reason=reason, problems=list(problems) or ([reason] if reason else []), kept=reason is None,
                    seconds=round(time.monotonic() - started, 2))
        if row is not None:
            info.update(tokens=row['tokens'], injected=len(row['injected']), masked_pool=len(row['masked']['pool']),
                        premise=bool(row['plan'].get('premise_note')), plan_scope=row['plan']['scope'], gate=row['gate'])
        return (row if reason is None else None), info

    if record is None:
        return done('no_teacher_output')
    if isinstance(record.get('skip'), str) and record['skip'].strip():
        info['skip'] = record['skip'].strip()
        return done('teacher_skip')
    reason, plan = CT.validate(job, record, store=ctx.store)
    if reason:
        return done(f'teacher_{reason}')
    served = router.serve(plan['question'])
    reason = route_reason(served)
    if reason:
        return done(reason)
    injected = []
    texts = [h.text for h in served.hits]
    missing = list(dict.fromkeys(c['passage_id'] for c in plan['chosen'] if C.lowest_holder(c['text'], texts) is None))
    if missing:
        served = router.serve(plan['question'], inject=missing)
        reason = route_reason(served)
        if reason:
            return done(reason)
        for pid in missing:
            slot = next((n for n, h in enumerate(served.hits, 1) if pid == h.id or pid in (h.metadata.get('merged_ids') or ())), None)
            injected.append({'passage_id': pid, 'slot': slot})
    row, reason = row_for(job, plan, served, injected, ctx)
    if reason:
        return done(reason)
    problems = check_row(row, served.hits, ctx, served.prompt, router)
    return done(problems[0] if problems else None, row, problems)


# ------------------------------------------------------------------ assemble
def read_teacher(path):
    """id -> teacher line; `path` is a file or a directory of teacher_*.jsonl. A later line for the same id wins."""
    path = Path(path).expanduser()
    files = sorted(path.glob('teacher_*.jsonl')) if path.is_dir() else [path]
    out = {}
    for f in files:
        for _, obj in CT.read_lines(f):
            if isinstance(obj, dict) and isinstance(obj.get('id'), str):
                out[obj['id']] = obj
    return out, files


def histogram(values, edges):
    out = {}
    for lo, hi in zip(edges, (*edges[1:], None)):
        label = f'{lo}-{hi}' if hi is not None else f'>{lo}'
        out[label] = sum(1 for v in values if v >= lo and (hi is None or v < hi))
    return out


def percentiles(values):
    if not values:
        return {}
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, max(0, -(-len(ordered) * q // 100) - 1))]  # noqa: E731 - nearest rank
    return {'p50': pick(50), 'p90': pick(90), 'max': ordered[-1], 'n': len(ordered)}


def manifest_for(jobs, attempts, rows, settings, identity):
    per_kind = {}
    for kind in KINDS:
        mine = [a for a in attempts if a['kind'] == kind]
        kept = [a for a in mine if a['kept']]
        per_kind[kind] = {'jobs': len([j for j in jobs if j['kind'] == kind]), 'attempted': len(mine), 'kept': len(kept),
                          'keep_rate': round(len(kept) / len(mine), 3) if mine else None,
                          'rows': {s: sum(1 for r in rows if r['kind'] == kind and r['split'] == s) for s in ('train', 'valid')},
                          'drop_reasons': dict(Counter(a['reason'] for a in mine if a['reason']).most_common())}
    measured = [a for a in attempts if a.get('tokens')]
    tokens = {part: {'histogram': histogram([a['tokens'][part] for a in measured], TOKEN_EDGES if part != 'target' else TARGET_EDGES),
                     'kept': percentiles([r['tokens'][part] for r in rows]), 'all_measured': percentiles([a['tokens'][part] for a in measured])}
              for part in ('prompt', 'target', 'total')}
    valid_rows = sum(r['split'] == 'valid' for r in rows)
    return {'identity': identity, 'settings': settings, 'jobs': len(jobs), 'attempted': len(attempts), 'kept': len(rows),
            'keep_rate': round(len(rows) / len(attempts), 3) if attempts else None, 'per_kind': per_kind,
            'drop_reasons': dict(Counter(a['reason'] for a in attempts if a['reason']).most_common()),
            'problem_counts': dict(Counter(p for a in attempts for p in a['problems']).most_common()),
            'tokens': tokens, 'max_tokens': settings.get('max_tokens'),
            # rows every other check keeps, by the token cap they would need (the reader window fills ~15.4k prompt tokens)
            'kept_at_cap': {str(cap): sum(1 for a in measured if a['tokens']['total'] <= cap
                                          and not [p for p in a['problems'] if p != 'too_long'])
                            for cap in sorted({settings.get('max_tokens') or MAX_TOKENS, 15500, 16000, 16384})},
            'split': {'valid_years': identity.get('valid_years'), 'train_rows': len(rows) - valid_rows, 'valid_rows': valid_rows,
                      'valid_share': round(valid_rows / len(rows), 3) if rows else None},
            'masked': {'rows_with_masked_candidates': sum(1 for r in rows if r['masked']['pool']),
                       'masked_candidates': sum(len(r['masked']['pool']) for r in rows),
                       'rows_with_masked_neighbours': sum(1 for r in rows if r['masked']['neighbours'])},
            'injected_rows': sum(1 for r in rows if r['injected']), 'premise_rows': dict(Counter(r['kind'] for r in rows if r['plan'].get('premise_note'))),
            'gate_would_refuse': sum(1 for r in rows if not (r['gate']['sufficient'] or r['gate']['soft'])),
            'refusal_retry_would_fire': sum(1 for r in rows if r.get('retry_would_fire')),
            'plan_scope': dict(Counter(f"{r['kind']}:{r['plan']['scope']}" for r in rows if r['kind'] == 'mixed_scope')),
            'plan_kinds': dict(Counter(r['plan']['kind'] for r in rows)),
            'sentences_per_answer_row': percentiles([len(r['sentences']) for r in rows if r['kind'] != 'refusal']),
            'letters_per_answer_row': percentiles([len({s['year'] for s in r['sentences']}) for r in rows if r['kind'] != 'refusal'])}


def settings_snapshot(max_tokens, prompt_version):
    import config
    from src.generation.token_count import count_tokens_and_source
    from src.services.schemas import AskRequest
    return {'finance': {k: config.finance_setting(k) for k in config.FINANCE_DEFAULTS},
            'retrieval': {k: config.retrieval_setting(k) for k in config.RETRIEVAL_DEFAULTS},
            'llm_n_ctx': config.llm_n_ctx(), 'llm_context_passages': config.llm_context_passages(),
            'max_new_tokens': AskRequest(query='x').max_new_tokens, 'prompt_version': prompt_version,
            'window_token_source': count_tokens_and_source('x')[1], 'window_tokenizer': os.environ.get('LLM_TOKENIZER_JSON'),
            'gemma_tokenizer': str(GEMMA_TOKENIZER), 'gemma_revision': GEMMA_REVISION, 'max_tokens': max_tokens,
            'question_overlap_max': QUESTION_OVERLAP_MAX, 'embedding_device': getattr(config, 'EMBEDDING_DEVICE', None)}


def write_outputs(out, rows, attempts, manifest):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r['split'], common.sha(f"{common.SEED}:fr4:{r['id']}".encode())))
    V6.write_jsonl_file(out / 'rows.jsonl', rows)
    for split in ('train', 'valid'):
        V6.write_jsonl_file(out / f'{split}.jsonl', [{'messages': r['messages']} for r in rows if r['split'] == split])
    V6.write_jsonl_file(out / 'attempts.jsonl', attempts)
    common.atomic_json(out / 'manifest.json', manifest)


def assemble(jobs, teacher, router, ctx, progress=None):
    rows, attempts = [], []
    for index, job in enumerate(jobs, 1):
        try:
            row, info = attempt(job, teacher.get(job['id']), router, ctx)
        except Exception as error:  # noqa: BLE001 - one bad job must not stop the run
            row, info = None, {'id': job['id'], 'kind': job['kind'], 'split': job['split'], 'kept': False,
                               'reason': f'error_{type(error).__name__}', 'problems': [f'error_{type(error).__name__}'],
                               'detail': str(error)[:300]}
        attempts.append(info)
        if row is not None:
            rows.append(row)
        if progress:
            progress(index, job, info)
    return rows, attempts


def assemble_command(args):
    out = Path(args.out).expanduser()
    if (out / 'rows.jsonl').exists() and not args.force:
        raise SystemExit(f'{out}/rows.jsonl exists; pick a new --out or pass --force')
    jobs = V6.read_jsonl_file(args.jobs)
    unknown = sorted({j['kind'] for j in jobs} - set(KINDS))
    if unknown:
        raise SystemExit(f'unknown job kind(s) {unknown}')
    teacher, files = read_teacher(args.teacher)
    full, store = load_corpus(), load_store()
    if {j['identity'].get('corpus_sha256') for j in jobs} - {file_sha(common.CORPUS)}:
        raise SystemExit('jobs were planned on another corpus; re-run plan')
    mask, exclude = read_ids(args.mask_file), read_ids(args.exclude_file)
    ctx = Context(full, store, exclude, frozenset(mask), set(V6.valid_years_of(full.values())), GemmaCounter(), args.max_tokens,
                  args.prompt_version)
    if args.limit:
        jobs = jobs[:args.limit]
    out.mkdir(parents=True, exist_ok=True)
    router = Router.load(full, store, mask, args.cache or out / CACHE_NAME)
    started = time.monotonic()

    def progress(index, job, info):
        print(f"[{index}/{len(jobs)}] {job['id']} kept={info['kept']} reason={info['reason']} s={info.get('seconds')}", flush=True)
        if index % 10 == 0:
            router.save_cache()

    rows, attempts = assemble(jobs, teacher, router, ctx, progress)
    router.save_cache()
    identity = {**(jobs[0]['identity'] if jobs else {}), 'jobs_sha256': file_sha(args.jobs),
                'teacher_files': {str(f): file_sha(f) for f in files}, 'mask_sha256': file_sha(args.mask_file),
                'exclude_sha256': file_sha(args.exclude_file), 'valid_years': sorted(ctx.valid_years)}
    manifest = manifest_for(jobs, attempts, rows, settings_snapshot(args.max_tokens, args.prompt_version), identity)
    manifest.update(teacher_records=len(teacher), jobs_without_teacher=sum(1 for j in jobs if j['id'] not in teacher),
                    wall_seconds=round(time.monotonic() - started, 1))
    write_outputs(out, rows, attempts, manifest)
    print('kept per kind: ' + ' '.join(f"{k}={v['kept']}/{v['attempted']}" for k, v in manifest['per_kind'].items())
          + f" total={len(rows)}/{len(attempts)}")
    print(f"drop reasons: {manifest['drop_reasons']}")
    print(f"tokens total (measured rows): {manifest['tokens']['total']['all_measured']} -> {out}/{{rows,train,valid,attempts}}.jsonl, manifest.json")


# ------------------------------------------------------------------ verify
def verify_rows(rows, ctx, router=None):
    """-> [(row id, problems)] of the bad rows. With a router every row is served again and its system and user turns
    must be the route's; without one (static) the hits are rebuilt from the row itself."""
    bad = []
    for row in rows:
        if row.get('kind') not in KINDS:
            bad.append((row.get('id'), ['unknown_kind']))
            continue
        if router is None:
            problems = check_row(row, static_hits(row), ctx)
        else:
            served = router.serve(row['question'], inject=[i['passage_id'] for i in row.get('injected') or ()])
            problems = [] if route_reason(served) is None else [route_reason(served)]
            if not problems:
                if [served.system, served.user] != [m['content'] for m in row['messages'][:2]]:
                    problems.append('user_turn_mismatch')
                if [{'id': h.id, 'merged_ids': list(h.metadata.get('merged_ids') or [h.id])} for h in served.hits] != \
                        [{'id': s['id'], 'merged_ids': s['merged_ids']} for s in row['served']]:
                    problems.append('served_list_mismatch')
                problems += check_row(row, served.hits, ctx, served.prompt, router)
        if problems:
            bad.append((row.get('id'), problems))
    return bad


def verify_command(args):
    rows = V6.read_jsonl_file(args.path)
    full, store = load_corpus(), load_store()
    mask = read_ids(args.mask_file)
    versions = {r.get('prompt_version') for r in rows} or {PROMPT_VERSION}
    if len(versions) > 1:
        raise SystemExit(f'rows carry several prompt versions {sorted(versions)}')
    apply_serving_env(versions.pop(), args.window_tokenizer, args.allow_env)
    ctx = Context(full, store, read_ids(args.exclude_file), frozenset(mask), set(V6.valid_years_of(full.values())), GemmaCounter(),
                  args.max_tokens, os.environ['PROMPT_VERSION'])
    router = None if args.static else Router.load(full, store, mask, args.cache or Path(args.path).with_name(CACHE_NAME))
    bad = verify_rows(rows, ctx, router)
    if router is not None:
        router.save_cache()
    for ident, problems in bad[:30]:
        print(f'BAD {ident}: {problems}')
    if bad:
        raise SystemExit(f'FAIL bad_rows={len(bad)} of {len(rows)}')
    print(f"OK rows={len(rows)} ({'static' if args.static else 're-served'})")


# ------------------------------------------------------------------ environment
def apply_serving_env(prompt_version=PROMPT_VERSION, window_tokenizer=GEMMA_TOKENIZER, allow_env=False):
    """The serving flags this builder runs with (module docstring); a retrieval / finance override in the environment
    stops it, because the rows would then not be the served chat (`allow_env` lets it pass, for experiments)."""
    import config
    from src.generation import token_count
    keys = set(config.FINANCE_DEFAULTS) | set(config.RETRIEVAL_DEFAULTS) | set(PINNED_ENV)
    overrides = sorted(k for k in keys - set(SERVING_FLAGS) if (os.environ.get(k) or '').strip())
    if overrides and not allow_env:
        raise SystemExit(f'environment overrides {overrides}: the rows must use the serving defaults (unset them or pass --allow-env)')
    os.environ.update(SERVING_FLAGS)
    os.environ['PROMPT_VERSION'] = prompt_version
    if window_tokenizer:
        os.environ['LLM_TOKENIZER_JSON'] = str(window_tokenizer)
    for key in ('REMOTE_LLM_URL', 'FINANCE_DEBUG'):
        os.environ.pop(key, None)
    token_count.reset()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    def common_args(p, ids=True):
        if ids:
            p.add_argument('--exclude-file', default=str(EXCLUDE_FILE), help='ids that are never a source of a target sentence')
            p.add_argument('--mask-file', default=str(MASK_FILE), help='ids that are never in a served training context')
        p.add_argument('--prompt-version', default=PROMPT_VERSION, help=f'system prompt of the reader rows (default {PROMPT_VERSION})')
        p.add_argument('--window-tokenizer', default=str(GEMMA_TOKENIZER),
                       help='tokenizer.json the reader window is counted with (LLM_TOKENIZER_JSON; default: the Gemma one)')
        p.add_argument('--allow-env', action='store_true', help='run although FINANCE_* / retrieval settings are overridden')

    p = sub.add_parser('plan', help='sample jobs + exact teacher messages -> jobs.jsonl (no model)')
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', required=True, help='jobs file; plan_manifest.json goes next to it')
    p.add_argument('--shares', help=f'kind shares of the jobs, e.g. refusal=.15 (default {SHARE})')
    common_args(p)

    e = sub.add_parser('export-batches', help='teacher batch files + TEACHER.md')
    e.add_argument('--jobs', required=True)
    e.add_argument('--size', type=int, default=50)
    e.add_argument('--out', required=True, help='batches directory; the teacher writes into its sibling out/')

    f = sub.add_parser('fake-teacher', help='deterministic teacher lines (template questions, first usable sentences)')
    f.add_argument('--jobs', required=True)
    f.add_argument('--out', required=True)

    a = sub.add_parser('assemble', help='teacher lines + jobs -> rows, checks, train/valid split, manifest (retrieval, no LLM)')
    a.add_argument('--jobs', required=True)
    a.add_argument('--teacher', required=True, help='a teacher jsonl file or a directory of teacher_*.jsonl')
    a.add_argument('--out', required=True)
    a.add_argument('--max-tokens', type=int, default=MAX_TOKENS, help=f'Gemma tokens of prompt + target at most (default {MAX_TOKENS})')
    a.add_argument('--cache', help=f'embedding / reranker cache (default <out>/{CACHE_NAME})')
    a.add_argument('--limit', type=int, help='first N jobs only')
    a.add_argument('--force', action='store_true', help='overwrite an existing --out')
    common_args(a)

    v = sub.add_parser('verify', help='re-serve every row and re-run the checks (--static: from the stored rows alone)')
    v.add_argument('path')
    v.add_argument('--static', action='store_true')
    v.add_argument('--max-tokens', type=int, default=MAX_TOKENS)
    v.add_argument('--cache')
    common_args(v)

    args = parser.parse_args(argv)
    if args.command == 'export-batches':
        index = export_batches(args.jobs, args.size, args.out)
        print(f'batches={len(index)} -> {args.out} (+ TEACHER.md, index.json)')
        return
    if args.command == 'fake-teacher':
        return fake_teacher_command(args)
    if args.command == 'verify':
        return verify_command(args)
    apply_serving_env(args.prompt_version, args.window_tokenizer, args.allow_env)
    if args.command == 'plan':
        return plan_command(args)
    return assemble_command(args)


if __name__ == '__main__':
    main()
