"""ft-r4-finance group 1: the FT-r4 finance training-data builder (scripts/ft/build_ft_finance.py, check_teacher.py).

A synthetic corpus of twelve passages and its knowledge layer; the finance route runs for real (`ask_flow.finance_route`)
over a fake retriever whose ranking is fixed and that has no reranker, so finance_search keeps that order.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'ft'))

import build_ft_finance as B  # noqa: E402
import check_teacher as CT  # noqa: E402
from src.generation import finance_answer  # noqa: E402
from src.generation.prompt import REFUSAL_LINE  # noqa: E402
from src.knowledge import KnowledgeStore  # noqa: E402
from src.storage import SearchHit, StoredDoc  # noqa: E402

TEXTS = {
    '1985_p0001': ('Our insurance float rose to $400 million during 1985 and cost us nothing to hold. National Indemnity wrote '
                   'premiums of $120 million while keeping its underwriting discipline intact. We believe that float of this '
                   'kind is worth far more than its balance sheet value suggests.'),
    '1990_p0001': ('The airline industry has been a graveyard for investors since the Wright brothers took flight. Our preferred '
                   'stock in USAir looked sound when we bought it, but the carrier lost money every quarter of 1990. Insurance '
                   'results at National Indemnity were much better and produced steady earnings for Berkshire.'),
    '1990_p0002': ('My mother celebrated her eightieth birthday with the whole family in Omaha that spring. My children flew in '
                   'from both coasts, and my sister baked the cake herself. We told old stories about my father until well '
                   'past midnight and nobody wanted to leave.'),
    '1995_p0001': ('GEICO joined Berkshire as a wholly owned subsidiary at the beginning of 1996 after years of partial '
                   'ownership. Its low cost structure gives it a durable advantage in auto insurance that rivals find hard to '
                   'copy. We expect the insurance operation to grow its policy count for many years.'),
    '2001_p0001': ('Insurance losses from the terrorist attacks reached $2.4 billion at General Re during 2001. We had priced '
                   'the risk of a mega-catastrophe far too cheaply for years and paid dearly for that mistake. Our managers '
                   'have since tightened underwriting terms across every insurance line we write.'),
    '2001_p0002': ('Insurance reserves at General Re needed further strengthening after the year closed. The reinsurance unit '
                   'had underestimated long-tail claims for several years running. Corrective pricing should restore '
                   'profitability over the next two or three years.'),
    '2008_p0001': ('The 2008 financial crisis froze credit markets and pushed many banks to the edge of failure. Housing prices '
                   'fell sharply across the country and dragged down the wider economy. Berkshire kept a large cushion of cash '
                   'so that it could act while others were forced to sell.'),
    '2008_p0002': ('During the crisis we invested $5 billion in Goldman Sachs preferred stock paying ten percent a year. We also '
                   'bought a large stake in General Electric when its commercial paper market seized up. Clayton Homes kept '
                   'lending to buyers through the panic because Berkshire stood behind it.'),
    '2009_p0001': ('The recession that followed the 2008 financial crisis cut demand at many of our manufacturing businesses. '
                   'Insurance float nonetheless kept growing and reached $62 billion by the end of 2009. We used the turmoil '
                   'to agree to the largest acquisition in our history.'),
    '2010_p0001': ('Two years after the 2008 financial crisis began, the economy was recovering at a slow pace. Our railroad and '
                   'utility businesses carried large capital programs through the downturn without strain. Insurance '
                   'underwriting remained profitable for the eighth consecutive year.'),
    '2010_p0002': ('Regulators reshaped the banking system after the 2008 financial crisis with new capital rules. Big banks '
                   'raised fresh equity and cut back on the leverage that had nearly sunk them. We continue to hold Wells '
                   'Fargo because its managers avoided the worst lending excesses.'),
    '2015_p0001': ('Insurance remains the engine that powers Berkshire, and our float now tops $87 billion. We have written an '
                   'underwriting profit in thirteen consecutive years, a record no large insurer can match. Underpricing to '
                   'win volume is a temptation that our managers have always refused.'),
}


def _ann(pid, sectors=(), themes=(), events=(), companies=(), stance=(), scope='financial', personal=False):
    return {'id': pid, 'scope': scope, 'has_personal': personal, 'is_table': False, 'sectors': list(sectors),
            'themes': list(themes), 'events': list(events), 'companies': list(companies), 'periods': [int(pid[:4])],
            'stance': [{'target': t, 'polarity': p, 'quote': 'x'} for t, p in stance], 'summary': TEXTS[pid][:80]}


ANNOTATIONS = [
    _ann('1985_p0001', sectors=['insurance'], themes=['float_underwriting'], companies=['National Indemnity'], stance=[('insurance', 'positive')]),
    _ann('1990_p0001', sectors=['airlines_aviation', 'insurance'], companies=['USAir'], stance=[('airlines_aviation', 'negative')]),
    _ann('1990_p0002', scope='personal_anecdote', personal=True),
    _ann('1995_p0001', sectors=['insurance'], companies=['GEICO'], stance=[('insurance', 'positive')]),
    _ann('2001_p0001', sectors=['insurance'], events=['sept11_2001'], themes=['mistakes_lessons'], stance=[('insurance', 'negative')]),
    _ann('2001_p0002', sectors=['insurance'], companies=['General Re']),
    _ann('2008_p0001', sectors=['banking', 'housing_construction'], events=['financial_crisis_2008'], stance=[('banking', 'negative')]),
    _ann('2008_p0002', events=['financial_crisis_2008'], companies=['Goldman Sachs', 'Clayton Homes']),
    _ann('2009_p0001', sectors=['insurance', 'manufacturing_industrial'], events=['financial_crisis_2008'], stance=[('insurance', 'positive')]),
    _ann('2010_p0001', sectors=['railroads', 'insurance'], events=['financial_crisis_2008']),
    _ann('2010_p0002', sectors=['banking'], events=['financial_crisis_2008'], companies=['Wells Fargo']),
    _ann('2015_p0001', sectors=['insurance'], themes=['float_underwriting'], stance=[('insurance', 'positive')]),
]
EVENTS = {'events': {
    'financial_crisis_2008': {'description': 'the 2007-09 financial crisis', 'event_years': [2007, 2008, 2009],
                              'letter_years': [2008, 2009, 2010], 'aliases': ['financial crisis', 'credit crisis']},
    'sept11_2001': {'description': 'the September 11 attacks', 'event_years': [2001], 'letter_years': [2001],
                    'aliases': ['september 11', 'terrorist attacks']}}}
NUMBERS = [{'metric': 'float', 'year': 1985, 'value': 400, 'unit': 'usd_millions', 'entity': 'Berkshire',
            'as_written': '$400 million', 'source_id': '1985_p0001'}]
DIGESTS = [{'topic': 'insurance', 'kind': 'sector', 'overview': 'Insurance is the core.',
            'timeline': [{'year': 1985, 'point': 'float', 'quote': 'x', 'source_id': '1985_p0001'},
                         {'year': 2001, 'point': 'losses', 'quote': 'x', 'source_id': '2001_p0001'},
                         {'year': 2001, 'point': 'reserves', 'quote': 'x', 'source_id': '2001_p0002'},   # excluded: never a seed
                         {'year': 2015, 'point': 'record', 'quote': 'x', 'source_id': '2015_p0001'}]}]
RANKING = ['2010_p0002', '2008_p0001', '2008_p0002', '2009_p0001', '2010_p0001', '1990_p0001', '1995_p0001', '2015_p0001',
           '2001_p0001', '2001_p0002', '1990_p0002']     # 1985_p0001 is never retrieved: a source of it must be injected
EXCLUDE = {'2001_p0002', '2010_p0002'}
MASK = frozenset({'2010_p0002'})
VALID_YEARS = {1995}


def _docs():
    docs, by_year = {}, {}
    for pid in sorted(TEXTS):
        by_year.setdefault(pid[:4], []).append(pid)
    for year, ids in by_year.items():
        for i, pid in enumerate(ids):
            docs[pid] = StoredDoc(id=pid, text=TEXTS[pid], metadata={
                'year': int(year), 'source_file': f'buffet_{year}.txt', 'chunk_index': i,
                'previous_chunk_id': ids[i - 1] if i else None, 'next_chunk_id': ids[i + 1] if i + 1 < len(ids) else None})
    return docs


class FakeRetriever:
    """Hybrid ranking = RANKING for every query and filter; no reranker, no embedder, no docs for the knowledge sources."""

    def __init__(self, full, ranking=RANKING):
        self.hits = [SearchHit(pid, full[pid].text, dict(full[pid].metadata), 1.0 - n / 100) for n, pid in enumerate(ranking)]
        self.reranker = None

    def _hybrid_for_filter(self, query, fetch_k, where):
        return list(self.hits)[:fetch_k]

    def _bm25_search(self, query, top_k, where=None):
        return []


@pytest.fixture(scope='module')
def counter():
    return B.GemmaCounter()


@pytest.fixture
def env(monkeypatch, counter, tmp_path):
    for key, value in {**B.SERVING_FLAGS, 'PROMPT_VERSION': B.PROMPT_VERSION, 'LLM_TOKENIZER_JSON': str(B.GEMMA_TOKENIZER)}.items():
        monkeypatch.setenv(key, value)
    for key in ('LLM_CONTEXT_PASSAGES', 'LLM_MODEL_PATH', 'LLM_N_CTX', 'LLM_PASSAGE_ORDER', 'REMOTE_LLM_URL', 'FINANCE_DEBUG'):
        monkeypatch.delenv(key, raising=False)
    from src.generation import token_count
    token_count.reset()
    full = _docs()
    store = KnowledgeStore(annotations=ANNOTATIONS, numbers=NUMBERS, events=EVENTS, digests=DIGESTS)
    ctx = B.Context(full, store, set(EXCLUDE), MASK, set(VALID_YEARS), counter)
    router = B.Router(FakeRetriever(full), full, store, MASK)
    return type('Env', (), {'full': full, 'store': store, 'ctx': ctx, 'router': router, 'tmp': tmp_path})


def _jobs(env, n=24, seed=5):
    return B.build_jobs(env.full, env.store, EXCLUDE, VALID_YEARS, n, seed, identity={'test': True})


def _job(kind='crisis_period', passages=None, **extra):
    passages = passages or [('2008_p0001', 2008), ('2009_p0001', 2009)]
    job = {'id': f'fr4:{kind}:train:0000', 'kind': kind, 'split': 'train', 'variant': extra.pop('variant', None),
           'topic': {'type': 'event', 'key': 'financial_crisis_2008', 'label': 'the 2008 financial crisis'},
           'source_ids': [p for p, _ in passages if p != '1990_p0002'], 'personal_ids': [], 'years': sorted({y for p, y in passages if p != '1990_p0002'}),
           'passages': [{'id': p, 'year': y, 'role': 'personal' if p == '1990_p0002' else 'financial',
                         'sentences': [[n, s] for n, s in enumerate(B.evidence_sentences(TEXTS[p]), 1)]} for p, y in passages]}
    job.update(extra)
    return job


# ------------------------------------------------------------------ plan
def test_plan_samples_every_kind_from_eligible_passages_and_is_deterministic(env):
    jobs, report = _jobs(env)
    again, _ = _jobs(env)
    assert [j['id'] for j in jobs] == [j['id'] for j in again] and jobs == again
    assert {j['kind'] for j in jobs} == set(B.KINDS)
    assert len({j['id'] for j in jobs}) == len(jobs)
    for job in jobs:
        assert not set(job['source_ids'] + job['personal_ids']) & EXCLUDE        # exclude_ids are never shown as sources
        assert all(env.store.is_financial(p) for p in job['source_ids'])
        assert (job['split'] == 'valid') == all(y in VALID_YEARS for y in job['years'])
        message = job['messages'][1]['content']
        assert f"{job['passages'][0]['id']}#1: " in message and 'Reply with JSON only' in message
        assert 'Did Buffett say anything about the oil crisis in the 80s?' in message and "the 2008 crisis" in message
    by_kind = {j['kind']: j for j in jobs}
    assert len(by_kind['crisis_period']['years']) >= 2
    view = by_kind['view_over_time']
    assert len(view['years']) >= 3 and len({y // 10 for y in view['years']}) >= 2
    mixed = by_kind['mixed_scope']
    assert mixed['personal_ids'] == ['1990_p0002'] and 'PERSONAL' in mixed['messages'][1]['content']
    premise = by_kind['premise_partial']
    assert premise['probe_note'] and premise['period_phrase'] in premise['messages'][1]['content']
    assert report['eligible_passages'] == 9                     # 12 passages - 2 excluded - the personal (non-financial) one


def test_seeds_prefer_the_topic_digest_timeline_passages(env):
    import random
    pools = B.Pools(env.full, env.store, EXCLUDE, VALID_YEARS)
    assert pools.digest[('sector', 'insurance')] == {'1985_p0001', '2001_p0001', '2015_p0001'}
    letters = pools.letters(pools.tags[('sector', 'insurance')], 'train')
    assert sorted(letters) == [1985, 1990, 2001, 2009, 2010, 2015]
    for seed in range(5):
        assert pools.pick_years(letters, 3, ('sector', 'insurance'), random.Random(seed)) == [1985, 2001, 2015]
    seed = pools.seed_view_over_time('train', random.Random(1), 0)     # insurance: the only topic with 3+ train letters
    assert seed['topic']['key'] == 'insurance' and set(seed['source_ids']) & pools.digest[('sector', 'insurance')]
    assert len({int(p[:4]) // 10 for p in seed['source_ids']}) >= 2


def test_unquotable_sentences_are_marked_for_the_teacher():
    texts = {'2002_p0001': ('Home lending standards collapsed across the industry in those years. Clayton Homes never joined the race '
                            'to the bottom on its loan terms. (The form we use is reproduced in its actual size on page 119.)'),
             '2002_p0002': ('Mortgage practices matter a great deal to every borrower and every lender in the country. My wife '
                            'reminded me that our own first mortgage carried a rate of seven percent. Clayton has written loans '
                            'that its borrowers could actually repay for decades.'),
             '2002_p0003': ('Our family gathered in Omaha for the holidays and my children came home from both coasts. We told '
                            'old stories about the early partnership days until well past midnight.')}
    ids = sorted(texts)
    full = {pid: StoredDoc(id=pid, text=texts[pid], metadata={
        'year': 2002, 'source_file': 'buffet_2002.txt', 'previous_chunk_id': ids[i - 1] if i else None,
        'next_chunk_id': ids[i + 1] if i + 1 < len(ids) else None}) for i, pid in enumerate(ids)}
    store = KnowledgeStore(annotations=[{'id': '2002_p0001', 'scope': 'financial'}, {'id': '2002_p0002', 'scope': 'financial'},
                                        {'id': '2002_p0003', 'scope': 'personal_anecdote', 'has_personal': True}])
    pools = B.Pools(full, store, set(), set())
    # #1 runs into the '.)' sentence before it in the merged text; #2 names a wife next to a personal neighbour
    assert pools.unquotable('2002_p0002') == [1, 2]
    assert B.Pools(full, store, set(), set(), mask={'2002_p0001', '2002_p0003'}).unquotable('2002_p0002') == []
    job = {**_job('single_year', [('2008_p0001', 2008)])}
    job['passages'][0]['unquotable'] = [1]
    message = B.passages_text(job)
    assert '2008_p0001#1 (not quotable): ' in message and '2008_p0001#2: ' in message
    q = 'In 2008, what steps did Buffett take as lenders panicked?'
    assert _ok(job, {'question': q, 'answer': [{'year': 2008, 'sentences': ['2008_p0001#1']}]})[0] == 'sentence_not_quotable'
    assert _ok(job, {'question': q, 'answer': [{'year': 2008, 'sentences': ['2008_p0001#3']}]})[0] is None


def test_allocate_and_shares():
    assert sum(B.allocate(40, B.SHARE).values()) == 40 and all(v > 0 for v in B.allocate(40, B.SHARE).values())
    shares = B.parse_shares('refusal=.2')
    assert shares['refusal'] == .2 and abs(sum(shares.values()) - 1) < 1e-9
    with pytest.raises(ValueError):
        B.parse_shares('nonsense=.2')


# ------------------------------------------------------------------ teacher check
def _ok(job, obj):
    return CT.validate(job, obj, route=False)


def test_check_teacher_accepts_a_valid_answer_and_orders_it_by_letter():
    job = _job()
    reason, plan = _ok(job, {'question': 'What lessons did Buffett draw from the great credit meltdown?',
                             'answer': [{'year': 2009, 'sentences': ['2009_p0001#2']}, {'year': 2008, 'sentences': ['2008_p0001#1']}]})
    assert reason is None
    assert [c['sid'] for c in plan['chosen']] == ['2008_p0001#1', '2009_p0001#2']


@pytest.mark.parametrize('answer, reason', [
    ([{'year': 2008, 'sentences': ['2008_p0001#9']}, {'year': 2009, 'sentences': ['2009_p0001#1']}], 'unknown_sentence_id'),
    ([{'year': 2009, 'sentences': ['2008_p0001#1']}], 'sentence_wrong_letter'),
    ([{'year': 2008, 'sentences': ['2008_p0001#1']}], 'too_few_letters'),
    ([{'year': 2008, 'sentences': ['2008_p0001#1', '2008_p0001#1']}, {'year': 2009, 'sentences': ['2009_p0001#1']}], 'duplicate_sentence'),
    ([], 'no_sentences'),
    ('x', 'bad_answer'),
])
def test_check_teacher_rejects_bad_answers(answer, reason):
    assert _ok(_job(), {'question': 'What lessons did Buffett draw from the great credit meltdown?', 'answer': answer})[0] == reason


def test_check_teacher_dependent_sentence_needs_the_one_before():
    job = _job(passages=[('2008_p0002', 2008), ('2009_p0001', 2009)])
    text = dict(job['passages'][0]['sentences'])[1]
    assert text.startswith('During')          # not dependent; build one that is
    job['passages'][0]['sentences'].append([4, 'It was a hard year for every bank in the country.'])
    q = 'What lessons did Buffett draw from the great credit meltdown?'
    alone = [{'year': 2008, 'sentences': ['2008_p0002#4']}, {'year': 2009, 'sentences': ['2009_p0001#1']}]
    assert _ok(job, {'question': q, 'answer': alone})[0] == 'dependent_start'
    alone[0]['sentences'] = ['2008_p0002#3', '2008_p0002#4']
    assert _ok(job, {'question': q, 'answer': alone})[0] is None


def test_check_teacher_question_rules():
    job = _job()
    answer = [{'year': 2008, 'sentences': ['2008_p0001#1']}, {'year': 2009, 'sentences': ['2009_p0001#1']}]
    assert _ok(job, {'question': 'What does the passage say?', 'answer': answer})[0] == 'question_refers_to_passage'
    assert _ok(job, {'question': 'no question mark here at all', 'answer': answer})[0] == 'bad_question'
    copied = 'Why did the 2008 financial crisis froze credit markets and pushed many banks?'
    assert _ok(job, {'question': copied, 'answer': answer})[0] == 'question_copies_passage'
    assert _ok(job, {'question': 'Did the financial crisis hurt housing prices and banks?', 'answer': answer})[0] == 'question_overlaps_passage'


def test_check_teacher_kind_rules():
    one = [('1985_p0001', 1985)]
    single = _job('single_year', one)
    answer = [{'year': 1985, 'sentences': ['1985_p0001#1']}]
    assert _ok(single, {'question': 'What was special about the insurance operation that year?', 'answer': answer})[0] == 'year_missing'
    assert _ok(single, {'question': 'In 1985, what was special about the insurance operation?', 'answer': answer})[0] is None
    figure = _job('figure', one)
    assert _ok(figure, {'question': 'In 1985, how did Buffett judge the worth of the money he held?',
                        'answer': [{'year': 1985, 'sentences': ['1985_p0001#3']}]})[0] == 'no_figure'
    one_decade = (('2001_p0001', 2001), ('2008_p0001', 2008), ('2009_p0001', 2009))
    view = _job('view_over_time', list(one_decade))
    three = [{'year': y, 'sentences': [f'{p}#1']} for p, y in one_decade]
    question = "How did Buffett's thinking on this evolve over time?"
    assert _ok(view, {'question': question, 'answer': three})[0] == 'too_few_decades'
    assert _ok(view, {'question': question, 'answer': three[:2]})[0] == 'too_few_letters'
    two_decades = (('1985_p0001', 1985), ('1990_p0001', 1990), ('2001_p0001', 2001))
    view = _job('view_over_time', list(two_decades))
    assert _ok(view, {'question': question, 'answer': [{'year': y, 'sentences': [f'{p}#1']} for p, y in two_decades]})[0] is None
    premise = _job('premise_partial', [('1990_p0001', 1990)], period_variants=['the 1970s', 'the 70s'])
    sentence = [{'year': 1990, 'sentences': ['1990_p0001#1']}]
    assert _ok(premise, {'question': 'Did Buffett say anything about airlines in the 2000s?', 'answer': sentence})[0] == 'period_missing'
    assert _ok(premise, {'question': 'Did Buffett say anything about airlines in the 70s?', 'answer': sentence})[0] is None


def test_check_teacher_partial_refusal_and_mixed_rules():
    one = [('1985_p0001', 1985)]
    partial = _job('partial', one)
    answer = [{'year': 1985, 'sentences': ['1985_p0001#1']}]
    q = 'In 1985, how did the float develop, and what did Zephyrine Holdings pay for its stake?'
    good_gap = {'ask': 'what Zephyrine Holdings paid for its stake', 'missing': ['Zephyrine Holdings']}
    assert _ok(partial, {'question': q, 'answer': answer, 'gap': good_gap})[0] is None
    assert _ok(partial, {'question': q, 'answer': answer, 'gap': {'ask': 'the float', 'missing': ['National Indemnity']}})[0] == 'bad_gap'
    near = _job('refusal', one, variant='near')
    rq = 'What did National Indemnity pay when it bought Quorvant Industries?'
    assert _ok(near, {'question': rq, 'answer': [], 'present': ['National Indemnity'], 'missing': ['Quorvant Industries']})[0] is None
    assert _ok(near, {'question': rq, 'answer': answer, 'present': ['National Indemnity'], 'missing': ['Quorvant Industries']})[0] == 'refusal_has_answer'
    assert _ok(near, {'question': rq, 'answer': [], 'present': ['Quorvant Industries'], 'missing': ['National Indemnity']})[0] == 'bad_missing'
    assert _ok(near, {'question': 'What will National Indemnity earn in 2031?', 'answer': [], 'present': ['National Indemnity'],
                      'missing': ['2031']})[0] == 'refusal_cue'
    mixed = _job('mixed_scope', [('1990_p0001', 1990), ('1990_p0002', 1990)])
    mq = 'What did Buffett say about USAir in 1990, and how old was his mother at the time?'
    base = {'question': mq, 'financial_part': 'USAir', 'personal_part': 'his mother', 'answer': [{'year': 1990, 'sentences': ['1990_p0001#2']}]}
    assert _ok(mixed, {**base, 'personal_sentences': ['1990_p0002#1']})[0] is None
    assert _ok(mixed, {**base, 'personal_sentences': ['1990_p0001#1']})[0] == 'bad_personal_sentences'
    leak = {**base, 'answer': [{'year': 1990, 'sentences': ['1990_p0002#1']}], 'personal_sentences': ['1990_p0002#2']}
    assert _ok(mixed, leak)[0] == 'personal_sentence_in_answer'


def test_check_teacher_rejects_questions_the_planner_routes_away(env):
    job = _job()
    answer = [{'year': 2008, 'sentences': ['2008_p0001#1']}, {'year': 2009, 'sentences': ['2009_p0001#1']}]
    q = 'How did banks fare in 2008 compared with 2009?'
    assert CT.validate(job, {'question': q, 'answer': answer}, store=env.store)[0] == 'comparison_question'
    premise = _job('premise_partial', [('1990_p0001', 1990)], period_variants=['the 1990s'])
    sentence = [{'year': 1990, 'sentences': ['1990_p0001#1']}]
    q = 'Did Buffett say anything about airlines in the 1990s?'            # the right period: nothing to correct
    assert CT.validate(premise, {'question': q, 'answer': sentence}, store=env.store)[0] == 'premise_not_fired'


def test_check_teacher_cli(env, capsys):
    jobs, _ = _jobs(env, n=8)
    batch, teacher = env.tmp / 'batch.jsonl', env.tmp / 'teacher.jsonl'
    B.V6.write_jsonl_file(batch, jobs)
    lines = [B.fake_answer(j, env.store) for j in jobs]
    ok = [line for line, j in zip(lines, jobs) if CT.validate(j, line, store=env.store)[0] is None]
    assert ok, 'the fake teacher must pass on some jobs'
    teacher.write_text(''.join(json.dumps(line) + '\n' for line in lines))
    CT.planner_store.cache_clear()
    rc_all = CT.check_file(jobs, CT.read_lines(teacher), store=env.store)[0]
    assert all(status in ('OK', 'REJECT') for status, _ in rc_all.values())
    teacher.write_text(json.dumps({'id': jobs[0]['id'], 'skip': 'nothing answers it'}) + '\nnot json\n')
    result, stray = CT.check_file(jobs, CT.read_lines(teacher), store=env.store)
    assert result[jobs[0]['id']] == ('SKIP', 'nothing answers it') and stray == [(2, 'json_invalid')]
    assert result[jobs[1]['id']][0] == 'MISSING'
    assert CT.main(['--batch', str(batch), str(teacher)]) == 1
    assert 'MISSING' in capsys.readouterr().out


# ------------------------------------------------------------------ serving replay
CRISIS_Q = 'Did Buffett say anything about the 2008 financial crisis?'


def test_router_masks_sealed_passages_and_backfills_from_the_next_candidates(env, monkeypatch):
    monkeypatch.setenv('FINANCE_K_PERIOD', '3')
    open_router = B.Router(FakeRetriever(env.full), env.full, env.store, frozenset())
    unmasked = open_router.serve(CRISIS_Q)
    masked = env.router.serve(CRISIS_Q)
    assert unmasked.route == masked.route == 'reader'
    assert '2010_p0002' in B.served_ids(unmasked.hits)
    ids = B.served_ids(masked.hits)
    assert '2010_p0002' not in ids and len(masked.hits) == len(unmasked.hits) == 3
    assert {h.id for h in masked.hits} == {'2008_p0001', '2008_p0002', '2009_p0001'}   # the 4th candidate took the slot
    assert masked.masked_pool == ['2010_p0002']
    assert '2010_p0002' not in masked.user and '2010_p0002' not in str([h.metadata.get('merged_ids') for h in masked.hits])
    assert finance_answer.is_v3_prompt(masked.prompt) and masked.system == B.system_prompt('v2')
    assert [h.metadata['year'] for h in masked.hits] == sorted(h.metadata['year'] for h in masked.hits)


def test_masked_store_hides_only_the_financial_flag(env):
    store = B.MaskedStore(env.store, {'2008_p0001'})
    assert not store.is_financial('2008_p0001') and store.is_financial('2009_p0001')
    assert store.annotation('2008_p0001') == env.store.annotation('2008_p0001') and len(store) == len(env.store)


def test_injection_serves_a_missing_source_in_its_letter_year_slot(env):
    served = env.router.serve(CRISIS_Q, inject=['1985_p0001'])
    assert served.hits[0].id == '1985_p0001'                     # the oldest letter: first in the year-ordered prompt
    assert '[1] (year=1985' in served.user and 'LETTER OF 1985' in served.user


def test_parse_user_passages_round_trips_the_v3_prompt(env):
    served = env.router.serve(CRISIS_Q)
    parsed = B.parse_user_passages(served.user)
    assert [text for _, _, text in parsed] == [h.text.strip() for h in served.hits]
    assert [int(year) for year, _, _ in parsed] == [h.metadata['year'] for h in served.hits]


# ------------------------------------------------------------------ targets and rows
def test_build_target_layout():
    sentences = [{'year': 2009, 'passage_id': '2009_p0001', 'n': 2, 'text': 'B two.', 'slot': 3},
                 {'year': 2008, 'passage_id': '2008_p0001', 'n': 3, 'text': 'A three.', 'slot': 2},
                 {'year': 2008, 'passage_id': '2008_p0001', 'n': 1, 'text': 'A one.', 'slot': 1}]
    assert B.build_target(sentences) == 'A one. [1] A three. [2]\nB two. [3]'
    assert B.build_target(sentences, 'Note.', {'ask': 'the price'}) == \
        'Note. A one. [1] A three. [2]\nB two. [3]\nThe retrieved passages do not cover the price.'


def _assemble(env, n=24, seed=5):
    jobs, _ = _jobs(env, n, seed)
    teacher = {j['id']: B.fake_answer(j, env.store) for j in jobs}
    rows, attempts = B.assemble(jobs, teacher, env.router, env.ctx)
    return jobs, rows, attempts


def test_assemble_end_to_end_rows_match_the_serving_rules(env):
    jobs, rows, attempts = _assemble(env)
    assert len(attempts) == len(jobs) and rows
    assert not [a for a in attempts if str(a['reason']).startswith('error_')], [a for a in attempts if a['reason']]
    kinds = {r['kind'] for r in rows}
    assert {'crisis_period', 'refusal', 'premise_partial', 'mixed_scope'} <= kinds
    for row in rows:
        system, user, target = (m['content'] for m in row['messages'])
        assert [m['role'] for m in row['messages']] == ['system', 'user', 'assistant']
        assert system == B.system_prompt('v2') and user.endswith('Answer:') and finance_answer.is_v3_prompt(f'{system}\n\n{user}')
        assert not set(row['served_ids']) & MASK
        assert row['tokens']['total'] <= B.MAX_TOKENS
        assert B.check_row(row, B.static_hits(row), env.ctx) == []          # static re-check agrees with assemble
        if row['kind'] == 'refusal':
            assert target == REFUSAL_LINE
            continue
        texts = [h.text for h in B.static_hits(row)]
        for s in row['sentences']:
            assert f"{s['text']} [{s['slot']}]" in target and B.C.lowest_holder(s['text'], texts) == s['slot']
            assert s['passage_id'] not in EXCLUDE
        if row['plan']['premise_note']:
            assert target.startswith(row['plan']['premise_note'])
    premise = next(r for r in rows if r['kind'] == 'premise_partial')
    assert 'do not discuss' in premise['plan']['premise_note']
    mixed = next(r for r in rows if r['kind'] == 'mixed_scope')
    assert mixed['plan']['scope'] == 'mixed' and 'mother' not in mixed['messages'][1]['content'].split('BEGIN USER QUESTION')[1]
    assert not any(f in mixed['messages'][2]['content'] for f in mixed['forbidden'])
    injected = [r for r in rows if r['injected']]
    assert injected and all('1985_p0001' in [i['passage_id'] for i in r['injected']] for r in injected)
    assert B.verify_rows(rows, env.ctx) == []
    assert B.verify_rows(rows, env.ctx, env.router) == []               # re-served: the same user turns and served lists


def test_check_row_flags_broken_rows(env):
    _, rows, _ = _assemble(env)
    row = next(r for r in rows if r['kind'] == 'crisis_period')
    hits = B.static_hits(row)
    import copy
    tight = B.Context(**{**env.ctx.__dict__, 'max_tokens': 100})
    assert 'too_long' in B.check_row(row, hits, tight)
    moved = copy.deepcopy(row)
    moved['sentences'][0]['slot'] = len(hits) + 1 if moved['sentences'][0]['slot'] != len(hits) + 1 else 1
    assert 'citation_not_lowest_holder' in B.check_row(moved, hits, env.ctx)
    edited = copy.deepcopy(row)
    edited['messages'][2]['content'] = edited['messages'][2]['content'].replace(' [', ' extra words [', 1)
    problems = B.check_row(edited, hits, env.ctx)
    assert 'target_mismatch' in problems and 'token_count_mismatch' in problems
    copied = copy.deepcopy(row)
    copied['question'] = 'Did the 2008 financial crisis freeze credit markets and push banks?'
    assert 'question_overlap' in B.check_row(copied, hits, env.ctx)
    masked = copy.deepcopy(row)
    masked_hits = hits[:-1] + [SearchHit('2010_p0002', hits[-1].text, {**hits[-1].metadata, 'merged_ids': ['2010_p0002']}, 0.)]
    assert 'masked_in_context' in B.check_row(masked, masked_hits, env.ctx)
    mixed = copy.deepcopy(next(r for r in rows if r['kind'] == 'mixed_scope'))
    mixed['messages'][2]['content'] += ' ' + mixed['forbidden'][0]
    assert 'personal_sentence' in B.check_row(mixed, B.static_hits(mixed), env.ctx)
    excluded = B.Context(**{**env.ctx.__dict__, 'exclude': env.ctx.exclude | {row['sentences'][0]['passage_id']}})
    assert 'sentence_from_excluded' in B.check_row(row, hits, excluded)


def test_postprocess_replay_flags_a_sentence_the_validator_drops(env):
    _, rows, _ = _assemble(env)
    row = next(r for r in rows if r['kind'] == 'crisis_period')
    hits = B.static_hits(row)
    fake = {'text': 'Berkshire tripled its railroad earnings in a single quarter.', 'slot': 1, 'year': 2099,
            'passage_id': 'x', 'n': 1}
    broken = {**row, 'sentences': row['sentences'] + [fake],
              'messages': row['messages'][:2] + [{'role': 'assistant', 'content': row['messages'][2]['content'] + '\n' + f"{fake['text']} [1]"}]}
    problems = B.postprocess_problems(broken, None or f"{row['messages'][0]['content']}\n\n{row['messages'][1]['content']}", hits, env.store)
    assert 'postprocess_drops_sentence' in problems


def test_write_outputs_and_manifest(env):
    jobs, rows, attempts = _assemble(env)
    manifest = B.manifest_for(jobs, attempts, rows, {'max_tokens': B.MAX_TOKENS}, {'valid_years': sorted(VALID_YEARS)})
    B.write_outputs(env.tmp / 'out', rows, attempts, manifest)
    train = [json.loads(line) for line in (env.tmp / 'out/train.jsonl').read_text().splitlines()]
    assert train and all(list(r) == ['messages'] for r in train)
    saved = json.loads((env.tmp / 'out/manifest.json').read_text())
    assert set(saved['per_kind']) == set(B.KINDS) and saved['kept'] == len(rows)
    assert sum(v['kept'] for v in saved['per_kind'].values()) == len(rows)
    assert sum(saved['tokens']['total']['histogram'].values()) == sum(1 for a in attempts if a.get('tokens'))
    assert saved['kept_at_cap'][str(B.MAX_TOKENS)] == len(rows)
    assert saved['split']['train_rows'] + saved['split']['valid_rows'] == len(rows)


def test_export_batches_writes_batches_and_teacher_instructions(env):
    jobs, _ = _jobs(env, n=8)
    path = env.tmp / 'jobs.jsonl'
    B.V6.write_jsonl_file(path, jobs)
    index = B.export_batches(path, 3, env.tmp / 'batches')
    assert list(index) == ['batch_000', 'batch_001', 'batch_002'] and sum(len(v['jobs']) for v in index.values()) == len(jobs)
    md = (env.tmp / 'batches/TEACHER.md').read_text()
    assert 'check_teacher.py' in md and str((env.tmp / 'out').resolve()) in md and 'teacher_batch_NNN.jsonl' in md
    assert (env.tmp / 'out').is_dir()
    assert B.V6.read_jsonl_file(env.tmp / 'batches/batch_001.jsonl')[0]['id'] == jobs[3]['id']
