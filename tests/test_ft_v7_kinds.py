"""ft_v7 data builder, items 3, 4, 5: refusal checks, refusal_near, partial, comparison (served per-period shape), exact-N multipart,
cross-letter planner and split, unknown kinds."""
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_env = dict(os.environ)
sys.path.insert(0, str(ROOT / 'scripts' / 'ft'))
import build_ft_v6 as B  # noqa: E402
import ft_v7_checks as C  # noqa: E402
os.environ.clear()
os.environ.update(_env)

from src.generation.compare import comparison_periods, generate_comparison_answer, period_label, period_question  # noqa: E402
from src.generation.prompt import REFUSAL_LINE  # noqa: E402
from src.retrieval.bm25 import _meta_matches  # noqa: E402
from src.storage import SearchHit  # noqa: E402


@pytest.fixture(autouse=True)
def training_prompt(monkeypatch):
    monkeypatch.setenv('PROMPT_VERSION', B.PROMPT_VERSION)
    monkeypatch.delenv('LLM_CONTEXT_PASSAGES', raising=False)
    monkeypatch.setattr(B, 'QUESTION_OVERLAP_MAX', 1.0)       # the tiny test passages share most words with any question; the overlap rule has its own test


S = [
    'Berkshire repurchased 120,000 shares of its common stock during 1995.',
    'The buyback was funded from retained earnings of the insurance operations.',
    'This explains why the float grew faster than book value in that year.',
    'Insurance float reached 3.4 billion dollars at the end of 1995.',
    'GEICO added 400,000 policyholders and its combined ratio improved to 96.',
    'Charlie and I paid 50 million dollars for Dexter Shoe during 1993.',
]
T = [
    'Berkshire spent 2 billion dollars on share repurchases during 2010.',
    'Insurance float reached 65 billion dollars at the end of 2010.',
    'GEICO added 900,000 policyholders and its combined ratio improved to 91.',
    'The railroad earned 1.8 billion dollars before taxes in 2010.',
]


def doc(ident, sentences, year):
    return SimpleNamespace(id=ident, text=' '.join(sentences), metadata={'year': year, 'source_file': f'buffet_{year}.txt'})


def hit(ident, sentences, year):
    return SearchHit(id=ident, text=' '.join(sentences), metadata={'year': year, 'source_file': f'buffet_{year}.txt'}, score=1.)


def filler(n, year=1990):
    return [hit(f'{year}_p9{i:03d}', [f'Filler passage number {i} talks about farm equipment weather and nothing else of note here.'], year) for i in range(n)]


class FakeServing:
    def __init__(self, extra=()):
        self.seen, self.extra = [], list(extra)

    def prompts(self, question, history=None):
        k = int(os.environ.get('LLM_CONTEXT_PASSAGES', 8))
        hits = (self.extra + filler(k))[:k]
        self.seen.append(question)
        return [{'period': None, 'hits': hits, 'gate': True, 'retrieved_ids': [h.id for h in hits], 'reranked': True}]


def generator(full, tmp_path, serving=None, recorded=None, **kw):
    return B.RecordedGenerator(recorded or {}, serving or FakeServing(), full, set(), tmp_path, None, B.MAX_SEQ_MIX, kw.pop('strict', False), [8])


# ------------------------------------------------------------------ tables, unknown kinds
def test_every_kind_has_its_share_expectation_and_parts():
    assert set(B.SHARE) == set(B.KINDS) == set(B.EXPECT)
    assert abs(sum(B.SHARE.values()) - 1) < 1e-9
    assert B.SHARE['refusal'] + B.SHARE['refusal_near'] >= .15 and abs(B.SHARE['partial'] - .05) < 1e-9
    assert {'refusal_near', 'partial', 'comparison', 'cross_letter'} <= set(B.KINDS)
    assert B.PARTS['multipart'] == (4, 5) and B.PARTS['cross_letter'] == (4, 5)


def test_unknown_kinds_fail_loudly_in_plan_and_assemble():
    d = doc('1995_p0001', S, 1995)
    with pytest.raises(ValueError, match='unknown job kind'):
        B.job_record({'id': 'x', 'kind': 'bogus', 'split': 'train', 'source_ids': ['1995_p0001']}, {'1995_p0001': d}, {})
    with pytest.raises(SystemExit, match='unknown job kind'):
        B.require_known_kinds([{'id': 'a', 'kind': 'multipart'}, {'id': 'b', 'kind': 'bogus'}])
    assert B.require_known_kinds([{'id': 'a', 'kind': 'refusal_near'}]) is None
    gen = generator({}, Path('.'))
    with pytest.raises(ValueError, match='unknown job kind'):
        gen._attempt({'id': 'x', 'kind': 'bogus', 'split': 'train', 'source_ids': []}, {'k': 8})


def test_quotas_count_comparison_in_jobs_of_two_rows():
    quotas = B.quotas_for(1000)
    rows = {k: quotas[(k, 'train')] + quotas[(k, 'valid')] for k in B.KINDS}
    assert rows['comparison'] == round(1000 * B.SHARE['comparison'] / 2)
    assert rows['refusal_near'] == round(1000 * B.SHARE['refusal_near'])


# ------------------------------------------------------------------ item 5: exact-N multipart (ASM-02, MPR-02)
def numbered(n):
    sentences = [f'Berkshire reported distinct fact number {i} about its insurance business in 1995 clearly.' for i in range(1, 8)]
    parts = [{'ask': f'fact {i}', 'passage': 1, 'sentences': [i]} for i in range(1, n + 1)]
    return doc('1995_p0001', sentences, 1995), {'question': 'In 1995, what were the first facts that Berkshire reported about its insurance business and its results?', 'parts': parts}


def test_multipart_job_asks_for_exactly_n_parts_and_shows_an_example_of_that_size():
    d, _ = numbered(5)
    job5 = {'id': 'v7:multipart:train:0000', 'kind': 'multipart', 'split': 'train', 'source_ids': ['1995_p0001'], 'parts_range': [5, 5]}
    prompt5 = B.teacher_prompt(job5, [d])
    assert 'EXACTLY 5 parts' in prompt5 and prompt5.count('"ask"') >= 7 and '5-part' in prompt5
    prompt4 = B.teacher_prompt(dict(job5, parts_range=[4, 4]), [d])
    assert 'EXACTLY 4 parts' in prompt4 and '4-part' in prompt4 and '5-part' not in prompt4


def test_exact_n_is_enforced_and_legacy_ranges_still_validate():
    d, obj = numbered(4)
    assert B.validate_plan(obj, 'multipart', [d], [5, 5])[0] == 'parts_count'
    assert B.validate_plan(obj, 'multipart', [d], [4, 4])[0] is None
    assert B.validate_plan(obj, 'multipart', [d])[0] is None            # default range is PARTS['multipart'] = (4, 5)
    d3, obj3 = numbered(3)
    assert B.validate_plan(obj3, 'multipart', [d3])[0] == 'parts_count'
    assert B.validate_plan(obj3, 'multipart', [d3], [3, 5])[0] is None   # an ft_v6 job carries its own [3, 5]


# ------------------------------------------------------------------ small corpus for the planners
def corpus(years, chunks=8):
    full = {}
    for year in years:
        for i in range(chunks):
            ident = f'{year}_p{i:04d}'
            sentences = [f'In {year} the company reported fact {j} of chunk {i} about its operations, insurance float and subsidiaries clearly.' for j in range(1, 6)]
            full[ident] = SimpleNamespace(id=ident, text=' '.join(sentences), metadata={
                'year': year, 'source_file': f'buffet_{year}.txt', 'previous_chunk_id': f'{year}_p{i - 1:04d}' if i else None,
                'next_chunk_id': f'{year}_p{i + 1:04d}' if i + 1 < chunks else None})
    return full


def test_cross_letter_runs_take_two_different_letters_from_one_split():
    full = corpus(range(1990, 2002))
    valid_years = {1991, 1995, 1999}
    rng = random.Random(3)
    for split in ('train', 'valid'):
        runs = B.build_cross_runs(full, set(), valid_years, 'cross_letter', split, 6, set(), rng)
        assert runs
        for run in runs:
            years = {int(full[i].metadata['year']) for i in run}
            assert len(years) == 2 and all((y in valid_years) == (split == 'valid') for y in years)
            assert [int(full[i].metadata['year']) for i in run] == sorted(int(full[i].metadata['year']) for i in run)


def test_plan_makes_every_kind_with_exact_parts_and_two_letter_jobs():
    full = corpus(range(1990, 2002))
    quotas = B.quotas_for(100)
    plan = B.build_plan(full, set(), {1991, 1995, 1999}, quotas, {k: 1. for k in B.KINDS}, {'x': 1}, 7)
    kinds = {j['kind'] for j in plan['jobs']}
    assert kinds == set(B.KINDS)
    for job in plan['jobs']:
        assert job['id'].startswith('v7:')
        if job['kind'] in ('multipart', 'cross_letter'):
            assert job['parts_range'] in ([4, 4], [5, 5])
        if job['kind'] in ('cross_letter', 'comparison'):
            assert len({int(full[i].metadata['year']) for i in job['source_ids']}) == 2
    assert {j['parts_range'][0] for j in plan['jobs'] if j['kind'] == 'multipart'} == {4, 5}
    records = [B.job_record(j, full, {'corpus_sha256': 'x'}) for j in plan['jobs']]
    assert all(r['needs_teacher'] == (r['messages'] is not None) for r in records)
    assert {r['kind'] for r in records if r['messages'] is None} <= {'refusal'}


def test_cross_letter_validate_plan_needs_facts_from_both_letters():
    a = doc('1990_p0001', [f'Berkshire reported distinct fact number {i} about its insurance business in 1990 clearly.' for i in range(1, 5)], 1990)
    b = doc('1995_p0001', [f'Berkshire reported distinct fact number {i} about its shoe business in 1995 clearly.' for i in range(1, 5)], 1995)
    question = 'What did Berkshire report about insurance business facts, about shoe business facts, and about the other distinct results it kept?'
    both = {'question': question, 'parts': [{'ask': 'a', 'passage': 1, 'sentences': [1]}, {'ask': 'b', 'passage': 1, 'sentences': [3]},
                                            {'ask': 'c', 'passage': 2, 'sentences': [1]}, {'ask': 'd', 'passage': 2, 'sentences': [3]}]}
    assert B.validate_plan(both, 'cross_letter', [a, b], [4, 4])[0] is None
    one = dict(both, parts=[{'ask': 'a', 'passage': 1, 'sentences': [i]} for i in (1, 2, 3, 4)])
    assert B.validate_plan(one, 'cross_letter', [a, b], [4, 4])[0] == 'single_letter_parts'
    head = B.teacher_prompt({'id': 'x', 'kind': 'cross_letter', 'split': 'train', 'source_ids': ['1990_p0001', '1995_p0001'], 'parts_range': [4, 4]}, [a, b])
    assert 'from the 1990 Berkshire' not in head and 'letter year 1990' in head and 'letter year 1995' in head


def test_a_row_must_keep_all_its_letters_on_one_side_of_the_split():
    assert B.split_problem({'split': 'valid', 'parts': [{'year': 1991}]}, {1991}) is None
    assert B.split_problem({'split': 'train', 'parts': [{'year': 1991}]}, {1991}) == 'split_straddle'
    assert B.split_problem({'split': 'valid', 'parts': [{'year': 1991}, {'year': 1992}]}, {1991}) == 'split_straddle'
    assert B.split_problem({'split': 'train', 'parts': [{'year': 1992}, {'year': 1993}]}, {1991}) is None
    assert B.split_problem({'split': 'valid', 'parts': [], 'source_year': 1991}, {1991}) is None


# ------------------------------------------------------------------ item 3a: adjacent refusals need distinctive missing terms
def refusal_job():
    return {'id': 'v7:refusal:train:0001', 'kind': 'refusal', 'split': 'train', 'source_ids': ['1995_p0001'], 'variant': 'adjacent'}


def adjacent_raw(missing, question='Which company did Berkshire buy in 1987 when Fechheimer joined the group of subsidiaries?'):
    return {'v7:refusal:train:0001': {'raw': json.dumps({'question': question, 'missing': missing}), 'calls': 1}}


def test_adjacent_refusal_rejects_generic_missing_terms(tmp_path):
    full = {'1995_p0001': doc('1995_p0001', S, 1995)}
    for missing in (['specific companies'], ['exact amount'], ['previous year'], ['the CEO']):
        question = f'What were the {missing[0]} that Berkshire named in 1987 for its subsidiaries and Fechheimer?'
        gen = generator(full, tmp_path, recorded=adjacent_raw(missing, question))
        row, reason = gen._attempt(refusal_job(), {'k': 8, 'teacher_calls': 0})
        assert row is None and reason == 'bad_part', missing
    gen = generator(full, tmp_path, recorded=adjacent_raw(['Fechheimer']))
    row, reason = gen._attempt(refusal_job(), {'k': 8, 'teacher_calls': 0})
    assert reason is None and row['messages'][2]['content'] == REFUSAL_LINE and row['absent_terms'] == ['Fechheimer']
    assert B.check_row(row) == []


def test_distinctive_missing_needs_a_name_or_a_number():
    for ok in ('Fechheimer', '1987', '$40 billion', 'Ajit Jain'):
        assert C.distinctive_missing(ok), ok
    for bad in ('specific companies', 'exact amount', '199', 'the year', 'March', '', 'Q', 'the CEO'):
        assert not C.distinctive_missing(bad), bad


def test_check_row_rejects_vacuous_adjacent_terms():
    passages = filler(2)
    messages = B.chat_messages('What exact amount did Berkshire pay for the specific companies?', passages) + [{'role': 'assistant', 'content': REFUSAL_LINE}]
    row = {'id': 'r', 'kind': 'refusal', 'variant': 'adjacent', 'type': 'unanswerable', 'source_ids': ['1990_p0001'], 'served_ids': ['1990_p0000'],
           'messages': messages, 'absent_terms': ['specific companies']}
    assert 'refusal_term_not_distinctive' in B.check_row(row)
    assert 'refusal_term_not_distinctive' not in B.check_row(dict(row, absent_terms=['Fechheimer']))


# ------------------------------------------------------------------ item 3b: refusal_near
def near_job():
    return {'id': 'v7:refusal_near:train:0001', 'kind': 'refusal_near', 'split': 'train', 'source_ids': ['1995_p0001']}


def near_raw(question, present=('GEICO',), missing=('Zenith',)):
    return {'v7:refusal_near:train:0001': {'raw': json.dumps({'question': question, 'present': list(present), 'missing': list(missing)}), 'calls': 1}}


def test_refusal_near_prompt_demands_a_topic_that_is_in_the_letters():
    d = doc('1995_p0001', S, 1995)
    prompt = B.refusal_near_prompt([d])
    assert 'IS' in prompt and 'present' in prompt and 'missing' in prompt and '2025' in prompt and 'price' in prompt
    rec = B.job_record(near_job(), {'1995_p0001': d}, {})
    assert rec['expect'] == 'refusal_near' and rec['needs_teacher'] and rec['messages'][1]['content'] == prompt


@pytest.mark.parametrize('question,missing,reason', [
    ('What did GEICO pay in 2026 to buy Zenith National Insurance from its owners in the market?', ['Zenith'], 'refusal_near_cue'),
    ('What was the closing stock price of GEICO when Zenith joined the group of subsidiaries?', ['Zenith'], 'refusal_near_cue'),
    ('What did GEICO say about the specific amount that was paid for the other companies it bought?', ['specific amount'], 'bad_part'),
    ('What did Berkshire say about GEICO policyholders and the Zenith acquisition in the letters?', ['Zenith'], None),
])
def test_refusal_near_teacher_output_checks(tmp_path, question, missing, reason):
    full = {'1995_p0001': doc('1995_p0001', S, 1995)}
    gen = generator(full, tmp_path, serving=FakeServing(extra=[hit('1995_p0001', S, 1995)]), recorded=near_raw(question, missing=missing))
    row, got = gen._attempt(near_job(), {'k': 8, 'teacher_calls': 0})
    assert got == reason
    if reason is None:
        assert row['kind'] == 'refusal_near' and row['present_terms'] == ['GEICO'] and B.check_row(row) == []


def test_refusal_near_rejects_when_the_missing_term_is_in_the_served_text_or_the_topic_is_not(tmp_path):
    full = {'1995_p0001': doc('1995_p0001', S, 1995)}
    q = 'What did Berkshire say about GEICO policyholders and the Zenith acquisition in the letters?'
    served = FakeServing(extra=[hit('1996_p0001', ['Berkshire acquired Zenith during the year with cash.'], 1996)])
    gen = generator(full, tmp_path, serving=served, recorded=near_raw(q))
    assert gen._attempt(near_job(), {'k': 8, 'teacher_calls': 0})[1] == 'refusal_terms_present_in_served'
    nothing = near_raw(q.replace('GEICO', 'Unicorn'), present=('Unicorn',))
    gen = generator(full, tmp_path, recorded=nothing)
    assert gen._attempt(near_job(), {'k': 8, 'teacher_calls': 0})[1] == 'bad_part'     # the present term is not in the source passage
    gen = generator(full, tmp_path, recorded=near_raw(q))                               # GEICO is in the letter but no served passage is about it
    assert gen._attempt(near_job(), {'k': 8, 'teacher_calls': 0})[1] == 'refusal_near_topic_not_served'


def test_check_row_flags_a_refusal_near_row_with_a_cue_or_an_absent_topic(tmp_path):
    full = {'1995_p0001': doc('1995_p0001', S, 1995)}
    q = 'What did Berkshire say about GEICO policyholders and the Zenith acquisition in the letters?'
    gen = generator(full, tmp_path, serving=FakeServing(extra=[hit('1995_p0001', S, 1995)]), recorded=near_raw(q))
    row, _ = gen._attempt(near_job(), {'k': 8, 'teacher_calls': 0})
    assert B.check_row(row) == []
    assert 'refusal_near_topic_absent' in B.check_row(dict(row, present_terms=['Unicorn']))
    assert 'refusal_near_cue' in B.check_row(dict(row, question=q + ' In 2026?'))


# ------------------------------------------------------------------ item 3c: partial
def partial_job():
    return {'id': 'v7:partial:train:0001', 'kind': 'partial', 'split': 'train', 'source_ids': ['1995_p0001']}


def partial_obj(**over):
    obj = {'question': 'In 1995, how many shares did Berkshire repurchase, how big was its float, and what did Berkshire pay for Fechheimer?',
           'parts': [{'ask': 'buyback', 'passage': 1, 'sentences': [1, 2]}, {'ask': 'float', 'passage': 1, 'sentences': [4]}],
           'absent': {'ask': 'the price Berkshire paid for Fechheimer', 'missing': ['Fechheimer']}}
    obj.update(over)
    return obj


def test_partial_prompt_and_plan_validation():
    d = doc('1995_p0001', S, 1995)
    prompt = B.partial_prompt({'id': 'x', 'kind': 'partial'}, [d])
    assert 'ONE part' in prompt and '"absent"' in prompt and 'The retrieved passages do not cover' in prompt
    reason, plan = B.validate_partial(partial_obj(), [d])
    assert reason is None and plan['gap'] == {'ask': 'the price Berkshire paid for Fechheimer', 'missing': ['Fechheimer']} and len(plan['obligations']) == 2
    assert B.validate_partial(partial_obj(absent={'ask': 'x', 'missing': ['Fechheimer']}), [d])[0] == 'bad_part'
    assert B.validate_partial(partial_obj(absent={'ask': 'the price', 'missing': ['GEICO']}), [d])[0] == 'bad_part'      # GEICO is in the passage
    assert B.validate_partial(partial_obj(absent={'ask': 'the price', 'missing': ['Zenith']}), [d])[0] == 'bad_part'     # not in the question
    assert B.validate_partial(partial_obj(parts=partial_obj()['parts'] * 2), [d])[0] in ('parts_count', 'duplicate_sentence')


def test_partial_row_is_verbatim_parts_plus_one_gap_line_and_passes_check_row(tmp_path):
    full = {'1995_p0001': doc('1995_p0001', S, 1995)}
    gen = generator(full, tmp_path, recorded={'v7:partial:train:0001': {'raw': json.dumps(partial_obj()), 'calls': 1}})
    row, reason = gen._attempt(partial_job(), {'k': 8, 'teacher_calls': 0})
    assert reason is None, reason
    lines = row['messages'][2]['content'].split('\n')
    assert len(lines) == 3 and lines[-1] == 'The retrieved passages do not cover the price Berkshire paid for Fechheimer.' and lines[-1] == C.gap_line(row['gap']['ask'])
    assert all(' [' in l for l in lines[:-1]) and row['kind'] == 'partial' and row['type'] == 'answerable'
    assert B.check_row(row) == []
    # serving: the validator removes the uncited gap line and nothing else; only the verbatim cited lines are shown to the user
    passages = [(int(y), t) for y, _, t in B.parse_prompt_passages(row['messages'][1]['content']).values()]
    assert C.serving_changes(row['messages'][2]['content'], passages) is True
    assert C.serving_changes(row['messages'][2]['content'], passages, gap_lines=[lines[-1]]) is False


def test_partial_row_fails_check_row_when_the_gap_is_wrong_or_the_missing_term_is_served(tmp_path):
    full = {'1995_p0001': doc('1995_p0001', S, 1995)}
    gen = generator(full, tmp_path, recorded={'v7:partial:train:0001': {'raw': json.dumps(partial_obj()), 'calls': 1}})
    row = gen._attempt(partial_job(), {'k': 8, 'teacher_calls': 0})[0]
    cut = dict(row, messages=row['messages'][:2] + [{'role': 'assistant', 'content': '\n'.join(row['messages'][2]['content'].split('\n')[:-1])}])
    assert 'obligation_count' in B.check_row(cut)
    other = row['messages'][2]['content'].replace('Fechheimer', 'Dexter')
    assert 'gap_line_mismatch' in B.check_row(dict(row, messages=row['messages'][:2] + [{'role': 'assistant', 'content': other}]))
    assert 'partial_gap_term_present' in B.check_row(dict(row, gap=dict(row['gap'], missing=['GEICO'])))


def test_partial_job_is_rejected_when_the_asked_detail_is_in_the_served_context(tmp_path):
    full = {'1995_p0001': doc('1995_p0001', S, 1995)}
    served = FakeServing(extra=[hit('1996_p0001', ['Berkshire bought Fechheimer for a fair price in 1996.'], 1996)])
    gen = generator(full, tmp_path, serving=served, recorded={'v7:partial:train:0001': {'raw': json.dumps(partial_obj()), 'calls': 1}})
    assert gen._attempt(partial_job(), {'k': 8, 'teacher_calls': 0})[1] == 'partial_gap_present_in_served'


# ------------------------------------------------------------------ item 4: comparison rows in the served per-period shape
class FakeCompareServing:
    """round5.Serving.prompts for a comparison: one dict per period with that period's subset of ONE fitted context.
    Its 'query' is a wrong focus string on purpose: the builder must not use it."""
    def __init__(self, context):
        self.context = context

    def prompts(self, question, history=None):
        return [{'period': period_label(p), 'query': f'{question} (focus: WRONG)', 'hits': [h for h in self.context if _meta_matches(h.metadata, p)],
                 'gate': True, 'retrieved_ids': [h.id for h in self.context], 'reranked': True} for p in comparison_periods(question)]


def comparison_job():
    return {'id': 'v7:comparison:train:0001', 'kind': 'comparison', 'split': 'train', 'source_ids': ['1995_p0001', '2010_p0001']}


def comparison_obj(**over):
    obj = {'question': 'How did insurance float and GEICO policyholder growth in 1995 compare with 2010?',
           'parts': [{'ask': '1995 float', 'passage': 1, 'sentences': [4]}, {'ask': '1995 GEICO', 'passage': 1, 'sentences': [5]},
                     {'ask': '2010 float', 'passage': 2, 'sentences': [2]}, {'ask': '2010 GEICO', 'passage': 2, 'sentences': [3]}]}
    obj.update(over)
    return obj


def compare_full():
    return {'1995_p0001': doc('1995_p0001', S, 1995), '2010_p0001': doc('2010_p0001', T, 2010)}


def compare_context():
    return [hit('1995_p0001', S, 1995), hit('2010_p0001', T, 2010)] + filler(2, 1995) + filler(2, 2010)


def test_comparison_validate_plan_needs_a_two_year_question_and_parts_in_both_letters():
    docs = list(compare_full().values())
    reason, plan = B.validate_plan(comparison_obj(), 'comparison', docs)
    assert reason is None and [o['source_id'] for o in plan['obligations']] == ['1995_p0001', '1995_p0001', '2010_p0001', '2010_p0001']
    assert B.validate_plan(comparison_obj(question='How did insurance float and GEICO policyholder growth change during 1995 at Berkshire?'), 'comparison', docs)[0] == 'not_comparison_question'
    only95 = comparison_obj(parts=comparison_obj()['parts'][:2])
    assert B.validate_plan(only95, 'comparison', docs)[0] in ('comparison_period_missing', 'parts_count')
    prompt = B.teacher_prompt(comparison_job(), docs)
    assert 'letter year 1995' in prompt and 'letter year 2010' in prompt and 'compar' in prompt


def test_comparison_job_builds_one_row_per_period_call_exactly_as_serving_does(tmp_path):
    full, context = compare_full(), compare_context()
    gen = generator(full, tmp_path, serving=FakeCompareServing(context), recorded={'v7:comparison:train:0001': {'raw': json.dumps(comparison_obj()), 'calls': 1}})
    rows, reason = gen._attempt(comparison_job(), {'k': 8, 'teacher_calls': 0})
    assert reason is None, reason
    question = comparison_obj()['question']
    periods = comparison_periods(question)
    assert [r['period'] for r in rows] == ['1995', '2010'] and all(r['kind'] == 'comparison' for r in rows)
    for row, period in zip(rows, periods):
        assert row['question'] == period_question(question, period, periods) and '(focus: ' in row['question'] and row['job_question'] == question
        years = {y for y, _, _ in B.parse_prompt_passages(row['messages'][1]['content']).values()}
        assert years == {str(period['year'])}
        assert B.check_row(row) == [] and 'WRONG' not in row['messages'][1]['content']
    # byte-compare with what the live compare path sends to the model
    sent = []

    class Llm:
        provider_name = 'llama'

        def generate(self, prompt, max_new_tokens=0):
            sent.append(prompt)
            return REFUSAL_LINE
    generate_comparison_answer(Llm(), question, context, max_new_tokens=300)
    assert len(sent) == 2
    for row, prompt in zip(rows, sent):
        assert row['messages'][0]['content'] + '\n\n' + row['messages'][1]['content'] == prompt


def test_comparison_job_is_rejected_when_the_question_does_not_take_the_comparison_path(tmp_path):
    full = compare_full()
    obj = comparison_obj(question='What happened to GEICO policyholders and insurance float at Berkshire during 1995?')
    gen = generator(full, tmp_path, serving=FakeCompareServing(compare_context()), recorded={'v7:comparison:train:0001': {'raw': json.dumps(obj), 'calls': 1}})
    rows, reason = gen._attempt(comparison_job(), {'k': 8, 'teacher_calls': 0})
    assert rows is None and reason in ('not_comparison_question', 'comparison_period_missing')
    # a normal single-path server for a comparison job
    gen = generator(full, tmp_path, serving=FakeServing(), recorded={'v7:comparison:train:0001': {'raw': json.dumps(comparison_obj()), 'calls': 1}})
    assert gen._attempt(comparison_job(), {'k': 8, 'teacher_calls': 0})[1] == 'not_comparison_path'


def test_attempt_records_both_rows_of_a_comparison_job(tmp_path):
    full = compare_full()
    gen = generator(full, tmp_path, serving=FakeCompareServing(compare_context()), recorded={'v7:comparison:train:0001': {'raw': json.dumps(comparison_obj()), 'calls': 1}})
    record = gen.attempt(dict(comparison_job(), split='train'))
    assert record['kept'] and len(gen.rows) == 2 and record['obligations'] == 4
    assert B.manifest_for(tmp_path, gen.rows, gen.attempts, {}, {}, 1., 10)['row_counts_per_kind']['comparison'] == 2


def test_comparison_rows_have_their_own_ids_and_sources_and_cite_every_line(tmp_path):
    full = compare_full()
    gen = generator(full, tmp_path, serving=FakeCompareServing(compare_context()), recorded={'v7:comparison:train:0001': {'raw': json.dumps(comparison_obj()), 'calls': 1}})
    rows, _ = gen._attempt(comparison_job(), {'k': 8, 'teacher_calls': 0})
    assert rows[0]['id'] != rows[1]['id'] and rows[0]['id'].startswith('v7:comparison:train:0001') and rows[0]['source_ids'] == ['1995_p0001'] and rows[1]['source_ids'] == ['2010_p0001']
    for row in rows:
        target = row['messages'][2]['content']
        assert target.count('\n') == 1 and all(line.endswith(']') for line in target.split('\n'))


# ------------------------------------------------------------------ exact-N multipart and cross_letter assemble into rows that pass check_row
def test_exact_n_multipart_and_cross_letter_jobs_assemble_into_valid_rows(tmp_path):
    d5, obj5 = numbered(5)
    gen = generator({'1995_p0001': d5}, tmp_path, recorded={'v7:multipart:train:0001': {'raw': json.dumps(obj5), 'calls': 1}})
    job = {'id': 'v7:multipart:train:0001', 'kind': 'multipart', 'split': 'train', 'source_ids': ['1995_p0001'], 'parts_range': [5, 5]}
    row, reason = gen._attempt(job, {'k': 8, 'teacher_calls': 0})
    assert reason is None, reason
    assert len(row['obligations']) == 5 and row['messages'][2]['content'].count('\n') == 4 and B.check_row(row) == []
    wrong = generator({'1995_p0001': d5}, tmp_path / 'w', recorded={'v7:multipart:train:0001': {'raw': json.dumps(obj5), 'calls': 1}})
    assert wrong._attempt(dict(job, parts_range=[4, 4]), {'k': 8, 'teacher_calls': 0})[1] == 'parts_count'

    a = doc('1990_p0001', [f'Berkshire reported distinct fact number {i} about its insurance business in 1990 clearly.' for i in range(1, 5)], 1990)
    b = doc('1995_p0001', [f'Berkshire reported distinct fact number {i} about its shoe business in 1995 clearly.' for i in range(1, 5)], 1995)
    question = 'What did Berkshire report about insurance business facts, about shoe business facts, and about the other distinct results it kept?'
    raw = {'question': question, 'parts': [{'ask': 'a', 'passage': 1, 'sentences': [1]}, {'ask': 'b', 'passage': 1, 'sentences': [3]},
                                           {'ask': 'c', 'passage': 2, 'sentences': [1]}, {'ask': 'd', 'passage': 2, 'sentences': [3]}]}
    gen = generator({'1990_p0001': a, '1995_p0001': b}, tmp_path / 'x', recorded={'v7:cross_letter:train:0001': {'raw': json.dumps(raw), 'calls': 1}})
    cross = {'id': 'v7:cross_letter:train:0001', 'kind': 'cross_letter', 'split': 'train', 'source_ids': ['1990_p0001', '1995_p0001'], 'parts_range': [4, 4]}
    row, reason = gen._attempt(cross, {'k': 8, 'teacher_calls': 0})
    assert reason is None, reason
    assert {p['year'] for p in row['parts']} == {1990, 1995} and len(row['injected']) == 2 and B.check_row(row) == []
    assert B.split_problem(dict(row, split='train'), {1990}) == 'split_straddle'
