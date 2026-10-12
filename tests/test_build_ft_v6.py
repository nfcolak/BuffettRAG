"""ft_v6 generator: teacher-plan validation, deterministic target, serving-slot injection, row checks (no teacher, no index)."""
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_env = dict(os.environ)   # scripts/ft/common.py sets offline env flags on import; keep them out of other tests
sys.path.insert(0, str(ROOT / 'scripts' / 'ft'))
import build_ft_v6 as B  # noqa: E402
os.environ.clear()
os.environ.update(_env)

from src.generation.prompt import REFUSAL_LINE  # noqa: E402
from src.storage import SearchHit  # noqa: E402


@pytest.fixture(autouse=True)
def v2_prompt(monkeypatch):
    monkeypatch.setenv('PROMPT_VERSION', 'v2')


S = [
    'Berkshire repurchased 120,000 shares of its common stock during 1995.',
    'The buyback was funded from retained earnings of the insurance operations.',
    'This explains why the float grew faster than book value in that year.',
    'Insurance float reached 3.4 billion dollars at the end of 1995.',
    'GEICO added 400,000 policyholders and its combined ratio improved to 96.',
]


def doc(ident, sentences, year=1995):
    return SimpleNamespace(id=ident, text=' '.join(sentences), metadata={'year': year, 'source_file': f'buffet_{year}.txt'})


def hit(ident, sentences, year=1995):
    return SearchHit(id=ident, text=' '.join(sentences), metadata={'year': year, 'source_file': f'buffet_{year}.txt'}, score=1.)


def plan_obj(**over):
    obj = {'question': 'In 1995, how many shares did Berkshire buy back, how big was its float and what happened at GEICO?',
           'parts': [{'ask': 'buyback', 'passage': 1, 'sentences': [1, 2]},
                     {'ask': 'float', 'passage': 1, 'sentences': [4]},
                     {'ask': 'GEICO', 'passage': 1, 'sentences': [5]}]}
    obj.update(over)
    return obj


def test_sentence_list_is_what_the_7b_sees():
    d = doc('1995_p0001', S)
    assert len(B.sentences_of(d)) == 5


def test_validate_plan_accepts_adjacent_pair_and_builds_obligations():
    d = doc('1995_p0001', S)
    reason, plan = B.validate_plan(plan_obj(), 'multipart', [d])
    assert reason is None and [o['sentence_numbers'] for o in plan['obligations']] == [[1, 2], [4], [5]]
    assert plan['obligations'][0]['sentences'] == S[:2]


@pytest.mark.parametrize('change,expected', [
    ({'parts': plan_obj()['parts'][:2]}, 'parts_count'),
    ({'parts': [{'ask': 'a', 'passage': 1, 'sentences': [1, 3]}] + plan_obj()['parts'][1:]}, 'non_adjacent'),
    ({'parts': [{'ask': 'a', 'passage': 1, 'sentences': [9]}] + plan_obj()['parts'][1:]}, 'sentence_index'),
    ({'parts': [{'ask': 'a', 'passage': 1, 'sentences': [3]}] + plan_obj()['parts'][1:]}, 'dependent_start'),
    ({'parts': [{'ask': 'a', 'passage': 1, 'sentences': [4]}] + plan_obj()['parts'][1:]}, 'duplicate_sentence'),
    ({'question': 'How big was float?'}, 'bad_question'),
    ({'question': 'What does the passage say about the buyback of shares and the float growth in the year?'}, 'question_refers_to_passage'),
])
def test_validate_plan_rejects(change, expected):
    reason, _ = B.validate_plan(plan_obj(**change), 'multipart', [doc('1995_p0001', S)])
    assert reason == expected


def test_temporal_needs_the_year_in_the_question():
    obj = plan_obj(question='How many shares did Berkshire buy back, how big was its float and what happened at GEICO?')
    assert B.validate_plan(obj, 'temporal', [doc('1995_p0001', S)])[0] == 'year_missing'


def test_follow_up_needs_pronoun_and_new_sentences():
    d = doc('1995_p0001', S)
    obj = {'first_question': 'How many shares did Berkshire repurchase in 1995?', 'first_part': {'passage': 1, 'sentences': [1]},
           'question': 'How was it funded, and how large did their float get?',
           'parts': [{'ask': 'funding', 'passage': 1, 'sentences': [2]}, {'ask': 'float', 'passage': 1, 'sentences': [4]}]}
    reason, plan = B.validate_plan(obj, 'follow_up', [d])
    assert reason is None and plan['history']['sentences'] == [S[0]]
    obj['parts'][0]['sentences'] = [1]
    assert B.validate_plan(obj, 'follow_up', [d])[0] == 'followup_same_sentences'
    obj['parts'][0]['sentences'] = [2]
    obj['question'] = 'How was the Berkshire buyback funded, and how large did the float get?'
    assert B.validate_plan(obj, 'follow_up', [d])[0] == 'followup_pronoun'


def test_unserved_source_goes_to_a_random_slot_and_target_is_numbered_against_served_context():
    source = doc('1995_p0001', S)
    hits = [hit(f'1990_p{i:04d}', [f'Other passage number {i} says something else entirely about nothing.']) for i in range(8)]
    plan = B.validate_plan(plan_obj(), 'multipart', [source])[1]
    placed, injected, error = B.place_sources(hits, plan['obligations'], {'1995_p0001': source}, random.Random(3))
    assert error is None and len(placed) == 8 and len(injected) == 1 and placed[injected[0]['slot'] - 1].id == '1995_p0001'
    target, slots = B.build_target(plan['obligations'], placed)
    marker = f"[{injected[0]['slot']}]"
    assert target.split('\n') == [f'{S[0]} {marker} {S[1]} {marker}', f'{S[3]} {marker}', f'{S[4]} {marker}']


def test_served_source_keeps_its_slot_and_is_not_injected():
    source = doc('1995_p0001', S)
    hits = [hit('1990_p0001', ['Unrelated words fill this other passage for the test.']), hit('1995_p0001', S)]
    plan = B.validate_plan(plan_obj(), 'multipart', [source])[1]
    placed, injected, error = B.place_sources(hits, plan['obligations'], {'1995_p0001': source}, random.Random(3))
    assert (error, injected) == (None, []) and B.build_target(plan['obligations'], placed)[0].endswith('[2]')


def make_row(target=None, passages=None, kind='multipart'):
    source = doc('1995_p0001', S)
    hits = passages or [hit('1990_p0001', ['Unrelated words fill this other passage for the test.']), hit('1995_p0001', S)]
    plan = B.validate_plan(plan_obj(), 'multipart', [source])[1]
    target = target or B.build_target(plan['obligations'], hits)[0]
    messages = B.chat_messages(plan['question'], hits) + [{'role': 'assistant', 'content': target}]
    obligations = [{k: o[k] for k in ('ask', 'source_id', 'sentence_numbers', 'sentences')} for o in plan['obligations']]
    return {'id': 'v6:test', 'kind': kind, 'type': 'answerable', 'source_ids': ['1995_p0001'], 'served_ids': ['1990_p0001', '1995_p0001'],
            'obligations': obligations, 'messages': messages}


def test_check_row_accepts_a_built_row():
    assert B.check_row(make_row()) == []


def test_check_row_rejects_paraphrase_wrong_slot_missing_marker_and_excluded_source():
    row = make_row()
    changed = dict(row, messages=row['messages'][:2] + [{'role': 'assistant', 'content': row['messages'][2]['content'].replace('repurchased', 'bought back')}])
    assert 'obligation_not_in_target' in B.check_row(changed)
    wrong = dict(row, messages=row['messages'][:2] + [{'role': 'assistant', 'content': row['messages'][2]['content'].replace('[2]', '[1]')}])
    assert 'sentence_not_in_cited_passage' in B.check_row(wrong)
    bare = dict(row, messages=row['messages'][:2] + [{'role': 'assistant', 'content': row['messages'][2]['content'].replace(' [2]', '', 1)}])
    assert 'sentence_without_marker' in B.check_row(bare)
    assert 'source_excluded' in B.check_row(row, excluded={'1995_p0001'})
    assert 'partnership_passage' in B.check_row(dict(row, served_ids=['bpl1960_p0001']))
    v1 = dict(row, messages=[{'role': 'system', 'content': 'old'}] + row['messages'][1:])
    assert 'system_prompt_not_v2' in B.check_row(v1)


def test_check_row_token_budget():
    assert 'too_long_for_max_seq' in B.check_row(make_row(), count_tokens=lambda messages: 6000)


def test_refusal_row_needs_exact_line_and_absent_terms():
    passages = [hit('1990_p0001', ['Berkshire owns no Tesla shares and reports nothing for the year.'])]
    messages = B.chat_messages('What was Nvidia revenue in 2025?', passages) + [{'role': 'assistant', 'content': REFUSAL_LINE}]
    row = {'id': 'r', 'kind': 'refusal', 'type': 'unanswerable', 'source_ids': ['1990_p0001'], 'served_ids': ['1990_p0001'], 'messages': messages,
           'absent_terms': ['Nvidia', '2025']}
    assert B.check_row(row) == []
    assert 'refusal_terms_not_absent' in B.check_row(dict(row, absent_terms=['Tesla']))
    row['messages'][2]['content'] = REFUSAL_LINE + ' Sorry.'
    assert 'refusal_target_not_exact' in B.check_row(row)


def test_distinctive_terms_and_quota_shape():
    assert B.distinctive_terms("What was Apple's exact audited revenue for fiscal year 2025?") == {'Apple', '2025'}
    quotas = B.quotas_for(1760)
    assert sum(quotas.values()) == 1760 and sum(n for (_, s), n in quotas.items() if s == 'valid') == 160
    assert quotas[('multipart', 'train')] + quotas[('multipart', 'valid')] == 968
