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
def training_prompt(monkeypatch):
    monkeypatch.setenv('PROMPT_VERSION', B.PROMPT_VERSION)


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
    reason, plan = B.validate_plan(plan_obj(), 'temporal', [d])
    assert reason is None and [o['sentence_numbers'] for o in plan['obligations']] == [[1, 2], [4], [5]]
    assert plan['obligations'][0]['sentences'] == S[:2]


@pytest.mark.parametrize('change,expected', [
    ({'parts': plan_obj()['parts'][:2]}, 'parts_count'),
    ({'parts': [{'ask': 'a', 'passage': 1, 'sentences': [1, 3]}] + plan_obj()['parts'][1:]}, 'non_adjacent'),
    ({'parts': [{'ask': 'a', 'passage': 1, 'sentences': [9]}] + plan_obj()['parts'][1:]}, 'sentence_index'),
    ({'parts': [{'ask': 'a', 'passage': 1, 'sentences': [2]}, {'ask': 'b', 'passage': 1, 'sentences': [3]}, {'ask': 'c', 'passage': 1, 'sentences': [5]}]}, 'dependent_start'),
    ({'parts': [{'ask': 'a', 'passage': 1, 'sentences': [4]}] + plan_obj()['parts'][1:]}, 'duplicate_sentence'),
    ({'question': 'How big was float?'}, 'bad_question'),
    ({'question': 'What does the passage say about the buyback of shares and the float growth in the year?'}, 'question_refers_to_passage'),
])
def test_validate_plan_rejects(change, expected):
    reason, _ = B.validate_plan(plan_obj(**change), 'temporal', [doc('1995_p0001', S)])
    assert reason == expected


def test_dependent_sentence_is_repaired_with_its_predecessor():
    parts = [{'ask': 'a', 'passage': 1, 'sentences': [1]}, {'ask': 'b', 'passage': 1, 'sentences': [3]}, {'ask': 'c', 'passage': 1, 'sentences': [5]}]
    reason, plan = B.validate_plan(plan_obj(parts=parts), 'temporal', [doc('1995_p0001', S)])
    assert reason is None and plan['obligations'][1]['sentence_numbers'] == [2, 3]


def test_layout_debris_is_not_a_fact():
    assert B.bad_sentence('Late in 2009 * * * * * * we agreed to buy BNSF.') and B.bad_sentence('limited way that I do with subsidiaries.')
    assert not B.bad_sentence(S[0])


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
    plan = B.validate_plan(plan_obj(), 'temporal', [source])[1]
    placed, injected, error = B.place_sources(hits, plan['obligations'], {'1995_p0001': source}, random.Random(3))
    assert error is None and len(placed) == 8 and len(injected) == 1 and placed[injected[0]['slot'] - 1].id == '1995_p0001'
    target, slots = B.build_target(plan['obligations'], placed)
    marker = f"[{injected[0]['slot']}]"
    assert target.split('\n') == [f'{S[0]} {marker} {S[1]} {marker}', f'{S[3]} {marker}', f'{S[4]} {marker}']


def test_served_source_keeps_its_slot_and_is_not_injected():
    source = doc('1995_p0001', S)
    hits = [hit('1990_p0001', ['Unrelated words fill this other passage for the test.']), hit('1995_p0001', S)]
    plan = B.validate_plan(plan_obj(), 'temporal', [source])[1]
    placed, injected, error = B.place_sources(hits, plan['obligations'], {'1995_p0001': source}, random.Random(3))
    assert (error, injected) == (None, []) and B.build_target(plan['obligations'], placed)[0].endswith('[2]')


def make_row(target=None, passages=None, kind='multipart'):
    source = doc('1995_p0001', S)
    hits = passages or [hit('1990_p0001', ['Unrelated words fill this other passage for the test.']), hit('1995_p0001', S)]
    plan = B.validate_plan(plan_obj(), 'temporal', [source])[1]
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
    assert 'system_prompt_mismatch' in B.check_row(v1)
    assert 'system_prompt_mismatch' in B.check_row(dict(row, prompt_version='v2'))   # a v3 prompt stored as a v2 row


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
    rows = sum(n * B.ROWS_PER_JOB.get(k, 1) for (k, _), n in quotas.items())       # a comparison job is two rows
    assert abs(rows - 1760) <= 6 and abs(sum(n * B.ROWS_PER_JOB.get(k, 1) for (k, s), n in quotas.items() if s == 'valid') - 1760 / 11) < 12
    assert quotas[('single', 'train')] + quotas[('single', 'valid')] == round(1760 * B.SHARE['single'])


# ------------------------------------------------------------------ plan / assemble split
def test_partnership_in_context_is_detected_by_id_source_and_collection():
    clean = [hit('1990_p0001', ['Plain Berkshire text for the test.'])]
    assert not B.served_has_partnership(clean)
    assert B.served_has_partnership(clean + [hit('bpl1960_p0003', ['Partnership text.'], year=1960)])
    merged = SearchHit(id='1990_p0001', text='x', metadata={'merged_ids': ['bpl1960_p0004'], 'year': 1990}, score=1.)
    assert B.served_has_partnership([merged])
    assert B.served_has_partnership([SearchHit(id='x', text='x', metadata={'collection': 'partnership', 'year': 1960}, score=1.)])
    row = make_row()
    assert B.check_row(row) == []
    assert 'partnership_passage' in B.check_row(dict(row, passage_ids=['1990_p0001', 'bpl1960_p0001']))
    assert 'partnership_passage' in B.check_row(dict(row, context_passage_ids=['bpl1961_p0002']))
    bad_source = row['messages'][1]['content'].replace('buffet_1995.txt', 'buffett_partnership_1960.txt', 1)
    assert 'partnership_passage' in B.check_row(dict(row, messages=[row['messages'][0], {'role': 'user', 'content': bad_source}, row['messages'][2]]))


def test_job_record_holds_the_exact_teacher_messages_and_shape_info():
    d = doc('1995_p0001', S)
    job = {'id': 'v6:multipart:train:0000', 'kind': 'multipart', 'split': 'train', 'source_ids': ['1995_p0001']}
    rec = B.job_record(job, {'1995_p0001': d}, {'corpus_sha256': 'x'}, lambda messages: 123)
    assert rec['messages'] == [{'role': 'system', 'content': B.SYSTEM}, {'role': 'user', 'content': B.teacher_prompt(job, [d])}]
    assert (rec['needs_teacher'], rec['expect'], rec['n_passages'], rec['sentence_counts'], rec['parts_range'], rec['prompt_tokens']) == \
        (True, 'plan', 1, [5], [4, 5], 123)      # ft_v7: multipart asks for 4-5 parts (an ft_v6 job file carries its own [3, 5])
    assert rec['sampling']['temperature'] == B.TEMPERATURE and rec['sampling']['max_tokens'] == B.MAX_TEACHER_TOKENS
    template = B.job_record({'id': 'r', 'kind': 'refusal', 'split': 'train', 'source_ids': ['1995_p0001'], 'variant': 'template', 'template': 'Q?'},
                            {'1995_p0001': d}, {}, None)
    assert template['messages'] is None and template['needs_teacher'] is False and template['expect'] == 'refusal'


def test_order_jobs_interleaves_kinds_and_keeps_every_job():
    jobs = [{'id': f'{k}{s}{i}', 'kind': k, 'split': s} for k, n in (('multipart', 8), ('refusal', 4)) for s in ('train',) for i in range(n)]
    ordered = B.order_jobs(jobs)
    assert sorted(j['id'] for j in ordered) == sorted(j['id'] for j in jobs)
    assert {j['kind'] for j in ordered[:3]} == {'multipart', 'refusal'}


def test_recorded_generator_validates_the_recorded_json_like_the_local_teacher():
    d = doc('1995_p0001', S)
    gen = B.RecordedGenerator({'a': {'raw': 'Sure: ' + __import__('json').dumps(plan_obj()), 'calls': 2}, 'b': {'raw': 'not json'}}, None, {}, set(), '.', None)
    value, reason = gen.ask_teacher({'id': 'a'}, '', {}, lambda obj: B.validate_plan(obj, 'temporal', [d]))
    assert reason is None and len(value['obligations']) == 3
    record = {}
    assert gen.ask_teacher({'id': 'b'}, '', record, lambda obj: B.validate_plan(obj, 'temporal', [d]))[1] == 'json_invalid'
    assert gen.ask_teacher({'id': 'missing'}, '', {}, lambda obj: (None, obj))[1] == 'no_teacher_output'


def test_legacy_flags_route_to_teach_local_and_subcommands_parse(monkeypatch):
    seen = {}
    monkeypatch.setattr(B, 'run', lambda args, parser: seen.setdefault('run', args))
    monkeypatch.setattr(B, 'plan_command', lambda args: seen.setdefault('plan', args))
    monkeypatch.setattr(B, 'assemble_command', lambda args: seen.setdefault('assemble', args))
    B.main(['--pilot', '2', '--kinds', 'refusal'])
    assert seen['run'].pilot == 2 and seen['run'].kinds == 'refusal' and seen['run'].command == 'teach-local'
    B.main(['plan', '--n', '2600', '--seed', '7', '--out', 'x.jsonl'])
    assert (seen['plan'].n, seen['plan'].seed) == (2600, 7)
    B.main(['assemble', '--jobs', 'j.jsonl', '--teacher', 't.jsonl'])
    assert seen['assemble'].out_dir == B.OUT_DIR and seen['assemble'].target is None


def test_training_rows_equal_the_serving_split_of_the_same_prompt(monkeypatch):
    """chat_messages (training) and llama_provider.split_prompt (serving) give the same system/user for the active version."""
    from src.generation.prompt import build_cited_prompt
    from src.generation.providers.llama_provider import split_prompt
    hits = [hit('1995_p0001', S)]
    for version in ('v1', B.PROMPT_VERSION):
        monkeypatch.setenv('PROMPT_VERSION', version)
        prompt = build_cited_prompt('What happened in 1995?', hits)
        system, user = split_prompt(prompt)
        if version == B.PROMPT_VERSION:
            rows = B.chat_messages('What happened in 1995?', hits)
            assert (system, user) == (rows[0]['content'], rows[1]['content'])
        else:
            assert user.startswith('BEGIN UNTRUSTED PASSAGES')


# ------------------------------------------------------------------ --passages-mix (FT-r3: mixed served context sizes)
def test_passages_mix_parsing():
    assert B.parse_passages_mix(None) == [8] and B.parse_passages_mix('') == [8]
    assert B.parse_passages_mix('8,12,16') == [8, 12, 16]
    for bad in ('8,x', '0', '8,40', ','):
        with pytest.raises(ValueError):
            B.parse_passages_mix(bad)


def test_pick_k_is_deterministic_in_the_mix_and_spread_over_it():
    mix = [8, 12, 16]
    ids = [f'v6:multipart:train:{i:04d}' for i in range(600)]
    first = [B.pick_k(i, mix) for i in ids]
    assert first == [B.pick_k(i, mix) for i in ids] and set(first) == set(mix)
    assert all(100 <= first.count(k) <= 300 for k in mix)
    assert {B.pick_k(i, [8]) for i in ids} == {8}
    assert B.k_histogram([{'k': 16}, {'k': 8}, {'k': 16}, {}]) == {'8': 2, '16': 2}


def test_k16_row_prompt_equals_the_serving_prompt_for_the_same_question():
    """The training user/system messages of a 16-passage row are exactly what ask_flow + llama_provider.split_prompt send at K=16."""
    from src.generation.prompt import build_cited_prompt
    from src.generation.providers.llama_provider import split_prompt
    hits = [hit(f'1990_p{i:04d}', [f'Passage number {i} says something distinct about topic {i} for the test.'], year=1990 + i % 5) for i in range(16)]
    history = [{'role': 'user', 'content': 'How many shares did Berkshire repurchase?'}, {'role': 'assistant', 'content': 'See [1].'}]
    for hist in (None, history):
        system, user = split_prompt(build_cited_prompt('What happened next in 1995?', hits, hist))
        rows = B.chat_messages('What happened next in 1995?', hits, hist)
        assert rows == [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        assert user.count('\n[16] (year=') == 1 and '[17]' not in user


class FakeServing:
    """Records the LLM_CONTEXT_PASSAGES each call sees and serves that many unrelated passages."""
    def __init__(self):
        self.seen = []

    def prompts(self, question, history=None):
        k = int(os.environ['LLM_CONTEXT_PASSAGES'])
        self.seen.append(k)
        hits = [hit(f'1990_p{i:04d}', [f'Unrelated passage {i} about something else entirely in the test.']) for i in range(k)]
        return [{'period': None, 'hits': hits, 'gate': True, 'retrieved_ids': [h.id for h in hits], 'reranked': True}]


def test_generator_serves_each_row_with_its_own_k_and_gold_sources_stay_inside(tmp_path, monkeypatch):
    monkeypatch.delenv('LLM_CONTEXT_PASSAGES', raising=False)
    source = doc('1995_p0001', S)
    serving = FakeServing()
    gen = B.Generator(None, serving, {'1995_p0001': source}, set(), tmp_path, None, B.MAX_SEQ_MIX, False, [8, 12, 16])
    plan = B.validate_plan(plan_obj(), 'temporal', [source])[1]
    rows = []
    for index in range(30):
        job = {'id': f'v6:multipart:train:{index:04d}', 'kind': 'multipart', 'split': 'train', 'source_ids': ['1995_p0001']}
        record = {'k': B.pick_k(job['id'], gen.mix)}
        row, reason = gen._finish(job, [source], plan, None, record)
        assert reason is None and row['k'] == record['k'] and row['passages'] == row['k'] and B.check_row(row) == []
        assert '1995_p0001' in row['passage_ids'] and len(row['injected']) == 1
        rows.append(row)
    assert serving.seen == [r['k'] for r in rows] and 'LLM_CONTEXT_PASSAGES' not in os.environ
    assert set(B.k_histogram(rows)) == {'8', '12', '16'}
    default = B.Generator(None, FakeServing(), {}, set(), tmp_path / 'd', None)
    assert default.mix == [8] and default.max_seq == B.MAX_SEQ


def test_assemble_flags_parse(monkeypatch):
    seen = {}
    monkeypatch.setattr(B, 'assemble_command', lambda args: seen.setdefault('a', args))
    B.main(['assemble', '--jobs', 'j.jsonl', '--teacher', 't.jsonl', '--passages-mix', '8,12,16'])
    assert seen['a'].passages_mix == '8,12,16' and seen['a'].max_seq is None


# ------------------------------------------------------------------ pv8: refusal term check, out-dir guard, refusal share
def test_refusal_terms_drop_bare_years_and_months_but_keep_entities():
    q = 'What was the closing price of Ethereum on March 3, 2025?'
    assert {'2025', 'March'} <= B.distinctive_terms(q)
    assert B.refusal_terms(q) == B.distinctive_terms(q) - {'2025', 'March'}
    assert 'Ethereum' in B.refusal_terms(q)


def test_template_refusal_uses_the_relaxed_terms():
    import inspect
    assert "sorted(refusal_terms(job['template']))" in inspect.getsource(B)


def test_refusal_share_check():
    rows = [{'kind': 'refusal'}] * 2 + [{'kind': 'single'}] * 8
    B.check_refusal_share(rows, 0.12)
    with pytest.raises(SystemExit):
        B.check_refusal_share(rows, 0.25)
    B.check_refusal_share([], 0)


def test_assemble_refuses_to_overwrite_an_existing_out_dir(tmp_path):
    (tmp_path / 'train.jsonl').write_text('{}\n')
    args = SimpleNamespace(passages_mix=None, max_seq=None, jobs=str(tmp_path / 'j.jsonl'), teacher=str(tmp_path / 't.jsonl'),
                           out_dir=str(tmp_path), force=False)
    with pytest.raises(SystemExit, match='exists'):
        B.assemble_command(args)


def test_default_out_dir_is_not_the_ft_r3_training_data():
    assert B.OUT_DIR != 'data/ft_v6'
