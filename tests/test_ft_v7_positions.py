"""ft_v7 data builder, items 6 (position mix), 9 (exclusions) and the assemble wiring (kinds, over-length, shares, manifest)."""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_env = dict(os.environ)
sys.path.insert(0, str(ROOT / 'scripts' / 'ft'))
import build_ft_v6 as B  # noqa: E402
import ft_v7_checks as C  # noqa: E402
import round5  # noqa: E402
os.environ.clear()
os.environ.update(_env)

from src.generation.compare import comparison_periods, period_label, period_question  # noqa: E402
from src.storage import SearchHit  # noqa: E402


@pytest.fixture(autouse=True)
def training_prompt(monkeypatch):
    monkeypatch.setenv('PROMPT_VERSION', B.PROMPT_VERSION)
    monkeypatch.delenv('LLM_CONTEXT_PASSAGES', raising=False)
    monkeypatch.setattr(B, 'QUESTION_OVERLAP_MAX', 1.0)


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


def plan_obj():
    return {'question': 'In 1995, how many shares did Berkshire buy back, how big was its float and what happened at GEICO?',
            'parts': [{'ask': 'buyback', 'passage': 1, 'sentences': [1, 2]}, {'ask': 'float', 'passage': 1, 'sentences': [4]},
                      {'ask': 'GEICO', 'passage': 1, 'sentences': [5]}]}


class FakeServing:
    def __init__(self, extra=()):
        self.extra = list(extra)

    def prompts(self, question, history=None):
        k = int(os.environ.get('LLM_CONTEXT_PASSAGES', 8))
        hits = (self.extra + [hit(f'1990_p9{i:03d}', [f'Filler passage number {i} talks about farm equipment weather and nothing else of note here.'], 1990)
                              for i in range(k)])[:k]
        return [{'period': None, 'hits': hits, 'gate': True, 'retrieved_ids': [h.id for h in hits], 'reranked': True}]


def generator(tmp_path, mix=None, extra=(), excluded=(), strict=False, k=16):
    full = {'1995_p0001': doc('1995_p0001', S)}
    return B.Generator(None, FakeServing(extra), full, set(excluded), tmp_path, None, B.MAX_SEQ_MIX, strict, [k], position_mix=mix), full


def finish(gen, index=0, plan=None):
    source = gen.full['1995_p0001']
    plan = plan or B.validate_plan(plan_obj(), 'temporal', [source])[1]
    job = {'id': f'v7:temporal:train:{index:04d}', 'kind': 'temporal', 'split': 'train', 'source_ids': ['1995_p0001']}
    return gen._finish(job, [source], plan, None, {'k': B.pick_k(job['id'], gen.mix)})


# ------------------------------------------------------------------ item 6: --position-mix
def test_position_mix_parsing():
    assert C.parse_position_mix(None) is None and C.parse_position_mix('') is None
    assert C.parse_position_mix('natural,serving,uniform=0.5,0.3,0.2') == {'natural': .5, 'serving': .3, 'uniform': .2}
    assert C.parse_position_mix('serving,uniform=1,0') == {'serving': 1., 'uniform': 0.}
    for bad in ('natural,serving=0.5,0.3', 'natural,serving,uniform=0.5,0.3,0.3', 'natural,bogus=0.5,0.5', 'natural,natural=0.5,0.5', 'natural=x'):
        with pytest.raises(ValueError):
            C.parse_position_mix(bad)


def test_serving_rank_distribution_matches_the_measured_one():
    w = C.rank_weights(16)
    assert abs(sum(w) - 1) < 1e-9 and abs(w[0] - .526) < 1e-9 and abs(w[1] - .18) < 1e-9 and abs(w[2] - .112) < 1e-9
    assert abs(sum(w[3:8]) - .115) < 1e-9 and abs(sum(w[8:]) - .067) < 1e-9
    assert abs(sum(C.rank_weights(8)) - 1) < 1e-9 and C.rank_weights(8)[0] == .526
    rng = random.Random(1)
    firsts = [C.draw_slots('serving', 1, 16, rng)[0] for _ in range(4000)]
    assert .48 < firsts.count(1) / 4000 < .58
    uniform = [C.draw_slots('uniform', 1, 16, rng)[0] for _ in range(4000)]
    assert .03 < uniform.count(1) / 4000 < .10 and max(uniform) == 16


def test_reposition_moves_sources_to_distinct_slots_and_keeps_the_rest_in_order():
    hits = [hit(f'x{i}', [f'Passage number {i} says something distinct.']) for i in range(10)]
    moved = C.reposition(hits, [3, 7], 'uniform', random.Random(5))
    assert sorted(h.id for h in moved) == sorted(h.id for h in hits)
    others = [h.id for h in moved if h.id not in ('x3', 'x7')]
    assert others == [h.id for h in hits if h.id not in ('x3', 'x7')]
    assert C.reposition(hits, [3, 7], 'natural', random.Random(5)) == hits and C.reposition(hits, [], 'uniform', random.Random(5)) == hits


def test_mode_draw_follows_the_mix():
    mix = C.parse_position_mix('natural,serving,uniform=0.5,0.3,0.2')
    modes = Counter(C.pick_mode(mix, random.Random(common_seed(i))) for i in range(3000))
    assert .45 < modes['natural'] / 3000 < .55 and .25 < modes['serving'] / 3000 < .35 and .15 < modes['uniform'] / 3000 < .25


def common_seed(i):
    return f'seed:{i}'


def test_rows_with_a_position_mix_pass_check_row_with_markers_from_the_final_order(tmp_path):
    gen, _ = generator(tmp_path, mix={'uniform': 1.0})
    slots_seen = set()
    for index in range(40):
        row, reason = finish(gen, index)
        assert reason is None, reason
        assert row['position_mode'] == 'uniform' and B.check_row(row) == []
        slot = row['injected'][0]['slot']
        assert row['passage_ids'][slot - 1] == '1995_p0001'          # injected[].slot is recomputed from the final order (ASM-08)
        slots_seen.add(slot)
    assert len(slots_seen) > 6
    natural, _ = generator(tmp_path / 'n', mix=None)
    row, _ = finish(natural, 3)
    assert row['position_mode'] == 'natural'


def test_position_stream_is_separate_from_the_injection_stream(tmp_path):
    base, _ = generator(tmp_path / 'a', mix=None)
    nat, _ = generator(tmp_path / 'b', mix={'natural': 1.0})
    for index in range(12):
        assert finish(base, index)[0]['messages'] == finish(nat, index)[0]['messages']      # same injection draw, no reposition
    moved, _ = generator(tmp_path / 'c', mix={'uniform': 1.0})
    assert [finish(moved, i)[0]['messages'] for i in range(5)] == [finish(moved, i)[0]['messages'] for i in range(5)]   # deterministic
    assert [r for r in (finish(moved, i)[0] for i in range(12))] != [finish(base, i)[0] for i in range(12)]


def test_overlap_neighbours_may_outrank_the_source_and_the_marker_follows_the_lowest_holder(tmp_path):
    neighbour = hit('1995_p0002', [S[3], S[4]])
    gen, _ = generator(tmp_path, mix={'uniform': 1.0}, extra=[neighbour, hit('1995_p0001', S)])
    for index in range(30):
        row, reason = finish(gen, index)
        assert reason is None and B.check_row(row) == []
        texts = [B.norm(t) for t in [p[2] for _, p in sorted(B.parse_prompt_passages(row['messages'][1]['content']).items())]]
        for line, ob in zip(row['messages'][2]['content'].split('\n'), row['obligations']):
            for sentence in ob['sentences']:
                assert f'{sentence} [{C.lowest_holder(sentence, texts)}]' in line


def test_follow_up_history_cites_the_slot_where_its_sentences_sit_after_the_move(tmp_path):
    gen, full = generator(tmp_path, mix={'uniform': 1.0})
    obj = {'first_question': 'How many shares did Berkshire repurchase in 1995?', 'first_part': {'passage': 1, 'sentences': [1]},
           'question': 'How was it funded, and how large did their float get?',
           'parts': [{'ask': 'funding', 'passage': 1, 'sentences': [2]}, {'ask': 'float', 'passage': 1, 'sentences': [4]}]}
    plan = B.validate_plan(obj, 'follow_up', [full['1995_p0001']])[1]
    for index in range(10):
        job = {'id': f'v7:follow_up:train:{index:04d}', 'kind': 'follow_up', 'split': 'train', 'source_ids': ['1995_p0001']}
        history = [{'role': 'user', 'content': plan['history']['first_question']}, {'role': 'assistant', 'content': f'{S[0]} [1]'}]
        row, reason = gen._finish(job, [full['1995_p0001']], plan, history, {'k': 16})
        assert reason is None, reason
        texts = [B.norm(p[2]) for _, p in sorted(B.parse_prompt_passages(row['messages'][1]['content']).items())]
        slot = C.lowest_holder(S[0], texts)
        assert row['history'][1]['content'] == (f'{S[0]} [{slot}]' if slot else S[0])


# ------------------------------------------------------------------ item 9: exclusions
def chunks():
    return {f'1995_p{i:04d}': SimpleNamespace(id=f'1995_p{i:04d}', text='x', metadata={'year': 1995, 'previous_chunk_id': f'1995_p{i - 1:04d}' if i else None,
                                                                                     'next_chunk_id': f'1995_p{i + 1:04d}' if i < 4 else None}) for i in range(5)}


def test_excluded_ids_get_their_neighbours():
    full = chunks()
    assert C.with_neighbours({'1995_p0002'}, full) == {'1995_p0001', '1995_p0002', '1995_p0003'}
    assert C.with_neighbours({'1995_p0000', '1999_p0001'}, full) == {'1995_p0000', '1995_p0001', '1999_p0001'}     # an id outside the corpus stays
    assert C.with_neighbours(set(), full) == set()


def test_exclude_ids_arguments_expand_to_neighbours(tmp_path):
    full = chunks()
    listing = tmp_path / 'ids.json'
    listing.write_text(json.dumps(['1995_p0003']))
    args = SimpleNamespace(exclude_ids=None, exclude_ids_file=str(listing))
    assert B.extra_exclusions(args, full) == {'1995_p0002', '1995_p0003', '1995_p0004'}
    assert B.extra_exclusions(SimpleNamespace(exclude_ids='1995_p0000, 1995_p0001', exclude_ids_file=None), full) == \
        {'1995_p0000', '1995_p0001', '1995_p0002'}
    legacy = tmp_path / 'ids.txt'
    legacy.write_text('# spent\n1995_p0004\n')
    assert B.extra_exclusions(SimpleNamespace(exclude_ids=str(legacy), exclude_ids_file=None), full) == {'1995_p0003', '1995_p0004'}
    assert B.extra_exclusions(SimpleNamespace(exclude_ids=None, exclude_ids_file=None), full) == set()


def test_cli_accepts_exclude_ids_file_and_the_new_assemble_flags(monkeypatch):
    seen = {}
    monkeypatch.setattr(B, 'assemble_command', lambda args: seen.setdefault('a', args))
    monkeypatch.setattr(B, 'plan_command', lambda args: seen.setdefault('p', args))
    B.main(['assemble', '--jobs', 'j', '--teacher', 't', '--exclude-ids-file', 'ids.json', '--exclude-ids', 'a,b', '--position-mix', 'natural,uniform=0.5,0.5',
            '--allow-drops', '--strict-context', '--passages-mix', '8,12,16,16,16,16'])
    a = seen['a']
    assert (a.exclude_ids_file, a.exclude_ids, a.position_mix, a.allow_drops, a.strict_context) == ('ids.json', 'a,b', 'natural,uniform=0.5,0.5', True, True)
    B.main(['plan', '--n', '60', '--seed', '7', '--out', 'x.jsonl', '--exclude-ids-file', 'ids.json'])
    assert seen['p'].exclude_ids_file == 'ids.json'


def test_strict_context_rejects_rows_whose_context_holds_an_excluded_id(tmp_path):
    spent = hit('1999_p0007', ['A spent evaluation passage that is served as a distractor in the prompt.'], 1999)
    gen, _ = generator(tmp_path / 'strict', extra=[spent], excluded={'1999_p0007'}, strict=True, k=8)
    assert finish(gen)[1] == 'excluded_passage_in_context'
    lax, _ = generator(tmp_path / 'lax', extra=[spent], excluded={'1999_p0007'}, strict=False, k=8)
    row, reason = finish(lax)
    assert reason is None and 'excluded_passage_in_context' in B.check_row(row, excluded={'1999_p0007'}, strict_context=True)
    assert 'excluded_passage_in_context' not in B.check_row(row, excluded={'1999_p0007'})


def test_strict_context_applies_to_refusal_rows_too(tmp_path):
    spent = hit('1999_p0007', ['A spent evaluation passage that is served as a distractor in the prompt.'], 1999)
    gen = B.RecordedGenerator({'r': {'raw': json.dumps({'question': 'Which company did Berkshire buy in 1987 when Fechheimer joined the subsidiaries?', 'missing': ['Fechheimer']})}},
                              FakeServing([spent]), {'1995_p0001': doc('1995_p0001', S)}, {'1999_p0007'}, tmp_path, None, B.MAX_SEQ_MIX, True, [8])
    job = {'id': 'r', 'kind': 'refusal', 'split': 'train', 'source_ids': ['1995_p0001'], 'variant': 'adjacent'}
    assert gen._attempt(job, {'k': 8, 'teacher_calls': 0})[1] == 'excluded_passage_in_context'


# ------------------------------------------------------------------ round5.Serving uses the live per-period question (PW-02)
def test_round5_serving_asks_each_period_with_compares_period_question(monkeypatch):
    ctx = [hit('1995_p0001', S, 1995), hit('2010_p0001', ['Insurance float reached 65 billion dollars at the end of 2010.'], 2010)]
    question = 'How did insurance float in 1995 compare with 2010?'
    prompt = 'prompt'
    monkeypatch.setattr(round5.ask_flow, '_prepare_ask', lambda request: (None, ctx, None, True, ctx, prompt, None))
    monkeypatch.setattr(round5.ask_flow, '_extra_queries', lambda request, expanded: [])
    out = round5.Serving(None, {}).prompts(question)
    periods = comparison_periods(question)
    assert [p['query'] for p in out] == [period_question(question, p, periods) for p in periods]
    assert [p['period'] for p in out] == [period_label(p) for p in periods]
    for p in out:
        assert p['query'] in p['messages'][1]['content']


# ------------------------------------------------------------------ assemble wiring
def tiny_inputs(monkeypatch, tmp_path, kinds=('temporal',)):
    full = {'1995_p0001': doc('1995_p0001', S)}
    jobs, teacher = [], []
    for kind in kinds:
        job = {'id': f'v7:{kind}:train:0001', 'kind': kind, 'split': 'train', 'source_ids': ['1995_p0001'], 'identity': {'corpus_sha256': 'c'}, 'needs_teacher': True}
        jobs.append(job)
        raw = plan_obj() if kind == 'temporal' else {'question': 'Which company did Berkshire buy in 1987 when Fechheimer joined the subsidiaries?', 'missing': ['Fechheimer']}
        if kind == 'refusal':
            job['variant'] = 'adjacent'
        teacher.append({'id': job['id'], 'raw': json.dumps(raw), 'ok': True, 'calls': 1})
    (tmp_path / 'jobs.jsonl').write_text(''.join(json.dumps(j) + '\n' for j in jobs))
    (tmp_path / 'teacher.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in teacher))
    monkeypatch.setattr(B, 'load_inputs', lambda teacher_path: (full, set(), {'with_neighbours': 0}, {1988}, {'corpus_sha256': 'c', 'valid_years': [1988]}))
    monkeypatch.setattr(B.round5, 'load_serving', lambda: FakeServing())
    monkeypatch.setattr(B.round4, 'token_counter', lambda path: (None, lambda messages: 100))
    monkeypatch.setattr(B.common, 'CORPUS', tmp_path / 'corpus.jsonl')
    (tmp_path / 'corpus.jsonl').write_text('')
    return full


def assemble_args(tmp_path, **over):
    args = dict(jobs=str(tmp_path / 'jobs.jsonl'), teacher=str(tmp_path / 'teacher.jsonl'), out_dir=str(tmp_path / 'out'), target=None, exclude_ids=None,
                exclude_ids_file=None, strict_context=False, passages_mix=None, position_mix=None, force=False, min_refusal_share=0., min_partial_share=0.,
                max_seq=None, allow_drops=False)
    args.update(over)
    return SimpleNamespace(**args)


def test_assemble_command_writes_rows_manifest_and_position_histograms(monkeypatch, tmp_path, capsys):
    tiny_inputs(monkeypatch, tmp_path, kinds=('temporal', 'refusal'))
    B.assemble_command(assemble_args(tmp_path, position_mix='uniform=1', passages_mix='8,12,16,16,16,16'))
    manifest = json.loads((tmp_path / 'out/manifest.json').read_text())
    assert manifest['row_counts_per_kind']['temporal'] == 1 and manifest['row_counts_per_kind']['refusal'] == 1
    assert manifest['position_mix'] == {'uniform': 1.0} and manifest['parts_per_row'] == {'3': 1} and manifest['max_seq'] == 12288
    assert manifest['source_rank_histogram'] and manifest['position_modes'] == {'uniform': 1}
    assert (tmp_path / 'out/train.jsonl').read_text().count('\n') == 2


def test_assemble_command_fails_loudly_on_unknown_kinds(monkeypatch, tmp_path):
    tiny_inputs(monkeypatch, tmp_path)
    jobs = [json.loads(line) for line in (tmp_path / 'jobs.jsonl').read_text().splitlines()]
    jobs.append(dict(jobs[0], id='v7:bogus:train:0001', kind='bogus'))
    (tmp_path / 'jobs.jsonl').write_text(''.join(json.dumps(j) + '\n' for j in jobs))
    with pytest.raises(SystemExit, match='unknown job kind'):
        B.assemble_command(assemble_args(tmp_path))
    assert not (tmp_path / 'out/train.jsonl').exists()


def test_assemble_command_fails_on_an_over_length_drop_unless_allowed(monkeypatch, tmp_path, capsys):
    tiny_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(B.round4, 'token_counter', lambda path: (None, lambda messages: 20000))
    with pytest.raises(SystemExit, match='over-length'):
        B.assemble_command(assemble_args(tmp_path))
    assert 'over-length drop v7:temporal:train:0001' in capsys.readouterr().out
    assert not (tmp_path / 'out/train.jsonl').exists()
    B.assemble_command(assemble_args(tmp_path, allow_drops=True))
    assert json.loads((tmp_path / 'out/manifest.json').read_text())['over_length_drops'] == ['v7:temporal:train:0001']


def test_assemble_command_checks_the_refusal_and_partial_shares(monkeypatch, tmp_path):
    tiny_inputs(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match='refusal share'):
        B.assemble_command(assemble_args(tmp_path, min_refusal_share=.15))
    shutil_rm(tmp_path / 'out')
    with pytest.raises(SystemExit, match='partial share'):
        B.assemble_command(assemble_args(tmp_path, min_partial_share=.03))


def shutil_rm(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def test_check_refusal_share_counts_refusal_near_and_partial_share_is_reported():
    rows = [{'kind': 'refusal'}, {'kind': 'refusal_near'}] + [{'kind': 'single'}] * 8
    B.check_refusal_share(rows, .15)
    with pytest.raises(SystemExit):
        B.check_refusal_share(rows, .25)
    B.check_partial_share([{'kind': 'partial'}] + [{'kind': 'single'}] * 9, .03)
    with pytest.raises(SystemExit):
        B.check_partial_share([{'kind': 'single'}] * 9, .03)
