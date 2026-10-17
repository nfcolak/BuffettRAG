"""ft_v7 data builder, items 1, 2, 6 (citation rule), 7, 8: complete aligned sentences, serving replay, lengths, question style."""
import os
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


def make_row(target=None, passages=None, kind='multipart', obligations=None):
    source = doc('1995_p0001', S)
    hits = passages or [hit('1990_p0001', ['Unrelated words fill this other passage for the test.']), hit('1995_p0001', S)]
    plan = B.validate_plan(plan_obj(), 'multipart', [source])[1]
    target = target or B.build_target(plan['obligations'], hits)[0]
    messages = B.chat_messages(plan['question'], hits) + [{'role': 'assistant', 'content': target}]
    obligations = obligations or [{k: o[k] for k in ('ask', 'source_id', 'sentence_numbers', 'sentences')} for o in plan['obligations']]
    return {'id': 'v7:test', 'kind': kind, 'type': 'answerable', 'source_ids': ['1995_p0001'], 'served_ids': ['1990_p0001', '1995_p0001'],
            'obligations': obligations, 'messages': messages}


# ------------------------------------------------------------------ item 1: complete, standalone sentences (SP-01, TARGET-02, TARGET-03)
@pytest.mark.parametrize('sentence', [
    'Again, the answer is no.', 'Clearly, the party is over.', 'Therefore, the buyback was cheap for owners.',
    'Consequently, float grew faster than book value.', 'Thus the insurance business kept its edge over rivals.',
    'Hence we retained the earnings of the subsidiary.', 'In addition, GEICO grew its policyholder base.',
    'As a result, the combined ratio improved a lot.', 'Indeed, nobody expected such a quick recovery.', 'Otherwise we would sell the stock.'])
def test_dependent_start_lists_the_connectives_the_audit_found(sentence):
    assert B.DEPENDENT_START.match(sentence)


def test_dependent_start_keeps_standalone_openers_out():
    for sentence in ('Berkshire repurchased shares.', 'Insurance float reached 3.4 billion dollars.', 'Againstthe odds we grew.', 'Thusly named.'):
        assert not B.DEPENDENT_START.match(sentence)


def test_connective_first_sentence_is_rejected_when_nothing_precedes_it():
    sentences = ['Therefore, the buyback was funded from retained earnings of the insurance operations.'] + S[1:]
    d = doc('1995_p0001', sentences)
    obj = plan_obj(parts=[{'ask': 'a', 'passage': 1, 'sentences': [1]}, {'ask': 'b', 'passage': 1, 'sentences': [4]}, {'ask': 'c', 'passage': 1, 'sentences': [5]}])
    assert B.validate_plan(obj, 'multipart', [d])[0] == 'dependent_start'


def test_fragment_without_terminal_punctuation_is_bad():
    assert B.bad_sentence('The U.S. - and much of the world - became trapped in a vicious negative-feedback')
    assert B.bad_sentence('Our mortgage portfolio loans, which')
    assert not B.bad_sentence('He said "we will do it."') and not B.bad_sentence('Float grew (a lot).')
    assert not B.bad_sentence(S[0])


def test_obligation_must_have_eight_words():
    short = ['Berkshire repurchased shares quickly.', 'Float reached 3.4 billion dollars at year end.']   # 4 words
    d = doc('1995_p0001', short + S[1:])
    obj = plan_obj(parts=[{'ask': 'a', 'passage': 1, 'sentences': [1]}, {'ask': 'b', 'passage': 1, 'sentences': [2]}, {'ask': 'c', 'passage': 1, 'sentences': [4]}])
    reason, _ = B.validate_plan(obj, 'multipart', [d])
    assert reason == 'sentence_too_short' or reason == 'obligation_too_short'
    five = ['Float reached 3.4 billion dollars.'] + S[1:]            # five words, passes the per-sentence minimum but not the 8 words per obligation
    d = doc('1995_p0001', five)
    obj = plan_obj(parts=[{'ask': 'a', 'passage': 1, 'sentences': [1]}, {'ask': 'b', 'passage': 1, 'sentences': [2]}, {'ask': 'c', 'passage': 1, 'sentences': [4]}])
    assert B.validate_plan(obj, 'multipart', [d])[0] == 'obligation_too_short'


def test_check_row_requires_the_target_sentence_to_be_a_whole_sentence_of_the_cited_passage():
    served = [hit('1990_p0001', ['Unrelated words fill this other passage for the test.']),
              hit('1995_p0001', ['Management said that ' + S[0]] + S[1:])]
    plan = B.validate_plan(plan_obj(), 'multipart', [doc('1995_p0001', S)])[1]
    target = B.build_target(plan['obligations'], [hit('1990_p0001', ['x']), hit('1995_p0001', S)])[0]
    row = make_row(target=target, passages=served)    # in the prompt S[0] is only a substring of the first served sentence now
    problems = B.check_row(row)
    assert 'target_not_sentence_aligned' in problems and 'sentence_not_in_cited_passage' not in problems
    assert B.check_row(make_row()) == []


def test_place_sources_injects_the_source_when_the_served_sentence_is_a_fragment_of_a_longer_one():
    source = doc('1995_p0001', S)
    served = [hit('1990_p0001', ['Management said that ' + S[0]] + S[1:])] + \
        [hit(f'1991_p{i:04d}', [f'Filler passage number {i} talks about nothing relevant at all here.']) for i in range(7)]
    plan = B.validate_plan(plan_obj(), 'multipart', [source])[1]
    import random
    placed, injected, error = B.place_sources(served, plan['obligations'], {'1995_p0001': source}, random.Random(1))
    assert error is None and [i['passage_id'] for i in injected] == ['1995_p0001']


def test_check_row_rejects_a_connective_opener_and_a_short_obligation():
    row = make_row()
    row['obligations'][1]['sentences'] = ['Therefore ' + S[3][0].lower() + S[3][1:]]
    assert 'dependent_start' in B.check_row(row)
    short = make_row()
    short['obligations'][2] = dict(short['obligations'][2], sentences=['GEICO added 400,000 policies.'], sentence_numbers=[5])
    assert 'obligation_too_short' in B.check_row(short)


# ------------------------------------------------------------------ item 2: every target survives the serving validator (B14)
def test_check_row_replays_the_target_through_the_serving_validator(monkeypatch):
    row = make_row()
    assert B.check_row(row) == []
    real = C.validate_and_filter_answer

    def drops_a_sentence(answer, hits, **kw):
        result = real(answer, hits, **kw)
        result.safe_answer = '\n'.join(answer.split('\n')[:-1])
        return result
    monkeypatch.setattr(C, 'validate_and_filter_answer', drops_a_sentence)
    assert 'serving_validator_drops_sentence' in B.check_row(row)


def test_check_row_flags_a_sentence_answer_expand_cannot_locate(monkeypatch):
    monkeypatch.setattr(C, 'locatable', lambda sentence, text: False)
    assert 'sentence_not_locatable' in B.check_row(make_row())


# ------------------------------------------------------------------ item 6: the citation rule is explicit (TARGET-06, TPC-01, TPC-03)
def overlap_passages():
    first = hit('1995_p0002', [S[3], S[4]])                         # a neighbour hit that overlaps the source and ranks first
    second = hit('1995_p0001', S)
    return [first, second]


def test_citation_is_the_lowest_numbered_passage_holding_the_whole_sentence():
    source = doc('1995_p0001', S)
    plan = B.validate_plan(plan_obj(), 'multipart', [source])[1]
    target, slots = B.build_target(plan['obligations'], overlap_passages())
    lines = target.split('\n')
    assert lines[0].endswith('[2]') and lines[1] == f'{S[3]} [1]' and lines[2] == f'{S[4]} [1]' and slots == [[2, 2], [1], [1]]
    row = make_row(passages=overlap_passages())
    assert B.check_row(row) == []


def test_check_row_rejects_a_marker_that_is_not_the_lowest_holder():
    row = make_row(passages=overlap_passages())
    wrong = row['messages'][2]['content'].replace(f'{S[3]} [1]', f'{S[3]} [2]')
    assert 'citation_not_lowest_holder' in B.check_row(dict(row, messages=row['messages'][:2] + [{'role': 'assistant', 'content': wrong}]))


# ------------------------------------------------------------------ item 7: lengths
def test_default_max_seq_with_a_mix_is_12288_and_mixed_weights_follow_repetition():
    assert B.MAX_SEQ_MIX == 12288
    mix = B.parse_passages_mix('8,12,16,16,16,16')
    assert mix == [8, 12, 16, 16, 16, 16]
    ids = [f'v7:multipart:train:{i:04d}' for i in range(1200)]
    counts = {k: sum(B.pick_k(i, mix) == k for i in ids) for k in (8, 12, 16)}
    assert 140 <= counts[8] <= 260 and 140 <= counts[12] <= 260 and 720 <= counts[16] <= 880


def test_over_length_drops_fail_unless_allowed(capsys):
    attempts = [{'id': 'a', 'kind': 'multipart', 'k': 16, 'reason': 'too_long_for_max_seq'}, {'id': 'b', 'kind': 'single', 'k': 8, 'reason': None}]
    with pytest.raises(SystemExit, match='over-length'):
        B.check_over_length(attempts, allow=False)
    assert 'a' in capsys.readouterr().out
    assert B.check_over_length(attempts, allow=True) == ['a']
    assert B.check_over_length(attempts[1:], allow=False) == []


# ------------------------------------------------------------------ item 8: question style (DS-06, PD-05)
def test_question_that_mirrors_its_passage_is_rejected_and_a_paraphrase_is_not():
    d = doc('1995_p0001', S)
    copy = 'How big was insurance float, how many policyholders did GEICO add, and how was repurchase funded from retained earnings, given insurance operations?'
    assert B.check_question(copy, 'multipart', [d]) == 'question_overlaps_passage'
    natural = 'How did Buffett pay for buying back stock in 1995, and what happened to insurance customers at GEICO?'
    assert C.question_overlap(natural, [d.text]) <= B.QUESTION_OVERLAP_MAX
    assert B.check_question(natural, 'multipart', [d]) is None


def test_teacher_prompts_ask_for_natural_reader_wording():
    d = doc('1995_p0001', S)
    prompt = B.teacher_prompt({'id': 'x', 'kind': 'multipart', 'split': 'train', 'source_ids': ['1995_p0001'], 'parts_range': [4, 4]}, [d])
    assert 'your own words' in prompt and 'natural' in prompt
