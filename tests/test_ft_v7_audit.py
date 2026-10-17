"""ft_v7 offline audit: counts per rule over a stored row set, and the train/valid + metadata join."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_env = dict(os.environ)
sys.path.insert(0, str(ROOT / 'scripts' / 'ft'))
import build_ft_v6 as B  # noqa: E402
import ft_v7_audit as A  # noqa: E402
os.environ.clear()
os.environ.update(_env)

from src.storage import SearchHit  # noqa: E402


@pytest.fixture(autouse=True)
def training_prompt(monkeypatch):
    monkeypatch.setenv('PROMPT_VERSION', B.PROMPT_VERSION)


S = ['Berkshire repurchased 120,000 shares of its common stock during 1995.', 'The buyback was funded from retained earnings of the insurance operations.',
     'Insurance float reached 3.4 billion dollars at the end of 1995.', 'GEICO added 400,000 policyholders and its combined ratio improved to 96.']


def row(split='train', sentences=None, question='How did Berkshire pay for repurchases and what changed at GEICO in 1995?'):
    sentences = sentences or [S[0], S[2], S[3]]
    hits = [SearchHit(id='1990_p0001', text='Unrelated words fill this other passage for the test.', metadata={'year': 1990, 'source_file': 'a.txt'}, score=1.),
            SearchHit(id='1995_p0001', text=' '.join(S), metadata={'year': 1995, 'source_file': 'b.txt'}, score=1.)]
    target = '\n'.join(f'{s} [2]' for s in sentences)
    messages = B.chat_messages(question, hits) + [{'role': 'assistant', 'content': target}]
    return {'id': f'v7:x:{split}', 'kind': 'temporal', 'split': split, 'type': 'answerable', 'source_ids': ['1995_p0001'], 'source_year': 1995,
            'served_ids': ['1990_p0001', '1995_p0001'], 'question': question, 'prompt_version': B.PROMPT_VERSION,
            'parts': [{'year': 1995}] * len(sentences), 'messages': messages,
            'obligations': [{'ask': 'x', 'source_id': '1995_p0001', 'sentence_numbers': [i], 'sentences': [s]} for i, s in enumerate(sentences, 1)]}


def test_audit_counts_each_rule_and_keeps_clean_rows():
    docs = {'1995_p0001': SimpleNamespace(id='1995_p0001', text=' '.join(S))}
    clean = row()
    short = row(sentences=['Float reached 3.4 billion.', S[0], S[3]])
    dependent = row(sentences=['Therefore the buyback was a good use of our retained cash.', S[0], S[3]])
    result = A.audit([clean, short, dependent], docs, set(), {1988})
    assert result['rows'] == 3 and result['by_rule'].get('obligation_too_short', {}).get('rows') == 1
    assert result['by_rule']['dependent_start']['rows'] == 1 and 'dependent_start_old_list' not in result['by_rule']      # only the extended list catches 'Therefore'
    assert result['kept_without_the_overlap_rule'] == 1
    assert result['question_overlap']['answerable_rows'] == 3


def test_audit_flags_a_train_row_that_quotes_a_valid_year():
    docs = {'1995_p0001': SimpleNamespace(id='1995_p0001', text=' '.join(S))}
    result = A.audit([row(split='train')], docs, set(), {1995})
    assert result['by_rule']['split_straddle']['rows'] == 1


def test_load_rows_joins_metadata_and_training_rows_in_order(tmp_path):
    rows = [row('train'), dict(row('valid'), id='v7:x:valid')]
    meta = [{k: v for k, v in r.items() if k != 'messages'} for r in rows]
    (tmp_path / 'split_metadata.jsonl').write_text(''.join(json.dumps(m) + '\n' for m in meta))
    for split in ('train', 'valid'):
        (tmp_path / f'{split}.jsonl').write_text(''.join(json.dumps({'messages': r['messages']}) + '\n' for r in rows if r['split'] == split))
    loaded = A.load_rows(tmp_path)
    assert [r['id'] for r in loaded] == ['v7:x:train', 'v7:x:valid'] and loaded[0]['messages'] == rows[0]['messages']
    (tmp_path / 'train.jsonl').write_text(json.dumps({'messages': rows[1]['messages'][:2] + [{'role': 'assistant', 'content': 'Something else.'}]}) + '\n')
    with pytest.raises(AssertionError, match='not the same row'):
        A.load_rows(tmp_path)
