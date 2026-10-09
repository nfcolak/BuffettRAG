"""FT-r2 v4 generator: serving-identical prompts, completeness filter, heldout_v4 refusal."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_env = dict(os.environ)   # scripts/ft/common.py sets offline env flags on import; keep them out of other tests
sys.path.insert(0, str(ROOT / 'scripts' / 'ft'))
import common  # noqa: E402
import round4  # noqa: E402
os.environ.clear()
os.environ.update(_env)

from src.generation.compare import generate_comparison_answer  # noqa: E402
from src.generation.prompt import REFUSAL_LINE  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.services import ask_flow  # noqa: E402
from src.services.schemas import AskRequest  # noqa: E402
from src.storage import SearchHit  # noqa: E402


@pytest.fixture(scope='module')
def corpus():
    docs = [d for d in common.load_chunks_as_docs(common.CORPUS) if 1985 <= int(d.metadata['year']) <= 1999]
    return BM25Retriever(docs), {d.id: d for d in docs}


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, **kwargs):
        return SimpleNamespace(hits=self.hits, used_filter=None, reranked=False)


def serving_context(monkeypatch, hits, lookup, query):
    monkeypatch.setattr(ask_flow, '_state', {'llm': round4.LlamaStub(), 'retriever': FakeRetriever(hits), 'docs_by_id': lookup})
    return ask_flow._prepare_ask(AskRequest(query=query, expand_query=False))


def joined(prompt):
    system, user = prompt['messages']
    return system['content'] + '\n\n' + user['content']


def test_single_prompt_is_what_ask_sends_the_7b(corpus, monkeypatch):
    bm25, lookup = corpus
    query = 'In the 1995 letter, what does Berkshire say about insurance float?'
    anchors = round4.retrieve(bm25, query)
    (prompt,) = round4.serving_prompts(query, anchors, lookup)
    _, _, _, _, context, served, _ = serving_context(monkeypatch, anchors, lookup, query)
    assert joined(prompt) == served
    assert [h.id for h in prompt['hits']] == [h.id for h in context]
    assert len(prompt['hits']) == round4.SERVING_PASSAGES == 8
    assert all(len(h.text) <= 1800 for h in prompt['hits'])


def test_temporal_job_yields_one_prompt_per_period_like_generate_comparison_answer(corpus, monkeypatch):
    bm25, lookup = corpus
    query = 'How did the 1990 and 1995 letters describe insurance float?'
    anchors = round4.retrieve(bm25, query)
    prompts = round4.serving_prompts(query, anchors, lookup)
    *_, context, _, _ = serving_context(monkeypatch, anchors, lookup, query)
    sent = []

    class Capture(round4.LlamaStub):
        def generate(self, prompt, max_new_tokens):
            sent.append(prompt)
            return 'Float is discussed. [1]'

    generate_comparison_answer(Capture(), query, context, history=[], max_new_tokens=900)
    assert [p['period'] for p in prompts] == ['1990', '1995']
    assert all(p['gate'] for p in prompts), 'pick a query whose periods both pass the evidence gate'
    assert sent == [joined(p) for p in prompts]
    assert all(f'{query} (focus: {p["period"]})' in joined(p) for p in prompts)
    for prompt in prompts:   # only that period's passages
        assert {h.metadata['year'] for h in prompt['hits']} == {int(prompt['period'])}


def _hits():
    return [SearchHit('p1', 'Berkshire repurchased 120,000 shares during 1995. The buyback was funded from retained earnings.', {'year': 1995}, 0.),
            SearchHit('p2', 'Insurance float reached $3.1 billion at yearend. Float is money held for later claims.', {'year': 1995}, 0.)]


EVIDENCE = [{'id': 'E1', 'source_id': 'p1', 'sentence': 'Berkshire repurchased 120,000 shares during 1995.'},
            {'id': 'E2', 'source_id': 'p2', 'sentence': 'Insurance float reached $3.1 billion at yearend.'}]
QUESTION = 'In 1995, how many shares did Berkshire repurchase and how large was insurance float?'


def test_completeness_filter_rejects_a_dropped_part():
    both = 'Berkshire repurchased 120,000 shares during 1995. [1]\nInsurance float reached $3.1 billion at yearend. [2]'
    assert round4.completeness_reason(both, _hits(), EVIDENCE, question_text=QUESTION) is None
    dropped = 'Berkshire repurchased 120,000 shares during 1995. [1]'
    assert round4.completeness_reason(dropped, _hits(), EVIDENCE, question_text=QUESTION) is not None
    assert round4.completeness_reason(REFUSAL_LINE, _hits(), EVIDENCE, question_text=QUESTION) == 'answerable_refusal'


def test_completeness_filter_accepts_explicit_do_not_state_sentence():
    missing = [{'id': 'M1', 'ask': 'the exact price paid per acre'}]
    question = QUESTION + ' What was the exact price paid per acre?'
    stated = 'Berkshire repurchased 120,000 shares during 1995. [1]\nThe passages do not state the exact price paid per acre.'
    assert round4.completeness_reason(stated, _hits(), EVIDENCE[:1], missing, question) is None
    silent = 'Berkshire repurchased 120,000 shares during 1995. [1]'
    assert round4.completeness_reason(silent, _hits(), EVIDENCE[:1], missing, question) == 'missing_part_not_named'
    cited = 'Berkshire repurchased 120,000 shares during 1995. [1]\nThe passages do not state the exact price paid per acre. [2]'
    assert round4.completeness_reason(cited, _hits(), EVIDENCE[:1], missing, question) is not None
    invented = 'Berkshire repurchased 130,000 shares during 1995. [1]'
    assert round4.completeness_reason(invented, _hits(), EVIDENCE[:1], question_text=QUESTION) is not None


def test_missing_heldout_v4_checker_refuses_to_run(tmp_path, monkeypatch):
    absent = tmp_path / 'check_heldout_v4.py'
    with pytest.raises(RuntimeError, match='refusing'):
        round4.heldout_v4_excluded_ids(absent)
    monkeypatch.setattr(round4, 'HELDOUT_V4_CHECKER', absent)
    monkeypatch.setattr(round4, 'PILOT_DIR', 'models/ft/_test_round4_refusal')
    with pytest.raises(RuntimeError, match='refusing'):
        round4.main(['--pilot', '1'])
    assert not (ROOT / 'models/ft/_test_round4_refusal').exists()


def test_heldout_v4_checker_output_is_validated(tmp_path):
    checker = tmp_path / 'check.py'
    checker.write_text('import json\nprint(json.dumps(["1990_p0001", "1990_p0000", "1990_p0001"]))\n')
    assert round4.heldout_v4_excluded_ids(checker) == ['1990_p0000', '1990_p0001']
    checker.write_text('print("not json")\n')
    with pytest.raises(RuntimeError, match='refusing'):
        round4.heldout_v4_excluded_ids(checker)
