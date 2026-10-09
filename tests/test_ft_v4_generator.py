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


# ------------------------------------------------------------------ v4 question writing (fake teacher, no model)
class ScriptedTeacher:
    def __init__(self, *outputs, finish='stop'):
        self.outputs, self.calls, self.finish = list(outputs), [], finish

    def generate(self, messages, key, max_tokens=350, temperature=.2):
        self.calls.append((key, [dict(m) for m in messages]))
        return self.outputs.pop(0), {'tokens': 10, 'seconds': 1., 'generation_tps': 1., 'finish_reason': self.finish}


def _lookup(*texts):
    docs = {}
    for index, (year, text) in enumerate(texts):
        ident = f'{year}_p{index:04d}'
        docs[ident] = SimpleNamespace(id=ident, text=text, metadata={'year': year})
    return docs


FOLLOWUP_DOC = ('Berkshire sold its textile operations in 1985 after years of losses. Management concluded that the capital was better used '
                'elsewhere in the insurance business. The decision was difficult but clearly correct for owners.')
TEMPORAL_DOCS = [(1990, 'Junk bonds were sold to investors who ignored the credit risk of the issuers. Many of them later defaulted badly during the recession.'),
                 (2000, 'Junk bonds became a smaller part of the market after the defaults of the early period. Buyers demanded much higher yields than before.')]


def _job(kind, lookup, **extra):
    ids = list(lookup)
    base = {'id': f'v4:{kind}:0000', 'kind': kind, 'source_id': ids[0], 'split': 'train', 'type': 'answerable'}
    if kind == 'temporal':
        base.update(topic='junk bonds', source_ids=ids)
    return {**base, **extra}


def _json(**obj):
    import json
    return json.dumps(obj)


GOOD_FOLLOWUP = _json(question='Why did Berkshire decide to sell it after the losses?', topic='the textile operations',
                      parts=[{'ask': 'why it was sold', 'evidence_id': 'E1'}])
STANDALONE_FOLLOWUP = _json(question='Which year did Berkshire sell its textile operations?', topic='the textile operations',
                            parts=[{'ask': 'the year of the sale', 'evidence_id': 'E1'}])
GOOD_TEMPORAL = _json(question='How did the 1990 and 2000 letters describe junk bonds?', topic='',
                      parts=[{'ask': 'junk bonds in 1990', 'evidence_id': 'E1'}, {'ask': 'junk bonds in 2000', 'evidence_id': 'E2'}])
ONE_YEAR_TEMPORAL = _json(question='What does the 1990 letter say about junk bonds?', topic='',
                          parts=[{'ask': 'junk bonds in 1990', 'evidence_id': 'E1'}, {'ask': 'junk bonds in 2000', 'evidence_id': 'E2'}])


def test_followup_prompt_has_worked_example_and_good_first_answer_needs_no_retry():
    lookup = _lookup((1985, FOLLOWUP_DOC))
    teacher = ScriptedTeacher(GOOD_FOLLOWUP)
    row, record = round4.synthesize(teacher, _job('followup', lookup), lookup, {}, set())
    assert row and record['kept'] == 1 and len(teacher.calls) == 1
    prompt = teacher.calls[0][1][1]['content']
    assert 'EXAMPLE answer' in prompt and 'it, they, that, this or their' in prompt and 'must NOT name the topic' in prompt
    assert row['style'] == 'v4' and row['origin'] == 'v4_new' and row['history'][0]['content'].startswith("Let's discuss the textile operations")


def test_followup_without_pronoun_gets_one_retry_naming_the_rule_and_is_kept_when_fixed():
    lookup = _lookup((1985, FOLLOWUP_DOC))
    teacher = ScriptedTeacher(STANDALONE_FOLLOWUP, GOOD_FOLLOWUP)
    row, record = round4.synthesize(teacher, _job('followup', lookup), lookup, {}, set())
    assert row and record['retry'] and record['first_reason'] == 'invalid_followup_topic_or_pronoun'
    assert [key for key, _ in teacher.calls] == ['v4:followup:0000:question', 'v4:followup:0000:question:retry']
    retry = teacher.calls[1][1]
    assert retry[2] == {'role': 'assistant', 'content': STANDALONE_FOLLOWUP}
    assert 'invalid_followup_topic_or_pronoun' in retry[3]['content'] and 'it, they, that, this or their' in retry[3]['content']
    assert record['seconds'] == 2.


def test_retry_is_capped_at_one_and_checks_stay_strict():
    lookup = _lookup((1985, FOLLOWUP_DOC))
    teacher = ScriptedTeacher(STANDALONE_FOLLOWUP, STANDALONE_FOLLOWUP, GOOD_FOLLOWUP)
    row, record = round4.synthesize(teacher, _job('followup', lookup), lookup, {}, set())
    assert row is None and record['reason'] == 'invalid_followup_topic_or_pronoun' and len(teacher.calls) == 2


def test_temporal_needs_both_years_and_retry_names_them():
    lookup = _lookup(*TEMPORAL_DOCS)
    teacher = ScriptedTeacher(ONE_YEAR_TEMPORAL, GOOD_TEMPORAL)
    row, record = round4.synthesize(teacher, _job('temporal', lookup), lookup, {}, set())
    assert row and record['first_reason'] == 'source_year_missing'
    assert '1990 and 2000' in teacher.calls[1][1][3]['content']
    assert '1990 and 2000 letters' in teacher.calls[0][1][1]['content'] or '1991 and 2003' in teacher.calls[0][1][1]['content']
    assert round4.round3.source_evidence(_job('temporal', lookup), lookup), 'fixture must give two clean topic sentences'
    teacher = ScriptedTeacher(ONE_YEAR_TEMPORAL, ONE_YEAR_TEMPORAL)
    row, record = round4.synthesize(teacher, _job('temporal', lookup), lookup, {}, set())
    assert row is None and record['reason'] == 'source_year_missing' and len(teacher.calls) == 2


def test_non_rule_failures_are_not_retried_and_other_kinds_keep_the_round3_prompt():
    lookup = _lookup((1985, FOLLOWUP_DOC))
    first, _ = round4.synthesize(ScriptedTeacher(GOOD_FOLLOWUP), _job('followup', lookup), lookup, {}, set())
    teacher = ScriptedTeacher(GOOD_FOLLOWUP)
    row, _ = round4.synthesize(teacher, _job('followup', lookup), lookup, {}, {round4.round3.fingerprint(first)})
    assert row is None and len(teacher.calls) == 1, 'duplicate question is not a rule the teacher can fix'
    teacher = ScriptedTeacher(_json(question='What did Berkshire conclude about the capital in 1985?', parts=[{'ask': 'conclusion', 'evidence_id': 'E1'}]))
    row, _ = round4.synthesize(teacher, _job('single', lookup), lookup, {}, set())
    assert row and len(teacher.calls) == 1 and 'EXAMPLE' not in teacher.calls[0][1][1]['content']


def test_budget_stop_is_not_retried():
    lookup = _lookup((1985, FOLLOWUP_DOC))
    teacher = ScriptedTeacher('', finish='budget')
    row, record = round4.synthesize(teacher, _job('followup', lookup), lookup, {}, set())
    assert row is None and record['finish_reason'] == 'budget' and len(teacher.calls) == 1


def test_temporal_plan_only_picks_pairs_whose_both_years_have_a_clean_topic_sentence(corpus):
    _, lookup = corpus
    docs = list(lookup.values())
    quotas = {'single': 0, 'multipart': 0, 'followup': 0, 'temporal': 12, 'refusal': 0}
    plan = round4.build_plan(docs, {1999}, {'frozen_set_hashes': {}}, quotas, 1.)
    jobs = [j for j in plan['jobs'] if j['kind'] == 'temporal']
    assert len(jobs) >= 6
    for job in jobs:
        evidence = round4.round3.source_evidence(job, lookup)
        assert len(evidence) == 2 and evidence[0]['year'] != evidence[1]['year'], job
        assert all(job['topic'] in e['sentence'].lower() for e in evidence)
