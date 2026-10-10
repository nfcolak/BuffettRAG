"""ft_v5 generator: serving-identical prompts, passage-based questions, serving-validator answer filter, exclusion union (fake teacher, no MLX)."""
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_env = dict(os.environ)   # scripts/ft/common.py sets offline env flags on import; keep them out of other tests
sys.path.insert(0, str(ROOT / 'scripts' / 'ft'))
import common  # noqa: E402
import round4  # noqa: E402
import round5  # noqa: E402
os.environ.clear()
os.environ.update(_env)

from src.generation.prompt import REFUSAL_LINE, SYSTEM_PROMPT, build_cited_prompt  # noqa: E402
from src.generation.providers.llama_provider import split_prompt  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402
from src.services import ask_flow  # noqa: E402
from src.services.schemas import AskRequest  # noqa: E402
from src.storage import SearchHit, StoredDoc  # noqa: E402


class BM25OnlyRetriever(Retriever):
    """Production fusion, year handling and period split; BM25 is the only candidate source (no model loads)."""

    def __init__(self, docs):
        self.bm25, self.reranker = BM25Retriever(docs), None

    def _hybrid_for_filter(self, query, fetch_k, where):
        return self.bm25.search(query, top_k=fetch_k, where=where)


# ------------------------------------------------------------------ tiny corpus
SOURCE = ('Berkshire repurchased 120,000 shares of its common stock during 1995. The buyback was funded from retained earnings '
          'rather than new borrowing. Management considers repurchases attractive only when the price is well below intrinsic value.')
OTHERS = [
    'Insurance float reached $3.1 billion at yearend 1995. Float is money held for later claims, and it costs less than other funding. '
    'Underwriting discipline matters more than premium growth in this business.',
    'The textile operation in New England kept losing money through 1996 despite repeated capital outlays. Management finally closed the mills.',
    'Our newspaper earnings declined in 1996 as circulation softened and newsprint prices rose sharply across the industry.',
    'The candy company set a sales record during the holiday season of 1996, helped by new retail stores in the western states.',
    'We prefer buying whole businesses run by managers we admire, and we rarely sell a company that we have acquired in the past.',
    'Annual meetings were crowded this spring as shareholders arrived from many countries to ask questions for several hours.',
    'The furniture retailer in Nebraska expanded its warehouse and added a second showroom to serve customers across the plains.',
    'Reinsurance premiums fell as competitors cut prices, so we wrote far less of that business during 1996 than a year earlier.',
    'Our jewelry subsidiary reported higher profits because of strong demand in the fourth quarter and tight inventory control.',
    'The shoe manufacturer faced intense foreign competition and cut its workforce while seeking cheaper suppliers abroad.',
    'Our investment in a large soft drink company has grown steadily, and we expect to hold the shares indefinitely.',
]
QUESTION = 'In 1995, how large was the buyback of its own common stock?'


def corpus():
    docs = [StoredDoc(id='1995_p0001', text=SOURCE, metadata={'year': 1995, 'source_file': 'letter_1995.pdf'})]
    docs += [StoredDoc(id=f'{1995 + (i > 0)}_p{i + 2:04d}', text=text, metadata={'year': 1995 + (i > 0), 'source_file': 'letter.pdf'})
             for i, text in enumerate(OTHERS)]
    return docs


class ScriptedTeacher:
    """Writes the question from a script and answers from the served prompt only (it never sees anything else)."""

    def __init__(self, question_json, answer=None):
        self.question_json, self.answer, self.calls = question_json, answer, []

    def generate(self, messages, key, max_tokens=350, temperature=.2):
        self.calls.append((key, [dict(m) for m in messages]))
        metrics = {'tokens': 10, 'seconds': 1., 'generation_tps': 1., 'finish_reason': 'stop'}
        if ':question' in key or ':q2' in key:
            return self.question_json, metrics
        if self.answer is not None:
            return self.answer, metrics
        user = messages[1]['content']
        number = next(int(n) for n, block in re.findall(r'\[(\d+)\] \(year=\d+, source=[^)]*\)\n([^\[]*)', user) if '120,000 shares' in block)
        return f'Berkshire repurchased 120,000 shares of its common stock during 1995. [{number}]', metrics


def question_json(question=QUESTION, key_terms=('shares',)):
    return json.dumps({'question': question, 'parts': [{'ask': 'size of the buyback', 'key_terms': list(key_terms)}]})


def make_generator(tmp_path, teacher, quotas=None):
    docs = corpus()
    full = {d.id: d for d in docs}
    loaded = (docs, set(), [], {2001}, {'style': 'v5'}, full)
    serving = round5.Serving(BM25OnlyRetriever(docs), full)
    return round5.Generator(teacher, serving, loaded, quotas or {k: 1 for k in round5.KINDS}, tmp_path / 'out'), full


def single_job():
    return {'id': 'v5:single:0000', 'kind': 'single', 'split': 'train', 'type': 'answerable', 'source_id': '1995_p0001'}


# ------------------------------------------------------------------ kept row = serving prompt, byte for byte
def test_kept_row_prompt_is_the_serving_split_prompt_byte_for_byte(tmp_path, monkeypatch):
    teacher = ScriptedTeacher(question_json())
    generator, full = make_generator(tmp_path, teacher)
    question, record = generator.synthesize(single_job())
    assert question and record['kept'], record
    assert generator.process(question) == 1
    (row,) = generator.rows
    assert row['natural_miss'] is False and row['source_in_top8'] is True

    monkeypatch.setattr(ask_flow, '_state', {'llm': round4.LlamaStub(), 'retriever': BM25OnlyRetriever(corpus()), 'docs_by_id': full})
    *_, context, prompt, _ = ask_flow._prepare_ask(AskRequest(query=question['question'], expand_query=False))
    system, user = split_prompt(prompt)
    assert system == SYSTEM_PROMPT and user.endswith('Answer:')
    assert row['messages'][:2] == [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    assert row['messages'][:2] == round5.messages_from(build_cited_prompt(question['question'], context))
    assert row['messages'][2]['role'] == 'assistant' and '120,000 shares' in row['messages'][2]['content']

    # the teacher saw the serving messages plus the style suffix, nothing else (no gold sentence, no checklist)
    answer_calls = [messages for key, messages in teacher.calls if ':question' not in key]
    assert len(answer_calls) == 1
    seen_system, seen_user = answer_calls[0]
    assert seen_system['content'] == system + round5.STYLE_V5 and seen_user['content'] == user
    assert 'CHECKLIST' not in seen_user['content'] and 'Direct supporting sentence' not in seen_user['content']


# ------------------------------------------------------------------ question copying passage text is rejected
def test_question_sharing_a_4gram_with_the_passage_is_rejected(tmp_path):
    copied = 'In 1995, how many shares of its common stock were repurchased?'     # "shares of its common stock" is in the passage
    generator, _ = make_generator(tmp_path, ScriptedTeacher(question_json(copied)))
    question, record = generator.synthesize(single_job())
    assert question is None and record['reason'] == 'question_copies_passage' and record['retry'] is True
    assert round5.copy_reason('What does the passage say about the buyback?', [SOURCE]) == 'question_refers_to_passage'
    assert round5.copy_reason('How large was the buyback of its own common stock in 1995?', [SOURCE]) is None


# ------------------------------------------------------------------ answer the serving validator mostly drops is rejected
def test_answer_dropped_by_the_serving_validator_is_rejected(tmp_path):
    hits = [SearchHit(id='1995_p0001', text=SOURCE, metadata={'year': 1995}, score=0.)]
    part = {'id': 'P1', 'ask': 'size of the buyback', 'key_terms': ['shares'], 'source_id': '1995_p0001', 'year': 1995}
    good = 'Berkshire repurchased 120,000 shares of its common stock during 1995. [1]'
    assert round5.evaluate_answer(good, hits, [part], {'1995_p0001'})[0] is None
    bad = good + ' Insurance float reached $9.9 billion at yearend. [1]'      # 1 of 2 sentences unsupported -> 50% < 85%
    reason, _, stats = round5.evaluate_answer(bad, hits, [part], {'1995_p0001'})
    assert reason == 'validator_dropped_over_15pct' and stats['keep_rate'] == .5
    # a gap sentence is only allowed for a part whose source is not served
    gap = good + ' The passages do not state the buyback price.'
    assert round5.evaluate_answer(gap, hits, [part], {'1995_p0001'})[0] == 'unmatched_gap_sentence'

    generator, _ = make_generator(tmp_path, ScriptedTeacher(question_json(), answer=bad))
    question, _ = generator.synthesize(single_job())
    generator.process(question)
    assert not generator.rows
    (attempt,) = generator.attempts
    assert attempt['reason'] == 'validator_dropped_over_15pct' and attempt['retried'] is True


# ------------------------------------------------------------------ exclusions
def test_exclusion_union_contains_heldout_v5_and_v4_ids_and_refuses_on_a_failing_checker(tmp_path):
    full = {d.id: d for d in common.load_chunks_as_docs(common.CORPUS)}
    v5 = round5.checker_ids(round5.HELDOUT_V5_CHECKER, 'heldout_v5')
    v4 = round5.checker_ids(round5.HELDOUT_V4_CHECKER, 'heldout_v4')
    dev, _ = round5.gold_ids(ROOT / round5.DEV_CASES)
    union = round5.exclusion_union(full, v4=v4, v5=v5, dev=dev)
    assert v5[0] in union and v4[0] in union and next(iter(dev)) in union
    assert set(v5) <= union and set(v4) <= union

    failing = tmp_path / 'check.py'
    failing.write_text('import sys\nsys.exit(3)\n')
    with pytest.raises(RuntimeError, match='refusing'):
        round5.checker_ids(failing, 'heldout_v5')
    with pytest.raises(RuntimeError, match='refusing'):
        round5.load_inputs(tmp_path / 'never', v5_checker=failing)
    assert not (tmp_path / 'never').exists()


# ------------------------------------------------------------------ dry run on 5 jobs
def test_dry_run_on_five_jobs_prints_token_stats_per_kind(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('FT_TEACHER', str(tmp_path / 'no-tokenizer'))        # falls back to chars/4
    loaded = round5.load_inputs(tmp_path / 'dry')
    docs, excluded, *_ = loaded
    assert not {d.id for d in docs} & excluded
    full = loaded[-1]
    serving = round5.Serving(BM25OnlyRetriever(list(full.values())), full)
    args = SimpleNamespace(mix=None, target=None, oversample=4.)
    stats = round5.dry_run(5, args, serving=serving, loaded=loaded, out=tmp_path / 'dry')
    out = capsys.readouterr().out
    assert set(stats) == set(round5.KINDS)
    assert 'tok_p50' in out and 'tok_p95' in out and 'tok_max' in out and 'ALL:' in out
    for kind in round5.KINDS:
        assert any(line.startswith(kind) for line in out.splitlines())
    assert sum(len(entry['tokens']) for entry in stats.values()) >= 5
