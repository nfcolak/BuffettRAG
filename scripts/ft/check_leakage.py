"""Audit the delivered splits, frozen neighbors, questions, and citation targets."""
from __future__ import annotations
from collections import defaultdict
import json
import re
from common import OUT, inputs, near_frozen, read_jsonl
from src.evaluation.claim_validator import validate_and_filter_answer, _split_original
from src.generation.evidence_gate import assess_evidence
from src.generation.prompt import REFUSAL_LINE, SYSTEM_PROMPT, build_cited_prompt, parse_citations
from src.retrieval.context import expand_hits_with_neighbors
from src.vector_store import SearchHit


def check():
    docs, excluded, frozen, valid_years, identity = inputs()
    lookup = {doc.id: doc for doc in docs}
    questions = read_jsonl(OUT / 'questions.jsonl')
    plan = json.loads((OUT / 'question_plan.json').read_text())
    assert plan['identity'] == identity, 'Corpus/frozen identity changed'
    examples = {row['id']: row for row in read_jsonl(OUT / 'examples.jsonl')}
    metadata = read_jsonl(OUT / 'split_metadata.jsonl')
    used = defaultdict(set)
    overlapping = set()
    near_duplicates = sum(near_frozen(row['question'], frozen) for row in questions)
    for row in questions:
        if row['source_id'] in excluded:
            overlapping.add(row['source_id'])
    expected = defaultdict(list)
    for row in metadata:
        full = examples[row['id']]
        expected[row['split']].append(full)
        used[row['split']].update(row['context_passage_ids'])
        if row['source_id']:
            used[row['split']].add(row['source_id'])
        overlapping.update(set(row['context_passage_ids']) & excluded)
    cross_split = used['train'] & used['valid']
    for split in ('train', 'valid'):
        delivered = read_jsonl(OUT / f'{split}.jsonl')
        assert len(delivered) == len(expected[split]), f'{split} metadata count mismatch (retry if generator writing)'
        for sample, full in zip(delivered, expected[split]):
            assert sample == {'messages': full['messages']}, 'Chat/metadata mismatch'
            assert [turn['role'] for turn in sample['messages']] == ['system', 'user', 'assistant']
            assert sample['messages'][0]['content'] == SYSTEM_PROMPT
            question = full['question']
            anchors = [SearchHit(id=ident, text=lookup[ident].text, metadata=lookup[ident].metadata, score=0.) for ident in full['passage_ids']]
            local_lookup = lookup
            if full['type'] == 'distractor':
                local_lookup = {ident: doc for ident, doc in lookup.items() if ident != full['source_id']}
            hits = expand_hits_with_neighbors(anchors, local_lookup, neighbors=1, max_chars=1800)
            assert all(len(hit.text) <= 1800 for hit in hits)
            user = build_cited_prompt(question, hits)[len(SYSTEM_PROMPT + '\n\n'):]
            assert user == sample['messages'][1]['content'], 'Serving prompt drift'
            answer = sample['messages'][2]['content']
            if full['type'] == 'answerable':
                result = validate_and_filter_answer(answer, hits)
                assert not result.blocked_claims, f'Blocked claims in {full["id"]}'
                cited = {ident for citation in parse_citations(answer, hits) for ident in citation['passage_ids']}
                assert full['source_id'] in cited
                for line in answer.splitlines():
                    for sentence in _split_original(line):
                        citations = parse_citations(sentence, hits)
                        assert citations and all(c['passage_indices'] and not c['invalid_numbers'] for c in citations)
            else:
                assert answer == REFUSAL_LINE
                if full['type'] == 'distractor':
                    assert full['source_id'] not in full['context_passage_ids']
                    assert not assess_evidence(question, hits).sufficient
    print(f'{len(overlapping)} overlapping passage ids / {near_duplicates} near-duplicate questions / {len(cross_split)} cross-split passage ids', flush=True)
    print(f'checked train={len(expected["train"])} valid={len(expected["valid"])} questions={len(questions)} excluded={len(excluded)}', flush=True)
    assert not overlapping and not near_duplicates and not cross_split
    return {'overlapping_passage_ids': len(overlapping), 'near_duplicate_questions': near_duplicates, 'cross_split_passage_ids': len(cross_split)}


if __name__ == '__main__':
    check()
