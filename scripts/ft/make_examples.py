"""Build resumable BM25-grounded MLX chat examples with strict claim filters."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import re
import time

from common import OUT, ROOT, SEED, Teacher, append, atomic_json, content_terms, inputs, read_jsonl, sha
from make_questions import generate_pending
from src.evaluation.claim_validator import validate_and_filter_answer, _split_original
from src.evaluation.citation_faithfulness import split_sentences
from src.generation.evidence_gate import assess_evidence
from src.generation.prompt import REFUSAL_LINE, SYSTEM_PROMPT, build_cited_prompt, parse_citations
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.context import expand_hits_with_neighbors

STYLE = '\nFor this answer, be brief: prefer one or two short, self-contained sentences. Preserve the supporting passage wording rather than introducing synonyms. Do not infer or calculate. Cite each sentence; copy the refusal exactly if unsupported.'


def chat_messages(question, hits):
    prompt = build_cited_prompt(question, hits)
    prefix = SYSTEM_PROMPT + '\n\n'
    assert prompt.startswith(prefix)
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt[len(prefix):]}]


def check_answer(answer, hits, source_id):
    if not answer or answer == REFUSAL_LINE:
        return 'answerable_refusal'
    result = validate_and_filter_answer(answer, hits)
    if result.blocked_claims:
        return 'blocked_claims'
    for line in answer.splitlines():
        for sentence in _split_original(line):
            citations = parse_citations(sentence, hits)
            if not citations or not all(c['passage_indices'] and not c['invalid_numbers'] for c in citations):
                return 'missing_or_invalid_sentence_citation'
    cited = {ident for citation in parse_citations(answer, hits) for ident in citation['passage_ids']}
    if source_id not in cited:
        return 'source_not_cited'
    return None


class Builder:
    def __init__(self, teacher, revision=0):
        self.teacher = teacher
        self.revision = revision
        self.docs, self.excluded, _, self.valid_years, self.identity = inputs()
        self.lookup = {doc.id: doc for doc in self.docs}
        self.bm25 = BM25Retriever(self.docs)
        self.years = {split: sorted({int(doc.metadata['year']) for doc in self.docs if ('valid' if int(doc.metadata['year']) in self.valid_years else 'train') == split}) for split in ('train', 'valid')}
        self.rows = read_jsonl(OUT / 'examples.jsonl')
        self.accepted = {row['id'] for row in self.rows}
        self.attempts = read_jsonl(OUT / 'example_attempts.jsonl')
        self.done = {row['attempt_id'] for row in self.attempts}
        self.materialize()

    def retrieve(self, question):
        years = self.years[question['split']]
        explicit_years = {int(value) for value in re.findall(r'\b(?:19|20)\d{2}\b', question['question'])}
        compatible = sorted(explicit_years.intersection(years))
        # Serving retrieval auto-detects source years in the query. Keep the
        # explicit year constraint, but never allow an out-of-split context.
        if compatible:
            years = compatible
        anchors = self.bm25.search(question['question'], top_k=5, where={'year': {'$in': years}})
        hits = expand_hits_with_neighbors(anchors, self.lookup, neighbors=1, max_chars=1800)
        assert all(len(hit.text) <= 1800 for hit in hits)
        return hits

    def context_ids(self, hits, lookup=None):
        lookup = self.lookup if lookup is None else lookup
        ids = set()
        for hit in hits:
            ids.add(hit.id)
            doc = lookup.get(hit.id)
            if doc:
                for field in ('previous_chunk_id', 'next_chunk_id'):
                    neighbor = doc.metadata.get(field)
                    if neighbor in lookup:
                        ids.add(neighbor)
        return sorted(ids)

    def process(self, question, retry=False):
        ident = question['id']
        attempt_id = f'{ident}:r{self.revision}'
        if ident in self.accepted or attempt_id in self.done:
            return False
        hits = self.retrieve(question)
        messages = chat_messages(question['question'], hits)
        generation_messages = [dict(message) for message in messages]
        generation_messages[0]['content'] += STYLE
        if retry:
            generation_messages[1]['content'] += ('\n\nFINAL OUTPUT REMINDER: Return exactly ONE short, self-contained factual sentence answering the question, followed immediately by its [n] citation. '
                                                  'Use the wording of a complete supporting sentence from a passage; do not add introductory or explanatory clauses. '
                                                  'Do not list multiple citations, do not say "as stated", and do not write a second sentence. '
                                                  'If unsupported, copy the refusal line alone, WITHOUT a citation.')
            if question['type'] == 'answerable':
                source_indices = [index + 1 for index, hit in enumerate(hits) if hit.id == question['source_id']]
                if source_indices:
                    number = source_indices[0]
                    generation_messages[1]['content'] += f'\nThe source passage for this question is [{number}]. Answer using only a directly supporting sentence from [{number}] and end with [{number}].'
                    terms = content_terms(question['question'])
                    source_text = re.sub(r'\s+', ' ', hits[number - 1].text)
                    candidates = []
                    for sentence in split_sentences(source_text):
                        if not 40 <= len(sentence) <= 650:
                            continue
                        cited_sentence = sentence.strip() + f' [{number}]'
                        if check_answer(cited_sentence, hits, question['source_id']) is None:
                            overlap = len(terms & content_terms(sentence))
                            if overlap:
                                candidates.append((overlap, cited_sentence))
                    candidates.sort(key=lambda value: (-value[0], len(value[1])))
                    if candidates:
                        generation_messages[1]['content'] += '\nSupporting wording already present in the source passage (choose the sentence that directly answers; do not invent a connecting explanation):\n' + '\n'.join(value[1] for value in candidates[:3])
        output, metrics = self.teacher.generate(generation_messages, attempt_id, max_tokens=350, temperature=.2)
        reason, replaced = None, False
        if metrics['finish_reason'] == 'length':
            reason = 'truncated_teacher_output'
        elif question['type'] == 'answerable':
            reason = check_answer(output, hits, question['source_id'])
        elif output != REFUSAL_LINE:
            terms = content_terms(question['question'])
            if all(len(terms & content_terms(hit.text)) < 2 for hit in hits):
                output, replaced = REFUSAL_LINE, True
            else:
                reason = 'unanswerable_teacher_answered_with_lexical_overlap'
        record = {'attempt_id': attempt_id, 'id': ident, 'type': question['type'], 'kept': reason is None, 'reason': reason, 'teacher_output': output, 'refusal_replaced': replaced, **metrics}
        if reason is None:
            self.keep(question, hits, messages, output, metrics, replaced=replaced)
        append(OUT / 'example_attempts.jsonl', record)
        self.attempts.append(record)
        self.done.add(attempt_id)
        print(f'example={ident} kept={reason is None} reason={reason} accepted={len(self.rows)} tok/s={metrics["generation_tps"]:.2f}', flush=True)
        return True

    def keep(self, question, hits, messages, answer, metrics, *, replaced=False, lookup=None):
        row = {**question, 'passage_ids': [hit.id for hit in hits], 'context_passage_ids': self.context_ids(hits, lookup), 'messages': messages + [{'role': 'assistant', 'content': answer}], 'teacher_metrics': metrics, 'refusal_replaced': replaced}
        assert not set(row['context_passage_ids']) & self.excluded
        append(OUT / 'examples.jsonl', row)
        self.rows.append(row)
        self.accepted.add(row['id'])
        if len(self.rows) % 5 == 0:
            self.materialize()

    def distractor(self, question):
        ident = question['id'] + ':distractor'
        if ident in self.accepted or ident in self.done:
            return
        if sum(row['type'] == 'distractor' for row in self.rows) >= 150:
            return
        hits = self.retrieve(question)
        if question['source_id'] not in {hit.id for hit in hits}:
            return
        lookup = {key: value for key, value in self.lookup.items() if key != question['source_id']}
        # Also re-expand the remaining anchors without the source, so a neighbor
        # cannot smuggle it back into a nominally source-free negative.
        remaining = [hit for hit in hits if hit.id != question['source_id']]
        remaining = expand_hits_with_neighbors(remaining, lookup, neighbors=1, max_chars=1800)
        sufficient = assess_evidence(question['question'], remaining).sufficient
        record = {'attempt_id': ident, 'id': ident, 'type': 'distractor', 'kept': not sufficient, 'reason': 'remaining_evidence_sufficient' if sufficient else None, 'tokens': 0, 'seconds': 0., 'generation_tps': 0.}
        if not sufficient:
            modified = {**question, 'id': ident, 'type': 'distractor'}
            assert question['source_id'] not in self.context_ids(remaining, lookup)
            self.keep(modified, remaining, chat_messages(question['question'], remaining), REFUSAL_LINE, {'tokens': 0, 'seconds': 0., 'generation_tps': 0.}, lookup=lookup)
        append(OUT / 'example_attempts.jsonl', record)
        self.attempts.append(record)
        self.done.add(ident)

    def materialize(self, *, final=False):
        selected = []
        for split in ('train', 'valid'):
            positives = [row for row in self.rows if row['split'] == split and row['type'] == 'answerable']
            negatives = [row for row in self.rows if row['split'] == split and row['type'] != 'answerable']
            # Never fabricate negatives or relax their checks. If the final
            # positive yield is unusually high, downsample positives to attain
            # the requested minimum 15% refusal share instead.
            if final and negatives:
                positive_limit = len(negatives) * 17 // 3
                positives.sort(key=lambda row: sha(f'{SEED}:positive:{row["id"]}'.encode()))
                positives = positives[:positive_limit]
            # Preserve accepted positives during progress snapshots; cap refusal
            # share at 25%. Mix negative types independently of creation order.
            negatives.sort(key=lambda row: sha(f'{SEED}:{row["id"]}'.encode()))
            negatives = negatives[:len(positives) // 3]
            rows = positives + negatives
            rows.sort(key=lambda row: sha(f'{SEED}:order:{row["id"]}'.encode()))
            path = OUT / f'{split}.jsonl'
            temporary = path.with_suffix('.jsonl.tmp')
            with temporary.open('w') as handle:
                for row in rows:
                    handle.write(json.dumps({'messages': row['messages']}, ensure_ascii=False) + '\n')
            temporary.replace(path)
            selected.extend(rows)
        temporary = OUT / 'split_metadata.jsonl.tmp'
        with temporary.open('w') as handle:
            for row in selected:
                handle.write(json.dumps({key: value for key, value in row.items() if key not in {'messages', 'teacher_metrics'}}, ensure_ascii=False) + '\n')
        temporary.replace(OUT / 'split_metadata.jsonl')
        return selected

    def finish(self):
        selected = self.materialize(final=True)
        counts = {split: dict(Counter(row['type'] for row in selected if row['split'] == split)) for split in ('train', 'valid')}
        totals = {split: sum(counts[split].values()) for split in counts}
        answer_attempts = [row for row in self.attempts if row['type'] != 'distractor']
        question_jobs = read_jsonl(OUT / 'question_attempts.jsonl')
        manifest = {**self.identity, 'counts_per_split_type': counts, 'counts_per_split': totals, 'accepted_before_refusal_cap': len(self.rows), 'question_count': len(read_jsonl(OUT / 'questions.jsonl')), 'question_jobs_completed': len(question_jobs), 'sampled_passages': sum(row['id'].startswith('passage:') for row in question_jobs), 'generation_attempts': len(answer_attempts), 'kept_generation_attempts': sum(row['kept'] for row in answer_attempts), 'keep_rate': sum(row['kept'] for row in answer_attempts) / max(1, len(answer_attempts)), 'rejection_reasons': dict(Counter(row['reason'] for row in self.attempts if row['reason'])), 'refusal_share': {split: sum(count for kind, count in counts[split].items() if kind != 'answerable') / max(1, totals[split]) for split in counts}, 'settings': {'temperature': .2, 'max_tokens': 350, 'LLM_CONTEXT_PASSAGES': 5, 'LLM_PASSAGE_MAX_CHARS': 1800, 'split_strategy': 'seeded source-year grouping; retrieval and neighbor expansion remain inside each split'}, 'target_met': totals['train'] >= 1000 and all(.15 <= share <= .25 for share in [sum(count for kind, count in counts['train'].items() if kind != 'answerable') / max(1, totals['train'])])}
        atomic_json(OUT / 'manifest.json', manifest)
        self.update_readme(totals, counts, manifest['refusal_share'])
        print(json.dumps(manifest, indent=2), flush=True)

    def update_readme(self, totals, counts, shares):
        path = OUT / 'README.md'
        if path.exists():
            lines = path.read_text().splitlines()
            lines = [f'Counts: train={totals["train"]}, valid={totals["valid"]}; types={json.dumps(counts, sort_keys=True)}; train refusal share={shares["train"]:.1%}.' if line.startswith('Counts:') else line for line in lines]
            temporary = path.with_suffix('.md.tmp')
            temporary.write_text('\n'.join(lines) + '\n')
            temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pilot', type=int, help='Run only this many new teacher answers; do not synthesize more questions')
    parser.add_argument('--prompt-revision', type=int, default=1, choices=(0, 1))
    parser.add_argument('--split-only', action='store_true', help='Rewrite train/valid/split_metadata/manifest from existing examples.jsonl; no generation')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.split_only:
        builder = Builder(None)
        builder.finish()
        from check_leakage import check
        check()
        valid = len(read_jsonl(OUT / 'valid.jsonl'))
        total = valid + len(read_jsonl(OUT / 'train.jsonl'))
        assert valid >= .05 * total, f'valid split too small: {valid}/{total}'
        return
    # The lock file is harmless and local; prevents two writers corrupting append logs.
    lock = (OUT / 'generation.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / 'examples.jsonl').exists() and not args.resume:
        parser.error('Output exists; use --resume.')
    teacher = Teacher()
    builder = Builder(teacher, args.prompt_revision)
    start_index = len(builder.attempts)
    started = time.monotonic()
    processed = 0
    questions = read_jsonl(OUT / 'questions.jsonl')
    for question in questions:
        if args.pilot is not None and processed >= args.pilot:
            break
        if builder.process(question, retry=args.prompt_revision > 0):
            processed += 1
        if args.pilot is None and question['type'] == 'answerable':
            builder.distractor(question)
    if args.pilot is not None:
        builder.materialize()
        attempts = builder.attempts[start_index:]
        kept = sum(row['kept'] for row in attempts)
        elapsed = time.monotonic() - started
        tokens = sum(row['tokens'] for row in attempts)
        summary = {'examples': len(attempts), 'kept': kept, 'keep_rate': kept / max(1, len(attempts)), 'generation_tok_per_second': sum(row['generation_tps'] for row in attempts) / max(1, len(attempts)), 'end_to_end_tok_per_second': tokens / max(.001, elapsed), 'seconds': elapsed, 'prompt_revision': args.prompt_revision}
        atomic_json(OUT / f'pilot_r{args.prompt_revision}.json', summary)
        print('PILOT ' + json.dumps(summary), flush=True)
        return
    # Finish question synthesis and teacher answers in interleaved resumable jobs;
    # no second GPU process, and progress becomes usable before all 900 sources finish.
    for batch in generate_pending(teacher):
        for question in batch:
            builder.process(question, retry=args.prompt_revision > 0)
            if question['type'] == 'answerable':
                builder.distractor(question)
    # Retry rejected records with the SAME fixed prompt and independent seeded
    # sampling. Stop retrying once both delivered training targets are satisfied.
    # A rejected output is never converted into a made-up positive.
    for revision in (1, 2, 3):
        builder.revision = revision
        for question in read_jsonl(OUT / 'questions.jsonl'):
            if question['id'] not in builder.accepted:
                builder.process(question, retry=True)
        selected = builder.materialize()
        train = [row for row in selected if row['split'] == 'train']
        refusal_share = sum(row['type'] != 'answerable' for row in train) / max(1, len(train))
        if len(train) >= 1000 and .15 <= refusal_share <= .25:
            break
    builder.finish()
    from check_leakage import check
    check()


if __name__ == '__main__':
    main()
