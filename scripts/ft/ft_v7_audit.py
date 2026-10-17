"""Offline audit: how many rows of an existing set (default data/ft_v6) would each ft_v7 rule drop? No model, no teacher.

  PY scripts/ft/ft_v7_audit.py [--dir data/ft_v6] [--tokens] [--out audit.json]

Rows are train.jsonl / valid.jsonl joined (in file order, as materialize writes them) with split_metadata.jsonl, then run through
build_ft_v6.check_row (alignment, citation rule, serving replay, dependent starts, refusal terms, split) plus the question-overlap rule
(which needs the source passages). The rows keep their own prompt version (ft_v6: v2)."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_ft_v6 as B  # noqa: E402
import common  # noqa: E402
import ft_v7_checks as C  # noqa: E402

OLD_DEPENDENT_START = re.compile(r'^(?:This|That|These|Those|It|They|He|She|Such|However|But|And|Also|Thus|Hence|Moreover|Furthermore|Yet|So|Then|Still|'
                                 r'Instead|Both|Here|There|Its|Their|His|Her)\b')
OVERLAP_THRESHOLDS = (.50, .55, .60, .65, .70)


def load_rows(directory):
    directory = Path(directory)
    meta = [json.loads(line) for line in (directory / 'split_metadata.jsonl').read_text().splitlines() if line.strip()]
    rows = []
    for split in ('train', 'valid'):
        messages = [json.loads(line)['messages'] for line in (directory / f'{split}.jsonl').read_text().splitlines() if line.strip()]
        chosen = [m for m in meta if m['split'] == split]
        assert len(chosen) == len(messages), f'{split}: {len(chosen)} metadata rows vs {len(messages)} training rows'
        for m, msgs in zip(chosen, messages):
            assert msgs[2]['content'] == joined_target(m, msgs), f'{m["id"]}: metadata and training row are not the same row'
            rows.append({**m, 'messages': msgs})
    return rows


def joined_target(meta, messages):
    """Join check: the stored target must start with the first obligation sentence (or be the refusal line); returns the target when it does."""
    target = messages[2]['content']
    if meta['type'] == 'unanswerable':
        return target if target == B.REFUSAL_LINE else None
    first = B.norm(meta['obligations'][0]['sentences'][0])
    return target if target.startswith(first) else None


def audit(rows, docs, excluded, valid_years, count_tokens=None, max_seq=None):
    """-> {'rows', 'rows_with_any_problem', 'kept', 'by_rule': {code: {'rows': n, 'train': n, 'valid': n}}, 'by_kind', 'question_overlap', ...}."""
    by_rule, any_problem, by_kind = {}, Counter(), Counter()
    kept, kept_without_overlap = Counter(), Counter()
    first_reason = Counter()
    overlaps = []
    for row in rows:
        problems = B.check_row(row, excluded=excluded, count_tokens=count_tokens, valid_years=valid_years)
        if row['type'] == 'answerable' and row['kind'] in B.KINDS and docs:
            texts = [docs[i].text for i in row['source_ids'] if i in docs]
            overlap = C.question_overlap(row['question'], texts)
            overlaps.append(overlap)
            if overlap > B.QUESTION_OVERLAP_MAX:
                problems.append('question_overlaps_passage')
        if row['type'] == 'answerable' and any(OLD_DEPENDENT_START.match(o['sentences'][0]) for o in row['obligations']):
            problems = [p for p in problems if p != 'dependent_start'] + ['dependent_start_old_list']
        for code in set(problems):
            entry = by_rule.setdefault(code, Counter())
            entry['rows'] += 1
            entry[row['split']] += 1
        if not [p for p in problems if p != 'question_overlaps_passage']:
            kept_without_overlap[row['split']] += 1
        if problems:
            any_problem[row['split']] += 1
            by_kind[row['kind']] += 1
            first_reason[problems[0]] += 1
        else:
            kept[row['split']] += 1
    total = Counter(r['split'] for r in rows)
    out = {'rows': len(rows), 'rows_per_split': dict(total), 'rows_with_any_problem': sum(any_problem.values()), 'any_problem_per_split': dict(any_problem),
           'kept_after_all_rules': sum(kept.values()), 'kept_per_split': dict(kept),
           'kept_without_the_overlap_rule': sum(kept_without_overlap.values()), 'kept_without_the_overlap_rule_per_split': dict(kept_without_overlap), 'rows_with_problem_by_kind': dict(by_kind),
           'rows_per_kind': dict(Counter(r['kind'] for r in rows)), 'by_rule': {k: dict(v) for k, v in sorted(by_rule.items())},
           'first_problem': dict(first_reason)}
    if overlaps:
        out['question_overlap'] = {'mean': round(sum(overlaps) / len(overlaps), 3),
                                   **{f'above_{t}': sum(o > t for o in overlaps) for t in OVERLAP_THRESHOLDS}, 'answerable_rows': len(overlaps)}
    refusals = [r for r in rows if r['type'] == 'unanswerable']
    out['refusals'] = {'rows': len(refusals), 'with_year_after_2024': sum(bool(C.FUTURE_YEAR.search(r['question'])) for r in refusals),
                       'adjacent': sum(r.get('variant') == 'adjacent' for r in refusals), 'template': sum(r.get('variant') == 'template' for r in refusals)}
    answerable = [r for r in rows if r['type'] == 'answerable']
    out['answerable_with_year_after_2024'] = sum(bool(C.FUTURE_YEAR.search(r['question'])) for r in answerable)
    if max_seq and count_tokens:
        out['token_lengths'] = {'over_max_seq': by_rule.get('too_long_for_max_seq', Counter())['rows'], 'max_seq': max_seq}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', default='data/ft_v6')
    parser.add_argument('--tokens', action='store_true', help='also count tokens (loads the Qwen2.5 tokenizer on CPU) against --max-seq')
    parser.add_argument('--max-seq', type=int, default=B.MAX_SEQ_MIX)
    parser.add_argument('--out', help='write the audit as JSON here')
    args = parser.parse_args(argv)
    import os
    os.environ['PROMPT_VERSION'] = B.PROMPT_VERSION
    directory = Path(args.dir)
    directory = directory if directory.is_absolute() else common.ROOT / directory
    rows = load_rows(directory)
    docs = {d.id: d for d in common.load_chunks_as_docs(common.CORPUS)}
    excluded, _ = B.exclusion_sets(docs)
    valid_years = B.valid_years_of(docs.values())
    count_tokens = None
    if args.tokens:
        _, count_tokens = B.round4.token_counter(B.DEFAULT_TEACHER)
        rows = [{**r, 'max_seq': args.max_seq} for r in rows]
    result = audit(rows, docs, excluded, valid_years, count_tokens, args.max_seq if args.tokens else None)
    result['dir'] = str(directory)
    context_excluded = sum(bool(set(r['context_passage_ids']) & excluded) for r in rows)
    result['rows_with_excluded_passage_in_context'] = context_excluded
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
