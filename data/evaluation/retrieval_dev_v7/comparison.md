# Retrieval dev v7: served-context recall

Claim-level served recall = share of gold claims whose `gold_passage_ids` hit the ids of the passages placed in the 7B prompt (the `hits` list of /ask). Pool: answerable cases of heldout_v3..v6 and dev_cases (all spent). Secondary column: ids merged into a served passage as neighbour text.

## Determinism (baseline vs stored heldout_v6 `passage_ids`)

100.0% of 79 cases have the same served ids (79 identical order, 0 same ids other order); 0 differ.


## Pooled

| config | claims served | claim recall % | delta pp | CI95 pp (paired bootstrap) | merged-ids recall % | case any-gold % | HN pass | p50 ms | max tokens | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 607/765 | 79.3 |  |  | 84.3 | 88.8 | 35/40 | 517 | 7389 |  |
| fu2 | 607/765 | 79.3 | +0.00 | [+0.00, +0.00] | 84.3 | 88.8 | 35/40 | 536 | 7389 | fail: ci_lower_gt_0 |
| fualways | 607/765 | 79.3 | +0.00 | [-0.65, +0.65] | 84.3 | 88.8 | 35/40 | 533 | 7389 | fail: ci_lower_gt_0 |
| multipart | 623/765 | 81.4 | +2.09 | [+1.05, +3.27] | 86.3 | 90.4 | 36/40 | 1108 | 7389 | PASS |
| p10 | 611/765 | 79.9 | +0.52 | [+0.13, +1.05] | 84.8 | 89.4 | 35/40 | 1376 | 8243 | fail: prompt_fits_n_ctx |
| res2 | 607/765 | 79.3 | +0.00 | [+0.00, +0.00] | 84.3 | 88.8 | 35/40 | 526 | 7389 | fail: ci_lower_gt_0 |

## Per set (claim recall %)

| config | dev_cases | heldout_v3 | heldout_v4 | heldout_v5a | heldout_v5b | heldout_v6a | heldout_v6b |
|---|---|---|---|---|---|---|---|
| baseline | 93.7 | 78.0 | 90.0 | 74.5 | 83.3 | 71.2 | 73.7 |
| fu2 | 93.7 | 78.0 | 90.0 | 74.5 | 83.3 | 71.2 | 73.7 |
| fualways | 93.7 | 78.0 | 90.0 | 72.5 | 83.3 | 72.9 | 74.3 |
| multipart | 93.7 | 78.0 | 90.0 | 75.2 | 84.7 | 76.3 | 78.3 |
| p10 | 94.7 | 82.9 | 90.0 | 74.5 | 83.3 | 72.0 | 73.7 |
| res2 | 93.7 | 78.0 | 90.0 | 74.5 | 83.3 | 71.2 | 73.7 |

## Per question type (claim recall %, claims in brackets)

| config | answerable [12] | company [173] | fact_number [170] | follow_up [101] | opinion [165] | temporal_comparison [144] |
|---|---|---|---|---|---|---|
| baseline | 83.3 | 83.8 | 83.5 | 85.1 | 73.9 | 70.8 |
| fu2 | 83.3 | 83.8 | 83.5 | 85.1 | 73.9 | 70.8 |
| fualways | 83.3 | 83.8 | 83.5 | 85.1 | 73.9 | 70.8 |
| multipart | 83.3 | 87.9 | 83.5 | 86.1 | 78.8 | 70.8 |
| p10 | 83.3 | 83.8 | 84.7 | 85.1 | 74.5 | 71.5 |
| res2 | 83.3 | 83.8 | 83.5 | 85.1 | 73.9 | 70.8 |

## Latency and prompt size

Latency = wall time of retrieval + rerank + context assembly plus the measured cost of cached model calls (machine shared with other jobs: indicative only). Tokens: Qwen2.5 tokenizer of the served FT model on the chat-templated request; n_ctx 8192, 768 tokens reserved for the answer.

| config | p50 ms | wall p50 ms | passages p50 | tokens p50 | tokens max | fits n_ctx-768 |
|---|---|---|---|---|---|---|
| baseline | 517 | 492 | 8.0 | 4421.5 | 7389 | True |
| fu2 | 536 | 478 | 8.0 | 4421.5 | 7389 | True |
| fualways | 533 | 483 | 8.0 | 4405.5 | 7389 | True |
| multipart | 1108 | 832 | 8.0 | 4396.0 | 7389 | True |
| p10 | 1376 | 43 | 10.0 | 5219.0 | 8243 | False |
| res2 | 526 | 474 | 8.0 | 4421.5 | 7389 | True |

## Overrides

- baseline: {'LLM_CONTEXT_PASSAGES': '8'}
- fu2: {'LLM_CONTEXT_PASSAGES': '8', 'RETRIEVAL_FOLLOWUP_TURNS': '2'}
- fualways: {'LLM_CONTEXT_PASSAGES': '8', 'RETRIEVAL_FOLLOWUP_ALWAYS': '1'}
- multipart: {'LLM_CONTEXT_PASSAGES': '8', 'RETRIEVAL_MULTIPART': '1'}
- p10: {'LLM_CONTEXT_PASSAGES': '10'}
- res2: {'LLM_CONTEXT_PASSAGES': '8', 'RETRIEVAL_PERIOD_RESERVE': '2'}

## Caveats and candidates not measured

- Measured: baseline, multipart, p10 (LLM_CONTEXT_PASSAGES=10), fu2, fualways, res2. NOT measured (the shared machine ran out of memory, swap 15-18 GB, and the runs stalled; they were killed): rerank depth 25/30, MMR (RETRIEVAL_MMR_LAMBDA), LLM_CONTEXT_PASSAGES=12, RRF k, BM25/dense weights, ANSWER_CONTEXT_NEIGHBORS=2, neighbour anchors (RETRIEVAL_NEIGHBOR_ANCHORS), and the combinations (multipart + 10 passages). Their settings exist behind env keys with the old default values (see `config.RETRIEVAL_DEFAULTS`) and can be run with `served_context_recall.py run --config NAME:KEY=VAL`.
- p10: gain +0.52pp with CI95 lower bound > 0, but the longest prompt is 8243 tokens (real Qwen2.5 tokenizer, chat template) against the 8192-token window: `fit_context_to_llm` budgets 4 chars per token and that bound is too optimistic for number-heavy passages. The baseline itself peaks at 7389 tokens (803 left for the answer). 10 or 12 passages need a stricter `LLM_CHARS_PER_TOKEN` (setting added, default 4.0 unchanged); the stricter value was not measured, so LLM_CONTEXT_PASSAGES stays 8.
- Determinism: baseline reproduced the stored heldout_v6 `passage_ids` for 79 of 79 answerable hv6 cases (identical order). The 16 hv6 refusal cases are not in the recall pool and their serve run stalled under the memory pressure, so they are unchecked. The run used the MPS device for the embedder and reranker (the stored runs' device is not recorded); results matched anyway.
- fu2, fualways, res2 changed nothing or nothing significant. Settings that act on a subset of questions (follow-ups, multi-part, temporal comparisons) reused the baseline rows for out-of-scope cases, which are byte-identical by construction; this also makes their p50 latency partly the baseline's.
- Latency is wall time plus the measured cost of cached model calls; the machine was shared and swapping, so p50 is indicative only (p10's wall p50 of 43 ms is the cache effect, its 1376 ms the cost estimate). multipart adds one retrieval + rerank per extra question part, on multi-part questions only.
- The pool was used for selection and is spent, so the multipart gain (+2.1pp, CI95 [+1.05, +3.27], hard_negatives_v4 36/40 vs 35/40, no type below baseline) is a development estimate, not a blind one.

## Rule and decision

A candidate becomes the default only when its pooled claim-level served-recall gain has a paired per-claim bootstrap (10,000 draws, seed 0) CI95 lower bound > 0, hard_negatives_v4 passes are not lower, no question type drops by more than 0.02, and the longest prompt leaves >= 768 tokens of the 8192-token window. Each setting is an environment variable read at call time (`config.retrieval_setting`, `config.llm_context_passages`).

Final configuration: `multipart` {'LLM_CONTEXT_PASSAGES': '8', 'RETRIEVAL_MULTIPART': '1'}.

Note: scripts that set `LLM_CONTEXT_PASSAGES=8` explicitly (Artifacts `run_arm.sh`) override the new default and must be changed to the new value to measure it.

DEFAULT: CHANGED RETRIEVAL_MULTIPART=1
