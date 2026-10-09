# heldout_v3 blind measurement: 7B base (arm 8) and Kaggle 7B FT-r1 (arm 9), each run once

Frozen blind set `data/evaluation/heldout_v3/answer_benchmark_heldout_v3.json` (40 cases, sha256 `23f3a71930ce761412a1366e313bd319c5c36ec9454538ca1f0b9d4beea1baf3`), BM25, temperature 0, `--save-raw`, llama provider, fresh process per arm, no code or setting change. Custody: `scripts/eval/check_heldout_v3.py` OK (FT answer passages of data/ft_v3 train/valid excluded). Screen written before the run (PREREGISTRATION.md). Arm 1 is the U7 extractive result re-scored from `../final/`, not re-run.

Engineering screen, not a statistical test: 40 cases, one run per arm; per-type differences of 1-2 claims are noise. local Mac, llama.cpp Metal (LLM_GPU_LAYERS=-1); latency not comparable with U7 arms.

## Screening verdict for arm 9 (7B FT-r1)

- FAIL: strict supported claims: arm9 >= arm1 + 3 AND arm9 >= arm8 -- arm9=14, arm1=13 (need >= 16), arm8=2
- PASS: correct refusals = all genuine unanswerable cases -- arm9 correct refusals=6/6
- FAIL: abstentions on answerable <= arm1 -- arm9=13, arm1=6
- PASS: zero invalid markers -- arm9=0
- PASS: zero provider failures -- arm9 failed cases=0, error events=0
- FAIL: no question type with fewer strict claims than arm1 by more than 1 -- opinion: arm9=1 < arm1=3 - 1

Verdict: **not a candidate** (no default changed; the user decides)

## Per-arm results

| arm | name | lexical acc. | NLI acc. | strict supported met/req | NLI claims met | invalid markers | unsupp. extra sent. | correct refusals | unexpected refusals | abstain answerable | provider failures | warm p50 ms | warm p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | local_extractive | 14/40 | 17/40 | 13/41 | 19/41 | 0 | 74 | 3/6 | 3 | 6/34 | 0 | 41.0 | 51.2 |
| 8 | llama7b_base | 9/40 | 11/40 | 2/41 | 5/41 | 0 | 12 | 6/6 | 21 | 27/34 | 0 | 4493.2 | 7190.7 |
| 9 | llama7b_ft | 19/40 | 16/40 | 14/41 | 14/41 | 0 | 13 | 6/6 | 8 | 13/34 | 0 | 4279.1 | 6342.1 |

## Strict supported claims per type (met / required)

| arm | opinion | company | fact_number | temporal_comparison | follow_up |
|---|---|---|---|---|---|
| 1 local_extractive | 3/9 | 6/9 | 1/9 | 1/9 | 2/5 |
| 8 llama7b_base | 0/9 | 2/9 | 0/9 | 0/9 | 0/5 |
| 9 llama7b_ft | 1/9 | 8/9 | 4/9 | 0/9 | 1/5 |

Paired arm9_vs_arm1: arm 9 better in 4 cases, worse in 4, equal in 26.

Paired arm9_vs_arm8: arm 9 better in 11 cases, worse in 0, equal in 23.
