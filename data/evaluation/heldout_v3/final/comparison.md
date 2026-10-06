# heldout_v3 blind measurement (U7): 7 arms, each run once

Frozen blind set `data/evaluation/heldout_v3/answer_benchmark_heldout_v3.json` (40 cases, sha256 `23f3a71930ce761412a1366e313bd319c5c36ec9454538ca1f0b9d4beea1baf3`), BM25, temperature 0, `--save-raw`, one arm after another, fresh process each. Frozen grounded defaults (T_RELEVANT 0.01, T_SLOT 0.001, MAX_UNITS 6, NUMERIC_GUARD bound); no code or setting changed.

Engineering screen, not a statistical test: 40 cases, one run per arm; per-type differences of 1-2 claims are noise. Human adjudication of arms 1, 4, 5 is required before any default change; the user decides.

Machine load: shared machine, other agents active; load average at start of the run 5.77 3.82 3.50, per-arm values below (3-15)

## Screening verdict for arm 5 (grounded + mlx 7B)

- FAIL: supported claims met: arm5 >= arm1 + 3 AND arm5 >= arm4 -- arm5=15, arm1=13 (need >= 16), arm4=4 (need >= 4)
- FAIL: correct refusals = all genuine refusal cases -- arm5 correct refusals=3/6
- PASS: abstentions on answerable <= arm1 -- arm5=5, arm1=6
- PASS: zero invalid markers -- arm5=0
- PASS: zero provider failures -- arm5 failed cases=0, error events=0
- PASS: no type with fewer supported claims than arm1 by more than 1 -- per type arm5 vs arm1: opinion 2 vs 3, company 7 vs 6, fact_number 4 vs 1, temporal_comparison 1 vs 1, follow_up 1 vs 2
- PASS: warm p95 <= 8 s -- arm5 warm p95=6695.3 ms (n warm=39)

Verdict: **not a candidate**

## Per-arm results

| arm | name | lexical acc. | NLI acc. | strict supported met/req | NLI claims met | invalid markers | unsupp. extra sent. | correct refusals | unexpected refusals | abstain answerable (incl. notes) | provider failures | warm p50 ms | warm p95 ms | cold n / max ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | local_extractive | 14/40 | 17/40 | 13/41 | 19/41 | 0 | 74 | 3/6 | 3 | 6/34 | 0 | 41.0 | 51.2 | 1 / 43.5 |
| 2 | llama_base | 7/40 | 6/40 | 1/41 | 0/41 | 0 | 12 | 6/6 | 26 | 32/34 | 0 | 713.4 | 1590.1 | 1 / 2178.5 |
| 3 | llama_ftr2 | 7/40 | 6/40 | 1/41 | 0/41 | 0 | 12 | 6/6 | 26 | 32/34 | 0 | 698.3 | 1508.7 | 1 / 1966.3 |
| 4 | mlx7b_oldpath | 10/40 | 9/40 | 4/41 | 5/41 | 0 | 12 | 6/6 | 21 | 27/34 | 0 | 3137.1 | 4353.8 | 1 / 6667.8 |
| 5 | grounded_mlx7b | 16/40 | 7/40 | 15/41 | 17/41 | 0 | 24 | 3/6 | 5 | 5/34 | 0 | 2396.6 | 6695.3 | 1 / 5794.3 |
| 6 | grounded_template | 14/40 | 9/40 | 14/41 | 16/41 | 0 | 25 | 3/6 | 5 | 5/34 | 0 | 1035.4 | 4981.5 | 1 / 2842.3 |
| 7 | grounded_llama_ftr2 | 14/40 | 8/40 | 14/41 | 14/41 | 0 | 24 | 3/6 | 5 | 5/34 | 0 | 1465.1 | 5368.2 | 1 / 3275.1 |

Definitions. Strict supported claim: the gold claim is met by an answer sentence that matches lexically and whose cited gold hit passes the deterministic verifier (`src.evaluation.supported_claims`, same scorer as the dev confirmation). Grounded arms use the saved `context_snapshot`; arms 1-4 have none, so their client context is rebuilt by re-running the retrieval and context-building path with a stub provider (no model call; ids checked against `passage_ids`, method validated against the saved snapshots of the grounded arms). Unsupported extra sentence: an answer sentence on an answerable case that is not a coverage note or refusal, not a strict supporting sentence of a required claim, and not NLI-supported (cited and entailed >= 0.5). Abstention on answerable: refusal, all-notes, empty or marker-only answer. Unexpected refusal: refusal line on an answerable case. Correct refusals are counted on the genuine unanswerable cases only (R5: a comparison with one side absent is answerable and is not scored as a refusal). Warm = every case except those that loaded a model/scorer (grounded: summed cold_load_ms > 50 ms; other arms: the first case that called the model).

## Strict supported claims per type (met / required)

| arm | opinion | company | fact_number | temporal_comparison | follow_up | unanswerable |
|---|---|---|---|---|---|---|
| 1 local_extractive | 3/9 | 6/9 | 1/9 | 1/9 | 2/5 | 0/0 |
| 2 llama_base | 1/9 | 0/9 | 0/9 | 0/9 | 0/5 | 0/0 |
| 3 llama_ftr2 | 1/9 | 0/9 | 0/9 | 0/9 | 0/5 | 0/0 |
| 4 mlx7b_oldpath | 1/9 | 2/9 | 1/9 | 0/9 | 0/5 | 0/0 |
| 5 grounded_mlx7b | 2/9 | 7/9 | 4/9 | 1/9 | 1/5 | 0/0 |
| 6 grounded_template | 1/9 | 7/9 | 4/9 | 1/9 | 1/5 | 0/0 |
| 7 grounded_llama_ftr2 | 1/9 | 6/9 | 4/9 | 2/9 | 1/5 | 0/0 |

## Grounded arms: fallback to template

- arm 5 grounded_mlx7b: groups composed 36, fallback to template 25 (rate 0.6944); reasons {'verifier_rejected': 10, 'model_refused': 15}; decisions {'answer': 32, 'refuse': 8}
- arm 6 grounded_template: groups composed 36, fallback to template 0 (rate 0.0); reasons {}; decisions {'answer': 32, 'refuse': 8}
- arm 7 grounded_llama_ftr2: groups composed 36, fallback to template 28 (rate 0.7778); reasons {'verifier_rejected': 14, 'model_refused': 14}; decisions {'answer': 32, 'refuse': 8}

## Abstentions on answerable cases (incl. coverage-note-only answers)

- arm 1 local_extractive: 6 (hv3_02 refusal, hv3_10 refusal, hv3_19 refusal, hv3_26 no cited sentence, hv3_27 no cited sentence, hv3_29 no cited sentence)
- arm 2 llama_base: 32 (hv3_01 refusal, hv3_02 refusal, hv3_04 refusal, hv3_05 refusal, hv3_06 refusal, hv3_08 refusal, hv3_09 refusal, hv3_10 refusal, hv3_11 refusal, hv3_12 refusal, hv3_13 refusal, hv3_14 refusal, hv3_15 refusal, hv3_16 refusal, hv3_17 refusal, hv3_18 refusal, hv3_19 refusal, hv3_20 no cited sentence, hv3_21 refusal, hv3_22 refusal, hv3_23 refusal, hv3_24 refusal, hv3_25 refusal, hv3_26 no cited sentence, hv3_27 no cited sentence, hv3_28 no cited sentence, hv3_29 no cited sentence, hv3_30 no cited sentence, hv3_31 refusal, hv3_32 refusal, hv3_33 refusal, hv3_34 refusal)
- arm 3 llama_ftr2: 32 (hv3_01 refusal, hv3_02 refusal, hv3_04 refusal, hv3_05 refusal, hv3_06 refusal, hv3_08 refusal, hv3_09 refusal, hv3_10 refusal, hv3_11 refusal, hv3_12 refusal, hv3_13 refusal, hv3_14 refusal, hv3_15 refusal, hv3_16 refusal, hv3_17 refusal, hv3_18 refusal, hv3_19 refusal, hv3_20 no cited sentence, hv3_21 refusal, hv3_22 refusal, hv3_23 refusal, hv3_24 refusal, hv3_25 refusal, hv3_26 no cited sentence, hv3_27 no cited sentence, hv3_28 no cited sentence, hv3_29 no cited sentence, hv3_30 no cited sentence, hv3_31 refusal, hv3_32 refusal, hv3_33 refusal, hv3_34 refusal)
- arm 4 mlx7b_oldpath: 27 (hv3_01 refusal, hv3_02 refusal, hv3_04 refusal, hv3_05 refusal, hv3_06 refusal, hv3_07 refusal, hv3_08 refusal, hv3_10 refusal, hv3_12 refusal, hv3_15 refusal, hv3_16 refusal, hv3_17 refusal, hv3_18 refusal, hv3_19 refusal, hv3_20 no cited sentence, hv3_21 refusal, hv3_22 refusal, hv3_23 refusal, hv3_26 no cited sentence, hv3_27 no cited sentence, hv3_28 no cited sentence, hv3_29 no cited sentence, hv3_30 no cited sentence, hv3_31 refusal, hv3_32 refusal, hv3_33 refusal, hv3_34 refusal)
- arm 5 grounded_mlx7b: 5 (hv3_02 refusal, hv3_05 refusal, hv3_10 refusal, hv3_19 refusal, hv3_26 refusal)
- arm 6 grounded_template: 5 (hv3_02 refusal, hv3_05 refusal, hv3_10 refusal, hv3_19 refusal, hv3_26 refusal)
- arm 7 grounded_llama_ftr2: 5 (hv3_02 refusal, hv3_05 refusal, hv3_10 refusal, hv3_19 refusal, hv3_26 refusal)

## Latency (ms)

| arm | all n/p50/p95 | warm n/p50/p95 | cold n/p50/p95 | setup |
|---|---|---|---|---|
| 1 local_extractive | 40/41.2/51.0 | 39/41.0/51.2 | 1/43.5/43.5 | 181.663 |
| 2 llama_base | 40/716.3/1694.9 | 39/713.4/1590.1 | 1/2178.5/2178.5 | 183.878 |
| 3 llama_ftr2 | 40/714.3/1681.9 | 39/698.3/1508.7 | 1/1966.3/1966.3 | 189.773 |
| 4 mlx7b_oldpath | 40/3146.7/4520.1 | 39/3137.1/4353.8 | 1/6667.8/6667.8 | 203.413 |
| 5 grounded_mlx7b | 40/2410.2/6636.6 | 39/2396.6/6695.3 | 1/5794.3/5794.3 | 187.858 |
| 6 grounded_template | 40/1042.8/4979.9 | 39/1035.4/4981.5 | 1/2842.3/2842.3 | 202.502 |
| 7 grounded_llama_ftr2 | 40/1479.2/5366.1 | 39/1465.1/5368.2 | 1/3275.1/3275.1 | 186.843 |

Machine load per arm start (from progress.txt): local_extractive 5.77 3.82 3.50; llama_base 5.11 3.75 3.48; llama_ftr2 5.54 3.94 3.56; mlx7b_oldpath 3.79 3.67 3.47; grounded_mlx7b 14.66 6.58 4.56; grounded_template 6.29 5.98 4.57; grounded_llama_ftr2 3.32 5.19 4.38. The machine is shared with other agents; latency is indicative only.

## Effective model / device / temperature (from the manifests)

| arm | provider/composer | model path | temperature (requested; asserted at provider) | device | code sha (dirty) |
|---|---|---|---|---|---|
| 1 local_extractive | local/None | embedded-extractive-v1 | 0.0; None | None, torch_mps=None; NLI on mps |  (None) |
| 2 llama_base | llama/None | qwen2.5-1.5b-instruct-q4_k_m.gguf | 0.0; None | None, torch_mps=None; NLI on mps |  (None) |
| 3 llama_ftr2 | llama/None | qwen2.5-1.5b-instruct-q4_k_m.gguf | 0.0; None | None, torch_mps=None; NLI on mps |  (None) |
| 4 mlx7b_oldpath | mlx/None | /Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/models/teacher-qwen2.5-7b-mlx4 | 0.0; True | arm64, torch_mps=True; NLI on mps | 7a4e8f6f4a (True) |
| 5 grounded_mlx7b | grounded/mlx | /Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/models/teacher-qwen2.5-7b-mlx4 | 0.0; True | arm64, torch_mps=True; NLI on mps | 7a4e8f6f4a (True) |
| 6 grounded_template | grounded/template | grounded:template | 0.0; True | arm64, torch_mps=True; NLI on mps | 7a4e8f6f4a (True) |
| 7 grounded_llama_ftr2 | grounded/llama | /Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/models/ft/round2/buffett-qwen2.5-1.5b-ft-r2-q4_k_m.gguf | 0.0; True | arm64, torch_mps=True; NLI on mps | 7a4e8f6f4a (True) |

- arm 5: context rebuild check against the saved snapshot: 40/40 rows identical (ids and text)
- arm 6: context rebuild check against the saved snapshot: 40/40 rows identical (ids and text)
- arm 7: context rebuild check against the saved snapshot: 40/40 rows identical (ids and text)

## Paired strict supported claims per case: arm 5 vs arm 1 (extractive)

| qid | type | required | arm5 | arm1 | delta |
|---|---|---|---|---|---|
| hv3_01 | opinion | 1 | 0 | 0 | +0 |
| hv3_02 | opinion | 1 | 0 | 0 | +0 |
| hv3_03 | opinion | 1 | 0 | 1 | -1 |
| hv3_04 | opinion | 1 | 1 | 1 | +0 |
| hv3_05 | opinion | 1 | 0 | 0 | +0 |
| hv3_06 | opinion | 1 | 0 | 0 | +0 |
| hv3_07 | opinion | 1 | 0 | 0 | +0 |
| hv3_08 | opinion | 1 | 0 | 0 | +0 |
| hv3_09 | opinion | 1 | 1 | 1 | +0 |
| hv3_10 | company | 1 | 0 | 0 | +0 |
| hv3_11 | company | 1 | 1 | 1 | +0 |
| hv3_12 | company | 1 | 1 | 1 | +0 |
| hv3_13 | company | 1 | 1 | 1 | +0 |
| hv3_14 | company | 1 | 1 | 1 | +0 |
| hv3_15 | company | 1 | 1 | 1 | +0 |
| hv3_16 | company | 1 | 0 | 1 | -1 |
| hv3_17 | company | 2 | 2 | 0 | +2 |
| hv3_18 | fact_number | 1 | 1 | 0 | +1 |
| hv3_19 | fact_number | 1 | 0 | 0 | +0 |
| hv3_20 | fact_number | 1 | 0 | 0 | +0 |
| hv3_21 | fact_number | 1 | 0 | 0 | +0 |
| hv3_22 | fact_number | 2 | 2 | 1 | +1 |
| hv3_23 | fact_number | 1 | 0 | 0 | +0 |
| hv3_24 | fact_number | 1 | 0 | 0 | +0 |
| hv3_25 | fact_number | 1 | 1 | 0 | +1 |
| hv3_26 | temporal_comparison | 2 | 0 | 0 | +0 |
| hv3_27 | temporal_comparison | 2 | 0 | 0 | +0 |
| hv3_28 | temporal_comparison | 2 | 1 | 1 | +0 |
| hv3_29 | temporal_comparison | 2 | 0 | 0 | +0 |
| hv3_30 | temporal_comparison | 1 | 0 | 0 | +0 |
| hv3_31 | follow_up | 1 | 0 | 0 | +0 |
| hv3_32 | follow_up | 1 | 1 | 1 | +0 |
| hv3_33 | follow_up | 2 | 0 | 1 | -1 |
| hv3_34 | follow_up | 1 | 0 | 0 | +0 |

Cases with a different count: arm5 better in 4, worse in 3, equal in 27.

## Paired strict supported claims per case: arm 5 vs arm 4 (mlx 7B old path)

| qid | type | required | arm5 | arm4 | delta |
|---|---|---|---|---|---|
| hv3_01 | opinion | 1 | 0 | 0 | +0 |
| hv3_02 | opinion | 1 | 0 | 0 | +0 |
| hv3_03 | opinion | 1 | 0 | 0 | +0 |
| hv3_04 | opinion | 1 | 1 | 0 | +1 |
| hv3_05 | opinion | 1 | 0 | 0 | +0 |
| hv3_06 | opinion | 1 | 0 | 0 | +0 |
| hv3_07 | opinion | 1 | 0 | 0 | +0 |
| hv3_08 | opinion | 1 | 0 | 0 | +0 |
| hv3_09 | opinion | 1 | 1 | 1 | +0 |
| hv3_10 | company | 1 | 0 | 0 | +0 |
| hv3_11 | company | 1 | 1 | 1 | +0 |
| hv3_12 | company | 1 | 1 | 0 | +1 |
| hv3_13 | company | 1 | 1 | 0 | +1 |
| hv3_14 | company | 1 | 1 | 1 | +0 |
| hv3_15 | company | 1 | 1 | 0 | +1 |
| hv3_16 | company | 1 | 0 | 0 | +0 |
| hv3_17 | company | 2 | 2 | 0 | +2 |
| hv3_18 | fact_number | 1 | 1 | 0 | +1 |
| hv3_19 | fact_number | 1 | 0 | 0 | +0 |
| hv3_20 | fact_number | 1 | 0 | 0 | +0 |
| hv3_21 | fact_number | 1 | 0 | 0 | +0 |
| hv3_22 | fact_number | 2 | 2 | 0 | +2 |
| hv3_23 | fact_number | 1 | 0 | 0 | +0 |
| hv3_24 | fact_number | 1 | 0 | 0 | +0 |
| hv3_25 | fact_number | 1 | 1 | 1 | +0 |
| hv3_26 | temporal_comparison | 2 | 0 | 0 | +0 |
| hv3_27 | temporal_comparison | 2 | 0 | 0 | +0 |
| hv3_28 | temporal_comparison | 2 | 1 | 0 | +1 |
| hv3_29 | temporal_comparison | 2 | 0 | 0 | +0 |
| hv3_30 | temporal_comparison | 1 | 0 | 0 | +0 |
| hv3_31 | follow_up | 1 | 0 | 0 | +0 |
| hv3_32 | follow_up | 1 | 1 | 0 | +1 |
| hv3_33 | follow_up | 2 | 0 | 0 | +0 |
| hv3_34 | follow_up | 1 | 0 | 0 | +0 |

Cases with a different count: arm5 better in 9, worse in 0, equal in 25.

