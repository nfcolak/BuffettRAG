# Corpus v5_pl (v3 + Buffett Partnership letters 1957-1970) vs v3: retrieval regression

**Basis: hybrid retrieval WITHOUT the cross-encoder rerank.** The reranked run (`CMP_RERANK=1`, bge-reranker-v2-m3) was started but did not finish within the time budget on the loaded machine (about 30 s/query on CPU, MPS stalled), so this verdict rests on FAISS + BM25 + RRF order (`CMP_RERANK=0`). Rerank reorders only the top-15 fused candidates; it was not measured here.

Serving hybrid retrieval as in `run_live_benchmark.py --retrieval hybrid` (FAISS bge-base-en-v1.5 + BM25 + RRF, NO cross-encoder rerank (hybrid RRF order only), auto year filter, top 8, fetch_k 30). Labels are the old v3 ids: gold-50 (year + keyword, text based), dev (gold passages of the answerable grounded_dev_v1 cases), hard_negatives_v4 (relevant ids). v3 baseline uses the 1977-2024 year bounds that served v3; v5_pl uses 1957-2024. Pooled = all three sets; MRR is over the returned top 8.

| set | n | corpus | recall@8 | MRR | top-8 slots on partnership passages | queries with >=1 partnership hit |
|---|---|---|---|---|---|---|
| gold50 | 50 | v3 | 0.9400 | 0.7557 | 0 / 400 | 0 |
| gold50 | 50 | v5_pl | 0.9400 | 0.7557 | 11 / 400 | 8 |
| dev | 71 | v3 | 0.9296 | 0.6891 | 0 / 568 | 0 |
| dev | 71 | v5_pl | 0.9296 | 0.6838 | 5 / 568 | 2 |
| hard_neg | 40 | v3 | 0.9000 | 0.7913 | 0 / 320 | 0 |
| hard_neg | 40 | v5_pl | 0.9000 | 0.8037 | 1 / 320 | 1 |
| pooled | 161 | v3 | 0.9255 | 0.7352 | 0 / 1288 | 0 |
| pooled | 161 | v5_pl | 0.9255 | 0.7359 | 17 / 1288 | 11 |

hard_negatives_v4 passes (best relevant rank < best hard-negative rank): v3 35/40, v5_pl 35/40; cases that changed: none.

Pooled recall@8 drop (v3 - v5_pl): +0.0000.

Queries whose recall@8 or reciprocal rank changed:

| query | recall@8 v3 -> v5_pl | RR v3 -> v5_pl | partnership slots (v5_pl) | year filter (v5_pl) |
|---|---|---|---|---|
| dev:adv04 | 1 -> 1 | 0.500 -> 0.125 | 4 | {'multi_subquery': [{'year': 1965}, {'year': 1985}]} |
| hard_neg:hn4_30_voting_weighing_machine | 1 -> 1 | 0.500 -> 1.000 | 1 | None |

REGRESSION: OK
