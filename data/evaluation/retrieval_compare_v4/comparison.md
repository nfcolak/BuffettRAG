# Corpus v4 (layout) vs v3 (paragraph): retrieval comparison

Serving retrieval as in `run_live_benchmark.py --retrieval hybrid` (FAISS bge-base-en-v1.5 + BM25 + RRF, bge-reranker-v2-m3, year filter, top 8, fetch_k 30); BM25 = the benchmark's BM25 mode. Labels: gold-50 (year + keyword, text based), dev (gold passages of the 71 answerable grounded_dev_v1 cases), hard_negatives_v4 (40 cases; relevant ids). Pooled = all three sets. MRR is over the returned top 8.

Label map: 164 (set, v3 id) entries, 119 method=map, 45 method=manual; unresolved 0; dropped queries none.

## Primary (text-verified label map)

| mode | set | n | corpus | recall@1 | recall@5 | recall@8 | MRR |
|---|---|---|---|---|---|---|---|
| hybrid | gold50 | 50 | v3 | 0.620 | 0.900 | 0.940 | 0.743 |
| hybrid | gold50 | 50 | v4 | 0.520 | 0.880 | 0.920 | 0.682 |
| hybrid | dev | 71 | v3 | 0.732 | 0.915 | 0.944 | 0.813 |
| hybrid | dev | 71 | v4 | 0.732 | 0.915 | 0.930 | 0.805 |
| hybrid | hard_neg | 40 | v3 | 0.775 | 0.850 | 0.900 | 0.819 |
| hybrid | hard_neg | 40 | v4 | 0.775 | 0.875 | 0.925 | 0.826 |
| hybrid | pooled | 161 | v3 | 0.708 | 0.894 | 0.932 | 0.793 |
| hybrid | pooled | 161 | v4 | 0.677 | 0.894 | 0.925 | 0.772 |
| bm25 | gold50 | 50 | v3 | 0.600 | 0.820 | 0.880 | 0.694 |
| bm25 | gold50 | 50 | v4 | 0.600 | 0.840 | 0.900 | 0.700 |
| bm25 | dev | 71 | v3 | 0.648 | 0.873 | 0.930 | 0.743 |
| bm25 | dev | 71 | v4 | 0.592 | 0.845 | 0.915 | 0.716 |
| bm25 | hard_neg | 40 | v3 | 0.675 | 0.850 | 0.875 | 0.760 |
| bm25 | hard_neg | 40 | v4 | 0.750 | 0.900 | 0.925 | 0.809 |
| bm25 | pooled | 161 | v3 | 0.640 | 0.851 | 0.901 | 0.732 |
| bm25 | pooled | 161 | v4 | 0.633 | 0.857 | 0.913 | 0.734 |

hard_negatives_v4 passed (existing criterion: best relevant rank < best hard-negative rank, within the returned list):

- v3/hybrid: 35/40
- v3/bm25: 33/40; plain BM25 top-20 (eval_hard_negatives.py path): 33/40
- v4/hybrid: 34/40
- v4/bm25: 35/40; plain BM25 top-20 (eval_hard_negatives.py path): 36/40

Paired per-query bootstrap (10,000 resamples, seed 0), v4 - v3:

- hybrid (n=161): recall@8 -0.0062 CI95 [-0.0311, 0.0186]; MRR -0.0209 CI95 [-0.0519, 0.0099]; queries recall@8 v4 better 2, v3 better 3
- bm25 (n=161): recall@8 +0.0124 CI95 [-0.0186, 0.0497]; MRR +0.0020 CI95 [-0.0312, 0.0358]; queries recall@8 v4 better 5, v3 better 3

Rules (PREREGISTRATION.md section 1), hybrid, pooled — primary:

- recall@8 v4 - v3 >= +0.03 and paired-bootstrap 95% CI lower bound > 0: not met (diff -0.0062, CI [-0.0311, 0.0186])
- MRR v4 >= v3: not met (v3 0.7929, v4 0.7720)
- hard_negatives_v4 passed v4 >= v3: not met (v3 35, v4 34)

Verdict under primary: STAY

## Sensitivity: literal id map (top-1 where Jaccard >= 0.5)

| mode | set | n | corpus | recall@1 | recall@5 | recall@8 | MRR |
|---|---|---|---|---|---|---|---|
| hybrid | gold50 | 50 | v3 | 0.620 | 0.900 | 0.940 | 0.743 |
| hybrid | gold50 | 50 | v4 | 0.520 | 0.880 | 0.920 | 0.682 |
| hybrid | dev | 71 | v3 | 0.732 | 0.915 | 0.944 | 0.813 |
| hybrid | dev | 71 | v4 | 0.648 | 0.859 | 0.873 | 0.726 |
| hybrid | hard_neg | 40 | v3 | 0.775 | 0.850 | 0.900 | 0.819 |
| hybrid | hard_neg | 40 | v4 | 0.750 | 0.825 | 0.900 | 0.792 |
| hybrid | pooled | 161 | v3 | 0.708 | 0.894 | 0.932 | 0.793 |
| hybrid | pooled | 161 | v4 | 0.633 | 0.857 | 0.894 | 0.729 |
| bm25 | gold50 | 50 | v3 | 0.600 | 0.820 | 0.880 | 0.694 |
| bm25 | gold50 | 50 | v4 | 0.600 | 0.840 | 0.900 | 0.700 |
| bm25 | dev | 71 | v3 | 0.648 | 0.873 | 0.930 | 0.743 |
| bm25 | dev | 71 | v4 | 0.521 | 0.775 | 0.859 | 0.643 |
| bm25 | hard_neg | 40 | v3 | 0.675 | 0.850 | 0.875 | 0.760 |
| bm25 | hard_neg | 40 | v4 | 0.700 | 0.850 | 0.875 | 0.759 |
| bm25 | pooled | 161 | v3 | 0.640 | 0.851 | 0.901 | 0.732 |
| bm25 | pooled | 161 | v4 | 0.590 | 0.814 | 0.876 | 0.689 |

hard_negatives_v4 passed (existing criterion: best relevant rank < best hard-negative rank, within the returned list):

- v3/hybrid: 35/40
- v3/bm25: 33/40; plain BM25 top-20 (eval_hard_negatives.py path): 33/40
- v4/hybrid: 34/40
- v4/bm25: 33/40; plain BM25 top-20 (eval_hard_negatives.py path): 34/40

Paired per-query bootstrap (10,000 resamples, seed 0), v4 - v3:

- hybrid (n=161): recall@8 -0.0373 CI95 [-0.0745, 0.0]; MRR -0.0641 CI95 [-0.1091, -0.0218]; queries recall@8 v4 better 2, v3 better 8
- bm25 (n=161): recall@8 -0.0248 CI95 [-0.0683, 0.0186]; MRR -0.0427 CI95 [-0.0891, 0.0015]; queries recall@8 v4 better 5, v3 better 9

Rules (PREREGISTRATION.md section 1), hybrid, pooled — literal_map:

- recall@8 v4 - v3 >= +0.03 and paired-bootstrap 95% CI lower bound > 0: not met (diff -0.0373, CI [-0.0745, 0.0])
- MRR v4 >= v3: not met (v3 0.7929, v4 0.7288)
- hard_negatives_v4 passed v4 >= v3: not met (v3 35, v4 34)

Verdict under literal_map: STAY

VERDICT: STAY
