| config | MRR | R@1 | R@5 | R@10 | HN hard_negatives_v2 | HN hard_negatives_v3 | latency ms |
|---|---|---|---|---|---|---|---|
| bm25 | 0.678 | 0.560 | 0.820 | 0.940 | 8/8 | 8/8 | 6 |
| vector:bge-base-en-v1.5 | 0.645 | 0.520 | 0.820 | 0.920 | 6/8 | 7/8 | 24 |
| hybrid:bge-base-en-v1.5 | 0.733 | 0.580 | 0.940 | 0.940 | 8/8 | 8/8 | 20 |
| vector:bge-small-en-v1.5 | 0.655 | 0.480 | 0.840 | 0.940 | 8/8 | 8/8 | 6 |
| hybrid:bge-small-en-v1.5 | 0.728 | 0.580 | 0.920 | 0.960 | 8/8 | 8/8 | 13 |
| vector:multilingual-e5-large | 0.648 | 0.460 | 0.900 | 0.960 | 7/8 | 8/8 | 16 |
| hybrid:multilingual-e5-large | 0.731 | 0.600 | 0.920 | 0.980 | 7/8 | 8/8 | 21 |
| vector:all-MiniLM-L6-v2 | 0.524 | 0.380 | 0.700 | 0.760 | 4/8 | 3/8 | 4 |
| hybrid:all-MiniLM-L6-v2 | 0.668 | 0.480 | 0.880 | 0.920 | 7/8 | 7/8 | 10 |
| hybrid:bge-base-en-v1.5+rerank:none | 0.733 | 0.580 | 0.940 | 0.940 | 8/8 | 8/8 | 14 |
| hybrid:bge-base-en-v1.5+rerank:bge-reranker-v2-m3 | 0.722 | 0.580 | 0.880 | 0.980 | 8/8 | 8/8 | 436 |
| hybrid:bge-base-en-v1.5+rerank:bge-reranker-base (weights not in offline cache) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
