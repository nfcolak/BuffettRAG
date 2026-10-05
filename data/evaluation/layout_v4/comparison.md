# Layout corpus v4 vs v3 (BM25, offline)

Gold set: 50 queries. Hard negatives: 8 cases (v4 ids mapped via v3->v4 map).

| corpus | chunks | recall@1 | recall@3 | recall@5 | recall@10 | mrr | year_hit_rate@5 | hard-neg passed |
|---|---|---|---|---|---|---|---|---|
| v3 paragraph | 5831 | 0.560 | 0.720 | 0.820 | 0.940 | 0.675 | 0.404 | 8/8 |
| v4 layout | 5779 | 0.580 | 0.740 | 0.840 | 0.920 | 0.690 | 0.412 | 6/8 |

## Audit, PDF years 1998-2024 (v3 -> v4)

| metric | v3 | v4 |
|---|---|---|
| chunks | 3446 | 3394 |
| mean_chunk_chars | 679.8 | 583.5 |
| line_break_in_sentence | 22845 | 0 |
| hyphen_break | 2 | 1 |
| header_page_number_residue | 0 | 0 |
| table_paragraphs | 0 | 221 |
| table_paragraphs_unmarked_in_prose | 514 | 0 |

v3 ids without a confident v4 match (token Jaccard < 0.5): 620 of 5831 (620 in PDF years).
