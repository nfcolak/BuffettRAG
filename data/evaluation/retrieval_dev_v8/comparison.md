# Retrieval dev v8: served-context recall (corpus v5_pl, pool incl. devpl)

Claim-level served recall = share of gold claims whose `gold_passage_ids` hit the ids of the passages placed in the 7B prompt (the `hits` list of /ask). Pool: answerable cases of heldout_v3..v6 and dev_cases (all spent). Secondary column: ids merged into a served passage as neighbour text.

## Determinism (baseline vs stored heldout_v6 `passage_ids`)

48.1% of 79 cases have the same served ids (38 identical order, 0 same ids other order); 41 differ.

- hv6a_01: now ['2014_p0240', '1981_p0039', '1989_p0138', '2016_p0025', '2018_p0055', '1987_p0120', '1987_p0123', '1982_p0060'] vs stored ['2014_p0240', '1981_p0039', '1989_p0138', '2016_p0025', '2018_p0055', '2016_p0035', '2014_p0230', '1987_p0121']
- hv6a_03: now ['1987_p0071', 'bpl1958_p0013', '2009_p0026', '1991_p0020', '2018_p0033', '2013_p0105', '1990_p0087', '1987_p0066'] vs stored ['2013_p0102', '1987_p0071', '2009_p0026', '1991_p0020', '2018_p0033', '2013_p0105', '1990_p0087', '1987_p0066']
- hv6a_08: now ['1989_p0132', '2014_p0241', '1989_p0136', '2000_p0090', '1986_p0124', '2014_p0238', '2009_p0096', '1992_p0097'] vs stored ['1989_p0132', '2014_p0241', '1989_p0136', '2000_p0090', '1986_p0124', '2014_p0230', '2018_p0025', '2024_p0003']
- hv6a_11: now ['1999_p0056', '1997_p0071', '1997_p0069', '1997_p0073', '1999_p0059', '1997_p0072', '1997_p0070', '2007_p0131'] vs stored ['1999_p0056', '1997_p0071', '1997_p0069', '1997_p0073', '1999_p0059', '1997_p0072', '1997_p0070', '1999_p0060']
- hv6a_12: now ['1997_p0096', '1997_p0093', '1990_p0100', '1987_p0127', '1995_p0082', '1987_p0111', '1996_p0099', '1996_p0098'] vs stored ['1997_p0096', '1997_p0093', '1990_p0100', '1987_p0127', '1995_p0082', '1987_p0111', '1997_p0098', '1995_p0092']
- hv6a_13: now ['1999_p0072', '1999_p0056', '1999_p0005', '1999_p0059', '1999_p0058', '1999_p0057', 'bpl1965_p0070', '2014_p0191'] vs stored ['1999_p0072', '1999_p0056', '1999_p0005', '1999_p0059', '1999_p0058', '1999_p0057', '1992_p0024', '1999_p0078']
- hv6a_15: now ['2011_p0093', '2007_p0023', '2016_p0098', '2007_p0024', '2012_p0075', '2014_p0139', '2016_p0034', '1980_p0012'] vs stored ['2011_p0093', '2007_p0023', '2016_p0098', '2007_p0024', '2012_p0075', '2014_p0139', '1988_p0098', '2015_p0019']
- hv6a_16: now ['1992_p0104', '1989_p0036', '1983_p0014', '1988_p0030', '2020_p0048', '1985_p0076', '1999_p0056', '2020_p0050'] vs stored ['1992_p0104', '1989_p0036', '1983_p0014', '1988_p0030', '2020_p0048', '1985_p0076', '1999_p0056', '1984_p0028']
- hv6a_17: now ['1992_p0020', '1981_p0009', '2010_p0156', '1982_p0088', '2000_p0013', '1981_p0008', '1981_p0012', '2006_p0040'] vs stored ['1992_p0020', '1981_p0009', '2010_p0156', '1982_p0088', '2000_p0013', '2014_p0243', '2014_p0224', '2006_p0033']
- hv6a_18: now ['1991_p0002', '1991_p0000', '1991_p0081', '1991_p0014', '2022_p0040', '2010_p0035', '2019_p0026', '1997_p0018'] vs stored ['1991_p0002', '1991_p0000', '1991_p0081', '1991_p0018', '1991_p0001', '1991_p0014', '1991_p0086', '1991_p0031']

## Pooled

| config | claims served | claim recall % | delta pp | CI95 pp (paired bootstrap) | merged-ids recall % | case any-gold % | HN pass | p50 ms | max tokens | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 685/875 | 78.3 |  |  | 83.3 | 88.9 | 36/40 | 1303 | 7389 |  |
| bm25w15 | 671/875 | 76.7 | -1.60 | [-2.74, -0.46] | 82.1 | 88.6 | 36/40 | 848 | 7389 | fail: ci_lower_gt_0,no_type_drop_gt_0.02 |
| combo16 | 718/875 | 82.1 | +3.77 | [+2.06, +5.60] | 86.2 | 91.6 | 36/40 | 1264 | 10847 | fail: no_type_drop_gt_0.02 |
| combo16r30 | 748/875 | 85.5 | +7.20 | [+5.37, +9.14] | 89.5 | 93.9 | 36/40 | 2146 | 13332 | PASS |
| combo20r30 | 757/875 | 86.5 | +8.23 | [+6.29, +10.29] | 89.9 | 94.7 | 36/40 | 2101 | 16407 | fail: prompt_fits_n_ctx |
| dedup0 | 685/875 | 78.3 | +0.00 | [+0.00, +0.00] | 83.3 | 88.9 | 36/40 | 779 | 7389 | fail: ci_lower_gt_0 |
| densew15 | 668/875 | 76.3 | -1.94 | [-3.31, -0.57] | 82.3 | 86.6 | 37/40 | 860 | 7389 | fail: ci_lower_gt_0,devpl_not_lower,no_type_drop_gt_0.02 |
| fu2 | 685/875 | 78.3 | +0.00 | [+0.00, +0.00] | 83.3 | 88.9 | 36/40 | 1286 | 7389 | fail: ci_lower_gt_0 |
| fualways | 684/875 | 78.2 | -0.11 | [-0.57, +0.34] | 83.2 | 88.9 | 36/40 | 1303 | 7389 | fail: ci_lower_gt_0 |
| n12p10 | 692/875 | 79.1 | +0.80 | [+0.23, +1.49] | 84.0 | 89.4 | 36/40 | 759 | 8243 | PASS |
| n12p12 | 693/875 | 79.2 | +0.91 | [+0.34, +1.60] | 84.1 | 89.7 | 36/40 | 762 | 9277 | PASS |
| n12p16 | 705/875 | 80.6 | +2.29 | [+1.37, +3.31] | 85.7 | 91.4 | 36/40 | 857 | 10847 | PASS |
| n12p8 | 685/875 | 78.3 | +0.00 | [+0.00, +0.00] | 83.3 | 88.9 | 36/40 | 759 | 7389 | fail: ci_lower_gt_0 |
| n16p10 | 692/875 | 79.1 | +0.80 | [+0.23, +1.49] | 84.0 | 89.4 | 36/40 | 780 | 8243 | PASS |
| n16p12 | 693/875 | 79.2 | +0.91 | [+0.34, +1.60] | 84.1 | 89.7 | 36/40 | 780 | 9277 | PASS |
| n16p16 | 705/875 | 80.6 | +2.29 | [+1.37, +3.31] | 85.7 | 91.4 | 36/40 | 826 | 10847 | PASS |
| n16p16r30 | 742/875 | 84.8 | +6.51 | [+4.91, +8.23] | 89.4 | 94.2 | 36/40 | 2054 | 12946 | PASS |
| n16p20r30 | 748/875 | 85.5 | +7.20 | [+5.49, +8.91] | 89.7 | 94.7 | 36/40 | 2052 | 16030 | fail: prompt_fits_n_ctx |
| n16p8 | 685/875 | 78.3 | +0.00 | [+0.00, +0.00] | 83.3 | 88.9 | 36/40 | 788 | 7389 | fail: ci_lower_gt_0 |
| rr30 | 704/875 | 80.5 | +2.17 | [+0.69, +3.77] | 85.5 | 90.5 | 35/40 | 2167 | 7486 | fail: hard_negatives_not_lower,no_type_drop_gt_0.02,prompt_fits_n_ctx |
| rrfk120 | 685/875 | 78.3 | +0.00 | [+0.00, +0.00] | 83.3 | 88.9 | 36/40 | 818 | 7389 | fail: ci_lower_gt_0 |
| rrfk20 | 685/875 | 78.3 | +0.00 | [+0.00, +0.00] | 83.3 | 88.9 | 36/40 | 811 | 7389 | fail: ci_lower_gt_0 |
| slots1 | 676/875 | 77.3 | -1.03 | [-1.83, -0.23] | 83.0 | 88.6 | 36/40 | 1265 | 7389 | fail: ci_lower_gt_0,devpl_not_lower,no_type_drop_gt_0.02 |
| slots3 | 672/875 | 76.8 | -1.49 | [-2.51, -0.57] | 82.1 | 88.6 | 36/40 | 1265 | 7389 | fail: ci_lower_gt_0,devpl_not_lower,no_type_drop_gt_0.02 |
| strip | 693/875 | 79.2 | +0.91 | [-0.34, +2.29] | 84.3 | 89.4 | 36/40 | 1001 | 7389 | fail: ci_lower_gt_0,no_type_drop_gt_0.02 |
| yr1957 | 689/875 | 78.7 | +0.46 | [-0.34, +1.26] | 83.4 | 88.9 | 36/40 | 787 | 7389 | fail: ci_lower_gt_0 |
| yrw3 | 685/875 | 78.3 | +0.00 | [+0.00, +0.00] | 83.3 | 88.9 | 36/40 | 787 | 7389 | fail: ci_lower_gt_0 |

## devpl only (claim recall, paired bootstrap on the devpl claims)

| config | claims served | claim recall % | delta pp | CI95 pp |
|---|---|---|---|---|
| baseline | 67/110 | 60.9 |  | |
| bm25w15 | 68/110 | 61.8 | +0.91 | [-2.73, +4.55] |
| combo16 | 84/110 | 76.4 | +15.45 | [+8.18, +23.64] |
| combo16r30 | 84/110 | 76.4 | +15.45 | [+7.27, +23.64] |
| combo20r30 | 87/110 | 79.1 | +18.18 | [+10.00, +27.27] |
| dedup0 | 67/110 | 60.9 | +0.00 | [+0.00, +0.00] |
| densew15 | 63/110 | 57.3 | -3.64 | [-8.18, +0.00] |
| fu2 | 67/110 | 60.9 | +0.00 | [+0.00, +0.00] |
| fualways | 67/110 | 60.9 | +0.00 | [+0.00, +0.00] |
| n12p10 | 69/110 | 62.7 | +1.82 | [+0.00, +4.55] |
| n12p12 | 70/110 | 63.6 | +2.73 | [+0.00, +6.36] |
| n12p16 | 73/110 | 66.4 | +5.45 | [+1.82, +10.00] |
| n12p8 | 67/110 | 60.9 | +0.00 | [+0.00, +0.00] |
| n16p10 | 69/110 | 62.7 | +1.82 | [+0.00, +4.55] |
| n16p12 | 70/110 | 63.6 | +2.73 | [+0.00, +6.36] |
| n16p16 | 73/110 | 66.4 | +5.45 | [+1.82, +10.00] |
| n16p16r30 | 80/110 | 72.7 | +11.82 | [+6.36, +18.18] |
| n16p20r30 | 82/110 | 74.5 | +13.64 | [+7.27, +20.91] |
| n16p8 | 67/110 | 60.9 | +0.00 | [+0.00, +0.00] |
| rr30 | 71/110 | 64.5 | +3.64 | [-1.82, +9.09] |
| rrfk120 | 67/110 | 60.9 | +0.00 | [+0.00, +0.00] |
| rrfk20 | 67/110 | 60.9 | +0.00 | [+0.00, +0.00] |
| slots1 | 65/110 | 59.1 | -1.82 | [-5.45, +1.82] |
| slots3 | 63/110 | 57.3 | -3.64 | [-9.09, +0.91] |
| strip | 70/110 | 63.6 | +2.73 | [-0.91, +7.27] |
| yr1957 | 72/110 | 65.5 | +4.55 | [-0.91, +10.00] |
| yrw3 | 67/110 | 60.9 | +0.00 | [+0.00, +0.00] |

## Per set (claim recall %)

| config | dev_cases | devpl | heldout_v3 | heldout_v4 | heldout_v5a | heldout_v5b | heldout_v6a | heldout_v6b |
|---|---|---|---|---|---|---|---|---|
| baseline | 93.7 | 60.9 | 78.0 | 90.0 | 75.2 | 82.7 | 74.6 | 78.3 |
| bm25w15 | 93.7 | 61.8 | 78.0 | 91.7 | 73.2 | 81.3 | 68.6 | 75.7 |
| combo16 | 95.8 | 76.4 | 85.4 | 90.0 | 77.9 | 82.7 | 75.4 | 82.2 |
| combo16r30 | 94.7 | 76.4 | 95.1 | 93.3 | 82.6 | 84.7 | 83.9 | 85.5 |
| combo20r30 | 95.8 | 79.1 | 97.6 | 93.3 | 85.2 | 84.7 | 83.9 | 85.5 |
| dedup0 | 93.7 | 60.9 | 78.0 | 90.0 | 75.2 | 82.7 | 74.6 | 78.3 |
| densew15 | 91.6 | 57.3 | 78.0 | 86.7 | 76.5 | 76.7 | 74.6 | 77.0 |
| fu2 | 93.7 | 60.9 | 78.0 | 90.0 | 75.2 | 82.7 | 74.6 | 78.3 |
| fualways | 93.7 | 60.9 | 78.0 | 90.0 | 73.2 | 82.7 | 76.3 | 78.3 |
| n12p10 | 94.7 | 62.7 | 82.9 | 90.0 | 75.2 | 82.7 | 75.4 | 78.9 |
| n12p12 | 94.7 | 63.6 | 82.9 | 90.0 | 75.2 | 82.7 | 75.4 | 78.9 |
| n12p16 | 95.8 | 66.4 | 85.4 | 91.7 | 77.2 | 83.3 | 77.1 | 78.9 |
| n12p8 | 93.7 | 60.9 | 78.0 | 90.0 | 75.2 | 82.7 | 74.6 | 78.3 |
| n16p10 | 94.7 | 62.7 | 82.9 | 90.0 | 75.2 | 82.7 | 75.4 | 78.9 |
| n16p12 | 94.7 | 63.6 | 82.9 | 90.0 | 75.2 | 82.7 | 75.4 | 78.9 |
| n16p16 | 95.8 | 66.4 | 85.4 | 91.7 | 77.2 | 83.3 | 77.1 | 78.9 |
| n16p16r30 | 94.7 | 72.7 | 90.2 | 91.7 | 82.6 | 84.7 | 83.9 | 86.2 |
| n16p20r30 | 95.8 | 74.5 | 92.7 | 91.7 | 83.9 | 84.7 | 83.9 | 86.2 |
| n16p8 | 93.7 | 60.9 | 78.0 | 90.0 | 75.2 | 82.7 | 74.6 | 78.3 |
| rr30 | 92.6 | 64.5 | 82.9 | 91.7 | 74.5 | 82.7 | 83.1 | 80.9 |
| rrfk120 | 93.7 | 60.9 | 78.0 | 90.0 | 75.2 | 82.7 | 74.6 | 78.3 |
| rrfk20 | 93.7 | 60.9 | 78.0 | 90.0 | 75.2 | 82.7 | 74.6 | 78.3 |
| slots1 | 93.7 | 59.1 | 78.0 | 90.0 | 74.5 | 82.7 | 72.9 | 75.7 |
| slots3 | 93.7 | 57.3 | 78.0 | 90.0 | 73.8 | 82.7 | 72.9 | 75.0 |
| strip | 93.7 | 63.6 | 78.0 | 88.3 | 75.2 | 82.7 | 75.4 | 81.6 |
| yr1957 | 93.7 | 65.5 | 78.0 | 90.0 | 75.2 | 82.7 | 73.7 | 78.3 |
| yrw3 | 93.7 | 60.9 | 78.0 | 90.0 | 75.2 | 82.7 | 74.6 | 78.3 |

## Per question type (claim recall %, claims in brackets)

| config | answerable [12] | company [193] | fact_number [197] | follow_up [119] | opinion [189] | temporal_comparison [165] |
|---|---|---|---|---|---|---|
| baseline | 83.3 | 81.9 | 81.7 | 88.2 | 74.6 | 66.7 |
| bm25w15 | 83.3 | 80.8 | 80.2 | 84.0 | 72.0 | 67.3 |
| combo16 | 83.3 | 87.0 | 85.8 | 84.0 | 81.0 | 71.5 |
| combo16r30 | 83.3 | 90.7 | 85.3 | 90.8 | 83.1 | 78.8 |
| combo20r30 | 83.3 | 90.7 | 86.8 | 91.6 | 84.1 | 80.6 |
| dedup0 | 83.3 | 81.9 | 81.7 | 88.2 | 74.6 | 66.7 |
| densew15 | 75.0 | 81.3 | 80.2 | 83.2 | 72.5 | 65.5 |
| fu2 | 83.3 | 81.9 | 81.7 | 88.2 | 74.6 | 66.7 |
| fualways | 83.3 | 81.9 | 81.7 | 87.4 | 74.6 | 66.7 |
| n12p10 | 83.3 | 81.9 | 83.2 | 88.2 | 76.2 | 67.3 |
| n12p12 | 83.3 | 81.9 | 83.2 | 88.2 | 76.2 | 67.9 |
| n12p16 | 83.3 | 82.9 | 83.8 | 89.1 | 77.8 | 70.9 |
| n12p8 | 83.3 | 81.9 | 81.7 | 88.2 | 74.6 | 66.7 |
| n16p10 | 83.3 | 81.9 | 83.2 | 88.2 | 76.2 | 67.3 |
| n16p12 | 83.3 | 81.9 | 83.2 | 88.2 | 76.2 | 67.9 |
| n16p16 | 83.3 | 82.9 | 83.8 | 89.1 | 77.8 | 70.9 |
| n16p16r30 | 83.3 | 88.1 | 85.3 | 90.8 | 82.0 | 79.4 |
| n16p20r30 | 83.3 | 88.1 | 86.3 | 91.6 | 83.1 | 80.0 |
| n16p8 | 83.3 | 81.9 | 81.7 | 88.2 | 74.6 | 66.7 |
| rr30 | 83.3 | 85.0 | 83.8 | 85.7 | 78.3 | 69.7 |
| rrfk120 | 83.3 | 81.9 | 81.7 | 88.2 | 74.6 | 66.7 |
| rrfk20 | 83.3 | 81.9 | 81.7 | 88.2 | 74.6 | 66.7 |
| slots1 | 83.3 | 80.8 | 82.2 | 88.2 | 70.9 | 66.1 |
| slots3 | 83.3 | 78.8 | 81.7 | 84.9 | 73.5 | 66.1 |
| strip | 83.3 | 83.4 | 82.2 | 84.0 | 77.2 | 69.1 |
| yr1957 | 83.3 | 84.5 | 81.2 | 88.2 | 74.6 | 66.7 |
| yrw3 | 83.3 | 81.9 | 81.7 | 88.2 | 74.6 | 66.7 |

## Latency and prompt size

Latency = wall time of retrieval + rerank + context assembly plus the measured cost of cached model calls (machine shared with other jobs: indicative only). Tokens: Qwen2.5 tokenizer of the served FT model on the chat-templated request; n_ctx 8192, 768 tokens reserved for the answer.

| config | p50 ms | wall p50 ms | passages p50 | tokens p50 | tokens max | fits n_ctx-768 |
|---|---|---|---|---|---|---|
| baseline | 1303 | 1233 | 8 | 4386 | 7389 | True |
| bm25w15 | 848 | 181 | 8 | 4409 | 7389 | True |
| combo16 | 1264 | 96 | 16 | 7679 | 10847 | True |
| combo16r30 | 2146 | 284 | 16 | 7638 | 13332 | True |
| combo20r30 | 2101 | 89 | 20 | 9326 | 16407 | False |
| dedup0 | 779 | 44 | 8 | 4386 | 7389 | True |
| densew15 | 860 | 201 | 8 | 4415 | 7389 | True |
| fu2 | 1286 | 767 | 8 | 4386 | 7389 | True |
| fualways | 1303 | 877 | 8 | 4373 | 7389 | True |
| n12p10 | 759 | 42 | 10 | 5185 | 8243 | True |
| n12p12 | 762 | 48 | 12 | 5969 | 9277 | True |
| n12p16 | 857 | 162 | 16 | 7642 | 10847 | True |
| n12p8 | 759 | 46 | 8 | 4386 | 7389 | True |
| n16p10 | 780 | 45 | 10 | 5185 | 8243 | True |
| n16p12 | 780 | 45 | 12 | 5969 | 9277 | True |
| n16p16 | 826 | 46 | 16 | 7642 | 10847 | True |
| n16p16r30 | 2054 | 57 | 16 | 7635 | 12946 | True |
| n16p20r30 | 2052 | 55 | 20 | 9346 | 16030 | False |
| n16p8 | 788 | 51 | 8 | 4386 | 7389 | True |
| rr30 | 2167 | 486 | 8 | 4356 | 7486 | False |
| rrfk120 | 818 | 52 | 8 | 4386 | 7389 | True |
| rrfk20 | 811 | 49 | 8 | 4386 | 7389 | True |
| slots1 | 1265 | 478 | 8 | 4388 | 7389 | True |
| slots3 | 1265 | 478 | 8 | 4381 | 7389 | True |
| strip | 1001 | 239 | 8 | 4393 | 7389 | True |
| yr1957 | 787 | 86 | 8 | 4404 | 7389 | True |
| yrw3 | 787 | 79 | 8 | 4386 | 7389 | True |

## Cause of the unserved claims (miss analysis, claims)

| cause | baseline (missed 190) |
|---|---|
| lexical_gap | 17 |
| rerank_demotion | 49 |
| candidate_depth_cut | 55 |
| cap_dedupe_merge_drop | 44 |
| period_year_filter | 13 |
| multipart_split_error | 10 |
| followup_context_missing | 2 |

Per set, first configuration: heldout_v3: 9; heldout_v4: 6; heldout_v5a: 37; heldout_v5b: 26; heldout_v6a: 30; heldout_v6b: 33; dev_cases: 6; devpl: 43

## Overrides

- baseline: {'LLM_CONTEXT_PASSAGES': '8'}
- bm25w15: {'LLM_CONTEXT_PASSAGES': '8', 'HYBRID_BM25_WEIGHT': '1.5'}
- combo16: {'LLM_CONTEXT_PASSAGES': '16', 'LLM_N_CTX': '16384', 'RERANK_CANDIDATES': '16', 'RETRIEVAL_YEAR_FILTER_MIN': '1957', 'RETRIEVAL_QUERY_STRIP': '1', 'MAX_NEW_TOKENS': '1024'}
- combo16r30: {'LLM_CONTEXT_PASSAGES': '16', 'LLM_N_CTX': '16384', 'RERANK_CANDIDATES': '30', 'RETRIEVAL_YEAR_FILTER_MIN': '1957', 'RETRIEVAL_QUERY_STRIP': '1', 'MAX_NEW_TOKENS': '1024'}
- combo20r30: {'LLM_CONTEXT_PASSAGES': '20', 'LLM_N_CTX': '16384', 'RERANK_CANDIDATES': '30', 'RETRIEVAL_YEAR_FILTER_MIN': '1957', 'RETRIEVAL_QUERY_STRIP': '1', 'MAX_NEW_TOKENS': '1024'}
- dedup0: {'LLM_CONTEXT_PASSAGES': '8', 'RETRIEVAL_DEDUP': '0'}
- densew15: {'LLM_CONTEXT_PASSAGES': '8', 'HYBRID_VECTOR_WEIGHT': '1.5'}
- fu2: {'LLM_CONTEXT_PASSAGES': '8', 'RETRIEVAL_FOLLOWUP_TURNS': '2'}
- fualways: {'LLM_CONTEXT_PASSAGES': '8', 'RETRIEVAL_FOLLOWUP_ALWAYS': '1'}
- n12p10: {'LLM_CONTEXT_PASSAGES': '10', 'LLM_N_CTX': '12288', 'MAX_NEW_TOKENS': '1024'}
- n12p12: {'LLM_CONTEXT_PASSAGES': '12', 'LLM_N_CTX': '12288', 'MAX_NEW_TOKENS': '1024'}
- n12p16: {'LLM_CONTEXT_PASSAGES': '16', 'LLM_N_CTX': '12288', 'RERANK_CANDIDATES': '16', 'MAX_NEW_TOKENS': '1024'}
- n12p8: {'LLM_CONTEXT_PASSAGES': '8', 'LLM_N_CTX': '12288', 'MAX_NEW_TOKENS': '1024'}
- n16p10: {'LLM_CONTEXT_PASSAGES': '10', 'LLM_N_CTX': '16384', 'MAX_NEW_TOKENS': '1024'}
- n16p12: {'LLM_CONTEXT_PASSAGES': '12', 'LLM_N_CTX': '16384', 'MAX_NEW_TOKENS': '1024'}
- n16p16: {'LLM_CONTEXT_PASSAGES': '16', 'LLM_N_CTX': '16384', 'RERANK_CANDIDATES': '16', 'MAX_NEW_TOKENS': '1024'}
- n16p16r30: {'LLM_CONTEXT_PASSAGES': '16', 'LLM_N_CTX': '16384', 'RERANK_CANDIDATES': '30', 'MAX_NEW_TOKENS': '1024'}
- n16p20r30: {'LLM_CONTEXT_PASSAGES': '20', 'LLM_N_CTX': '16384', 'RERANK_CANDIDATES': '30', 'MAX_NEW_TOKENS': '1024'}
- n16p8: {'LLM_CONTEXT_PASSAGES': '8', 'LLM_N_CTX': '16384', 'MAX_NEW_TOKENS': '1024'}
- rr30: {'LLM_CONTEXT_PASSAGES': '8', 'RERANK_CANDIDATES': '30'}
- rrfk120: {'LLM_CONTEXT_PASSAGES': '8', 'RRF_K': '120'}
- rrfk20: {'LLM_CONTEXT_PASSAGES': '8', 'RRF_K': '20'}
- slots1: {'LLM_CONTEXT_PASSAGES': '8', 'RETRIEVAL_PART_SLOTS': '1'}
- slots3: {'LLM_CONTEXT_PASSAGES': '8', 'RETRIEVAL_PART_SLOTS': '3'}
- strip: {'LLM_CONTEXT_PASSAGES': '8', 'RETRIEVAL_QUERY_STRIP': '1'}
- yr1957: {'LLM_CONTEXT_PASSAGES': '8', 'RETRIEVAL_YEAR_FILTER_MIN': '1957'}
- yrw3: {'LLM_CONTEXT_PASSAGES': '8', 'RETRIEVAL_YEAR_FILTER_WEIGHT': '3'}

## Caveats and candidates not measured

- Pool: 359 answerable cases / 875 claims (heldout_v3..v6, dev_cases, devpl 37 cases / 110 claims) served on the active corpus v5_pl (chunks_v5_pl + faiss_v5_pl, MPS embedder/reranker) with the base commit's defaults as baseline (multipart on, 8 passages, n_ctx 8192, rerank depth 15). Stub LLM, nothing generated, no 7B/14B model loaded.
- All pool sets except devpl are spent (used for selection before); gains are development estimates, not blind. devpl was never used to tune before this run but is part of the pooled selection here.
- Miss analysis (baseline, 190 unserved claims): see the cause table and `misses_baseline.csv`. `cap_dedupe_merge_drop` (44) are claims whose gold chunk text IS in the prompt as neighbour text of another served passage (`gold_text_in_prompt=True`) but whose id is not a served passage id, so the id metric counts them as misses; the merged-ids column shows that coverage. Ranks are for the best gold id of the claim over the whole-question and per-part queries, unfiltered, 200 deep (BM25, dense, RRF of both); `rerank_rank_top50` re-ranks the fused top 50 with the cross-encoder (gold-year scoped when the question names a year/period).
- After-state cause table: `causes_<config>.json` exists for the baseline only (the miss analysis takes ~12 min on the shared machine); rerun `served_context_recall.py misses --out DIR --config NAME[:K=V,...]` for a final config.
- Not changed/measured: RETRIEVAL_RERANK_FETCH=1 with RERANK_CANDIDATES>30 (fetch_k is 30 by default, so depths above 30 need it; rr50 not completed within the time box), ANSWER_CONTEXT_NEIGHBORS, neighbour anchors, MMR.
- RRF k 20/120, RETRIEVAL_DEDUP=0, followup turns 2/always, RETRIEVAL_YEAR_FILTER_WEIGHT=3 changed nothing or nothing significant; BM25 or dense weight 1.5 and part slots 1/3 lowered recall. Rerank order is decided by the cross-encoder, so fusion constants only matter through the candidate set.
- Determinism section: baseline vs stored heldout_v6 ids differs for 41/79 cases because the stored runs used corpus v3 and the active corpus now also holds the partnership letters (bpl ids occur in old answers); not a regression signal.
- Latency: p50 is wall time plus the measured cost of cached model calls (shared, swapping machine): indicative only. Deeper rerank costs ~2x (30 vs 15 cross-encoder pairs per query).
- Prompt size: with n_ctx 16384 and 1024 tokens reserved for the answer the prompt must be <= 15360 tokens. 20 passages reached 16030 tokens with the 4 chars/token budget (`LLM_CHARS_PER_TOKEN`), so 20 passages need a stricter value (~3.2) and were not adopted; 16 passages peak at 12946 tokens. KV cache (f16, llama.cpp formula, GGUF header of the served model, no model load): 56 KiB/token = 448 MiB at 8192, 672 MiB at 12288, 896 MiB at 16384 (+448 MiB vs today). See `kv_cache.json`.
- A longer prompt (16 passages ~ 8-13k tokens vs ~5-7k) slows prefill roughly in proportion and may change 7B behaviour (lost-in-the-middle, more refusals/hallucination); this run only measures whether the gold passage reaches the prompt. An answer-level check with the 7B is still required before relying on it.
- Defaults applied in config.py: LLM_N_CTX 16384, LLM_CONTEXT_PASSAGES 16 (7B), RERANK_CANDIDATES 30, (each still overridable by its environment variable; `served_context_recall.py` pins the old values as BASELINE_ENV). Two tests in tests/test_ft_v4_generator.py fail afterwards: scripts/ft/round4.py (outside this unit's Files) asserts `LLM_CONTEXT_PASSAGES == SERVING_PASSAGES (8)`; its constant must become 16 for the generator-parity guard to match serving.
- Target not reached: pooled 85.5% (devpl 76.4%) against the 90% goal; remaining misses are mostly rerank/depth (gold ranked 31-200 or demoted) and neighbour-only coverage; 20 passages (86.5%, devpl 79.1%) need a stricter LLM_CHARS_PER_TOKEN to fit 16384.
- RETRIEVAL_QUERY_STRIP=1 and RETRIEVAL_YEAR_FILTER_MIN=1957 stay OFF: alone each has a CI95 lower bound < 0 (strip +0.91pp [-0.34, +2.29]; yr1957 +0.46pp [-0.34, +1.26], devpl 65.5%). Together with 16 passages/rerank 30 (combo16r30: 85.5% pooled, devpl 76.4%) they add +0.7pp pooled / +3.7pp devpl over n16p16r30 (84.8% / 72.7%), not tested separately; they are the next candidates for the default after a blind check.

## Rule and decision

A candidate becomes the default only when its pooled claim-level served-recall gain has a paired per-claim bootstrap (10,000 draws, seed 0) CI95 lower bound > 0, hard_negatives_v4 passes are not lower, no question type drops by more than 0.02, and the longest prompt leaves >= 768 tokens of the 8192-token window. Each setting is an environment variable read at call time (`config.retrieval_setting`, `config.llm_context_passages`).

Final configuration: `n16p16r30` {'LLM_CONTEXT_PASSAGES': '16', 'LLM_N_CTX': '16384', 'RERANK_CANDIDATES': '30', 'MAX_NEW_TOKENS': '1024'}.

Note: scripts that set `LLM_CONTEXT_PASSAGES=8` explicitly (Artifacts `run_arm.sh`) override the new default and must be changed to the new value to measure it.

DEFAULT: CHANGED LLM_CONTEXT_PASSAGES=16 LLM_N_CTX=16384 RERANK_CANDIDATES=30
