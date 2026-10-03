| System | Accepted | Required claims met | Correct refusals | Unexpected refusals | Provider failures | Mean latency ms |
| --- | --- | --- | --- | --- | --- | --- |
| local | 18/24 | 23/31 | 2/2 | 0 | 0 | 43.578 |
| base llama | 3/24 | 2/31 | 2/2 | 18 | 0 | 1171.486 |
| round 1 | 5/24 | 4/31 | 2/2 | 7 | 0 | 652.433 |
| round 2 | 10/24 | 10/31 | 2/2 | 3 | 0 | 1139.950 |

local: replaces heldout_v1_local_bm25_fixA3.json (accepted 14/24 -> 18/24; delta +4); heldout_v1_local_bm25_fixB.json (accepted 17/24 -> 18/24; delta +1).
base llama: replaces heldout_v1_llama_base_bm25.json (accepted 3/24 -> 3/24; delta +0).
round 1: replaces heldout_v1_llama_ft_r1_bm25.json (accepted 5/24 -> 5/24; delta +0).
round 2: replaces heldout_v1_llama_ft_r2_bm25.json (accepted 10/24 -> 10/24; delta +0).
