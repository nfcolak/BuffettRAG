# Cloud dev-set CPU check (v1)

Dev cases only: `data/evaluation/grounded_dev_v1/confirm/dev_cases.json` (80 cases, sha256 `fae69723daaacea8...`; the union of the four dev sets heldout_v1, heldout_v2, answer_benchmark_v3 and adversarial_dev that the file marks `dev_only`). No frozen held-out set (heldout_v2/ or heldout_v3/ folders) was read, run or scored.

BM25 retrieval, temperature 0, CPU only (`LLM_GPU_LAYERS=0`, 18 vCPU), one arm after another, fresh process each, `--save-raw`. All cases run for every arm.

Engineering read-out for the owner, not a statistical test; no default setting was changed on these numbers.

| arm | strict supported claims met/required | NLI accepted | lexical accepted | correct refusals | unexpected refusals | citation-only answers caught by guard | provider failures | p50 latency ms | p95 latency ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 extractive, main code (old) | 46/95 | 36/80 | 38/80 | 4/9 | 1/71 | n/a | 0 | 165.9 | 227.4 |
| 2 extractive, task A (lead + <=2 supporting) | 36/95 | 30/80 | 31/80 | 4/9 | 1/71 | n/a | 0 | 179.6 | 246.7 |
| 3 info: task A with the original unstemmed ranking | 36/95 | 27/80 | 29/80 | 4/9 | 2/71 | n/a | 0 | 166.1 | 241.2 |
| 4 llama FT-r2 1.5B, task B guard | 17/95 | 18/80 | 20/80 | 7/9 | 18/71 | 0 | 0 | 34361.3 | 47752.5 |
| 5 llama base Qwen2.5-1.5B | 5/95 | 10/80 | 11/80 | 7/9 | 55/71 | 0 | 0 | 35789.6 | 59975.7 |
| 6 llama base Qwen2.5-7B | 5/95 | 13/80 | 15/80 | 8/9 | 43/71 | 0 | 0 | 5572.7 | 9712.4 |
| 7 llama FT-r1 7B (ft_v3, LoRA) | 42/95 | 31/80 | 44/80 | 8/9 | 13/71 | 0 | 0 | 5117.3 | 8233.7 |

Definitions. Strict supported claim: a gold claim met by an answer sentence that matches lexically and whose cited gold hit passes the deterministic verifier (`src.evaluation.supported_claims`), scored against the client context rebuilt with a stub provider (row passage ids checked: arm 1 0 mismatches, arm 2 0 mismatches, arm 3 0 mismatches, arm 4 0 mismatches, arm 5 0 mismatches, arm 6 0 mismatches, arm 7 0 mismatches). NLI accepted: `scripts/eval/rescore_nli.py` with `models/nli-deberta-v3-base` (entailment >= 0.5). Correct refusals are over the unanswerable cases; unexpected refusals are refusal-line answers on answerable cases. Citation-only: a raw model answer with fewer than three words once `[n]` markers and punctuation are removed (`ask_flow.is_citation_only`); the guard replaces it with the extractive answer. Latency is per case end to end on this CPU, including the first (cold) case.

Citation-only qids: arm 4: none; arm 5: none; arm 6: none; arm 7: none
