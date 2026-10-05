# Blind heldout_v2 comparison — round 3 included

All systems used the frozen paragraph-v3 corpus, BM25 retrieval, temperature 0, and saved raw answers. The local provider is extractive, not a generative LLM. Acceptance denominators are scored answers; claims denominators are required claims. Citation support is NLI-supported cited sentences divided by all cited sentences in answerable cases; N/A means no cited sentences.

| System | Lexical accepted | NLI accepted | Lexical claims met | NLI claims met | Citation support rate | Correct refusals | Unexpected refusals | Provider failures | Mean latency ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| extractive local | 11/40 | 13/40 | 12/45 | 16/45 | 0.980 (198/202) | 2/4 | 1 | 0 | 50.101 |
| base llama | 4/40 | 4/40 | 0/45 | 0/45 | N/A (0/0) | 4/4 | 30 | 0 | 4951.930 |
| round 1 | 3/40 | 4/40 | 0/45 | 2/45 | 0.833 (15/18) | 3/4 | 13 | 0 | 1964.117 |
| round 2 | 10/40 | 9/40 | 6/45 | 6/45 | 0.944 (34/36) | 4/4 | 13 | 0 | 1003.319 |
| round 3 | 6/40 | 7/40 | 2/45 | 4/45 | 0.840 (21/25) | 4/4 | 8 | 0 | 1895.402 |

## Same systems on heldout_v1

These are saved heldout_v1 results, not reruns. The round-3 result was copied unchanged from the existing round-3 artifact and rescored with the same NLI scorer.

| System | Lexical accepted | NLI accepted |
| --- | --- | --- |
| extractive local | 18/24 | 18/24 |
| base llama | 3/24 | 3/24 |
| round 1 | 5/24 | 5/24 |
| round 2 | 10/24 | 9/24 |
| round 3 | 6/24 | 4/24 |

Blind heldout_v2 cases SHA256: 642501e574649d73ebada40740137d52030d051c66f4827c19144a0d4cf5b609; each arm ran exactly once, with no reruns or code/config changes between arms.

NLI rescoring used the offline DeBERTa model on MPS with the existing fixed entailment threshold (0.5) and window rule (400 tokens). Mean latency covers each full in-process /ask case, including retrieval and postprocessing, but excludes provider/retriever setup. Latency was measured while other agents shared the machine and is not an isolated speed comparison.
