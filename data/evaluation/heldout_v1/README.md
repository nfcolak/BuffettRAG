Held-out V1: 24 new questions covering 1977–2024; no development question or gold-passage-ID overlap.
Frozen on 2026-10-02, with corpus SHA256 and per-case source verification notes; do not tune on this set.
Includes 3 temporal comparisons, 2 contextual follow-ups, and 2 unanswerable questions requiring the exact refusal line.
The runner calls the actual in-process /ask path; no HTTP server or request-level LLM overrides are used.
Offline local/BM25 result: 24 scored, 12 accepted, 12 rejected; 17/31 required claims met; 1/2 correct refusals.
All 25 distinct gold/relevant IDs exist; 0 missing fixture IDs and 0 provider failures; output: heldout_v1_local_bm25.json.
This is an extractive smoke benchmark, NOT live LLM quality; claim-term/citation scoring is deterministic, not semantic judging.
Embedded-LLM run needs no API keys; provider failures remain unscored and return exit 1.
From the repository root (the GGUF model must exist under models/; hybrid also needs embedding/reranker weights and the vector backend):
env -u PYTHONPATH PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 python3 scripts/run_live_benchmark.py --provider llama --retrieval bm25 --temperature 0 --cases data/evaluation/heldout_v1/answer_benchmark_heldout_v1.json --output data/evaluation/heldout_v1/heldout_v1_llama_base_bm25.json
# hybrid: use --retrieval hybrid
