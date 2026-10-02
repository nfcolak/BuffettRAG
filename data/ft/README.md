# BuffettRAG offline MLX fine-tuning data
Teacher: local Qwen2.5-7B-Instruct, MLX 4-bit, Apache-2.0; models/ is ignored; no weights or external API calls.
Counts: train=112, valid=12; types={"train": {"answerable": 99, "unanswerable": 13}, "valid": {"answerable": 12}}; train refusal share=11.6%.
Sampling: seed 13; 900 source passages stratified across all 48 letters; two teacher questions/source; 200 unavailable-fact templates paraphrased locally.
Retrieval: repository BM25 top 5, explicit query-year filtering, anchor + immediate neighbors, max 1800 chars; build_cited_prompt serving system/user split.
Filters: zero blocked claims, every sentence validly cited, source cited; refusal line is a stop sequence, not a rewritten target (replacement only with <2 shared non-stopwords/passage).
Negatives: up to 150 source-removed distractors, kept only when assess_evidence fails; refuse targets capped at 25% per split.
Leakage: both frozen sets' gold/relevant IDs and immediate neighbors excluded from sampling, retrieval and context; token-Jaccard >=0.5 questions dropped.
Splits: seed-selected source-year groups (~90/10); even retrieved/neighbor IDs cannot occur in both splits; fixed source hints are teacher-only, never training turns.
Resume: append question/example audit logs, skip completed IDs; rejected targets retried with deterministic independent seeds, same fixed prompt; atomic chat snapshots.
From /Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/.worktrees/ft-data: V=/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/.venv/bin/python
Questions: env -u PYTHONPATH PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 "$V" scripts/ft/make_questions.py --resume --limit 100
Pilot: env -u PYTHONPATH PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 "$V" scripts/ft/make_examples.py --resume --pilot 20
Detached (also completes remaining questions): nohup caffeinate -i env -u PYTHONPATH PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 "$V" -u scripts/ft/make_examples.py --resume > data/ft/gen.log 2>&1 < /dev/null & printf '%s\n' "$!" > data/ft/gen.pid
Leakage check (retry if racing a snapshot): env -u PYTHONPATH PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 "$V" scripts/ft/check_leakage.py
