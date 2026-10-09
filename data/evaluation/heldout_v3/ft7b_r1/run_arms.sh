#!/bin/bash
# heldout_v3 blind run: arm 8 = 7B base, arm 9 = 7B FT-r1; each ONCE, fresh process, BM25, t=0, Metal. No code change.
R=/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG
OUT=/Users/necatifurkancolak/AI-Workplace/Artifacts/BuffettRAG/heldout7b/results
cd "$R" || exit 99
CASES=data/evaluation/heldout_v3/answer_benchmark_heldout_v3.json
PYBIN=$R/.venv/bin/python
BASE7B=$R/models/qwen2.5-7b-gguf/qwen2.5-7b-instruct-q4_k_m.gguf
FT7B=$R/models/ft/7b_r1/buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf
export PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 LLM_GPU_LAYERS=-1
unset PYTHONPATH
COMMON="--cases $CASES --retrieval bm25 --temperature 0 --save-raw"
any_fail=0
run_arm() {
  local name=$1 model=$2; shift 2
  local t0=$(date +%s)
  echo "START $name $(date -u +%Y-%m-%dT%H:%M:%SZ) load=$(uptime | sed 's/.*load averages*: //')" >> "$OUT/progress.txt"
  LLM_MODEL_PATH="$model" "$PYBIN" scripts/eval/run_live_benchmark.py $COMMON --provider llama --model-path "$model" \
    --output "$OUT/$name.json" > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  echo "END $name exit=$rc seconds=$(( $(date +%s) - t0 )) $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/progress.txt"
  [ $rc -ne 0 ] && any_fail=1
}
run_arm llama7b_base "$BASE7B"
run_arm llama7b_ft "$FT7B"
echo "ALL_DONE any_fail=$any_fail $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/progress.txt"
