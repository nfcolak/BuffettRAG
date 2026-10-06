#!/bin/bash
# U7: the 7 planned arms on heldout_v3, ONE AFTER ANOTHER, each a fresh process. No code/setting change.
cd "$(dirname "$0")/../../../.." || exit 99
FINAL=data/evaluation/heldout_v3/final
CASES=data/evaluation/heldout_v3/answer_benchmark_heldout_v3.json
PYBIN=/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/.venv/bin/python
M=/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/models
BASE=$M/qwen2.5-1.5b-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
FTR2=$M/ft/round2/buffett-qwen2.5-1.5b-ft-r2-q4_k_m.gguf
MLX7=$M/teacher-qwen2.5-7b-mlx4
mkdir -p "$FINAL/logs"
export PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1
unset PYTHONPATH
COMMON="--cases $CASES --retrieval bm25 --temperature 0 --save-raw"
any_fail=0
run_arm() {
  local name=$1; shift
  local t0 t1 rc
  t0=$(date +%s)
  echo "START $name $(date -u +%Y-%m-%dT%H:%M:%SZ) load=$(uptime | sed 's/.*load averages*: //')" >> "$FINAL/progress.txt"
  "$PYBIN" scripts/eval/run_live_benchmark.py $COMMON "$@" --output "$FINAL/$name.json" > "$FINAL/logs/$name.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "END $name exit=$rc seconds=$((t1-t0)) $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$FINAL/progress.txt"
  [ $rc -ne 0 ] && any_fail=1
}
run_arm local_extractive --provider local
run_arm llama_base --provider llama --model-path "$BASE"
run_arm llama_ftr2 --provider llama --model-path "$FTR2"
run_arm mlx7b_oldpath --provider mlx --model-path "$MLX7"
run_arm grounded_mlx7b --provider grounded --composer mlx --model-path "$MLX7"
run_arm grounded_template --provider grounded --composer template
run_arm grounded_llama_ftr2 --provider grounded --composer llama --model-path "$FTR2"
echo "$any_fail" > "$FINAL/full_exit.txt"
echo "ALL_DONE any_fail=$any_fail $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$FINAL/progress.txt"
