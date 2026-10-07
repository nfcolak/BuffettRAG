#!/bin/bash
# Cloud dev-set CPU check: 4 arms on the 80 dev cases, one after another, each a fresh process.
# BM25, temperature 0, CPU only (LLM_GPU_LAYERS=0). Dev cases only; no frozen held-out set is used.
# Arm 1 runs the extractive engine from a checkout of main (MAIN_CHECKOUT, default ../main_checkout),
# whose models/ and data/processed/ must match this checkout's.
cd "$(dirname "$0")/../../.." || exit 99
OUT=data/evaluation/cloud_dev_check_v1
CASES=data/evaluation/grounded_dev_v1/confirm/dev_cases.json
MAIN_CHECKOUT=${MAIN_CHECKOUT:-../main_checkout}
PYBIN=${PYBIN:-.venv/bin/python}
PYBIN=$(cd "$(dirname "$PYBIN")" && pwd)/$(basename "$PYBIN")
M=$(pwd)/models
BASE=$M/qwen2.5-1.5b-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
FTR2=$M/ft/round2/buffett-qwen2.5-1.5b-ft-r2-q4_k_m.gguf
BASE7B=${BASE7B:-$M/qwen2.5-7b-gguf/qwen2.5-7b-instruct-q4_k_m.gguf}
FT7B=${FT7B:-$M/ft/7b_r1/buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf}
mkdir -p "$OUT/logs"
export PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 LLM_GPU_LAYERS=${LLM_GPU_LAYERS:-0}
unset PYTHONPATH
COMMON="--cases $CASES --retrieval bm25 --temperature 0 --save-raw"
run_arm() {
  local name=$1 dir=$2; shift 2
  local t0=$(date +%s)
  echo "START $name $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/progress.txt"
  (cd "$dir" && "$PYBIN" scripts/eval/run_live_benchmark.py $COMMON "$@" --output "$OLDPWD/$OUT/$name.json") \
    > "$OUT/logs/$name.log" 2>&1
  echo "END $name exit=$? seconds=$(( $(date +%s) - t0 ))" >> "$OUT/progress.txt"
}
# --model-path alone is not enough: LlamaCppProvider binds config.LLM_MODEL_PATH as a default
# argument at import, so the model must also be selected through the environment.
ARMS=${ARMS:-"extractive_main extractive_lead llama_ftr2_guard llama_base"}
want() { [[ " $ARMS " == *" $1 "* ]]; }
want extractive_main && run_arm extractive_main "$MAIN_CHECKOUT" --provider local
want extractive_lead && run_arm extractive_lead . --provider local
want llama_ftr2_guard && LLM_MODEL_PATH="$FTR2" run_arm llama_ftr2_guard . --provider llama --model-path "$FTR2"
want llama_base && LLM_MODEL_PATH="$BASE" run_arm llama_base . --provider llama --model-path "$BASE"
# 7B arms run only when named in ARMS (not in the default list); pass LLM_GPU_LAYERS=-1 for Metal.
want llama7b_base && LLM_MODEL_PATH="$BASE7B" run_arm llama7b_base . --provider llama --model-path "$BASE7B"
want llama7b_ft && LLM_MODEL_PATH="$FT7B" run_arm llama7b_ft . --provider llama --model-path "$FT7B"
echo "ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/progress.txt"
