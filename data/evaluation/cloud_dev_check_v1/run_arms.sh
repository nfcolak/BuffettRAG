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
mkdir -p "$OUT/logs"
export PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 LLM_GPU_LAYERS=0
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
run_arm extractive_main "$MAIN_CHECKOUT" --provider local
run_arm extractive_lead . --provider local
run_arm llama_ftr2_guard . --provider llama --model-path "$FTR2"
run_arm llama_base . --provider llama --model-path "$BASE"
echo "ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/progress.txt"
