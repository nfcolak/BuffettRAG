#!/usr/bin/env bash
# Usage: scripts/ft/train_lora.sh <round> [iters]
# Checkpointed MLX LoRA training; re-running resumes from adapters/adapters.safetensors.
set -u
ROUND="${1:?usage: train_lora.sh <round> [iters]}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/.venv/bin/python}"
BATCH=2
OUT="models/ft/round${ROUND}"
mkdir -p "$OUT/adapters"
TRAIN_N=$(wc -l < data/ft/train.jsonl | tr -d ' ')
TOTAL="${2:-$(( (2 * TRAIN_N + BATCH - 1) / BATCH ))}"   # 2 epochs
PROGRESS="$OUT/progress.json"

DONE=0
RESUME=()
if [ -f "$OUT/adapters/adapters.safetensors" ] && [ -f "$PROGRESS" ]; then
  DONE=$(env -u PYTHONPATH "$PY" -c "import json;print(json.load(open('$PROGRESS')).get('completed_iters',0))")
  RESUME=(--resume-adapter-file "$OUT/adapters/adapters.safetensors")
fi
REMAIN=$(( TOTAL - DONE ))
if [ "$REMAIN" -le 0 ]; then echo "nothing to do: $DONE/$TOTAL iters complete"; echo 0 > "$OUT/train_exit.txt"; exit 0; fi
rm -f "$OUT/train_exit.txt"
echo "round=$ROUND total=$TOTAL done=$DONE remaining=$REMAIN train_n=$TRAIN_N $(date)" | tee -a "$OUT/train.log"

(
  env -u PYTHONPATH PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 "$PY" -m mlx_lm lora \
    -c scripts/ft/lora_config.yaml --iters "$REMAIN" --adapter-path "$OUT/adapters" ${RESUME[@]+"${RESUME[@]}"} 2>&1
  echo "EXIT:$?"
) | tee -a "$OUT/train.log" | env -u PYTHONPATH "$PY" scripts/ft/track_progress.py "$PROGRESS" "$TOTAL" "$DONE"
RC=$(grep -o 'EXIT:[0-9]*' "$OUT/train.log" | tail -1 | cut -d: -f2)
RC="${RC:-1}"
echo "$RC" > "$OUT/train_exit.txt"
# final state: a clean exit means all iters done
if [ "$RC" = 0 ]; then echo "{\"total_iters\": $TOTAL, \"completed_iters\": $TOTAL, \"last_logged_iter\": $TOTAL}" > "$PROGRESS"; fi
exit "$RC"
