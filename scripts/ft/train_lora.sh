#!/usr/bin/env bash
# Usage: [DATA_DIR=data/ft_v2] scripts/ft/train_lora.sh <round> [iters]   (DATA_DIR default: data/ft = round 1 data)
# Checkpointed MLX LoRA training; re-running resumes from adapters/adapters.safetensors.
set -u
ROUND="${1:?usage: train_lora.sh <round> [iters]}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/.venv/bin/python}"
BATCH=2
OUT="models/ft/round${ROUND}"
mkdir -p "$OUT/adapters"
DATA_DIR="${DATA_DIR:-data/ft}"
TRAIN_N=$(wc -l < "$DATA_DIR/train.jsonl" | tr -d ' ')
TOTAL="${2:-$(( (2 * TRAIN_N + BATCH - 1) / BATCH ))}"   # 2 epochs
PROGRESS="$OUT/progress.json"

DONE=0
RESUME=()
if [ -f "$OUT/adapters/adapters.safetensors" ]; then
  # completed iters: progress.json, else the newest absolute checkpoint copy (abs_<iter>.safetensors)
  DONE=$(env -u PYTHONPATH "$PY" - "$PROGRESS" "$OUT/adapters" <<'EOF2'
import glob, json, os, re, sys
done = 0
if os.path.exists(sys.argv[1]):
    done = json.load(open(sys.argv[1])).get('completed_iters', 0)
absolute = glob.glob(sys.argv[2] + '/abs_*.safetensors')
for f in absolute:
    done = max(done, int(re.search(r'abs_(\d+)', f).group(1)))
if not done and not absolute:
    # first run of this round (never resumed): mlx_lm checkpoint numbers are already absolute
    for f in glob.glob(sys.argv[2] + '/0*_adapters.safetensors'):
        done = max(done, int(re.match(r'(\d+)_', os.path.basename(f)).group(1)))
print(done)
EOF2
)
  [ "$DONE" -gt 0 ] && RESUME=(--resume-adapter-file "$OUT/adapters/adapters.safetensors")
fi
REMAIN=$(( TOTAL - DONE ))
if [ "$REMAIN" -le 0 ]; then echo "nothing to do: $DONE/$TOTAL iters complete"; echo 0 > "$OUT/train_exit.txt"; exit 0; fi
rm -f "$OUT/train_exit.txt"
echo "round=$ROUND total=$TOTAL done=$DONE remaining=$REMAIN train_n=$TRAIN_N $(date)" | tee -a "$OUT/train.log"

(
  env -u PYTHONPATH PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 "$PY" -m mlx_lm lora \
    -c scripts/ft/lora_config.yaml --data "$DATA_DIR" --iters "$REMAIN" --adapter-path "$OUT/adapters" ${RESUME[@]+"${RESUME[@]}"} 2>&1
  echo "EXIT:$?"
) | tee -a "$OUT/train.log" | env -u PYTHONPATH "$PY" scripts/ft/track_progress.py "$PROGRESS" "$TOTAL" "$DONE" "$OUT/adapters"
RC=$(grep -a -o 'EXIT:[0-9]*' "$OUT/train.log" | tail -1 | cut -d: -f2)
RC="${RC:-1}"
echo "$RC" > "$OUT/train_exit.txt"
# final state: a clean exit means all iters done
if [ "$RC" = 0 ]; then echo "{\"total_iters\": $TOTAL, \"completed_iters\": $TOTAL, \"last_logged_iter\": $TOTAL}" > "$PROGRESS"; fi
exit "$RC"
