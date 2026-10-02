#!/usr/bin/env bash
# Usage: scripts/ft/run_round.sh <round> <target_examples> <max_gen_minutes>
# One resumable pipeline: generate (data/ft_v2) -> split -> LoRA train -> export + eval -> comparison.
# Every stage skips itself when its marker exists, so re-running after an interruption continues where it stopped:
#   nohup caffeinate -i bash scripts/ft/run_round.sh 2 900 150 >> models/ft/round2/pipeline.log 2>&1 &
# To redo a stage, delete its marker in models/ft/round<N>/stages/. pipeline_exit.txt: 0 only after the comparison is written.
set -uo pipefail
ROUND="${1:?usage: run_round.sh <round> <target_examples> <max_gen_minutes>}"
TARGET="${2:?target examples}"
MAXMIN="${3:?max generation minutes}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/.venv/bin/python}"
R1=/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/.worktrees/ft-r1b/data/evaluation/heldout_v1/heldout_v1_llama_ft_r1_bm25.json
DATA=data/ft_v2
OUT="models/ft/round${ROUND}"
STAGES="$OUT/stages"
HELD=data/evaluation/heldout_v1
RESULT="$HELD/heldout_v1_llama_ft_r${ROUND}_bm25.json"
mkdir -p "$STAGES"
rm -f "$OUT/pipeline_exit.txt"
run() { env -u PYTHONPATH PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 "$PY" "$@"; }
log() { echo "[$(date '+%F %T')] $*"; }
fail() { log "FAILED stage=$1 rc=$2"; echo "$2" > "$OUT/pipeline_exit.txt"; echo "$1" > "$OUT/pipeline_failed_stage.txt"; exit "$2"; }

# (a) generation: stops at TARGET kept examples or when MAXMIN minutes of teacher time are used (cumulative across restarts)
if [ ! -f "$STAGES/a_generate.done" ]; then
  USED=$(run - "$DATA" <<'PYEOF'
import json, sys
total = 0.0
for name in ('example_attempts.jsonl', 'question_attempts.jsonl'):
    try:
        for line in open(f'{sys.argv[1]}/{name}'):
            try: total += float(json.loads(line).get('seconds') or 0)
            except ValueError: pass   # torn last line from a kill
    except FileNotFoundError: pass
print(int(total))
PYEOF
)
  REMAIN=$(awk -v m="$MAXMIN" -v u="$USED" 'BEGIN{r=m-u/60; if (r<0) r=0; printf "%.2f", r}')
  log "stage a: generation target=$TARGET max_minutes=$MAXMIN used_s=$USED remaining_min=$REMAIN"
  if awk -v r="$REMAIN" 'BEGIN{exit !(r>0)}'; then
    run -u scripts/ft/make_examples.py --style v2 --target "$TARGET" --max-minutes "$REMAIN" || fail a_generate $?
  fi
  touch "$STAGES/a_generate.done"
fi

# (b) split + manifest + leakage check (check_leakage asserts 0 overlaps; split-only fails if valid < 5%)
if [ ! -f "$STAGES/b_split.done" ]; then
  log "stage b: split-only"
  run scripts/ft/make_examples.py --style v2 --split-only || fail b_split $?
  touch "$STAGES/b_split.done"
fi

# (c) LoRA training on data/ft_v2: 2 epochs, round-1 config, checkpoint every 25 iters; train_lora.sh resumes itself
if [ ! -f "$STAGES/c_train.done" ]; then
  log "stage c: train"
  DATA_DIR="$DATA" bash scripts/ft/train_lora.sh "$ROUND"
  RC=$?
  [ "$RC" = 0 ] || fail c_train "$RC"
  touch "$STAGES/c_train.done"
fi

# (d) fuse -> GGUF -> Q4_K_M -> heldout_v1 eval with --save-raw
if [ ! -f "$STAGES/d_export_eval.done" ]; then
  log "stage d: export + eval"
  bash scripts/ft/export_and_eval.sh "$ROUND" | tee "$OUT/export_eval.log"
  RC=${PIPESTATUS[0]}
  [ "$RC" = 0 ] || fail d_export_eval "$RC"
  [ -f "$RESULT" ] || fail d_export_eval 3
  touch "$STAGES/d_export_eval.done"
fi

# (e) 4-row comparison + exit marker
log "stage e: comparison"
run - "$HELD" "$RESULT" "$R1" "$ROUND" > "$OUT/comparison.txt" <<'PYEOF' || fail e_comparison $?
import json, sys
held, result, r1, rnd = sys.argv[1:5]
rows = [('extractive local', f'{held}/heldout_v1_local_bm25.json'),
        ('base llama (untuned)', f'{held}/heldout_v1_llama_base_bm25.json'),
        ('fine-tuned round 1', r1),
        (f'fine-tuned round {rnd}', result)]
print(f"{'system':26} {'accepted':>10} {'unexp.refusals':>15} {'correct.refusals':>17} {'provider.fail':>14} {'claims met':>11}")
for name, path in rows:
    s = json.load(open(path))['summary']
    print(f"{name:26} {s['accepted']:>4}/{s['scored_answers']:<5} {s['unexpected_refusals']:>15} "
          f"{s['correct_refusals']:>9}/{s['unanswerable_scored']:<7} {s['provider_failures']:>14} {s['required_claims_met']:>5}/{s['required_claims']:<5}")
PYEOF
cat "$OUT/comparison.txt"
echo 0 > "$OUT/pipeline_exit.txt"
log "pipeline complete"
