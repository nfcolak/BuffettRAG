#!/usr/bin/env bash
# Usage: scripts/ft/run_round.sh <round> <target_examples> <max_gen_minutes>
# Round 3's target counts NEW kept examples, then mixes leakage-safe, deduplicated v2 splits.
# PILOT=1 with round 3 runs generation ONLY into models/ft/round3_pilot (never train/export).
set -uo pipefail
ROUND="${1:?usage: run_round.sh <round> <target_examples> <max_gen_minutes>}"
TARGET="${2:?target examples}"
MAXMIN="${3:?max generation minutes}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/.venv/bin/python}"
run() { env -u PYTHONPATH PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 "$PY" "$@"; }
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
case "$ROUND" in
  1|2) DATA=data/ft_v2; STYLE=v2 ;;
  3)
    DATA=data/ft_v3; STYLE=v3
    if [ "${PILOT:-0}" = 1 ]; then
      run -u scripts/ft/make_examples.py --style v3 --pilot --target "$TARGET" --max-minutes "$MAXMIN"
      exit $?
    fi
    # Fail closed BEFORE any stage/output writes, including when resuming marked stages.
    [ -f data/evaluation/heldout_v2/answer_benchmark_heldout_v2.json ] || {
      printf '%s\n' 'Round 3 requires blind heldout_v2 exclusion; PILOT=1 is generation-only.' >&2
      exit 2
    }
    ;;
  *) printf 'unsupported round: %s\n' "$ROUND" >&2; exit 2 ;;
esac
export FT_DATA_DIR="$DATA" FT_STYLE="$STYLE"
R1=data/evaluation/heldout_v1/final/llama_ft_r1_bm25.json
R2=data/evaluation/heldout_v1/final/llama_ft_r2_bm25.json
OUT="models/ft/round${ROUND}"
STAGES="$OUT/stages"
HELD=data/evaluation/heldout_v1
RESULT="$HELD/heldout_v1_llama_ft_r${ROUND}_bm25.json"
[ "$ROUND" = 3 ] && RESULT="$OUT/heldout_v1_llama_ft_r3_bm25.json"
mkdir -p "$STAGES"
rm -f "$OUT/pipeline_exit.txt"
fail() { log "FAILED stage=$1 rc=$2"; printf '%s\n' "$2" > "$OUT/pipeline_exit.txt"; printf '%s\n' "$1" > "$OUT/pipeline_failed_stage.txt"; exit "$2"; }

# (a) Time is cumulative over question AND answer generation, including restarts.
if [ ! -f "$STAGES/a_generate.done" ]; then
  REMAIN=$(run - "$DATA" "$MAXMIN" <<'PYEOF'
import json, pathlib, sys
used = 0.0
for name in ('example_attempts.jsonl', 'question_attempts.jsonl'):
    path = pathlib.Path(sys.argv[1]) / name
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                used += float(json.loads(line).get('seconds') or 0)
            except ValueError:
                pass  # generator recovers a torn final line
print(max(0., float(sys.argv[2]) - used / 60))
PYEOF
) || fail a_generate $?
  log "stage a: generation target=$TARGET max_minutes=$MAXMIN remaining_min=$REMAIN style=$STYLE"
  if run -c 'import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)' "$REMAIN"; then
    run -u scripts/ft/make_examples.py --style "$STYLE" --target "$TARGET" --max-minutes "$REMAIN" || fail a_generate $?
  fi
  touch "$STAGES/a_generate.done"
fi

# (b) Reused v2 rows AND new v3 rows must pass frozen-set/neighbor/question audits.
if [ ! -f "$STAGES/b_split.done" ]; then
  log "stage b: split-only"
  run scripts/ft/make_examples.py --style "$STYLE" --target "$TARGET" --split-only || fail b_split $?
  touch "$STAGES/b_split.done"
fi
# Round 3 re-audits on resume, so stale markers cannot bypass a changed blind set.
if [ "$ROUND" = 3 ]; then
  run scripts/ft/check_leakage.py || fail b_leakage $?
fi

# (c) Two epochs; absolute student weights in lora_config.yaml, worktree-local outputs.
if [ ! -f "$STAGES/c_train.done" ]; then
  log "stage c: train"
  DATA_DIR="$DATA" bash scripts/ft/train_lora.sh "$ROUND"
  RC=$?
  [ "$RC" = 0 ] || fail c_train "$RC"
  touch "$STAGES/c_train.done"
fi

# (d) Export/evaluate heldout_v1 ONLY; never run an answer system on heldout_v2.
if [ ! -f "$STAGES/d_export_eval.done" ]; then
  log "stage d: export + eval"
  RESULT="$RESULT" bash scripts/ft/export_and_eval.sh "$ROUND" | tee "$OUT/export_eval.log"
  RC=${PIPESTATUS[0]}
  [ "$RC" = 0 ] || fail d_export_eval "$RC"
  [ -f "$RESULT" ] || fail d_export_eval 3
  touch "$STAGES/d_export_eval.done"
fi

# (e) Canonical final baselines; avoid duplicate rows for the current round.
log "stage e: comparison"
run - "$HELD" "$RESULT" "$R1" "$R2" "$ROUND" > "$OUT/comparison.txt" <<'PYEOF' || fail e_comparison $?
import json, sys
held, result, r1, r2, rnd = sys.argv[1:6]
rows = [('extractive local', f'{held}/final/local_bm25.json'),
        ('base llama (untuned)', f'{held}/final/llama_base_bm25.json'),
        ('fine-tuned round 1', result if rnd == '1' else r1),
        ('fine-tuned round 2', result if rnd == '2' else r2)]
if rnd not in ('1', '2'):
    rows.append((f'fine-tuned round {rnd}', result))
print(f"{'system':26} {'accepted':>10} {'unexp.refusals':>15} {'correct.refusals':>17} {'provider.fail':>14} {'claims met':>11}")
for name, path in rows:
    s = json.load(open(path))['summary']
    print(f"{name:26} {s['accepted']:>4}/{s['scored_answers']:<5} {s['unexpected_refusals']:>15} "
          f"{s['correct_refusals']:>9}/{s['unanswerable_scored']:<7} {s['provider_failures']:>14} {s['required_claims_met']:>5}/{s['required_claims']:<5}")
PYEOF
run -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text(), end="")' "$OUT/comparison.txt"
printf '0\n' > "$OUT/pipeline_exit.txt"
log "pipeline complete"
