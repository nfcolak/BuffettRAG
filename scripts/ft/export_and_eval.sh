#!/usr/bin/env bash
# Usage: scripts/ft/export_and_eval.sh <round>
# Fuse the best (lowest logged val loss) saved LoRA checkpoint, convert to GGUF f16 -> Q4_K_M,
# score on the frozen heldout_v1 set with the embedded llama provider, print a 3-row comparison.
# Refuses to run until training finished cleanly (models/ft/round<N>/train_exit.txt == 0).
# Overrides: ADAPTER_ITER=<n|last> (default: best val loss), FORCE=1 (skip the exit check).
set -euo pipefail
ROUND="${1:?usage: export_and_eval.sh <round>}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/.venv/bin/python}"
M=/Users/necatifurkancolak/AI-Workplace/Projects/done/BuffettRAG/models
STUDENT="$M/student-qwen2.5-1.5b-hf"
CONVERT="$M/llama.cpp-src/convert_hf_to_gguf.py"
QUANT=/opt/homebrew/bin/llama-quantize
OUT="$ROOT/models/ft/round${ROUND}"
GGUF="$OUT/buffett-qwen2.5-1.5b-ft-r${ROUND}-q4_k_m.gguf"
HELD=data/evaluation/heldout_v1
RESULT="$ROOT/$HELD/heldout_v1_llama_ft_r${ROUND}_bm25.json"
run() { env -u PYTHONPATH PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 "$PY" "$@"; }

if [ "${FORCE:-0}" != 1 ] && [ "$(cat "$OUT/train_exit.txt" 2>/dev/null || echo missing)" != 0 ]; then
  echo "training round $ROUND not finished cleanly (train_exit.txt != 0); not exporting" >&2; exit 1
fi
[ -f "$OUT/adapters/adapters.safetensors" ] || { echo "no adapters in $OUT/adapters" >&2; exit 1; }

# (a) pick the adapter checkpoint and fuse it into an HF model
TOTAL=$(run - "$OUT" <<'EOF3'
import json, os, re, sys
out = sys.argv[1]
if os.path.exists(out + '/progress.json'):
    print(json.load(open(out + '/progress.json'))['total_iters'])
else:  # fall back to the "round=N total=T" header train_lora.sh writes to train.log
    print(re.findall(r'^round=\S+ total=(\d+)', open(out + '/train.log', errors='replace').read(), re.M)[0])
EOF3
)
PICK=$(ADAPTER_ITER="${ADAPTER_ITER:-}" run - "$OUT" "$TOTAL" <<'EOF'
import json, os, re, sys
out, total = sys.argv[1], int(sys.argv[2])
want = os.environ.get('ADAPTER_ITER', '')
if want and want != 'last':
    print(int(want)); raise SystemExit
hist = out + '/val_history.jsonl'
if os.path.exists(hist):   # absolute iters written by track_progress.py (survives resumes)
    pairs = [(d['iter'], d['val_loss']) for d in map(json.loads, open(hist))]
else:                      # run was never resumed: log iters are absolute
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', open(out + '/train.log', errors='replace').read()).replace('\r', '\n')
    pairs = [(int(i), float(l)) for i, l in re.findall(r'^\s*(\d+)\s+val\s+([0-9.]+)', text, re.M) if int(i) > 1]
best = None
for it, loss in pairs:
    ckpt = out + ('/adapters/adapters.safetensors' if it >= total else f'/adapters/abs_{it:07d}.safetensors')
    if not os.path.exists(ckpt):
        ckpt = out + f'/adapters/{it:07d}_adapters.safetensors'   # unresumed run: relative == absolute
    if os.path.exists(ckpt) and (best is None or loss < best[1]):
        best = (it, loss)
print(best[0] if best else total)
EOF
)
echo "using adapter checkpoint iter=$PICK (of $TOTAL)"
SEL="$OUT/adapter_selected"
rm -rf "$SEL" "$OUT/fused"; mkdir -p "$SEL"
cp "$OUT/adapters/adapter_config.json" "$SEL/"
if [ "$PICK" -ge "$TOTAL" ]; then SRC="$OUT/adapters/adapters.safetensors"
elif [ -f "$(printf '%s/adapters/abs_%07d.safetensors' "$OUT" "$PICK")" ]; then SRC="$(printf '%s/adapters/abs_%07d.safetensors' "$OUT" "$PICK")"
else SRC="$(printf '%s/adapters/%07d_adapters.safetensors' "$OUT" "$PICK")"; fi
cp "$SRC" "$SEL/adapters.safetensors"
echo "$PICK" > "$OUT/selected_iter.txt"
run -m mlx_lm fuse --model "$STUDENT" --adapter-path "$SEL" --save-path "$OUT/fused"

# (b) HF -> GGUF f16 -> Q4_K_M
run "$CONVERT" "$OUT/fused" --outfile "$OUT/buffett-ft-r${ROUND}-f16.gguf" --outtype f16
"$QUANT" "$OUT/buffett-ft-r${ROUND}-f16.gguf" "$GGUF" Q4_K_M
ls -la "$GGUF"

# (c) benchmark. The runner needs main's src/ (embedded llama provider), so run it from a
# git-archive of main with this worktree's runner (and --save-raw) overlaid; cases/output stay here.
BENCH="$OUT/bench_tree"
rm -rf "$BENCH"; mkdir -p "$BENCH"
git archive main | tar -x -C "$BENCH"
cp scripts/run_live_benchmark.py "$BENCH/scripts/run_live_benchmark.py"
mkdir -p "$ROOT/$HELD"
LLM_MODEL_PATH="$GGUF" run "$BENCH/scripts/run_live_benchmark.py" --provider llama --retrieval bm25 \
  --cases "$ROOT/$HELD/answer_benchmark_heldout_v1.json" --output "$RESULT" --save-raw \
  || echo "runner exit $? (non-zero = some provider failures; see summary)"

# (d) 3-row comparison
run - "$ROOT/$HELD" "$RESULT" "$ROUND" <<'EOF'
import json, sys
held, result, rnd = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [('extractive local', f'{held}/heldout_v1_local_bm25.json'),
        ('base llama (untuned)', f'{held}/heldout_v1_llama_base_bm25.json'),
        (f'fine-tuned round {rnd}', result)]
print(f"{'system':26} {'accepted':>10} {'unexp.refusals':>15} {'correct.refusals':>17} {'provider.fail':>14}")
for name, path in rows:
    s = json.load(open(path))['summary']
    print(f"{name:26} {s['accepted']:>4}/{s['scored_answers']:<5} {s['unexpected_refusals']:>15} "
          f"{s['correct_refusals']:>9}/{s['unanswerable_scored']:<7} {s['provider_failures']:>14}")
EOF
