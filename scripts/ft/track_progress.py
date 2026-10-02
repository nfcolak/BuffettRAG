"""Read mlx_lm lora output on stdin; record completed iters and keep absolute-iteration checkpoints.

usage: track_progress.py <progress.json> <total_iters> <already_done_iters> <adapters_dir>

mlx_lm numbers its checkpoints relative to each (re)started run, so on every "save NNN_adapters"
line we copy adapters.safetensors to <adapters_dir>/abs_<absolute iter>.safetensors and write
progress.json {total_iters, completed_iters}. Validation losses are appended to val_history.jsonl
(next to progress.json) with absolute iteration numbers, for best-checkpoint selection.
"""
import json
import os
import re
import shutil
import sys

path, total, base, adapters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ansi = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
buf = ''
while True:
    chunk = sys.stdin.read(1)
    if not chunk:
        break
    if chunk not in '\r\n':
        buf += chunk
        continue
    line = ansi.sub('', buf)
    val = re.match(r'\s*(\d+)\s+val\s+([0-9.]+)', line)
    if val and int(val.group(1)) > 1:
        with open(os.path.join(os.path.dirname(path), 'val_history.jsonl'), 'a') as handle:
            handle.write(json.dumps({'iter': base + int(val.group(1)), 'val_loss': float(val.group(2))}) + '\n')
    match = re.search(r'save\s+(\d+)_adapters\.safetensors', line)
    if match:
        done = base + int(match.group(1))
        src = os.path.join(adapters, 'adapters.safetensors')
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(adapters, f'abs_{done:07d}.safetensors.tmp'))
            os.replace(os.path.join(adapters, f'abs_{done:07d}.safetensors.tmp'), os.path.join(adapters, f'abs_{done:07d}.safetensors'))
        json.dump({'total_iters': total, 'completed_iters': done}, open(path + '.tmp', 'w'))
        os.replace(path + '.tmp', path)
    buf = ''
