"""Read mlx_lm lora output on stdin; record the last saved checkpoint iter in progress.json.

usage: track_progress.py <progress.json> <total_iters> <already_done_iters>
"""
import json
import re
import sys

path, total, base = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
buf = ''
while True:
    chunk = sys.stdin.read(1)
    if not chunk:
        break
    if chunk not in '\r\n':
        buf += chunk
        continue
    match = re.search(r'save\s+(\d+)_adapters\.safetensors', buf)
    if match:
        done = base + int(match.group(1))
        json.dump({'total_iters': total, 'completed_iters': done}, open(path + '.tmp', 'w'))
        import os
        os.replace(path + '.tmp', path)
    buf = ''
