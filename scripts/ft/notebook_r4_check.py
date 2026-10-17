"""Local smoke test of colab/train_7b_lora_r4.ipynb's pure-Python cells (settings, helpers, data identity + MAX_SEQ) on a real data dir.

python scripts/ft/notebook_r4_check.py --data-dir data/ft_v6 --tokenizer models/ft/7b_r1/adapter --prompt-version v2
No GPU, no model: only the Qwen tokenizer (a local dir with tokenizer.json + chat template) is loaded on CPU.
"""
import argparse
import hashlib
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / 'colab/train_7b_lora_r4.ipynb'


def cell(marker):
    found = [''.join(c['source']) for c in json.loads(NB.read_text())['cells'] if c['cell_type'] == 'code' and marker in ''.join(c['source'])]
    assert len(found) == 1, (marker, len(found))
    return found[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--prompt-version', default='v3')
    a = ap.parse_args()
    from transformers import AutoTokenizer
    data = Path(a.data_dir).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / 'bundle'
        (bundle / data.name).mkdir(parents=True)
        lines = []
        for name in ('train.jsonl', 'valid.jsonl', 'manifest.json'):
            (bundle / data.name / name).symlink_to(data / name)
            lines.append(f'{hashlib.sha256((data / name).read_bytes()).hexdigest()}  {data.name}/{name}\n')
        (bundle / 'SHA256SUMS').write_text(''.join(lines))
        try:
            from datasets import Dataset
        except ImportError:
            class Dataset(list):
                from_list = classmethod(lambda cls, rows: cls(rows))
        sys.modules.setdefault('datasets', types.SimpleNamespace(Dataset=Dataset))
        ns = {}
        exec(cell('RUN_NAME = '), ns)
        ns.update(DATA_NAME=data.name, EXPECT_PROMPT_VERSION=a.prompt_version, EXPECT_SYSTEM_SHA256=None if a.prompt_version != 'v3' else ns['EXPECT_SYSTEM_SHA256'],
                  BUNDLE_DIR=str(bundle), tokenizer=AutoTokenizer.from_pretrained(a.tokenizer))
        exec(cell('def checkpoint_problem'), ns)
        exec(cell('from datasets import Dataset'), ns)
        print('---\nMAX_SEQ', ns['MAX_SEQ'], '| MAX_LEN', ns['MAX_LEN'], '| dropped', ns['TRAIN_DROPPED'], ns['VALID_DROPPED'], '| steps/epoch if EPOCHS=2:', -(-len(ns['train_ds']) // 2),
              '| est peak GB', ns['estimate_peak_gb'](ns['MAX_SEQ']), '| eval subset', len(ns['EVAL_IDX']), 'of', len(ns['valid_ds']))


if __name__ == '__main__':
    main()
