"""FT-r3 training notebook: valid JSON, compilable cells, per-epoch adapters and GGUFs."""
import json
import re
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parents[1] / 'colab' / 'train_7b_lora.ipynb'


@pytest.fixture(scope='module')
def cells():
    nb = json.loads(NB.read_text(encoding='utf-8'))
    return [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']


def _strip_magics(code):
    return '\n'.join(l for l in code.split('\n') if not l.lstrip().startswith(('!', '%')))


def test_notebook_parses():
    nb = json.loads(NB.read_text(encoding='utf-8'))
    assert nb['nbformat'] == 4 and nb['cells']


def test_code_cells_compile(cells):
    assert cells
    for i, code in cells:
        compile(_strip_magics(code), f'cell{i}', 'exec')


def test_settings_and_epoch_saving(cells):
    allsrc = '\n'.join(code for _, code in cells)
    assert "DATA_NAME = 'ft_v5'" in allsrc and "'ft_v4'" in allsrc
    assert re.search(r'^MAX_SEQ = 6144', allsrc, re.M)
    assert re.search(r'^EXPORT_BASE_GGUF = False', allsrc, re.M)
    assert "save_strategy='epoch'" in allsrc and "eval_strategy='epoch'" in allsrc
    assert re.search(r'save_total_limit=(max\(2, EPOCHS\)|[2-9])', allsrc)
    assert 'eval_loss_per_epoch' in allsrc


def test_two_epoch_ggufs_and_sums(cells):
    allsrc = '\n'.join(code for _, code in cells)
    assert 'buffett-qwen2.5-7b-ft-r3-e1-q4_k_m.gguf' in allsrc
    assert 'buffett-qwen2.5-7b-ft-r3-e2-q4_k_m.gguf' in allsrc
    assert 'SHA256SUMS' in allsrc and 'os.chdir(work)' in allsrc


def test_no_rmtree_of_epoch_adapters(cells):
    for i, code in cells:
        for line in code.split('\n'):
            if 'rmtree' in line:
                assert not re.search(r'checkpoint|CKPT_DIR|OUT_DIR|adapter', line, re.I), (i, line)
