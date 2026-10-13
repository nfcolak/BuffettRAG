"""colab/train_7b_lora_r3.ipynb: FT-r1 settings kept, MAX_SEQ from the data (p99 -> next 512, cap 12288), longer rows dropped and counted,
per-epoch eval/adapters, final-model GGUF export. The data cell is executed with a word-count tokenizer (no model, no GPU, no datasets package)."""
import json
import math
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / 'colab/train_7b_lora_r3.ipynb'
R1 = ROOT / 'colab/train_7b_lora.ipynb'

# FT-r1 (run-7b-r1, notebook at 7c522f5): r=16, alpha=32, lr=1e-4, batch 1 x grad-accum 2, all seven projections
R1_SETTINGS = {'RANK': 16, 'ALPHA': 32, 'LR': 1e-4, 'BATCH': 1, 'GRAD_ACCUM': 2}
TARGET_MODULES = "target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']"


def cells(path):
    return ["".join(c['source']) for c in json.loads(path.read_text())['cells'] if c['cell_type'] == 'code']


def code_cell(marker, path=R3):
    found = [c for c in cells(path) if marker in c]
    assert len(found) == 1, (marker, len(found))
    return found[0]


def settings_ns(path=R3):
    ns = {}
    exec(code_cell('EPOCHS = ', path), ns)
    return ns


def test_settings_are_ft_r1s_with_ft_v6_and_two_epochs():
    ns = settings_ns()
    assert {k: ns[k] for k in R1_SETTINGS} == R1_SETTINGS
    old = settings_ns(R1)
    assert all(ns[k] == old[k] for k in R1_SETTINGS) and ns['BATCH'] * ns['GRAD_ACCUM'] == 2     # effective batch of FT-r1
    assert (ns['DATA_NAME'], ns['EPOCHS'], ns['EXPORT_BASE_GGUF'], ns['MAX_SEQ_CAP'], ns['MAX_SEQ_OVERRIDE']) == ('ft_v6', 1, False, 12288, None)
    assert ns['BASE_MODEL'] == 'unsloth/Qwen2.5-7B-Instruct-bnb-4bit' and ns['OUT_DIR'].endswith('/run-7b-r3')
    assert TARGET_MODULES in code_cell('get_peft_model')
    assert "use_gradient_checkpointing='unsloth'" in code_cell('get_peft_model') and 'lora_dropout=0' in code_cell('get_peft_model')


def test_trainer_evals_and_saves_every_epoch_and_masks_the_prompt():
    cell = code_cell('SFTConfig(')
    for text in ("eval_strategy='epoch'", "save_strategy='epoch'", 'num_train_epochs=EPOCHS', 'learning_rate=LR', 'per_device_train_batch_size=BATCH',
                 'gradient_accumulation_steps=GRAD_ACCUM', 'per_device_eval_batch_size=1', 'max_seq_length=MAX_SEQ,', "packing=False"):
        assert text in cell, text
    assert "train_on_responses_only(trainer, instruction_part='<|im_start|>user\\n', response_part='<|im_start|>assistant\\n')" in cell
    assert 'FastLanguageModel.from_pretrained(BASE_MODEL, max_seq_length=MAX_SEQ_CAP' in code_cell('get_peft_model')
    adapters = code_cell('adapter_e')
    assert "OUT_DIR + f'/adapter_e{ep}'" in adapters and "range(1, EPOCHS + 1)" in adapters


def test_export_is_one_merged_q4_k_m_gguf_of_the_final_adapter_with_sha256():
    cell = code_cell('FT_GGUF = ')
    assert "run-7b-r3" in settings_ns()['OUT_DIR'] and 'buffett-qwen2.5-7b-ft-r3-q4_k_m.gguf' in cell
    assert "quantization_method='q4_k_m'" in cell and "OUT_DIR + '/adapter'" in cell and 'sha256_file(FT_GGUF)' in cell and 'EPOCH_GGUFS' not in cell
    assert 'EXPORT_BASE_GGUF' in code_cell('BASE_GGUF = OUT_DIR')


class WordTokenizer:
    """ChatML like Qwen's template; one token per whitespace-separated word."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': text.split()}


class FakeDataset(list):
    @classmethod
    def from_list(cls, rows):
        return cls(rows)


def write_bundle(tmp_path, train_words, valid_words, prompt_version='v2'):
    directory = tmp_path / 'ft_v6'
    directory.mkdir()
    system = 'SYSTEM v2 prompt\n'

    def row(i, words):
        user = 'BEGIN UNTRUSTED PASSAGES ' + ' '.join(['w'] * words) + ' Answer with inline citations'
        return {'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}, {'role': 'assistant', 'content': f'Answer {i} [1]'}]}
    for name, sizes in (('train', train_words), ('valid', valid_words)):
        (directory / f'{name}.jsonl').write_text(''.join(json.dumps(row(i, n)) + '\n' for i, n in enumerate(sizes)))
    (directory / 'manifest.json').write_text(json.dumps({'prompt_version': prompt_version, 'passages_mix': [8, 12, 16], 'k_histogram': {'8': 1}}))
    return tmp_path


def run_data_cell(bundle, monkeypatch, **extra):
    monkeypatch.setitem(sys.modules, 'datasets', types.SimpleNamespace(Dataset=FakeDataset))
    ns = {'tokenizer': WordTokenizer(), 'BUNDLE_DIR': str(bundle), 'DATA_NAME': 'ft_v6', 'MAX_SEQ_OVERRIDE': None, 'MAX_SEQ_CAP': 12288, **extra}
    exec(code_cell('from datasets import Dataset'), ns)
    return ns


def test_max_seq_is_p99_rounded_up_to_512_and_longer_rows_are_dropped_and_counted(tmp_path, monkeypatch):
    bundle = write_bundle(tmp_path, [200 + 40 * i for i in range(99)] + [9000], [300, 5000])
    ns = run_data_cell(bundle, monkeypatch)
    lengths = sorted(n for _, n in ns['train_enc'] + ns['valid_enc'])
    p99 = lengths[math.ceil(.99 * len(lengths)) - 1]
    assert ns['P99'] == p99 and ns['MAX_SEQ'] == math.ceil(p99 / 512) * 512 and ns['MAX_SEQ'] % 512 == 0
    assert ns['TRAIN_DROPPED'] + ns['VALID_DROPPED'] == sum(n > ns['MAX_SEQ'] for n in lengths) >= 1
    assert len(ns['train_ds']) + ns['TRAIN_DROPPED'] == 100 and len(ns['valid_ds']) + ns['VALID_DROPPED'] == 2
    assert all(len(r['text'].split()) <= ns['MAX_SEQ'] for r in ns['train_ds'] + ns['valid_ds'])


def test_cap_and_override(tmp_path, monkeypatch):
    bundle = write_bundle(tmp_path, [14000] * 5 + [100] * 95, [100])
    ns = run_data_cell(bundle, monkeypatch)
    assert ns['MAX_SEQ'] == 12288 and ns['TRAIN_DROPPED'] == 5 and len(ns['train_ds']) == 95
    ns = run_data_cell(bundle, monkeypatch, MAX_SEQ_OVERRIDE=2048)
    assert ns['MAX_SEQ'] == 2048 and ns['TRAIN_DROPPED'] == 5


def test_data_cell_refuses_a_non_v2_bundle(tmp_path, monkeypatch):
    bundle = write_bundle(tmp_path, [100] * 3, [100], prompt_version='v1')
    with pytest.raises(AssertionError):
        run_data_cell(bundle, monkeypatch)
