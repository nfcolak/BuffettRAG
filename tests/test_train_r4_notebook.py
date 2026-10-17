"""colab/train_7b_lora_r4.ipynb (FT-r4, Colab + Drive): static checks of every pv8-nb item plus execution of the pure-Python helper and data cells
(word-count tokenizer, no model, no GPU, no datasets package)."""
import hashlib
import json
import math
import re
import struct
import sys
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
R4 = ROOT / 'colab/train_7b_lora_r4.ipynb'
R3 = ROOT / 'colab/train_7b_lora_r3.ipynb'

# FT-r1/FT-r3 hyperparameters stay
R1_SETTINGS = {'RANK': 16, 'ALPHA': 32, 'LR': 1e-4, 'BATCH': 1, 'GRAD_ACCUM': 2}


def nb_cells(path=R4, kind='code'):
    return [''.join(c['source']) for c in json.loads(path.read_text())['cells'] if c['cell_type'] == kind]


def code_cell(marker, path=R4):
    found = [c for c in nb_cells(path) if marker in c]
    assert len(found) == 1, (marker, len(found))
    return found[0]


ALL_CODE = '\n'.join(nb_cells())


def run_settings():
    ns = {}
    exec(code_cell('RUN_NAME = '), ns)
    return ns


def run_helpers(ns=None):
    ns = ns if ns is not None else run_settings()
    exec(code_cell('def checkpoint_problem'), ns)
    return ns


# ---- structure -------------------------------------------------------------------------------------------------------------

def test_parses_and_every_code_cell_compiles():
    nb = json.loads(R4.read_text())
    assert nb['nbformat'] == 4 and nb['cells']
    for i, code in enumerate(nb_cells()):
        compile(code, f'cell{i}', 'exec')


def test_r3_notebook_is_untouched_and_r4_keeps_r1_hyperparameters():
    ns = run_settings()
    assert {k: ns[k] for k in R1_SETTINGS} == R1_SETTINGS and ns['BATCH'] * ns['GRAD_ACCUM'] == 2
    assert ns['EPOCHS'] == 2 and ns['SEED'] == 13 and ns['BASE_MODEL'] == 'unsloth/Qwen2.5-7B-Instruct-bnb-4bit'
    assert "target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']" in code_cell('get_peft_model')
    assert "lr_scheduler_type='linear', warmup_steps=5" in code_cell('SFTConfig(') and 'random_state=SEED' in code_cell('get_peft_model')
    assert "fp16=not is_bf16_supported(), bf16=is_bf16_supported()" in code_cell('SFTConfig(')
    assert 'run-7b-r3' in R3.read_text()                    # the r3 notebook still says r3


# ---- 1. RUN_NAME drives everything ------------------------------------------------------------------------------------------

def test_run_name_drives_every_path_and_no_r3_path_is_left():
    ns = run_settings()
    assert ns['RUN_NAME'] == 'r4' and ns['OUT_DIR'].endswith('/run-7b-r4') and ns['CKPT_DIR'].startswith(ns['OUT_DIR']) and ns['ADAPTER_DIR'].startswith(ns['OUT_DIR'])
    assert "f'/run-7b-{RUN_NAME}'" in code_cell('RUN_NAME = ') and "f'{OUT_DIR}/buffett-qwen2.5-7b-ft-{RUN_NAME}-s{step}-{quant}.gguf'" in code_cell('def gguf_name')
    assert not re.findall(r"run-7b-r3|[-_/.]r3[-_/.']|ft-r3", ALL_CODE.replace("FT-r3", ''))      # prose in comments says "FT-r3", no path or name does
    # no hard-coded r4 in a path or file name outside the one RUN_NAME line
    assert not re.findall(r"(?<!RUN_NAME = ')[-/]r4[-/.'_]", ALL_CODE)
    assert "metrics.json" in code_cell("'run_name': RUN_NAME") and "'run_name': RUN_NAME" in ALL_CODE


def test_run_guard_refuses_another_runs_dir_and_existing_gguf(tmp_path):
    ns = run_helpers()
    guard = ns['run_guard']
    out = tmp_path / 'run-7b-r4'
    out.mkdir()
    assert guard(str(out), 'r4') == {}
    with pytest.raises(AssertionError):
        guard(str(tmp_path / 'run-7b-r3'), 'r4')                         # directory name must carry RUN_NAME
    (out / 'run_meta.json').write_text(json.dumps({'run_name': 'r3', 'status': 'finished'}))
    with pytest.raises(SystemExit, match="'r3'"):
        guard(str(out), 'r4')
    (out / 'run_meta.json').write_text(json.dumps({'run_name': 'r4', 'status': 'training'}))
    (out / 'metrics.json').write_text(json.dumps({'run_name': 'r3'}))
    with pytest.raises(SystemExit):
        guard(str(out), 'r4')
    (out / 'metrics.json').write_text(json.dumps({'run_name': 'r4'}))
    assert guard(str(out), 'r4')['status'] == 'training'
    (out / 'buffett-qwen2.5-7b-ft-r3-q4_k_m.gguf').write_bytes(b'x')
    with pytest.raises(SystemExit, match='another run'):
        guard(str(out), 'r4', allow_existing_export=True)
    (out / 'buffett-qwen2.5-7b-ft-r3-q4_k_m.gguf').unlink()
    (out / 'buffett-qwen2.5-7b-ft-r4-s800-q4_k_m.gguf').write_bytes(b'x')
    with pytest.raises(SystemExit, match='already exist'):
        guard(str(out), 'r4')
    assert guard(str(out), 'r4', allow_existing_export=True)


def test_identity_check_refuses_resume_on_other_data():
    ns = run_helpers()
    meta = {'train_sha256': 'a' * 64, 'seed': 13}
    ns['check_identity'](meta, train_sha256='a' * 64, seed=13)
    with pytest.raises(SystemExit):
        ns['check_identity'](meta, train_sha256='b' * 64, seed=13)
    assert 'check_identity(RUN_META, train_sha256=DATA_SHA' in code_cell('SFTConfig(')


# ---- 2. save/eval by steps --------------------------------------------------------------------------------------------------

def test_step_based_save_and_eval_on_a_subset_full_valid_at_the_end():
    ns = run_settings()
    assert ns['SAVE_STEPS'] == 50 and ns['SAVE_TOTAL_LIMIT'] >= 3 and ns['EVAL_STEPS'] >= 50 and ns['EVAL_ROWS'] == 40
    cfg = code_cell('SFTConfig(')
    for text in ("eval_strategy='steps', eval_steps=EVAL_STEPS", "save_strategy='steps', save_steps=SAVE_STEPS", 'save_total_limit=SAVE_TOTAL_LIMIT',
                 'eval_on_start=resume_path is None', 'eval_dataset=eval_sub_ds', "per_device_eval_batch_size=1", 'seed=SEED, data_seed=SEED',
                 "train_on_responses_only(trainer, instruction_part='<|im_start|>user\\n', response_part='<|im_start|>assistant\\n')"):
        assert text in cfg, text
    assert "eval_strategy='epoch'" not in ALL_CODE and "save_strategy='epoch'" not in ALL_CODE
    fin = code_cell("full_valid_eval_loss")
    assert 'trainer.eval_dataset = valid_ds' in fin and "metric_key_prefix='full_valid'" in fin and 'labels are not masked' in fin


def test_eval_subset_is_fixed_seeded_and_kind_balanced():
    ns = run_helpers()
    pick = ns['pick_eval_subset']
    kinds = ['a'] * 60 + ['b'] * 40 + ['c'] * 20
    first = pick(kinds, 40, 13)
    assert first == pick(kinds, 40, 13) and first != pick(kinds, 40, 14) and len(set(first)) == 40 and first == sorted(first)
    assert {k: sum(kinds[i] == k for i in first) for k in 'abc'} == {'a': 14, 'b': 13, 'c': 13}
    assert pick(['a'] * 10, 40, 13) == list(range(10)) and len(pick([None] * 100, 40, 13)) == 40


def test_no_epoch_adapter_assert_and_epoch_ends_are_saved():
    assert 'adapter_e' not in ALL_CODE and 'sorted(EPOCH_ADAPTERS)' not in ALL_CODE
    cb = code_cell('class R4Callback')
    assert 'EPOCH_END_STEPS' in cb and 'control.should_save = True' in cb and "copy_adapter(src, f'{ADAPTER_DIR}/step-{state.global_step}')" in cb
    assert 'STEPS_PER_EPOCH * e for e in range(1, EPOCHS + 1)' in cb and 'math.ceil(len(train_ds) / (BATCH * GRAD_ACCUM))' in cb


# ---- 3. resume --------------------------------------------------------------------------------------------------------------

def write_safetensors(path, ok=True):
    header = json.dumps({'w': {'dtype': 'F32', 'shape': [4], 'data_offsets': [0, 16]}}).encode()
    body = b'\0' * (16 if ok else 7)
    Path(path).write_bytes(struct.pack('<Q', len(header)) + header + body)


def write_checkpoint(root, step, drop=None, truncate=None):
    d = Path(root) / f'checkpoint-{step}'
    d.mkdir(parents=True)
    write_safetensors(d / 'adapter_model.safetensors')
    (d / 'adapter_config.json').write_text('{}')
    (d / 'trainer_state.json').write_text(json.dumps({'global_step': step, 'log_history': []}))
    for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
        with zipfile.ZipFile(d / name, 'w') as z:
            z.writestr('data.pkl', b'x')
    if drop:
        (d / drop).unlink()
    if truncate:
        raw = (d / truncate).read_bytes()
        (d / truncate).write_bytes(raw[: len(raw) // 2])
    return d


def test_resume_picks_newest_complete_checkpoint_numerically_and_skips_half_written(tmp_path):
    ns = run_helpers()
    newest = ns['newest_complete_checkpoint']
    assert newest(str(tmp_path)) is None
    write_checkpoint(tmp_path, 50)
    write_checkpoint(tmp_path, 100)
    assert newest(str(tmp_path)).endswith('checkpoint-100')                    # 100 > 50 numerically (lexicographic order is the r3 bug)
    write_checkpoint(tmp_path, 150, drop='optimizer.pt')
    write_checkpoint(tmp_path / 'x', 200)
    assert newest(str(tmp_path)).endswith('checkpoint-100')
    write_checkpoint(tmp_path, 250, truncate='optimizer.pt')
    write_checkpoint(tmp_path, 300, truncate='adapter_model.safetensors')
    d = write_checkpoint(tmp_path, 350)
    (d / 'trainer_state.json').write_text('{"global_step": 350')
    assert newest(str(tmp_path)).endswith('checkpoint-100')
    write_checkpoint(tmp_path, 400)
    assert newest(str(tmp_path)).endswith('checkpoint-400')
    train = code_cell('SFTConfig(')
    assert 'resume_path = newest_complete_checkpoint(CKPT_DIR)' in train and 'trainer.train(resume_from_checkpoint=resume_path)' in train
    assert "bool(glob.glob(CKPT_DIR" not in ALL_CODE


def test_stop_at_step_ends_a_leg_without_export():
    cb = code_cell('class R4Callback')
    assert 'if STOP_AT_STEP and state.global_step >= STOP_AT_STEP' in cb and 'control.should_training_stop = True' in cb and 'control.should_evaluate' in cb
    assert run_settings()['STOP_AT_STEP'] is None
    assert 'LEG_STOPPED = FINAL_STEP < TOTAL_STEPS' in code_cell('SFTConfig(')
    assert 'if LEG_STOPPED:' in code_cell('def export_ggufs') and 'if LEG_STOPPED:' in code_cell("full_valid_eval_loss")


# ---- 4. export --------------------------------------------------------------------------------------------------------------

def test_export_lists_checkpoints_and_writes_q4_k_m_and_q8_0_with_sums():
    ns = run_settings()
    assert ns['EXPORT_QUANTS'] == ['q4_k_m', 'q8_0'] and ns['EXPORT_STEPS'] == []
    cell = code_cell('def export_ggufs')
    assert 'EXPORT_LIST = sorted({FINAL_STEP, *EXPORT_STEPS})' in cell
    assert 'quantization_method=EXPORT_QUANTS' in cell and "FastLanguageModel.from_pretrained(f'{ADAPTER_DIR}/step-{s}'" in cell
    assert "'sha256': sha256_file(p)" in cell and "open(OUT_DIR + '/SHA256SUMS', 'w')" in cell and "glob.glob(OUT_DIR + '/*.gguf')" in cell
    assert 'os.replace(gguf_name(step, q) + \'.part\', gguf_name(step, q))' in cell                  # no half-written GGUF under its final name
    assert 'EPOCH_ADAPTERS' not in ALL_CODE and 'EXPORT_BASE_GGUF' not in ALL_CODE


# ---- 5. MAX_SEQ -------------------------------------------------------------------------------------------------------------

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


SYSTEM_V3 = 'SYSTEM v3 prompt\n'


def write_bundle(tmp_path, train_words, valid_words, prompt_version='v3', system=SYSTEM_V3, row_version=None, manifest_extra=None, bad_sums=False):
    bundle = tmp_path / 'bundle'
    d = bundle / 'ft_v7'
    d.mkdir(parents=True)

    def row(i, words):
        user = 'BEGIN UNTRUSTED PASSAGES ' + ' '.join(['w'] * words) + ' Answer with inline citations'
        r = {'id': f'r{i}', 'kind': 'single' if i % 2 else 'multipart',
             'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}, {'role': 'assistant', 'content': f'Answer {i} [1]'}]}
        if row_version:
            r['prompt_version'] = row_version
        return r
    for name, sizes in (('train', train_words), ('valid', valid_words)):
        (d / f'{name}.jsonl').write_text(''.join(json.dumps(row(i, n)) + '\n' for i, n in enumerate(sizes)))
    manifest = {'prompt_version': prompt_version, 'passages_mix': [8, 12, 16], 'k_histogram': {'8': 1}, **(manifest_extra or {})}
    (d / 'manifest.json').write_text(json.dumps(manifest))
    sums = ''.join(f'{hashlib.sha256((d / n).read_bytes()).hexdigest()}  ft_v7/{n}\n' for n in ('train.jsonl', 'valid.jsonl', 'manifest.json'))
    (bundle / 'SHA256SUMS').write_text(sums.replace('0', '1', 1) if bad_sums else sums)
    return bundle


def run_data_cell(bundle, monkeypatch, **extra):
    monkeypatch.setitem(sys.modules, 'datasets', types.SimpleNamespace(Dataset=FakeDataset))
    ns = run_settings()
    ns.update({'tokenizer': WordTokenizer(), 'BUNDLE_DIR': str(bundle), 'EXPECT_SYSTEM_SHA256': hashlib.sha256(SYSTEM_V3.encode()).hexdigest(), **extra})
    run_helpers(ns)
    exec(code_cell('from datasets import Dataset'), ns)
    return ns


def test_max_seq_is_the_real_maximum_rounded_to_512_and_nothing_is_dropped(tmp_path, monkeypatch, capsys):
    bundle = write_bundle(tmp_path, [200 + 40 * i for i in range(99)] + [9000], [300, 5000])
    ns = run_data_cell(bundle, monkeypatch)
    lengths = sorted(n for _, n in ns['train_enc'] + ns['valid_enc'])
    assert ns['MAX_LEN'] == lengths[-1] and ns['MAX_SEQ'] == math.ceil(lengths[-1] / 512) * 512 and ns['MAX_SEQ'] >= lengths[-1] > ns['P99']
    assert ns['TRAIN_DROPPED'] == ns['VALID_DROPPED'] == 0 and len(ns['train_ds']) == 100 and len(ns['valid_ds']) == 2
    out = capsys.readouterr().out
    assert 'dropped(>' in out and ' #' in out and f"max={lengths[-1]}" in out                # histogram and length stats are printed


def test_longest_row_above_the_cap_stops_instead_of_dropping(tmp_path, monkeypatch):
    bundle = write_bundle(tmp_path, [14000] + [100] * 5, [100])
    with pytest.raises(AssertionError, match='MAX_SEQ_CAP'):
        run_data_cell(bundle, monkeypatch)
    ns = run_data_cell(write_bundle(tmp_path / 'b', [14000] + [100] * 5, [100]), monkeypatch, MAX_SEQ_CAP=16384)
    assert ns['MAX_SEQ'] == 14336 and ns['TRAIN_DROPPED'] == 0
    with pytest.raises(AssertionError, match='would be dropped'):
        run_data_cell(write_bundle(tmp_path / 'c', [14000] + [100] * 5, [100]), monkeypatch, MAX_SEQ_OVERRIDE=2048)


def test_memory_precheck_extrapolates_r3_and_fails_early():
    ns = run_helpers()
    assert ns['estimate_peak_gb'](9728) == 7.7 and ns['estimate_peak_gb'](11776) < 8.5
    assert ns['check_memory_fits'](11776, 14.6) == ns['estimate_peak_gb'](11776)
    with pytest.raises(SystemExit):
        ns['check_memory_fits'](11776, 8.0)
    probe = code_cell('probe_loss')
    assert 'model(input_ids=ids, labels=ids).loss' in probe and 'PROBE_PEAK_GB <= 0.92 * GPU_TOTAL_GB' in probe and 'max(train_enc + valid_enc' in probe


# ---- 6. pins ----------------------------------------------------------------------------------------------------------------

def test_versions_are_pinned_and_recorded():
    ns = run_settings()
    assert ns['PINS']['unsloth'] == '2026.10.6' and ns['PINS']['transformers'] == '5.16.1' and ns['PINS']['trl'] == '1.15.0'
    assert set(ns['PINS']) >= {'unsloth', 'transformers', 'trl', 'peft', 'bitsandbytes'} and 'BASE_REVISION' in ns and ns['REQUIRE_ALL_PINS'] is True
    assert re.fullmatch(r'[0-9a-f]{40}', ns['LLAMA_CPP_COMMIT'])
    inst = code_cell('pip')
    assert "f'{k}=={v}' if v else k" in inst and 'assert not v or VERSIONS[k] == v' in inst and 'unfilled pins' in inst and 'pip_freeze.txt' in inst
    assert "git('checkout', '-q', LLAMA_CPP_COMMIT)" in inst and "assert git('rev-parse', 'HEAD') == LLAMA_CPP_COMMIT" in inst
    assert 'revision=BASE_REVISION' in code_cell('get_peft_model') and 'model_info(BASE_MODEL, revision=BASE_REVISION).sha' in code_cell('get_peft_model')
    assert "pip install -q unsloth\n" not in ALL_CODE
    meta = code_cell("full_valid_eval_loss")
    for key in ("'versions': VERSIONS", "'base_revision': BASE_SHA", "'llama_cpp_commit': LLAMA_CPP_COMMIT", "'seed': SEED", "'data_sha256': DATA_SHA"):
        assert key in meta, key
    assert "work + '/llama.cpp'" in code_cell('def export_ggufs') and 'export did not use the pinned llama.cpp' in code_cell('def export_ggufs')


# ---- 7. data identity -------------------------------------------------------------------------------------------------------

def test_data_cell_verifies_sha_prompt_version_and_system_prompt(tmp_path, monkeypatch, capsys):
    bundle = write_bundle(tmp_path, [100] * 4, [100] * 3, row_version='v3')
    ns = run_data_cell(bundle, monkeypatch)
    d = bundle / 'ft_v7'
    assert ns['DATA_SHA'] == {s: hashlib.sha256((d / f'{s}.jsonl').read_bytes()).hexdigest() for s in ('train', 'valid')}
    assert 'on 7/7 rows' in capsys.readouterr().out and ns['SYSTEM_SHA'] == hashlib.sha256(SYSTEM_V3.encode()).hexdigest()


def test_data_cell_refuses_wrong_prompt_version_changed_data_or_mixed_prompts(tmp_path, monkeypatch):
    with pytest.raises(AssertionError):
        run_data_cell(write_bundle(tmp_path / 'a', [100] * 3, [100], prompt_version='v2'), monkeypatch)             # manifest says v2
    with pytest.raises(AssertionError, match='another prompt_version'):
        run_data_cell(write_bundle(tmp_path / 'b', [100] * 3, [100], row_version='v2'), monkeypatch)                # a row says v2
    with pytest.raises(AssertionError, match='SHA256SUMS'):
        run_data_cell(write_bundle(tmp_path / 'c', [100] * 3, [100], bad_sums=True), monkeypatch)
    with pytest.raises(AssertionError, match='differs from the manifest'):
        run_data_cell(write_bundle(tmp_path / 'd', [100] * 3, [100], manifest_extra={'train_sha256': 'f' * 64}), monkeypatch)
    with pytest.raises(AssertionError, match='is not the v3 prompt'):
        run_data_cell(write_bundle(tmp_path / 'e', [100] * 3, [100], system='another system prompt\n'), monkeypatch)
    with pytest.raises(AssertionError, match='Partnership'):
        sysp = 'Buffett Partnership letters\n'
        run_data_cell(write_bundle(tmp_path / 'f', [100] * 3, [100], system=sysp), monkeypatch, EXPECT_SYSTEM_SHA256=None)
    bundle = write_bundle(tmp_path / 'g', [100] * 3, [100])
    rows = [json.loads(l) for l in (bundle / 'ft_v7/train.jsonl').read_text().splitlines()]
    rows[1]['messages'][0]['content'] = 'different system prompt'
    (bundle / 'ft_v7/train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    sums = (bundle / 'SHA256SUMS').read_text().splitlines()
    new = hashlib.sha256((bundle / 'ft_v7/train.jsonl').read_bytes()).hexdigest()
    (bundle / 'SHA256SUMS').write_text('\n'.join([f'{new}  ft_v7/train.jsonl'] + sums[1:]) + '\n')
    with pytest.raises(AssertionError, match='different system prompts'):
        run_data_cell(bundle, monkeypatch)
    s = run_settings()
    assert s['DATA_NAME'] == 'ft_v7' and s['EXPECT_PROMPT_VERSION'] == 'v3' and re.fullmatch(r'[0-9a-f]{64}', s['EXPECT_SYSTEM_SHA256'])


def test_expected_system_sha_is_the_v3_prompt_of_build_ft_v6():
    from scripts.ft import build_ft_v6
    assert hashlib.sha256(build_ft_v6.system_prompt('v3').encode()).hexdigest() == run_settings()['EXPECT_SYSTEM_SHA256']


# ---- 8. slice generation ----------------------------------------------------------------------------------------------------

def test_slice_cell_is_greedy_900_tokens_and_writes_per_step_files():
    ns = run_settings()
    assert ns['SLICE_MAX_NEW_TOKENS'] == 900 and ns['SLICE_FILE'].endswith('/slice_prompts.jsonl') and ns['SLICE_STEPS'] is None
    cell = code_cell('slice_prompts.jsonl on Drive')
    assert 'do_sample=False' in cell and 'max_new_tokens=SLICE_MAX_NEW_TOKENS' in cell and "f'{OUT_DIR}/slice_answers_{step}.jsonl'" in cell
    assert "r['id']" in cell and "r['messages']" in cell and "f'{ADAPTER_DIR}/step-{step}'" in cell and 'add_generation_prompt=True' in cell
    assert 'if not os.path.isfile(SLICE_FILE)' in cell and 'os.replace(dst' in cell


# ---- 9. markdown ------------------------------------------------------------------------------------------------------------

def test_top_markdown_has_procedure_time_math_and_disconnect_advice():
    top = nb_cells(kind='markdown')[0]
    for text in ('Run procedure', 'Time math', '40.1 s', '32,080 s', 'T4', 'L4', 'A100', 'estimate', 'After a disconnect', 'STOP_AT_STEP', 'Drive'):
        assert text in top, text
    assert top.count('estimate') >= 3 and 'not measured' in top
    assert 'kaggle' not in ALL_CODE.lower()


def test_save_adapter_copy_helper_is_complete_or_fails(tmp_path):
    ns = run_helpers()
    src = write_checkpoint(tmp_path / 'ck', 50)
    ns['copy_adapter'](str(src), str(tmp_path / 'ad/step-50'))
    assert sorted(p.name for p in (tmp_path / 'ad/step-50').iterdir()) == sorted(ns['ADAPTER_KEEP'])
    write_safetensors(src / 'adapter_model.safetensors', ok=False)
    with pytest.raises(AssertionError):
        ns['copy_adapter'](str(src), str(tmp_path / 'ad/step-bad'))
