"""colab/train_gemma4_12b_r4.ipynb (FT-r4, Gemma 4 12B QLoRA on Colab A100) and its builder colab/build_train_gemma4_r4.py.

No GPU, no model, no network: every cell is compiled, the embedded rows are decoded, the pure helpers run for real, the chat template is rendered with the
local Gemma 4 tokenizer (data/tokenizer/gemma4_12b_it holds tokenizer.json but no chat template, so a stand-in of Gemma 4's turn format is set on it),
and Drive is replaced by an in-memory fake (plus a stub of googleapiclient for the real transport class). The Colab-only parts (unsloth, trl, llama.cpp)
are checked statically."""
import base64
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / 'colab/build_train_gemma4_r4.py'
NOTEBOOK = ROOT / 'colab/train_gemma4_12b_r4.ipynb'
TOKENIZER_DIR = ROOT / 'data/tokenizer/gemma4_12b_it'


def load_builder():
    spec = importlib.util.spec_from_file_location('build_train_gemma4_r4', BUILDER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['build_train_gemma4_r4'] = mod
    spec.loader.exec_module(mod)
    return mod


B = load_builder()
C = B.load_common()


def make_rows(n, prefix='t', words=30, with_id=True):
    rows = []
    for i in range(n):
        body = ' '.join(f'[{k}] ({1980 + k}) synthetic passage {prefix}{i} sentence {k}.' for k in range(1, 4)) + ' filler' * (words * (i + 1))
        row = {'messages': [{'role': 'system', 'content': 'SYSTEM synthetic: answer from the passages only.'},
                            {'role': 'user', 'content': f'BEGIN UNTRUSTED PASSAGES\n{body}\nEND UNTRUSTED PASSAGES\nQuestion {prefix}{i}?'},
                            {'role': 'assistant', 'content': f'Synthetic passage {prefix}{i} sentence 1. [1] Synthetic passage {prefix}{i} sentence 2. [2]'}]}
        if with_id:
            row['id'] = f'{prefix}{i}'
        rows.append(row)
    return rows


def to_lines(rows):
    return [json.dumps(r) + '\n' for r in rows]


def build(tmp_path, train=None, valid=None, **over):
    opts = {'run_name': 'r4', 'smoke': False, 'epochs': 2, 'save_steps': 50, 'eval_steps': 50, 'pins': {}, 'max_seq': B.MAX_SEQ,
            'tokenizer_dir': str(TOKENIZER_DIR), 'source': 'rows', **over}
    return B.build_notebook(to_lines(train if train is not None else make_rows(6)), to_lines(valid if valid is not None else make_rows(3, 'v')), opts)


def cells(nb, kind='code'):
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == kind]


def code(nb, marker):
    found = [c for c in cells(nb) if marker in c]
    assert len(found) == 1, (marker, len(found))
    return found[0][len('%%guarded\n'):] if found[0].startswith('%%guarded\n') else found[0]


def body(src):
    """The Python a code cell holds: a %%writefile cell is the file after its first line."""
    if src.startswith('%%guarded\n'):
        return src.split('\n', 1)[1]
    return src.split('\n', 1)[1] if src.startswith('%%writefile') else src


def written_module(nb, name):
    return body(code(nb, f'%%writefile /content/work/{name}'))


def decode_rows(nb, split):
    meta = embedded_meta(nb)
    ns = {}
    exec(code(nb, f'DATA_{split.upper()}_B64 = '), ns)
    raw = gzip.decompress(base64.b64decode(''.join(ns[f'DATA_{split.upper()}_B64'].split())))
    assert hashlib.sha256(raw).hexdigest() == meta[split]['sha256']
    return raw


def embedded_meta(nb):
    ns = {}
    exec(code(nb, 'DATA_META = '), ns)
    return ns['DATA_META']


def settings(nb):
    ns = {}
    exec(code(nb, 'RUN_NAME = '), ns)
    return ns


@pytest.fixture(scope='module')
def hf_tok():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    tok.chat_template = B.STANDIN_TEMPLATE
    return tok


@pytest.fixture(scope='module')
def nb_fixture(tmp_path_factory):
    return build(tmp_path_factory.mktemp('nb'))


# ---- 2.2 stub tests: compile, decode, template, token counts -------------------------------------------------------------------------------------

def test_every_cell_compiles_and_the_written_modules_too(nb_fixture):
    nb = nb_fixture
    assert nb['nbformat'] == 4 and nb['cells']
    for i, src in enumerate(cells(nb)):
        compile(body(src), f'cell{i}', 'exec')
    for name in ('r4_common.py', 'train_gemma4.py', 'export_gguf.py'):
        compile(written_module(nb, name), name, 'exec')


def test_committed_notebook_compiles_and_is_the_builders_output():
    nb = json.loads(NOTEBOOK.read_text())
    for i, src in enumerate(cells(nb)):
        compile(body(src), f'cell{i}', 'exec')
    meta = embedded_meta(nb)
    opts = {**meta['options'], 'pins': settings(nb)['PINS'], 'tokenizer_dir': str(TOKENIZER_DIR)}
    if meta['placeholder']:
        assert opts['source'] == 'fixture' and meta['placeholder'] == B.FIXTURE_PLACEHOLDER
        train, valid = B.fixture_rows()
    else:                                                  # real rows: the notebook must be what the builder makes of the rows it embeds
        assert not meta['smoke_subset'] and opts['source'] == 'rows'
        train, valid = (decode_rows(nb, sp).decode().splitlines(keepends=True) for sp in ('train', 'valid'))
    again = B.build_notebook(train, valid, opts)
    assert again == nb, 'colab/train_gemma4_12b_r4.ipynb is out of date: re-run colab/build_train_gemma4_r4.py'


def test_runtime_is_a_100_and_settings_follow_the_task(nb_fixture):
    nb = nb_fixture
    assert nb['metadata']['colab']['gpuType'] == 'A100' and nb['metadata']['accelerator'] == 'GPU'
    s = settings(nb)
    assert (s['RANK'], s['ALPHA'], s['LR'], s['BATCH'], s['GRAD_ACCUM']) == (16, 32, 1e-4, 1, 4) and s['WARMUP_STEPS'] > 0
    assert s['MAX_SEQ'] == 16384 and s['BASE_MODEL'] == 'google/gemma-4-12B-it' and s['EPOCHS'] == 2 and s['MAX_STEPS'] is None
    assert re.fullmatch(r'[0-9a-f]{40}', s['BASE_REVISION']) and re.fullmatch(r'[0-9a-f]{40}', s['LLAMA_CPP_COMMIT'])
    assert s['BASE_REVISION'] == '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7' and s['EXPORT_QUANT'] == 'Q4_K_M'
    assert s['SMOKE'] is False and s['RESUME'] is False and s['REQUIRE_ALL_PINS'] is True and s['RELEASE_SLEEP_SECONDS'] == 900
    assert (s['SMOKE_ROWS'], s['SMOKE_STEPS']) == (20, 10) and s['SAVE_STEPS'] == 50 and s['EVAL_STEPS'] == 50
    assert set(s['SETTING_NAMES']) <= set(s) and len(set(s['SETTING_NAMES'])) == len(s['SETTING_NAMES'])


def test_embedded_rows_decode_to_exactly_the_input_bytes(tmp_path):
    train, valid = make_rows(5), make_rows(2, 'v')
    nb = build(tmp_path, train, valid)
    for split, rows in (('train', train), ('valid', valid)):
        raw = decode_rows(nb, split)
        assert raw == ''.join(to_lines(rows)).encode()
        meta = embedded_meta(nb)[split]
        assert meta['rows'] == len(rows) and len(meta['tokens']) == len(rows) and meta['sha256'] == hashlib.sha256(raw).hexdigest()
    # the unpack cell does the same inside the notebook and writes the files
    ns = {'DATA_META': embedded_meta(nb), 'ALLOW_PLACEHOLDER': False, 'C': C}
    for split in ('train', 'valid'):
        exec(code(nb, f'DATA_{split.upper()}_B64 = '), ns)
    exec_unpack(nb, ns, tmp_path)
    assert ns['RAW']['train'] == ''.join(to_lines(train)).encode()


def exec_unpack(nb, ns, tmp_path):
    src = code(nb, 'RAW = {}').replace('/content/work/data', str(tmp_path / 'data'))
    exec(src, ns)


def test_unpack_cell_refuses_a_changed_blob_and_a_fixture_notebook(tmp_path):
    nb = build(tmp_path)
    ns = {'DATA_META': embedded_meta(nb), 'ALLOW_PLACEHOLDER': False, 'C': C}
    for split in ('train', 'valid'):
        exec(code(nb, f'DATA_{split.upper()}_B64 = '), ns)
    ns['DATA_TRAIN_B64'] = base64.b64encode(gzip.compress(b'{"messages": []}\n')).decode()
    with pytest.raises(AssertionError, match='sha256'):
        exec_unpack(nb, ns, tmp_path)
    placeholder = build(tmp_path, source='fixture')
    ns2 = {'DATA_META': embedded_meta(placeholder), 'ALLOW_PLACEHOLDER': False, 'C': C}
    with pytest.raises(SystemExit, match='fixture'):
        exec_unpack(placeholder, ns2, tmp_path)


def test_standin_template_in_the_builder_equals_the_jinja_template(hf_tok):
    for row in make_rows(3):
        msgs = row['messages']
        for gen in (False, True):
            m = msgs[:2] if gen else msgs
            assert hf_tok.apply_chat_template(m, tokenize=False, add_generation_prompt=gen) == B.render_standin(m, gen)


def test_chat_template_rendering_of_3_rows_ends_with_the_assistant_target(hf_tok):
    render = lambda m, g: hf_tok.apply_chat_template(m, tokenize=False, add_generation_prompt=g)
    turn_end = C.derive_turn_end(render)
    assert turn_end == '<turn|>\n'
    for row in make_rows(3):
        msgs = row['messages']
        target = msgs[-1]['content']
        text, mode = C.render_training_text(render, msgs, turn_end)
        assert mode == 'template' and text.endswith(target + turn_end) and text[:-len(turn_end)].endswith(target)
        # the prompt is exactly what serving would send (system + user + generation prompt)
        assert text.startswith(render(msgs[:2], True)) and render(msgs[:2], True).endswith('<|turn>model\n')


def test_markers_come_from_the_template_and_the_loss_covers_only_the_target(hf_tok):
    render = lambda m, g: hf_tok.apply_chat_template(m, tokenize=False, add_generation_prompt=g)
    instruction, response = C.derive_markers(render, hf_tok.bos_token)
    assert (instruction, response) == ('<|turn>user\n', '<|turn>model\n')
    encode = lambda t: hf_tok(t, add_special_tokens=False)['input_ids']
    turn_end = C.derive_turn_end(render)
    rows = make_rows(3)
    prep = C.prepare_split('train', rows, render, encode, hf_tok.bos_token_id, turn_end, 16384)
    # what unsloth's train_on_responses_only leaves unmasked: everything after the response marker, i.e. target + turn end
    dataset = []
    for text, row in zip(prep['texts'], rows):
        ids, prompt_ids = encode(text), encode(render(row['messages'][:2], True))
        dataset.append({'input_ids': ids, 'labels': [-100] * len(prompt_ids) + ids[len(prompt_ids):]})
    live = C.check_loss_masks('train', dataset, hf_tok.decode, [r['messages'][2]['content'] for r in rows], turn_end)
    assert live == sum(len(d['labels']) - d['labels'].count(-100) for d in dataset) > 0
    assert dataset[0]['labels'][0] == -100 and dataset[0]['labels'][-1] != -100
    bad = [{'input_ids': d['input_ids'], 'labels': [-100] * (len(d['labels']) - 3) + d['labels'][-3:]} for d in dataset]
    with pytest.raises(AssertionError):
        C.check_loss_masks('train', bad, hf_tok.decode, [r['messages'][2]['content'] for r in rows], turn_end)
    unmasked_prompt = [{'input_ids': d['input_ids'], 'labels': list(d['input_ids'])} for d in dataset]
    with pytest.raises(AssertionError):
        C.check_loss_masks('train', unmasked_prompt, hf_tok.decode, [r['messages'][2]['content'] for r in rows], turn_end)
    # the trainer may append the EOS token to the text column: tolerated only when the caller says so
    with_eos = [{'input_ids': d['input_ids'] + [hf_tok.eos_token_id], 'labels': d['labels'] + [hf_tok.eos_token_id]} for d in dataset]
    with pytest.raises(AssertionError):
        C.check_loss_masks('train', with_eos, hf_tok.decode, [r['messages'][2]['content'] for r in rows], turn_end)
    assert C.check_loss_masks('train', with_eos, hf_tok.decode, [r['messages'][2]['content'] for r in rows], turn_end, (hf_tok.eos_token,)) == live + len(rows)


def test_token_counts_match_the_builder_and_a_second_tokenizer_path(hf_tok, tmp_path):
    train, valid = make_rows(4), make_rows(2, 'v')
    nb = build(tmp_path, train, valid)
    meta = embedded_meta(nb)
    render = lambda m, g: hf_tok.apply_chat_template(m, tokenize=False, add_generation_prompt=g)
    encode = lambda t: hf_tok(t, add_special_tokens=False)['input_ids']
    turn_end = C.derive_turn_end(render)
    for split, rows in (('train', train), ('valid', valid)):
        prep = C.prepare_split(split, rows, render, encode, hf_tok.bos_token_id, turn_end, 16384, expect_tokens=meta[split]['tokens'], tolerance=0)
        assert prep['tokens'] == meta[split]['tokens'] and prep['max_diff'] == 0     # HF tokenizer + jinja template == builder (tokenizers + python render)
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(TOKENIZER_DIR / 'tokenizer.json'))
        assert prep['tokens'] == [len(tk.encode(B.render_standin(r['messages']), add_special_tokens=False).ids) for r in rows]
    assert meta['train']['tokens_max'] == max(meta['train']['tokens']) and meta['train']['tokens_total'] == sum(meta['train']['tokens'])


def test_prepare_split_refuses_gap_overlong_rows_and_template_changes(hf_tok):
    render = lambda m, g: hf_tok.apply_chat_template(m, tokenize=False, add_generation_prompt=g)
    encode = lambda t: hf_tok(t, add_special_tokens=False)['input_ids']
    rows = make_rows(2)
    turn_end = C.derive_turn_end(render)
    counts = [len(encode(B.render_standin(r['messages']))) for r in rows]
    assert C.prepare_split('t', rows, render, encode, 2, turn_end, 16384, counts, 0)['max_diff'] == 0
    with pytest.raises(AssertionError, match='builder counted'):
        C.prepare_split('t', rows, render, encode, 2, turn_end, 16384, [counts[0] + 20, counts[1]], 16)
    with pytest.raises(AssertionError, match='MAX_SEQ'):
        C.prepare_split('t', rows, render, encode, 2, turn_end, counts[1] - 1)
    # a template that closes the turn with something else than the derived turn end
    with pytest.raises(AssertionError, match='turn end'):
        C.prepare_split('t', rows, lambda m, g: render(m, g) + ('' if g else '<eos>'), encode, 2, turn_end, 16384)
    # a template whose generation prompt differs from the history rendering falls back to prompt + target + turn end
    quirky = lambda m, g: render(m, g) + ('<|channel>thought\n<channel|>' if g else '')
    text, mode = C.render_training_text(quirky, rows[0]['messages'], turn_end)
    assert mode == 'prompt+target' and text.endswith('<|channel>thought\n<channel|>' + rows[0]['messages'][2]['content'] + turn_end)
    # one BOS only
    with pytest.raises(AssertionError, match='BOS'):
        C.prepare_split('t', rows, lambda m, g: '<bos>' + render(m, g), encode, 2, turn_end, 16384)


def test_builder_refuses_overlong_rows_duplicate_ids_and_bad_roles(tmp_path):
    long_row = make_rows(1)
    long_row[0]['messages'][1]['content'] = 'word ' * 20000
    with pytest.raises(AssertionError, match='MAX_SEQ'):
        build(tmp_path, long_row)
    with pytest.raises(AssertionError, match='duplicate row ids'):
        build(tmp_path, make_rows(2), make_rows(2))                      # both splits use the ids t0, t1
    swapped = make_rows(1, 'a')
    swapped[0]['messages'][0]['role'] = 'user'
    with pytest.raises(AssertionError):
        build(tmp_path, swapped)


def test_rows_may_use_system_user_assistant_keys():
    row = {'id': 'k', 'system': 'S', 'user': 'U', 'assistant': 'A [1]'}
    assert C.rows_to_messages(row) == [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'U'}, {'role': 'assistant', 'content': 'A [1]'}]
    with pytest.raises(AssertionError):
        C.rows_to_messages({'system': 'S', 'user': '', 'assistant': 'A'})


# ---- smoke ------------------------------------------------------------------------------------------------------------------------------------------

def test_smoke_picks_20_rows_with_the_longest_and_builds_a_small_notebook(tmp_path):
    train, valid = make_rows(60), make_rows(12, 'v')
    full = build(tmp_path, train, valid)
    smoke = build(tmp_path, train, valid, smoke=True)
    s = settings(smoke)
    assert s['SMOKE'] is True and s['REQUIRE_ALL_PINS'] is False and settings(full)['SMOKE'] is False
    meta_s, meta_f = embedded_meta(smoke), embedded_meta(full)
    assert meta_s['train']['rows'] == 20 and meta_s['valid']['rows'] == 6 and meta_s['smoke_subset'] is True
    assert meta_s['full']['train_rows'] == 60 and meta_s['full']['tokens_mean'] == meta_f['full']['tokens_mean']
    assert max(meta_s['train']['tokens']) == max(meta_f['train']['tokens']) == meta_f['train']['tokens_max']          # the longest row is in the smoke
    assert len(json.dumps(smoke)) < len(json.dumps(full))
    ids = [json.loads(line)['id'] for line in decode_rows(smoke, 'train').decode().splitlines()]
    assert len(set(ids)) == 20 and 't59' in ids


def test_effective_settings_for_smoke_and_full():
    S = {'SMOKE': True, 'RUN_NAME': 'r4', 'SAVE_STEPS': 50, 'EVAL_STEPS': 50, 'WARMUP_STEPS': 10, 'MAX_STEPS': None, 'SMOKE_STEPS': 10, 'SMOKE_SAVE_STEPS': 5,
         'SMOKE_ROWS': 20, 'SMOKE_VALID_ROWS': 6, 'SEED': 13, 'DRIVE_ROOT': 'buffettrag'}
    meta = {'train': {'tokens': list(range(100, 160))}, 'valid': {'tokens': list(range(10))}}
    E = C.effective_settings(S, meta)
    assert (E['MAX_STEPS'], E['SAVE_STEPS'], E['EVAL_STEPS'], E['WARMUP_STEPS']) == (10, 5, 5, 1) and E['RUN_TAG'] == 'r4-smoke'
    assert E['DRIVE_PATH'] == ['buffettrag', 'ftr4-gemma4-12b-r4-smoke'] and len(E['TRAIN_INDICES']) == 20 and 59 in E['TRAIN_INDICES'] and len(E['VALID_INDICES']) == 6
    assert E['EXPECT_TOKENS']['train'] == [meta['train']['tokens'][i] for i in E['TRAIN_INDICES']]
    full = C.effective_settings({**S, 'SMOKE': False}, meta)
    assert full['MAX_STEPS'] is None and full['SAVE_STEPS'] == 50 and full['RUN_TAG'] == 'r4' and full['TRAIN_INDICES'] is None
    assert full['EXPECT_TOKENS']['train'] == meta['train']['tokens']
    tied = C.pick_smoke_rows([5, 9, 9, 1], 2, 13)
    assert C.pick_smoke_rows([5, 9, 9, 1], 10, 13) == [0, 1, 2, 3] and len(tied) == 2 and 1 in tied and tied == C.pick_smoke_rows([5, 9, 9, 1], 2, 13)   # ties: the earlier longest row


def test_smoke_cell_writes_timing_pins_and_checks_the_round_trip(nb_fixture):
    src = code(nb_fixture, 'SMOKE_REPORT = {}')
    for text in ("SMOKE_STEPS", "assert BEGINS == [0]", "STORE.restore_newest_checkpoint(E['CKPT_LOCAL'])", "run_leg('resume'", "'--max-steps', str(SMOKE_STEPS + SMOKE_RESUME_EXTRA_STEPS)",
                 "'seconds_per_step_mean'", "'estimate_full_train_seconds'", "'peak_gpu_gb'", "BEGINS[-1] == SMOKE_STEPS"):
        assert text in src, text
    fin = code(nb_fixture, 'from google.colab import files')
    assert "pins_resolved.json" in fin and "smoke_report.json" in fin and "if not SMOKE:" in fin and "'pins_resolved'" in fin


# ---- Drive: no mount, API round trip -----------------------------------------------------------------------------------------------------------------

class FakeDrive:
    """In-memory stand-in of C.GoogleDrive: same methods, md5 and size like Drive reports them."""

    def __init__(self, tmp_path):
        self.root = tmp_path / 'drive'
        self.root.mkdir()
        self.files = {}
        self.n = 0
        self.deleted = []

    def _info(self, fid):
        f = self.files[fid]
        data = f['path'].read_bytes() if f['path'] else b''
        return {'id': fid, 'name': f['name'], 'size': str(len(data)), 'md5Checksum': hashlib.md5(data).hexdigest(), 'appProperties': dict(f['props']),
                'mimeType': f['mime'], 'parent': f['parent']}

    def find(self, parent, name=None, folders_only=False):
        out = [self._info(fid) for fid, f in self.files.items() if f['parent'] == parent and fid not in self.deleted]
        if name is not None:
            out = [o for o in out if o['name'] == name]
        if folders_only:
            out = [o for o in out if o['mimeType'].endswith('folder')]
        return out

    def mkdir(self, parent, name):
        self.n += 1
        fid = f'dir{self.n}'
        self.files[fid] = {'name': name, 'parent': parent, 'path': None, 'props': {}, 'mime': 'application/vnd.google-apps.folder'}
        return fid

    def put(self, local_path, parent, name, props=None, replace_id=None):
        data = Path(local_path).read_bytes()
        if replace_id:
            fid = replace_id
        else:
            self.n += 1
            fid = f'f{self.n}'
        dest = self.root / fid
        dest.write_bytes(data)
        self.files[fid] = {'name': name, 'parent': parent, 'path': dest, 'props': dict(props or {}), 'mime': 'application/octet-stream'}
        return self._info(fid)

    def get(self, file_id, dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        Path(dest).write_bytes(self.files[file_id]['path'].read_bytes())

    def delete(self, file_id):
        self.deleted.append(file_id)
        self.files.pop(file_id)


def write_safetensors(path, ok=True):
    import struct
    header = json.dumps({'w': {'dtype': 'F32', 'shape': [4], 'data_offsets': [0, 16]}}).encode()
    Path(path).write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * (16 if ok else 7))


def write_checkpoint(root, step, drop=None, truncate=None):
    import zipfile
    d = Path(root) / f'checkpoint-{step}'
    d.mkdir(parents=True)
    write_safetensors(d / 'adapter_model.safetensors')
    (d / 'adapter_config.json').write_text('{}')
    (d / 'trainer_state.json').write_text(json.dumps({'global_step': step, 'log_history': []}))
    for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
        with zipfile.ZipFile(d / name, 'w') as z:
            z.writestr('data.pkl', b'x' * 100)
    if drop:
        (d / drop).unlink()
    if truncate:
        raw = (d / truncate).read_bytes()
        (d / truncate).write_bytes(raw[: len(raw) // 2])
    return d


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'WORK', str(tmp_path / 'work'))
    return tmp_path


def save_like_the_train_script(tmp_path, step):
    """What R4Callback.on_save does: adapter copy + the two tars."""
    ckpt = write_checkpoint(tmp_path / 'ckpt', step)
    adapter = tmp_path / 'adapters' / f'step-{step}'
    adapter.mkdir(parents=True)
    for name in C.ADAPTER_FILES:
        (adapter / name).write_bytes((ckpt / name).read_bytes())
    tar = C.make_tar(str(ckpt), str(tmp_path / 'uploads' / f'checkpoint-{step}.tar'), f'checkpoint-{step}')
    atar = C.make_tar(str(adapter), str(tmp_path / 'uploads' / f'adapter-step-{step}.tar'), f'step-{step}')
    return tar, atar


def test_drive_round_trip_upload_restore_resume_and_retention(work):
    drive = FakeDrive(work)
    store = C.DriveStore(drive, ['buffettrag', 'ftr4-gemma4-12b-r4-smoke'])
    again = C.DriveStore(drive, ['buffettrag', 'ftr4-gemma4-12b-r4-smoke'])
    assert again.folder_id == store.folder_id and len([f for f in drive.files.values() if f['mime'].endswith('folder')]) == 2          # folders are reused
    up = C.Uploader(store, keep=2)
    for step in (5, 10, 15, 20):
        tar, atar = save_like_the_train_script(work, step)
        up.submit(step, tar, atar, best=5)
    up.join()
    assert [s for s, _ in store.list_checkpoints()] == [20, 15, 5]                  # newest 2 + the best (5); 10 pruned
    assert [s for s, _ in store.list_adapters()] == [20, 15, 10, 5]                  # adapters are never pruned
    assert not list((work / 'uploads').iterdir()), 'local tars are removed after a verified upload'
    # a fresh runtime: nothing local, restore the newest
    import shutil
    shutil.rmtree(work / 'ckpt')
    shutil.rmtree(work / 'adapters')
    path = store.restore_newest_checkpoint(str(work / 'fresh'))
    assert path.endswith('checkpoint-20') and C.checkpoint_problem(path) is None and C.newest_complete_checkpoint(str(work / 'fresh')) == path
    adapter = store.restore_adapter(10, str(work / 'fresh_adapters'))
    assert C.safetensors_complete(os.path.join(adapter, 'adapter_model.safetensors'))
    assert C.best_step([{'step': 0, 'eval_loss': 1.0}, {'step': 5, 'eval_loss': 0.9}, {'step': 10, 'eval_loss': 0.7}, {'step': 15, 'eval_loss': 0.7}]) == 10
    assert C.best_step([{'step': 5, 'eval_loss': 0.9}, {'step': 10, 'eval_loss': 0.7}], available={5}) == 5


def test_drive_restore_refuses_corruption_and_falls_back_to_an_older_checkpoint(work):
    drive = FakeDrive(work)
    store = C.DriveStore(drive, ['x'])
    for step in (5, 10):
        tar, atar = save_like_the_train_script(work, step)
        store.upload(tar, props={'kind': 'checkpoint', 'step': str(step)})
    newest = next(f for f in drive.files.values() if f['name'] == 'checkpoint-10.tar')
    newest['path'].write_bytes(newest['path'].read_bytes()[:-9] + b'corrupted!')           # Drive md5 is recomputed, the stored sha256 is not
    with pytest.raises(AssertionError, match='sha256'):
        store.restore_newest_checkpoint(str(work / 'r'))
    # an incomplete checkpoint inside a tar that matches its hash is skipped for the older one
    d = write_checkpoint(work / 'bad', 15, truncate='optimizer.pt')
    store.upload(C.make_tar(str(d), str(work / 'u' / 'checkpoint-15.tar'), 'checkpoint-15'), props={'step': '15'})
    store.drive.delete(next(f['id'] for f in store.listing() if f['name'] == 'checkpoint-10.tar'))
    assert store.restore_newest_checkpoint(str(work / 'r2')).endswith('checkpoint-5')
    (work / 'other').mkdir()
    empty = C.DriveStore(FakeDrive(work / 'other'), ['y'])
    assert empty.restore_newest_checkpoint(str(work / 'r3')) is None


def test_upload_verifies_size_and_md5_and_replaces_same_name(work):
    drive = FakeDrive(work)
    store = C.DriveStore(drive, ['x'])
    p = work / 'a.json'
    p.write_text('{"a": 1}')
    first = store.upload(str(p))
    p.write_text('{"a": 2}')
    second = store.upload(str(p))
    assert first['id'] == second['id'] and store.fetch_json('a.json') == {'a': 2} and len(store.find('a.json')) == 1
    store.write_json('run_meta.json', {'run_name': 'r4'})
    assert store.fetch_json('run_meta.json') == {'run_name': 'r4'} and store.fetch_json('missing.json') == {}
    original_put = drive.put
    drive.put = lambda *a, **k: {**original_put(*a, **k), 'md5Checksum': '0' * 32}
    with pytest.raises(AssertionError, match='md5'):
        store.upload(str(p), name='b.json')
    (work / 'empty.bin').write_bytes(b'')
    with pytest.raises(AssertionError, match='empty'):
        store.upload(str(work / 'empty.bin'))


def test_uploader_surfaces_a_failed_upload(work):
    drive = FakeDrive(work)
    store = C.DriveStore(drive, ['x'])
    up = C.Uploader(store)
    tar, atar = save_like_the_train_script(work, 5)
    os.remove(tar)
    up.submit(5, tar, atar, best=5)
    with pytest.raises(RuntimeError, match='drive upload failed'):
        up.join()
    with pytest.raises(RuntimeError):
        up.submit(10, tar, atar, best=5)


def test_untar_rejects_paths_outside_the_target(tmp_path):
    import tarfile
    evil = tmp_path / 'evil.tar'
    payload = tmp_path / 'p.txt'
    payload.write_text('x')
    with tarfile.open(evil, 'w') as tf:
        tf.add(payload, arcname='../escape.txt')
    with pytest.raises(AssertionError, match='unsafe'):
        C.safe_untar(str(evil), str(tmp_path / 'dest'))
    assert not (tmp_path / 'escape.txt').exists()


def test_resume_picks_newest_complete_checkpoint_numerically(tmp_path):
    assert C.newest_complete_checkpoint(str(tmp_path)) is None
    write_checkpoint(tmp_path, 50)
    write_checkpoint(tmp_path, 100)
    assert C.newest_complete_checkpoint(str(tmp_path)).endswith('checkpoint-100')                 # 100 > 50 numerically
    write_checkpoint(tmp_path, 150, drop='optimizer.pt')
    write_checkpoint(tmp_path, 250, truncate='optimizer.pt')
    write_checkpoint(tmp_path, 300, truncate='adapter_model.safetensors')
    assert C.newest_complete_checkpoint(str(tmp_path)).endswith('checkpoint-100')
    with pytest.raises(SystemExit):
        C.check_identity({'train_sha256': 'a'}, train_sha256='b')
    C.check_identity({'train_sha256': 'a', 'seed': 13}, train_sha256='a', seed=13)


class StubGoogle:
    """googleapiclient stand-in recording how the real transport calls it."""

    def __init__(self):
        self.builds = 0
        self.calls = []
        self.pages = [{'files': [{'id': '1', 'name': 'a'}], 'nextPageToken': 'p2'}, {'files': [{'id': '2', 'name': 'b'}]}]

    def install(self, monkeypatch):
        outer = self

        class Req:
            def __init__(self, result=None, chunks=2):
                self.result, self.chunks = result, chunks
            def execute(self, num_retries=0):
                return self.result
            def next_chunk(self, num_retries=0):
                self.chunks -= 1
                return (None, {'id': 'new', 'size': '3', 'md5Checksum': hashlib.md5(b'abc').hexdigest()}) if self.chunks <= 0 else (None, None)

        class Files:
            def list(self, **kw):
                outer.calls.append(('list', kw))
                return Req(outer.pages.pop(0))
            def create(self, **kw):
                outer.calls.append(('create', kw))
                return Req({'id': 'dirnew'}) if 'media_body' not in kw else Req()
            def update(self, **kw):
                outer.calls.append(('update', kw))
                return Req()
            def get_media(self, **kw):
                outer.calls.append(('get_media', kw))
                return 'media-request'
            def delete(self, **kw):
                outer.calls.append(('delete', kw))
                return Req({})

        class Svc:
            def files(self):
                return Files()

        def build(name, version, cache_discovery=None):
            outer.builds += 1
            outer.calls.append(('build', name, version, cache_discovery))
            return Svc()

        class MediaFileUpload:
            def __init__(self, path, mimetype=None, resumable=False, chunksize=None):
                outer.calls.append(('MediaFileUpload', path, resumable, chunksize))

        class MediaIoBaseDownload:
            def __init__(self, fh, request, chunksize=None):
                self.fh, self.left = fh, 2
                outer.calls.append(('MediaIoBaseDownload', request, chunksize))
            def next_chunk(self, num_retries=0):
                self.left -= 1
                self.fh.write(b'abc')
                return None, self.left <= 0

        gac, disc, http = types.ModuleType('googleapiclient'), types.ModuleType('googleapiclient.discovery'), types.ModuleType('googleapiclient.http')
        disc.build, http.MediaFileUpload, http.MediaIoBaseDownload = build, MediaFileUpload, MediaIoBaseDownload
        gac.discovery, gac.http = disc, http
        monkeypatch.setitem(sys.modules, 'googleapiclient', gac)
        monkeypatch.setitem(sys.modules, 'googleapiclient.discovery', disc)
        monkeypatch.setitem(sys.modules, 'googleapiclient.http', http)


def test_real_transport_uses_a_fresh_client_a_lock_resumable_chunks_and_shared_drive_flags(tmp_path, monkeypatch):
    stub = StubGoogle()
    stub.install(monkeypatch)
    drive = C.GoogleDrive(chunk=8 * C.MIB)
    found = drive.find('PARENT', name="it's", folders_only=True)
    assert [f['id'] for f in found] == ['1', '2'] and stub.builds == 1                           # paged, one client for one call
    q = [c for c in stub.calls if c[0] == 'list'][0][1]
    assert "name = 'it\\'s'" in q['q'] and "'PARENT' in parents" in q['q'] and 'trashed = false' in q['q'] and 'folder' in q['q']
    assert q['supportsAllDrives'] is True and q['includeItemsFromAllDrives'] is True
    local = tmp_path / 'f.bin'
    local.write_bytes(b'abc')
    resp = drive.put(str(local), 'PARENT', 'f.bin', {'sha256': 'x'})
    assert resp['id'] == 'new' and stub.builds == 2
    assert ('MediaFileUpload', str(local), True, 8 * C.MIB) in stub.calls
    create = [c for c in stub.calls if c[0] == 'create'][-1][1]
    assert create['body'] == {'name': 'f.bin', 'parents': ['PARENT'], 'appProperties': {'sha256': 'x'}} and create['supportsAllDrives'] is True
    drive.put(str(local), 'PARENT', 'f.bin', {}, replace_id='ID')
    assert [c for c in stub.calls if c[0] == 'update'][0][1]['fileId'] == 'ID' and stub.builds == 3
    out = tmp_path / 'sub' / 'g.bin'
    drive.get('ID', str(out))
    assert out.read_bytes() == b'abcabc' and not Path(str(out) + '.part').exists() and stub.builds == 4
    assert [c for c in stub.calls if c[0] == 'get_media'][0][1] == {'fileId': 'ID', 'supportsAllDrives': True}
    assert drive.mkdir('PARENT', 'dir') == 'dirnew'
    drive.delete('ID')
    assert stub.builds == 6


def test_notebook_does_not_mount_drive_and_uses_the_api(nb_fixture):
    allcode = '\n'.join(body(c) for c in cells(nb_fixture))
    assert 'drive.mount' not in allcode and 'flush_and_unmount' not in allcode
    assert 'auth.authenticate_user()' in allcode and 'googleapiclient' in allcode and 'MediaFileUpload' in allcode and 'resumable=True' in allcode
    assert allcode.count('build(') == 1 and "cache_discovery=False" in allcode and 'threading.Lock()' in allcode


# ---- notebook structure and the pitfalls of the colab-chrome-runs skill -------------------------------------------------------------------------------

def test_train_script_follows_the_task_spec(nb_fixture):
    src = written_module(nb_fixture, 'train_gemma4.py')
    for text in ("from unsloth import FastModel", "load_in_4bit=True", "revision=S['BASE_REVISION']", "r=S['RANK'], lora_alpha=S['ALPHA']", "random_state=S['SEED']",
                 "train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)", "C.derive_markers(render",
                 "tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)", "max_seq_length=S['MAX_SEQ']", "learning_rate=S['LR']",
                 "per_device_train_batch_size=S['BATCH']", "gradient_accumulation_steps=S['GRAD_ACCUM']", "warmup_steps=S['WARMUP_STEPS']",
                 "eval_strategy='steps', eval_steps=S['EVAL_STEPS']", "save_strategy='steps', save_steps=S['SAVE_STEPS']", "max_steps=MAX_STEPS", "num_train_epochs=S['EPOCHS']",
                 "trainer.train(resume_from_checkpoint=A.resume_from)", "disable_tqdm=True", "C.check_loss_masks('train'", "C.check_loss_masks('valid'",
                 "control.should_evaluate = control.should_save = True", "eval_on_start=A.resume_from is None", 'C.status(f"saved step={step}'):
        assert text in src, text
    assert 'googleapiclient' not in src and 'STORE' not in src and 'import torch' in src          # the training subprocess never talks to Drive
    # r4_common never imports torch or the Google client at import time
    common = written_module(nb_fixture, 'r4_common.py')
    head = common.split('class GoogleDrive')[0]
    assert 'import torch' not in common and 'googleapiclient' not in head


def test_export_script_merges_converts_quantizes_with_the_pinned_llama_cpp(nb_fixture):
    src = written_module(nb_fixture, 'export_gguf.py')
    for text in ("save_pretrained_merged(MERGED, tokenizer, save_method='merged_16bit')", "convert_hf_to_gguf.py", "'--outtype', 'bf16'", "build/bin/llama-quantize",
                 "QUANT = S['EXPORT_QUANT']", "'git', 'rev-parse', 'HEAD'", "S['LLAMA_CPP_COMMIT']", "snapshot_download(S['BASE_MODEL'], revision=S['BASE_REVISION']",
                 "cfg['base_model_name_or_path'] = base_dir", "C.gguf_summary(GGUF, LLAMA)", "summary['architecture'] == 'gemma4'", "'sha256': sha", "SHA256SUMS",
                 "os.replace(GGUF + '.part', GGUF)", "template_parity"):
        assert text in src, text
    run = code(nb_fixture, "HISTORY = REPORTS")
    assert 'C.best_step(HISTORY, available=ON_DRIVE)' in run and 'BUILD_THREAD.join()' in run and 'disk_usage' in run
    install = code(nb_fixture, 'import importlib.metadata as md_')
    assert "git('checkout', '-q', LLAMA_CPP_COMMIT)" in install and "git('rev-parse', 'HEAD') == LLAMA_CPP_COMMIT" in install
    assert 'llama-quantize' in install and "'-DGGML_CUDA=OFF'" in install


def test_pins_are_installed_before_anything_imports_torch_and_recorded(nb_fixture):
    install = code(nb_fixture, 'import importlib.metadata as md_')
    assert "f'{k}=={v}' if v else k" in install and 'assert not v or VERSIONS[k] == v' in install and 'unfilled pins' in install and 'pip_freeze.txt' in install
    assert "'torch': torch.__version__" in install                                              # the pin is proven from a subprocess
    kernel_code = '\n'.join(body(c) for c in cells(nb_fixture) if not c.startswith('%%writefile'))
    assert not re.search(r'^\s*(import|from)\s+(torch|unsloth|trl|peft|transformers)\b', kernel_code, re.M), 'the kernel must not import the GPU stack'
    fin = code(nb_fixture, 'from google.colab import files')
    assert "'versions': VERSIONS" in fin and "'llama_cpp_commit': LLAMA_CPP_COMMIT" in fin and "'base_revision_resolved'" in fin and "'chat_template_sha256'" in fin
    s = settings(nb_fixture)
    assert s['PINS']['unsloth'] == '2026.10.6' and s['PINS']['peft'] is None and s['REQUIRE_ALL_PINS'] is True


def test_pins_json_fills_every_pin(tmp_path):
    pins_json = tmp_path / 'pins_resolved.json'
    pins_json.write_text(json.dumps({'versions': {'unsloth': '1', 'unsloth_zoo': '2', 'transformers': '3', 'trl': '4', 'peft': '5', 'bitsandbytes': '6', 'torch': '7'}}))
    import subprocess
    out = tmp_path / 'nb.ipynb'
    rows = tmp_path / 'rows'
    rows.mkdir()
    (rows / 'train.jsonl').write_text(''.join(to_lines(make_rows(4))))
    (rows / 'valid.jsonl').write_text(''.join(to_lines(make_rows(2, 'v'))))
    res = subprocess.run([sys.executable, str(BUILDER), '--train', str(rows / 'train.jsonl'), '--valid', str(rows / 'valid.jsonl'), '--out', str(out), '--pins-json', str(pins_json),
                          '--save-steps', '25', '--epochs', '3', '--run-name', 'r4x'], capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': ''})
    assert res.returncode == 0, res.stderr
    s = settings(json.loads(out.read_text()))
    assert s['PINS'] == {'unsloth': '1', 'unsloth_zoo': '2', 'transformers': '3', 'trl': '4', 'peft': '5', 'bitsandbytes': '6'}
    assert (s['SAVE_STEPS'], s['EVAL_STEPS'], s['EPOCHS'], s['RUN_NAME']) == (25, 25, 3, 'r4x') and embedded_meta(json.loads(out.read_text()))['placeholder'] is None
    bad = subprocess.run([sys.executable, str(BUILDER)], capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': ''})
    assert bad.returncode != 0 and '--train' in (bad.stderr + bad.stdout)
    both = subprocess.run([sys.executable, str(BUILDER), '--fixture', '--train', str(rows / 'train.jsonl'), '--valid', str(rows / 'valid.jsonl')], capture_output=True, text=True,
                          env={**os.environ, 'PYTHONPATH': ''})
    assert both.returncode != 0


def test_finish_uploads_to_drive_before_download_and_releases_last(nb_fixture):
    fin = code(nb_fixture, 'from google.colab import files')
    assert fin.index('STORE.upload(GGUF_PATH') < fin.index('files.download(GGUF_PATH)')
    assert fin.index("STORE.upload(f\"{E['OUT_LOCAL']}/{name}\")") < fin.index('files.download(f"')
    release = code(nb_fixture, 'if RELEASE_RUNTIME:')
    assert 'time.sleep(RELEASE_SLEEP_SECONDS)' in release and release.index('time.sleep') < release.index('runtime.unassign()')
    all_cells = [body(c) for c in cells(nb_fixture)]
    assert all_cells.index(fin) < all_cells.index(release) and all_cells[-1] == release                    # the release is the last cell
    assert settings(nb_fixture)['RELEASE_SLEEP_SECONDS'] >= 600
    top = cells(nb_fixture, 'markdown')[0]
    for text in ('İzin ver', 'RESUME = True', 'smoke_report.json', 'Q4_K_M', 'pins_resolved.json', 'not measured'):
        assert text in top, text


def test_run_logged_echoes_status_lines_only_and_prints_the_tail_on_failure(tmp_path, capsys):
    script = tmp_path / 's.py'
    script.write_text("import sys\nfor i in range(100):\n    print('noise line', i)\nprint('STATUS step=1 loss=2.5')\nsys.stdout.flush()\nprint('traceback: boom', file=sys.stderr)\nsys.exit(int(sys.argv[1]))\n")
    seen = []
    log = tmp_path / 'logs' / 'run.log'
    assert C.run_logged([sys.executable, str(script), '0'], str(log), on_line=seen.append) == 0
    out = capsys.readouterr().out
    assert out.strip() == 'STATUS step=1 loss=2.5' and seen == ['STATUS step=1 loss=2.5'] and 'noise line 99' in log.read_text()
    with pytest.raises(SystemExit, match='exit code 3'):
        C.run_logged([sys.executable, str(script), '3'], str(log))
    out = capsys.readouterr().out
    assert 'traceback: boom' in out and 'noise line 99' in out and 'noise line 10\n' not in out         # the tail, not the whole log
    assert max(len(line) for line in out.splitlines()) <= 300
    C.status('x ' * 300)
    assert len(capsys.readouterr().out.strip()) <= 200 and C.parse_kv('STATUS saved step=40 tar=/a/b best=40') == {'step': '40', 'tar': '/a/b', 'best': '40'}


def test_cells_print_only_short_status_lines(nb_fixture):
    for src in cells(nb_fixture):
        if src.startswith('%%writefile'):
            continue
        for line in body(src).splitlines():
            if re.match(r'\s*print\(', line):
                pytest.fail('driver cells use C.status, not print: ' + line.strip())
    train = written_module(nb_fixture, 'train_gemma4.py')
    assert not re.findall(r'^\s*print\(', train, re.M) and 'tqdm' not in train.replace('disable_tqdm=True', '')


def test_templates_are_derived_not_hard_coded_in_the_training_script(nb_fixture):
    train = written_module(nb_fixture, 'train_gemma4.py')
    assert '<|turn>' not in train and '<turn|>' not in train and '<start_of_turn>' not in train        # markers come from the tokenizer's own template at run time
    assert "tok.chat_template" in train and "TEMPLATE_SHA" in train


# ---- every cell, in order, against fakes: Colab modules, in-memory Drive, a mini trainer --------------------------------------------------------------------
# The real train_gemma4.py and export_gguf.py run in-process (fake unsloth/trl/torch/datasets around the real HF tokenizer with the stand-in template),
# so a NameError, a wrong key between cells or a broken STATUS protocol shows up here and not in the first Colab session.

class FakeDS(list):
    @classmethod
    def from_list(cls, rows):
        return cls(dict(r) for r in rows)


class FakeTrainer:
    """Just enough of trl.SFTTrainer + transformers.Trainer: tokenises `text`, loops over steps, calls the callbacks in Trainer order, writes checkpoints."""

    def __init__(self, model=None, processing_class=None, train_dataset=None, eval_dataset=None, args=None, callbacks=()):
        self.processing_class, self.args, self.callbacks = processing_class, args, list(callbacks)
        tokenise = lambda ds: FakeDS({**r, 'input_ids': processing_class(r['text'])['input_ids']} for r in ds)
        self.train_dataset, self.eval_dataset = tokenise(train_dataset), tokenise(eval_dataset)
        self.state = types.SimpleNamespace(global_step=0, max_steps=0, log_history=[])
        self.control = types.SimpleNamespace(should_evaluate=False, should_save=False)

    def _each(self, name, **kw):
        for cb in self.callbacks:
            getattr(cb, name)(self.args, self.state, self.control, **kw)

    best_at = 5                       # the valid loss is lowest at this step and rises on both sides: the last step is not the best one

    def _eval(self):
        loss = round(0.1 + 0.05 * abs(self.state.global_step - self.best_at), 4)
        self.state.log_history.append({'eval_loss': loss, 'step': self.state.global_step})
        self._each('on_evaluate', metrics={'eval_loss': loss})

    def _save(self):
        d = write_checkpoint(self.args.output_dir, self.state.global_step)
        (d / 'trainer_state.json').write_text(json.dumps({'global_step': self.state.global_step, 'log_history': self.state.log_history}))
        self._each('on_save')

    def train(self, resume_from_checkpoint=None):
        a, s = self.args, self.state
        if resume_from_checkpoint:
            st = json.loads((Path(resume_from_checkpoint) / 'trainer_state.json').read_text())
            s.global_step, s.log_history = st['global_step'], st['log_history']
        per_epoch = -(-len(self.train_dataset) // (a.per_device_train_batch_size * a.gradient_accumulation_steps))
        s.max_steps = a.max_steps if a.max_steps > 0 else per_epoch * a.num_train_epochs
        self._each('on_train_begin')
        if a.eval_on_start and not resume_from_checkpoint:
            self._eval()
        while s.global_step < s.max_steps:
            self._each('on_step_begin')
            s.global_step += 1
            self.control.should_evaluate = s.global_step % a.eval_steps == 0
            self.control.should_save = s.global_step % a.save_steps == 0
            for cb in self.callbacks:
                cb.on_step_end(a, s, self.control)
            if s.global_step % a.logging_steps == 0:
                s.log_history.append({'loss': 3.0 / s.global_step, 'step': s.global_step})
                self._each('on_log', logs={'loss': 3.0 / s.global_step, 'learning_rate': 1e-4})
            if self.control.should_evaluate:
                self._eval()
            if self.control.should_save:
                self._save()
        return types.SimpleNamespace(training_loss=0.5)


def fake_train_on_responses_only(trainer, instruction_part, response_part):
    """unsloth's rule: match the marker token sequences in input_ids, keep what follows a response marker until the next instruction marker."""
    tok = trainer.processing_class
    resp = tok(response_part, add_special_tokens=False)['input_ids']
    instr = tok(instruction_part, add_special_tokens=False)['input_ids']

    def labels_of(ids):
        labels, live, i = [-100] * len(ids), False, 0
        while i < len(ids):
            if ids[i:i + len(resp)] == resp:
                live, i = True, i + len(resp)
            elif ids[i:i + len(instr)] == instr:
                live, i = False, i + len(instr)
            else:
                labels[i] = ids[i] if live else -100
                i += 1
        return labels
    for ds in (trainer.train_dataset, trainer.eval_dataset):
        for row in ds:
            row['labels'] = labels_of(row['input_ids'])
    return trainer


class Harness:
    def __init__(self, tmp_path, monkeypatch, hf_tok, drive):
        self.tmp, self.mp, self.tok, self.drive = tmp_path, monkeypatch, hf_tok, drive
        self.work, self.llama = tmp_path / 'work', tmp_path / 'llama.cpp'
        self.downloads, self.slept, self.unassigned, self.calls, self.model_calls, self.lines = [], [], [], [], [], []
        (self.llama / '.git').mkdir(parents=True, exist_ok=True)
        (self.llama / 'build/bin').mkdir(parents=True, exist_ok=True)
        (self.llama / 'convert_hf_to_gguf.py').write_text('# fake')
        (self.llama / 'build/bin/llama-quantize').write_text('# fake')
        (tmp_path / 'hub_snapshot').mkdir(exist_ok=True)
        self.install_fakes()

    def sub(self, text):
        return text.replace('/content/work', str(self.work)).replace('/content/llama.cpp', str(self.llama)).replace("'/content'", repr(str(self.tmp)))

    def install_fakes(self):
        import subprocess
        mp, h = self.mp, self
        real_run = subprocess.run
        commit = B.LLAMA_CPP_COMMIT

        def fake_run(cmd, *a, **kw):
            h.calls.append([str(c) for c in cmd])
            out = ''
            if cmd[0] == 'nvidia-smi':
                out = 'NVIDIA A100-SXM4-40GB, 40960 MiB\n'
            elif cmd[0] == 'git':
                out = (commit[:7] if '--short' in cmd else commit) + '\n' if 'rev-parse' in cmd else ''
            elif cmd[0] == sys.executable and cmd[1:3] == ['-m', 'pip']:
                out = 'torch==2.99\n' if cmd[3] == 'freeze' else ''
            elif cmd[0] == sys.executable and cmd[1] == '-c':
                names = ['unsloth', 'unsloth_zoo', 'transformers', 'trl', 'peft', 'bitsandbytes', 'accelerate', 'datasets']
                known = {'unsloth': '2026.10.6', 'transformers': '5.16.1', 'trl': '1.15.0'}
                out = json.dumps({**{n: known.get(n, '9.9.9') for n in names}, 'torch': '2.99.0', 'cuda': '12.8'}) + '\n'
            elif str(cmd[1]).endswith('convert_hf_to_gguf.py'):
                Path(cmd[cmd.index('--outfile') + 1]).write_bytes(b'GGUF-bf16' * 10)
            elif str(cmd[0]).endswith('llama-quantize'):
                Path(cmd[2]).write_bytes(b'GGUF-q4' * 10)
            else:
                return real_run(cmd, *a, **kw)
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr='')
        mp.setattr(subprocess, 'run', fake_run)
        mp.setattr('time.sleep', lambda s: h.slept.append(s))

        colab = types.ModuleType('google.colab')
        colab.auth = types.SimpleNamespace(authenticate_user=lambda: h.calls.append(['authenticate_user']))
        colab.userdata = types.SimpleNamespace(get=lambda k: (_ for _ in ()).throw(RuntimeError('no secret')))
        colab.files = types.SimpleNamespace(download=lambda p: h.downloads.append(str(p)))
        colab.runtime = types.SimpleNamespace(unassign=lambda: h.unassigned.append(True))
        google = types.ModuleType('google')
        google.colab, google.__path__ = colab, []
        for name, mod in (('google', google), ('google.colab', colab)):
            mp.setitem(sys.modules, name, mod)

        tok = self.tok

        class FastModel:
            @staticmethod
            def from_pretrained(model_name=None, **kw):
                h.model_calls.append(('from_pretrained', model_name, kw))
                model = types.SimpleNamespace(save_pretrained_merged=lambda path, t, save_method=None: (os.makedirs(path, exist_ok=True), Path(path, 'model.safetensors').write_bytes(b'm' * 64)))
                return model, tok

            @staticmethod
            def get_peft_model(model, **kw):
                h.model_calls.append(('get_peft_model', kw))
                return model

        fake = {
            'unsloth': types.SimpleNamespace(FastModel=FastModel),
            'unsloth.chat_templates': types.SimpleNamespace(train_on_responses_only=fake_train_on_responses_only),
            'torch': types.SimpleNamespace(__version__='2.99.0', version=types.SimpleNamespace(cuda='12.8'), cuda=types.SimpleNamespace(
                max_memory_allocated=lambda: 12 * 2**30, reset_peak_memory_stats=lambda: None, get_device_name=lambda i: 'NVIDIA A100-SXM4-40GB', empty_cache=lambda: None)),
            'datasets': types.SimpleNamespace(Dataset=FakeDS),
            'transformers': types.SimpleNamespace(TrainerCallback=object),
            'trl': types.SimpleNamespace(SFTConfig=lambda **kw: types.SimpleNamespace(**kw), SFTTrainer=FakeTrainer),
            'huggingface_hub': types.SimpleNamespace(HfApi=lambda: types.SimpleNamespace(model_info=lambda *a, **k: types.SimpleNamespace(sha=B.BASE_REVISION)),
                                                     snapshot_download=lambda *a, **k: str(h.tmp / 'hub_snapshot')),
        }
        for name, mod in fake.items():
            mp.setitem(sys.modules, name, mod)
        mp.delitem(sys.modules, 'r4_common', raising=False)

    def run_script(self, cmd, on_line):
        """What run_logged does for the two scripts, in-process: STATUS lines are echoed and handed to on_line."""
        import contextlib
        import io
        import runpy
        buf = io.StringIO()
        self.mp.setattr(sys, 'argv', [cmd[1], *cmd[2:]])
        with contextlib.redirect_stdout(buf):
            runpy.run_path(cmd[1], run_name='__main__')
        for line in buf.getvalue().splitlines():
            if line.startswith('STATUS '):
                self.lines.append(line)
                print(line[:200])
                if on_line:
                    on_line(line)

    def patch_common(self, C_mod):
        h = self

        def run_logged(cmd, log_path, env=None, cwd=None, on_line=None, tail=40):
            if len(cmd) > 1 and str(cmd[1]).endswith(('train_gemma4.py', 'export_gguf.py')):
                h.calls.append([str(c) for c in cmd])
                h.run_script(cmd, on_line)
            else:
                h.calls.append([str(c) for c in cmd])
            return 0
        C_mod.run_logged = run_logged
        C_mod.GoogleDrive = lambda: h.drive
        C_mod.gguf_summary = lambda path, llama: {'architecture': 'gemma4', 'file_type': 15, 'tensors': 3, 'chat_template_sha256': 'x', 'chat_template_has_turn_marker': True}

    def run(self, nb, patch=None):
        ns = {'__name__': 'nb'}
        self.work.mkdir(parents=True, exist_ok=True)
        for src in cells(nb):
            src = self.sub(src)
            if patch:
                src = patch(src)
            if src.startswith('%%writefile'):
                head, text = src.split('\n', 1)
                Path(head.split()[1]).write_text(text)
                continue
            exec(compile(body(src), 'cell', 'exec'), ns)
            if 'import r4_common as C' in src:
                self.patch_common(ns['C'])
        return ns


@pytest.fixture
def harness(tmp_path, monkeypatch, hf_tok):
    h = Harness(tmp_path, monkeypatch, hf_tok, FakeDrive(tmp_path))
    yield h
    sys.modules.pop('r4_common', None)


FULL_PINS = {'unsloth': '2026.10.6', 'transformers': '5.16.1', 'trl': '1.15.0', 'unsloth_zoo': '9.9.9', 'peft': '9.9.9', 'bitsandbytes': '9.9.9'}     # what the fake pip reports


def notebook_for_run(tmp_path, smoke, **over):
    return build(tmp_path, make_rows(30 if smoke else 8), make_rows(9 if smoke else 3, 'v'), smoke=smoke, **{'save_steps': 2, 'eval_steps': 2, 'pins': FULL_PINS, **over})


def test_smoke_notebook_runs_every_cell_with_the_drive_round_trip_and_export_check(harness, capsys):
    nb = notebook_for_run(harness.tmp, smoke=True)
    ns = harness.run(nb)
    out = capsys.readouterr().out
    assert out.count('STATUS') == len([l for l in out.splitlines() if l.startswith('STATUS')]) and all(len(l) <= 200 for l in out.splitlines())
    names = {f['name'] for f in harness.drive.files.values()}
    assert {'checkpoint-10.tar', 'adapter-step-5.tar', 'adapter-step-10.tar', 'adapter-step-12.tar', 'run_meta.json', 'pins_resolved.json', 'smoke_report.json', 'metrics.json',
            'export.json', 'SHA256SUMS'} <= names
    assert not any(n.endswith('.gguf') for n in names), 'the smoke GGUF is not uploaded'
    assert ns['SAVED'] == [5, 10, 12] and ns['BEGINS'] == [0, 10] and ns['FINAL_STEP'] == 12
    rep = ns['SMOKE_REPORT']
    assert rep['round_trip'] == {'restored': 'checkpoint-10', 'resume_begin_step': 10, 'resume_final_step': 12}
    assert rep['seconds_per_step_mean'] >= 0 and rep['estimate_full_train_seconds'] >= 0 and rep['full_train_rows'] == 30 and rep['peak_gpu_gb'] == 12
    smoke_json = json.loads((harness.work / 'out/smoke_report.json').read_text())
    assert len(smoke_json['seconds_per_step_all']) == 10
    pins = json.loads((harness.work / 'out/pins_resolved.json').read_text())
    assert pins['versions']['unsloth'] == '2026.10.6' and pins['base_revision_resolved'] == B.BASE_REVISION and pins['llama_cpp_commit'] == B.LLAMA_CPP_COMMIT
    assert len(pins['chat_template_sha256']) == 64 and pins['markers'][0] == '<|turn>user\n'
    # the fake valid loss is lowest at step 5 and worse at 10 and 12: the frozen choice is the lowest valid loss, not the last step, and the resume leg still
    # knows it (the checkpoint tar of the best step is what the pruner must keep)
    assert ns['BEST_STEP'] == 5 and (harness.work / 'adapters/step-5/adapter_model.safetensors').exists()
    assert [C.parse_kv(l)['best'] for l in harness.lines if l.startswith('STATUS saved ')] == ['5', '5', '5']
    assert any(l.startswith('STATUS train_begin step=10 ') and l.endswith('best_step=5') for l in harness.lines)
    assert 'checkpoint-5.tar' in names and 'checkpoint-12.tar' in names
    exported = json.loads((harness.work / 'out/export.json').read_text())
    assert exported['step'] == 5 and exported['gguf'].endswith('-s5-q4_k_m.gguf') and exported['quant'] == 'Q4_K_M'
    assert harness.downloads and all(Path(p).name in {'pins_resolved.json', 'metrics.json', 'export.json', 'SHA256SUMS', 'smoke_report.json'} for p in harness.downloads)
    assert harness.slept == [900] and harness.unassigned == [True]
    # the load calls: 4-bit, pinned revision, LoRA settings
    load = next(c for c in harness.model_calls if c[0] == 'from_pretrained' and c[1] == 'google/gemma-4-12B-it')
    assert load[2]['load_in_4bit'] is True and load[2]['revision'] == B.BASE_REVISION and load[2]['max_seq_length'] == 16384
    peft = next(c for c in harness.model_calls if c[0] == 'get_peft_model')[1]
    assert (peft['r'], peft['lora_alpha'], peft['random_state'], peft['finetune_vision_layers']) == (16, 32, 13, False)
    export_load = [c for c in harness.model_calls if c[0] == 'from_pretrained' and c[1] != 'google/gemma-4-12B-it']
    assert export_load and export_load[0][1].endswith('export_adapter')
    cfg = json.loads((harness.work / 'export_adapter/adapter_config.json').read_text())
    assert cfg['base_model_name_or_path'].endswith('hub_snapshot'), 'the merge uses the pinned base snapshot'


def test_full_notebook_runs_uploads_the_gguf_then_downloads_then_releases(harness, capsys):
    nb = notebook_for_run(harness.tmp, smoke=False)
    ns = harness.run(nb)
    names = {f['name']: f for f in harness.drive.files.values()}
    gguf = [n for n in names if n.endswith('.gguf')]
    assert gguf == [f"buffett-gemma4-12b-ft-r4-r4-s{ns['BEST_STEP']}-q4_k_m.gguf"] and names[gguf[0]]['props']['kind'] == 'gguf'
    assert ns['FINAL_STEP'] == 4 and ns['SAVED'] == [2, 4] and ns['BEGINS'] == [0] and ns['SMOKE_REPORT'] == {}
    assert [Path(p).name for p in harness.downloads][-1] == gguf[0]
    assert 'smoke_report.json' not in names and 'checkpoint-4.tar' in names
    meta = json.loads(next(f for f in harness.drive.files.values() if f['name'] == 'run_meta.json')['path'].read_text())
    assert meta['status'] == 'finished' and meta['gguf_sha256'] == ns['EXPORT']['sha256'] and meta['best_step'] == ns['BEST_STEP']
    metrics = json.loads((harness.work / 'out/metrics.json').read_text())
    assert metrics['best_valid_loss'] == min(h['eval_loss'] for h in ns['HISTORY'] if h['step'] > 0) and metrics['export']['gguf'] == gguf[0]
    assert metrics['data']['train']['rows'] == 8 and metrics['lora']['rank'] == 16
    assert harness.slept == [900] and harness.unassigned == [True]
    # drive uploads came before the downloads: every file in downloads is on Drive
    on_drive = set(names)
    assert {Path(p).name for p in harness.downloads} <= on_drive


def test_resume_after_a_disconnect_restores_from_drive_and_refuses_the_wrong_cases(harness, tmp_path_factory, monkeypatch, hf_tok):
    nb = notebook_for_run(harness.tmp, smoke=False)
    harness.run(nb)
    # a fresh runtime, same Drive: without RESUME the folder is refused
    h2 = Harness(tmp_path_factory.mktemp('second'), monkeypatch, hf_tok, harness.drive)
    with pytest.raises(SystemExit, match='already holds'):
        h2.run(nb)
    sys.modules.pop('r4_common', None)
    h3 = Harness(tmp_path_factory.mktemp('third'), monkeypatch, hf_tok, harness.drive)
    ns = h3.run(nb, patch=lambda s: s.replace('RESUME = False', 'RESUME = True', 1))
    assert ns['BEGINS'] == [4] and ns['RESUME_PATH'].endswith('checkpoint-4') and ns['FINAL_STEP'] == 4          # continued from the restored step, nothing left to train
    assert ns['HISTORY'] and ns['BEST_STEP'] > 0 and h3.unassigned == [True]
    # other data: the identity check refuses the resume
    sys.modules.pop('r4_common', None)
    other = build(harness.tmp, make_rows(8, 'z'), make_rows(3, 'zv'), save_steps=2, eval_steps=2)
    h4 = Harness(tmp_path_factory.mktemp('fourth'), monkeypatch, hf_tok, harness.drive)
    with pytest.raises(SystemExit, match='differs'):
        h4.run(other, patch=lambda s: s.replace('RESUME = False', 'RESUME = True', 1))
    # other training settings: refused as well
    sys.modules.pop('r4_common', None)
    h6 = Harness(tmp_path_factory.mktemp('sixth'), monkeypatch, hf_tok, harness.drive)
    with pytest.raises(SystemExit, match='train_settings'):
        h6.run(nb, patch=lambda s: s.replace('RESUME = False', 'RESUME = True', 1).replace('LR = 1e-4', 'LR = 2e-4', 1))
    # RESUME with an empty Drive
    sys.modules.pop('r4_common', None)
    h5 = Harness(tmp_path_factory.mktemp('fifth'), monkeypatch, hf_tok, FakeDrive(tmp_path_factory.mktemp('emptydrive')))
    with pytest.raises(SystemExit, match='no checkpoint'):
        h5.run(nb, patch=lambda s: s.replace('RESUME = False', 'RESUME = True', 1))


def test_notebook_stops_before_training_on_unfilled_pins_or_a_small_gpu(harness):
    nb = notebook_for_run(harness.tmp, smoke=False, pins={})
    with pytest.raises(SystemExit, match='unfilled pins'):
        harness.run(nb)                       # unsloth_zoo, peft and bitsandbytes are still None and REQUIRE_ALL_PINS is True for a full run
    assert not any('train_gemma4.py' in ' '.join(c) for c in harness.calls) and not any(c[:3] == [sys.executable, '-m', 'pip'] for c in harness.calls)
    sys.modules.pop('r4_common', None)
    real_run = __import__('subprocess').run
    harness.mp.setattr('subprocess.run', lambda cmd, *a, **k: __import__('subprocess').CompletedProcess(cmd, 0, stdout='NVIDIA L4, 23034 MiB\n', stderr='')
                       if cmd[0] == 'nvidia-smi' else real_run(cmd, *a, **k))
    with pytest.raises(AssertionError, match='needs the A100'):
        harness.run(notebook_for_run(harness.tmp, smoke=False))


def test_gguf_summary_reads_architecture_file_type_tensors_and_the_chat_template(tmp_path):
    import gguf
    import numpy as np
    template = '<|turn>user\n{{ messages }}<turn|>\n'
    path = tmp_path / 'tiny.gguf'
    writer = gguf.GGUFWriter(str(path), 'gemma4')
    writer.add_file_type(15)
    writer.add_string('tokenizer.chat_template', template)
    writer.add_tensor('t.weight', np.zeros((4, 4), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    summary = C.gguf_summary(str(path), str(tmp_path / 'no_llama_checkout'))              # falls back to the installed gguf-py
    assert summary == {'architecture': 'gemma4', 'file_type': 15, 'tensors': 1, 'chat_template_sha256': hashlib.sha256(template.encode()).hexdigest(),
                       'chat_template_has_turn_marker': True}


def test_kernel_cells_are_guarded_and_failures_print_plain_text(nb_fixture):
    """Smoke lesson: an error output hides the log tail in a cross-origin iframe; kernel cells run through %%guarded."""
    s = code(nb_fixture, 'RUN_NAME = ')
    assert 'def guarded(line, cell)' in s and "'CELL_FAILED '" in s and 'failure.json' in s
    kernel = [c for c in cells(nb_fixture) if not c.startswith('%%writefile')][1:]
    kernel = [c for c in kernel if not c.startswith('# @title')]
    assert kernel and all(c.startswith('%%guarded\n') for c in kernel), [c[:40] for c in kernel if not c.startswith('%%guarded')]
