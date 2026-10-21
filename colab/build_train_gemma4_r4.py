"""Build colab/train_gemma4_12b_r4.ipynb: FT-r4, a QLoRA fine-tune of google/gemma-4-12B-it on the ft_finance rows (Colab A100, Unsloth).

python colab/build_train_gemma4_r4.py --train <dir>/train.jsonl --valid <dir>/valid.jsonl
python colab/build_train_gemma4_r4.py --train ... --valid ... --smoke --out colab/train_gemma4_12b_r4_smoke.ipynb
python colab/build_train_gemma4_r4.py --train ... --valid ... --pins-json pins_resolved.json   # after the smoke run
python colab/build_train_gemma4_r4.py --fixture                                                # placeholder from synthetic rows; it refuses to train

The notebook embeds the rows (gzip+base64, sha256 checked), needs no upload step and no Drive mount: checkpoints and the GGUF go to Drive through
googleapiclient. Every GPU step runs in its own subprocess; the kernel never imports torch. Rows are `messages` (system, user, assistant) or the
keys system/user/assistant; the training text is the tokenizer's chat template and the loss covers the assistant turn only.

Token counts here use the local tokenizer.json (revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7) with a stand-in of Gemma 4's turn format:
the local copy has no chat template file. The notebook recounts with the real template and refuses on a gap above TOKEN_TOLERANCE.
Rebuilding is the only way to change the notebook: do not edit the JSON.
"""
import argparse
import base64
import gzip
import hashlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'colab/train_gemma4_12b_r4.ipynb'
TOKENIZER_DIR = ROOT / 'data/tokenizer/gemma4_12b_it'
BASE_REVISION = '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'   # google/gemma-4-12B-it main; also the revision of the local tokenizer copy
MAX_SEQ = 16384
SEED = 13
SMOKE_ROWS, SMOKE_VALID_ROWS = 20, 6
LLAMA_CPP_COMMIT = '46ca246de9bb1c35269722a6240d37d9dfd79cad'
PIN_KEYS = ('unsloth', 'unsloth_zoo', 'transformers', 'trl', 'peft', 'bitsandbytes')
DEFAULT_PINS = {'unsloth': '2026.10.6', 'transformers': '5.16.1', 'trl': '1.15.0'}   # FT-r3 log; the rest is filled from the smoke's pins_resolved.json
FIXTURE_PLACEHOLDER = 'FIXTURE ROWS ONLY: synthetic text, rebuild with --train/--valid'

# Gemma 4's turn format as the local tokenizer_config declares it (sot `<|turn>`, eot `<turn|>`, start anchor `<|turn>model\n`); the same text
# as render_standin(). The real chat template replaces it in the notebook.
STANDIN_TEMPLATE = ("{{ bos_token }}{% for m in messages %}<|turn>{{ 'model' if m['role'] == 'assistant' else m['role'] }}\n"
                    "{{ m['content'] | trim }}<turn|>\n{% endfor %}{% if add_generation_prompt %}<|turn>model\n{% endif %}")


def render_standin(messages, add_generation_prompt=False):
    out = '<bos>'
    for m in messages:
        out += f"<|turn>{'model' if m['role'] == 'assistant' else m['role']}\n{m['content'].strip()}<turn|>\n"
    return out + ('<|turn>model\n' if add_generation_prompt else '')


# ---------------------------------------------------------------------------------------------------------------------------------------------
# notebook sources (cells). `@@NAME@@` markers are replaced by the builder; the code cells are plain Python (3.12 on Colab).
# ---------------------------------------------------------------------------------------------------------------------------------------------

TOP_MD = r'''# BuffettRAG: QLoRA fine-tune of Gemma 4 12B (FT-r4) on the ft_finance rows, Colab A100 + Drive API
Unsloth QLoRA on `google/gemma-4-12B-it` (pinned revision), LoRA r16 / alpha 32 / lr 1e-4, batch 1 x accum 4, max_seq 16,384. The rows are the exact chat the finance reader route serves; the training text is the tokenizer's own chat template and the loss covers the assistant turn only. The pre-registered choice is the checkpoint with the lowest valid loss (full valid set at every eval step), frozen before any eval.

Everything heavy runs in a subprocess (training, merge + GGUF); the kernel never imports torch. Drive is used through `googleapiclient` only (no mount). The cells print short `STATUS` lines; the full logs are in `/content/work/logs` and on Drive.

**Run procedure**
1. Runtime type: A100 (the notebook asks for it). Keep the tab open.
2. Optional: a Colab secret `HF_TOKEN` with notebook access, if `google/gemma-4-12B-it` is gated for the account.
3. Run all. **The first run asks for a Drive permission dialog ("İzin ver") in the Colab tab: the user approves it.** Timing marks start after it.
4. Smoke first (`SMOKE = True` in section a, or the `_smoke` notebook): 20 rows, 10 steps, then a Drive round trip (wipe local checkpoints, restore from Drive, resume to step 12), then merge + GGUF Q4_K_M with the pinned llama.cpp. It writes `pins_resolved.json` and `smoke_report.json` (seconds per step, peak GPU memory, an estimate of the full run) and downloads both. Paste the resolved pins back with `build_train_gemma4_r4.py --pins-json`.
5. Full run: `SMOKE = False` and all pins filled (`REQUIRE_ALL_PINS`). Every `SAVE_STEPS` a checkpoint tar goes to `MyDrive/buffettrag/ftr4-gemma4-12b-<RUN_NAME>/`; the newest two are kept on Drive plus the best one, adapters are kept for every step.
6. **After a disconnect:** reconnect, set `RESUME = True`, Run all. The notebook restores the newest complete checkpoint from Drive (sha256 and md5 checked), refuses other data or settings, and continues. Without `RESUME`, a Drive folder that already holds checkpoints is refused.
7. At the end: GGUF (Q4_K_M) + `SHA256SUMS` + `export.json` + `metrics.json` go to Drive, then to Downloads (`files.download`); the last cell waits about 15 minutes and releases the runtime.

Timings of this run are not measured yet: the ETA comes from `smoke_report.json`.
'''

SETTINGS_MD = '## a. Settings'

SETTINGS_PY = r'''import os
try:
    os.makedirs('/content/work', exist_ok=True)  # the %%writefile cells below need the folder (IPython does not create it)
except OSError:
    pass  # not on Colab (local stub tests)
FAILED = None
try:
    from IPython.core.magic import register_cell_magic

    @register_cell_magic
    def guarded(line, cell):
        """Run the cell; on failure write the traceback as plain text and stop every later guarded cell.

        Smoke lesson: an error output makes Colab render the whole cell output in a cross-origin iframe, so the log tail
        printed before the error could not be read from Chrome. The failing cell therefore ends without an error output,
        its message also goes to Drive as failure.json, and the next guarded cell raises to stop Run all."""
        import sys, traceback
        ns = get_ipython().user_ns
        if ns.get('FAILED'):
            raise RuntimeError('stopped: an earlier cell failed, its plain-text output says why: ' + ns['FAILED'])
        try:
            exec(compile(cell, '<cell>', 'exec'), ns)
        except BaseException as e:
            text = traceback.format_exc()
            ns['FAILED'] = (type(e).__name__ + ': ' + str(e))[:300]
            sys.stdout.write('CELL_FAILED ' + ns['FAILED'] + '\n' + text[-4000:] + '\n')
            if ns.get('STORE') is not None:
                try:
                    ns['STORE'].write_json('failure.json', {'error': ns['FAILED'], 'traceback': text[-20000:]})
                except Exception:
                    pass
except Exception:
    pass  # not in IPython (local stub tests)
RUN_NAME = '@@RUN_NAME@@'              # drives the Drive folder, the GGUF name and every record
SMOKE = @@SMOKE@@                    # True: 20 rows, 10 steps, Drive round trip, export check, no GGUF upload; writes pins_resolved.json + smoke_report.json
RESUME = False                       # True: continue from the newest complete checkpoint on Drive (after a disconnect)
ALLOW_PLACEHOLDER = False            # a notebook built with --fixture refuses to train
RELEASE_RUNTIME = True               # last cell: sleep, then runtime.unassign()
RELEASE_SLEEP_SECONDS = 900          # time for files.download to finish before the runtime goes away
BASE_MODEL = 'google/gemma-4-12B-it'
BASE_REVISION = '@@BASE_REVISION@@'  # main of google/gemma-4-12B-it when the local tokenizer copy was taken; the unsloth 4-bit mirror would need its own hash
PINS = @@PINS@@
REQUIRE_ALL_PINS = @@REQUIRE_ALL_PINS@@   # True: refuse to train while a pin is None (the smoke run writes the resolved values)
LLAMA_CPP_COMMIT = '@@LLAMA_CPP_COMMIT@@'  # has LLM_ARCH_GEMMA4 and Gemma4UnifiedModel (conversion/gemma.py); converter and quantizer are built from it
MAX_SEQ = @@MAX_SEQ@@
TOKEN_TOLERANCE = 16                 # |recount with the real template - builder count| per row; above it the run stops
# LoRA / optimiser
EPOCHS = @@EPOCHS@@
MAX_STEPS = None                     # None: EPOCHS decide; int: stop at that optimizer step
LR = 1e-4
RANK = 16
ALPHA = 32
BATCH = 1
GRAD_ACCUM = 4
WARMUP_STEPS = 10
SEED = 13
SAVE_STEPS = @@SAVE_STEPS@@          # a checkpoint tar goes to Drive every SAVE_STEPS
EVAL_STEPS = @@EVAL_STEPS@@          # full valid set, so the lowest valid loss picks the checkpoint
KEEP_DRIVE_CHECKPOINTS = 2           # newest full checkpoints kept on Drive besides the best one; adapter tars are never deleted
# smoke
SMOKE_ROWS = 20
SMOKE_VALID_ROWS = 6
SMOKE_STEPS = 10
SMOKE_SAVE_STEPS = 5
SMOKE_RESUME_EXTRA_STEPS = 2         # second leg after the restore from Drive: steps SMOKE_STEPS .. SMOKE_STEPS + 2
EXPORT_QUANT = 'Q4_K_M'
DRIVE_ROOT = 'buffettrag'            # My Drive/<DRIVE_ROOT>/ftr4-gemma4-12b-<RUN_NAME>
DRIVE_FOLDER_ID = None               # set to use an existing Drive folder directly
SETTING_NAMES = ['RUN_NAME', 'SMOKE', 'RESUME', 'ALLOW_PLACEHOLDER', 'RELEASE_RUNTIME', 'RELEASE_SLEEP_SECONDS', 'BASE_MODEL', 'BASE_REVISION', 'PINS',
                 'REQUIRE_ALL_PINS', 'LLAMA_CPP_COMMIT', 'MAX_SEQ', 'TOKEN_TOLERANCE', 'EPOCHS', 'MAX_STEPS', 'LR', 'RANK', 'ALPHA', 'BATCH', 'GRAD_ACCUM',
                 'WARMUP_STEPS', 'SEED', 'SAVE_STEPS', 'EVAL_STEPS', 'KEEP_DRIVE_CHECKPOINTS', 'SMOKE_ROWS', 'SMOKE_VALID_ROWS', 'SMOKE_STEPS',
                 'SMOKE_SAVE_STEPS', 'SMOKE_RESUME_EXTRA_STEPS', 'EXPORT_QUANT', 'DRIVE_ROOT', 'DRIVE_FOLDER_ID']
'''

DATA_MD = '## b. Embedded data (gzip + base64, sha256 checked in the next cells)'

COMMON_PY = r'''%%writefile /content/work/r4_common.py
"""Shared helpers of the FT-r4 Gemma 4 notebook: pure Python, no torch, no Google client at import time."""
import collections
import glob
import gzip
import hashlib
import json
import math
import os
import queue
import random
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import threading
import time
import zipfile

WORK = '/content/work'
STATUS_PREFIX = 'STATUS '
MIB = 1 << 20
RESUME_FILES = ('adapter_model.safetensors', 'adapter_config.json', 'trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth')
ADAPTER_FILES = ('adapter_model.safetensors', 'adapter_config.json')


def status(msg, limit=190):
    """One short line; the notebook cells echo only these. limit=None for the lines the notebook parses (a cut-off key=value would be lost)."""
    text = ' '.join(str(msg).split())
    print(STATUS_PREFIX + (text[:limit] if limit else text), flush=True)


def parse_kv(line):
    return dict(p.split('=', 1) for p in line.split() if '=' in p)


# ---- files ------------------------------------------------------------------------------------------------------------------------------------

def sha256_file(path):
    return file_digests(path)[0]


def file_digests(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path + '.part', 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(path + '.part', path)


def load_settings(path=WORK + '/settings.json'):
    return read_json(path)


def make_tar(src_dir, tar_path, arcname, only=None):
    """Tar `src_dir` (or only the named files of it) under `arcname/`; written to .part first, so a cut-off tar never has the final name."""
    os.makedirs(os.path.dirname(tar_path) or '.', exist_ok=True)
    with tarfile.open(tar_path + '.part', 'w') as tf:
        if only:
            for name in only:
                tf.add(os.path.join(src_dir, name), arcname=f'{arcname}/{name}')
        else:
            tf.add(src_dir, arcname=arcname)
    os.replace(tar_path + '.part', tar_path)
    return tar_path


def safe_untar(tar_path, dest):
    os.makedirs(dest, exist_ok=True)
    root = os.path.realpath(dest)
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            target = os.path.realpath(os.path.join(dest, m.name))
            assert target == root or target.startswith(root + os.sep), f'unsafe path in tar: {m.name}'
            assert m.isfile() or m.isdir(), f'unsupported tar member: {m.name}'
        try:
            tf.extractall(dest, filter='data')
        except TypeError:
            tf.extractall(dest)


# ---- settings and rows ---------------------------------------------------------------------------------------------------------------------------

def pick_smoke_rows(lengths, n, seed):
    """The longest row plus a seeded sample of the others, so the smoke meets the worst memory case; sorted indices."""
    if n >= len(lengths):
        return list(range(len(lengths)))
    longest = max(range(len(lengths)), key=lambda i: (lengths[i], -i))
    rest = [i for i in range(len(lengths)) if i != longest]
    return sorted([longest] + random.Random(seed).sample(rest, n - 1))


def effective_settings(S, meta=None):
    """The settings as the subprocesses see them: smoke overrides, run tag, local dirs, the Drive path and (smoke) the picked row indices."""
    E = dict(S)
    smoke = bool(S['SMOKE'])
    E['RUN_TAG'] = S['RUN_NAME'] + ('-smoke' if smoke else '')
    E.update(WORK=WORK, OUT_LOCAL=WORK + '/out', CKPT_LOCAL=WORK + '/ckpt', ADAPTER_LOCAL=WORK + '/adapters', UPLOAD_LOCAL=WORK + '/uploads',
             LOG_DIR=WORK + '/logs', DATA_DIR=WORK + '/data', LLAMA_CPP_DIR='/content/llama.cpp')
    E['DRIVE_PATH'] = [S['DRIVE_ROOT'], 'ftr4-gemma4-12b-' + E['RUN_TAG']]
    E['TRAIN_INDICES'] = E['VALID_INDICES'] = None
    if smoke:
        E.update(MAX_STEPS=S['SMOKE_STEPS'], SAVE_STEPS=S['SMOKE_SAVE_STEPS'], EVAL_STEPS=S['SMOKE_SAVE_STEPS'], WARMUP_STEPS=1)
        if meta:
            E['TRAIN_INDICES'] = pick_smoke_rows(meta['train']['tokens'], S['SMOKE_ROWS'], S['SEED'])
            E['VALID_INDICES'] = pick_smoke_rows(meta['valid']['tokens'], S['SMOKE_VALID_ROWS'], S['SEED'])
    if meta:
        E['EXPECT_TOKENS'] = {}
        for split in ('train', 'valid'):
            idx = E[split.upper() + '_INDICES']
            E['EXPECT_TOKENS'][split] = meta[split]['tokens'] if idx is None else [meta[split]['tokens'][i] for i in idx]
    return E


def rows_to_messages(row):
    msgs = row['messages'] if 'messages' in row else [{'role': r, 'content': row[r]} for r in ('system', 'user', 'assistant')]
    assert [m['role'] for m in msgs] == ['system', 'user', 'assistant'], [m['role'] for m in msgs]
    assert all(isinstance(m['content'], str) and m['content'].strip() for m in msgs), 'empty or non-string content'
    return [{'role': m['role'], 'content': m['content']} for m in msgs]


def parse_jsonl(raw):
    return [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]


def best_step(history, available=None):
    """Step of the lowest valid loss (earliest on a tie); only steps in `available` count when given. Step 0 (eval on start) has no adapter."""
    cands = [h for h in history if h['step'] > 0 and (available is None or h['step'] in available)]
    assert cands, 'no evaluated step with a saved adapter'
    return min(cands, key=lambda h: (h['eval_loss'], h['step']))['step']


# ---- chat template ----------------------------------------------------------------------------------------------------------------------------

def derive_markers(render, bos_token=''):
    """(instruction_part, response_part) for train_on_responses_only, read from the tokenizer's own template: the generation prompt is the response
    marker, the head of a user-only rendering (without BOS) is the instruction marker."""
    probe = [{'role': 'system', 'content': 'SYSPROBE'}, {'role': 'user', 'content': 'USERPROBE'}]
    plain, gen = render(probe, False), render(probe, True)
    assert gen.startswith(plain) and len(gen) > len(plain), 'the generation prompt is not a suffix of the plain rendering'
    response_part = gen[len(plain):]
    user_only = render(probe[1:], False)
    head = user_only[:user_only.index('USERPROBE')]
    instruction_part = head[len(bos_token):] if bos_token and head.startswith(bos_token) else head
    assert instruction_part.strip() and response_part.strip() and instruction_part in plain and instruction_part != response_part
    return instruction_part, response_part


def derive_turn_end(render):
    full = render([{'role': 'user', 'content': 'UPROBE'}, {'role': 'assistant', 'content': 'APROBE'}], False)
    return full[full.index('APROBE') + len('APROBE'):]


def render_training_text(render, messages, turn_end):
    """The template's own rendering of the whole chat when it extends the serving prompt (generation prompt) by the target; else prompt + target
    + turn end. Returns (text, mode)."""
    prompt = render(messages[:-1], True)
    target = messages[-1]['content'].strip()
    full = render(messages, False)
    if full.startswith(prompt + target):
        return full, 'template'
    return prompt + target + turn_end, 'prompt+target'


def prepare_split(name, rows, render, encode, bos_id, turn_end, max_seq, expect_tokens=None, tolerance=16):
    """Training texts and token counts of one split, with the checks that make the run trustworthy: one BOS, the serving prompt is a token prefix,
    the completion is the target plus the turn end, no row above max_seq, counts within `tolerance` of the builder's."""
    texts, tokens, modes, worst = [], [], collections.Counter(), 0
    for i, row in enumerate(rows):
        msgs = rows_to_messages(row)
        text, mode = render_training_text(render, msgs, turn_end)
        prompt = render(msgs[:-1], True)
        target = msgs[-1]['content'].strip()
        assert text.startswith(prompt + target) and text[len(prompt) + len(target):].strip() == turn_end.strip(), f'{name} row {i}: completion is not target + turn end'
        ids, prompt_ids = encode(text), encode(prompt)
        assert ids[:len(prompt_ids)] == prompt_ids, f'{name} row {i}: the serving prompt tokens are not a prefix of the training tokens'
        if bos_id is not None:
            assert ids[0] == bos_id and ids[1] != bos_id, f'{name} row {i}: BOS missing or doubled'
        assert len(ids) <= max_seq, f'{name} row {i} has {len(ids)} tokens, above MAX_SEQ={max_seq}'
        if expect_tokens is not None:
            worst = max(worst, abs(len(ids) - expect_tokens[i]))
            assert abs(len(ids) - expect_tokens[i]) <= tolerance, f'{name} row {i}: {len(ids)} tokens, builder counted {expect_tokens[i]}'
        texts.append(text)
        tokens.append(len(ids))
        modes[mode] += 1
    return {'texts': texts, 'tokens': tokens, 'modes': dict(modes), 'max_diff': worst}


def check_loss_masks(name, dataset, decode, targets, turn_end, tail_ok=()):
    """Every row: the labels left unmasked decode to the target plus the turn end, nothing else (a trailing token of `tail_ok`, such as the EOS
    the trainer may append to the text, is allowed). Returns the unmasked token total."""
    squash = lambda s: ''.join(s.split())
    ok = {squash(turn_end)} | {squash(turn_end) + squash(t) for t in tail_ok if t}
    total = 0
    for i, target in enumerate(targets):
        live = [t for t in dataset[i]['labels'] if t != -100]
        got = decode(live).strip()
        want = target.strip()
        assert got.startswith(want) and squash(got[len(want):]) in ok, (name, i, got[:60], got[-40:])
        total += len(live)
    return total


# ---- checkpoints ---------------------------------------------------------------------------------------------------------------------------------

def safetensors_complete(path):
    """The file is exactly header + the data the header promises (a half-copied file fails)."""
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            if n <= 0 or 8 + n > size:
                return False
            header = json.loads(f.read(n))
        end = max((v['data_offsets'][1] for k, v in header.items() if k != '__metadata__'), default=0)
        return end > 0 and size == 8 + n + end
    except Exception:
        return False


def checkpoint_problem(d):
    """None if the checkpoint dir is complete and loadable, else the reason."""
    for name in RESUME_FILES:
        p = os.path.join(d, name)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            return f'missing or empty {name}'
    if not safetensors_complete(os.path.join(d, 'adapter_model.safetensors')):
        return 'adapter_model.safetensors incomplete'
    for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
        if not zipfile.is_zipfile(os.path.join(d, name)):      # torch.save writes a zip; a cut-off write has no central directory
            return f'{name} is not a complete torch file'
    try:
        state = read_json(os.path.join(d, 'trainer_state.json'))
        step = int(os.path.basename(d).split('-')[-1])
        if state.get('global_step') != step or not isinstance(state.get('log_history'), list):
            return 'trainer_state.json does not match the directory step'
    except Exception as e:
        return f'trainer_state.json unreadable: {e}'
    return None


def newest_complete_checkpoint(ckpt_dir):
    dirs = [d for d in glob.glob(ckpt_dir + '/checkpoint-*') if re.fullmatch(r'checkpoint-\d+', os.path.basename(d))]
    for d in sorted(dirs, key=lambda d: int(d.rsplit('-', 1)[1]), reverse=True):       # numeric: checkpoint-100 is newer than checkpoint-50
        why = checkpoint_problem(d)
        if why is None:
            return d
        status(f'skipping incomplete checkpoint {os.path.basename(d)}: {why}')
    return None


def check_identity(meta, **now):
    """A resume must see the same data, seed and run as the run that wrote the checkpoint."""
    for k, v in now.items():
        if k in meta and meta[k] != v:
            raise SystemExit(f'run_meta {k} differs from this session ({meta[k]!r} != {v!r}); refusing to resume on different data/settings')


# ---- Drive ---------------------------------------------------------------------------------------------------------------------------------------

class GoogleDrive:
    """Drive v3 through googleapiclient, no Drive mount. httplib2 is not thread-safe: every call holds one lock and builds a fresh client."""
    FIELDS = 'id,name,size,md5Checksum,appProperties,mimeType'

    def __init__(self, chunk=64 * MIB):
        self.lock = threading.Lock()
        self.chunk = chunk

    def service(self):
        from googleapiclient.discovery import build
        return build('drive', 'v3', cache_discovery=False)

    def find(self, parent, name=None, folders_only=False):
        q = [f"'{parent}' in parents", 'trashed = false']
        if name is not None:
            q.append("name = '" + name.replace('\\', '\\\\').replace("'", "\\'") + "'")
        if folders_only:
            q.append("mimeType = 'application/vnd.google-apps.folder'")
        out, token = [], None
        with self.lock:
            svc = self.service()
            while True:
                res = svc.files().list(q=' and '.join(q), spaces='drive', fields='nextPageToken,files(' + self.FIELDS + ')', pageSize=200, pageToken=token,
                                       orderBy='createdTime', supportsAllDrives=True, includeItemsFromAllDrives=True).execute(num_retries=5)
                out += res.get('files', [])
                token = res.get('nextPageToken')
                if not token:
                    return out

    def mkdir(self, parent, name):
        with self.lock:
            body = {'name': name, 'parents': [parent], 'mimeType': 'application/vnd.google-apps.folder'}
            return self.service().files().create(body=body, fields='id', supportsAllDrives=True).execute(num_retries=5)['id']

    def put(self, local_path, parent, name, props=None, replace_id=None):
        from googleapiclient.http import MediaFileUpload
        media = MediaFileUpload(local_path, mimetype='application/octet-stream', resumable=True, chunksize=self.chunk)
        with self.lock:
            files = self.service().files()
            if replace_id:
                req = files.update(fileId=replace_id, body={'appProperties': props or {}}, media_body=media, fields=self.FIELDS, supportsAllDrives=True)
            else:
                req = files.create(body={'name': name, 'parents': [parent], 'appProperties': props or {}}, media_body=media, fields=self.FIELDS, supportsAllDrives=True)
            resp = None
            while resp is None:
                _, resp = req.next_chunk(num_retries=5)
        return resp

    def get(self, file_id, dest):
        from googleapiclient.http import MediaIoBaseDownload
        os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
        with self.lock:
            request = self.service().files().get_media(fileId=file_id, supportsAllDrives=True)
            with open(dest + '.part', 'wb') as f:
                dl, done = MediaIoBaseDownload(f, request, chunksize=self.chunk), False
                while not done:
                    _, done = dl.next_chunk(num_retries=5)
        os.replace(dest + '.part', dest)

    def delete(self, file_id):
        with self.lock:
            self.service().files().delete(fileId=file_id, supportsAllDrives=True).execute(num_retries=5)


class DriveStore:
    """The run folder on Drive: verified uploads (size, md5 from Drive, sha256 in appProperties), restore with the same checks, retention."""

    def __init__(self, drive, path_parts, folder_id=None):
        self.drive = drive
        parent = 'root'
        if folder_id is None:
            for part in path_parts:
                found = drive.find(parent, name=part, folders_only=True)
                parent = found[0]['id'] if found else drive.mkdir(parent, part)
        self.folder_id = folder_id or parent

    def find(self, name):
        return self.drive.find(self.folder_id, name=name)

    def listing(self):
        return self.drive.find(self.folder_id)

    def upload(self, local_path, name=None, props=None):
        name = name or os.path.basename(local_path)
        size = os.path.getsize(local_path)
        assert size > 0, f'{name} is empty'
        sha, md5 = file_digests(local_path)
        props = {**(props or {}), 'sha256': sha}
        same = self.find(name)
        t0 = time.time()
        resp = self.drive.put(local_path, self.folder_id, name, props, replace_id=same[0]['id'] if same else None)
        assert int(resp.get('size', -1)) == size, f'{name}: Drive holds {resp.get("size")} bytes, local file has {size}'
        assert resp.get('md5Checksum') == md5, f'{name}: Drive md5 {resp.get("md5Checksum")} != local {md5}'
        for extra in same[1:]:
            self.drive.delete(extra['id'])
        status(f'drive up {name} {size / MIB:.0f} MiB in {time.time() - t0:.0f}s sha256 {sha[:12]}')
        return {'id': resp['id'], 'name': name, 'size': size, 'sha256': sha}

    def download(self, info, dest):
        """Fetch a Drive file and check it against the sha256 stored with it and the md5 Drive reports."""
        self.drive.get(info['id'], dest)
        sha, md5 = file_digests(dest)
        want = (info.get('appProperties') or {}).get('sha256')
        assert want is None or want == sha, f'{info["name"]}: sha256 {sha} != stored {want}'
        assert info.get('md5Checksum') in (None, md5), f'{info["name"]}: md5 {md5} != Drive {info.get("md5Checksum")}'
        return dest

    def write_json(self, name, obj):
        tmp = os.path.join(WORK, 'drive_tmp', name)
        write_json(tmp, obj)
        return self.upload(tmp, name)

    def fetch_json(self, name):
        found = self.find(name)
        if not found:
            return {}
        tmp = os.path.join(WORK, 'drive_tmp', 'fetched_' + name)
        self.drive.get(found[0]['id'], tmp)
        return read_json(tmp)

    def steps(self, pattern):
        out = []
        for f in self.listing():
            m = re.fullmatch(pattern, f['name'])
            if m:
                out.append((int(m.group(1)), f))
        return sorted(out, key=lambda t: -t[0])

    def list_checkpoints(self):
        return self.steps(r'checkpoint-(\d+)\.tar')

    def list_adapters(self):
        return self.steps(r'adapter-step-(\d+)\.tar')

    def restore_newest_checkpoint(self, dest_dir):
        """Newest checkpoint on Drive that downloads clean and passes checkpoint_problem; None if there is none."""
        for step, info in self.list_checkpoints():
            tmp = os.path.join(WORK, 'restore', info['name'])
            try:
                self.download(info, tmp)
                safe_untar(tmp, dest_dir)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            path = os.path.join(dest_dir, f'checkpoint-{step}')
            why = checkpoint_problem(path)
            if why is None:
                return path
            status(f'drive checkpoint {step} unusable: {why}')
            shutil.rmtree(path, ignore_errors=True)
        return None

    def restore_adapter(self, step, adapters_dir):
        match = [info for s, info in self.list_adapters() if s == step]
        assert match, f'no adapter-step-{step}.tar on Drive'
        tmp = os.path.join(WORK, 'restore', match[0]['name'])
        try:
            self.download(match[0], tmp)
            safe_untar(tmp, adapters_dir)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        path = os.path.join(adapters_dir, f'step-{step}')
        assert safetensors_complete(os.path.join(path, 'adapter_model.safetensors')), f'restored adapter {step} is incomplete'
        return path

    def prune_checkpoints(self, keep, protect=()):
        """Delete full checkpoints beyond the newest `keep` (and not in `protect`); adapter tars are never touched."""
        removed = []
        for i, (step, info) in enumerate(self.list_checkpoints()):
            if i >= keep and step not in protect:
                self.drive.delete(info['id'])
                removed.append(step)
        return removed


class Uploader:
    """One background thread uploads the checkpoint and adapter tars in order; a failure is re-raised in the caller (submit or join)."""

    def __init__(self, store, keep=2):
        self.store, self.keep, self.q, self.error = store, keep, queue.Queue(), None
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            job = self.q.get()
            try:
                if self.error is None:
                    self._upload(*job)
            except BaseException as e:
                self.error = e
            finally:
                self.q.task_done()

    def _upload(self, step, tar, adapter_tar, best):
        self.store.upload(adapter_tar, props={'kind': 'adapter', 'step': str(step)})        # small, and the one that matters: first
        self.store.upload(tar, props={'kind': 'checkpoint', 'step': str(step)})
        for p in (adapter_tar, tar):
            os.remove(p)
        gone = self.store.prune_checkpoints(self.keep, protect={best})
        status(f'drive step {step} stored, best {best}, pruned {gone}')

    def check(self):
        if self.error is not None:
            raise RuntimeError(f'drive upload failed: {self.error!r}') from self.error

    def submit(self, step, tar, adapter_tar, best):
        self.check()
        self.q.put((step, tar, adapter_tar, best))

    def join(self):
        self.q.join()
        self.check()


# ---- subprocesses --------------------------------------------------------------------------------------------------------------------------------

def run_logged(cmd, log_path, env=None, cwd=None, on_line=None, tail=40):
    """Run cmd; every output line goes to log_path, only STATUS lines are echoed (and passed to on_line). On failure the log tail is printed as
    plain text (cross-origin output iframes cannot be read by the watcher) and SystemExit is raised."""
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    last = collections.deque(maxlen=tail)
    with open(log_path, 'a') as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, cwd=cwd)
        for line in p.stdout:
            log.write(line)
            line = line.rstrip('\n')
            last.append(line)
            if line.startswith(STATUS_PREFIX):
                print(line[:200], flush=True)
                if on_line:
                    on_line(line)
        code = p.wait()
    if code != 0:
        print(f'--- exit {code}: last {len(last)} lines of {os.path.basename(log_path)} ---')
        for line in last:
            print(line[:300])
        raise SystemExit(f'{os.path.basename(str(cmd[1] if len(cmd) > 1 else cmd[0]))} failed with exit code {code}')
    return code


def gguf_summary(path, llama_dir):
    """Architecture, file type, tensor count and the embedded chat template of a GGUF, read with the pinned llama.cpp's gguf-py."""
    sys.path.insert(0, os.path.join(llama_dir, 'gguf-py'))
    from gguf import GGUFReader
    reader = GGUFReader(path)

    def field(name):
        f = reader.fields.get(name)
        if f is None:
            return None
        if hasattr(f, 'contents'):
            return f.contents()
        v = f.parts[f.data[0]]
        return bytes(v).decode('utf-8') if f.types and f.types[0].name == 'STRING' else v.tolist()[0]

    template = field('tokenizer.chat_template')
    return {'architecture': field('general.architecture'), 'file_type': field('general.file_type'), 'tensors': len(reader.tensors),
            'chat_template_sha256': hashlib.sha256(template.encode()).hexdigest() if isinstance(template, str) else None,
            'chat_template_has_turn_marker': isinstance(template, str) and '<|turn>' in template}
'''

IMPORT_COMMON_PY = r'''import sys
sys.path.insert(0, '/content/work')
import r4_common as C
'''

UNPACK_MD = '## c. Decode and verify the embedded rows'

UNPACK_PY = r'''import base64, gzip, hashlib, json, os
if DATA_META.get('placeholder') and not ALLOW_PLACEHOLDER:
    raise SystemExit(DATA_META['placeholder'] + '. This notebook was built with --fixture and refuses to train; rebuild it with --train/--valid.')
os.makedirs('/content/work/data', exist_ok=True)
RAW = {}
for split, b64 in (('train', DATA_TRAIN_B64), ('valid', DATA_VALID_B64)):
    raw = gzip.decompress(base64.b64decode(''.join(b64.split())))
    got = hashlib.sha256(raw).hexdigest()
    assert got == DATA_META[split]['sha256'], f'{split}: sha256 {got} != embedded {DATA_META[split]["sha256"]}'
    assert len(C.parse_jsonl(raw)) == DATA_META[split]['rows'], f'{split}: row count differs from the builder'
    open(f'/content/work/data/{split}.jsonl', 'wb').write(raw)
    RAW[split] = raw
C.status(f"data ok: train {DATA_META['train']['rows']} rows sha256 {DATA_META['train']['sha256'][:12]}, valid {DATA_META['valid']['rows']} rows sha256 {DATA_META['valid']['sha256'][:12]}"
         + (' (smoke subset)' if DATA_META.get('smoke_subset') else ''))
'''

EFFECTIVE_PY = r'''import json, os, shutil, subprocess, sys, threading, time
S = {k: globals()[k] for k in SETTING_NAMES}
E = C.effective_settings(S, DATA_META)
for d in (E['OUT_LOCAL'], E['LOG_DIR']):
    os.makedirs(d, exist_ok=True)
C.write_json('/content/work/settings.json', E)
ENV = {**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'HF_HUB_DISABLE_PROGRESS_BARS': '1', 'TOKENIZERS_PARALLELISM': 'false'}
C.status(f"run {E['RUN_TAG']}: smoke={SMOKE} resume={RESUME} steps={E['MAX_STEPS'] or 'epochs=' + str(EPOCHS)} save/eval every {E['SAVE_STEPS']}/{E['EVAL_STEPS']}")
'''

GPU_MD = '## d. GPU check'

GPU_PY = r'''out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
if not out:
    raise SystemExit('No GPU. Runtime > Change runtime type > A100, then re-run.')
GPU_NAME, GPU_MEM = [p.strip() for p in out.splitlines()[0].split(',')]
GPU_TOTAL_GB = float(GPU_MEM.split()[0]) / 1024
assert GPU_TOTAL_GB >= 38, f'{GPU_NAME} has {GPU_TOTAL_GB:.0f} GB; max_seq {MAX_SEQ} needs the A100 (40 GB)'
C.status(f'gpu {GPU_NAME} {GPU_TOTAL_GB:.0f} GB')
'''

DRIVE_MD = '## e. Drive through googleapiclient (the permission dialog needs the user), run guard'

DRIVE_PY = r'''from google.colab import auth
auth.authenticate_user()                 # a permission dialog opens in the Colab tab: the user approves it; the cell blocks until then
T_START = time.time()                    # timing marks start after the dialog
DRIVE = C.GoogleDrive()
STORE = C.DriveStore(DRIVE, E['DRIVE_PATH'], DRIVE_FOLDER_ID)
EXISTING = STORE.list_checkpoints()
RUN_META = STORE.fetch_json('run_meta.json')
IDENTITY = dict(run_name=RUN_NAME, smoke=SMOKE, train_sha256=DATA_META['train']['sha256'], valid_sha256=DATA_META['valid']['sha256'], seed=SEED, base_revision=BASE_REVISION,
                train_settings={k: E[k] for k in ('EPOCHS', 'MAX_STEPS', 'LR', 'RANK', 'ALPHA', 'BATCH', 'GRAD_ACCUM', 'WARMUP_STEPS', 'MAX_SEQ')})
if EXISTING and not RESUME:
    raise SystemExit(f'Drive folder {"/".join(E["DRIVE_PATH"])} already holds {len(EXISTING)} checkpoint(s); set RESUME = True to continue or use another RUN_NAME')
if RESUME and not EXISTING:
    raise SystemExit('RESUME = True but Drive holds no checkpoint of this run')
if RESUME:
    C.check_identity(RUN_META, **IDENTITY)
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = ENV['HF_TOKEN'] = userdata.get('HF_TOKEN')
    HF_TOKEN_STATE = 'from the Colab secret'
except Exception as e:
    HF_TOKEN_STATE = 'none (' + type(e).__name__ + ')'
C.status(f"drive folder {'/'.join(E['DRIVE_PATH'])} id {STORE.folder_id[:10]}, {len(EXISTING)} checkpoint(s); HF token {HF_TOKEN_STATE}")
'''

INSTALL_MD = '## f. Install the pinned libraries (before anything imports torch), build the pinned llama.cpp in the background'

INSTALL_PY = r'''import importlib.metadata as md_
missing = [k for k, v in PINS.items() if not v]
if (missing or not BASE_REVISION) and REQUIRE_ALL_PINS:
    raise SystemExit(f'unfilled pins: {missing + ([] if BASE_REVISION else ["BASE_REVISION"])}. Rebuild with --pins-json, or SMOKE = True / REQUIRE_ALL_PINS = False for a smoke run.')
specs = [f'{k}=={v}' if v else k for k, v in PINS.items()]
C.run_logged([sys.executable, '-m', 'pip', 'install', '-q', *specs, 'sentencepiece', 'protobuf'], E['LOG_DIR'] + '/pip_install.log')
VERSION_PROBE = ("import importlib.metadata as m, json, torch\n"
                 "names = ['unsloth', 'unsloth_zoo', 'transformers', 'trl', 'peft', 'bitsandbytes', 'accelerate', 'datasets']\n"
                 "print(json.dumps({**{n: m.version(n) for n in names}, 'torch': torch.__version__, 'cuda': torch.version.cuda}))\n")
VERSIONS = json.loads(subprocess.run([sys.executable, '-c', VERSION_PROBE], capture_output=True, text=True, check=True, env=ENV).stdout.strip().splitlines()[-1])
for k, v in PINS.items():
    assert not v or VERSIONS[k] == v, f'{k} is {VERSIONS[k]}, pinned {v}'
open(E['OUT_LOCAL'] + '/pip_freeze.txt', 'w').write(subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout)
C.status('versions ' + ' '.join(f'{k}={v}' for k, v in VERSIONS.items()))

LLAMA = E['LLAMA_CPP_DIR']
BUILD = {'error': None}

def build_llama():
    try:
        git = lambda *a: subprocess.run(['git', *a], cwd=LLAMA, capture_output=True, text=True, check=True).stdout.strip()
        if not os.path.isdir(LLAMA + '/.git'):
            subprocess.run(['git', 'clone', '-q', 'https://github.com/ggml-org/llama.cpp', LLAMA], check=True, capture_output=True)
        git('checkout', '-q', LLAMA_CPP_COMMIT)
        assert git('rev-parse', 'HEAD') == LLAMA_CPP_COMMIT, 'llama.cpp is not at the pinned commit'
        if not os.path.isfile(LLAMA + '/build/bin/llama-quantize'):
            C.run_logged(['cmake', '-B', 'build', '-DGGML_CUDA=OFF', '-DLLAMA_CURL=OFF'], E['LOG_DIR'] + '/llama_build.log', cwd=LLAMA)
            C.run_logged(['cmake', '--build', 'build', '--target', 'llama-quantize', '-j', str(os.cpu_count() or 4)], E['LOG_DIR'] + '/llama_build.log', cwd=LLAMA)
        assert os.path.isfile(LLAMA + '/convert_hf_to_gguf.py') and os.path.isfile(LLAMA + '/build/bin/llama-quantize')
    except BaseException as e:
        BUILD['error'] = e

BUILD_THREAD = threading.Thread(target=build_llama, daemon=True)
BUILD_THREAD.start()
'''

TRAIN_SCRIPT_MD = '## g. Training script (runs as a subprocess)'

TRAIN_SCRIPT_PY = r'''%%writefile /content/work/train_gemma4.py
"""One FT-r4 training leg: Gemma 4 12B in 4 bit, LoRA, loss on the assistant turn only, valid loss per eval step. Drive is the notebook's job:
this script writes checkpoint and adapter tars to disk and prints `STATUS saved ...`."""
import argparse, gc, hashlib, json, os, shutil, sys, time
sys.path.insert(0, '/content/work')
import r4_common as C

ap = argparse.ArgumentParser()
ap.add_argument('--leg', default='main')
ap.add_argument('--resume-from', default=None)
ap.add_argument('--max-steps', type=int, default=None)
A = ap.parse_args()
S = C.load_settings()
for d in (S['OUT_LOCAL'], S['CKPT_LOCAL'], S['ADAPTER_LOCAL'], S['UPLOAD_LOCAL']):
    os.makedirs(d, exist_ok=True)

from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
import torch
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

T0 = time.time()
model, tokenizer = FastModel.from_pretrained(model_name=S['BASE_MODEL'], max_seq_length=S['MAX_SEQ'], load_in_4bit=True, load_in_8bit=False,
                                             full_finetuning=False, revision=S['BASE_REVISION'] or None, token=os.environ.get('HF_TOKEN'))
LOAD_SECONDS = round(time.time() - T0, 1)
try:
    from huggingface_hub import HfApi
    BASE_SHA = HfApi().model_info(S['BASE_MODEL'], revision=S['BASE_REVISION'] or None, token=os.environ.get('HF_TOKEN')).sha
except Exception as e:
    BASE_SHA = 'unresolved: ' + type(e).__name__
assert not S['BASE_REVISION'] or BASE_SHA == S['BASE_REVISION'], (BASE_SHA, S['BASE_REVISION'])
C.status(f"model loaded in {LOAD_SECONDS}s, base {BASE_SHA[:12]}")
model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True,
                                 finetune_mlp_modules=True, r=S['RANK'], lora_alpha=S['ALPHA'], lora_dropout=0, bias='none',
                                 random_state=S['SEED'], use_gradient_checkpointing='unsloth')

# the text tokenizer: a multimodal processor keeps it in .tokenizer
tok = getattr(tokenizer, 'tokenizer', tokenizer)
if not getattr(tok, 'chat_template', None) and getattr(tokenizer, 'chat_template', None):
    tok.chat_template = tokenizer.chat_template
assert tok.chat_template, 'the tokenizer has no chat template'
render = lambda msgs, gen: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
encode = lambda text: tok(text, add_special_tokens=False)['input_ids']
INSTRUCTION_PART, RESPONSE_PART = C.derive_markers(render, tok.bos_token or '')
TURN_END = C.derive_turn_end(render)
TEMPLATE_SHA = hashlib.sha256((tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template, sort_keys=True)).encode()).hexdigest()
C.status(f'markers user={INSTRUCTION_PART!r} model={RESPONSE_PART!r} turn_end={TURN_END!r}')

def load_split(split):
    rows = C.parse_jsonl(open(f"{S['DATA_DIR']}/{split}.jsonl", 'rb').read())
    idx = S[split.upper() + '_INDICES']
    return rows if idx is None else [rows[i] for i in idx]

train_rows, valid_rows = load_split('train'), load_split('valid')
prep = {sp: C.prepare_split(sp, rows, render, encode, tok.bos_token_id, TURN_END, S['MAX_SEQ'], S['EXPECT_TOKENS'][sp], S['TOKEN_TOLERANCE'])
        for sp, rows in (('train', train_rows), ('valid', valid_rows))}
C.status(f"rows train={len(train_rows)} valid={len(valid_rows)} max_tokens={max(prep['train']['tokens'] + prep['valid']['tokens'])} "
         f"count_gap={max(p['max_diff'] for p in prep.values())} modes={prep['train']['modes']}")
train_ds = Dataset.from_list([{'text': t} for t in prep['train']['texts']])
valid_ds = Dataset.from_list([{'text': t} for t in prep['valid']['texts']])
STEPS_PER_EPOCH = -(-len(train_ds) // (S['BATCH'] * S['GRAD_ACCUM']))
MAX_STEPS = A.max_steps or S['MAX_STEPS'] or -1
TOTAL_STEPS = MAX_STEPS if MAX_STEPS > 0 else STEPS_PER_EPOCH * S['EPOCHS']

class R4Callback(TrainerCallback):
    def __init__(self):
        self.step_seconds, self.t_step, self.best = [], None, (None, None)
    def on_train_begin(self, args, state, control, **kw):
        for h in state.log_history:                                           # after a resume the history holds the evals before the restore
            if 'eval_loss' in h and h['step'] > 0 and (self.best[1] is None or h['eval_loss'] < self.best[1]):
                self.best = (h['step'], h['eval_loss'])
        C.status(f'train_begin step={state.global_step} max_steps={state.max_steps} best_step={self.best[0]}')
    def on_step_begin(self, args, state, control, **kw):
        self.t_step = time.time()
    def on_step_end(self, args, state, control, **kw):
        self.step_seconds.append(round(time.time() - self.t_step, 2))
        if state.global_step >= state.max_steps:                              # the last step is always evaluated and saved
            control.should_evaluate = control.should_save = True
        return control
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and 'loss' in logs:
            recent = self.step_seconds[-5:]
            C.status(f"step={state.global_step}/{state.max_steps} loss={logs['loss']:.4f} lr={logs.get('learning_rate', 0):.2e} "
                     f"s_per_step={sum(recent) / len(recent):.1f} peak_gb={torch.cuda.max_memory_allocated() / 2**30:.1f}")
    def on_evaluate(self, args, state, control, metrics=None, **kw):
        loss = (metrics or {}).get('eval_loss')
        if loss is not None and state.global_step > 0 and (self.best[1] is None or loss < self.best[1]):
            self.best = (state.global_step, loss)
        C.status(f"eval step={state.global_step} valid_loss={loss} best_step={self.best[0]} best_loss={self.best[1]}")
    def on_save(self, args, state, control, **kw):
        step = state.global_step
        ckpt = f"{args.output_dir}/checkpoint-{step}"
        adapter = f"{S['ADAPTER_LOCAL']}/step-{step}"
        os.makedirs(adapter, exist_ok=True)
        for name in C.ADAPTER_FILES:
            shutil.copy(f'{ckpt}/{name}', f'{adapter}/{name}')
        why = C.checkpoint_problem(ckpt)
        assert why is None, f'checkpoint {step} is not complete: {why}'
        tar = C.make_tar(ckpt, f"{S['UPLOAD_LOCAL']}/checkpoint-{step}.tar", f'checkpoint-{step}')
        atar = C.make_tar(adapter, f"{S['UPLOAD_LOCAL']}/adapter-step-{step}.tar", f'step-{step}')
        # best is known here: the evaluation of this step ran before the save; an eval-less save keeps the earlier best
        C.status(f"saved step={step} tar={tar} adapter={atar} best={self.best[0] or step}", limit=None)

cb = R4Callback()
args = SFTConfig(
    output_dir=S['CKPT_LOCAL'], dataset_text_field='text', max_seq_length=S['MAX_SEQ'], packing=False,
    num_train_epochs=S['EPOCHS'], max_steps=MAX_STEPS, learning_rate=S['LR'], per_device_train_batch_size=S['BATCH'],
    gradient_accumulation_steps=S['GRAD_ACCUM'], per_device_eval_batch_size=1, logging_steps=5, disable_tqdm=True,
    eval_strategy='steps', eval_steps=S['EVAL_STEPS'], eval_on_start=A.resume_from is None,
    save_strategy='steps', save_steps=S['SAVE_STEPS'], save_total_limit=3,
    bf16=True, seed=S['SEED'], data_seed=S['SEED'], optim='adamw_8bit', lr_scheduler_type='linear',
    warmup_steps=S['WARMUP_STEPS'], weight_decay=0.01, report_to='none')
trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=train_ds, eval_dataset=valid_ds, args=args, callbacks=[cb])
trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)
first = trainer.train_dataset[0]['input_ids']
assert first[0] == tok.bos_token_id and first[1] != tok.bos_token_id, 'BOS missing or doubled after the trainer tokenised the text'
TAIL_OK = (tok.eos_token,)                                                     # TRL may append the EOS token to a text column
LIVE_TRAIN = C.check_loss_masks('train', trainer.train_dataset, tok.decode, [C.rows_to_messages(r)[2]['content'] for r in train_rows], TURN_END, TAIL_OK)
LIVE_VALID = C.check_loss_masks('valid', trainer.eval_dataset, tok.decode, [C.rows_to_messages(r)[2]['content'] for r in valid_rows], TURN_END, TAIL_OK)
C.status(f'assistant-only loss ok on all rows: {LIVE_TRAIN} train and {LIVE_VALID} valid target tokens')

torch.cuda.reset_peak_memory_stats()
T1 = time.time()
result = trainer.train(resume_from_checkpoint=A.resume_from)
SECONDS = round(time.time() - T1, 1)
hist = trainer.state.log_history
report = {
    'leg': A.leg, 'resumed_from': A.resume_from, 'final_step': trainer.state.global_step, 'max_steps': trainer.state.max_steps, 'total_steps_planned': TOTAL_STEPS,
    'steps_per_epoch': STEPS_PER_EPOCH, 'train_rows': len(train_ds), 'valid_rows': len(valid_ds), 'train_seconds': SECONDS, 'model_load_seconds': LOAD_SECONDS,
    'step_seconds': cb.step_seconds, 'peak_gpu_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2), 'gpu': torch.cuda.get_device_name(0),
    'torch': torch.__version__, 'eval_history': [{'step': h['step'], 'eval_loss': h['eval_loss']} for h in hist if 'eval_loss' in h],
    'train_loss_last': next((h['loss'] for h in reversed(hist) if 'loss' in h), None), 'tokens_train': prep['train']['tokens'], 'tokens_valid': prep['valid']['tokens'],
    'count_gap_max': max(p['max_diff'] for p in prep.values()), 'template_modes': prep['train']['modes'], 'template_sha256': TEMPLATE_SHA,
    'markers': [INSTRUCTION_PART, RESPONSE_PART, TURN_END], 'base_revision_resolved': BASE_SHA, 'unmasked_tokens': {'train': LIVE_TRAIN, 'valid': LIVE_VALID},
}
C.write_json(f"{S['OUT_LOCAL']}/train_report_{A.leg}.json", report)
C.status(f"leg {A.leg} done: step {report['final_step']}, {SECONDS}s, peak {report['peak_gpu_gb']} GB")
'''

TRAIN_RUN_MD = '## h. Train (checkpoints go to Drive while it runs; the smoke adds the restore + resume round trip)'

TRAIN_RUN_PY = r'''SAVED = []
BEGINS = []
UP = C.Uploader(STORE, keep=KEEP_DRIVE_CHECKPOINTS)

def on_line(line):
    kv = C.parse_kv(line)
    if line.startswith('STATUS saved '):
        SAVED.append(int(kv['step']))
        UP.submit(int(kv['step']), kv['tar'], kv['adapter'], int(kv['best']))
    elif line.startswith('STATUS train_begin '):
        BEGINS.append(int(kv['step']))

def run_leg(name, *extra):
    t0 = time.time()
    C.run_logged([sys.executable, '/content/work/train_gemma4.py', '--leg', name, *extra], f"{E['LOG_DIR']}/train_{name}.log", env=ENV, on_line=on_line)
    UP.join()
    return C.read_json(f"{E['OUT_LOCAL']}/train_report_{name}.json"), round(time.time() - t0, 1)

RESUME_PATH = STORE.restore_newest_checkpoint(E['CKPT_LOCAL']) if RESUME else None
if RESUME:
    assert RESUME_PATH, 'no complete checkpoint could be restored from Drive'
    C.status(f'restored {os.path.basename(RESUME_PATH)} from Drive')
STORE.write_json('run_meta.json', {**IDENTITY, 'status': 'training'})
REPORTS = {}
REPORTS['main'], LEG_WALL = run_leg('main', *(['--resume-from', RESUME_PATH] if RESUME_PATH else []))
FINAL_STEP = REPORTS['main']['final_step']
C.status(f"training done at step {FINAL_STEP}, valid history {[(h['step'], round(h['eval_loss'], 4)) for h in REPORTS['main']['eval_history']]}")
'''

SMOKE_MD = '## i. Smoke only: wipe the local checkpoints, restore from Drive, resume, report'

SMOKE_PY = r'''SMOKE_REPORT = {}
if SMOKE:
    assert BEGINS == [0] and SAVED == sorted(SAVED) and SAVED[-1] == SMOKE_STEPS, (BEGINS, SAVED)
    shutil.rmtree(E['CKPT_LOCAL'], ignore_errors=True)               # a fresh runtime has none of these
    shutil.rmtree(E['ADAPTER_LOCAL'], ignore_errors=True)
    restored = STORE.restore_newest_checkpoint(E['CKPT_LOCAL'])
    assert restored and restored.endswith(f'checkpoint-{SMOKE_STEPS}'), restored
    REPORTS['resume'], _ = run_leg('resume', '--resume-from', restored, '--max-steps', str(SMOKE_STEPS + SMOKE_RESUME_EXTRA_STEPS))
    assert BEGINS[-1] == SMOKE_STEPS and REPORTS['resume']['final_step'] == SMOKE_STEPS + SMOKE_RESUME_EXTRA_STEPS, (BEGINS, REPORTS['resume']['final_step'])
    FINAL_STEP = REPORTS['resume']['final_step']
    steady = REPORTS['main']['step_seconds'][2:] or REPORTS['main']['step_seconds']
    mean_step = sum(steady) / len(steady)
    sub_tokens = sum(REPORTS['main']['tokens_train']) / len(REPORTS['main']['tokens_train'])
    full = DATA_META.get('full') or {'train_rows': DATA_META['train']['rows'], 'tokens_mean': sub_tokens}
    full_steps = -(-full['train_rows'] // (BATCH * GRAD_ACCUM)) * EPOCHS
    SMOKE_REPORT = {
        'seconds_per_step_mean': round(mean_step, 2), 'seconds_per_step_all': REPORTS['main']['step_seconds'], 'model_load_seconds': REPORTS['main']['model_load_seconds'],
        'peak_gpu_gb': max(r['peak_gpu_gb'] for r in REPORTS.values()), 'smoke_mean_row_tokens': round(sub_tokens), 'full_train_rows': full['train_rows'],
        'full_mean_row_tokens': full['tokens_mean'], 'full_steps_planned': full_steps,
        'estimate_full_train_seconds': round(mean_step * full_steps * full['tokens_mean'] / sub_tokens),
        'estimate_note': 'linear in tokens from the steady smoke steps; eval, save, upload, load and export come on top (see eval/upload lines)',
        'round_trip': {'restored': os.path.basename(restored), 'resume_begin_step': BEGINS[-1], 'resume_final_step': FINAL_STEP}, 'saved_steps': SAVED,
    }
    C.status(f"smoke: {SMOKE_REPORT['seconds_per_step_mean']} s/step, peak {SMOKE_REPORT['peak_gpu_gb']} GB, full run estimate {SMOKE_REPORT['estimate_full_train_seconds']} s")
'''

EXPORT_SCRIPT_MD = '## j. Export script (merge, convert with the pinned llama.cpp, quantize; runs as a subprocess)'

EXPORT_SCRIPT_PY = r'''%%writefile /content/work/export_gguf.py
"""Merge one LoRA step into the pinned base (16 bit), convert it with the pinned llama.cpp and quantize. Prints the result in export.json."""
import argparse, gc, hashlib, json, os, shutil, subprocess, sys, time
sys.path.insert(0, '/content/work')
import r4_common as C

ap = argparse.ArgumentParser()
ap.add_argument('--step', type=int, required=True)
A = ap.parse_args()
S = C.load_settings()
LLAMA = S['LLAMA_CPP_DIR']
OUT = S['OUT_LOCAL']
MERGED, BF16 = S['WORK'] + '/merged', S['WORK'] + '/model-bf16.gguf'
QUANT = S['EXPORT_QUANT']
GGUF = f"{OUT}/buffett-gemma4-12b-ft-r4-{S['RUN_TAG']}-s{A.step}-{QUANT.lower()}.gguf"
assert subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=LLAMA, capture_output=True, text=True).stdout.strip() == S['LLAMA_CPP_COMMIT'], 'llama.cpp is not at the pinned commit'
shutil.rmtree(MERGED, ignore_errors=True)
T0 = time.time()

# the adapter points at the pinned base snapshot, so the merge cannot pick up a newer revision of the model repo
from huggingface_hub import snapshot_download
base_dir = snapshot_download(S['BASE_MODEL'], revision=S['BASE_REVISION'], token=os.environ.get('HF_TOKEN'))
adapter = f"{S['WORK']}/export_adapter"
shutil.rmtree(adapter, ignore_errors=True)
shutil.copytree(f"{S['ADAPTER_LOCAL']}/step-{A.step}", adapter)
cfg = C.read_json(adapter + '/adapter_config.json')
cfg['base_model_name_or_path'] = base_dir
C.write_json(adapter + '/adapter_config.json', cfg)

from unsloth import FastModel
import torch
model, tokenizer = FastModel.from_pretrained(adapter, max_seq_length=S['MAX_SEQ'], load_in_4bit=True, load_in_8bit=False, full_finetuning=False, token=os.environ.get('HF_TOKEN'))
model.save_pretrained_merged(MERGED, tokenizer, save_method='merged_16bit')
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
C.status(f'merged step {A.step} in {time.time() - T0:.0f}s')

T1 = time.time()
subprocess.run([sys.executable, f'{LLAMA}/convert_hf_to_gguf.py', MERGED, '--outtype', 'bf16', '--outfile', BF16], check=True)
C.status(f'converted to bf16 gguf in {time.time() - T1:.0f}s')
shutil.rmtree(MERGED, ignore_errors=True)
T2 = time.time()
subprocess.run([f'{LLAMA}/build/bin/llama-quantize', BF16, GGUF + '.part', QUANT], check=True)
os.replace(GGUF + '.part', GGUF)
os.remove(BF16)
C.status(f'quantized {QUANT} in {time.time() - T2:.0f}s, {os.path.getsize(GGUF) / 2**30:.2f} GiB')

summary = C.gguf_summary(GGUF, LLAMA)
assert summary['architecture'] == 'gemma4' and summary['tensors'] > 0, summary
sha = C.sha256_file(GGUF)
report = C.read_json(f"{OUT}/train_report_main.json")
export = {'gguf': os.path.basename(GGUF), 'bytes': os.path.getsize(GGUF), 'sha256': sha, 'step': A.step, 'quant': QUANT, 'llama_cpp_commit': S['LLAMA_CPP_COMMIT'],
          'summary': summary, 'training_template_sha256': report['template_sha256'],
          'template_parity': summary['chat_template_sha256'] == report['template_sha256'], 'seconds': round(time.time() - T0)}
C.write_json(f'{OUT}/export.json', export)
with open(f'{OUT}/SHA256SUMS', 'w') as f:
    f.write(f'{sha}  {os.path.basename(GGUF)}\n')
C.status(f"export ok: {export['gguf']} sha256 {sha[:12]} arch {summary['architecture']} template_in_gguf={summary['chat_template_has_turn_marker']} parity={export['template_parity']}")
'''

EXPORT_RUN_MD = '## k. Pick the checkpoint with the lowest valid loss, export it'

EXPORT_RUN_PY = r'''HISTORY = REPORTS['resume']['eval_history'] if SMOKE else REPORTS['main']['eval_history']
ON_DRIVE = {s for s, _ in STORE.list_adapters()}
BEST_STEP = C.best_step(HISTORY, available=ON_DRIVE)
C.status(f"lowest valid loss at step {BEST_STEP}: {min(h['eval_loss'] for h in HISTORY if h['step'] == BEST_STEP):.4f} (frozen choice, no dev-driven picking)")
if not os.path.isfile(f"{E['ADAPTER_LOCAL']}/step-{BEST_STEP}/adapter_model.safetensors"):
    STORE.restore_adapter(BEST_STEP, E['ADAPTER_LOCAL'])
BUILD_THREAD.join()
if BUILD['error'] is not None:
    raise SystemExit(f'llama.cpp build failed: {BUILD["error"]!r}; see {E["LOG_DIR"]}/llama_build.log')
C.status('llama.cpp ' + subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=LLAMA, capture_output=True, text=True).stdout.strip() + ' built')
free_gb = shutil.disk_usage('/content').free / 2**30
assert free_gb >= 90, f'{free_gb:.0f} GB free; merge + bf16 gguf + quantized gguf need about 90'
C.run_logged([sys.executable, '/content/work/export_gguf.py', '--step', str(BEST_STEP)], E['LOG_DIR'] + '/export.log', env=ENV)
EXPORT = C.read_json(E['OUT_LOCAL'] + '/export.json')
GGUF_PATH = f"{E['OUT_LOCAL']}/{EXPORT['gguf']}"
'''

FINISH_MD = '## l. Records, Drive upload, download, runtime release'

FINISH_PY = r'''from google.colab import files
PINS_RESOLVED = {'versions': VERSIONS, 'pins_used': PINS, 'base_model': BASE_MODEL, 'base_revision_pinned': BASE_REVISION,
                 'base_revision_resolved': REPORTS['main']['base_revision_resolved'], 'llama_cpp_commit': LLAMA_CPP_COMMIT, 'gpu': GPU_NAME,
                 'torch_cuda': VERSIONS.get('cuda'), 'chat_template_sha256': REPORTS['main']['template_sha256'], 'markers': REPORTS['main']['markers']}
C.write_json(E['OUT_LOCAL'] + '/pins_resolved.json', PINS_RESOLVED)
METRICS = {'run_name': RUN_NAME, 'run_tag': E['RUN_TAG'], 'smoke': SMOKE, 'data': {s: {'rows': DATA_META[s]['rows'], 'sha256': DATA_META[s]['sha256']} for s in ('train', 'valid')},
           'lora': {'rank': RANK, 'alpha': ALPHA, 'lr': LR, 'batch': BATCH, 'grad_accum': GRAD_ACCUM, 'warmup_steps': E['WARMUP_STEPS'], 'seed': SEED},
           'max_seq': MAX_SEQ, 'count_gap_max': REPORTS['main']['count_gap_max'], 'steps_per_epoch': REPORTS['main']['steps_per_epoch'], 'final_step': FINAL_STEP,
           'valid_history': HISTORY, 'best_step': BEST_STEP, 'best_valid_loss': min(h['eval_loss'] for h in HISTORY if h['step'] == BEST_STEP),
           'peak_gpu_gb': max(r['peak_gpu_gb'] for r in REPORTS.values()), 'train_seconds_this_session': sum(r['train_seconds'] for r in REPORTS.values()),
           'export': EXPORT, 'pins_resolved': PINS_RESOLVED, 'wall_seconds_since_auth': round(time.time() - T_START)}
if SMOKE:
    METRICS['smoke_report'] = SMOKE_REPORT
    C.write_json(E['OUT_LOCAL'] + '/smoke_report.json', SMOKE_REPORT)
C.write_json(E['OUT_LOCAL'] + '/metrics.json', METRICS)
SMALL = ['pins_resolved.json', 'metrics.json', 'export.json', 'SHA256SUMS'] + (['smoke_report.json'] if SMOKE else [])
for name in SMALL:
    STORE.upload(f"{E['OUT_LOCAL']}/{name}")
for log in sorted(os.listdir(E['LOG_DIR'])):
    STORE.upload(f"{E['LOG_DIR']}/{log}", name=f'log_{log}') if os.path.getsize(f"{E['LOG_DIR']}/{log}") > 0 else None
if not SMOKE:                                                    # the smoke GGUF is not worth 7 GB of Drive and download time
    STORE.upload(GGUF_PATH, props={'kind': 'gguf', 'step': str(BEST_STEP)})
STORE.write_json('run_meta.json', {**IDENTITY, 'status': 'finished', 'best_step': BEST_STEP, 'gguf': EXPORT['gguf'], 'gguf_sha256': EXPORT['sha256']})
C.status('everything is on Drive; starting the downloads')
for name in SMALL:
    files.download(f"{E['OUT_LOCAL']}/{name}")
if not SMOKE:
    files.download(GGUF_PATH)
C.status('downloads started: ' + ', '.join(SMALL + ([EXPORT['gguf']] if not SMOKE else [])))
'''

RELEASE_MD = '## m. Release the runtime (waits so the downloads can finish)'

RELEASE_PY = r'''if RELEASE_RUNTIME:
    C.status(f'ALL_DONE; releasing the runtime in {RELEASE_SLEEP_SECONDS} s')
    time.sleep(RELEASE_SLEEP_SECONDS)
    from google.colab import runtime
    runtime.unassign()
else:
    C.status('ALL_DONE; RELEASE_RUNTIME is False, the runtime stays connected')
'''


# ---------------------------------------------------------------------------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------------------------------------------------------------------------

def load_common():
    """r4_common as a module, from the notebook's own source (the builder and the notebook run the same code)."""
    src = COMMON_PY.split('\n', 1)[1]
    mod = types.ModuleType('r4_common')
    exec(compile(src, 'r4_common.py', 'exec'), mod.__dict__)
    return mod


def lines_of(raw):
    return [line for line in raw.decode('utf-8').splitlines(keepends=True) if line.strip()]


def tokenizer_counter(tokenizer_dir):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(Path(tokenizer_dir) / 'tokenizer.json'))
    return (lambda text: tk.encode(text, add_special_tokens=False).ids), tk.token_to_id('<bos>')


def split_stats(rows, C, encode, bos_id, max_seq):
    """Token counts per row with the stand-in template; the same prepare_split the notebook runs."""
    turn_end = C.derive_turn_end(render_standin)
    prep = C.prepare_split('rows', rows, render_standin, encode, bos_id, turn_end, max_seq)
    return prep['tokens']


def summarise(tokens):
    s = sorted(tokens)
    q = lambda p: s[min(len(s) - 1, max(0, -(-p * len(s) // 100) - 1))]
    return {'rows': len(s), 'tokens_total': sum(s), 'tokens_mean': round(sum(s) / len(s), 1), 'tokens_p50': q(50), 'tokens_p95': q(95), 'tokens_max': s[-1]}


def fixture_rows():
    """Synthetic rows for the placeholder notebook and the tests; no real passage, no evaluation text."""
    system = 'FIXTURE system prompt: answer only with sentences of the passages, each ending with its [n].'
    rows = []
    for i in range(8):
        passages = ' '.join(f'[{n}] ({1990 + n}) Synthetic passage {i}.{n} says something about topic {i}. Second synthetic sentence {i}.{n}.' for n in (1, 2, 3))
        rows.append({'id': f'fx{i}', 'kind': ('single_year', 'figure', 'partial', 'refusal')[i % 4],
                     'messages': [{'role': 'system', 'content': system},
                                  {'role': 'user', 'content': f'BEGIN UNTRUSTED PASSAGES\n{passages}\nEND UNTRUSTED PASSAGES\nQuestion: what about topic {i}?'},
                                  {'role': 'assistant', 'content': f'Synthetic passage {i}.1 says something about topic {i}. [1] Second synthetic sentence {i}.2. [2]'}]})
    return [json.dumps(r) + '\n' for r in rows[:6]], [json.dumps(r) + '\n' for r in rows[6:]]


def code_cell(source):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': source.splitlines(keepends=True)}


def hidden_code_cell(source, title):
    cell = code_cell(f'# @title {title}\n' + source)
    cell['metadata'] = {'cellView': 'form', 'collapsed': True, 'jupyter': {'source_hidden': True}}
    return cell


def md_cell(source):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(keepends=True)}


def fill(template, values):
    for key, val in values.items():
        template = template.replace(f'@@{key}@@', str(val))
    assert '@@' not in template, 'unfilled marker in a cell'
    return template


GUARD = '%%guarded\n'


def g(source):
    """A kernel cell run through the %%guarded magic of the settings cell."""
    return code_cell(GUARD + source)


def build_notebook(train_lines, valid_lines, opts):
    """Notebook dict from the jsonl lines of both splits. opts: run_name, smoke, epochs, save_steps, eval_steps, pins, max_seq, tokenizer_dir, source."""
    C = load_common()
    encode, bos_id = tokenizer_counter(opts['tokenizer_dir'])
    embedded = {'train': train_lines, 'valid': valid_lines}
    parsed = {sp: [json.loads(line) for line in lines] for sp, lines in embedded.items()}
    ids = [r['id'] for rows in parsed.values() for r in rows if r.get('id') is not None]
    assert len(ids) == len(set(ids)), 'duplicate row ids across train/valid'
    tokens = {sp: split_stats(rows, C, encode, bos_id, opts['max_seq']) for sp, rows in parsed.items()}
    full = {'train_rows': len(train_lines), 'valid_rows': len(valid_lines), **{k: v for k, v in summarise(tokens['train']).items() if k != 'rows'}}
    if opts['smoke']:                      # the smoke notebook embeds only the rows the smoke trains and evaluates on
        for sp, n in (('train', SMOKE_ROWS), ('valid', SMOKE_VALID_ROWS)):
            keep = C.pick_smoke_rows(tokens[sp], n, SEED)
            embedded[sp] = [embedded[sp][i] for i in keep]
            tokens[sp] = [tokens[sp][i] for i in keep]
    meta = {'placeholder': FIXTURE_PLACEHOLDER if opts['source'] == 'fixture' else None, 'smoke_subset': bool(opts['smoke']), 'full': full, 'max_seq': opts['max_seq'],
            'tokenizer': {'revision': BASE_REVISION, 'tokenizer_json_sha256': hashlib.sha256((Path(opts['tokenizer_dir']) / 'tokenizer.json').read_bytes()).hexdigest(),
                          'template': 'stand-in Gemma 4 turn format (render_standin); the notebook recounts with the real chat template'},
            'builder': 'colab/build_train_gemma4_r4.py', 'options': {k: opts[k] for k in ('run_name', 'smoke', 'epochs', 'save_steps', 'eval_steps', 'max_seq', 'source')}}
    blobs = {}
    for sp in ('train', 'valid'):
        raw = ''.join(embedded[sp]).encode('utf-8')
        blobs[sp] = base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode('ascii')
        meta[sp] = {'rows': len(embedded[sp]), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'tokens': tokens[sp], **{k: v for k, v in summarise(tokens[sp]).items() if k != 'rows'}}
    pins = {k: opts['pins'].get(k, DEFAULT_PINS.get(k)) for k in PIN_KEYS}
    settings = fill(SETTINGS_PY, {
        'RUN_NAME': opts['run_name'], 'SMOKE': bool(opts['smoke']), 'BASE_REVISION': BASE_REVISION, 'PINS': repr(pins),
        'REQUIRE_ALL_PINS': not opts['smoke'], 'LLAMA_CPP_COMMIT': LLAMA_CPP_COMMIT, 'MAX_SEQ': opts['max_seq'], 'EPOCHS': opts['epochs'],
        'SAVE_STEPS': opts['save_steps'], 'EVAL_STEPS': opts['eval_steps']})
    b64_cells = [hidden_code_cell(f'{name} = """\n' + '\n'.join(blobs[sp][i:i + 4096] for i in range(0, len(blobs[sp]), 4096)) + '\n"""\n', f'embedded {sp} rows (gzip+base64)')
                 for sp, name in (('train', 'DATA_TRAIN_B64'), ('valid', 'DATA_VALID_B64'))]
    meta_cell = hidden_code_cell('DATA_META = ' + repr(meta) + '\n', 'what the builder recorded about the rows')
    cells = [md_cell(TOP_MD), md_cell(SETTINGS_MD), code_cell(settings), md_cell(DATA_MD), *b64_cells, meta_cell,
             code_cell(COMMON_PY), g(IMPORT_COMMON_PY), md_cell(UNPACK_MD), g(UNPACK_PY), g(EFFECTIVE_PY),
             md_cell(GPU_MD), g(GPU_PY), md_cell(DRIVE_MD), g(DRIVE_PY), md_cell(INSTALL_MD), g(INSTALL_PY),
             md_cell(TRAIN_SCRIPT_MD), code_cell(TRAIN_SCRIPT_PY), md_cell(TRAIN_RUN_MD), g(TRAIN_RUN_PY), md_cell(SMOKE_MD), g(SMOKE_PY),
             md_cell(EXPORT_SCRIPT_MD), code_cell(EXPORT_SCRIPT_PY), md_cell(EXPORT_RUN_MD), g(EXPORT_RUN_PY),
             md_cell(FINISH_MD), g(FINISH_PY), md_cell(RELEASE_MD), g(RELEASE_PY)]
    return {'nbformat': 4, 'nbformat_minor': 0,
            'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': opts.get('gpu', 'A100'), 'machine_shape': 'hm', 'provenance': []},
                         'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'language_info': {'name': 'python'}},
            'cells': cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', help='train.jsonl (rows with messages, or system/user/assistant)')
    ap.add_argument('--valid', help='valid.jsonl')
    ap.add_argument('--fixture', action='store_true', help='placeholder notebook from synthetic rows; it refuses to train')
    ap.add_argument('--out', default=None)
    ap.add_argument('--gpu', default='A100', choices=['A100', 'H100'], help='Colab accelerator (A100 was unavailable once and Colab handed out an L4)')
    ap.add_argument('--smoke', action='store_true', help='smoke notebook: SMOKE = True and only the 20 + 6 rows the smoke uses')
    ap.add_argument('--run-name', default='r4')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--save-steps', type=int, default=50)
    ap.add_argument('--eval-steps', type=int, default=None, help='default: --save-steps')
    ap.add_argument('--max-seq', type=int, default=MAX_SEQ)
    ap.add_argument('--pins-json', help='pins_resolved.json from the smoke run: its versions become the pins')
    ap.add_argument('--tokenizer-dir', default=str(TOKENIZER_DIR))
    a = ap.parse_args()
    if bool(a.fixture) == bool(a.train or a.valid) or (not a.fixture and not (a.train and a.valid)):
        sys.exit('give --train and --valid, or --fixture')
    if a.fixture:
        train_lines, valid_lines = fixture_rows()
    else:
        train_lines, valid_lines = (lines_of(Path(p).read_bytes()) for p in (a.train, a.valid))
    pins = dict(DEFAULT_PINS)
    if a.pins_json:
        versions = json.loads(Path(a.pins_json).read_text())['versions']
        pins = {k: versions[k] for k in PIN_KEYS}
    out = Path(a.out) if a.out else (DEFAULT_OUT.with_name('train_gemma4_12b_r4_smoke.ipynb') if a.smoke else DEFAULT_OUT)
    opts = {'run_name': a.run_name, 'smoke': a.smoke, 'epochs': a.epochs, 'save_steps': a.save_steps, 'eval_steps': a.eval_steps or a.save_steps,
            'pins': pins, 'max_seq': a.max_seq, 'tokenizer_dir': a.tokenizer_dir, 'source': 'fixture' if a.fixture else 'rows', 'gpu': a.gpu}
    nb = build_notebook(train_lines, valid_lines, opts)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
    print(f"wrote {out} ({out.stat().st_size / 2**20:.2f} MiB), {len(nb['cells'])} cells, source={opts['source']}, smoke={a.smoke}")
    print(f"train {len(train_lines)} rows, valid {len(valid_lines)} rows; notebook sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == '__main__':
    main()
