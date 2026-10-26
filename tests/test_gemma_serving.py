"""ft-r4-finance group 3: serving a Gemma GGUF from deploy/modal/app.py, and scripts/cloud/upload_gguf.py.

The `modal` package is not installed here, so app.py is executed against a small fake `modal` module whose decorators
return the function unchanged: `serve()` is then the real function and its Popen calls are captured.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import config
from src.generation import finance_answer
from src.generation import prompt as prompt_module
from src.generation.providers.llama_provider import LlamaCppProvider
from src.generation.providers.remote_provider import RemoteProvider
from src.storage import SearchHit

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "deploy" / "modal" / "app.py"
UPLOAD = ROOT / "scripts" / "cloud" / "upload_gguf.py"

FT_R1 = "buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf"
GEMMA = "buffett-gemma4-12b-ft-r4-q4_k_m.gguf"
GEMMA_NAMES = [GEMMA, "buffett-gemma4-12b-finance-q4_k_m.gguf", "Gemma-4-12B-it-Q4_K_M.gguf"]
OLD_FT_R1_COMMAND = (
    "$(command -v llama-server || echo /app/llama-server) --model /models/" + FT_R1 + " --ctx-size 16384 "
    '--n-gpu-layers 999 --parallel 1 --jinja --api-key "$LLM_TOKEN" --host 127.0.0.1 --port 8080'
)  # the command app.py built before LLM_GGUF existed
SETTINGS = ("LLM_GGUF", "LLM_PROMPT_VERSION", "PROMPT_VERSION", "REMOTE_LLM_MODEL", "REMOTE_LLM_URL", "LLM_MODEL_PATH",
            "LLM_TOKENIZER_JSON", "ALLOW_PROMPT_MISMATCH", "LLM_CONTEXT_PASSAGES", "KNOWLEDGE_DIR", "FINANCE_PIPELINE",
            "FINANCE_ANSWER_V3")


# --- a fake `modal` to execute app.py against ---------------------------------------------------------------------
class _Chain:
    """modal.Image: every builder call is recorded and returns the chain."""

    def __init__(self) -> None:
        self.calls: list = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self

        return call


def _decorator(**_kwargs):
    return lambda fn: fn


def _fake_modal(local: bool) -> types.ModuleType:
    mod = types.ModuleType("modal")
    mod.chain = _Chain()
    mod.is_local = lambda: local
    mod.Image = types.SimpleNamespace(from_registry=lambda *a, **k: mod.chain, debian_slim=lambda *a, **k: mod.chain)
    mod.App = lambda name: types.SimpleNamespace(name=name, function=_decorator)
    mod.Volume = types.SimpleNamespace(from_name=lambda name: name)
    mod.Secret = types.SimpleNamespace(from_name=lambda name: name)
    mod.concurrent = _decorator
    mod.web_server = _decorator
    mod.fastapi_endpoint = _decorator
    return mod


@pytest.fixture
def load_app(monkeypatch):
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)

    def load(local: bool = True, **env: str):
        monkeypatch.setenv("APP_NAME", "buffettrag-app-staging")
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        fake = _fake_modal(local)
        monkeypatch.setitem(sys.modules, "modal", fake)
        spec = importlib.util.spec_from_file_location("modal_app_under_test", APP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.fake_modal = fake
        return module

    return load


class _Popen:
    calls: list = []

    def __init__(self, cmd, **kwargs) -> None:
        type(self).calls.append((cmd, kwargs))


def _serve(module, monkeypatch, tmp_path, gguf_name=None):
    """Run the real serve() with a models dir that holds the GGUF; returns (llama command, app Popen kwargs)."""
    assert module.MODELS_DIR == "/models"  # the volume mount; the test serves from a temp dir and maps it back
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    (models / (gguf_name or module.GGUF_NAME)).write_bytes(b"GGUF")
    monkeypatch.setattr(module, "MODELS_DIR", str(models))
    monkeypatch.setenv("LLM_TOKEN", "llm-secret-for-the-test")
    _Popen.calls = []
    monkeypatch.setattr(module.subprocess, "Popen", _Popen)
    monkeypatch.setattr(module, "_gpu_name", lambda: "NVIDIA L4")  # nvidia-smi would be one more Popen call
    module.serve()
    (llama, llama_kw), (app, app_kw) = _Popen.calls
    assert llama_kw == {"shell": True} and app[:3] == ["python", "-m", "uvicorn"]
    return llama.replace(str(models), "/models"), app_kw["env"]


# --- selection by setting; the default does not move --------------------------------------------------------------
def test_default_is_the_ft_r1_file_with_the_command_it_always_had(load_app, monkeypatch, tmp_path):
    module = load_app()
    assert module.GGUF_NAME == FT_R1 == module.DEFAULT_GGUF_NAME and module.MODEL_FAMILY == "qwen"
    llama, env = _serve(module, monkeypatch, tmp_path)
    assert llama == OLD_FT_R1_COMMAND
    for name in ("PROMPT_VERSION", "REMOTE_LLM_MODEL", "LLM_TOKENIZER_JSON", "LLM_MODEL_PATH"):
        assert name not in env  # nothing new reaches the app: the model file's own registry decides, as before
    assert env["DEFAULT_LLM_PROVIDER"] == "remote" and env["REMOTE_LLM_RETRIES"] == "24"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_setting_is_the_default(load_app, blank):
    assert load_app(LLM_GGUF=blank).GGUF_NAME == FT_R1


def test_llm_gguf_selects_the_file(load_app, monkeypatch, tmp_path):
    module = load_app(LLM_GGUF=f"  {GEMMA} ")
    assert module.GGUF_NAME == GEMMA and module.MODEL_FAMILY == "gemma"
    llama, _env = _serve(module, monkeypatch, tmp_path)
    assert f"--model /models/{GEMMA} " in llama


@pytest.mark.parametrize("bad", ["../x.gguf", "/models/x.gguf", "a/b.gguf", "x y.gguf", "x;rm -rf.gguf", "$(id).gguf",
                                 "x.gguf; id", "model.bin", ".hidden.gguf", "-x.gguf", "a..b.gguf", "x\n.gguf"])
def test_bad_file_names_are_refused(load_app, bad):
    with pytest.raises(ValueError, match="LLM_GGUF"):
        load_app(LLM_GGUF=bad)


def test_a_file_missing_on_the_volume_fails_at_start_and_lists_what_is_there(load_app, monkeypatch, tmp_path):
    module = load_app(LLM_GGUF=GEMMA)
    models = tmp_path / "models"
    models.mkdir()
    (models / FT_R1).write_bytes(b"GGUF")
    monkeypatch.setattr(module, "MODELS_DIR", str(models))
    monkeypatch.setenv("LLM_TOKEN", "t")
    monkeypatch.setattr(module.subprocess, "Popen", _Popen)
    _Popen.calls = []
    with pytest.raises(RuntimeError, match=f"{GEMMA}.*{FT_R1}"):
        module.serve()
    assert _Popen.calls == []  # neither llama-server nor the app was started


def test_the_setting_is_baked_into_the_image_and_other_variables_are_not(load_app, monkeypatch):
    module = load_app(LLM_GGUF=GEMMA, LLM_PROMPT_VERSION="v3", FINANCE_PIPELINE="1", KNOWLEDGE_DIR="/k", SOME_OTHER="x")
    image_envs = [args[0] for name, args, _kw in module.fake_modal.chain.calls if name == "env"]
    assert len(image_envs) == 2  # the serving image, then the CPU status image (only its app name)
    image_env, status_env = image_envs
    assert status_env == {"STATUS_APP_NAME": module.APP_NAME}
    assert image_env["LLM_GGUF"] == GEMMA and image_env["LLM_PROMPT_VERSION"] == "v3"
    assert image_env["FINANCE_PIPELINE"] == "1" and image_env["KNOWLEDGE_DIR"] == "/k"
    assert "SOME_OTHER" not in image_env and "LLM_TOKEN" not in image_env
    assert load_app(local=False, LLM_GGUF=GEMMA).GGUF_NAME == GEMMA  # the container re-reads it from the image env
    assert load_app(local=False)._baked_env() == {}


# --- flags per model family ---------------------------------------------------------------------------------------
def _argv_of(command: str, tmp_path: Path) -> list:
    """Run the llama-server command in a shell with a fake llama-server that prints its arguments."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    script = fake_bin / "llama-server"
    script.write_text("#!/bin/sh\nfor a in \"$@\"; do printf '%s\\n' \"$a\"; done\n")
    script.chmod(0o755)
    env = {"PATH": f"{fake_bin}:/usr/bin:/bin", "LLM_TOKEN": "tok"}
    out = subprocess.run(command, shell=True, env=env, capture_output=True, text=True, check=True).stdout
    return out.splitlines()


def test_qwen_family_flags_are_the_common_ones_only(load_app, tmp_path):
    module = load_app()
    argv = _argv_of(module.llama_server_command(FT_R1, "/models"), tmp_path)
    assert argv == ["--model", f"/models/{FT_R1}", "--ctx-size", "16384", "--n-gpu-layers", "999", "--parallel", "1",
                    "--jinja", "--api-key", "tok", "--host", "127.0.0.1", "--port", "8080"]
    assert module.llama_server_command("other-qwen-model.gguf", "/models").count("--reasoning") == 0


@pytest.mark.parametrize("name", GEMMA_NAMES)
def test_gemma_family_flags(load_app, tmp_path, name):
    module = load_app()
    assert module.model_family(name) == "gemma"
    argv = _argv_of(module.llama_server_command(name, "/models"), tmp_path)
    assert argv[:2] == ["--model", f"/models/{name}"]
    assert argv[argv.index("--ctx-size") + 1] == "16384" and "--jinja" in argv  # the chat template of the GGUF applies
    assert argv[argv.index("--reasoning") + 1] == "off"  # the reader needs `content`, not a reasoning block
    assert json.loads(argv[argv.index("--chat-template-kwargs") + 1]) == {"enable_thinking": False}
    assert argv[argv.index("--api-key") + 1] == "tok"
    assert argv[-4:] == ["--host", "127.0.0.1", "--port", "8080"]


def test_gemma_container_settings(load_app, monkeypatch, tmp_path):
    module = load_app(LLM_GGUF=GEMMA)
    llama, env = _serve(module, monkeypatch, tmp_path)
    assert "--reasoning off" in llama
    assert env["PROMPT_VERSION"] == "v2"  # the finance system prompt stays v2 text
    assert env["REMOTE_LLM_MODEL"] == GEMMA
    assert env["LLM_TOKENIZER_JSON"] == "/root/app/data/tokenizer/gemma4_12b_it/tokenizer.json"
    assert "LLM_MODEL_PATH" not in env  # config.py would read its 7B name and cut the served context to 5 passages
    assert env["REMOTE_LLM_URL"] == "http://127.0.0.1:8080" and env["REMOTE_LLM_TOKEN"] == "llm-secret-for-the-test"


def test_prompt_version_can_be_set_per_deploy_and_is_checked(load_app, monkeypatch, tmp_path):
    _llama, env = _serve(load_app(LLM_GGUF=GEMMA, LLM_PROMPT_VERSION="V3"), monkeypatch, tmp_path)
    assert env["PROMPT_VERSION"] == "v3"
    with pytest.raises(ValueError, match="LLM_PROMPT_VERSION"):  # refused at deploy time, when app.py is imported
        load_app(LLM_GGUF=GEMMA, LLM_PROMPT_VERSION="v9")


def test_a_different_qwen_file_names_itself_but_keeps_the_ft_r1_flags(load_app, monkeypatch, tmp_path):
    name = "buffett-qwen2.5-7b-ft-r3-q4_k_m.gguf"
    llama, env = _serve(load_app(LLM_GGUF=name), monkeypatch, tmp_path)
    assert llama == OLD_FT_R1_COMMAND.replace(FT_R1, name)
    assert env["REMOTE_LLM_MODEL"] == name and "PROMPT_VERSION" not in env and "LLM_TOKENIZER_JSON" not in env


def test_the_gemma_tokenizer_ships_only_for_gemma(load_app):
    def shipped(module) -> list:
        return [args[1] for name, args, _kw in module.fake_modal.chain.calls if name == "add_local_file"]

    gemma_tokenizer = ROOT / "data" / "tokenizer" / "gemma4_12b_it" / "tokenizer.json"  # local only, git-ignored
    default_dests = shipped(load_app())
    assert not any("gemma4_12b_it" in dest for dest in default_dests)
    assert any(d.endswith("data/tokenizer/qwen_tokenizer.json") for d in default_dests)
    gemma_dests = shipped(load_app(LLM_GGUF=GEMMA))
    assert any(d.endswith("data/tokenizer/gemma4_12b_it/tokenizer.json") for d in gemma_dests) == gemma_tokenizer.is_file()
    assert any(d.endswith("data/tokenizer/qwen_tokenizer.json") for d in gemma_dests)  # the Qwen file stays for every family


def test_the_llama_image_tag_was_not_bumped_for_gemma():
    # Evidence in app.py: b11718 (18 Oct 2026) is newer than every Gemma 4 / 12B Unified loader change (#21326 ... #24118,
    # #23398). Changing the tag needs a new finding that this build cannot load the Gemma GGUF.
    assert 'IMAGE_TAG = "server-cuda-b11718"' in APP.read_text()
    assert "Not bumped for Gemma 4 12B" in APP.read_text()


# --- the prompt reaches Gemma as system + user chat messages, v2 text -------------------------------------------
def _hits() -> list:
    return [SearchHit("1987_p0001", "Oil prices fell in 1986. We lost money.", {"year": 1987, "source_file": "1987.pdf"}, 1.0),
            SearchHit("2008_p0004", "The 2008 crisis hurt insurers. We kept our cash.", {"year": 2008, "source_file": "2008.pdf"}, 1.0)]


@pytest.mark.parametrize("name", GEMMA_NAMES)
def test_the_app_settings_build_a_remote_provider_that_sends_system_and_user(load_app, monkeypatch, name):
    module = load_app(LLM_GGUF=name)
    for key in SETTINGS:
        monkeypatch.delenv(key, raising=False)
    for key, value in module.app_env_for_model(name, os.environ).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("REMOTE_LLM_URL", "http://127.0.0.1:8080")

    provider = RemoteProvider()  # must not raise "PROMPT_VERSION contradicts the model" for a name like ...-ft-r4-...
    assert provider.model == name and prompt_module.prompt_version() == "v2"
    assert config.llm_context_passages() == 16  # the served-context budget did not follow the file name

    reader = finance_answer.reader_prompt_v3("Did Buffett discuss the oil crisis?", _hits(), None, "Note: no 1999 oil crisis.")
    messages = provider._messages(reader)
    assert [m["role"] for m in messages] == ["system", "user"]  # Gemma 4 has a native system role
    assert messages[0]["content"] == prompt_module.PROMPT_VERSIONS["v2"]
    assert messages[1]["content"].startswith("BEGIN UNTRUSTED PASSAGES") and messages[1]["content"].endswith("Answer:")
    assert "LETTER OF 1987" in messages[1]["content"] and "Note: no 1999 oil crisis." in messages[1]["content"]
    assert provider._payload(reader, 900, stream=False)["messages"] == messages
    llama = LlamaCppProvider.__new__(LlamaCppProvider)
    assert llama._messages(reader) == messages  # the embedded provider sends the same two messages


def test_a_gemma_file_name_is_not_read_as_a_qwen_round():
    for name in GEMMA_NAMES:
        assert prompt_module.expected_prompt_version(name) is None
    assert prompt_module.expected_prompt_version("buffett-qwen2.5-7b-ft-r4-q4_k_m.gguf") == "v3"  # the Qwen registry is as before
    assert prompt_module.expected_prompt_version(FT_R1) == "v1"


# --- scripts/cloud/upload_gguf.py -----------------------------------------------------------------------------------
@pytest.fixture
def upload():
    spec = importlib.util.spec_from_file_location("upload_gguf_under_test", UPLOAD)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


class _FakeModal:
    """Replaces subprocess.run (put) and subprocess.Popen (get ... -) in the upload script."""

    def __init__(self, remote_bytes: bytes, put_code: int = 0, put_err: str = "", get_code: int = 0) -> None:
        self.remote_bytes, self.put_code, self.put_err, self.get_code = remote_bytes, put_code, put_err, get_code
        self.run_calls: list = []
        self.popen_calls: list = []

    def run(self, cmd, **kwargs):
        self.run_calls.append(cmd)
        assert kwargs.get("shell") is not True
        return types.SimpleNamespace(returncode=self.put_code, stdout="", stderr=self.put_err)

    def popen(self, cmd, **kwargs):
        self.popen_calls.append(cmd)
        assert kwargs.get("shell") is not True
        code = self.get_code
        return types.SimpleNamespace(stdout=io.BytesIO(self.remote_bytes), wait=lambda: code)

    @property
    def calls(self) -> list:
        return self.run_calls + self.popen_calls


@pytest.fixture
def gguf(tmp_path) -> tuple:
    data = b"GGUF" + b"\x03\x00\x00\x00" + bytes(range(256)) * 40
    path = tmp_path / GEMMA
    path.write_bytes(data)
    return path, data


def _patch(upload, monkeypatch, fake):
    monkeypatch.setattr(upload.shutil, "which", lambda name: f"/usr/local/bin/{name}")
    monkeypatch.setattr(upload.subprocess, "run", fake.run)
    monkeypatch.setattr(upload.subprocess, "Popen", fake.popen)
    return fake


def test_upload_puts_the_file_then_verifies_the_sha256(upload, monkeypatch, gguf, capsys):
    import hashlib
    path, data = gguf
    fake = _patch(upload, monkeypatch, _FakeModal(data))
    assert upload.main([str(path), "--sha256", hashlib.sha256(data).hexdigest().upper()]) == 0
    assert fake.run_calls == [["modal", "volume", "put", "buffettrag-models", str(path.resolve()), f"/{GEMMA}"]]
    assert fake.popen_calls == [["modal", "volume", "get", "buffettrag-models", f"/{GEMMA}", "-"]]
    out = capsys.readouterr().out
    assert hashlib.sha256(data).hexdigest() in out and "OK sha256 matches" in out and f"LLM_GGUF={GEMMA}" in out


def test_name_volume_and_force_are_passed_through(upload, monkeypatch, gguf):
    path, data = gguf
    fake = _patch(upload, monkeypatch, _FakeModal(data))
    assert upload.main([str(path), "--name", "gemma-r4.gguf", "--volume", "vol2", "--force", "--modal", "mymodal"]) == 0
    assert fake.run_calls == [["mymodal", "volume", "put", "--force", "vol2", str(path.resolve()), "/gemma-r4.gguf"]]
    assert fake.popen_calls == [["mymodal", "volume", "get", "vol2", "/gemma-r4.gguf", "-"]]


def test_a_changed_remote_file_fails_verification(upload, monkeypatch, gguf, capsys):
    path, data = gguf
    _patch(upload, monkeypatch, _FakeModal(data[:-1] + b"X"))
    assert upload.main([str(path)]) == 4
    assert "VERIFY FAILED" in capsys.readouterr().err
    _patch(upload, monkeypatch, _FakeModal(data[:-10]))  # truncated
    assert upload.main([str(path)]) == 4


def test_dry_run_calls_nothing(upload, monkeypatch, gguf, capsys):
    path, data = gguf
    fake = _patch(upload, monkeypatch, _FakeModal(data))
    assert upload.main([str(path), "--dry-run"]) == 0
    assert fake.calls == []
    assert "dry run: would run `modal volume put buffettrag-models" in capsys.readouterr().out


@pytest.mark.parametrize("make,args", [
    ("missing", []),
    ("not_gguf_suffix", []),
    ("bad_magic", []),
    ("good", ["--name", "../x.gguf"]),
    ("good", ["--name", "a b.gguf"]),
    ("good", ["--name", "x;id.gguf"]),
    ("good", ["--name", "model.bin"]),
    ("good", ["--name", FT_R1]),
    ("good", ["--name", FT_R1, "--force"]),
    ("good", ["--sha256", "0" * 64]),
    ("good", ["--sha256", "not-a-hash"]),
    ("good", ["--sha256", "ab" * 31]),
])
def test_argument_and_file_checks_stop_before_any_modal_call(upload, monkeypatch, tmp_path, gguf, make, args):
    path, data = gguf
    target = {"missing": tmp_path / "nope.gguf", "not_gguf_suffix": tmp_path / "m.bin",
              "bad_magic": tmp_path / "m.gguf", "good": path}[make]
    if make in ("not_gguf_suffix", "bad_magic"):
        target.write_bytes(data if make == "not_gguf_suffix" else b"GGML" + data[4:])
    fake = _patch(upload, monkeypatch, _FakeModal(data))
    assert upload.main([str(target), *args]) == 2
    assert fake.calls == []


def test_a_missing_modal_cli_is_reported(upload, monkeypatch, gguf, capsys):
    path, data = gguf
    fake = _patch(upload, monkeypatch, _FakeModal(data))
    monkeypatch.setattr(upload.shutil, "which", lambda name: None)
    assert upload.main([str(path)]) == 2 and fake.calls == []
    assert "not found" in capsys.readouterr().err


def test_modal_failures_exit_3_and_never_print_secrets(upload, monkeypatch, gguf, capsys):
    path, data = gguf
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "supersecret-value-123")
    monkeypatch.setenv("SOME_API_KEY", "apikey-value-456")
    leak = "File exists. token supersecret-value-123 key apikey-value-456 id ak-AbCdEfGhIjKlMnOpQr secret as-ZyXwVuTsRqPoNmLkJi hf_abcdefghij1234"
    fake = _patch(upload, monkeypatch, _FakeModal(data, put_code=1, put_err=leak))
    assert upload.main([str(path)]) == 3
    assert fake.popen_calls == []  # no verification after a failed upload
    seen = capsys.readouterr()
    text = seen.out + seen.err
    for secret in ("supersecret-value-123", "apikey-value-456", "ak-AbCdEfGhIjKlMnOpQr", "as-ZyXwVuTsRqPoNmLkJi", "hf_abcdefghij1234"):
        assert secret not in text
    assert "File exists" in text and "failed with exit code 1" in text
    assert not any("supersecret" in part or "apikey" in part for cmd in fake.calls for part in cmd)  # none on a command line

    fake = _patch(upload, monkeypatch, _FakeModal(data, get_code=1))
    assert upload.main([str(path)]) == 3  # the read-back failed


def test_scrub_and_tail(upload):
    env = {"MODAL_TOKEN_SECRET": "abcdef123456", "PATH": "/usr/bin", "SHORT_TOKEN": "abc"}
    assert upload.scrub("x abcdef123456 y /usr/bin abc", env) == "x *** y /usr/bin abc"
    long = "\n".join(f"line {i} " + "z" * 500 for i in range(50))
    shown = upload.tail(long).splitlines()
    assert len(shown) == upload.TAIL_LINES and all(len(line) <= upload.TAIL_LINE_CHARS for line in shown)


def test_the_script_is_python311_clean_and_stdlib_only():
    source = UPLOAD.read_text()
    compile(source, str(UPLOAD), "exec")
    imported = {line.split()[1].split(".")[0] for line in source.splitlines() if line.startswith(("import ", "from "))}
    assert imported <= {"__future__", "argparse", "hashlib", "os", "re", "shutil", "subprocess", "sys", "tempfile", "time",
                        "pathlib", "typing"}
