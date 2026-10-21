"""BuffettRAG on Modal: one L4 container runs the whole app (UI, API, retrieval on CUDA, FT-r1 via llama.cpp).

Why one container: the bge reranker takes ~17-19 s per question on CPU (measured on the Mac) and ~1-2 s on the GPU.
Deploy:  modal deploy deploy/modal/app.py   (needs the volume `buffettrag-models` with the FT-r1 GGUF and the
secrets `buffettrag-llm-token` (LLM_TOKEN) and `buffettrag-site-password` (SITE_PASSWORD)).
Staging:  APP_NAME=buffettrag-app-staging FINANCE_PIPELINE=1 [FINANCE_DEBUG=1] modal deploy deploy/modal/app.py
The live name `buffettrag-app` is deployed only with DEPLOY_LIVE=1.
Reader model: LLM_GGUF=<file name on the volume> picks the GGUF (default: FT-r1). A Gemma GGUF gets its own llama-server
flags, prompt version and tokenizer (`MODEL_FLAGS`, `app_env_for_model`); upload it with scripts/cloud/upload_gguf.py.
"""
import ast
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import modal

LIVE_APP_NAME = "buffettrag-app"
APP_NAME = os.environ.get("APP_NAME", LIVE_APP_NAME)
IMAGE_TAG = "server-cuda-b11718"  # llama.cpp build, pinned 2026-10-18; bump it deliberately
# Not bumped for Gemma 4 12B (checked 2026-10-21 in the llama.cpp release notes). b11718 is from 18 Oct 2026 (#30239); the
# Gemma 4 work it contains is older: parser #21326 and tokenizer #21343 (2-3 Apr 2026), "Fix Gemma 4 Unified conversion"
# #24118 (4 Jun), Gemma 4 MTP #23398 (7 Jun). Builds after it touch Gemma only in the multimodal projector (b11772, #30882)
# and the Hexagon backend (b11723, #30731): neither is on this text-only CUDA path. The flags used below are in the server
# README at tag b11718 (--jinja, -rea/--reasoning, --chat-template-kwargs). Bump only after a Gemma load fails on this tag.
DEFAULT_GGUF_NAME = "buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf"  # FT-r1: serves unless LLM_GGUF names another file
TOKENIZER_FILE = "data/tokenizer/qwen_tokenizer.json"  # Qwen BPE tokenizer for the reader-window count (about 7 MB)
GEMMA_TOKENIZER_FILE = "data/tokenizer/gemma4_12b_it/tokenizer.json"  # Gemma 4 12B tokenizer (32 MB, kept local, not in git)
MODELS_DIR = "/models"
APP_DIR = "/root/app"
GPU = "L4"
CTX_SIZE = 16384
LLM_PORT = 8080
WEB_PORT = 8000
SCALEDOWN_WINDOW_S = 300
STARTUP_TIMEOUT_S = 900
MAX_CONTAINERS = 1

# --- reader model: which GGUF, and what its family needs -------------------------------------------------------------
FAMILY_QWEN, FAMILY_GEMMA = "qwen", "gemma"  # every file name without "gemma" keeps the FT-r1 (Qwen) flags
PROMPT_VERSIONS = ("v1", "v2", "v3")  # the keys of src/generation/prompt.py PROMPT_VERSIONS
GEMMA_PROMPT_VERSION = "v2"  # FT-r4 finance: the system prompt stays v2 text (override per deploy with LLM_PROMPT_VERSION)
_GGUF_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}\.gguf")
# Gemma 4 is a thinking model: llama-server would put the answer behind a reasoning block, and the remote provider reads
# only `content`. Thinking is switched off in both documented places (the server flag and the template argument).
MODEL_FLAGS = {
    FAMILY_QWEN: (),  # the ChatML template in the GGUF metadata, through --jinja; nothing else
    FAMILY_GEMMA: ("--reasoning off", "--chat-template-kwargs " + shlex.quote(json.dumps({"enable_thinking": False}))),
}


def resolve_gguf(env) -> str:
    """The reader GGUF: LLM_GGUF (a file name on the volume), else the FT-r1 file. The name goes into a shell command."""
    name = (env.get("LLM_GGUF") or "").strip()
    if not name:
        return DEFAULT_GGUF_NAME
    if not _GGUF_NAME_RE.fullmatch(name) or ".." in name:
        raise ValueError(f"LLM_GGUF must be a plain .gguf file name on the volume (letters, digits, . _ -), got {name!r}")
    return name


def model_family(gguf_name: str) -> str:
    return FAMILY_GEMMA if "gemma" in gguf_name.lower() else FAMILY_QWEN


def resolve_prompt_version(env, family: str):
    """PROMPT_VERSION for the app: LLM_PROMPT_VERSION, else the family's (Gemma: v2). None leaves it to the model file."""
    raw = (env.get("LLM_PROMPT_VERSION") or "").strip().lower()
    if raw and raw not in PROMPT_VERSIONS:
        raise ValueError(f"LLM_PROMPT_VERSION must be one of {PROMPT_VERSIONS}, got {raw!r}")
    return raw or (GEMMA_PROMPT_VERSION if family == FAMILY_GEMMA else None)


def llama_server_command(gguf_name: str, models_dir: str) -> str:
    """The llama-server shell command: the flags every model gets, then the ones of its family."""
    return " ".join([
        "$(command -v llama-server || echo /app/llama-server)", f"--model {models_dir}/{gguf_name}",
        f"--ctx-size {CTX_SIZE}", "--n-gpu-layers 999", "--parallel 1", "--jinja",
        *MODEL_FLAGS[model_family(gguf_name)],
        '--api-key "$LLM_TOKEN"', "--host 127.0.0.1", f"--port {LLM_PORT}",
    ])


def app_env_for_model(gguf_name: str, env) -> dict:
    """App settings this model needs on top of the container defaults; nothing for the FT-r1 file.

    LLM_MODEL_PATH is left unset on purpose: config.py derives the 16-passage budget of the 7B reader from its name, and
    a Gemma file name would drop the served context to 5 passages.
    """
    family = model_family(gguf_name)
    out = {}
    if gguf_name != DEFAULT_GGUF_NAME:
        out["REMOTE_LLM_MODEL"] = gguf_name  # the model field of each request and the prompt-version check name the real file
    version = resolve_prompt_version(env, family)
    if version:
        out["PROMPT_VERSION"] = version
    if family == FAMILY_GEMMA:  # the reader-window count while llama.cpp still loads (src/generation/token_count.py)
        out["LLM_TOKENIZER_JSON"] = f"{APP_DIR}/{GEMMA_TOKENIZER_FILE}"
    return out


def require_gguf(models_dir: str, gguf_name: str) -> None:
    """Fail at container start with the list of GGUF files on the volume, not with a llama-server that never comes up."""
    path = Path(models_dir) / gguf_name
    if not path.is_file():
        found = sorted(p.name for p in Path(models_dir).glob("*.gguf")) if Path(models_dir).is_dir() else []
        raise RuntimeError(f"{path} is not on the volume buffettrag-models (GGUF files there: {found or 'none'}); "
                           "upload it with scripts/cloud/upload_gguf.py")


GGUF_NAME = resolve_gguf(os.environ)
MODEL_FAMILY = model_family(GGUF_NAME)
resolve_prompt_version(os.environ, MODEL_FAMILY)  # a bad LLM_PROMPT_VERSION stops the deploy, not the first request

# Local paths only matter when building/deploying; inside the container this file sits at /root/<name>.py.
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path(APP_DIR)


if modal.is_local():
    print(f"modal app name: {APP_NAME}")
    if APP_NAME == LIVE_APP_NAME and os.environ.get("DEPLOY_LIVE") != "1":
        raise SystemExit(f"refusing to deploy the live app '{LIVE_APP_NAME}' without DEPLOY_LIVE=1 (set APP_NAME for staging)")


def _baked_env() -> dict:
    """Deploy-time settings copied into the image: a local env variable does not reach the container otherwise."""
    if not modal.is_local():
        return {}
    # FINANCE_* settings (config.FINANCE_DEFAULTS overrides, e.g. a staging trial of a new answer setting), KNOWLEDGE_DIR and
    # the reader model (LLM_GGUF, LLM_PROMPT_VERSION): the container reads them again when it imports this file
    baked = ("KNOWLEDGE_DIR", "LLM_GGUF", "LLM_PROMPT_VERSION")
    return {k: v for k, v in os.environ.items() if v and (k.startswith("FINANCE_") or k in baked)}


def _download_retrieval_models() -> None:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
    CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")


def _shipped_finance_default(name: str):
    """A FINANCE_DEFAULTS value of the config.py that ships in the image, read without importing it."""
    for node in ast.parse((REPO / "config.py").read_text()).body:
        target = node.target if isinstance(node, ast.AnnAssign) else (node.targets[0] if isinstance(node, ast.Assign) else None)
        if getattr(target, "id", None) == "FINANCE_DEFAULTS":
            return ast.literal_eval(node.value).get(name)
    return None


def _warn_missing_summary_index() -> None:
    """Deploy-time check, never a failure: the summary source is on in the shipped config but its index is absent.

    `data/knowledge/summary_index/` is git-ignored, so a fresh clone deploys without it; the app then serves without
    the summary candidates (the finance route records the skip in its debug dict). Build it with
    scripts/index/build_summary_index.py before deploying. The container runs the shipped config (the environment is
    not baked), so that is the "on" the check reads.
    """
    try:
        index = REPO / "data/knowledge/summary_index"
        if _shipped_finance_default("FINANCE_SUMMARY_SOURCE") and not (index / "index_manifest.json").is_file():
            print("WARNING: FINANCE_SUMMARY_SOURCE is on but data/knowledge/summary_index/ is missing; the deployed app "
                  "will serve without the summary candidates. Run scripts/index/build_summary_index.py first.")
    except Exception as exc:  # noqa: BLE001 - a warning must never stop a deploy
        print(f"WARNING: could not check the summary index ({type(exc).__name__}: {exc})")


def _warn_missing_tokenizer() -> None:
    """Deploy-time check, never a failure: the token window is on in the shipped config but the tokenizer file is absent.

    The window trim (src/generation/token_count.py) counts with the llama.cpp server's /tokenize. While the server loads
    the model it answers 503, and the trim then needs the shipped tokenizer; without it the count falls back to
    characters / 4, which undercounts number-dense prompts by a fifth or more.
    """
    try:
        if _shipped_finance_default("FINANCE_WINDOW_TOKENS") and not (REPO / TOKENIZER_FILE).is_file():
            print(f"WARNING: FINANCE_WINDOW_TOKENS is on but {TOKENIZER_FILE} is missing; the deployed app will count "
                  "the reader window in characters / 4 while llama.cpp is loading.")
    except Exception as exc:  # noqa: BLE001 - a warning must never stop a deploy
        print(f"WARNING: could not check the tokenizer file ({type(exc).__name__}: {exc})")


def _warn_missing_gemma_tokenizer() -> None:
    """Deploy-time check, never a failure: a Gemma reader is selected but its tokenizer file is absent.

    The count then falls back to the Qwen tokenizer while llama.cpp loads, which is the wrong vocabulary for Gemma.
    """
    try:
        if MODEL_FAMILY == FAMILY_GEMMA and not (REPO / GEMMA_TOKENIZER_FILE).is_file():
            print(f"WARNING: a Gemma GGUF is selected but {GEMMA_TOKENIZER_FILE} is missing; the deployed app will count "
                  "the reader window with the Qwen tokenizer while llama.cpp is loading.")
    except Exception as exc:  # noqa: BLE001 - a warning must never stop a deploy
        print(f"WARNING: could not check the Gemma tokenizer file ({type(exc).__name__}: {exc})")


image = (
    modal.Image.from_registry(f"ghcr.io/ggml-org/llama.cpp:{IMAGE_TAG}", add_python="3.11")
    .entrypoint([])
    .pip_install_from_requirements(str(REPO / "requirements-cloud.txt"))
    .run_function(_download_retrieval_models)
    .env({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", **_baked_env()})
    .add_local_file(str(REPO / "config.py"), f"{APP_DIR}/config.py")
    .add_local_dir(str(REPO / "src"), f"{APP_DIR}/src", ignore=["**/__pycache__/**"])
    .add_local_dir(str(REPO / "frontend" / "dist"), f"{APP_DIR}/frontend/dist")
    .add_local_file(str(REPO / "data/processed/chunks_v3_paragraph.jsonl"),
                    f"{APP_DIR}/data/processed/chunks_v3_paragraph.jsonl")
    .add_local_file(str(REPO / "data/processed/chunks_v3_paragraph_manifest.json"),
                    f"{APP_DIR}/data/processed/chunks_v3_paragraph_manifest.json")
    .add_local_dir(str(REPO / "data/indices/faiss_v3"), f"{APP_DIR}/data/indices/faiss_v3")
)

if (REPO / "data/knowledge").is_dir():
    image = image.add_local_dir(str(REPO / "data/knowledge"), f"{APP_DIR}/data/knowledge")

if (REPO / TOKENIZER_FILE).is_file():  # the window trim's second token source (src/generation/token_count.py)
    image = image.add_local_file(str(REPO / TOKENIZER_FILE), f"{APP_DIR}/{TOKENIZER_FILE}")

if MODEL_FAMILY == FAMILY_GEMMA and (REPO / GEMMA_TOKENIZER_FILE).is_file():  # a Gemma reader counts with its own vocabulary
    image = image.add_local_file(str(REPO / GEMMA_TOKENIZER_FILE), f"{APP_DIR}/{GEMMA_TOKENIZER_FILE}")

if modal.is_local():
    _warn_missing_summary_index()
    _warn_missing_tokenizer()
    _warn_missing_gemma_tokenizer()
    print(f"reader model: {GGUF_NAME} ({MODEL_FAMILY} flags)")

app = modal.App(APP_NAME)
models = modal.Volume.from_name("buffettrag-models")


@app.function(
    image=image,
    gpu=GPU,
    volumes={MODELS_DIR: models},
    secrets=[modal.Secret.from_name("buffettrag-llm-token"), modal.Secret.from_name("buffettrag-site-password")],
    scaledown_window=SCALEDOWN_WINDOW_S,
    max_containers=MAX_CONTAINERS,
    timeout=3600,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=WEB_PORT, startup_timeout=STARTUP_TIMEOUT_S)
def serve() -> None:
    """Start llama.cpp on localhost and the BuffettRAG app on WEB_PORT; Modal proxies WEB_PORT."""
    require_gguf(MODELS_DIR, GGUF_NAME)
    print(f"serving {GGUF_NAME} ({MODEL_FAMILY} flags)")
    subprocess.Popen(llama_server_command(GGUF_NAME, MODELS_DIR), shell=True)
    env = dict(os.environ)
    env.update(
        DEFAULT_LLM_PROVIDER="remote",
        REMOTE_LLM_URL=f"http://127.0.0.1:{LLM_PORT}",
        REMOTE_LLM_TOKEN=os.environ["LLM_TOKEN"],
        REMOTE_LLM_RETRIES="24",  # llama.cpp answers 503 while the model loads (~20 s)
        VECTOR_BACKEND="faiss",
        FAISS_INDEX_NAME="faiss_v3",
        EMBEDDING_DEVICE="cuda",
        # pv8 decision 2026-10-19 (user): validator v2 flags on in serving; code defaults stay 0 so stored runs replay unchanged.
        VALIDATOR_SPLIT_V2="1",
        VALIDATOR_VERBATIM_KEEP="1",
        FRONTEND_DIST=f"{APP_DIR}/frontend/dist",
    )
    env.update(app_env_for_model(GGUF_NAME, os.environ))
    subprocess.Popen(
        ["python", "-m", "uvicorn", "src.services.space_app:app", "--host", "0.0.0.0", "--port", str(WEB_PORT)],
        cwd=APP_DIR, env=env,
    )
