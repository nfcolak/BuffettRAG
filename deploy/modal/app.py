"""BuffettRAG on Modal: one L4 container runs the whole app (UI, API, retrieval on CUDA, FT-r1 via llama.cpp).

Why one container: the bge reranker takes ~17-19 s per question on CPU (measured on the Mac) and ~1-2 s on the GPU.
Deploy:  modal deploy deploy/modal/app.py   (needs the volume `buffettrag-models` with the FT-r1 GGUF and the
secrets `buffettrag-llm-token` (LLM_TOKEN) and `buffettrag-site-password` (SITE_PASSWORD)).
Staging:  APP_NAME=buffettrag-app-staging FINANCE_PIPELINE=1 [FINANCE_DEBUG=1] modal deploy deploy/modal/app.py
The live name `buffettrag-app` is deployed only with DEPLOY_LIVE=1.
"""
import ast
import os
import subprocess
from pathlib import Path

import modal

LIVE_APP_NAME = "buffettrag-app"
APP_NAME = os.environ.get("APP_NAME", LIVE_APP_NAME)
IMAGE_TAG = "server-cuda-b11718"  # llama.cpp build, pinned 2026-10-18; bump it deliberately
GGUF_NAME = "buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf"
MODELS_DIR = "/models"
APP_DIR = "/root/app"
GPU = "L4"
CTX_SIZE = 16384
LLM_PORT = 8080
WEB_PORT = 8000
SCALEDOWN_WINDOW_S = 300
STARTUP_TIMEOUT_S = 900
MAX_CONTAINERS = 1

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
    return {k: os.environ[k] for k in ("FINANCE_PIPELINE", "FINANCE_DEBUG", "KNOWLEDGE_DIR") if os.environ.get(k)}


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

if modal.is_local():
    _warn_missing_summary_index()

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
    llm = (
        f"$(command -v llama-server || echo /app/llama-server) --model {MODELS_DIR}/{GGUF_NAME} "
        f"--ctx-size {CTX_SIZE} --n-gpu-layers 999 --parallel 1 --jinja "
        f'--api-key "$LLM_TOKEN" --host 127.0.0.1 --port {LLM_PORT}'
    )
    subprocess.Popen(llm, shell=True)
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
    subprocess.Popen(
        ["python", "-m", "uvicorn", "src.services.space_app:app", "--host", "0.0.0.0", "--port", str(WEB_PORT)],
        cwd=APP_DIR, env=env,
    )
