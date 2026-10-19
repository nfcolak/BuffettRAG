"""Modal deployment of the BuffettRAG FT-r1 model behind an OpenAI-compatible llama.cpp server.

Deploy with `modal deploy deploy/modal/serve_llm.py` (see README.md). The HF Space calls
`<endpoint>/v1/chat/completions` with `Authorization: Bearer <LLM_TOKEN>`.
"""
import subprocess

import modal

APP_NAME = "buffettrag-llm"
# Pinned llama.cpp build (newest server-cuda-b<N> tag on ghcr.io on 2026-10-18).
IMAGE_TAG = "server-cuda-b11718"
IMAGE_REF = f"ghcr.io/ggml-org/llama.cpp:{IMAGE_TAG}"
GGUF_NAME = "buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf"
VOLUME_NAME = "buffettrag-models"
MODELS_DIR = "/models"
SECRET_NAME = "buffettrag-llm-token"
GPU = "L4"
CTX_SIZE = 16384
N_GPU_LAYERS = 999
PORT = 8080
SCALEDOWN_WINDOW_S = 300
STARTUP_TIMEOUT_S = 600
MAX_CONTAINERS = 1

app = modal.App(APP_NAME)
image = modal.Image.from_registry(IMAGE_REF, add_python="3.11").entrypoint([])  # the llama.cpp image has no Python
models = modal.Volume.from_name(VOLUME_NAME)


@app.function(
    image=image,
    gpu=GPU,
    volumes={MODELS_DIR: models},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    scaledown_window=SCALEDOWN_WINDOW_S,
    max_containers=MAX_CONTAINERS,
)
@modal.concurrent(max_inputs=1)
@modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT_S)
def serve() -> None:
    """Start the llama.cpp server in the background; Modal proxies the port."""
    # The shell expands $LLM_TOKEN from the secret env, so the key never appears in this file.
    cmd = (
        # The image keeps the binary in /app, which is not always on PATH.
        f"$(command -v llama-server || echo /app/llama-server) --model {MODELS_DIR}/{GGUF_NAME} "
        f"--ctx-size {CTX_SIZE} --n-gpu-layers {N_GPU_LAYERS} --parallel 1 --jinja "
        f'--api-key "$LLM_TOKEN" --host 0.0.0.0 --port {PORT}'
    )
    subprocess.Popen(cmd, shell=True)
