"""LLM service.

Deployment: runs on the Nuvolos **Trainer T4** instance (GPU).
Responsibilities:
    - Loads Qwen2.5-7B-Instruct on the T4 GPU
    - Exposes a single /generate endpoint that the Backend calls

Why a separate service?
    The Backend app on Nuvolos has no GPU. The Trainer has the GPU but is
    isolated from the rest of the apps (its own network namespace). To bridge
    this, we run a thin FastAPI server on the Trainer and have the Backend
    POST prompts to it. The Trainer must expose a public port (or use Nuvolos
    instance-wide networking if available) for the Backend to reach it.

Run:
    uvicorn services.llm_service:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LLM_MODEL_PRIMARY
from src.generation.llm import LocalLLM


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 600


class GenerateResponse(BaseModel):
    text: str
    model: str


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

app = FastAPI(title="BuffettRAG LLM Service", version="1.0")
_state: dict = {"llm": None}


@app.on_event("startup")
def _startup() -> None:
    print(f"[llm] Loading {LLM_MODEL_PRIMARY} on GPU...")
    _state["llm"] = LocalLLM(model_name=LLM_MODEL_PRIMARY, device="cuda")
    print("[llm] startup complete")


@app.get("/health")
def health() -> dict:
    return {
        "ok": _state["llm"] is not None,
        "model": LLM_MODEL_PRIMARY,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    llm: Optional[LocalLLM] = _state["llm"]
    if llm is None:
        raise HTTPException(503, "LLM not initialized")
    text = llm.generate(req.prompt, max_new_tokens=req.max_new_tokens)
    return GenerateResponse(text=text, model=LLM_MODEL_PRIMARY)
