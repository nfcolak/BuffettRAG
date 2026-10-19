"""Static checks of deploy/modal/serve_llm.py (the modal package is not installed locally)."""
import ast
import re
from pathlib import Path

SERVE = Path(__file__).resolve().parent.parent / "deploy" / "modal" / "serve_llm.py"
REQUIRED_CONSTANTS = {
    "APP_NAME", "IMAGE_TAG", "GGUF_NAME", "VOLUME_NAME", "SECRET_NAME",
    "GPU", "CTX_SIZE", "PORT", "MAX_CONTAINERS",
}


def _constants() -> dict:
    tree = ast.parse(SERVE.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value.value
    return out


def test_parses_and_defines_constants() -> None:
    consts = _constants()
    assert REQUIRED_CONSTANTS <= consts.keys()
    assert consts["GGUF_NAME"] == "buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf"
    assert consts["APP_NAME"] == "buffettrag-llm"


def test_server_settings() -> None:
    consts = _constants()
    text = SERVE.read_text()
    assert consts["GPU"] == "L4"
    assert consts["CTX_SIZE"] == 16384
    assert consts["PORT"] == 8080
    assert consts["MAX_CONTAINERS"] == 1
    assert "--jinja" in text
    assert "--ctx-size" in text
    assert '--api-key "$LLM_TOKEN"' in text
    assert "max_containers=MAX_CONTAINERS" in text
    assert "web_server(port=PORT" in text


def test_no_secret_literal() -> None:
    text = SERVE.read_text()
    assert not re.search(r"""(?i)(token|api[-_]?key|secret)\s*=\s*["'][^"']{12,}["']""", text)
    assert not re.search(r"\b(?:hf|sk|ghp)_[A-Za-z0-9]{16,}", text)
    # the key is only ever read from the secret env
    assert "$LLM_TOKEN" in text
