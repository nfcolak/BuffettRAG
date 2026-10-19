"""Static checks of deploy/modal/app.py (the modal package is not installed locally)."""
import re
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "deploy" / "modal" / "app.py"
TEXT = APP.read_text()


def test_app_name_default_and_env_override() -> None:
    assert 'LIVE_APP_NAME = "buffettrag-app"' in TEXT
    assert 'APP_NAME = os.environ.get("APP_NAME", LIVE_APP_NAME)' in TEXT
    assert "modal.App(APP_NAME)" in TEXT


def test_live_name_needs_deploy_live() -> None:
    guard = re.search(r"if modal\.is_local\(\):\n(.*?)\n\n\n", TEXT, re.S)
    assert guard, "deploy-time guard missing"
    body = guard.group(1)
    assert "print(" in body and "APP_NAME" in body
    assert "APP_NAME == LIVE_APP_NAME" in body
    assert 'os.environ.get("DEPLOY_LIVE") != "1"' in body
    assert "SystemExit" in body


def test_finance_settings_baked_through_image_env() -> None:
    assert '"FINANCE_PIPELINE"' in TEXT and '"KNOWLEDGE_DIR"' in TEXT
    assert "**_baked_env()" in TEXT
    fn = TEXT.split("def _baked_env", 1)[1].split("def _download_retrieval_models", 1)[0]
    assert "modal.is_local()" in fn
    # image.env must come before the add_local_* steps (Modal rejects build steps after them)
    img = TEXT.split("image = (", 1)[1].split("\n)\n", 1)[0]
    assert img.index("_baked_env()") < img.index(".add_local_file")


def test_knowledge_dir_included_only_when_present() -> None:
    m = re.search(r'if \(REPO / "data/knowledge"\)\.is_dir\(\):\n\s+image = image\.add_local_dir\(([^\n]+)\)', TEXT)
    assert m and 'f"{APP_DIR}/data/knowledge"' in m.group(1)
    assert TEXT.index("add_local_dir(str(REPO / \"data/knowledge\")") > TEXT.index("image = (")


def test_no_secret_literal() -> None:
    assert not re.search(r"(?i)(password|token|secret|api[_-]?key)\s*=\s*[\"'][^\"']+[\"']", TEXT)
    assert not re.search(r"\b(sk-|hf_|ghp_)[A-Za-z0-9]{10,}", TEXT)
    assert 'os.environ["LLM_TOKEN"]' in TEXT


def test_cloud_serving_defaults_and_requirements_path() -> None:
    assert 'DEFAULT_LLM_PROVIDER="remote"' in TEXT
    assert 'VECTOR_BACKEND="faiss"' in TEXT
    assert 'REPO / "requirements-cloud.txt"' in TEXT
    assert "requirements-space" not in TEXT and "serve_llm" not in TEXT
    assert (APP.parents[2] / "requirements-cloud.txt").is_file()
