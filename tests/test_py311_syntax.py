"""The Modal image runs Python 3.11: every tracked source file must compile there (no 3.12-only syntax such as
backslashes inside f-string expressions). Uses a local 3.11 interpreter when one exists."""
import glob
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _python311():
    found = shutil.which("python3.11")
    if found:
        return found
    hits = sorted(glob.glob(os.path.expanduser("~/.local/share/uv/python/cpython-3.11*/bin/python3.11")))
    return hits[-1] if hits else None


def test_sources_compile_on_python311():
    py = _python311()
    if not py:
        pytest.skip("no Python 3.11 interpreter on this machine")
    files = [str(p) for d in ("src", "scripts", "deploy") for p in (ROOT / d).rglob("*.py")] + [str(ROOT / "config.py")]
    code = ("import sys\nbad=[]\nfor f in sys.argv[1:]:\n    try: compile(open(f).read(), f, 'exec')\n"
            "    except SyntaxError as e: bad.append(f'{f}:{e.lineno} {e.msg}')\nprint('\\n'.join(bad)); sys.exit(1 if bad else 0)")
    res = subprocess.run([py, "-c", code, *files], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
