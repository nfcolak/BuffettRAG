"""Put a local GGUF on the Modal volume `buffettrag-models` and check it there by sha256.

    python scripts/cloud/upload_gguf.py path/to/model-q4_k_m.gguf [--name FILE] [--sha256 HEX] [--dry-run] [--force]

1. The local file must be a GGUF (magic bytes) with a plain file name; its sha256 is computed, and compared with --sha256
   when you pass the one the export printed (nothing is uploaded on a mismatch).
2. `modal volume put` uploads it as /<name>. An existing file is not replaced unless --force; the live FT-r1 file is never
   replaced.
3. `modal volume get <volume> /<name> -` streams the file back and its sha256 and size are compared with the local ones.
   That reads the whole file again (as long as the upload took); nothing is written to disk.

Only the Modal CLI is used (it holds your login). This script never reads a token, passes none on a command line, and
scrubs token-shaped text and the values of TOKEN/SECRET/KEY/PASSWORD environment variables from anything it prints.
Exit codes: 0 ok, 2 bad arguments or file, 3 the Modal CLI failed, 4 the sha256 on the volume differs.
Then serve it:  LLM_GGUF=<name> APP_NAME=buffettrag-app-staging modal deploy deploy/modal/app.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

DEFAULT_VOLUME = "buffettrag-models"
PROTECTED_NAMES = ("buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf",)  # the live FT-r1 reader: upload under another name
GGUF_MAGIC = b"GGUF"
CHUNK = 16 << 20
TAIL_LINES = 12
TAIL_LINE_CHARS = 300

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}\.gguf")
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_SECRET_ENV = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|CREDENTIAL", re.I)
_SECRET_SHAPE = re.compile(r"\b(?:a[ks]-[A-Za-z0-9]{12,}|hf_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{10,})\b")


class UploadError(Exception):
    """A failure with an exit code; the message is already safe to print."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def scrub(text: str, env: Optional[dict] = None) -> str:
    """`text` without the values of secret-named environment variables and without token-shaped strings."""
    for name, value in (os.environ if env is None else env).items():
        if value and len(value) >= 6 and _SECRET_ENV.search(name):
            text = text.replace(value, "***")
    return _SECRET_SHAPE.sub("***", text)


def tail(text: str) -> str:
    lines = [line[:TAIL_LINE_CHARS] for line in text.strip().splitlines()[-TAIL_LINES:]]
    return scrub("\n".join(lines))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Upload a GGUF to the Modal volume and verify its sha256.")
    ap.add_argument("gguf", type=Path, help="local .gguf file")
    ap.add_argument("--name", help="file name on the volume (default: the local file name)")
    ap.add_argument("--volume", default=DEFAULT_VOLUME, help=f"Modal volume (default {DEFAULT_VOLUME})")
    ap.add_argument("--sha256", help="expected sha256 of the local file (from the export step)")
    ap.add_argument("--force", action="store_true", help="replace a file of the same name on the volume")
    ap.add_argument("--dry-run", action="store_true", help="check the file and print the plan; call no Modal command")
    ap.add_argument("--modal", default="modal", help="Modal CLI executable (default: modal on PATH)")
    return ap.parse_args(argv)


def check_name(name: str) -> str:
    if not _NAME_RE.fullmatch(name) or ".." in name:
        raise UploadError(f"bad file name {name!r}: use letters, digits, . _ - and end with .gguf (no path)", 2)
    return name


def check_local(path: Path) -> Path:
    if not path.is_file():
        raise UploadError(f"{path} is not a file", 2)
    if path.suffix != ".gguf":
        raise UploadError(f"{path.name} does not end with .gguf", 2)
    try:
        with path.open("rb") as handle:
            magic = handle.read(len(GGUF_MAGIC))
    except OSError as exc:
        raise UploadError(f"cannot read {path}: {type(exc).__name__}", 2) from exc
    if magic != GGUF_MAGIC:
        raise UploadError(f"{path.name} is not a GGUF file (magic bytes {magic!r})", 2)
    return path


def sha256_file(path: Path) -> Tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def run_modal(cmd: List[str]) -> None:
    """Run one Modal CLI command; on failure raise with the scrubbed tail of its output."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise UploadError(f"`{' '.join(cmd[:3])} ...` failed with exit code {result.returncode}\n"
                          f"{tail(result.stderr or '')}\n{tail(result.stdout or '')}".rstrip(), 3)


def remote_sha256(modal: str, volume: str, name: str) -> Tuple[str, int]:
    """sha256 and size of /<name> on the volume, read through `modal volume get ... -` (stdout), not stored."""
    digest, size = hashlib.sha256(), 0
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen([modal, "volume", "get", volume, f"/{name}", "-"], stdout=subprocess.PIPE, stderr=errors)
        try:
            while chunk := proc.stdout.read(CHUNK):
                digest.update(chunk)
                size += len(chunk)
        finally:
            proc.stdout.close()
            code = proc.wait()
        if code != 0:
            errors.seek(0)
            raise UploadError(f"`modal volume get` failed with exit code {code}\n"
                              f"{tail(errors.read().decode('utf-8', 'replace'))}".rstrip(), 3)
    return digest.hexdigest(), size


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        path = check_local(args.gguf)
        name = check_name(args.name or path.name)
        if name in PROTECTED_NAMES:
            raise UploadError(f"{name} is the live FT-r1 reader; upload under another --name", 2)
        if args.sha256 is not None and not _SHA_RE.fullmatch(args.sha256.lower()):
            raise UploadError("--sha256 must be 64 hex characters", 2)
        local_sha, local_size = sha256_file(path)
        print(f"local   {path.name}  {local_size} bytes  sha256 {local_sha}")
        if args.sha256 is not None and args.sha256.lower() != local_sha:
            raise UploadError(f"local sha256 differs from --sha256 ({args.sha256.lower()}); nothing uploaded", 2)
        put = [args.modal, "volume", "put", *(["--force"] if args.force else []), args.volume, str(path.resolve()), f"/{name}"]
        if args.dry_run:
            print(f"dry run: would run `{' '.join(put)}`, then read /{name} back and compare the sha256")
            return 0
        if shutil.which(args.modal) is None:
            raise UploadError(f"Modal CLI {args.modal!r} not found on PATH (pass --modal PATH)", 2)
        started = time.monotonic()
        print(f"upload  {args.volume}:/{name}  (the Modal CLI shows no progress here; this can take several minutes)")
        run_modal(put)
        print(f"uploaded in {time.monotonic() - started:.0f} s; reading it back to verify")
        remote_sha, remote_size = remote_sha256(args.modal, args.volume, name)
        if (remote_sha, remote_size) != (local_sha, local_size):
            raise UploadError(f"VERIFY FAILED: volume has {remote_size} bytes, sha256 {remote_sha}; local has {local_size} "
                              f"bytes, sha256 {local_sha}. Upload again with --force.", 4)
        print(f"remote  /{name}  {remote_size} bytes  sha256 {remote_sha}")
        print(f"OK sha256 matches. Serve it: LLM_GGUF={name} APP_NAME=<staging name> modal deploy deploy/modal/app.py")
        return 0
    except UploadError as exc:
        print(f"ERROR: {scrub(str(exc))}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
