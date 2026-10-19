"""Join the GGUF parts uploaded to the Modal volume and verify the sha256 of the whole file (CPU only).

Run once after scripts upload `/parts/part_*`:  modal run deploy/modal/join_parts.py
The result must be byte-identical to the local FT-r1 file (sha256 below), so serving matches the measured model.
"""
import hashlib
import os

import modal

VOLUME_NAME = "buffettrag-models"
MODELS_DIR = "/models"
GGUF_NAME = "buffett-qwen2.5-7b-ft-r1-q4_k_m.gguf"
EXPECTED_SHA256 = "ec41100e1457ffc88289d42c9c56de5bddb648c511db693cb75fd2d628182ec1"  # models/ft/7b_r1/SHA256SUMS

app = modal.App("buffettrag-join-gguf")
volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(volumes={MODELS_DIR: volume}, timeout=1800, cpu=2, memory=2048)
def join() -> str:
    parts_dir = os.path.join(MODELS_DIR, "parts")
    parts = sorted(p for p in os.listdir(parts_dir) if p.startswith("part_"))
    if not parts:
        raise RuntimeError("no parts in /models/parts")
    target = os.path.join(MODELS_DIR, GGUF_NAME)
    digest = hashlib.sha256()
    with open(target + ".tmp", "wb") as out:
        for name in parts:
            with open(os.path.join(parts_dir, name), "rb") as src:
                while chunk := src.read(16 << 20):
                    digest.update(chunk)
                    out.write(chunk)
    got = digest.hexdigest()
    if got != EXPECTED_SHA256:
        os.remove(target + ".tmp")
        raise RuntimeError(f"sha256 mismatch: {got}")
    os.replace(target + ".tmp", target)
    for name in os.listdir(parts_dir):
        os.remove(os.path.join(parts_dir, name))
    volume.commit()
    return f"{GGUF_NAME} ok, {len(parts)} parts, {os.path.getsize(target)} bytes, sha256 {got}"


@app.local_entrypoint()
def main() -> None:
    print(join.remote())
