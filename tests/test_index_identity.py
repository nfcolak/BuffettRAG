import json
from pathlib import Path

import pytest

from src.storage import index_manifest as im
from src.storage import StoredDoc


class FakePg:
    table = "t_fake"

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def test_pgvector_like_store_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(im, "MANIFEST_SIDECAR_DIR", tmp_path / "m")
    corpus = tmp_path / "c.jsonl"
    corpus.write_text("x")
    docs = [StoredDoc(id="a", text="a", metadata={}), StoredDoc(id="b", text="b", metadata={})]
    kw = dict(backend="pgvector", corpus=corpus, docs=docs, model_name="m", dimension=4)
    store = FakePg(2)
    with pytest.raises(RuntimeError, match="rebuild"):
        im.ensure_index_identity(store, **kw)  # missing manifest
    im.write_index_identity(store, **kw)
    im.ensure_index_identity(store, **kw)
    with pytest.raises(RuntimeError, match="rebuild"):
        im.ensure_index_identity(store, **{**kw, "model_name": "other"})
    corpus.write_text("changed")
    with pytest.raises(RuntimeError, match="rebuild"):
        im.ensure_index_identity(store, **kw)
