"""End-to-end RAG pipeline.

Composes: vector_store + embedder + retriever + (optional) reranker + LLM.

Builds the components only. Answers come from one path, ``src/services/ask_flow``
(the same one ``/ask`` serves); see ``scripts/ask.py`` for how a script drives it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    CHUNKS_V3_FILE,
    DEFAULT_LLM_PROVIDER,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_PRIMARY,
    VECTOR_BACKEND,
)
from src.storage.embeddings import BGEEmbedder
from src.generation.providers import LLMProvider, create_llm_provider
from src.storage.index_manifest import ensure_index_identity, write_index_identity
from src.retrieval import CrossEncoderReranker, Retriever
from src.retrieval.context import build_doc_lookup
from src.storage import (
    StoredDoc,
    get_vector_store,
    load_chunks_as_docs,
)


@dataclass
class PipelineConfig:
    chunks_file: Path = CHUNKS_V3_FILE
    embedding_model: str = EMBEDDING_MODEL_PRIMARY
    vector_backend: str = VECTOR_BACKEND
    device: str = EMBEDDING_DEVICE
    use_reranker: bool = True
    use_llm: bool = True
    llm_provider: str = DEFAULT_LLM_PROVIDER


class BuffettRAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        docs_by_id: Optional[Dict[str, StoredDoc]] = None,
        llm: Optional[LLMProvider] = None,
    ) -> None:
        self.retriever = retriever
        self.docs_by_id = docs_by_id or {}
        self.llm = llm

    # --------------------------------------------------------------- factory

    @classmethod
    def build(cls, cfg: Optional[PipelineConfig] = None) -> "BuffettRAGPipeline":
        cfg = cfg or PipelineConfig()
        chunks_path = cfg.chunks_file
        if not chunks_path.exists():
            raise FileNotFoundError(f"No chunks file found. Run ingestion first. Looked at: {chunks_path}")
        print(f"Loading chunks from {chunks_path}")
        docs: List[StoredDoc] = load_chunks_as_docs(chunks_path)
        print(f"  loaded {len(docs)} chunks")

        embedder = BGEEmbedder(model_name=cfg.embedding_model, device=cfg.device)

        store = get_vector_store(backend=cfg.vector_backend, dim=embedder.dimension)
        identity = dict(
            backend=cfg.vector_backend, corpus=chunks_path, docs=docs,
            model_name=embedder.model_name, dimension=embedder.dimension,
        )
        if len(store) == 0:
            print(f"Vector store empty -- building index ({cfg.vector_backend})")
            embeddings = embedder.embed_documents([d.text for d in docs])
            store.add(docs, embeddings)
            write_index_identity(store, **identity)
        else:
            ensure_index_identity(store, **identity)
            print(f"Vector store already populated: {len(store)} vectors")

        reranker = CrossEncoderReranker(device=cfg.device) if cfg.use_reranker else None

        retriever = Retriever(
            vector_store=store,
            embedder=embedder,
            docs=docs,
            reranker=reranker,
        )

        if not cfg.use_llm:
            llm = None
        else:
            llm = create_llm_provider(provider=cfg.llm_provider)
        return cls(retriever=retriever, docs_by_id=build_doc_lookup(docs), llm=llm)
