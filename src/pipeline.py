"""End-to-end RAG pipeline.

Composes: vector_store + embedder + retriever + (optional) reranker + LLM.

Usage:
    pipeline = BuffettRAGPipeline.build()
    out = pipeline.ask("How did Buffett react to the 2008 financial crisis?")
    print(out['answer'])
    for c in out['citations']:
        print(c)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    CHUNKS_FILE,
    CHUNKS_V2_FILE,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_TOP_K,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_PRIMARY,
    RETRIEVAL_FETCH_K,
    ANSWER_CONTEXT_MAX_CHARS,
    ANSWER_CONTEXT_NEIGHBORS,
    VECTOR_BACKEND,
)
from src.embeddings import BGEEmbedder
from src.generation.prompt import (
    build_cited_prompt,
    parse_citations,
    strip_chat_artifacts,
)
from src.generation.providers import LLMProvider, create_llm_provider
from src.retrieval import CrossEncoderReranker, Retriever
from src.retrieval.context import build_doc_lookup, expand_hits_with_neighbors
from src.vector_store import (
    SearchHit,
    StoredDoc,
    get_vector_store,
    load_chunks_as_docs,
)


@dataclass
class PipelineConfig:
    chunks_file: Path = CHUNKS_V2_FILE
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
        # Fall back to v1 chunks if v2 hasn't been generated yet.
        chunks_path = cfg.chunks_file if cfg.chunks_file.exists() else CHUNKS_FILE
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"No chunks file found. Run ingestion first. Looked at: "
                f"{cfg.chunks_file} and {CHUNKS_FILE}"
            )
        print(f"Loading chunks from {chunks_path}")
        docs: List[StoredDoc] = load_chunks_as_docs(chunks_path)
        print(f"  loaded {len(docs)} chunks")

        embedder = BGEEmbedder(model_name=cfg.embedding_model, device=cfg.device)

        store = get_vector_store(backend=cfg.vector_backend, dim=embedder.dimension)
        if len(store) == 0:
            print(f"Vector store empty -- building index ({cfg.vector_backend})")
            texts = [d.text for d in docs]
            embeddings = embedder.embed_documents(texts)
            store.add(docs, embeddings)
        else:
            print(f"Vector store already populated: {len(store)} vectors")

        reranker = CrossEncoderReranker(device=cfg.device) if cfg.use_reranker else None

        retriever = Retriever(
            vector_store=store,
            embedder=embedder,
            docs=docs,
            reranker=reranker,
        )

        llm = create_llm_provider(provider=cfg.llm_provider) if cfg.use_llm else None
        return cls(retriever=retriever, docs_by_id=build_doc_lookup(docs), llm=llm)

    # --------------------------------------------------------------- query API

    def ask(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = DEFAULT_TOP_K,
        fetch_k: int = RETRIEVAL_FETCH_K,
        rerank: bool = True,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a single query end-to-end."""
        result = self.retriever.search(
            query,
            strategy=strategy,
            top_k=top_k,
            fetch_k=fetch_k,
            rerank=rerank,
            where=where,
        )

        if self.llm is None:
            return {
                "query": query,
                "strategy": strategy,
                "answer": None,
                "passages": [_hit_to_dict(h) for h in result.hits],
                "citations": [],
                "used_filter": result.used_filter,
                "reranked": result.reranked,
            }

        context_hits = expand_hits_with_neighbors(
            result.hits,
            self.docs_by_id,
            neighbors=ANSWER_CONTEXT_NEIGHBORS,
            max_chars=ANSWER_CONTEXT_MAX_CHARS,
        )
        prompt = build_cited_prompt(query, context_hits)
        raw_answer = self.llm.generate(prompt)
        answer = strip_chat_artifacts(raw_answer)
        citations = parse_citations(answer, context_hits)

        return {
            "query": query,
            "strategy": strategy,
            "answer": answer,
            "passages": [_hit_to_dict(h) for h in result.hits],
            "citations": citations,
            "used_filter": result.used_filter,
            "reranked": result.reranked,
        }

    def compare_decades(
        self,
        query: str,
        decades=(1980, 1990, 2000, 2010, 2020),
        per_decade_k: int = 3,
        rerank: bool = True,
    ) -> Dict[int, List[Dict]]:
        """Cross-decade retrieval -- one bucket per decade."""
        per_decade = self.retriever.cross_decade_compare(
            query, decades=decades, per_decade_k=per_decade_k, rerank=rerank
        )
        return {d: [_hit_to_dict(h) for h in hits] for d, hits in per_decade.items()}


def _hit_to_dict(hit: SearchHit) -> Dict[str, Any]:
    return {
        "id": hit.id,
        "score": hit.score,
        "year": hit.metadata.get("year"),
        "source_file": hit.metadata.get("source_file"),
        "topics": hit.metadata.get("topics", ""),
        "text": hit.text,
    }
