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
    CHUNKS_V3_FILE,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_TOP_K,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_PRIMARY,
    FAISS_DIR,
    LLM_CONTEXT_PASSAGES,
    LLM_MAX_NEW_TOKENS,
    LLM_N_CTX,
    LLM_PASSAGE_MAX_CHARS,
    RETRIEVAL_FETCH_K,
    ANSWER_CONTEXT_MAX_CHARS,
    ANSWER_CONTEXT_NEIGHBORS,
    VECTOR_BACKEND,
)
from src.embeddings import BGEEmbedder
from src.generation.prompt import (
    REFUSAL_LINE,
    build_cited_prompt,
    parse_citations,
    strip_chat_artifacts,
)
from src.generation.compare import comparison_periods, generate_comparison_answer, prepare_answer_context
from src.generation.evidence_gate import assess_evidence
from src.retrieval.query_expansion import build_followup_retrieval_query
from src.evaluation.claim_validator import validate_and_filter_answer
from src.generation.providers import LLMProvider, create_llm_provider
from src.index_manifest import ensure_index_identity, write_index_identity
from src.retrieval import CrossEncoderReranker, Retriever
from src.retrieval.context import build_doc_lookup, expand_hits_with_neighbors, fit_context_to_llm
from src.vector_store import (
    FaissStore,
    SearchHit,
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
        # Fall back to the legacy corpus only when the active V3 corpus is absent.
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
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Run a query end-to-end; history resolves retrieval intent, not facts."""
        history = history or []
        retrieval_query = build_followup_retrieval_query(query, history)
        result = self.retriever.search(
            retrieval_query,
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

        extras = [turn["content"] for turn in history if turn.get("role") == "user"]
        periods = comparison_periods(query)
        evidence = assess_evidence(query, result.hits, extra_queries=extras)
        if not evidence.sufficient and not periods:
            return {
                "query": query,
                "strategy": strategy,
                "answer": REFUSAL_LINE,
                "passages": [_hit_to_dict(h) for h in result.hits],
                "retrieved_passages": [_hit_to_dict(h) for h in result.hits],
                "citations": [],
                "used_filter": result.used_filter,
                "reranked": result.reranked,
                "evidence": {"sufficient": False, "reason": evidence.reason,
                             "best_overlap": evidence.best_overlap},
            }

        context_hits = prepare_answer_context(
            self.llm, result.hits, self.docs_by_id, query, history=history,
            followup=retrieval_query != query,
            neighbors=ANSWER_CONTEXT_NEIGHBORS, max_chars=ANSWER_CONTEXT_MAX_CHARS,
            max_new_tokens=LLM_MAX_NEW_TOKENS, n_ctx=LLM_N_CTX,
            max_passages=LLM_CONTEXT_PASSAGES, passage_max_chars=LLM_PASSAGE_MAX_CHARS,
        )
        if periods:
            comparison = generate_comparison_answer(
                self.llm, query, context_hits, history=history,
                max_new_tokens=LLM_MAX_NEW_TOKENS, extra_queries=extras,
            )
            answer, citations, validation = comparison.answer, comparison.citations, comparison.validation
        else:
            prompt = build_cited_prompt(query, context_hits, history=history)
            raw_answer = self.llm.generate(prompt)
            answer = strip_chat_artifacts(raw_answer)
            validation = validate_and_filter_answer(answer, context_hits)
            answer = validation.safe_answer or REFUSAL_LINE
            citations = parse_citations(answer, context_hits)

        return {
            "query": query,
            "strategy": strategy,
            "answer": answer,
            "passages": [_hit_to_dict(h) for h in context_hits],
            "retrieved_passages": [_hit_to_dict(h) for h in result.hits],
            "citations": citations,
            "used_filter": result.used_filter,
            "reranked": result.reranked,
            "evidence": {"sufficient": True, "reason": evidence.reason,
                         "best_overlap": evidence.best_overlap},
            "citation_validation": {"blocked_claims": validation.blocked_claims,
                                     "validations": validation.validations,
                                     "nli_available": validation.nli_available},
        }


def _hit_to_dict(hit: SearchHit) -> Dict[str, Any]:
    return {
        "id": hit.id,
        "score": hit.score,
        "year": hit.metadata.get("year"),
        "source_file": hit.metadata.get("source_file"),
        "topics": hit.metadata.get("topics", ""),
        "text": hit.text,
    }
