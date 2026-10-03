"""Run retrieval ablation with honest unavailable entries for nonlocal model stacks."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from time import perf_counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import EMBEDDING_MODEL_PRIMARY, RERANKER_MODEL
from src.storage.embeddings import BGEEmbedder
from src.evaluation.ablation import benchmark_configurations
from src.storage.index_manifest import load_and_validate_index_manifest
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.storage import load_chunks_as_docs
from src.storage.faiss_store import FaissStore


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_index_manifest(manifest, *, corpus: Path, document_ids, model_name: str, dimension: int) -> None:
    """Reject an index unless its corpus, IDs, and embedding config match exactly."""
    if manifest.get("corpus_sha256") != _sha256(corpus):
        raise ValueError("index corpus hash does not match requested corpus")
    ids_hash = hashlib.sha256("\n".join(document_ids).encode("utf-8")).hexdigest()
    if manifest.get("document_ids_sha256") != ids_hash:
        raise ValueError("index document ID hash does not match requested corpus")
    if int(manifest.get("document_count", -1)) != len(document_ids):
        raise ValueError("index document count does not match requested corpus")
    embedding = manifest.get("embedding") or {}
    if embedding.get("model") != model_name or int(embedding.get("dimension", -1)) != dimension:
        raise ValueError("index embedding configuration does not match requested model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--faiss-dir", type=Path, default=None,
                        help="Optional corpus-bound FAISS directory for embedding runs.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    corpus = args.corpus.resolve()
    cases_path = args.cases.resolve()
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    docs = load_chunks_as_docs(corpus)
    bm25 = BM25Retriever(docs)

    def run_bm25(case):
        started = perf_counter()
        ids = [hit.id for hit in bm25.search(case["query"], top_k=8)]
        return ids, (perf_counter() - started) * 1000

    runners = {"bm25": run_bm25}
    model_config = {
        "embedding": {"model": EMBEDDING_MODEL_PRIMARY, "status": "unavailable"},
        "reranker": {"model": RERANKER_MODEL, "status": "unavailable"},
    }

    if args.faiss_dir is not None:
        manifest_path = args.faiss_dir / "index_manifest.json"
        if not manifest_path.exists():
            parser.error(f"Missing index manifest: {manifest_path}")
        embedder = BGEEmbedder(model_name=EMBEDDING_MODEL_PRIMARY, device=args.device)
        try:
            load_and_validate_index_manifest(
                args.faiss_dir,
                corpus=corpus,
                docs=docs,
                backend="faiss",
                model_name=embedder.model_name,
                dimension=embedder.dimension,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        vector_store = FaissStore(persist_dir=args.faiss_dir)
        if len(vector_store) != len(docs):
            parser.error("FAISS row count does not match the requested corpus")

        def run_embedding(case):
            started = perf_counter()
            query_embedding = embedder.embed_query(case["query"])
            ids = [hit.id for hit in vector_store.search(query_embedding, top_k=8)]
            return ids, (perf_counter() - started) * 1000

        reranker = CrossEncoderReranker(model_name=RERANKER_MODEL, device=args.device)

        def run_embedding_reranker(case):
            started = perf_counter()
            query_embedding = embedder.embed_query(case["query"])
            candidates = vector_store.search(query_embedding, top_k=30)
            ids = [hit.id for hit in reranker.rerank(case["query"], candidates, top_k=8)]
            return ids, (perf_counter() - started) * 1000

        runners.update({
            "embedding_only": run_embedding,
            "embedding_reranker": run_embedding_reranker,
        })
        model_config = {
            "embedding": {"model": embedder.model_name, "dimension": embedder.dimension, "status": "ok"},
            "reranker": {"model": RERANKER_MODEL, "status": "ok"},
        }

    result = benchmark_configurations(cases, runners)
    result.update({
        "mode": "offline_retrieval_ablation_no_generation_provider",
        "corpus": str(corpus.relative_to(ROOT)),
        "corpus_sha256": _sha256(corpus),
        "cases": str(cases_path.relative_to(ROOT)),
        "cases_sha256": _sha256(cases_path),
        "models": model_config,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({name: value["status"] for name, value in result["configurations"].items()}, indent=2))
