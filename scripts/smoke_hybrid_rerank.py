from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.embedding_store import EmbeddingConfig  # noqa: E402
from src.rag.hybrid_retriever import (  # noqa: E402
    HybridRetriever,
    HybridRetrieverConfig,
    save_retrieval_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test dense + BM25 + RRF + BGE reranker retrieval."
    )
    parser.add_argument(
        "--query",
        default="How can O-RAN KPI monitoring help troubleshoot 5G network performance?",
    )
    parser.add_argument("--chunks", default="knowledge_base/chunks/knowledge_chunks.jsonl")
    parser.add_argument("--persist-dir", default="knowledge_base/chroma")
    parser.add_argument("--collection", default="telecom_knowledge_bge_m3")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-max-length", type=int, default=1536)
    parser.add_argument("--embedding-batch-size", type=int, default=1)
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--embedding-fp16", action="store_true", default=True)
    parser.add_argument("--dense-top-k", type=int, default=30)
    parser.add_argument("--bm25-top-k", type=int, default=30)
    parser.add_argument("--fused-top-k", type=int, default=30)
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--reranker-max-length", type=int, default=1024)
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    parser.add_argument("--reranker-device", default="cuda:0")
    parser.add_argument("--reranker-fp16", action="store_true", default=True)
    parser.add_argument("--disable-reranker", action="store_true")
    parser.add_argument("--output", default="knowledge_base/retrieval/hybrid_rerank_smoke.json")
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    embedding_config = EmbeddingConfig(
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        use_fp16=args.embedding_fp16,
        return_sparse=False,
        devices=args.embedding_device,
    )
    retriever_config = HybridRetrieverConfig(
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        fused_top_k=args.fused_top_k,
        final_top_k=args.final_top_k,
        reranker_model=args.reranker_model,
        reranker_max_length=args.reranker_max_length,
        reranker_batch_size=args.reranker_batch_size,
        reranker_use_fp16=args.reranker_fp16,
        reranker_device=args.reranker_device,
        reranker_enabled=not args.disable_reranker,
    )
    retriever = HybridRetriever(
        chunks_path=resolve_project_path(args.chunks),
        chroma_dir=resolve_project_path(args.persist_dir),
        collection_name=args.collection,
        embedding_config=embedding_config,
        retriever_config=retriever_config,
    )
    result = retriever.retrieve(args.query)
    output_path = resolve_project_path(args.output)
    save_retrieval_result(result, output_path)

    print(f"Query: {result['query']}")
    print(
        f"Dense={result['dense_count']} BM25={result['bm25_count']} "
        f"Fused={result['fused_count']} Final={result['final_count']}"
    )
    print(
        f"Ranking mode: {result.get('ranking_mode')} "
        f"reranker_available={result.get('reranker_available')}"
    )
    if result.get("reranker_error"):
        print(f"Reranker fallback reason: {result['reranker_error']}")
    print(f"Output: {output_path.relative_to(PROJECT_ROOT)}")
    for rank, hit in enumerate(result["hits"], start=1):
        score = hit.get("rerank_score")
        label = "rerank"
        if score is None:
            score = hit.get("rank_score", hit.get("rrf_score", 0.0))
            label = hit.get("ranker", "rank")
        print(f"\n#{rank} {label}={float(score):.4f} rrf={hit['rrf_score']:.4f}")
        print(f"source={hit['source']}")
        print(f"section={hit['section']}")
        print(
            f"dense_rank={hit['dense_rank']} bm25_rank={hit['bm25_rank']} "
            f"bm25={hit['bm25_score']:.4f}"
        )
        print(f"preview={hit['preview']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
