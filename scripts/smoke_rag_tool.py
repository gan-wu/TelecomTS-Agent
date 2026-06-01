from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.telecom_knowledge_tool import KnowledgeSearchConfig, TelecomKnowledgeTool  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test search_telecom_knowledge RAG tool.")
    parser.add_argument(
        "--query",
        default="What does high UL_BLER mean for 5G uplink troubleshooting?",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--reranker-batch-size", type=int, default=None)
    parser.add_argument("--disable-reranker", action="store_true")
    parser.add_argument("--output", default="knowledge_base/retrieval/rag_tool_smoke.json")
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    config = KnowledgeSearchConfig.from_env()
    config = replace(
        config,
        embedding_device=args.embedding_device or config.embedding_device,
        reranker_device=args.reranker_device or config.reranker_device,
        reranker_batch_size=args.reranker_batch_size or config.reranker_batch_size,
        reranker_enabled=not args.disable_reranker,
    )
    tool = TelecomKnowledgeTool(config)
    result = tool.search_telecom_knowledge(args.query, top_k=args.top_k)

    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

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
        print(f"\n#{rank} {label}={float(score):.4f}")
        print(f"source={hit['source']}")
        print(f"section={hit['section']}")
        print(f"dense_rank={hit['dense_rank']} bm25_rank={hit['bm25_rank']}")
        print(f"evidence={hit['evidence'][:260]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
