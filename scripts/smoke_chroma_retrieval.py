from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.embedding_store import EmbeddingConfig, query_chroma  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test BGE-M3 + Chroma retrieval.")
    parser.add_argument(
        "--query",
        default="How can O-RAN KPI monitoring help troubleshoot 5G network performance?",
    )
    parser.add_argument("--persist-dir", default="knowledge_base/chroma")
    parser.add_argument("--collection", default="telecom_knowledge_bge_m3")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--device", default=None, help="Embedding device, for example cuda:0.")
    parser.add_argument("--use-fp16", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    config = EmbeddingConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_fp16=args.use_fp16,
        return_sparse=False,
        devices=args.device,
    )
    result = query_chroma(
        persist_dir=resolve_project_path(args.persist_dir),
        collection_name=args.collection,
        query=args.query,
        config=config,
        top_k=args.top_k,
    )
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    documents = result.get("documents", [[]])[0]
    print(f"Query: {args.query}")
    print(f"TopK: {len(ids)}")
    for rank, (chunk_id, distance, metadata, document) in enumerate(
        zip(ids, distances, metadatas, documents), start=1
    ):
        source = metadata.get("source") or metadata.get("source_path")
        section = metadata.get("section", "")
        preview = " ".join(document.split())[:220]
        print(f"\n#{rank} distance={distance:.4f}")
        print(f"id={chunk_id}")
        print(f"source={source}")
        print(f"section={section}")
        print(f"preview={preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
