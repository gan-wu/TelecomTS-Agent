from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.embedding_store import EmbeddingConfig, build_chroma_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a BGE-M3 dense Chroma index.")
    parser.add_argument("--chunks", default="knowledge_base/chunks/knowledge_chunks.jsonl")
    parser.add_argument("--persist-dir", default="knowledge_base/chroma")
    parser.add_argument("--collection", default="telecom_knowledge_bge_m3")
    parser.add_argument("--manifest", default="knowledge_base/chroma/embedding_manifest.json")
    parser.add_argument("--sparse-sidecar", default="knowledge_base/chroma/bge_m3_sparse_vectors.jsonl")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--device", default=None, help="Embedding device, for example cuda:0.")
    parser.add_argument("--limit", type=int, default=None, help="Index only the first N chunks.")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the collection.")
    parser.add_argument("--use-fp16", action="store_true", help="Use fp16. Keep disabled on CPU.")
    parser.add_argument(
        "--no-sparse-sidecar",
        action="store_true",
        help="Only build dense Chroma vectors; skip BGE-M3 lexical weights sidecar.",
    )
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
        return_sparse=not args.no_sparse_sidecar,
        devices=args.device,
    )
    manifest = build_chroma_index(
        chunks_path=resolve_project_path(args.chunks),
        persist_dir=resolve_project_path(args.persist_dir),
        collection_name=args.collection,
        manifest_path=resolve_project_path(args.manifest),
        sparse_sidecar_path=resolve_project_path(args.sparse_sidecar),
        config=config,
        limit=args.limit,
        reset=args.reset,
    )
    print(f"Indexed chunks: {manifest['chunk_count']}")
    print(f"Collection count: {manifest['collection_count']}")
    print(f"Dense dim: {manifest['dense_dim']}")
    print(f"Chroma dir: {Path(manifest['persist_dir']).relative_to(PROJECT_ROOT)}")
    print(f"Manifest: {Path(args.manifest)}")
    if manifest["sparse_sidecar"]:
        print(f"Sparse sidecar records: {manifest['sparse_records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
