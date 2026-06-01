from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunking import (  # noqa: E402
    ChunkConfig,
    build_chunks,
    summarize_chunks,
    write_chunk_manifest,
    write_chunks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build structure-aware, context-enhanced RAG chunks for TelecomTS_QA."
    )
    parser.add_argument(
        "--source-dir",
        default="knowledge_base/source_docs",
        help="Directory containing curated source documents.",
    )
    parser.add_argument(
        "--manifest",
        default="knowledge_base/manifests/source_manifest.json",
        help="Source manifest with repo/commit/path metadata.",
    )
    parser.add_argument(
        "--output",
        default="knowledge_base/chunks/knowledge_chunks.jsonl",
        help="Output JSONL file for generated chunks.",
    )
    parser.add_argument(
        "--chunk-manifest",
        default="knowledge_base/chunks/chunk_manifest.json",
        help="Output JSON summary for generated chunks.",
    )
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--target-tokens", type=int, default=700)
    parser.add_argument("--min-tokens", type=int, default=160)
    parser.add_argument("--overlap-tokens", type=int, default=120)
    parser.add_argument("--context-prefix-tokens", type=int, default=100)
    parser.add_argument(
        "--semantic-break-threshold",
        type=float,
        default=0.10,
        help="Lower lexical similarity can trigger a semantic boundary once target size is reached.",
    )
    parser.add_argument(
        "--extra-file",
        action="append",
        default=[],
        help="Extra source file, for example the dataset paper PDF. Can be repeated.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF parsing even if extra PDF files are provided.",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    config = ChunkConfig(
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        context_prefix_tokens=args.context_prefix_tokens,
        semantic_break_threshold=args.semantic_break_threshold,
    )

    source_dir = resolve_project_path(args.source_dir)
    manifest_path = resolve_project_path(args.manifest)
    output_path = resolve_project_path(args.output)
    chunk_manifest_path = resolve_project_path(args.chunk_manifest)
    extra_files = [resolve_project_path(item) for item in args.extra_file]

    chunks = build_chunks(
        source_dir=source_dir,
        manifest_path=manifest_path,
        project_root=PROJECT_ROOT,
        config=config,
        extra_files=extra_files,
        include_pdf=not args.no_pdf,
    )
    write_chunks(chunks, output_path)
    write_chunk_manifest(chunks, chunk_manifest_path, config)

    summary = summarize_chunks(chunks)
    print(f"Built {summary['chunk_count']} chunks")
    print(f"Output: {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Manifest: {chunk_manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"By group: {summary['by_group']}")
    print(f"Content tokens: {summary['content_tokens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

