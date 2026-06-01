"""BGE-M3 embedding and Chroma vector store utilities."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
try:
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except Exception:
    pass


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 8
    max_length: int = 1536
    use_fp16: bool = False
    return_sparse: bool = True
    devices: Optional[str] = None


@dataclass
class KnowledgeChunk:
    chunk_id: str
    text: str
    content: str
    context_prefix: str
    token_count: int
    content_token_count: int
    metadata: Dict[str, Any]


class BgeM3Embedder:
    """Thin wrapper around FlagEmbedding's BGEM3FlagModel."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "FlagEmbedding is required for BGE-M3. Install project requirements first."
            ) from exc

        init_kwargs = {
            "use_fp16": config.use_fp16,
            "query_max_length": config.max_length,
            "passage_max_length": config.max_length,
        }
        if config.devices:
            init_kwargs["devices"] = config.devices
        self.model_path = resolve_hf_model_path(config.model_name)
        with quiet_model_output():
            self.model = BGEM3FlagModel(self.model_path, **init_kwargs)

    def encode(
        self,
        texts: Sequence[str],
        *,
        return_sparse: Optional[bool] = None,
        return_colbert_vecs: bool = False,
    ) -> Dict[str, Any]:
        use_sparse = self.config.return_sparse if return_sparse is None else return_sparse
        kwargs = {
            "batch_size": self.config.batch_size,
            "max_length": self.config.max_length,
            "return_dense": True,
            "return_sparse": use_sparse,
            "return_colbert_vecs": return_colbert_vecs,
        }
        try:
            with quiet_model_output():
                return self.model.encode(list(texts), **kwargs)
        except TypeError:
            kwargs.pop("max_length", None)
            with quiet_model_output():
                return self.model.encode(list(texts), **kwargs)


def load_chunks(path: Path, limit: Optional[int] = None) -> List[KnowledgeChunk]:
    chunks: List[KnowledgeChunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            chunks.append(
                KnowledgeChunk(
                    chunk_id=str(row["chunk_id"]),
                    text=str(row["text"]),
                    content=str(row.get("content", "")),
                    context_prefix=str(row.get("context_prefix", "")),
                    token_count=int(row.get("token_count", 0)),
                    content_token_count=int(row.get("content_token_count", 0)),
                    metadata=dict(row.get("metadata", {})),
                )
            )
            if limit and len(chunks) >= limit:
                break
    return chunks


def resolve_hf_model_path(model_name_or_path: str) -> str:
    """Use a local HF snapshot when available to avoid network checks during demos."""
    path = Path(model_name_or_path)
    if path.exists():
        return str(path)
    if "/" not in model_name_or_path:
        return model_name_or_path

    escaped_name = "models--" + model_name_or_path.replace("/", "--")
    cache_roots = []
    hf_hub_cache = os.getenv("HF_HUB_CACHE")
    if hf_hub_cache:
        cache_roots.append(Path(hf_hub_cache))
    transformers_cache = os.getenv("TRANSFORMERS_CACHE")
    if transformers_cache:
        cache_roots.append(Path(transformers_cache))
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        cache_roots.append(Path(hf_home) / "hub")
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    seen: set[Path] = set()
    for cache_root in cache_roots:
        cache_root = cache_root.expanduser()
        if cache_root in seen:
            continue
        seen.add(cache_root)
        snapshots = cache_root / escaped_name / "snapshots"
        if not snapshots.exists():
            continue
        candidates = [item for item in snapshots.iterdir() if item.is_dir()]
        candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        for candidate in candidates:
            if (candidate / "config.json").exists() and (
                (candidate / "tokenizer.json").exists()
                or (candidate / "tokenizer_config.json").exists()
                or (candidate / "sentencepiece.bpe.model").exists()
            ):
                return str(candidate)
    return model_name_or_path


@contextmanager
def quiet_model_output():
    """Suppress verbose third-party progress bars unless RAG_VERBOSE=1."""
    if os.getenv("RAG_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}:
        with nullcontext():
            yield
        return

    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def to_plain_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list):
        return [to_plain_list(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain_list(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain_list(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def sanitize_metadata(metadata: Dict[str, Any], chunk: KnowledgeChunk) -> Dict[str, Any]:
    merged = dict(metadata)
    merged["chunk_id"] = chunk.chunk_id
    merged["token_count"] = chunk.token_count
    merged["content_token_count"] = chunk.content_token_count
    merged["has_context_prefix"] = bool(chunk.context_prefix)

    clean: Dict[str, Any] = {}
    for key, value in merged.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = json.dumps(to_plain_list(value), ensure_ascii=False)
    return clean


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def ensure_chroma_collection(
    persist_dir: Path,
    collection_name: str,
    reset: bool = False,
    embedding_model: str = "BAAI/bge-m3",
):
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is required. Install project requirements first.") from exc

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
    )


def write_sparse_sidecar(
    path: Path,
    chunks: Sequence[KnowledgeChunk],
    lexical_weights: Sequence[Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for chunk, weights in zip(chunks, lexical_weights):
            record = {
                "chunk_id": chunk.chunk_id,
                "source": chunk.metadata.get("source", ""),
                "repo": chunk.metadata.get("repo", ""),
                "source_path": chunk.metadata.get("source_path", ""),
                "lexical_weights": to_plain_list(weights),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_chroma_index(
    chunks_path: Path,
    persist_dir: Path,
    collection_name: str,
    manifest_path: Path,
    sparse_sidecar_path: Path,
    config: EmbeddingConfig,
    *,
    limit: Optional[int] = None,
    reset: bool = False,
) -> Dict[str, Any]:
    chunks = load_chunks(chunks_path, limit=limit)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    embedder = BgeM3Embedder(config)
    collection = ensure_chroma_collection(
        persist_dir,
        collection_name,
        reset=reset,
        embedding_model=config.model_name,
    )

    total = len(chunks)
    dense_dim: Optional[int] = None
    sparse_written = 0
    all_sparse_records: List[Any] = []

    for chunk_batch in batched(chunks, config.batch_size):
        texts = [chunk.text for chunk in chunk_batch]
        output = embedder.encode(texts, return_sparse=config.return_sparse)
        dense_vecs = to_plain_list(output["dense_vecs"])
        if dense_vecs and dense_dim is None:
            dense_dim = len(dense_vecs[0])

        ids = [chunk.chunk_id for chunk in chunk_batch]
        metadatas = [sanitize_metadata(chunk.metadata, chunk) for chunk in chunk_batch]
        documents = [chunk.text for chunk in chunk_batch]
        collection.upsert(
            ids=ids,
            embeddings=dense_vecs,
            documents=documents,
            metadatas=metadatas,
        )

        if config.return_sparse:
            sparse = output.get("lexical_weights") or []
            all_sparse_records.extend(sparse)
            sparse_written += len(sparse)

    if config.return_sparse:
        write_sparse_sidecar(sparse_sidecar_path, chunks, all_sparse_records)

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "chunks_path": str(chunks_path),
        "persist_dir": str(persist_dir),
        "collection_name": collection_name,
        "embedding_model": config.model_name,
        "dense_dim": dense_dim,
        "chunk_count": total,
        "collection_count": collection.count(),
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "use_fp16": config.use_fp16,
        "devices": config.devices or "auto",
        "dense_store": "Chroma",
        "sparse_sidecar": str(sparse_sidecar_path) if config.return_sparse else "",
        "sparse_records": sparse_written,
        "multi_vector": "not_materialized_by_default",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def query_chroma(
    persist_dir: Path,
    collection_name: str,
    query: str,
    config: EmbeddingConfig,
    *,
    top_k: int = 5,
    embedder: BgeM3Embedder | None = None,
    collection: Any | None = None,
) -> Dict[str, Any]:
    embedder = embedder or BgeM3Embedder(config)
    if collection is None:
        collection = ensure_chroma_collection(
            persist_dir,
            collection_name,
            reset=False,
            embedding_model=config.model_name,
        )
    output = embedder.encode([query], return_sparse=False)
    query_embedding = to_plain_list(output["dense_vecs"])[0]
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
