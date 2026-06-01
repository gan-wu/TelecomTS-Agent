"""Hybrid retrieval: Chroma dense + BM25 + RRF + optional BGE reranker."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.rag.embedding_store import (
    BgeM3Embedder,
    EmbeddingConfig,
    KnowledgeChunk,
    ensure_chroma_collection,
    load_chunks,
    quiet_model_output,
    query_chroma,
    resolve_hf_model_path,
)


LOGGER = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./:+#-]*|[\u4e00-\u9fff]+", re.UNICODE)


@dataclass(frozen=True)
class HybridRetrieverConfig:
    dense_top_k: int = 30
    bm25_top_k: int = 30
    fused_top_k: int = 30
    final_top_k: int = 5
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_max_length: int = 1024
    reranker_batch_size: int = 4
    reranker_use_fp16: bool = True
    reranker_device: Optional[str] = "cuda:0"
    reranker_enabled: bool = True
    reranker_fallback: bool = True


@dataclass
class RetrievalHit:
    chunk_id: str
    document: str
    metadata: Dict[str, Any]
    dense_rank: Optional[int] = None
    dense_distance: Optional[float] = None
    dense_score: float = 0.0
    bm25_rank: Optional[int] = None
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None


def tokenize(text: str) -> List[str]:
    tokens = []
    for token in TOKEN_RE.findall(text.lower()):
        tokens.append(token)
        if "_" in token:
            tokens.extend(part for part in token.split("_") if part)
        if "/" in token:
            tokens.extend(part for part in token.split("/") if part)
        if "-" in token:
            tokens.extend(part for part in token.split("-") if part)
    return tokens


class BM25Index:
    def __init__(self, chunks: Sequence[KnowledgeChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.term_freqs: List[Dict[str, int]] = []
        doc_freq: Dict[str, int] = {}

        for tokens in self.doc_tokens:
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs.append(tf)
            for token in tf:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        total_docs = len(self.doc_tokens)
        self.idf = {
            term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def search(self, query: str, top_k: int) -> List[RetrievalHit]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: List[tuple[int, float]] = []
        for index, tf in enumerate(self.term_freqs):
            score = 0.0
            doc_len = self.doc_lens[index] or 1
            norm = self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                score += idf * (freq * (self.k1 + 1)) / (freq + norm)
            if score > 0:
                scores.append((index, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        hits: List[RetrievalHit] = []
        for rank, (index, score) in enumerate(scores[:top_k], start=1):
            chunk = self.chunks[index]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    document=chunk.text,
                    metadata=chunk.metadata,
                    bm25_rank=rank,
                    bm25_score=score,
                )
            )
        return hits


class BgeReranker:
    def __init__(self, config: HybridRetrieverConfig):
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RuntimeError("FlagEmbedding is required for BGE reranking.") from exc

        kwargs: Dict[str, Any] = {
            "use_fp16": config.reranker_use_fp16,
            "max_length": config.reranker_max_length,
            "batch_size": config.reranker_batch_size,
        }
        if config.reranker_device:
            kwargs["devices"] = config.reranker_device
        self.model_path = resolve_hf_model_path(config.reranker_model)
        with quiet_model_output():
            self.model = FlagReranker(self.model_path, **kwargs)

    def rerank(self, query: str, hits: Sequence[RetrievalHit]) -> List[RetrievalHit]:
        if not hits:
            return []
        pairs = [(query, hit.document) for hit in hits]
        with quiet_model_output():
            scores = self.model.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        reranked: List[RetrievalHit] = []
        for hit, score in zip(hits, scores):
            hit.rerank_score = float(score)
            reranked.append(hit)
        reranked.sort(key=lambda item: item.rerank_score if item.rerank_score is not None else -1, reverse=True)
        return reranked


class HybridRetriever:
    def __init__(
        self,
        chunks_path: Path,
        chroma_dir: Path,
        collection_name: str,
        embedding_config: EmbeddingConfig,
        retriever_config: HybridRetrieverConfig,
    ):
        self.chunks = load_chunks(chunks_path)
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.embedding_config = embedding_config
        self.config = retriever_config
        self.embedder = BgeM3Embedder(embedding_config)
        self.collection = ensure_chroma_collection(
            chroma_dir,
            collection_name,
            reset=False,
            embedding_model=embedding_config.model_name,
        )
        self.bm25 = BM25Index(
            self.chunks,
            k1=retriever_config.bm25_k1,
            b=retriever_config.bm25_b,
        )
        self.reranker: Optional[BgeReranker] = None
        self.reranker_error: Optional[str] = None
        if retriever_config.reranker_enabled:
            try:
                self.reranker = BgeReranker(retriever_config)
            except Exception as exc:
                if not retriever_config.reranker_fallback:
                    raise
                self.reranker_error = _format_error(exc)
                LOGGER.warning(
                    "BGE reranker unavailable; using Chroma + BM25 + RRF fallback. error=%s",
                    self.reranker_error,
                )
        else:
            self.reranker_error = "reranker disabled by config"

    def dense_search(self, query: str) -> List[RetrievalHit]:
        result = query_chroma(
            persist_dir=self.chroma_dir,
            collection_name=self.collection_name,
            query=query,
            config=self.embedding_config,
            top_k=self.config.dense_top_k,
            embedder=self.embedder,
            collection=self.collection,
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: List[RetrievalHit] = []
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            hits.append(
                RetrievalHit(
                    chunk_id=chunk_id,
                    document=document,
                    metadata=dict(metadata),
                    dense_rank=rank,
                    dense_distance=float(distance),
                    dense_score=1.0 / (1.0 + float(distance)),
                )
            )
        return hits

    def retrieve(self, query: str) -> Dict[str, Any]:
        dense_hits = self.dense_search(query)
        bm25_hits = self.bm25.search(query, self.config.bm25_top_k)
        fused_hits = rrf_fuse(
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
            rrf_k=self.config.rrf_k,
        )[: self.config.fused_top_k]
        ranked_hits, ranking_mode = self._rank(query, fused_hits)
        final_hits = ranked_hits[: self.config.final_top_k]
        return {
            "query": query,
            "dense_count": len(dense_hits),
            "bm25_count": len(bm25_hits),
            "fused_count": len(fused_hits),
            "final_count": len(final_hits),
            "ranking_mode": ranking_mode,
            "reranker_available": self.reranker is not None and ranking_mode == "bge_reranker",
            "reranker_error": self.reranker_error,
            "hits": [hit_to_dict(hit) for hit in final_hits],
        }

    def _rank(self, query: str, fused_hits: Sequence[RetrievalHit]) -> tuple[List[RetrievalHit], str]:
        if self.reranker is None:
            return list(fused_hits), "rrf_fallback"

        try:
            return self.reranker.rerank(query, fused_hits), "bge_reranker"
        except Exception as exc:
            if not self.config.reranker_fallback:
                raise
            self.reranker = None
            self.reranker_error = _format_error(exc)
            LOGGER.warning(
                "BGE reranker failed during scoring; using Chroma + BM25 + RRF fallback. error=%s",
                self.reranker_error,
            )
            return list(fused_hits), "rrf_fallback"


def rrf_fuse(
    dense_hits: Sequence[RetrievalHit],
    bm25_hits: Sequence[RetrievalHit],
    rrf_k: int = 60,
) -> List[RetrievalHit]:
    fused: Dict[str, RetrievalHit] = {}

    def add(hit: RetrievalHit, source: str, rank: int) -> None:
        current = fused.get(hit.chunk_id)
        if current is None:
            current = RetrievalHit(
                chunk_id=hit.chunk_id,
                document=hit.document,
                metadata=dict(hit.metadata),
            )
            fused[hit.chunk_id] = current

        current.rrf_score += 1.0 / (rrf_k + rank)
        if source == "dense":
            current.dense_rank = rank
            current.dense_distance = hit.dense_distance
            current.dense_score = hit.dense_score
        else:
            current.bm25_rank = rank
            current.bm25_score = hit.bm25_score

    for rank, hit in enumerate(dense_hits, start=1):
        add(hit, "dense", rank)
    for rank, hit in enumerate(bm25_hits, start=1):
        add(hit, "bm25", rank)

    return sorted(fused.values(), key=lambda item: item.rrf_score, reverse=True)


def hit_to_dict(hit: RetrievalHit) -> Dict[str, Any]:
    rank_score = hit.rerank_score if hit.rerank_score is not None else hit.rrf_score
    ranker = "bge_reranker" if hit.rerank_score is not None else "rrf"
    return {
        "chunk_id": hit.chunk_id,
        "source": hit.metadata.get("source") or hit.metadata.get("source_path", ""),
        "section": hit.metadata.get("section", ""),
        "repo": hit.metadata.get("repo", ""),
        "dense_rank": hit.dense_rank,
        "dense_distance": hit.dense_distance,
        "bm25_rank": hit.bm25_rank,
        "bm25_score": hit.bm25_score,
        "rrf_score": hit.rrf_score,
        "rerank_score": hit.rerank_score,
        "rank_score": rank_score,
        "ranker": ranker,
        "preview": " ".join(hit.document.split())[:360],
        "document": hit.document,
        "metadata": hit.metadata,
    }


def _format_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip()
    if len(message) > 500:
        message = message[:500].rstrip() + "..."
    return f"{type(exc).__name__}: {message}"


def save_retrieval_result(result: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
