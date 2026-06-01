"""RAG tool for telecom domain knowledge search."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.rag.embedding_store import EmbeddingConfig
from src.rag.hybrid_retriever import HybridRetriever, HybridRetrieverConfig
from src.tools.telecom_tools import ToolCallRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class KnowledgeSearchConfig:
    chunks_path: Path = PROJECT_ROOT / "knowledge_base" / "chunks" / "knowledge_chunks.jsonl"
    chroma_dir: Path = PROJECT_ROOT / "knowledge_base" / "chroma"
    collection_name: str = "telecom_knowledge_bge_m3"
    embedding_model: str = "BAAI/bge-m3"
    embedding_max_length: int = 1536
    embedding_batch_size: int = 1
    embedding_device: str | None = "cuda:0"
    embedding_use_fp16: bool = True
    dense_top_k: int = 30
    bm25_top_k: int = 30
    fused_top_k: int = 30
    default_top_k: int = 5
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_max_length: int = 1024
    reranker_batch_size: int = 4
    reranker_device: str | None = "cuda:0"
    reranker_use_fp16: bool = True
    reranker_enabled: bool = True
    reranker_fallback: bool = True
    evidence_chars_per_hit: int = 900
    budget_aware: bool = True
    high_confidence_threshold: float = 0.90
    low_confidence_threshold: float = 0.75
    high_confidence_top_k: int = 2
    medium_confidence_top_k: int = 3
    high_confidence_evidence_chars: int = 520
    medium_confidence_evidence_chars: int = 700
    low_confidence_evidence_chars: int = 900

    @classmethod
    def from_env(cls) -> "KnowledgeSearchConfig":
        return cls(
            chunks_path=Path(os.getenv("RAG_CHUNKS_PATH", str(cls.chunks_path))),
            chroma_dir=Path(os.getenv("RAG_CHROMA_DIR", str(cls.chroma_dir))),
            collection_name=os.getenv("RAG_COLLECTION", cls.collection_name),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", cls.embedding_model),
            embedding_max_length=int(os.getenv("RAG_EMBEDDING_MAX_LENGTH", cls.embedding_max_length)),
            embedding_batch_size=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", cls.embedding_batch_size)),
            embedding_device=_none_if_auto(os.getenv("RAG_EMBEDDING_DEVICE", cls.embedding_device or "")),
            embedding_use_fp16=_env_bool("RAG_EMBEDDING_FP16", cls.embedding_use_fp16),
            dense_top_k=int(os.getenv("RAG_DENSE_TOP_K", cls.dense_top_k)),
            bm25_top_k=int(os.getenv("RAG_BM25_TOP_K", cls.bm25_top_k)),
            fused_top_k=int(os.getenv("RAG_FUSED_TOP_K", cls.fused_top_k)),
            default_top_k=int(os.getenv("RAG_FINAL_TOP_K", cls.default_top_k)),
            reranker_model=os.getenv("RAG_RERANKER_MODEL", cls.reranker_model),
            reranker_max_length=int(os.getenv("RAG_RERANKER_MAX_LENGTH", cls.reranker_max_length)),
            reranker_batch_size=int(os.getenv("RAG_RERANKER_BATCH_SIZE", cls.reranker_batch_size)),
            reranker_device=_none_if_auto(os.getenv("RAG_RERANKER_DEVICE", cls.reranker_device or "")),
            reranker_use_fp16=_env_bool("RAG_RERANKER_FP16", cls.reranker_use_fp16),
            reranker_enabled=_env_bool("RAG_RERANKER_ENABLED", cls.reranker_enabled),
            reranker_fallback=_env_bool("RAG_RERANKER_FALLBACK", cls.reranker_fallback),
            evidence_chars_per_hit=int(os.getenv("RAG_EVIDENCE_CHARS", cls.evidence_chars_per_hit)),
            budget_aware=_env_bool("RAG_BUDGET_AWARE", cls.budget_aware),
            high_confidence_threshold=float(os.getenv("RAG_HIGH_CONFIDENCE_THRESHOLD", cls.high_confidence_threshold)),
            low_confidence_threshold=float(os.getenv("RAG_LOW_CONFIDENCE_THRESHOLD", cls.low_confidence_threshold)),
            high_confidence_top_k=int(os.getenv("RAG_HIGH_CONFIDENCE_TOP_K", cls.high_confidence_top_k)),
            medium_confidence_top_k=int(os.getenv("RAG_MEDIUM_CONFIDENCE_TOP_K", cls.medium_confidence_top_k)),
            high_confidence_evidence_chars=int(os.getenv("RAG_HIGH_CONFIDENCE_EVIDENCE_CHARS", cls.high_confidence_evidence_chars)),
            medium_confidence_evidence_chars=int(os.getenv("RAG_MEDIUM_CONFIDENCE_EVIDENCE_CHARS", cls.medium_confidence_evidence_chars)),
            low_confidence_evidence_chars=int(os.getenv("RAG_LOW_CONFIDENCE_EVIDENCE_CHARS", cls.low_confidence_evidence_chars)),
        )


class TelecomKnowledgeTool:
    """Agent-facing wrapper around dense + BM25 + RRF + rerank retrieval."""

    def __init__(self, config: KnowledgeSearchConfig | None = None):
        self.config = config or KnowledgeSearchConfig.from_env()
        self._retriever: HybridRetriever | None = None
        self._retriever_top_k: int = 0

    def search_telecom_knowledge(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """Search telecom knowledge and return compact evidence for a SolverAgent."""
        clean_query = (query or "").strip()
        if not clean_query:
            raise ValueError("query must not be empty")

        final_top_k = top_k or self.config.default_top_k
        retriever = self._get_retriever(final_top_k)
        result = retriever.retrieve(clean_query)
        hits = result.get("hits", [])[:final_top_k]
        ranking_mode = str(result.get("ranking_mode") or "")
        rag_budget = self._select_rag_budget(hits, final_top_k, ranking_mode)
        selected_hits = hits[: rag_budget["selected_top_k"]]
        compact_hits = [
            compact_hit(hit, int(rag_budget["evidence_chars_per_hit"]))
            for hit in selected_hits
        ]
        context_block = format_knowledge_context(compact_hits)

        tool_result = {
            "query": clean_query,
            "top_k": final_top_k,
            "dense_count": result.get("dense_count", 0),
            "bm25_count": result.get("bm25_count", 0),
            "fused_count": result.get("fused_count", 0),
            "final_count": len(compact_hits),
            "ranking_mode": ranking_mode,
            "reranker_available": result.get("reranker_available", False),
            "reranker_error": result.get("reranker_error"),
            "hits": compact_hits,
            "context_block": context_block,
            "rag_budget": rag_budget,
        }
        record = ToolCallRecord(
            tool="search_telecom_knowledge",
            args={"query": clean_query, "top_k": final_top_k},
            result={
                "final_count": len(compact_hits),
                "selected_top_k": rag_budget["selected_top_k"],
                "confidence_bucket": rag_budget["confidence_bucket"],
                "ranking_mode": ranking_mode,
                "reranker_available": result.get("reranker_available", False),
                "sources": [hit["source"] for hit in compact_hits],
                "sections": [hit["section"] for hit in compact_hits],
            },
        )
        tool_result["tool_call"] = record.to_dict()
        return tool_result

    def _get_retriever(self, top_k: int) -> HybridRetriever:
        if self._retriever is not None and top_k <= self._retriever_top_k:
            return self._retriever

        if not self.config.chunks_path.exists():
            raise FileNotFoundError(f"knowledge chunks not found: {self.config.chunks_path}")
        if not self.config.chroma_dir.exists():
            raise FileNotFoundError(f"Chroma index not found: {self.config.chroma_dir}")

        embedding_config = EmbeddingConfig(
            model_name=self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
            max_length=self.config.embedding_max_length,
            use_fp16=self.config.embedding_use_fp16,
            return_sparse=False,
            devices=self.config.embedding_device,
        )
        retriever_config = HybridRetrieverConfig(
            dense_top_k=self.config.dense_top_k,
            bm25_top_k=self.config.bm25_top_k,
            fused_top_k=self.config.fused_top_k,
            final_top_k=max(top_k, self.config.default_top_k),
            reranker_model=self.config.reranker_model,
            reranker_max_length=self.config.reranker_max_length,
            reranker_batch_size=self.config.reranker_batch_size,
            reranker_use_fp16=self.config.reranker_use_fp16,
            reranker_device=self.config.reranker_device,
            reranker_enabled=self.config.reranker_enabled,
            reranker_fallback=self.config.reranker_fallback,
        )
        self._retriever = HybridRetriever(
            chunks_path=self.config.chunks_path,
            chroma_dir=self.config.chroma_dir,
            collection_name=self.config.collection_name,
            embedding_config=embedding_config,
            retriever_config=retriever_config,
        )
        self._retriever_top_k = retriever_config.final_top_k
        return self._retriever

    def _select_rag_budget(
        self,
        hits: list[dict[str, Any]],
        requested_top_k: int,
        ranking_mode: str,
    ) -> dict[str, Any]:
        top_score = _safe_float(hits[0].get("rerank_score")) if hits else None
        full_context_chars = sum(len(str(hit.get("document") or "")) for hit in hits[:requested_top_k])

        if not self.config.budget_aware or top_score is None:
            if self.config.budget_aware and ranking_mode == "rrf_fallback":
                selected_top_k = min(self.config.medium_confidence_top_k, requested_top_k, len(hits))
                evidence_chars = self.config.medium_confidence_evidence_chars
                bucket = "fallback"
            else:
                selected_top_k = min(requested_top_k, len(hits))
                evidence_chars = self.config.evidence_chars_per_hit
                bucket = "fixed"
        elif top_score >= self.config.high_confidence_threshold:
            selected_top_k = min(self.config.high_confidence_top_k, requested_top_k, len(hits))
            evidence_chars = self.config.high_confidence_evidence_chars
            bucket = "high"
        elif top_score >= self.config.low_confidence_threshold:
            selected_top_k = min(self.config.medium_confidence_top_k, requested_top_k, len(hits))
            evidence_chars = self.config.medium_confidence_evidence_chars
            bucket = "medium"
        else:
            selected_top_k = min(requested_top_k, len(hits))
            evidence_chars = self.config.low_confidence_evidence_chars
            bucket = "low"

        compressed_chars = sum(
            min(len(str(hit.get("document") or "")), evidence_chars)
            for hit in hits[:selected_top_k]
        )
        return {
            "enabled": self.config.budget_aware,
            "requested_top_k": requested_top_k,
            "selected_top_k": selected_top_k,
            "evidence_chars_per_hit": evidence_chars,
            "rag_top_score": top_score,
            "confidence_bucket": bucket,
            "ranking_mode": ranking_mode,
            "full_context_chars_est": full_context_chars,
            "compressed_context_chars_est": compressed_chars,
            "saved_context_chars_est": max(0, full_context_chars - compressed_chars),
        }


def compact_hit(hit: dict[str, Any], evidence_chars: int) -> dict[str, Any]:
    document = str(hit.get("document") or "")
    evidence = " ".join(document.split())
    if len(evidence) > evidence_chars:
        evidence = evidence[:evidence_chars].rstrip() + "..."

    return {
        "chunk_id": hit.get("chunk_id", ""),
        "source": hit.get("source", ""),
        "repo": hit.get("repo", ""),
        "section": hit.get("section", ""),
        "rerank_score": hit.get("rerank_score"),
        "rank_score": hit.get("rank_score"),
        "ranker": hit.get("ranker"),
        "rrf_score": hit.get("rrf_score"),
        "dense_rank": hit.get("dense_rank"),
        "bm25_rank": hit.get("bm25_rank"),
        "evidence": evidence,
        "metadata": {
            "source_path": (hit.get("metadata") or {}).get("source_path", ""),
            "page": (hit.get("metadata") or {}).get("page", ""),
            "group": (hit.get("metadata") or {}).get("group", ""),
        },
    }


def format_knowledge_context(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "No retrieved telecom knowledge was found."

    sections = ["## Retrieved Telecom Knowledge"]
    for index, hit in enumerate(hits, start=1):
        ranker = hit.get("ranker") or "unknown"
        score = hit.get("rerank_score") if hit.get("rerank_score") is not None else hit.get("rank_score")
        score_text = f"{score:.4f}" if isinstance(score, float) else str(score)
        sections.append(
            "\n".join(
                [
                    f"[{index}] source: {hit.get('source', '')}",
                    f"section: {hit.get('section', '')}",
                    f"ranker: {ranker} | score: {score_text}",
                    f"dense_rank: {hit.get('dense_rank')} | bm25_rank: {hit.get('bm25_rank')}",
                    f"evidence: {hit.get('evidence', '')}",
                ]
            )
        )
    return "\n\n".join(sections)


def search_telecom_knowledge(query: str, top_k: int = 5) -> dict[str, Any]:
    """Convenience function for agent tool-calling demos."""
    return get_default_knowledge_tool().search_telecom_knowledge(query, top_k=top_k)


@lru_cache(maxsize=1)
def get_default_knowledge_tool() -> TelecomKnowledgeTool:
    return TelecomKnowledgeTool(KnowledgeSearchConfig.from_env())


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _none_if_auto(value: str | None) -> str | None:
    if value is None:
        return None
    clean = value.strip()
    if clean.lower() in {"", "none", "auto"}:
        return None
    return clean
