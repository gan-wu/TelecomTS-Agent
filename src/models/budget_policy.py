from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenBudget:
    task: str
    max_tokens: int
    reason: str
    signals: dict[str, Any]


class TokenBudgetPolicy:
    """Explainable token-budget policy for Agent calls.

    The policy is deliberately heuristic and transparent. It uses task type,
    RAG state, retrieval confidence, context size, and formatting risk to pick a
    conservative output budget instead of giving every call the same max_tokens.
    """

    def budget(
        self,
        task: str,
        prompt_chars: int = 0,
        route_signals: dict[str, Any] | None = None,
    ) -> TokenBudget:
        signals = dict(route_signals or {})
        signals["prompt_chars"] = prompt_chars
        has_rag = bool(signals.get("has_rag"))
        rag_top_score = _safe_float(signals.get("rag_top_score"))
        rag_bucket = str(signals.get("rag_confidence_bucket") or "").lower()
        low_rag_confidence = has_rag and (
            rag_bucket == "low" or (rag_top_score is not None and rag_top_score < 0.75)
        )
        format_sensitive = bool(signals.get("format_sensitive"))
        item_type = signals.get("item_type")
        long_context = prompt_chars >= 8000

        if task == "critic":
            return TokenBudget(
                task=task,
                max_tokens=64 if format_sensitive or item_type == "timeseries" else 96,
                reason="critic repair uses a compact output budget",
                signals=_compact(
                    signals,
                    has_rag=has_rag,
                    rag_top_score=rag_top_score,
                    low_rag_confidence=low_rag_confidence,
                    format_sensitive=format_sensitive,
                ),
            )

        if task == "analyst":
            if long_context:
                max_tokens = 384
                reason = "long context analyst summary"
            else:
                max_tokens = 256
                reason = "standard analyst summary"
            return TokenBudget(
                task=task,
                max_tokens=max_tokens,
                reason=reason,
                signals=_compact(signals, has_rag=has_rag, format_sensitive=format_sensitive),
            )

        if task in {"solver", "rag", "diagnosis"}:
            if has_rag and low_rag_confidence:
                max_tokens = 384
                reason = "low-confidence RAG answer gets larger budget"
            elif has_rag:
                max_tokens = 256
                reason = "RAG answer gets concise knowledge budget"
            elif format_sensitive or item_type == "timeseries":
                max_tokens = 96
                reason = "structured solver answer uses compact format budget"
            else:
                max_tokens = 128
                reason = "standard solver answer"
            return TokenBudget(
                task=task,
                max_tokens=max_tokens,
                reason=reason,
                signals=_compact(
                    signals,
                    has_rag=has_rag,
                    rag_top_score=rag_top_score,
                    low_rag_confidence=low_rag_confidence,
                    format_sensitive=format_sensitive,
                ),
            )

        return TokenBudget(
            task=task,
            max_tokens=128,
            reason="default compact budget",
            signals=_compact(signals, has_rag=has_rag, format_sensitive=format_sensitive),
        )


def estimate_text_tokens(text: str) -> int:
    """Cheap token estimate for budget metadata.

    This avoids adding tokenizer dependencies in the hot path. It is not used
    for billing, only for local budget diagnostics.
    """

    clean = text or ""
    if not clean:
        return 0
    return max(1, int(len(clean) / 4))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact(signals: dict[str, Any], **extra: Any) -> dict[str, Any]:
    keep = {
        "case_id",
        "item_type",
        "knowledge_question",
        "prompt_chars",
        "has_rag",
        "rag_top_score",
        "rag_confidence_bucket",
        "rag_selected_top_k",
        "low_rag_confidence",
        "format_sensitive",
    }
    compact = {key: value for key, value in signals.items() if key in keep}
    compact.update({key: value for key, value in extra.items() if value is not None})
    return compact
