import logging
import os
import re
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from src.tools.llm_client import LLMClient


@dataclass
class ModelBackendConfig:
    provider: str
    base_url: str
    model: str
    api_key: str
    extra_body: dict[str, Any] | None = None


@dataclass
class RouteDecision:
    backends: list[str]
    reason: str
    signals: dict[str, Any]


class ModelRouterClient:
    """Signal-driven model router with cascade escalation.

    Routing is intentionally lightweight and explainable:
    - task signals decide the first backend;
    - output quality checks decide whether to escalate;
    - failures fall back to the next available backend.
    """

    REASONING_LEAK_PATTERNS = [
        r"^\s*we need to\b",
        r"^\s*we are asked\b",
        r"^\s*i need to\b",
        r"^\s*let'?s\b",
        r"^\s*the question asks\b",
        r"^\s*original question\b",
        r"^\s*proposed answer\b",
        r"^\s*analysis\s*:",
        r"<think",
        r"^\s*我们需要",
        r"^\s*需要回答",
    ]

    def __init__(
        self,
        mode: str,
        local_config: ModelBackendConfig,
        deepseek_flash_config: ModelBackendConfig | None = None,
        deepseek_pro_config: ModelBackendConfig | None = None,
    ):
        self.mode = mode
        self.backends: dict[str, LLMClient] = {
            "local": self._build_client(local_config),
        }
        if deepseek_flash_config and deepseek_flash_config.api_key:
            self.backends["deepseek_flash"] = self._build_client(deepseek_flash_config)
        if deepseek_pro_config and deepseek_pro_config.api_key:
            self.backends["deepseek_pro"] = self._build_client(deepseek_pro_config)

        self._thread_state = threading.local()
        self._last_call: dict[str, Any] | None = None
        self.local_max_concurrency = self._read_positive_int(os.getenv("LOCAL_MAX_CONCURRENCY"))
        self._local_semaphore = (
            threading.Semaphore(self.local_max_concurrency)
            if self.local_max_concurrency
            else None
        )

    @property
    def last_call(self) -> dict[str, Any] | None:
        return getattr(self._thread_state, "last_call", self._last_call)

    @last_call.setter
    def last_call(self, value: dict[str, Any] | None) -> None:
        self._last_call = value
        self._thread_state.last_call = value

    @classmethod
    def from_env(
        cls,
        mode: str = "local",
        local_base_url: str | None = None,
        local_model: str | None = None,
        local_api_key: str | None = None,
        deepseek_base_url: str | None = None,
        deepseek_api_key: str | None = None,
        deepseek_flash_model: str | None = None,
        deepseek_pro_model: str | None = None,
    ) -> "ModelRouterClient":
        local_config = ModelBackendConfig(
            provider="local",
            base_url=local_base_url or os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8080/v1"),
            model=local_model or os.getenv("LOCAL_MODEL", "qwen3.5-9b-q4"),
            api_key=local_api_key or os.getenv("LOCAL_API_KEY", "EMPTY"),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        deepseek_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        deepseek_url = deepseek_base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        flash_config = ModelBackendConfig(
            provider="deepseek_flash",
            base_url=deepseek_url,
            model=deepseek_flash_model or os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
            api_key=deepseek_key,
        )
        pro_config = ModelBackendConfig(
            provider="deepseek_pro",
            base_url=deepseek_url,
            model=deepseek_pro_model or os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-v4-pro"),
            api_key=deepseek_key,
        )

        return cls(
            mode=mode,
            local_config=local_config,
            deepseek_flash_config=flash_config,
            deepseek_pro_config=pro_config,
        )

    @staticmethod
    def _build_client(config: ModelBackendConfig) -> LLMClient:
        return LLMClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model_name=config.model,
            provider=config.provider,
            default_extra_body=config.extra_body,
        )

    def query(
        self,
        messages,
        temperature=0.0,
        max_tokens=256,
        max_retries=5,
        task: str | None = None,
        route_signals: dict[str, Any] | None = None,
    ):
        decision = self.plan_route(
            messages=messages,
            task=task,
            max_tokens=max_tokens,
            route_signals=route_signals,
        )
        backend_order = decision.backends
        last_error = None
        attempts: list[dict[str, Any]] = []
        best_rejected_answer: str | None = None

        for backend_name in backend_order:
            client = self.backends.get(backend_name)
            if not client:
                continue

            start = time.perf_counter()
            try:
                request_max_tokens = self._backend_max_tokens(backend_name, task, max_tokens)
                guard = self._local_semaphore if backend_name == "local" else None
                with guard if guard is not None else nullcontext():
                    answer = client.query(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=request_max_tokens,
                        max_retries=max_retries,
                        task=task,
                    )
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                quality_issue = self._quality_issue(answer, task, max_tokens, decision.signals)
                attempt = {
                    "backend": backend_name,
                    "provider": client.provider,
                    "model": client.model_name,
                    "latency_ms": latency_ms,
                    "request_max_tokens": request_max_tokens,
                    "quality_issue": quality_issue,
                    "accepted": quality_issue is None,
                }
                if backend_name == "local" and self.local_max_concurrency:
                    attempt["local_max_concurrency"] = self.local_max_concurrency
                usage = getattr(client, "last_usage", None)
                if usage:
                    attempt["usage"] = usage
                attempts.append(attempt)
                self.last_call = {
                    "mode": self.mode,
                    "task": task,
                    "provider": client.provider,
                    "model": client.model_name,
                    "latency_ms": latency_ms,
                    "fallback_used": backend_name != backend_order[0],
                    "route_reason": decision.reason,
                    "route_backends": backend_order,
                    "route_signals": decision.signals,
                    "quality_issue": quality_issue,
                    "attempts": list(attempts),
                }
                if usage:
                    self.last_call["usage"] = usage

                if answer and answer.strip() and quality_issue is None:
                    return answer
                if answer and answer.strip():
                    best_rejected_answer = answer
                    logging.warning(
                        "Model output rejected by router quality gate: backend=%s task=%s issue=%s",
                        backend_name,
                        task,
                        quality_issue,
                    )
            except Exception as exc:
                last_error = exc
                attempts.append(
                    {
                        "backend": backend_name,
                        "error": str(exc),
                        "accepted": False,
                    }
                )
                logging.warning("Model backend failed: %s task=%s error=%s", backend_name, task, exc)

        if best_rejected_answer:
            if self.last_call:
                self.last_call = {
                    **self.last_call,
                    "accepted": False,
                    "returned_rejected_answer": True,
                    "attempts": list(attempts),
                }
            return best_rejected_answer
        if last_error:
            raise last_error
        return None

    def plan_route(
        self,
        messages=None,
        task: str | None = None,
        max_tokens: int = 256,
        route_signals: dict[str, Any] | None = None,
    ) -> RouteDecision:
        signals = self._build_signals(messages, task, max_tokens, route_signals)
        backends, reason = self._select_backends(task, signals)
        available = [backend for backend in backends if backend in self.backends]
        return RouteDecision(
            backends=available,
            reason=reason,
            signals=signals,
        )

    def _select_backends(self, task: str | None, signals: dict[str, Any] | None = None) -> tuple[list[str], str]:
        signals = signals or {}
        if self.mode == "local":
            return ["local"], "mode=local"
        if self.mode == "deepseek":
            if task == "critic":
                return ["deepseek_pro"], "critic task uses pro only"
            if signals.get("has_rag") and signals.get("low_rag_confidence"):
                return ["deepseek_flash", "deepseek_pro"], "low-confidence RAG uses flash first with pro escalation"
            return ["deepseek_flash", "deepseek_pro"], "deepseek mode uses flash first with pro fallback"
        if self.mode == "hybrid":
            if task == "critic":
                return ["deepseek_pro"], "critic task uses pro only"
            if signals.get("has_rag"):
                if signals.get("low_rag_confidence"):
                    return ["deepseek_flash", "deepseek_pro"], "low-confidence RAG uses flash first with pro escalation"
                return ["deepseek_flash", "deepseek_pro"], "high-confidence RAG uses flash first with pro escalation"
            if task == "analyst":
                if signals.get("long_context"):
                    return ["deepseek_flash", "local", "deepseek_pro"], "long analysis context uses flash first"
                return ["local", "deepseek_flash", "deepseek_pro"], "simple analysis uses local first"
            if task in {"solver", "diagnosis", "rag"}:
                if signals.get("long_context") or signals.get("format_sensitive"):
                    return ["deepseek_flash", "deepseek_pro", "local"], "complex solver task uses flash first"
                return ["local", "deepseek_flash", "deepseek_pro"], "simple solver task uses local first"
            return ["local", "deepseek_flash", "deepseek_pro"], "default hybrid route"
        raise ValueError(f"Unsupported backend mode: {self.mode}")

    def _build_signals(
        self,
        messages,
        task: str | None,
        max_tokens: int,
        route_signals: dict[str, Any] | None,
    ) -> dict[str, Any]:
        text_parts: list[str] = []
        if messages:
            for message in messages:
                content = message.get("content", "") if isinstance(message, dict) else ""
                if isinstance(content, str):
                    text_parts.append(content)
        joined = "\n".join(text_parts)
        prompt_chars = len(joined)
        signals: dict[str, Any] = {
            "task": task,
            "prompt_chars": prompt_chars,
            "max_tokens": max_tokens,
            "long_context": prompt_chars >= 8000,
            "short_context": prompt_chars <= 3000,
        }
        if route_signals:
            signals.update(route_signals)

        has_rag = bool(
            signals.get("has_rag")
            or "retrieved telecom-domain evidence" in joined.lower()
            or "retrieved telecom knowledge" in joined.lower()
            or "ragtool" in joined.lower()
        )
        signals["has_rag"] = has_rag

        rag_top_score = self._safe_float(signals.get("rag_top_score"))
        rag_bucket = str(signals.get("rag_confidence_bucket") or "").lower()
        if has_rag and rag_top_score is not None:
            signals["rag_top_score"] = round(rag_top_score, 4)
            signals["low_rag_confidence"] = rag_top_score < 0.75
        else:
            signals["low_rag_confidence"] = has_rag and rag_bucket == "low"

        signals["format_sensitive"] = bool(
            signals.get("format_sensitive")
            or task == "critic"
            or any(marker in joined.lower() for marker in ["required format", "output only", "strictly"])
        )
        return self._compact_signals(signals)

    def _quality_issue(
        self,
        answer: str | None,
        task: str | None,
        max_tokens: int,
        signals: dict[str, Any],
    ) -> str | None:
        if not answer or not answer.strip():
            return "empty_answer"

        text = answer.strip()
        lower = text.lower()
        if task in {"solver", "critic", "rag", "diagnosis"}:
            for pattern in self.REASONING_LEAK_PATTERNS:
                if re.search(pattern, lower):
                    return "reasoning_leak"

        if task == "critic":
            if len(text) > 280:
                return "critic_answer_too_long"
            if any(marker in lower for marker in ["original question", "proposed answer", "required format"]):
                return "critic_meta_answer"

        if task in {"solver", "rag", "diagnosis"}:
            if signals.get("format_sensitive") and len(text) > 500:
                return "format_sensitive_answer_too_long"
            if max_tokens <= 128 and len(text) > 900:
                return "solver_answer_too_long"

        return None

    @staticmethod
    def _backend_max_tokens(backend_name: str, task: str | None, max_tokens: int) -> int:
        """Give reasoning backends enough room for hidden reasoning plus final content."""
        if backend_name.startswith("deepseek"):
            if task == "critic":
                return max(max_tokens, 256)
            if task in {"solver", "rag", "diagnosis"}:
                return max(max_tokens, 512)
            if task == "analyst":
                return max(max_tokens, 384)
        return max_tokens

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _read_positive_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _compact_signals(signals: dict[str, Any]) -> dict[str, Any]:
        keep = {
            "task",
            "prompt_chars",
            "max_tokens",
            "long_context",
            "short_context",
            "has_rag",
            "rag_top_score",
            "rag_confidence_bucket",
            "rag_selected_top_k",
            "low_rag_confidence",
            "format_sensitive",
            "knowledge_question",
            "item_type",
            "case_id",
            "budget_max_tokens",
            "budget_reason",
            "estimated_prompt_tokens",
        }
        return {key: value for key, value in signals.items() if key in keep}
