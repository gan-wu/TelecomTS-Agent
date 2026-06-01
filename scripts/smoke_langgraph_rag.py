from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.telecom_graph import TelecomLangGraphWorkflow  # noqa: E402
from src.tools.telecom_knowledge_tool import KnowledgeSearchConfig  # noqa: E402


class FakeLLMClient:
    """Tiny deterministic LLM stub for graph wiring tests."""

    def __init__(self):
        self.last_call: dict[str, Any] | None = None

    def query(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 128,
        task: str | None = None,
        route_signals: dict[str, Any] | None = None,
    ) -> str:
        user_content = messages[-1]["content"] if messages else ""
        self.last_call = {
            "backend": "fake",
            "task": task or "unknown",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "input_chars": len(user_content),
            "route_signals": route_signals or {},
        }
        if task == "solver":
            return (
                "High UL_BLER means the uplink has a high block error rate, "
                "usually pointing to poor radio quality, interference, or resource pressure."
            )
        if task == "critic":
            return user_content.rsplit(":", 1)[-1].strip()
        return "Fake analysis summary."


def compact_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": state.get("case_id"),
        "question": state.get("question"),
        "knowledge_question": state.get("knowledge_question"),
        "final_answer": state.get("final_answer"),
        "tool_calls": state.get("tool_calls"),
        "rag_hit_count": len(state.get("rag_hits") or []),
        "trace": state.get("trace"),
        "error": state.get("error"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test LangGraph RAG branch without a live LLM.")
    parser.add_argument("--benchmark", default=str(PROJECT_ROOT / "data" / "benchmark.json"))
    parser.add_argument("--case-id", default="ts_0_11164_16")
    parser.add_argument(
        "--question",
        default="What does high UL_BLER mean for 5G uplink troubleshooting?",
    )
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--reranker-batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rag_config = KnowledgeSearchConfig.from_env()
    rag_config = replace(
        rag_config,
        embedding_device=args.embedding_device or rag_config.embedding_device,
        reranker_device=args.reranker_device or rag_config.reranker_device,
        reranker_batch_size=args.reranker_batch_size or rag_config.reranker_batch_size,
    )
    workflow = TelecomLangGraphWorkflow(
        llm_client=FakeLLMClient(),
        benchmark_path=args.benchmark,
        use_tools=True,
        use_rag=True,
        use_critic=True,
        rag_config=rag_config,
    )
    state = workflow.invoke(args.case_id, args.question)
    print(json.dumps(compact_state(state), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
