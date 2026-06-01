from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.telecom_graph import TelecomLangGraphWorkflow  # noqa: E402
from src.models.model_router import ModelRouterClient  # noqa: E402
from src.tools.telecom_knowledge_tool import KnowledgeSearchConfig  # noqa: E402


def compact_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": state.get("case_id"),
        "question": state.get("question"),
        "knowledge_question": state.get("knowledge_question"),
        "tool_answer_used": state.get("tool_answer_used"),
        "final_answer": state.get("final_answer"),
        "critic_triggered": state.get("critic_triggered"),
        "tool_calls": state.get("tool_calls"),
        "rag_sources": [
            {
                "source": hit.get("source"),
                "section": hit.get("section"),
                "rerank_score": hit.get("rerank_score"),
            }
            for hit in (state.get("rag_hits") or [])
        ],
        "model_calls": state.get("model_calls"),
        "trace": state.get("trace"),
        "error": state.get("error"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test LangGraph with a real local/remote LLM backend.")
    parser.add_argument("--benchmark", default=str(PROJECT_ROOT / "data" / "benchmark.json"))
    parser.add_argument("--case-id", default="ts_0_11164_16")
    parser.add_argument("--backend", default="local", choices=["local", "deepseek", "hybrid"])
    parser.add_argument("--local-base-url", default=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--local-model", default=os.getenv("LOCAL_MODEL", "qwen3.5-9b-q4"))
    parser.add_argument("--local-api-key", default=os.getenv("LOCAL_API_KEY", "EMPTY"))
    parser.add_argument("--deepseek-base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--deepseek-api-key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--deepseek-flash-model", default=os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--deepseek-pro-model", default=os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-v4-pro"))
    parser.add_argument(
        "--knowledge-question",
        default="What does high UL_BLER mean for 5G uplink troubleshooting?",
    )
    parser.add_argument("--numeric-question", default="What is the mean of UL_BLER?")
    parser.add_argument("--embedding-device", default=os.getenv("RAG_EMBEDDING_DEVICE", "cuda:0"))
    parser.add_argument("--reranker-device", default=os.getenv("RAG_RERANKER_DEVICE", "cuda:0"))
    parser.add_argument("--reranker-batch-size", type=int, default=int(os.getenv("RAG_RERANKER_BATCH_SIZE", "2")))
    parser.add_argument("--no-critic", action="store_true")
    parser.add_argument("--only", choices=["knowledge", "numeric", "both"], default="both")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    llm_client = ModelRouterClient.from_env(
        mode=args.backend,
        local_base_url=args.local_base_url,
        local_model=args.local_model,
        local_api_key=args.local_api_key,
        deepseek_base_url=args.deepseek_base_url,
        deepseek_api_key=args.deepseek_api_key,
        deepseek_flash_model=args.deepseek_flash_model,
        deepseek_pro_model=args.deepseek_pro_model,
    )
    rag_config = replace(
        KnowledgeSearchConfig.from_env(),
        embedding_device=args.embedding_device,
        reranker_device=args.reranker_device,
        reranker_batch_size=args.reranker_batch_size,
    )
    workflow = TelecomLangGraphWorkflow(
        llm_client=llm_client,
        benchmark_path=args.benchmark,
        use_tools=True,
        use_rag=True,
        use_critic=not args.no_critic,
        rag_config=rag_config,
    )

    tests: list[tuple[str, str]] = []
    if args.only in {"numeric", "both"}:
        tests.append(("numeric_tool_branch", args.numeric_question))
    if args.only in {"knowledge", "both"}:
        tests.append(("rag_knowledge_branch", args.knowledge_question))

    for name, question in tests:
        print("=" * 80)
        print(name)
        state = workflow.invoke(args.case_id, question)
        print(json.dumps(compact_state(state), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
