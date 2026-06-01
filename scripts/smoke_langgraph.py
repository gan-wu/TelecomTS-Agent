import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.graph.telecom_graph import TelecomLangGraphWorkflow
from src.models.model_router import ModelRouterClient


def compact_state(state: dict) -> dict:
    return {
        "case_id": state.get("case_id"),
        "question": state.get("question"),
        "ground_truth": state.get("ground_truth"),
        "tool_answer_used": state.get("tool_answer_used"),
        "final_answer": state.get("final_answer"),
        "critic_triggered": state.get("critic_triggered"),
        "tool_calls": state.get("tool_calls"),
        "model_calls": state.get("model_calls"),
        "trace": state.get("trace"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test Telecom LangGraph workflow.")
    parser.add_argument("--benchmark", default=os.path.join(PROJECT_ROOT, "data", "benchmark.json"))
    parser.add_argument("--tool-case-id", default="ts_0_11164_16")
    parser.add_argument("--model-case-id", default="ts_0_11164_16")
    parser.add_argument(
        "--model-question",
        default="Give a concise health summary for this 5G network session.",
    )
    parser.add_argument("--no-critic", action="store_true")
    args = parser.parse_args()

    llm_client = ModelRouterClient.from_env(mode="local")
    workflow = TelecomLangGraphWorkflow(
        llm_client=llm_client,
        benchmark_path=args.benchmark,
        use_tools=True,
        use_critic=not args.no_critic,
    )

    tests = [
        ("tool_branch", workflow.invoke(args.tool_case_id)),
        ("model_branch", workflow.invoke(args.model_case_id, args.model_question)),
    ]

    for name, state in tests:
        print("=" * 80)
        print(name)
        print(json.dumps(compact_state(state), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
