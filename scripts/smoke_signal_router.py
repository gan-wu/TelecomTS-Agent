from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_router import ModelBackendConfig, ModelRouterClient  # noqa: E402


def build_router(mode: str = "hybrid") -> ModelRouterClient:
    return ModelRouterClient(
        mode=mode,
        local_config=ModelBackendConfig(
            provider="local",
            base_url="http://127.0.0.1:8080/v1",
            model="qwen3.5-9b-q4",
            api_key="EMPTY",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
        deepseek_flash_config=ModelBackendConfig(
            provider="deepseek_flash",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            api_key="DUMMY",
        ),
        deepseek_pro_config=ModelBackendConfig(
            provider="deepseek_pro",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-pro",
            api_key="DUMMY",
        ),
    )


def main() -> int:
    router = build_router("hybrid")
    cases = [
        {
            "name": "simple_analyst",
            "task": "analyst",
            "messages": [{"role": "user", "content": "Summarize this short KPI report."}],
            "signals": {"item_type": "network"},
        },
        {
            "name": "rag_high_confidence",
            "task": "solver",
            "messages": [{"role": "user", "content": "Retrieved telecom-domain evidence.\nWhat does high UL_BLER mean?"}],
            "signals": {"knowledge_question": True, "has_rag": True, "rag_top_score": 0.93},
        },
        {
            "name": "rag_low_confidence",
            "task": "solver",
            "messages": [{"role": "user", "content": "Retrieved telecom-domain evidence.\nExplain a vague KPI issue."}],
            "signals": {"knowledge_question": True, "has_rag": True, "rag_top_score": 0.62},
        },
        {
            "name": "knowledge_question_without_rag",
            "task": "solver",
            "messages": [{"role": "user", "content": "What does high UL_BLER mean?"}],
            "signals": {"knowledge_question": True},
        },
        {
            "name": "critic_repair",
            "task": "critic",
            "messages": [{"role": "user", "content": "ORIGINAL QUESTION: ...\nPROPOSED ANSWER: ..."}],
            "signals": {"format_sensitive": True},
        },
    ]

    report = []
    for case in cases:
        decision = router.plan_route(
            messages=case["messages"],
            task=case["task"],
            max_tokens=128,
            route_signals=case["signals"],
        )
        report.append(
            {
                "name": case["name"],
                "task": case["task"],
                "backends": decision.backends,
                "reason": decision.reason,
                "signals": decision.signals,
            }
        )

    report.append(
        {
            "name": "quality_gate_reasoning_leak",
            "issue": router._quality_issue(
                "We need to answer the question based on the evidence...",
                task="solver",
                max_tokens=128,
                signals={},
            ),
        }
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
