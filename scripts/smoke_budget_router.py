from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.budget_policy import TokenBudgetPolicy  # noqa: E402
from src.tools.telecom_knowledge_tool import KnowledgeSearchConfig, TelecomKnowledgeTool  # noqa: E402


def main() -> int:
    policy = TokenBudgetPolicy()
    budget_cases = [
        ("analyst_short", "analyst", 2500, {"item_type": "network"}),
        ("solver_timeseries", "solver", 6000, {"item_type": "timeseries", "format_sensitive": True}),
        ("rag_high", "solver", 5000, {"has_rag": True, "rag_top_score": 0.94}),
        ("rag_medium", "solver", 5000, {"has_rag": True, "rag_top_score": 0.82}),
        ("rag_low", "solver", 5000, {"has_rag": True, "rag_top_score": 0.61}),
        ("critic", "critic", 1000, {"item_type": "timeseries", "format_sensitive": True}),
    ]
    budgets = [
        {
            "name": name,
            **policy.budget(task=task, prompt_chars=prompt_chars, route_signals=signals).__dict__,
        }
        for name, task, prompt_chars, signals in budget_cases
    ]

    tool = TelecomKnowledgeTool(KnowledgeSearchConfig())
    fake_hits = [
        {"rerank_score": score, "document": "x" * 1200}
        for score in [0.94, 0.86, 0.81, 0.72, 0.68]
    ]
    rag_budgets = {
        "high": tool._select_rag_budget(fake_hits, requested_top_k=5, ranking_mode="bge_reranker"),
        "medium": tool._select_rag_budget(
            [{"rerank_score": 0.82, "document": "x" * 1200}] * 5,
            requested_top_k=5,
            ranking_mode="bge_reranker",
        ),
        "low": tool._select_rag_budget(
            [{"rerank_score": 0.61, "document": "x" * 1200}] * 5,
            requested_top_k=5,
            ranking_mode="bge_reranker",
        ),
    }

    print(json.dumps({"agent_budgets": budgets, "rag_budgets": rag_budgets}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
