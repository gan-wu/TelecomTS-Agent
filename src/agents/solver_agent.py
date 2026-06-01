from src.tools.llm_client import LLMClient
from src.tools.prompt_loader import PromptLoader
from src.models.budget_policy import TokenBudgetPolicy


class SolverAgent:
    """
    角色：问答解答专家 (QA Solver)
    职责：接收 AnalystAgent 产出的网络分析摘要 + 用户提出的具体问题。
         在充分理解了网络语境后，生成精准、简洁的最终答案。

    Prompt 来源：src/prompts/solver.yaml（外置管理，与代码解耦）
    论文角度：这就是整个系统的"推理核心 (Reasoning Core)"。
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.budget_policy = TokenBudgetPolicy()

    def solve(
        self,
        analysis_summary: str,
        question: str,
        original_context_report: str = "",
        route_signals: dict | None = None,
    ) -> str:
        """
        传入分析摘要 + 问题，输出最终简洁答案。
        original_context_report 作为原始数据的最后一道保险，防止 Analyst 丢失细节。
        """
        messages = PromptLoader.build_messages(
            agent="solver",
            analysis_summary=analysis_summary,
            context_report=original_context_report,
            question=question,
        )
        prompt_chars = sum(len(message.get("content", "")) for message in messages)
        budget = self.budget_policy.budget(
            task="solver",
            prompt_chars=prompt_chars,
            route_signals=route_signals,
        )

        answer = self.llm.query(
            messages,
            temperature=0.0,
            max_tokens=budget.max_tokens,
            task="solver",
            route_signals={
                **(route_signals or {}),
                "budget_max_tokens": budget.max_tokens,
                "budget_reason": budget.reason,
                "estimated_prompt_tokens": int(prompt_chars / 4),
            },
        )
        return answer or "[SolverAgent]: Answer Generation Failed"
