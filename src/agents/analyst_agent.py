from src.tools.llm_client import LLMClient
from src.tools.context_builder import ContextBuilder
from src.tools.prompt_loader import PromptLoader
from src.models.budget_policy import TokenBudgetPolicy


class AnalystAgent:
    """
    角色：高级网络分析师 (Senior Network Analyst)
    职责：不直接回答用户问题。
         专注于读入并"理解"当前基站的通信状态，生成一段结构化的分析摘要报告 (Analysis Report)。
         这份摘要是 SolverAgent 做出正确判断的核心上下文信息源。

    Prompt 来源：src/prompts/analyst.yaml（外置管理，与代码解耦）
    论文角度：体现了 Agent "感知-分析(Perception-Analysis)分离" 的专业架构思想。
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.budget_policy = TokenBudgetPolicy()

    def analyze(self, context_dict: dict, route_signals: dict | None = None) -> str:
        """
        传入原始数据 context 字典（来自 benchmark.json 中每条记录的 context 字段）。
        调用 LLM 生成分析摘要并返回字符串。
        """
        context_report = ContextBuilder.json_to_markdown_report(context_dict)

        messages = PromptLoader.build_messages(
            agent="analyst",
            context_report=context_report,
        )
        prompt_chars = sum(len(message.get("content", "")) for message in messages)
        budget = self.budget_policy.budget(
            task="analyst",
            prompt_chars=prompt_chars,
            route_signals=route_signals,
        )

        analysis_summary = self.llm.query(
            messages,
            temperature=0.0,
            max_tokens=budget.max_tokens,
            task="analyst",
            route_signals={
                **(route_signals or {}),
                "budget_max_tokens": budget.max_tokens,
                "budget_reason": budget.reason,
                "estimated_prompt_tokens": int(prompt_chars / 4),
            },
        )
        return analysis_summary or "[AnalystAgent]: Analysis Failed (No Response)"
