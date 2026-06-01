import re
from src.tools.llm_client import LLMClient
from src.tools.prompt_loader import PromptLoader
from src.tools.tool_router import TelecomToolRouter
from src.models.budget_policy import TokenBudgetPolicy


class CriticAgent:
    """
    角色：输出格式校验官 + 自我修正触发器 (Output Critic & Self-Correction Gatekeeper)
    职责：这是整个多智能体系统中最具技术含量的中间件，也是区别于"单纯调一次API"的关键创新点。

    核心逻辑：
        1. 用规则引擎 (Regex) 先做本地轻量级格式校验（零API成本）。
           ── 正则表达式从 src/prompts/critic.yaml 动态加载，与代码完全解耦。
        2. 如果规则引擎判断格式不合规，才触发 LLM 级的深度纠错（按需花费 Token）。
        3. 最终确保流入 Evaluator 的每一条输出都是合法的字符串，保障评测准确率的统计合法性。

    Prompt 来源：src/prompts/critic.yaml（外置管理，与代码解耦）
    论文角度：Critic 机制是一个独立的 Ablation Study 消融分析维度。
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.budget_policy = TokenBudgetPolicy()
        # 从 YAML 动态加载正则，避免硬编码在代码里
        self._patterns: dict[str, str] = PromptLoader.get_regex_patterns("critic")

    def _classify_question(self, question: str) -> str:
        """快速判断问题类型，用于挑选规则匹配的正则式。"""
        metric = TelecomToolRouter._detect_metric(question)
        if metric == "periodicity":
            return "periodicity"
        if metric == "trend":
            return "trend"
        if metric in {"mean", "variance"}:
            return "stat"
        return "classification"

    def _is_format_valid(self, answer: str, question_type: str) -> bool:
        """
        用轻量级正则进行格式验证。这是核心省钱节点：
        大多数回答格式正确则跳过 LLM 纠错直接通过，不花一分钱。
        """
        if question_type == "classification":
            # 分类题只需非空即合格，由 Evaluator 做 Exact Match
            return bool(answer and len(answer.strip()) > 0)

        pattern = self._patterns.get(question_type, "")
        if not pattern:
            return True

        return bool(re.match(pattern, answer.strip().lower()))

    def critique_and_correct(
        self,
        question: str,
        proposed_answer: str,
        route_signals: dict | None = None,
    ) -> tuple[str, bool]:
        """
        主方法：接收问题 + Solver 的答案，返回 (最终通过的答案, 是否触发了纠正)。

        Returns:
            str:  经格式核验/修正后的最终答案。
            bool: True 表示触发了 LLM 级纠正，用于统计 "Correction Rate" 这一高价值指标。
        """
        q_type = self._classify_question(question)

        # ===== 阶段1：本地规则快速校验（零 Token 花销）=====
        if self._is_format_valid(proposed_answer, q_type):
            return proposed_answer, False  # 格式已合格，直接放行

        # ===== 阶段2：触发 LLM 深度纠错（按需消费 Token）=====
        messages = PromptLoader.build_messages(
            agent="critic",
            question=question,
            proposed_answer=proposed_answer,
        )
        prompt_chars = sum(len(message.get("content", "")) for message in messages)
        merged_signals = {
            "format_sensitive": True,
            **(route_signals or {}),
        }
        budget = self.budget_policy.budget(
            task="critic",
            prompt_chars=prompt_chars,
            route_signals=merged_signals,
        )
        corrected_answer = self.llm.query(
            messages,
            temperature=0.0,
            max_tokens=budget.max_tokens,
            task="critic",
            route_signals={
                **merged_signals,
                "budget_max_tokens": budget.max_tokens,
                "budget_reason": budget.reason,
                "estimated_prompt_tokens": int(prompt_chars / 4),
            },
        )

        if corrected_answer:
            return corrected_answer, True   # 返回纠正后结果，并标记已触发纠正
        else:
            return proposed_answer, True    # 纠错也失败了，只好原样返回，保留标记
