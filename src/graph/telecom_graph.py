import json
import os
import sys
import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.agents.analyst_agent import AnalystAgent
from src.agents.critic_agent import CriticAgent
from src.agents.solver_agent import SolverAgent
from src.tools.context_builder import ContextBuilder
from src.tools.telecom_knowledge_tool import KnowledgeSearchConfig, TelecomKnowledgeTool
from src.tools.tool_router import TelecomToolRouter


class TelecomGraphState(TypedDict, total=False):
    case_id: str
    question: str
    context: dict[str, Any]
    ground_truth: str
    item_type: str
    tool_calls: list[dict[str, Any]]
    tool_answer: str | None
    tool_answer_used: bool
    knowledge_question: bool
    rag_context: str | None
    rag_hits: list[dict[str, Any]]
    rag_budget: dict[str, Any]
    analysis_summary: str | None
    proposed_answer: str | None
    final_answer: str | None
    critic_triggered: bool
    model_calls: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    error: str | None


def _trace(node: str, **metadata: Any) -> dict[str, Any]:
    clean_metadata = {k: v for k, v in metadata.items() if v is not None}
    return {
        "node": node,
        "timestamp": round(time.time(), 3),
        **clean_metadata,
    }


class TelecomLangGraphWorkflow:
    """LangGraph orchestration for KPI tools, RAG tool, Analyst, Solver, and Critic."""

    def __init__(
        self,
        llm_client,
        benchmark_path: str,
        use_tools: bool = True,
        use_rag: bool = True,
        use_critic: bool = True,
        rag_config: KnowledgeSearchConfig | None = None,
    ):
        self.llm_client = llm_client
        self.tool_router = TelecomToolRouter(benchmark_path) if use_tools else None
        self.knowledge_tool = TelecomKnowledgeTool(rag_config) if use_rag else None
        self.analyst = AnalystAgent(llm_client)
        self.solver = SolverAgent(llm_client)
        self.critic = CriticAgent(llm_client)
        self.use_critic = use_critic
        self.benchmark_path = os.path.abspath(benchmark_path)
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            self.benchmark = json.load(f)
        self.by_id = {item["id"]: item for item in self.benchmark}
        self.graph = self._build_graph()

    def invoke(self, case_id: str, question: str | None = None) -> TelecomGraphState:
        item = self.by_id.get(case_id)
        if not item:
            raise ValueError(f"case_id not found: {case_id}")

        initial_state: TelecomGraphState = {
            "case_id": case_id,
            "question": question or item["question"],
            "context": {},
            "ground_truth": "",
            "item_type": "",
            "tool_calls": [],
            "tool_answer": None,
            "tool_answer_used": False,
            "knowledge_question": False,
            "rag_context": None,
            "rag_hits": [],
            "rag_budget": {},
            "analysis_summary": None,
            "proposed_answer": None,
            "final_answer": None,
            "critic_triggered": False,
            "model_calls": [],
            "trace": [],
            "error": None,
        }
        return self.graph.invoke(initial_state)

    def _build_graph(self):
        builder = StateGraph(TelecomGraphState)

        builder.add_node("load_case", self._load_case)
        builder.add_node("tool_router", self._tool_router)
        builder.add_node("rag_tool", self._rag_tool)
        builder.add_node("analyst", self._analyst)
        builder.add_node("solver", self._solver)
        builder.add_node("critic", self._critic)

        builder.add_edge(START, "load_case")
        builder.add_edge("load_case", "tool_router")
        builder.add_conditional_edges(
            "tool_router",
            self._route_after_tools,
            {
                "critic": "critic",
                "rag_tool": "rag_tool",
                "analyst": "analyst",
            },
        )
        builder.add_edge("rag_tool", "solver")
        builder.add_edge("analyst", "solver")
        builder.add_edge("solver", "critic")
        builder.add_edge("critic", END)

        return builder.compile()

    def _load_case(self, state: TelecomGraphState) -> TelecomGraphState:
        item = self.by_id[state["case_id"]]
        return {
            "context": item["context"],
            "ground_truth": item.get("ground_truth", ""),
            "item_type": item.get("type", ""),
            "trace": state.get("trace", []) + [
                _trace("load_case", case_id=state["case_id"], item_type=item.get("type"))
            ],
        }

    def _tool_router(self, state: TelecomGraphState) -> TelecomGraphState:
        question = state["question"]
        case_data_question = is_case_data_question(question)
        knowledge_question = is_knowledge_question(question) and not case_data_question
        skip_deterministic_tools = knowledge_question

        if not self.tool_router:
            return {
                "tool_calls": [],
                "tool_answer": None,
                "tool_answer_used": False,
                "knowledge_question": knowledge_question,
                "trace": state.get("trace", []) + [
                    _trace(
                        "tool_router",
                        routed=False,
                        disabled=True,
                        knowledge_question=knowledge_question,
                    )
                ],
            }

        if skip_deterministic_tools:
            return {
                "tool_calls": [],
                "tool_answer": None,
                "tool_answer_used": False,
                "knowledge_question": True,
                "trace": state.get("trace", []) + [
                    _trace(
                        "tool_router",
                        routed=False,
                        skipped_for_knowledge=True,
                        knowledge_question=True,
                    )
                ],
            }

        route_result = self.tool_router.route(state["case_id"], question)
        return {
            "tool_calls": route_result.tool_calls,
            "tool_answer": route_result.answer,
            "tool_answer_used": bool(route_result.answer),
            "knowledge_question": knowledge_question,
            "trace": state.get("trace", []) + [
                _trace(
                    "tool_router",
                    routed=route_result.routed,
                    tool_count=len(route_result.tool_calls),
                    tool_answer_used=bool(route_result.answer),
                    knowledge_question=knowledge_question,
                )
            ],
        }

    def _route_after_tools(self, state: TelecomGraphState) -> Literal["critic", "rag_tool", "analyst"]:
        if state.get("tool_answer"):
            return "critic"
        if self.knowledge_tool and state.get("knowledge_question"):
            return "rag_tool"
        return "analyst"

    def _rag_tool(self, state: TelecomGraphState) -> TelecomGraphState:
        if not self.knowledge_tool:
            return {
                "rag_context": None,
                "rag_hits": [],
                "trace": state.get("trace", []) + [_trace("rag_tool", disabled=True)],
            }

        try:
            result = self.knowledge_tool.search_telecom_knowledge(state["question"], top_k=5)
            tool_calls = list(state.get("tool_calls", []))
            tool_calls.append(result["tool_call"])
            return {
                "tool_calls": tool_calls,
                "rag_context": result["context_block"],
                "rag_hits": result["hits"],
                "rag_budget": result.get("rag_budget", {}),
                "trace": state.get("trace", []) + [
                    _trace(
                        "rag_tool",
                        final_count=result["final_count"],
                        dense_count=result["dense_count"],
                        bm25_count=result["bm25_count"],
                        fused_count=result["fused_count"],
                        ranking_mode=result.get("ranking_mode"),
                        reranker_available=result.get("reranker_available"),
                        confidence_bucket=(result.get("rag_budget") or {}).get("confidence_bucket"),
                        selected_top_k=(result.get("rag_budget") or {}).get("selected_top_k"),
                        saved_context_chars_est=(result.get("rag_budget") or {}).get("saved_context_chars_est"),
                    )
                ],
            }
        except Exception as exc:
            return {
                "rag_context": None,
                "rag_hits": [],
                "error": str(exc),
                "trace": state.get("trace", []) + [_trace("rag_tool", error=str(exc))],
            }

    def _analyst(self, state: TelecomGraphState) -> TelecomGraphState:
        analysis_summary = self.analyst.analyze(
            state["context"],
            route_signals={
                "case_id": state["case_id"],
                "item_type": state.get("item_type", ""),
            },
        )
        model_calls = self._append_last_model_call(state)
        return {
            "analysis_summary": analysis_summary,
            "model_calls": model_calls,
            "trace": state.get("trace", []) + [
                _trace("analyst", output_chars=len(analysis_summary or ""))
            ],
        }

    def _solver(self, state: TelecomGraphState) -> TelecomGraphState:
        case_context_report = ContextBuilder.json_to_markdown_report(state["context"])
        if state.get("rag_context"):
            context_report = (
                f"{state['rag_context']}\n\n"
                "## Current Case KPI Data\n"
                f"{case_context_report}"
            )
            analysis_summary = (
                state.get("analysis_summary")
                or "[RAGTool] Retrieved telecom-domain evidence. Answer with the evidence first; use current case data only when the question asks about this session."
            )
        else:
            context_report = case_context_report
            analysis_summary = state.get("analysis_summary") or ""

        rag_hits = state.get("rag_hits") or []
        rag_budget = state.get("rag_budget") or {}
        rag_top_score = rag_budget.get("rag_top_score")
        if rag_top_score is None and rag_hits:
            rag_top_score = rag_hits[0].get("rerank_score")
        proposed_answer = self.solver.solve(
            analysis_summary,
            state["question"],
            context_report,
            route_signals={
                "case_id": state["case_id"],
                "item_type": state.get("item_type", ""),
                "knowledge_question": bool(state.get("knowledge_question")),
                "has_rag": bool(state.get("rag_context")),
                "rag_top_score": rag_top_score,
                "rag_confidence_bucket": rag_budget.get("confidence_bucket"),
                "rag_selected_top_k": rag_budget.get("selected_top_k"),
            },
        )
        model_calls = self._append_last_model_call(state)
        return {
            "proposed_answer": proposed_answer,
            "model_calls": model_calls,
            "trace": state.get("trace", []) + [
                _trace("solver", output_chars=len(proposed_answer or ""))
            ],
        }

    def _critic(self, state: TelecomGraphState) -> TelecomGraphState:
        proposed_answer = state.get("tool_answer") or state.get("proposed_answer") or ""
        if self.use_critic:
            final_answer, critic_triggered = self.critic.critique_and_correct(
                state["question"],
                proposed_answer,
                route_signals={
                    "case_id": state["case_id"],
                    "item_type": state.get("item_type", ""),
                },
            )
        else:
            final_answer, critic_triggered = proposed_answer, False

        model_calls = self._append_last_model_call(state)
        return {
            "proposed_answer": proposed_answer,
            "final_answer": final_answer,
            "critic_triggered": critic_triggered,
            "model_calls": model_calls,
            "trace": state.get("trace", []) + [
                _trace(
                    "critic",
                    triggered=critic_triggered,
                    final_chars=len(final_answer or ""),
                )
            ],
        }

    def _append_last_model_call(self, state: TelecomGraphState) -> list[dict[str, Any]]:
        calls = list(state.get("model_calls", []))
        last_call = getattr(self.llm_client, "last_call", None)
        if last_call and (not calls or calls[-1] != last_call):
            calls.append(dict(last_call))
        return calls


KNOWLEDGE_INTENT_MARKERS = [
    "what is",
    "what does",
    "explain",
    "define",
    "meaning",
    "why",
    "how does",
    "how to",
    "difference",
    "relationship",
    "principle",
    "troubleshoot",
    "best practice",
    "什么",
    "解释",
    "定义",
    "含义",
    "为什么",
    "如何",
    "怎么",
    "原理",
    "区别",
    "排查",
]

TELECOM_DOMAIN_MARKERS = [
    "5g",
    "ran",
    "o-ran",
    "oran",
    "gnb",
    "cu",
    "du",
    "ric",
    "e2",
    "a1",
    "srsran",
    "openairinterface",
    "kpi",
    "rsrp",
    "bler",
    "ul_bler",
    "dl_bler",
    "prb",
    "mcs",
    "qci",
    "jamming",
    "jammer",
    "interference",
    "handover",
    "throughput",
    "latency",
    "congestion",
    "anomaly",
    "异常",
    "拥塞",
    "干扰",
    "基站",
    "指标",
]

CASE_DATA_MARKERS = [
    "this",
    "current",
    "session",
    "case",
    "user",
    "present",
    "detected",
    "identify",
    "which",
    "whether",
    "这条",
    "当前",
    "本次",
    "用户",
    "是否",
    "哪",
]


def is_knowledge_question(question: str) -> bool:
    q = (question or "").lower()
    has_intent = any(marker in q for marker in KNOWLEDGE_INTENT_MARKERS)
    has_domain = any(marker in q for marker in TELECOM_DOMAIN_MARKERS)
    return has_intent and has_domain


def is_case_data_question(question: str) -> bool:
    q = question or ""
    q_lower = q.lower()
    if TelecomToolRouter._detect_metric(q) and TelecomToolRouter._detect_kpi(q):
        return True
    if TelecomToolRouter._detect_label(q) and any(marker in q_lower for marker in CASE_DATA_MARKERS):
        return True
    if TelecomToolRouter._is_anomaly_question(q) and any(marker in q_lower for marker in CASE_DATA_MARKERS):
        return True
    return False
