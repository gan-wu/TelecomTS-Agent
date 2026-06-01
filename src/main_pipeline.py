"""
TelecomTS Multi-Agent QA Pipeline
====================================
论文方法论核心实现：
    Analyst → Solver → Critic 三层 Agent 协同工作流。
支持：
    - 自动加载 benchmark.json 执行批量推理
    - 分层指标统计 (MAE for TimeSeries / Accuracy for Network)
    - 结果落盘 predictions.csv（用于论文表格与 Error Analysis）
    - 消融实验模式（--no-analyst, --no-critic 开关）
    - 基于 tqdm 的进度监控与实时花费 Token 统计
"""

import os
import re
import sys
import json
import logging
import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from tqdm import tqdm
import pandas as pd

# On Windows, loading torch before sklearn avoids occasional c10.dll initialization
# failures when RAG embedding/reranker models are first used inside this process.
try:
    import torch as _torch  # noqa: F401
except Exception:
    _torch = None

from sklearn.metrics import accuracy_score, mean_absolute_error

# 把项目根目录加入 Python 模块搜索路径，确保无论从哪里执行脚本 import 都不报错
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.llm_client import LLMClient
from src.tools.context_builder import ContextBuilder
from src.tools.tool_router import TelecomToolRouter
from src.models.model_router import ModelRouterClient
from src.agents.analyst_agent import AnalystAgent
from src.agents.solver_agent import SolverAgent
from src.agents.critic_agent import CriticAgent
from src.tools.telecom_knowledge_tool import KnowledgeSearchConfig, TelecomKnowledgeTool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("FlagEmbedding").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# ──────────────────────────────────────────────
# 工具函数：从模型输出的自然语言中提取原始数值
# ──────────────────────────────────────────────

def extract_number(text: str):
    """
    用正则表达式从答案字符串里安全地抠出第一个浮点数。
    覆盖: 负数, 科学计数法, 整数
    例: "The variance of RSRP is -0.031" → -0.031
    """
    if not text:
        return None
    pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    matches = re.findall(pattern, text.replace(",", ""))
    return float(matches[-1]) if matches else None


def extract_number_from_ground_truth(text: str):
    """从 ground_truth 字符串中提取数字，逻辑同上。"""
    return extract_number(text)


def normalize_classification(text: str) -> str:
    """
    对网络分类题 (Network QA) 做关键词匹配归一化。
    统一映射到小写关键字，防止因大小写或措辞不同导致的假 Mismatch。
    """
    if not text:
        return ""
    t = text.lower().strip()

    # 地区问题：只匹配明确的 Zone A/B/C 或单字母答案，避免普通句子里的 a/b/c 误判。
    zone_match = re.search(r"\bzone[\s_-]*([abc])\b", t)
    if zone_match:
        return zone_match.group(1)
    if t in {"a", "b", "c"}:
        return t

    # 移动性问题
    if any(k in t for k in ["not moving", "no movement", "stationary", "static", "still"]):
        return "stationary"
    if any(k in t for k in ["in motion", "moving", "motion", "mobile", "movement between zones", "changed zones"]):
        return "in_motion"

    # 应用类型问题
    for app in ["youtube", "twitch", "file"]:
        if app in t:
            return app

    # 拥塞/干扰/异常问题：先处理否定表达，避免 "not congested" 命中 congested。
    negative_markers = [
        "not congest",
        "no congest",
        "uncongested",
        "not overload",
        "no overload",
        "no signs of overload",
        "within normal",
        "normal throughput",
        "traffic flowed normally",
        "operating normally",
        "normal operation",
        "performing well",
        "no anomaly",
        "anomaly was absent",
        "absent",
        "unaffected by jamming",
        "free of jamming",
        "jammer-free",
        "without jamming",
        "interference-free",
        "normal",
        "no",
    ]
    if any(k in t for k in negative_markers):
        return "no"

    positive_markers = [
        "congested",
        "congestion",
        "overloaded",
        "overload",
        "saturation",
        "heavy load",
        "jammed",
        "jamming",
        "interference",
        "anomaly",
        "anomalous",
        "present",
        "yes",
    ]
    if any(k in t for k in positive_markers):
        return "yes"

    return t  # 若未命中任何规则，返回原始文本参与 Exact Match


# ──────────────────────────────────────────────
# 主流水线类
# ──────────────────────────────────────────────

class TeleQnAPipeline:
    """
    5G 通信多智能体问答推理主管线。
    AnalystAgent → SolverAgent → CriticAgent 的完整协同工作流。
    """

    def __init__(
        self,
        llm_client: LLMClient,
        benchmark_path: str,
        output_path: str,
        use_analyst: bool = True,
        use_critic: bool = True,
        analyst_mode: str = "auto",
        use_tools: bool = True,
        use_rag: bool = True,
        rag_config: KnowledgeSearchConfig | None = None,
        request_interval: float = 0.3,
        workers: int = 1,
    ):
        """
        Args:
            llm_client:       已配置好的 LLM 接入客户端
            benchmark_path:   benchmark.json 的绝对路径
            output_path:      predictions.csv 的输出路径
            use_analyst:      消融开关 - 是否启用 AnalystAgent（论文消融实验 Variant B）
            use_critic:       消融开关 - 是否启用 CriticAgent（论文消融实验 Variant C）
            request_interval: 每次 API 请求后的主动睡眠时间（防触发限流，单位：秒）
        """
        self.analyst = AnalystAgent(llm_client)
        self.solver = SolverAgent(llm_client)
        self.critic = CriticAgent(llm_client)
        self.context_builder = ContextBuilder()
        self.tool_router = TelecomToolRouter(benchmark_path) if use_tools else None
        self.knowledge_tool = TelecomKnowledgeTool(rag_config) if use_rag else None
        self.rag_lock = threading.Lock()

        self.use_analyst = use_analyst
        self.use_critic = use_critic
        self.analyst_mode = analyst_mode
        self.use_tools = use_tools
        self.use_rag = use_rag
        self.interval = request_interval
        self.workers = max(1, workers)

        with open(benchmark_path, "r", encoding="utf-8") as f:
            self.benchmark = json.load(f)

        self.output_path = output_path
        logging.info(f"流水线已启动。共载入 {len(self.benchmark)} 条评测题目。")
        logging.info(f"消融配置 => AnalystAgent: {use_analyst} | CriticAgent: {use_critic} | AnalystMode: {analyst_mode}")
        logging.info(f"Tool Calling => ToolRouter: {use_tools}")
        logging.info(f"RAG Tool => TelecomKnowledgeTool: {use_rag}")
        logging.info(f"Execution => workers: {self.workers}")

    def _run_single(self, item: dict) -> dict:
        """
        对单条 benchmark 记录执行完整的三层 Agent 推理流程。
        返回一条包含预测结果的记录字典。
        """
        context = item["context"]
        question = item["question"]
        ground_truth = item["ground_truth"]
        tool_calls = []
        model_calls = []
        tool_answer_used = False
        route_policy = item.get("route_policy", "")
        rag_used = False
        rag_error = ""
        rag_ranking_mode = ""
        rag_confidence_bucket = ""
        rag_top_score = None
        rag_selected_top_k = None
        rag_sources: list[str] = []
        pipeline_route: list[str] = []
        analysis_mode_used = ""
        self._clear_last_model_call()

        route_result = self.tool_router.route(item["id"], question) if self.tool_router else None
        if route_result:
            tool_calls = route_result.tool_calls
            tool_answer_used = bool(route_result.answer)

        if route_result and route_result.answer:
            pipeline_route.append("tool_router")
            analysis_summary = "[ToolRouter] Deterministic tool answer used."
            analysis_mode_used = "tool_router"
            proposed_answer = route_result.answer
        elif route_policy == "rag_first" and self.knowledge_tool:
            pipeline_route.append("rag_tool")
            try:
                with self.rag_lock:
                    rag_result = self.knowledge_tool.search_telecom_knowledge(question, top_k=5)

                rag_used = True
                tool_calls.append(rag_result["tool_call"])
                rag_budget = rag_result.get("rag_budget") or {}
                rag_ranking_mode = str(rag_result.get("ranking_mode") or "")
                rag_confidence_bucket = str(rag_budget.get("confidence_bucket") or "")
                rag_top_score = rag_budget.get("rag_top_score")
                rag_selected_top_k = rag_budget.get("selected_top_k")
                rag_sources = [str(hit.get("source", "")) for hit in rag_result.get("hits", [])]

                case_context_report = ContextBuilder.json_to_markdown_report(context)
                context_report = (
                    f"{rag_result['context_block']}\n\n"
                    "## Current Case KPI Data\n"
                    f"{case_context_report}"
                )
                analysis_summary = (
                    "[RAGTool] Retrieved telecom-domain evidence. "
                    "Answer with the evidence first; use current case data only when the question asks about this session."
                )
                analysis_mode_used = "rag_tool"
                proposed_answer = self.solver.solve(
                    analysis_summary,
                    question,
                    context_report,
                    route_signals={
                        "case_id": item["id"],
                        "item_type": item["type"],
                        "knowledge_question": True,
                        "has_rag": True,
                        "rag_top_score": rag_top_score,
                        "rag_confidence_bucket": rag_confidence_bucket,
                        "rag_selected_top_k": rag_selected_top_k,
                    },
                )
                model_calls = self._append_last_model_call(model_calls)
                time.sleep(self.interval)
            except Exception as exc:
                rag_error = str(exc)
                logging.warning(f"[RAG] ID={item.get('id')} failed, fallback to Analyst/Solver: {rag_error}")
                pipeline_route.append("rag_fallback_agent")
                analysis_summary, proposed_answer, model_calls, analysis_mode_used = self._run_agent_path(
                    item=item,
                    context=context,
                    question=question,
                    model_calls=model_calls,
                )
        else:
            pipeline_route.append("agent")
            # ── 阶段一：感知层 (Analyst) ──
            analysis_summary, model_calls, analysis_mode_used = self._get_analysis_summary(
                item=item,
                context=context,
                model_calls=model_calls,
            )

            # ── 阶段二：推理层 (Solver) ──
            context_report = ContextBuilder.json_to_markdown_report(context)
            proposed_answer = self.solver.solve(
                analysis_summary,
                question,
                context_report,
                route_signals={
                    "case_id": item["id"],
                    "item_type": item["type"],
                    "format_sensitive": item["type"] == "timeseries",
                },
            )
            model_calls = self._append_last_model_call(model_calls)
            time.sleep(self.interval)

        # ── 阶段三：校验层 (Critic) ──
        is_corrected = False
        if self.use_critic:
            pipeline_route.append("critic")
            final_answer, is_corrected = self.critic.critique_and_correct(
                question,
                proposed_answer,
                route_signals={
                    "case_id": item["id"],
                    "item_type": item["type"],
                    "has_rag": rag_used,
                    "rag_top_score": rag_top_score,
                    "rag_confidence_bucket": rag_confidence_bucket,
                },
            )
            model_calls = self._append_last_model_call(model_calls)
            if is_corrected:
                time.sleep(self.interval)
        else:
            final_answer = proposed_answer

        return {
            "id": item["id"],
            "type": item["type"],
            "benchmark_group": item.get("benchmark_group", ""),
            "route_policy": route_policy,
            "expected_route": item.get("expected_route", ""),
            "difficulty": item.get("difficulty", ""),
            "skill_tags": json.dumps(item.get("skill_tags", []), ensure_ascii=False),
            "question": question,
            "ground_truth": ground_truth,
            "analysis_summary": analysis_summary,
            "analysis_mode": analysis_mode_used,
            "proposed_answer": proposed_answer,
            "final_answer": final_answer,
            "critic_triggered": is_corrected,
            "tool_answer_used": tool_answer_used,
            "rag_used": rag_used,
            "rag_ranking_mode": rag_ranking_mode,
            "rag_confidence_bucket": rag_confidence_bucket,
            "rag_top_score": rag_top_score,
            "rag_selected_top_k": rag_selected_top_k,
            "rag_sources": json.dumps(rag_sources, ensure_ascii=False),
            "rag_error": rag_error,
            "pipeline_route": " -> ".join(pipeline_route),
            "tool_calls": json.dumps(tool_calls, ensure_ascii=False),
            "model_calls": json.dumps(model_calls, ensure_ascii=False),
        }

    def _run_agent_path(
        self,
        item: dict,
        context: dict,
        question: str,
        model_calls: list[dict],
    ) -> tuple[str, str, list[dict], str]:
        analysis_summary, model_calls, analysis_mode_used = self._get_analysis_summary(
            item=item,
            context=context,
            model_calls=model_calls,
        )

        context_report = ContextBuilder.json_to_markdown_report(context)
        proposed_answer = self.solver.solve(
            analysis_summary,
            question,
            context_report,
            route_signals={
                "case_id": item["id"],
                "item_type": item["type"],
                "format_sensitive": item["type"] == "timeseries",
            },
        )
        model_calls = self._append_last_model_call(model_calls)
        time.sleep(self.interval)
        return analysis_summary, proposed_answer, model_calls, analysis_mode_used

    def _get_analysis_summary(
        self,
        item: dict,
        context: dict,
        model_calls: list[dict],
    ) -> tuple[str, list[dict], str]:
        if not self.use_analyst:
            return "[No Analyst] Skipped in ablation mode.", model_calls, "disabled"

        if self._use_llm_analyst(item):
            analysis_summary = self.analyst.analyze(
                context,
                route_signals={
                    "case_id": item["id"],
                    "item_type": item["type"],
                },
            )
            model_calls = self._append_last_model_call(model_calls)
            time.sleep(self.interval)
            return analysis_summary, model_calls, "llm"

        return self._build_deterministic_analysis(context), model_calls, "deterministic"

    def _use_llm_analyst(self, item: dict) -> bool:
        if self.analyst_mode == "llm":
            return True
        if self.analyst_mode == "deterministic":
            return False
        # Auto mode keeps the costly LLM analyst only for ticket-style diagnosis.
        return item.get("type") == "diagnosis" or item.get("benchmark_group") == "diagnosis_reasoning"

    @staticmethod
    def _build_deterministic_analysis(context: dict) -> str:
        stats = context.get("statistics", {}) or {}

        def mean(kpi: str, default: float = 0.0) -> float:
            value = (stats.get(kpi) or {}).get("mean", default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        rsrp = mean("RSRP")
        ul_snr = mean("UL_SNR")
        ul_bler = mean("UL_BLER")
        dl_bler = mean("DL_BLER")
        prb_dl = mean("PRB_Utilization_DL")
        prb_ul = mean("PRB_Utilization_UL")
        tx_bytes = mean("TX_Bytes")
        rx_bytes = mean("RX_Bytes")

        signal_state = "poor" if rsrp < -105 else "moderate" if rsrp < -90 else "good"
        reliability_state = "high error" if max(ul_bler, dl_bler) >= 0.08 else "normal error"
        resource_state = "high load" if max(prb_dl, prb_ul) >= 80 else "moderate load" if max(prb_dl, prb_ul) >= 60 else "light load"

        return (
            f"- Signal Quality: {signal_state}; RSRP={rsrp:.3f}, UL_SNR={ul_snr:.3f}.\n"
            f"- Reliability: {reliability_state}; UL_BLER={ul_bler:.3f}, DL_BLER={dl_bler:.3f}.\n"
            f"- Resource/Traffic: {resource_state}; PRB_DL={prb_dl:.3f}, PRB_UL={prb_ul:.3f}, "
            f"TX_Bytes={tx_bytes:.3f}, RX_Bytes={rx_bytes:.3f}.\n"
            "- Note: This deterministic summary is derived only from KPI statistics; "
            "it does not expose benchmark labels or ground-truth categories."
        )

    def run(self):
        """批量运行所有题目，计算评测指标，并将详细结果落盘到 CSV。"""
        if self.workers == 1:
            records = []
            for item in tqdm(self.benchmark, desc="Running TeleQnA-Agent Pipeline", unit="sample"):
                try:
                    result = self._run_single(item)
                    records.append(result)
                except Exception as e:
                    logging.error(f"[FATAL] 处理 ID={item.get('id')} 时出错: {e}")
                    records.append(self._error_record(item))
        else:
            records = [None] * len(self.benchmark)
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(self._run_single, item): (index, item)
                    for index, item in enumerate(self.benchmark)
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Running TeleQnA-Agent Pipeline ({self.workers} workers)",
                    unit="sample",
                ):
                    index, item = futures[future]
                    try:
                        records[index] = future.result()
                    except Exception as e:
                        logging.error(f"[FATAL] 处理 ID={item.get('id')} 时出错: {e}")
                        records[index] = self._error_record(item)
            records = [record for record in records if record is not None]

        df = pd.DataFrame(records)
        df.to_csv(self.output_path, index=False, encoding="utf-8-sig")
        logging.info(f"全部结果已落盘至: {self.output_path}")

        self._evaluate(df)

    def _evaluate(self, df: pd.DataFrame):
        """对结果 DataFrame 分类型计算评测指标，并在终端打印论文级成绩报告。"""
        print("\n" + "=" * 65)
        print("          TeleQnA Pipeline - Evaluation Report")
        print("=" * 65)

        correction_rate = df["critic_triggered"].sum() / len(df) * 100
        print(f"  [CriticAgent] Correction Rate: {correction_rate:.2f}%  "
              f"({int(df['critic_triggered'].sum())} / {len(df)} triggered)")

        # ── 时序统计题：MAE 评测 ──
        ts_df = df[df["type"] == "timeseries"].copy()
        if not ts_df.empty:
            ts_df["pred_num"] = ts_df["final_answer"].apply(extract_number)
            ts_df["true_num"] = ts_df["ground_truth"].apply(extract_number_from_ground_truth)
            valid_ts = ts_df.dropna(subset=["pred_num", "true_num"])
            if not valid_ts.empty:
                mae = mean_absolute_error(valid_ts["true_num"], valid_ts["pred_num"])
                parse_rate = len(valid_ts) / len(ts_df) * 100
                print(f"\n  [TimeSeries QA]")
                print(f"    Total Samples : {len(ts_df)}")
                print(f"    Parseable Rate: {parse_rate:.1f}%")
                print(f"    MAE           : {mae:.4f}  ← 对标论文 Table 6")
            else:
                print(f"\n  [TimeSeries QA] 无法解析出任何数值，请检查 Prompt 设计。")

        # ── 网络分类题：Accuracy 评测 ──
        net_df = df[df["type"] == "network"].copy()
        if not net_df.empty:
            net_df["pred_norm"] = net_df["final_answer"].apply(normalize_classification)
            net_df["true_norm"] = net_df["ground_truth"].apply(normalize_classification)
            acc = accuracy_score(net_df["true_norm"], net_df["pred_norm"])
            print(f"\n  [Network QA]")
            print(f"    Total Samples : {len(net_df)}")
            print(f"    Accuracy      : {acc * 100:.2f}%  ← 对标论文 Table 6")

        print("\n" + "=" * 65)
        print(f"  详细错误分析请查阅: {self.output_path}")
        print("=" * 65 + "\n")

    @staticmethod
    def _error_record(item: dict) -> dict:
        return {
            "id": item.get("id"), "type": item.get("type"),
            "benchmark_group": item.get("benchmark_group", ""),
            "route_policy": item.get("route_policy", ""),
            "expected_route": item.get("expected_route", ""),
            "difficulty": item.get("difficulty", ""),
            "skill_tags": json.dumps(item.get("skill_tags", []), ensure_ascii=False),
            "question": item.get("question"), "ground_truth": item.get("ground_truth"),
            "analysis_summary": "ERROR", "proposed_answer": "ERROR",
            "analysis_mode": "error",
            "final_answer": "ERROR", "critic_triggered": False,
            "tool_answer_used": False,
            "rag_used": False,
            "rag_ranking_mode": "",
            "rag_confidence_bucket": "",
            "rag_top_score": None,
            "rag_selected_top_k": None,
            "rag_sources": "[]",
            "rag_error": "",
            "pipeline_route": "error",
            "tool_calls": "[]",
            "model_calls": "[]",
        }

    def _append_last_model_call(self, model_calls: list[dict]) -> list[dict]:
        llm = self.analyst.llm
        last_call = getattr(llm, "last_call", None)
        if last_call and (not model_calls or model_calls[-1] != last_call):
            model_calls.append(dict(last_call))
        return model_calls

    def _clear_last_model_call(self) -> None:
        llm = self.analyst.llm
        if hasattr(llm, "last_call"):
            llm.last_call = None


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="TeleQnA Multi-Agent Pipeline")
    parser.add_argument("--api-key",     type=str, default=None, help="LLM API Key（优先于环境变量）")
    parser.add_argument("--base-url",    type=str, default="https://api.deepseek.com", help="API Base URL")
    parser.add_argument("--model",       type=str, default="deepseek-chat", help="模型名称")
    parser.add_argument("--backend",     type=str, default="single", choices=["single", "local", "deepseek", "hybrid"], help="模型后端：single/local/deepseek/hybrid")
    parser.add_argument("--local-base-url", type=str, default=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8080/v1"), help="本地 OpenAI-compatible 服务地址")
    parser.add_argument("--local-model", type=str, default=os.getenv("LOCAL_MODEL", "qwen3.5-9b-q4"), help="本地模型名称")
    parser.add_argument("--local-api-key", type=str, default=os.getenv("LOCAL_API_KEY", "EMPTY"), help="本地模型 API Key 占位")
    parser.add_argument("--local-max-concurrency", type=int, default=None, help="限制本地模型同时请求数，建议与 llama-server -np 保持一致")
    parser.add_argument("--deepseek-base-url", type=str, default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), help="DeepSeek API Base URL")
    parser.add_argument("--deepseek-api-key", type=str, default=os.getenv("DEEPSEEK_API_KEY", ""), help="DeepSeek API Key")
    parser.add_argument("--deepseek-flash-model", type=str, default=os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"), help="DeepSeek Flash 模型名")
    parser.add_argument("--deepseek-pro-model", type=str, default=os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-v4-pro"), help="DeepSeek Pro 模型名")
    parser.add_argument("--benchmark",   type=str, default="data/benchmark.json", help="评测集路径")
    parser.add_argument("--output",      type=str, default="results/predictions.csv", help="输出 CSV 路径")
    parser.add_argument("--interval",    type=float, default=0.3, help="每次请求后的限流防护间隔（秒）")
    parser.add_argument("--workers",     type=int, default=1, help="并行样本数。单条样本内部仍按 Analyst->Solver->Critic 顺序执行")
    # 消融实验开关 (Ablation Study Switches)
    parser.add_argument("--no-analyst",  action="store_true", help="消融实验：禁用 AnalystAgent")
    parser.add_argument(
        "--analyst-mode",
        choices=["auto", "llm", "deterministic"],
        default=os.getenv("ANALYST_MODE", "auto"),
        help="Analyst 执行方式：auto=诊断题用 LLM、普通网络题用规则摘要；llm=全部用 LLM；deterministic=全部用规则摘要",
    )
    parser.add_argument("--no-critic",   action="store_true", help="消融实验：禁用 CriticAgent")
    parser.add_argument("--no-tools",    action="store_true", help="消融实验：禁用 Tool Calling")
    parser.add_argument("--no-rag",      action="store_true", help="Disable RAG Tool for rag_first samples")
    parser.add_argument("--rag-embedding-device", default=None, help="RAG embedding device, e.g. cuda:0 or cpu")
    parser.add_argument("--rag-reranker-device", default=None, help="RAG reranker device, e.g. cuda:0 or cpu")
    parser.add_argument("--rag-reranker-batch-size", type=int, default=None, help="RAG reranker batch size")
    parser.add_argument("--disable-rag-reranker", action="store_true", help="Use Chroma + BM25 + RRF fallback without BGE reranker")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "results", exist_ok=True)

    if args.backend == "single":
        client = LLMClient(
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.model,
        )
    else:
        if args.local_max_concurrency:
            os.environ["LOCAL_MAX_CONCURRENCY"] = str(args.local_max_concurrency)
        client = ModelRouterClient.from_env(
            mode=args.backend,
            local_base_url=args.local_base_url,
            local_model=args.local_model,
            local_api_key=args.local_api_key,
            deepseek_base_url=args.deepseek_base_url,
            deepseek_api_key=args.deepseek_api_key,
            deepseek_flash_model=args.deepseek_flash_model,
            deepseek_pro_model=args.deepseek_pro_model,
        )

    rag_config = KnowledgeSearchConfig.from_env()
    rag_config = replace(
        rag_config,
        embedding_device=args.rag_embedding_device or rag_config.embedding_device,
        reranker_device=args.rag_reranker_device or rag_config.reranker_device,
        reranker_batch_size=args.rag_reranker_batch_size or rag_config.reranker_batch_size,
        reranker_enabled=not args.disable_rag_reranker,
    )

    pipeline = TeleQnAPipeline(
        llm_client=client,
        benchmark_path=args.benchmark,
        output_path=args.output,
        use_analyst=not args.no_analyst,
        use_critic=not args.no_critic,
        analyst_mode=args.analyst_mode,
        use_tools=not args.no_tools,
        use_rag=not args.no_rag,
        rag_config=rag_config,
        request_interval=args.interval,
        workers=args.workers,
    )

    pipeline.run()
