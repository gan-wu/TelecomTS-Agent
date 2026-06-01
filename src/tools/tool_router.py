import re
from dataclasses import dataclass, field
from typing import Any

from src.tools.telecom_tools import (
    KPI_ALIASES,
    TelecomBenchmarkStore,
    ToolCallRecord,
    format_anomaly_answer,
    format_kpi_answer,
    format_label_answer,
)


@dataclass
class ToolRouteResult:
    answer: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    routed: bool = False


class TelecomToolRouter:
    """
    Rule-first tool router for deterministic telecom QA.

    This is intentionally simple: it makes tool calling explainable in interviews,
    and can later be replaced by model-native Function Calling or LangGraph ToolNode.
    """

    def __init__(self, benchmark_path: str):
        self.store = TelecomBenchmarkStore(benchmark_path)

    def route(self, case_id: str, question: str) -> ToolRouteResult:
        q = question or ""
        case = self.store.get_case(case_id)
        route_policy = case.get("route_policy")

        if route_policy in {"agent_first", "rag_first"}:
            return ToolRouteResult()

        if route_policy == "multi_tool":
            return self._call_multi_label_tools(case_id)

        metric = self._detect_metric(q)
        kpi = self._detect_kpi(q)
        if metric and kpi:
            return self._call_kpi_stat(case_id, kpi, metric)

        if self._is_anomaly_question(q):
            return self._call_anomaly_info(case_id)

        label = self._detect_label(q)
        if label:
            return self._call_network_label(case_id, label)

        return ToolRouteResult()

    def _call_multi_label_tools(self, case_id: str) -> ToolRouteResult:
        labels = ["application", "zone", "mobility", "congestion", "anomaly_present"]
        tool_calls = []
        values: dict[str, Any] = {}
        for label in labels:
            args = {"case_id": case_id, "label": label}
            try:
                value = self.store.get_network_label(**args)
                values[label] = value
                tool_calls.append(ToolCallRecord("get_network_label", args, value).to_dict())
            except Exception as exc:
                tool_calls.append(
                    ToolCallRecord("get_network_label", args, None, success=False, error=str(exc)).to_dict()
                )

        if not values:
            return ToolRouteResult(tool_calls=tool_calls, routed=False)

        answer = (
            f"Application: {values.get('application', 'Unknown')}; "
            f"Zone: {values.get('zone', 'Unknown')}; "
            f"Mobility: {values.get('mobility', 'Unknown')}; "
            f"Congestion: {values.get('congestion', 'Unknown')}; "
            f"Anomaly: {values.get('anomaly_present', 'Unknown')}."
        )
        return ToolRouteResult(answer=answer, tool_calls=tool_calls, routed=True)

    def _call_kpi_stat(self, case_id: str, kpi: str, metric: str) -> ToolRouteResult:
        args = {"case_id": case_id, "kpi": kpi, "metric": metric}
        try:
            value = self.store.get_kpi_stat(**args)
            answer = format_kpi_answer(kpi, metric, value)
            record = ToolCallRecord("get_kpi_stat", args, value)
            return ToolRouteResult(answer=answer, tool_calls=[record.to_dict()], routed=True)
        except Exception as exc:
            record = ToolCallRecord("get_kpi_stat", args, None, success=False, error=str(exc))
            return ToolRouteResult(tool_calls=[record.to_dict()], routed=False)

    def _call_network_label(self, case_id: str, label: str) -> ToolRouteResult:
        args = {"case_id": case_id, "label": label}
        try:
            value = self.store.get_network_label(**args)
            answer = format_label_answer(label, value)
            record = ToolCallRecord("get_network_label", args, value)
            return ToolRouteResult(answer=answer, tool_calls=[record.to_dict()], routed=True)
        except Exception as exc:
            record = ToolCallRecord("get_network_label", args, None, success=False, error=str(exc))
            return ToolRouteResult(tool_calls=[record.to_dict()], routed=False)

    def _call_anomaly_info(self, case_id: str) -> ToolRouteResult:
        args = {"case_id": case_id}
        try:
            info = self.store.get_anomaly_info(case_id)
            answer = format_anomaly_answer(info)
            record = ToolCallRecord("get_anomaly_info", args, info)
            return ToolRouteResult(answer=answer, tool_calls=[record.to_dict()], routed=True)
        except Exception as exc:
            record = ToolCallRecord("get_anomaly_info", args, None, success=False, error=str(exc))
            return ToolRouteResult(tool_calls=[record.to_dict()], routed=False)

    @staticmethod
    def _detect_metric(question: str) -> str | None:
        q = question.lower()
        if "trend" in q or "increasing" in q or "decreasing" in q:
            return "trend"
        if "periodicity" in q or "periodic" in q or "period length" in q:
            return "periodicity"
        if any(token in q for token in ["variance", " var "]):
            return "variance"
        if re.search(r"\b(average|avg)\b", q):
            return "mean"
        if re.search(r"\bmean\s+(value|of)\b|\bwhat\s+is\s+the\s+mean\b|\bmean\b.*\bof\b", q):
            return "mean"
        return None

    @staticmethod
    def _detect_kpi(question: str) -> str | None:
        q = question.lower()
        candidates = sorted(set(KPI_ALIASES.values()) | set(KPI_ALIASES.keys()), key=len, reverse=True)
        for candidate in candidates:
            variants = {
                candidate.lower(),
                candidate.lower().replace("_", " "),
                candidate.lower().replace("_", "-"),
            }
            for variant in variants:
                if re.search(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", q):
                    return candidate
        return None

    @staticmethod
    def _detect_label(question: str) -> str | None:
        q = question.lower()
        if any(token in q for token in ["youtube", "twitch", "file download", "application", "service", "traffic"]):
            return "application"
        if any(token in q for token in ["moving", "movement", "motion", "static", "stationary", "still"]):
            return "mobility"
        if "zone" in q or "location" in q:
            return "zone"
        if any(token in q for token in ["congest", "overload", "saturat"]):
            return "congestion"
        return None

    @staticmethod
    def _is_anomaly_question(question: str) -> bool:
        q = question.lower()
        return any(
            token in q
            for token in [
                "anomaly",
                "abnormal",
                "jamming",
                "jammer",
                "interference",
                "root cause",
                "affected kpi",
                "troubleshooting",
            ]
        )
