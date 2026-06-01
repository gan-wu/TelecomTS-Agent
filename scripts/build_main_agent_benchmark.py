from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.ipc as ipc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.tool_router import TelecomToolRouter  # noqa: E402


DEFAULT_QUOTAS = {
    "deterministic_tool": 1200,
    "regular_network_agent": 500,
    "heavy_load_reasoning": 500,
    "diagnosis_reasoning": 300,
    "multi_tool_composite": 250,
    "rag_knowledge": 150,
    "low_confidence_complex": 100,
}

INTERNAL_TOOL_SPLIT = {
    "deterministic_tool_timeseries": 700,
    "deterministic_tool_network": 500,
}


@dataclass
class Reservoir:
    quota: int
    rng: random.Random
    items: list[dict[str, Any]] = field(default_factory=list)
    seen: int = 0

    def add(self, item: dict[str, Any]) -> None:
        self.seen += 1
        if len(self.items) < self.quota:
            self.items.append(item)
            return
        index = self.rng.randrange(self.seen)
        if index < self.quota:
            self.items[index] = item


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build one integrated 3000-sample benchmark that showcases Tool Calling, Agent routing, RAG, and cascade behavior."
    )
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "benchmark_main_agent_3000.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=3000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    quotas = dict(DEFAULT_QUOTAS)
    scale = args.max_samples / sum(quotas.values())
    if scale != 1:
        quotas = {key: max(1, round(value * scale)) for key, value in quotas.items()}
        delta = args.max_samples - sum(quotas.values())
        quotas["deterministic_tool"] += delta

    internal_quotas = dict(quotas)
    if quotas.get("deterministic_tool") == DEFAULT_QUOTAS["deterministic_tool"]:
        internal_quotas.pop("deterministic_tool")
        internal_quotas.update(INTERNAL_TOOL_SPLIT)

    samplers = {
        group: Reservoir(quota=quota, rng=random.Random(args.seed + idx + 1))
        for idx, (group, quota) in enumerate(internal_quotas.items())
    }

    data_dir = Path(args.data_dir)
    arrow_paths = sorted(data_dir.glob("telecom_ts-full-*.arrow"))
    if not arrow_paths:
        raise FileNotFoundError(f"No Arrow shards found in {data_dir}")

    row_counts = 0
    native_qna_counts = Counter()
    for file_idx, path in enumerate(arrow_paths):
        with ipc.open_stream(path) as reader:
            row_offset = 0
            for batch in reader:
                rows = batch.to_pylist()
                for local_idx, row in enumerate(rows):
                    row_idx = row_offset + local_idx
                    row_counts += 1
                    context = build_context(row)
                    qna = row.get("QnA") or {}
                    labels = context.get("labels", {})

                    for q_idx, item in enumerate(qna.get("timeseries") or []):
                        if not valid_qna(item):
                            continue
                        native_qna_counts["timeseries"] += 1
                        entry = native_entry(
                            case_id=f"ts_{file_idx}_{row_idx}_{q_idx}",
                            item_type="timeseries",
                            context=context,
                            question=item["q"],
                            answer=item["a"],
                            group="deterministic_tool",
                            skill_tags=["tool_calling", "kpi_stat_tool", "zero_token_answer"],
                            expected_route="tool",
                            difficulty="easy",
                            route_policy="tool_first",
                            origin="native_qna",
                        )
                        if "deterministic_tool_timeseries" in samplers:
                            samplers["deterministic_tool_timeseries"].add(entry)
                        else:
                            samplers["deterministic_tool"].add(entry)

                    for q_idx, item in enumerate(qna.get("network") or []):
                        if not valid_qna(item):
                            continue
                        native_qna_counts["network"] += 1
                        question = item["q"]
                        base_id = f"net_{file_idx}_{row_idx}_{q_idx}"
                        if is_heavy_load_question(question):
                            samplers["heavy_load_reasoning"].add(
                                native_entry(
                                    case_id=f"heavy_{base_id}",
                                    item_type="network",
                                    context=context,
                                    question=question,
                                    answer=item["a"],
                                    group="heavy_load_reasoning",
                                    skill_tags=["analyst_agent", "solver_agent", "llm_reasoning", "budget_routing"],
                                    expected_route="flash",
                                    difficulty="hard",
                                    route_policy="agent_first",
                                    origin="native_qna",
                                )
                            )
                        elif is_tool_routable_network(question):
                            tool_group = (
                                "deterministic_tool_network"
                                if "deterministic_tool_network" in samplers
                                else "deterministic_tool"
                            )
                            samplers[tool_group].add(
                                native_entry(
                                    case_id=base_id,
                                    item_type="network",
                                    context=context,
                                    question=question,
                                    answer=item["a"],
                                    group="deterministic_tool",
                                    skill_tags=["tool_calling", "label_tool", "zero_token_answer"],
                                    expected_route="tool",
                                    difficulty="easy",
                                    route_policy="tool_first",
                                    origin="native_qna",
                                )
                            )
                            samplers["regular_network_agent"].add(
                                native_entry(
                                    case_id=f"agent_{base_id}",
                                    item_type="network",
                                    context=context,
                                    question=question,
                                    answer=item["a"],
                                    group="regular_network_agent",
                                    skill_tags=["analyst_agent", "solver_agent", "local_first_routing"],
                                    expected_route="local_or_flash",
                                    difficulty="medium",
                                    route_policy="agent_first",
                                    origin="native_qna",
                                )
                            )

                    maybe_add_derived_cases(
                        samplers=samplers,
                        file_idx=file_idx,
                        row_idx=row_idx,
                        row=row,
                        context=context,
                        labels=labels,
                        rng=rng,
                    )
                row_offset += batch.num_rows

    records: list[dict[str, Any]] = []
    for group, sampler in samplers.items():
        if len(sampler.items) < sampler.quota:
            raise RuntimeError(f"Not enough samples for {group}: {len(sampler.items)} / {sampler.quota}")
        records.extend(sampler.items[: sampler.quota])

    rng.shuffle(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "output": str(output),
        "seed": args.seed,
        "rows_scanned": row_counts,
        "native_qna_counts": dict(native_qna_counts),
        "target_quotas": quotas,
        "internal_quotas": internal_quotas,
        "actual_groups": dict(Counter(item["benchmark_group"] for item in records)),
        "actual_expected_routes": dict(Counter(item["expected_route"] for item in records)),
        "actual_route_policies": dict(Counter(item["route_policy"] for item in records)),
        "actual_types": dict(Counter(item["type"] for item in records)),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def valid_qna(item: Any) -> bool:
    return isinstance(item, dict) and bool(item.get("q")) and bool(item.get("a"))


def build_context(row: dict[str, Any]) -> dict[str, Any]:
    anomalies = row.get("anomalies") or {}
    return {
        "labels": row.get("labels") or {},
        "statistics": row.get("statistics") or {},
        "anomalies": {
            "exists": anomalies.get("exists"),
            "type": anomalies.get("type"),
            "anomaly_duration": anomalies.get("anomaly_duration"),
            "affected_kpis": row.get("affected_kpis") or [],
            "troubleshooting_tickets": row.get("troubleshooting_tickets") or "",
        },
        "description": row.get("description") or "",
    }


def native_entry(
    case_id: str,
    item_type: str,
    context: dict[str, Any],
    question: str,
    answer: str,
    group: str,
    skill_tags: list[str],
    expected_route: str,
    difficulty: str,
    route_policy: str,
    origin: str,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "type": item_type,
        "context": context,
        "question": question,
        "ground_truth": answer,
        "benchmark_group": group,
        "skill_tags": skill_tags,
        "expected_route": expected_route,
        "difficulty": difficulty,
        "route_policy": route_policy,
        "origin": origin,
    }


def derived_entry(
    case_id: str,
    item_type: str,
    context: dict[str, Any],
    question: str,
    answer: str,
    group: str,
    skill_tags: list[str],
    expected_route: str,
    difficulty: str,
    route_policy: str,
    eval_type: str,
) -> dict[str, Any]:
    item = native_entry(
        case_id=case_id,
        item_type=item_type,
        context=context,
        question=question,
        answer=answer,
        group=group,
        skill_tags=skill_tags,
        expected_route=expected_route,
        difficulty=difficulty,
        route_policy=route_policy,
        origin="derived_from_case",
    )
    item["eval_type"] = eval_type
    return item


def is_heavy_load_question(question: str) -> bool:
    q = question.lower()
    return "heavy load" in q and "performing well" in q


def is_tool_routable_network(question: str) -> bool:
    return bool(
        TelecomToolRouter._is_anomaly_question(question)
        or TelecomToolRouter._detect_label(question)
    )


def maybe_add_derived_cases(
    samplers: dict[str, Reservoir],
    file_idx: int,
    row_idx: int,
    row: dict[str, Any],
    context: dict[str, Any],
    labels: dict[str, Any],
    rng: random.Random,
) -> None:
    if labels:
        samplers["multi_tool_composite"].add(build_multi_tool_case(file_idx, row_idx, context, labels))
        samplers["low_confidence_complex"].add(build_low_confidence_case(file_idx, row_idx, context, labels))

    anomalies = context.get("anomalies") or {}
    if anomalies.get("exists"):
        samplers["diagnosis_reasoning"].add(build_diagnosis_case(file_idx, row_idx, context, anomalies))

    if should_add_rag_case(context, rng):
        samplers["rag_knowledge"].add(build_rag_case(file_idx, row_idx, context, rng))


def build_multi_tool_case(file_idx: int, row_idx: int, context: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    answer = (
        f"Application: {labels.get('application', 'Unknown')}; "
        f"Zone: {labels.get('zone', 'Unknown')}; "
        f"Mobility: {labels.get('mobility', 'Unknown')}; "
        f"Congestion: {labels.get('congestion', 'Unknown')}; "
        f"Anomaly: {labels.get('anomaly_present', 'Unknown')}."
    )
    return derived_entry(
        case_id=f"mix_{file_idx}_{row_idx}_0",
        item_type="composite",
        context=context,
        question="Return the application, zone, mobility, congestion, and anomaly status for this session.",
        answer=answer,
        group="multi_tool_composite",
        skill_tags=["tool_calling", "multi_tool", "structured_answer"],
        expected_route="multi_tool",
        difficulty="medium",
        route_policy="multi_tool",
        eval_type="structured_labels",
    )


def build_diagnosis_case(file_idx: int, row_idx: int, context: dict[str, Any], anomalies: dict[str, Any]) -> dict[str, Any]:
    affected = anomalies.get("affected_kpis") or []
    affected_text = ", ".join(affected[:6]) if affected else "unknown KPIs"
    issue = anomalies.get("type") or "Unknown anomaly"
    answer = f"Issue: {issue}. Affected KPIs: {affected_text}."
    return derived_entry(
        case_id=f"diag_{file_idx}_{row_idx}_0",
        item_type="diagnosis",
        context=context,
        question="Based on the KPI summary and incident ticket, identify the main issue type and the key changed KPIs.",
        answer=answer,
        group="diagnosis_reasoning",
        skill_tags=["analyst_agent", "solver_agent", "diagnosis_reasoning", "incident_ticket"],
        expected_route="flash",
        difficulty="hard",
        route_policy="agent_first",
        eval_type="reference_contains",
    )


def build_low_confidence_case(file_idx: int, row_idx: int, context: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    stats = context.get("statistics") or {}
    rsrp_mean = safe_float((stats.get("RSRP") or {}).get("mean"))
    congestion = labels.get("congestion")
    anomaly = labels.get("anomaly_present")
    if congestion == "Yes":
        label = "capacity_limited"
        reason = "congestion is marked as present"
    elif anomaly == "Yes":
        label = "fault_degraded"
        reason = "anomaly is marked as present"
    elif rsrp_mean is not None and rsrp_mean < -105:
        label = "coverage_degraded"
        reason = "RSRP mean is below -105 dBm"
    else:
        label = "healthy_or_light_load"
        reason = "no congestion or anomaly is marked in the labels"
    return derived_entry(
        case_id=f"complex_{file_idx}_{row_idx}_0",
        item_type="complex_network",
        context=context,
        question=(
            "Classify this session as one of capacity_limited, coverage_degraded, "
            "fault_degraded, or healthy_or_light_load, and give one short reason."
        ),
        answer=f"{label}: {reason}.",
        group="low_confidence_complex",
        skill_tags=["signal_router", "cascade_escalation", "budget_routing", "llm_reasoning"],
        expected_route="flash_then_pro_if_needed",
        difficulty="hard",
        route_policy="agent_first",
        eval_type="label_plus_reason",
    )


def should_add_rag_case(context: dict[str, Any], rng: random.Random) -> bool:
    stats = context.get("statistics") or {}
    labels = context.get("labels") or {}
    if labels.get("anomaly_present") == "Yes":
        return True
    ul_bler = safe_float((stats.get("UL_BLER") or {}).get("mean"))
    rsrp = safe_float((stats.get("RSRP") or {}).get("mean"))
    prb_dl = safe_float((stats.get("PRB_Utilization_DL") or {}).get("mean"))
    return bool(
        (ul_bler is not None and ul_bler > 0.05)
        or (rsrp is not None and rsrp < -105)
        or (prb_dl is not None and prb_dl > 60)
        or rng.random() < 0.01
    )


def build_rag_case(file_idx: int, row_idx: int, context: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    templates = [
        (
            "What does high UL_BLER mean for 5G uplink troubleshooting, and which current KPIs should be checked first?",
            "High UL_BLER means uplink block errors and retransmissions; check UL_SNR, UL_MCS, UL_NPRB or uplink PRB utilization, RSRP, and interference or jamming evidence.",
        ),
        (
            "Why does very low RSRP matter in a 5G RAN health check?",
            "Very low RSRP indicates poor radio coverage or edge-cell conditions; it can reduce reliability and should be checked with BLER, SNR, MCS, and PRB utilization.",
        ),
        (
            "How should PRB utilization be interpreted when diagnosing possible network congestion?",
            "High PRB utilization suggests capacity pressure or congestion risk; compare it with throughput, BLER, buffer, packet counts, and the congestion label.",
        ),
        (
            "What telecom evidence helps distinguish jamming or interference from normal traffic bursts?",
            "Jamming or interference is supported by degraded radio quality, BLER/SNR changes, affected KPIs, anomaly labels, and incident-ticket evidence rather than traffic volume alone.",
        ),
    ]
    question, answer = rng.choice(templates)
    return derived_entry(
        case_id=f"rag_{file_idx}_{row_idx}_0",
        item_type="knowledge",
        context=context,
        question=question,
        answer=answer,
        group="rag_knowledge",
        skill_tags=["rag", "chroma", "bm25", "reranker", "solver_agent"],
        expected_route="rag_flash",
        difficulty="medium",
        route_policy="rag_first",
        eval_type="reference_contains",
    )


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
