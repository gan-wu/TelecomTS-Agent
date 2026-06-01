from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TELECOM_TERMS = {
    "ul_bler",
    "dl_bler",
    "bler",
    "ul_snr",
    "rsrp",
    "sinr",
    "prb",
    "prb_utilization_dl",
    "prb_utilization_ul",
    "ul_nprb",
    "mcs",
    "ul_mcs",
    "throughput",
    "buffer",
    "congestion",
    "jamming",
    "jammer",
    "interference",
    "uplink",
    "downlink",
    "heavy",
    "load",
    "anomaly",
    "radio",
    "quality",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the integrated Agent main experiment.")
    parser.add_argument("--results", required=True, help="Pipeline CSV output.")
    parser.add_argument("--benchmark", default="data/benchmark_main_agent_3000.json")
    parser.add_argument("--output-md", default="results/main_agent_experiment_report.md")
    parser.add_argument("--output-json", default="results/main_agent_experiment_report.json")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_json_loads(value: Any, default: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def extract_number(text: Any) -> float | None:
    if pd.isna(text):
        return None
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text).replace(",", ""))
    return float(matches[-1]) if matches else None


def normalize_classification(text: Any) -> str:
    if pd.isna(text):
        return ""
    t = str(text).lower().strip()
    zone_match = re.search(r"\bzone[\s_-]*([abc])\b", t)
    if zone_match:
        return zone_match.group(1)
    if t in {"a", "b", "c"}:
        return t
    if any(k in t for k in ["not moving", "no movement", "stationary", "static", "still"]):
        return "stationary"
    if any(k in t for k in ["in motion", "moving", "motion", "mobile", "changed zones"]):
        return "in_motion"
    for app in ["youtube", "twitch", "file"]:
        if app in t:
            return app
    negative = [
        "not congest",
        "no congest",
        "uncongested",
        "not overload",
        "no overload",
        "normal",
        "no anomaly",
        "absent",
        "without jamming",
        "interference-free",
        "no",
    ]
    if any(k in t for k in negative):
        return "no"
    positive = [
        "congested",
        "congestion",
        "overloaded",
        "overload",
        "saturation",
        "heavy load",
        "jamming",
        "jammed",
        "interference",
        "anomaly",
        "present",
        "yes",
    ]
    if any(k in t for k in positive):
        return "yes"
    return t


def extract_terms(text: Any) -> set[str]:
    clean = str(text or "").lower()
    terms = set()
    for token in re.findall(r"[a-z][a-z0-9_]{2,}", clean):
        if token in TELECOM_TERMS or "_" in token:
            terms.add(token)
    for phrase in ["heavy load", "radio quality", "uplink", "downlink"]:
        if phrase in clean:
            terms.add(phrase)
    return terms


def keyword_recall(ground_truth: Any, answer: Any) -> float | None:
    terms = extract_terms(ground_truth)
    if not terms:
        return None
    answer_text = str(answer or "").lower()
    hits = sum(1 for term in terms if term in answer_text)
    return hits / len(terms)


def model_call_stats(df: pd.DataFrame) -> dict[str, Any]:
    accepted_counter: Counter[str] = Counter()
    attempt_counter: Counter[str] = Counter()
    route_reason_counter: Counter[str] = Counter()
    rows_with_model = 0
    total_call_records = 0

    for raw in df.get("model_calls", []):
        calls = safe_json_loads(raw, [])
        if calls:
            rows_with_model += 1
        total_call_records += len(calls)
        for call in calls:
            model_key = f"{call.get('provider', '')}/{call.get('model', '')}".strip("/")
            if model_key:
                accepted_counter[model_key] += 1
            reason = call.get("route_reason")
            if reason:
                route_reason_counter[str(reason)] += 1
            for attempt in call.get("attempts") or []:
                attempt_key = f"{attempt.get('provider', '')}/{attempt.get('model', '')}".strip("/")
                if not attempt_key:
                    attempt_key = str(attempt.get("backend") or "unknown")
                attempt_counter[attempt_key] += 1

    return {
        "rows_with_model": rows_with_model,
        "total_call_records": total_call_records,
        "accepted_model_counts": dict(accepted_counter),
        "attempt_model_counts": dict(attempt_counter),
        "route_reason_counts": dict(route_reason_counter),
    }


def group_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    group_col = "benchmark_group" if "benchmark_group" in df.columns else "type"
    for group, part in df.groupby(group_col, dropna=False):
        model_stats = model_call_stats(part)
        rows.append(
            {
                "group": str(group),
                "rows": int(len(part)),
                "types": dict(Counter(part.get("type", []))),
                "route_policies": dict(Counter(part.get("route_policy", []))),
                "tool_rate": pct(bool_mean(part, "tool_answer_used")),
                "rag_rate": pct(bool_mean(part, "rag_used")),
                "critic_rate": pct(bool_mean(part, "critic_triggered")),
                "ranking_modes": dict(Counter(nonempty_values(part, "rag_ranking_mode"))),
                "rag_buckets": dict(Counter(nonempty_values(part, "rag_confidence_bucket"))),
                "rows_with_model": model_stats["rows_with_model"],
                "accepted_model_counts": model_stats["accepted_model_counts"],
                "attempt_model_counts": model_stats["attempt_model_counts"],
            }
        )
    return rows


def task_metrics(df: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if "type" not in df.columns:
        return metrics

    ts = df[df["type"] == "timeseries"].copy()
    if not ts.empty:
        ts["pred_num"] = ts["final_answer"].apply(extract_number)
        ts["true_num"] = ts["ground_truth"].apply(extract_number)
        valid = ts.dropna(subset=["pred_num", "true_num"])
        metrics["timeseries"] = {
            "rows": int(len(ts)),
            "parseable_rate": pct(len(valid) / max(1, len(ts))),
            "mae": float(mean_absolute_error(valid["true_num"], valid["pred_num"])) if not valid.empty else None,
        }

    net = df[df["type"] == "network"].copy()
    if not net.empty:
        net["pred_norm"] = net["final_answer"].apply(normalize_classification)
        net["true_norm"] = net["ground_truth"].apply(normalize_classification)
        metrics["network"] = {
            "rows": int(len(net)),
            "accuracy": pct(accuracy_score(net["true_norm"], net["pred_norm"])),
        }

    semantic = df[df["type"].isin(["knowledge", "diagnosis", "complex_network"])].copy()
    if not semantic.empty:
        semantic["keyword_recall"] = semantic.apply(
            lambda row: keyword_recall(row.get("ground_truth"), row.get("final_answer")),
            axis=1,
        )
        valid_scores = semantic["keyword_recall"].dropna()
        metrics["semantic_keyword_recall"] = {
            "rows": int(len(semantic)),
            "scored_rows": int(len(valid_scores)),
            "mean_recall": pct(valid_scores.mean()) if not valid_scores.empty else None,
        }
    return metrics


def pct(value: Any) -> float | None:
    if value is None:
        return None


def nonempty_values(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    values = []
    for value in df[column].dropna():
        text = str(value).strip()
        if text:
            values.append(text)
    return values


def bool_mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns or df.empty:
        return None
    values = [to_bool(value) for value in df[column]]
    return sum(values) / len(values)


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        if math.isnan(float(value)):
            return None
        return round(float(value) * 100, 3)
    except (TypeError, ValueError):
        return None


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Main Agent Experiment Report",
        "",
        "## Overall",
        "",
        f"- rows: {report['rows']}",
        f"- benchmark rows: {report.get('benchmark_rows')}",
        f"- rows with model calls: {report['model_calls']['rows_with_model']}",
        f"- total model call records: {report['model_calls']['total_call_records']}",
        "",
        "## Task Metrics",
        "",
    ]
    for name, metrics in report["task_metrics"].items():
        lines.append(f"- {name}: `{json.dumps(metrics, ensure_ascii=False)}`")

    lines.extend(
        [
            "",
            "## Model Calls",
            "",
            f"- accepted model counts: `{json.dumps(report['model_calls']['accepted_model_counts'], ensure_ascii=False)}`",
            f"- attempt model counts: `{json.dumps(report['model_calls']['attempt_model_counts'], ensure_ascii=False)}`",
            "",
            "## Group Summary",
            "",
            "| Group | Rows | Tool % | RAG % | Critic % | Rows with model | Ranking modes | Accepted models |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in report["groups"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["group"],
                    str(row["rows"]),
                    fmt(row["tool_rate"]),
                    fmt(row["rag_rate"]),
                    fmt(row["critic_rate"]),
                    str(row["rows_with_model"]),
                    json.dumps(row["ranking_modes"], ensure_ascii=False),
                    json.dumps(row["accepted_model_counts"], ensure_ascii=False),
                ]
            )
            + " |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def main() -> int:
    args = parse_args()
    results_path = resolve_path(args.results)
    benchmark_path = resolve_path(args.benchmark)
    df = pd.read_csv(results_path)

    benchmark_rows = None
    if benchmark_path.exists():
        benchmark_rows = len(json.loads(benchmark_path.read_text(encoding="utf-8")))

    report = {
        "results": str(results_path),
        "benchmark": str(benchmark_path),
        "rows": int(len(df)),
        "benchmark_rows": benchmark_rows,
        "task_metrics": task_metrics(df),
        "model_calls": model_call_stats(df),
        "groups": group_summary(df),
    }

    output_json = resolve_path(args.output_json)
    output_md = resolve_path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, output_md)

    print(f"Rows: {report['rows']}")
    print(f"Markdown: {output_md.relative_to(PROJECT_ROOT)}")
    print(f"JSON: {output_json.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
