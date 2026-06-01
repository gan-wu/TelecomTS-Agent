from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.telecom_graph import TelecomLangGraphWorkflow  # noqa: E402
from src.models.model_router import ModelRouterClient  # noqa: E402
from src.rag.embedding_store import EmbeddingConfig, query_chroma  # noqa: E402
from src.tools.telecom_knowledge_tool import KnowledgeSearchConfig, TelecomKnowledgeTool  # noqa: E402


TABLE6_COLUMNS = [
    "stat_min",
    "stat_max",
    "period_min",
    "period_max",
    "trend_acc",
    "traffic_acc",
    "mobility_acc",
    "location_acc",
    "congestion_acc",
]


def extract_number(text: Any) -> float | None:
    if pd.isna(text):
        return None
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text).replace(",", ""))
    if not matches:
        return None
    return float(matches[-1])


def question_kind(question: str) -> str:
    q = str(question).lower()
    if "trend" in q or "decrease" in q or "increase" in q:
        return "trend"
    if "periodic" in q or "period" in q:
        return "periodicity"
    return "statistics"


def question_kpi(question: str) -> str:
    return str(question).split()[-1].strip("?.")


def network_category(question: str) -> str:
    q = str(question).lower()
    if any(token in q for token in ["traffic", "youtube", "twitch", "file", "service", "application"]):
        return "Traffic"
    if any(token in q for token in ["moving", "static", "motion", "still", "stationary", "mobile"]):
        return "Mobility"
    if any(token in q for token in ["zone", "location"]):
        return "Location"
    if any(token in q for token in ["congest", "saturat", "overload", "jamming", "interfere"]):
        return "Congestion"
    return "Other"


def normalize_class(text: Any) -> str:
    t = str(text).lower().strip()
    zone_match = re.search(r"\bzone[\s_-]*([abc])\b", t)
    if zone_match:
        return f"zone_{zone_match.group(1)}"
    if t in {"a", "b", "c"}:
        return f"zone_{t}"
    if "not moving" in t or "no movement" in t or "stationary" in t or "static" in t or "still" in t:
        return "stationary"
    if "changed zones" in t or "movement between zones" in t or "in motion" in t or "moving" in t or "motion" in t or "mobile" in t:
        return "moving"
    if "youtube" in t:
        return "youtube"
    if "twitch" in t:
        return "twitch"
    if "file" in t:
        return "file"
    if any(
        token in t
        for token in [
            "not congest",
            "no congestion",
            "uncongest",
            "not overloaded",
            "no overload",
            "no signs of overload",
            "within normal",
            "normal throughput",
            "traffic flowed normally",
            "operating normally",
            "normal operation",
            "performing well",
            "no anomaly",
            "absent",
            "unaffected by jamming",
            "free of jamming",
            "jammer-free",
            "without jamming",
            "interference-free",
        ]
    ):
        return "no"
    if any(token in t for token in ["congested", "congestion", "overload", "saturation", "heavy load", "yes", "present", "jamming", "jammed", "anomaly"]):
        return "yes"
    if any(token in t for token in ["no", "normal", "absent", "interference-free", "clean"]):
        return "no"
    return t


def table6_metrics(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    ts = df[df["type"] == "timeseries"].copy()
    ts["gt_num"] = ts["ground_truth"].apply(extract_number)
    ts["pred_num"] = ts["final_answer"].apply(extract_number)
    ts["mae"] = (ts["gt_num"] - ts["pred_num"]).abs()
    ts["kind"] = ts["question"].apply(question_kind)
    ts["kpi"] = ts["question"].apply(question_kpi)

    stat_mae = ts[ts["kind"] == "statistics"].groupby("kpi")["mae"].mean()
    period_mae = ts[ts["kind"] == "periodicity"].groupby("kpi")["mae"].mean()
    trend = ts[ts["kind"] == "trend"]

    net = df[df["type"] == "network"].copy()
    net["category"] = net["question"].apply(network_category)
    net["gt_norm"] = net["ground_truth"].apply(normalize_class)
    net["pred_norm"] = net["final_answer"].apply(normalize_class)
    net["correct"] = net["gt_norm"] == net["pred_norm"]
    net_acc = net.groupby("category")["correct"].mean() * 100

    return {
        "path": str(path),
        "rows": int(len(df)),
        "stat_min": float(stat_mae.min()),
        "stat_max": float(stat_mae.max()),
        "period_min": float(period_mae.min()),
        "period_max": float(period_mae.max()),
        "trend_acc": float((trend["gt_num"] == trend["pred_num"]).mean() * 100),
        "traffic_acc": float(net_acc.get("Traffic", float("nan"))),
        "mobility_acc": float(net_acc.get("Mobility", float("nan"))),
        "location_acc": float(net_acc.get("Location", float("nan"))),
        "congestion_acc": float(net_acc.get("Congestion", float("nan"))),
        "tool_rate": float(df["tool_answer_used"].mean() * 100) if "tool_answer_used" in df.columns else None,
        "critic_rate": float(df["critic_triggered"].mean() * 100) if "critic_triggered" in df.columns else None,
        "network_category_counts": net["category"].value_counts().to_dict(),
        "timeseries_kind_counts": ts["kind"].value_counts().to_dict(),
    }


def compact_trace(state: dict[str, Any]) -> list[str]:
    return [str(item.get("node")) for item in state.get("trace", [])]


def run_graph_branches(
    benchmark: Path,
    embedding_device: str,
    reranker_device: str,
    reranker_batch_size: int,
) -> dict[str, Any]:
    client = ModelRouterClient.from_env(mode="local")
    rag_config = replace(
        KnowledgeSearchConfig.from_env(),
        embedding_device=embedding_device,
        reranker_device=reranker_device,
        reranker_batch_size=reranker_batch_size,
    )
    workflow = TelecomLangGraphWorkflow(
        llm_client=client,
        benchmark_path=str(benchmark),
        use_tools=True,
        use_rag=True,
        use_critic=True,
        rag_config=rag_config,
    )
    cases = {
        "numeric_tool": workflow.invoke("ts_0_11164_16", "What is the mean of UL_BLER?"),
        "knowledge_rag": workflow.invoke(
            "ts_0_11164_16",
            "What does high UL_BLER mean for 5G uplink troubleshooting?",
        ),
        "normal_agent": workflow.invoke(
            "ts_0_11164_16",
            "Give a concise health summary for this 5G network session.",
        ),
    }
    return {
        name: {
            "trace": compact_trace(state),
            "final_answer": state.get("final_answer"),
            "tool_answer_used": state.get("tool_answer_used"),
            "tool_calls": state.get("tool_calls"),
            "rag_hit_count": len(state.get("rag_hits") or []),
            "rag_sources": [
                {
                    "source": hit.get("source"),
                    "section": hit.get("section"),
                    "rerank_score": hit.get("rerank_score"),
                }
                for hit in (state.get("rag_hits") or [])[:5]
            ],
            "model_calls": state.get("model_calls"),
            "error": state.get("error"),
        }
        for name, state in cases.items()
    }


def retrieval_experiment(
    query: str,
    embedding_device: str,
    reranker_device: str,
    reranker_batch_size: int,
) -> dict[str, Any]:
    embedding_config = EmbeddingConfig(
        model_name="BAAI/bge-m3",
        batch_size=1,
        max_length=1536,
        use_fp16=False,
        return_sparse=False,
        devices=embedding_device,
    )
    dense = query_chroma(
        persist_dir=PROJECT_ROOT / "knowledge_base" / "chroma",
        collection_name="telecom_knowledge_bge_m3",
        query=query,
        config=embedding_config,
        top_k=5,
    )
    dense_hits = []
    for rank, (doc_id, metadata, distance) in enumerate(
        zip(dense["ids"][0], dense["metadatas"][0], dense["distances"][0]),
        start=1,
    ):
        dense_hits.append(
            {
                "rank": rank,
                "id": doc_id,
                "source": metadata.get("source"),
                "section": metadata.get("section"),
                "distance": float(distance),
            }
        )

    rag_config = replace(
        KnowledgeSearchConfig.from_env(),
        embedding_device=embedding_device,
        reranker_device=reranker_device,
        reranker_batch_size=reranker_batch_size,
    )
    hybrid = TelecomKnowledgeTool(rag_config).search_telecom_knowledge(query, top_k=5)
    return {
        "query": query,
        "dense_top5": dense_hits,
        "hybrid_counts": {
            "dense": hybrid["dense_count"],
            "bm25": hybrid["bm25_count"],
            "fused": hybrid["fused_count"],
            "final": hybrid["final_count"],
        },
        "hybrid_top5": [
            {
                "rank": index,
                "source": hit.get("source"),
                "section": hit.get("section"),
                "rerank_score": hit.get("rerank_score"),
                "rank_score": hit.get("rank_score"),
                "ranker": hit.get("ranker"),
                "rrf_score": hit.get("rrf_score"),
                "dense_rank": hit.get("dense_rank"),
                "bm25_rank": hit.get("bm25_rank"),
            }
            for index, hit in enumerate(hybrid["hits"], start=1)
        ],
    }


def round_metrics(row: dict[str, Any]) -> dict[str, Any]:
    rounded = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = round(value, 3)
        else:
            rounded[key] = value
    return rounded


def format_retrieval_score(hit: dict[str, Any]) -> str:
    rerank_score = hit.get("rerank_score")
    if isinstance(rerank_score, (int, float)):
        return f"rerank={rerank_score:.4f}"
    rank_score = hit.get("rank_score")
    if isinstance(rank_score, (int, float)):
        return f"{hit.get('ranker') or 'rank'}={rank_score:.4f}"
    rrf_score = hit.get("rrf_score")
    if isinstance(rrf_score, (int, float)):
        return f"rrf={rrf_score:.4f}"
    return "score=NA"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Added Technology Effects Report",
        "",
        "## Table 6 Style Metrics",
        "",
        "| Run | Stat min | Stat max | Period min | Period max | Trend | Traffic | Mobility | Location | Congestion | Tool rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in report["table6"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{metrics['stat_min']:.3f}",
                    f"{metrics['stat_max']:.3f}",
                    f"{metrics['period_min']:.3f}",
                    f"{metrics['period_max']:.3f}",
                    f"{metrics['trend_acc']:.3f}",
                    f"{metrics['traffic_acc']:.3f}",
                    f"{metrics['mobility_acc']:.3f}",
                    f"{metrics['location_acc']:.3f}",
                    f"{metrics['congestion_acc']:.3f}",
                    "NA" if metrics["tool_rate"] is None else f"{metrics['tool_rate']:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## LangGraph Branches", ""])
    for name, branch in report["branches"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- trace: `{' -> '.join(branch['trace'])}`",
                f"- final_answer: {branch['final_answer']}",
                f"- rag_hit_count: {branch['rag_hit_count']}",
                f"- model_calls: `{json.dumps(branch['model_calls'], ensure_ascii=False)}`",
                "",
            ]
        )
    lines.extend(["## Retrieval Experiments", ""])
    for item in report["retrieval"]:
        lines.extend([f"### {item['query']}", "", "**Dense top5**", ""])
        for hit in item["dense_top5"]:
            lines.append(f"- #{hit['rank']} {hit['source']} | {hit['section']} | distance={hit['distance']:.4f}")
        lines.extend(["", "**Hybrid top5**", ""])
        for hit in item["hybrid_top5"]:
            lines.append(
                f"- #{hit['rank']} {hit['source']} | {hit['section']} | "
                f"{format_retrieval_score(hit)} | dense={hit['dense_rank']} | bm25={hit['bm25_rank']}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the effect of added Agent/RAG technologies.")
    parser.add_argument("--previous", default="results/predictions_deepseekV3.2.csv")
    parser.add_argument("--current", default="results/current_local_tool_rag_fixed.csv")
    parser.add_argument("--benchmark", default="data/benchmark.json")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--reranker-device", default="cpu")
    parser.add_argument("--reranker-batch-size", type=int, default=2)
    parser.add_argument("--output-json", default="results/added_tech_effect_report.json")
    parser.add_argument("--output-md", default="results/added_tech_effect_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    previous_path = PROJECT_ROOT / args.previous
    current_path = PROJECT_ROOT / args.current
    benchmark_path = PROJECT_ROOT / args.benchmark

    report = {
        "table6": {
            "previous": table6_metrics(previous_path),
            "current": table6_metrics(current_path),
        },
        "branches": run_graph_branches(
            benchmark=benchmark_path,
            embedding_device=args.embedding_device,
            reranker_device=args.reranker_device,
            reranker_batch_size=args.reranker_batch_size,
        ),
        "retrieval": [
            retrieval_experiment(
                "What does high UL_BLER mean for 5G uplink troubleshooting?",
                args.embedding_device,
                args.reranker_device,
                args.reranker_batch_size,
            ),
            retrieval_experiment(
                "UL_BLER PRB_Utilization_DL gNB jamming KPI troubleshooting",
                args.embedding_device,
                args.reranker_device,
                args.reranker_batch_size,
            ),
        ],
    }

    output_json = PROJECT_ROOT / args.output_json
    output_md = PROJECT_ROOT / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, output_md)

    print("Table 6 metrics:")
    for name, metrics in report["table6"].items():
        print(name, json.dumps(round_metrics(metrics), ensure_ascii=False))
    print(f"JSON: {output_json.relative_to(PROJECT_ROOT)}")
    print(f"Markdown: {output_md.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
