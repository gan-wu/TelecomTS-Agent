import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any


KPI_ALIASES = {
    "rsrp": "RSRP",
    "dl_bler": "DL_BLER",
    "downlink_bler": "DL_BLER",
    "ul_bler": "UL_BLER",
    "uplink_bler": "UL_BLER",
    "dl_mcs": "DL_MCS",
    "ul_mcs": "UL_MCS",
    "ul_nprb": "UL_NPRB",
    "ul_snr": "UL_SNR",
    "tx_bytes": "TX_Bytes",
    "rx_bytes": "RX_Bytes",
    "estimated_ul_buffer": "Estimated_UL_Buffer",
    "prb_utilization_dl": "PRB_Utilization_DL",
    "prb_utilization_ul": "PRB_Utilization_UL",
    "prbs_dl_current": "PRBs_DL_Current",
    "prbs_ul_current": "PRBs_UL_Current",
    "ul_numberofpackets": "UL_NumberOfPackets",
    "dl_numberofpackets": "DL_NumberOfPackets",
}


@dataclass
class ToolCallRecord:
    tool: str
    args: dict[str, Any]
    result: Any
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TelecomToolError(ValueError):
    pass


class TelecomBenchmarkStore:
    """Small indexed view over benchmark.json plus optional raw Arrow lookup."""

    def __init__(self, benchmark_path: str):
        self.benchmark_path = os.path.abspath(benchmark_path)
        self.data_dir = os.path.dirname(self.benchmark_path)
        self.project_root = os.path.dirname(self.data_dir)
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        self.by_id = {item["id"]: item for item in self.items}
        self.raw_ids_by_file: dict[int, set[int]] = defaultdict(set)
        self._raw_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self._loaded_raw_files: set[int] = set()
        for item in self.items:
            parsed = parse_case_id(item.get("id", ""))
            if parsed:
                file_idx, row_idx = parsed
                self.raw_ids_by_file[file_idx].add(row_idx)

    def get_case(self, case_id: str) -> dict[str, Any]:
        item = self.by_id.get(case_id)
        if not item:
            raise TelecomToolError(f"case_id not found in benchmark: {case_id}")
        return item

    def get_context(self, case_id: str) -> dict[str, Any]:
        return self.get_case(case_id).get("context", {})

    def get_kpi_stat(self, case_id: str, kpi: str, metric: str) -> Any:
        canonical_kpi = normalize_kpi(kpi)
        canonical_metric = normalize_metric(metric)
        stats = self.get_context(case_id).get("statistics", {})
        if canonical_metric == "periodicity":
            stored_period = (stats.get(canonical_kpi) or {}).get("periodicity")
            if stored_period == 1:
                return 1
            try:
                return compute_periodicity(self.get_kpi_series(case_id, canonical_kpi))
            except Exception:
                pass

        if canonical_kpi not in stats:
            raise TelecomToolError(f"KPI not found: {canonical_kpi}")
        if canonical_metric not in stats[canonical_kpi]:
            raise TelecomToolError(f"metric not found: {canonical_metric}")
        return stats[canonical_kpi][canonical_metric]

    def get_kpi_series(self, case_id: str, kpi: str) -> list[float]:
        canonical_kpi = normalize_kpi(kpi)
        raw = self._get_raw_row(case_id)
        if not raw:
            raise TelecomToolError(f"raw case not available: {case_id}")
        kpis = raw.get("KPIs") or {}
        if canonical_kpi not in kpis:
            raise TelecomToolError(f"raw KPI not found: {canonical_kpi}")
        return kpis[canonical_kpi]

    def get_network_label(self, case_id: str, label: str) -> Any:
        canonical_label = normalize_label(label)
        labels = self.get_context(case_id).get("labels", {})
        if canonical_label not in labels:
            raise TelecomToolError(f"label not found: {canonical_label}")
        return labels[canonical_label]

    def get_anomaly_info(self, case_id: str) -> dict[str, Any]:
        raw = self._get_raw_row(case_id)
        if raw and isinstance(raw.get("anomalies"), dict):
            return raw["anomalies"]

        labels = self.get_context(case_id).get("labels", {})
        return {
            "exists": labels.get("anomaly_present") == "Yes",
            "type": None,
            "anomaly_duration": None,
            "affected_kpis": [],
            "troubleshooting_tickets": "",
        }

    def get_case_summary(self, case_id: str) -> dict[str, Any]:
        item = self.get_case(case_id)
        context = item.get("context", {})
        return {
            "id": item.get("id"),
            "type": item.get("type"),
            "question": item.get("question"),
            "labels": context.get("labels", {}),
            "available_kpis": sorted((context.get("statistics") or {}).keys()),
        }

    @lru_cache(maxsize=64)
    def _get_raw_row(self, case_id: str) -> dict[str, Any] | None:
        parsed = parse_case_id(case_id)
        if not parsed:
            return None

        file_idx, row_idx = parsed
        arrow_path = os.path.join(
            self.data_dir,
            f"telecom_ts-full-{file_idx:05d}-of-00002.arrow",
        )
        if not os.path.exists(arrow_path):
            return None

        try:
            import pyarrow as pa
        except ImportError:
            return None

        cache_key = (file_idx, row_idx)
        if cache_key in self._raw_cache:
            return self._raw_cache[cache_key]

        if file_idx in self._loaded_raw_files:
            return None

        targets = self.raw_ids_by_file.get(file_idx, {row_idx})
        target_set = set(targets)
        seen = 0
        with pa.ipc.open_stream(arrow_path) as reader:
            for batch in reader:
                batch_start = seen
                batch_end = seen + batch.num_rows
                hits = [idx for idx in target_set if batch_start <= idx < batch_end]
                for target_idx in hits:
                    local_idx = target_idx - batch_start
                    self._raw_cache[(file_idx, target_idx)] = batch.slice(local_idx, 1).to_pylist()[0]
                seen = batch_end

        self._loaded_raw_files.add(file_idx)
        return self._raw_cache.get(cache_key)


def parse_case_id(case_id: str) -> tuple[int, int] | None:
    match = re.match(r"^(?:ts|net)_(\d+)_(\d+)_\d+$", case_id or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def normalize_kpi(text: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    if key in KPI_ALIASES:
        return KPI_ALIASES[key]

    for canonical in KPI_ALIASES.values():
        if key == canonical.lower():
            return canonical
    raise TelecomToolError(f"unknown KPI: {text}")


def normalize_metric(text: str) -> str:
    metric = (text or "").lower().strip()
    if metric in {"mean", "average", "avg"}:
        return "mean"
    if metric in {"variance", "var"}:
        return "variance"
    if metric in {"trend"}:
        return "trend"
    if metric in {"period", "periodicity", "period length", "dominant period"}:
        return "periodicity"
    raise TelecomToolError(f"unknown metric: {text}")


def normalize_label(text: str) -> str:
    label = (text or "").lower().strip()
    if label in {"zone", "location"}:
        return "zone"
    if label in {"application", "app", "traffic", "service"}:
        return "application"
    if label in {"mobility", "movement", "moving", "motion"}:
        return "mobility"
    if label in {"congestion", "overload", "saturation"}:
        return "congestion"
    if label in {"anomaly", "anomaly_present"}:
        return "anomaly_present"
    raise TelecomToolError(f"unknown label: {text}")


def format_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        rounded = round(value, 3)
        if rounded.is_integer():
            return f"{rounded:.3f}"
        return f"{rounded:.3f}"
    return str(value)


def format_period(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.2f}"


def format_kpi_answer(kpi: str, metric: str, value: Any) -> str:
    canonical_kpi = normalize_kpi(kpi)
    canonical_metric = normalize_metric(metric)
    value_text = format_number(value)

    if canonical_metric == "periodicity":
        return f"The period length of {canonical_kpi} is {format_period(value)}"
    if canonical_metric == "trend":
        return f"The trend of {canonical_kpi} is {int(float(value))}"
    return f"The {canonical_metric} of {canonical_kpi} is {value_text}"


def format_label_answer(label: str, value: Any) -> str:
    canonical_label = normalize_label(label)
    value_text = str(value)

    if canonical_label == "application":
        if value_text.lower() == "file":
            value_text = "File download"
        return f"{value_text} was active."
    if canonical_label == "zone":
        if value_text.lower() in {"in motion", "moving"}:
            return "The user was moving across locations."
        return f"Zone {value_text} was the location of the user."
    if canonical_label == "mobility":
        return "The user was moving." if value_text == "Yes" else "The user was stationary."
    if canonical_label == "congestion":
        return "The network was congested." if value_text == "Yes" else "The network was not congested."
    if canonical_label == "anomaly_present":
        return "An anomaly was present." if value_text == "Yes" else "No anomaly was present."
    return value_text


def compute_periodicity(series: list[Any]) -> float | int:
    try:
        import numpy as np
    except ImportError as exc:
        raise TelecomToolError("numpy is required for raw periodicity computation") from exc

    values = np.array(series, dtype=float)
    if values.size < 2:
        return 128

    if np.isnan(values).any():
        mean_value = np.nanmean(values)
        values = np.nan_to_num(values, nan=mean_value)

    centered = values - values.mean()
    if np.allclose(centered, 0):
        return 128

    magnitudes = np.abs(np.fft.rfft(centered))
    if magnitudes.size <= 1:
        return 128
    magnitudes[0] = 0
    dominant_idx = int(np.argmax(magnitudes))
    if dominant_idx <= 0:
        return 128

    period = values.size / dominant_idx
    if abs(period - round(period)) < 1e-9:
        return int(round(period))
    return round(float(period), 2)


def format_anomaly_answer(info: dict[str, Any]) -> str:
    if not info or not info.get("exists"):
        return "No anomaly was present."

    anomaly_type = info.get("type") or "Unknown anomaly"
    affected = info.get("affected_kpis") or []
    affected_text = ", ".join(affected) if affected else "unknown KPIs"
    ticket = (info.get("troubleshooting_tickets") or "").replace("\n", " ").strip()
    if ticket:
        return f"Anomaly present: {anomaly_type}. Affected KPIs: {affected_text}. {ticket}"
    return f"Anomaly present: {anomaly_type}. Affected KPIs: {affected_text}."
