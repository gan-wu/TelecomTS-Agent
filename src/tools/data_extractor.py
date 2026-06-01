import json
import random
from pathlib import Path

import pyarrow as pa


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATHS = [
    PROJECT_ROOT / "data" / "telecom_ts-full-00000-of-00002.arrow",
    PROJECT_ROOT / "data" / "telecom_ts-full-00001-of-00002.arrow",
]
OUTPUT_FILE = PROJECT_ROOT / "data" / "benchmark.json"


def extract_data():
    all_qna_pairs = {"network": [], "timeseries": []}

    for file_idx, filepath in enumerate(RAW_DATA_PATHS):
        print(f"Reading file {file_idx + 1}/2: {filepath} ...")
        if not filepath.exists():
            print(f"Skipped missing raw dataset shard: {filepath}")
            continue

        try:
            with pa.ipc.open_stream(str(filepath)) as reader:
                table = reader.read_all()

            data = table.to_pylist()

            for row_idx, row in enumerate(data):
                context = {
                    "labels": row.get("labels", {}),
                    "statistics": row.get("statistics", {}),
                }

                qna_dict = row.get("QnA") or {}

                network_qnas = qna_dict.get("network") or []
                for i, item in enumerate(network_qnas):
                    if isinstance(item, dict) and "q" in item and "a" in item:
                        all_qna_pairs["network"].append(
                            {
                                "id": f"net_{file_idx}_{row_idx}_{i}",
                                "type": "network",
                                "context": context,
                                "question": item["q"],
                                "ground_truth": item["a"],
                            }
                        )

                ts_qnas = qna_dict.get("timeseries") or []
                for i, item in enumerate(ts_qnas):
                    if isinstance(item, dict) and "q" in item and "a" in item:
                        all_qna_pairs["timeseries"].append(
                            {
                                "id": f"ts_{file_idx}_{row_idx}_{i}",
                                "type": "timeseries",
                                "context": context,
                                "question": item["q"],
                                "ground_truth": item["a"],
                            }
                        )
        except Exception as exc:
            print(f"Failed to process {filepath}. Error: {exc}")

    random.seed(42)

    net_samples = random.sample(all_qna_pairs["network"], min(500, len(all_qna_pairs["network"])))
    ts_samples = random.sample(all_qna_pairs["timeseries"], min(500, len(all_qna_pairs["timeseries"])))

    final_benchmark = net_samples + ts_samples
    random.shuffle(final_benchmark)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(final_benchmark, f, indent=2, ensure_ascii=False)

    print("-" * 40)
    print("Extraction complete")
    print(f"Total available Network QAs: {len(all_qna_pairs['network'])}")
    print(f"Total available TimeSeries QAs: {len(all_qna_pairs['timeseries'])}")
    print(f"Saved {len(net_samples)} Network QAs and {len(ts_samples)} TimeSeries QAs into benchmark.json")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_data()
