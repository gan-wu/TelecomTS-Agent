import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.tools.tool_router import TelecomToolRouter


def load_cases(benchmark_path: str) -> list[dict]:
    with open(benchmark_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_case(cases: list[dict], contains: str) -> dict:
    needle = contains.lower()
    for item in cases:
        if needle in item.get("question", "").lower():
            return item
    raise RuntimeError(f"No benchmark case matched: {contains}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for TelecomTS Tool Calling.")
    parser.add_argument("--benchmark", default=os.path.join(PROJECT_ROOT, "data", "benchmark.json"))
    args = parser.parse_args()

    router = TelecomToolRouter(args.benchmark)
    cases = load_cases(args.benchmark)

    examples = [
        pick_case(cases, "variance"),
        pick_case(cases, "trend"),
        pick_case(cases, "period"),
        pick_case(cases, "active"),
        pick_case(cases, "zone"),
        pick_case(cases, "movement"),
    ]

    for item in examples:
        result = router.route(item["id"], item["question"])
        print("=" * 80)
        print(f"case_id: {item['id']}")
        print(f"question: {item['question']}")
        print(f"ground_truth: {item['ground_truth']}")
        print(f"tool_answer: {result.answer}")
        print(f"tool_calls: {json.dumps(result.tool_calls, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
