import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.model_router import ModelRouterClient


def main() -> int:
    client = ModelRouterClient.from_env(
        mode="local",
        local_base_url=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8080/v1"),
        local_model=os.getenv("LOCAL_MODEL", "qwen3.5-9b-q4"),
        local_api_key=os.getenv("LOCAL_API_KEY", "EMPTY"),
    )
    answer = client.query(
        messages=[
            {
                "role": "user",
                "content": "Answer with exactly one short sentence: are you available?",
            }
        ],
        temperature=0.1,
        max_tokens=128,
        max_retries=int(os.getenv("LOCAL_SMOKE_MAX_RETRIES", "0")),
        task="smoke",
    )
    print("answer:", answer)
    print("last_call:", client.last_call)
    if not answer:
        print("Local model smoke failed. Start the local OpenAI-compatible server and retry.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())