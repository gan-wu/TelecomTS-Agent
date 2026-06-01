# Data Directory

This GitHub package includes:

- `benchmark.json`: 1000 QA samples used by the original pipeline.
- `benchmark_main_agent_3000.json`: 3000-sample integrated benchmark for Tool/RAG/Agent routing experiments.
- `benchmark_main_agent_3000.summary.json`: sampling quotas and route distribution for the integrated benchmark.
- `dataset_info.json`: dataset metadata.

The raw Arrow shards are intentionally excluded because they are large:

- `telecom_ts-full-00000-of-00002.arrow`
- `telecom_ts-full-00001-of-00002.arrow`

The included JSON benchmarks can run directly. Rebuilding `benchmark_main_agent_3000.json` from scratch requires placing the raw Arrow shards in this directory and running `scripts/build_main_agent_benchmark.py`.
