# Main Agent Experiment Report

## Overall

- rows: 3000
- benchmark rows: 3000
- rows with model calls: 1550
- total model call records: 1850

## Task Metrics

- timeseries: `{"rows": 700, "parseable_rate": null, "mae": 0.0}`
- network: `{"rows": 1500, "accuracy": null}`
- semantic_keyword_recall: `{"rows": 550, "scored_rows": 448, "mean_recall": null}`

## Model Calls

- accepted model counts: `{"deepseek_flash/deepseek-v4-flash": 1269, "local/qwen3.5-9b-q4": 403, "deepseek_pro/deepseek-v4-pro": 178}`
- attempt model counts: `{"deepseek_flash/deepseek-v4-flash": 1550, "local/qwen3.5-9b-q4": 403, "deepseek_pro/deepseek-v4-pro": 281}`

## Group Summary

| Group | Rows | Tool % | RAG % | Critic % | Rows with model | Ranking modes | Accepted models |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| deterministic_tool | 1200 | NA | NA | NA | 0 | {} | {} |
| diagnosis_reasoning | 300 | NA | NA | NA | 300 | {} | {"local/qwen3.5-9b-q4": 336, "deepseek_flash/deepseek-v4-flash": 191, "deepseek_pro/deepseek-v4-pro": 73} |
| heavy_load_reasoning | 500 | NA | NA | NA | 500 | {} | {"deepseek_flash/deepseek-v4-flash": 450, "local/qwen3.5-9b-q4": 3, "deepseek_pro/deepseek-v4-pro": 47} |
| low_confidence_complex | 100 | NA | NA | NA | 100 | {} | {"deepseek_flash/deepseek-v4-flash": 96, "deepseek_pro/deepseek-v4-pro": 2, "local/qwen3.5-9b-q4": 2} |
| multi_tool_composite | 250 | NA | NA | NA | 0 | {} | {} |
| rag_knowledge | 150 | NA | NA | NA | 150 | {"bge_reranker": 150} | {"deepseek_pro/deepseek-v4-pro": 5, "deepseek_flash/deepseek-v4-flash": 145} |
| regular_network_agent | 500 | NA | NA | NA | 500 | {} | {"deepseek_flash/deepseek-v4-flash": 387, "local/qwen3.5-9b-q4": 62, "deepseek_pro/deepseek-v4-pro": 51} |