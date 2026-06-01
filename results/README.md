# Results

Selected outputs retained for GitHub/interview display:

- `main_agent_3000_hybrid_fair_rag_fast.csv`: latest 3000-sample hybrid run with ToolRouter, RAG, local Qwen, DeepSeek Flash/Pro, signal-driven routing, and budget-aware routing.
- `main_agent_experiment_report.md`: concise analysis generated from the latest 3000-sample run.
- `main_agent_experiment_report.json`: machine-readable analysis generated from the latest 3000-sample run.
- `added_tech_effect_report.md`: analysis report for Tool Calling, LangGraph, RAG, Chroma, BM25, RRF, and reranker effects.
- `added_tech_effect_report.json`: machine-readable version of the same report.
- `current_local_tool_rag_fixed.csv`: local Qwen + Tool/RAG benchmark output.
- `predictions_deepseekV3.2.csv`: historical DeepSeek V3.2 baseline output.
- `predictions_deepseekV4_flash_tool_legacy.csv`: legacy run using DeepSeek V4 Flash first with Tool Calling.
- `predictions_deepseekV4_flash_no_tools_legacy.csv`: legacy no-tool ablation run.

New experiment outputs should be reviewed before committing to avoid uploading large temporary CSV files accidentally.

