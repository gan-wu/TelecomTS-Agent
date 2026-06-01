# TelecomTS-Agent: 5G Network Multi-Agent QA System

面向 5G 网络运维场景的多智能体问答系统。项目基于 TelecomTS benchmark，重点展示 Agent 工作流、Tool Calling、RAG、向量检索、本地大模型部署、线上模型路由和 token 成本控制能力。

## Project Summary

本项目不是单纯 prompt 包装，而是把 5G KPI 问答拆成不同执行路径：

- 结构化统计问题由 ToolRouter 调用本地确定性 KPI 工具，避免大模型算数不稳定。
- 通信知识问题走 RAG Tool，使用 Chroma + BM25 + RRF + bge-reranker-v2-m3 检索证据后再回答。
- 推断型网络问题保留 Agent 推理，隐藏 benchmark 标签，避免把答案直接泄露给模型。
- 模型调用支持本地 Qwen OpenAI-compatible 服务、DeepSeek V4 Flash、DeepSeek V4 Pro。
- 路由策略已加入 Signal-driven Router + Cascade Escalation 和 Budget-Aware Agent Routing。

## Highlights

- **Agent Workflow**: `ToolRouter / RAG Tool / Analyst / Solver / Critic` 多节点流水线，并提供 LangGraph 编排版本。
- **Tool Calling**: 支持 KPI 均值、方差、趋势、周期、拥塞等结构化问题的确定性工具调用。
- **RAG Pipeline**: `BGE-M3 embedding -> Chroma dense retrieval + BM25 -> RRF fusion -> bge-reranker-v2-m3 rerank`。
- **Model Routing**: 高置信 RAG 优先 Flash，低置信 RAG 增加证据后 Flash，不通过再 Pro；Critic 修正优先 Pro。
- **Budget Control**: 按任务类型动态控制 `max_tokens` 和 RAG 证据长度，减少无效 token。
- **Local Model Deployment**: 支持 llama.cpp 本地 Qwen3.5-9B-Q4 OpenAI-compatible API。
- **Fair Evaluation**: 推断题不直接读取标签；`context_builder` 已隐藏 benchmark labels，避免答案泄露。

## Project Structure

```text
.
├── src/
│   ├── agents/              # Analyst, Solver, Critic agents
│   ├── graph/               # LangGraph workflow
│   ├── models/              # model router and token budget policy
│   ├── prompts/             # YAML prompts
│   ├── rag/                 # chunking, embedding, Chroma, hybrid retrieval
│   ├── tools/               # Tool Calling, KPI tools, RAG tool, LLM client
│   └── main_pipeline.py     # batch benchmark pipeline
├── scripts/                 # smoke tests, benchmark build, index build, analysis
├── data/
│   ├── benchmark.json       # 1000-sample benchmark
│   ├── benchmark_main_agent_3000.json
│   ├── benchmark_main_agent_3000.summary.json
│   └── dataset_info.json
├── knowledge_base/
│   ├── source_docs/         # curated telecom/RAG source documents
│   ├── chunks/              # contextual RAG chunks
│   ├── manifests/           # source provenance and extraction metadata
│   └── retrieval/           # saved retrieval smoke outputs
├── results/                 # selected experiment outputs
├── docs/                    # paper PDF and interview notes
├── requirements.txt
└── .env.example
```

Raw Arrow shards, generated Chroma indexes, virtual environments, caches, and real API keys are intentionally excluded.

## Dataset Information

Included data:

- `data/benchmark.json`: 1000 QA samples used by the original project workflow.
- `data/benchmark_main_agent_3000.json`: 3000-sample integrated benchmark for showing Tool, RAG, Agent routing, and model escalation.
- `data/benchmark_main_agent_3000.summary.json`: group quotas and route distribution for the 3000-sample benchmark.
- `data/dataset_info.json`: dataset metadata.

Excluded data:

- `data/telecom_ts-full-00000-of-00002.arrow`
- `data/telecom_ts-full-00001-of-00002.arrow`

The included JSON benchmarks can run directly. Rebuilding the 3000-sample benchmark from scratch requires placing the raw Arrow shards under `data/`.

## Environment

Recommended:

- Python 3.10+
- CUDA GPU optional, recommended for BGE-M3 embedding and reranker
- Local OpenAI-compatible model server optional
- DeepSeek API key optional

Install dependencies in Windows cmd:

```cmd
cd /d D:\WG\python_code\TelecomTS_QA_github
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

When HuggingFace downloads are slow:

```cmd
set HF_ENDPOINT=https://hf-mirror.com
```

Configure API keys by environment variables. Do not commit real keys.

```cmd
set DEEPSEEK_API_KEY=your_api_key_here
set DEEPSEEK_BASE_URL=https://api.deepseek.com
set DEEPSEEK_FLASH_MODEL=deepseek-v4-flash
set DEEPSEEK_PRO_MODEL=deepseek-v4-pro
```

## Optional Local Qwen Server

The latest 3000-sample run used a local llama.cpp Qwen server first. The following command is the actual Windows cmd command used on this machine; replace the paths if your model directory is different.

```cmd
G:\Qwen3.5-9B_Q8_0\llama-cpp\llama-server.exe -m "G:\Qwen3.5-9B_Q8_0\Qwen3.5-9B-Q4_K_M.gguf" -a qwen3.5-9b-q4 --host 0.0.0.0 --port 8080 --jinja --reasoning-format deepseek -ngl 30 -fa on -c 8192 -n 1024 -np 4 -b 256 -ub 256 --cache-type-k q4_0 --cache-type-v q4_0 --cache-ram 0 --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 -t 4
```

Then set the local backend variables in the experiment terminal:

```cmd
set LOCAL_BASE_URL=http://127.0.0.1:8080/v1
set LOCAL_MODEL=qwen3.5-9b-q4
set LOCAL_API_KEY=EMPTY
```
## Quick Smoke Tests

No LLM required:

```cmd
.\.venv\Scripts\python.exe scripts\smoke_tool_calling.py
.\.venv\Scripts\python.exe scripts\smoke_signal_router.py
.\.venv\Scripts\python.exe scripts\smoke_budget_router.py
```

Local model connectivity:

```cmd
.\.venv\Scripts\python.exe scripts\smoke_local_model.py
```

RAG retrieval requires a Chroma index. Rebuild it first if `knowledge_base/chroma/` does not exist.

```cmd
set HF_ENDPOINT=https://hf-mirror.com
.\.venv\Scripts\python.exe scripts\build_chroma_index.py ^
  --reset ^
  --batch-size 1 ^
  --max-length 1536 ^
  --device cuda:0 ^
  --use-fp16
```

RAG smoke test:

```cmd
.\.venv\Scripts\python.exe scripts\smoke_rag_tool.py ^
  --query "What does high UL_BLER mean for 5G uplink troubleshooting?" ^
  --top-k 5 ^
  --embedding-device cuda:0 ^
  --reranker-device cuda:0 ^
  --reranker-batch-size 2
```

If GPU memory is tight, replace `cuda:0` with `cpu`.

## Main Experiment

Full 3000-sample hybrid experiment, using ToolRouter, RAG, local Qwen, DeepSeek Flash/Pro, signal-driven routing, and budget-aware routing.

The latest reported run was executed from the original working folder `D:\WG\python_code\TelecomTS_QA`. For this GitHub display package, use `D:\WG\python_code\TelecomTS_QA_github` as the working directory after dependencies and optional Chroma index are prepared.

Actual command used for the latest run:

```cmd
cd /d D:\WG\python_code\TelecomTS_QA

.\.venv\Scripts\python.exe src\main_pipeline.py ^
  --backend hybrid ^
  --benchmark data\benchmark_main_agent_3000.json ^
  --output results\main_agent_3000_hybrid_fair_rag_fast.csv ^
  --workers 6 ^
  --local-max-concurrency 4 ^
  --interval 0.03 ^
  --analyst-mode auto ^
  --rag-embedding-device cuda:0 ^
  --rag-reranker-device cuda:0 ^
  --rag-reranker-batch-size 2
```

GitHub package equivalent:

```cmd
cd /d D:\WG\python_code\TelecomTS_QA_github

.\.venv\Scripts\python.exe src\main_pipeline.py ^
  --backend hybrid ^
  --benchmark data\benchmark_main_agent_3000.json ^
  --output results\main_agent_3000_hybrid_fair_rag_fast.csv ^
  --workers 6 ^
  --local-max-concurrency 4 ^
  --interval 0.03 ^
  --analyst-mode auto ^
  --rag-embedding-device cuda:0 ^
  --rag-reranker-device cuda:0 ^
  --rag-reranker-batch-size 2
```
Analyze the output:

```cmd
.\.venv\Scripts\python.exe scripts\analyze_main_agent_experiment.py ^
  --results results\main_agent_3000_hybrid_fair_rag_fast.csv ^
  --benchmark data\benchmark_main_agent_3000.json ^
  --output-md results\main_agent_experiment_report.md ^
  --output-json results\main_agent_experiment_report.json
```

## Latest Result Snapshot

Latest 3000-sample run on 2026-06-01:

| Metric | Value |
| --- | ---: |
| Total samples | 3000 |
| Runtime | 36m55s |
| TimeSeries QA samples | 700 |
| TimeSeries MAE | 0.0000 |
| Network QA samples | 1500 |
| Network QA accuracy | 80.53% |
| RAG knowledge samples | 150 |
| Tool/multi-tool routed samples | 1450 |
| Agent routed samples | 1400 |
| DeepSeek Flash request increment | 1550 |
| DeepSeek Pro request increment | 281 |
| Local Qwen final-call count | 403 |
| Approx. API cost increment | 3.09 RMB |

Notes:

- `regular_network_agent` contains label-inference questions. These are not routed to label tools in the fair main experiment.
- ToolRouter handles explicit structured/statistical questions and composite KPI tasks.
- RAG questions use reranked external evidence rather than benchmark labels.
- `analyst-mode auto` uses deterministic KPI summaries for simple KPI contexts, reducing unnecessary LLM analyst calls without writing answer labels into prompts.

## Reproducibility Notes

- The Chroma binary index is excluded from GitHub; rebuild it from `knowledge_base/chunks/knowledge_chunks.jsonl`.
- The raw Arrow dataset is excluded; included JSON benchmarks are enough for running the displayed experiments.
- Reranker and embedding models are downloaded from HuggingFace or mirror on first use.
- Real API keys should be provided through environment variables only.
- Full online evaluation consumes API tokens; run smoke tests first.

## Interview Talking Points

- Built an Agent system with explicit routing and tools, not a single prompt chain.
- Used deterministic tools for KPI statistics to improve stability and reduce cost.
- Implemented industrial RAG stack: Chroma dense retrieval, BM25, RRF fusion, and BGE reranker.
- Integrated local Qwen deployment with online DeepSeek Flash/Pro escalation.
- Added budget-aware routing to control evidence length and output tokens.
- Audited label leakage so fair inference tasks stay fair.



