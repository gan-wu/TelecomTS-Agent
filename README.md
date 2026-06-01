# TelecomTS-Agent：面向 5G 网络运维问答的多智能体系统

本项目面向 5G 网络运维问答场景，基于 TelecomTS benchmark 构建多智能体问答流程，重点展示 Agent 工作流、Tool Calling、RAG 检索增强、向量数据库、本地大模型部署、线上模型路由和 token 成本控制能力。

项目不是单纯的 prompt 包装，而是将问题按类型分流：结构化统计问题由工具确定性计算，通信知识问题走 RAG 检索增强，复杂推断问题交给 Analyst、Solver、Critic 多智能体流程，并根据任务信号调度 DeepSeek Flash、DeepSeek Pro 或本地 Qwen。

## 核心能力

- **多智能体工作流**：包含 ToolRouter、RAG Tool、Analyst、Solver、Critic 等节点，并提供 LangGraph 编排版本。
- **Tool Calling**：对 KPI 均值、方差、趋势、周期、拥塞等结构化问题使用本地工具计算，减少大模型算数不稳定。
- **RAG 检索增强**：使用 BGE-M3 embedding、Chroma dense retrieval、BM25、RRF 融合和 bge-reranker-v2-m3 重排序。
- **模型调度**：支持 DeepSeek Flash、DeepSeek Pro 和本地 Qwen OpenAI-compatible 服务；高置信任务优先低成本模型，必要时升级到 Pro。
- **成本控制**：按任务类型动态控制证据数量、prompt 长度和输出 token，避免无效消耗。
- **公平评测**：推断型网络问题不直接读取 benchmark 标签，避免答案泄露。

## 目录结构

```text
.
├── src/
│   ├── agents/              # Analyst、Solver、Critic
│   ├── graph/               # LangGraph 工作流
│   ├── models/              # 模型路由与 token 预算策略
│   ├── prompts/             # YAML prompt 模板
│   ├── rag/                 # 分块、embedding、Chroma、混合检索
│   ├── tools/               # Tool Calling、KPI 工具、RAG 工具、LLM 客户端
│   └── main_pipeline.py     # 批量实验入口
├── scripts/                 # 索引构建、smoke test、实验分析脚本
├── data/                    # 可直接运行的 benchmark JSON
├── knowledge_base/          # RAG 知识源、chunk、来源记录
├── results/                 # 示例结果与分析报告
├── docs/                    # 论文 PDF、简历项目说明、上传说明
├── requirements.txt
└── .env.example
```

说明：虚拟环境、真实 API key、原始 Arrow 数据、生成的 Chroma 二进制索引不会提交到 GitHub。

## 数据说明

仓库中已包含可直接运行的 JSON benchmark：

- `data/benchmark.json`：1000 条基础评测样本。
- `data/benchmark_main_agent_3000.json`：3000 条主实验样本，用于展示 Tool、RAG、Agent 推理和模型调度能力。
- `data/benchmark_main_agent_3000.summary.json`：3000 条样本的分布说明。
- `data/dataset_info.json`：数据集元信息。

未包含的原始文件：

- `data/telecom_ts-full-00000-of-00002.arrow`
- `data/telecom_ts-full-00001-of-00002.arrow`

如果只是复现实验结果，可以直接使用仓库内 JSON benchmark；如果要从原始数据重新抽样构建 benchmark，需要额外放入 Arrow 文件。

## 环境安装

推荐环境：

- Python 3.10+
- NVIDIA GPU 可选；如果要加速 BGE-M3 embedding 和 reranker，建议使用 CUDA
- DeepSeek API key 可选；运行在线模型实验时需要
- 本地 Qwen 服务可选；只在展示本地部署能力时需要

Windows cmd 中执行：

```cmd
cd /d <你的项目路径>\TelecomTS_QA_github
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果 HuggingFace 下载较慢，可以先设置镜像：

```cmd
set HF_ENDPOINT=https://hf-mirror.com
```

配置 DeepSeek API key，真实 key 只放在本机环境变量中，不要写进代码或提交到 GitHub：

```cmd
set DEEPSEEK_API_KEY=your_api_key_here
set DEEPSEEK_BASE_URL=https://api.deepseek.com
set DEEPSEEK_FLASH_MODEL=deepseek-v4-flash
set DEEPSEEK_PRO_MODEL=deepseek-v4-pro
```

## 第一步：构建 RAG 向量索引

`knowledge_base/chroma/` 是本地生成的 Chroma 向量索引，不上传 GitHub。首次运行主实验前必须先构建。

GPU 版本：

```cmd
.\.venv\Scripts\python.exe scripts\build_chroma_index.py ^
  --reset ^
  --batch-size 1 ^
  --max-length 1536 ^
  --device cuda:0 ^
  --use-fp16
```

CPU 版本，速度较慢，但不需要显卡：

```cmd
.\.venv\Scripts\python.exe scripts\build_chroma_index.py ^
  --reset ^
  --batch-size 1 ^
  --max-length 1536 ^
  --device cpu
```

构建成功后应看到类似输出：

```text
Indexed chunks: 1272
Collection count: 1272
Dense dim: 1024
Chroma dir: knowledge_base\chroma
```

## 第二步：测试 RAG 是否可用

先测试 Chroma 能不能正常返回 TopK：

```cmd
.\.venv\Scripts\python.exe scripts\smoke_chroma_retrieval.py ^
  --device cuda:0 ^
  --use-fp16
```

再测试完整 RAG Tool，包括 Chroma、BM25、RRF 和 reranker：

```cmd
.\.venv\Scripts\python.exe scripts\smoke_rag_tool.py ^
  --query "What does high UL_BLER mean for 5G uplink troubleshooting?" ^
  --top-k 5 ^
  --embedding-device cuda:0 ^
  --reranker-device cuda:0 ^
  --reranker-batch-size 2
```

如果显存紧张，把 `cuda:0` 改成 `cpu`，并去掉 `--use-fp16`。

## 第三步：基础功能 smoke test

这些测试不需要大模型 API，适合面试官快速检查代码结构：

```cmd
.\.venv\Scripts\python.exe scripts\smoke_tool_calling.py
.\.venv\Scripts\python.exe scripts\smoke_signal_router.py
.\.venv\Scripts\python.exe scripts\smoke_budget_router.py
```

## 主实验方案 A：没有本地模型时运行

这是更适合面试官复现的命令：不需要本地 Qwen，只需要 DeepSeek API key。模型调度在 DeepSeek Flash 和 DeepSeek Pro 之间进行，仍然可以展示 Tool Calling、RAG、Agent 工作流、信号路由和 token 预算控制。

```cmd
.\.venv\Scripts\python.exe src\main_pipeline.py ^
  --backend deepseek ^
  --benchmark data\benchmark_main_agent_3000.json ^
  --output results\main_agent_3000_deepseek_only.csv ^
  --workers 6 ^
  --interval 0.03 ^
  --analyst-mode auto ^
  --rag-embedding-device cuda:0 ^
  --rag-reranker-device cuda:0 ^
  --rag-reranker-batch-size 2
```

如果没有 GPU：

```cmd
.\.venv\Scripts\python.exe src\main_pipeline.py ^
  --backend deepseek ^
  --benchmark data\benchmark_main_agent_3000.json ^
  --output results\main_agent_3000_deepseek_only_cpu_rag.csv ^
  --workers 3 ^
  --interval 0.1 ^
  --analyst-mode auto ^
  --rag-embedding-device cpu ^
  --rag-reranker-device cpu ^
  --rag-reranker-batch-size 1
```

## 主实验方案 B：展示本地 Qwen + DeepSeek 混合调度

这个方案用于展示本地大模型部署能力。先启动本地 Qwen OpenAI-compatible 服务，再运行 `hybrid` 模式。

本机示例使用 llama.cpp 启动 Qwen3.5-9B-Q4_K_M：

```cmd
set QWEN_DIR=G:\Qwen3.5-9B_Q8_0
%QWEN_DIR%\llama-cpp\llama-server.exe -m "%QWEN_DIR%\Qwen3.5-9B-Q4_K_M.gguf" -a qwen3.5-9b-q4 --host 0.0.0.0 --port 8080 --jinja --reasoning-format deepseek -ngl 30 -fa on -c 8192 -n 1024 -np 4 -b 256 -ub 256 --cache-type-k q4_0 --cache-type-v q4_0 --cache-ram 0 --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 -t 4
```

另开一个 cmd，设置本地模型环境变量：

```cmd
set LOCAL_BASE_URL=http://127.0.0.1:8080/v1
set LOCAL_MODEL=qwen3.5-9b-q4
set LOCAL_API_KEY=EMPTY
```

先确认本地模型可连通：

```cmd
.\.venv\Scripts\python.exe scripts\smoke_local_model.py
```

再运行完整 hybrid 主实验：

```cmd
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

## 结果分析

主实验结束后生成 CSV 结果文件，可以运行分析脚本：

```cmd
.\.venv\Scripts\python.exe scripts\analyze_main_agent_experiment.py ^
  --results results\main_agent_3000_hybrid_fair_rag_fast.csv ^
  --benchmark data\benchmark_main_agent_3000.json ^
  --output-md results\main_agent_experiment_report.md ^
  --output-json results\main_agent_experiment_report.json
```

如果分析的是 DeepSeek-only 结果，把 `--results` 改成对应文件名：

```cmd
.\.venv\Scripts\python.exe scripts\analyze_main_agent_experiment.py ^
  --results results\main_agent_3000_deepseek_only.csv ^
  --benchmark data\benchmark_main_agent_3000.json ^
  --output-md results\main_agent_deepseek_only_report.md ^
  --output-json results\main_agent_deepseek_only_report.json
```

## 已有结果快照

最近一次 3000 条 hybrid 主实验结果：

| 指标 | 结果 |
| --- | ---: |
| 总样本数 | 3000 |
| 运行时间 | 36m55s |
| 时序 QA 样本 | 700 |
| 时序 MAE | 0.0000 |
| 网络 QA 样本 | 1500 |
| 网络 QA 准确率 | 80.53% |
| RAG 知识类样本 | 150 |
| Tool / multi-tool 路由样本 | 1450 |
| Agent 推理样本 | 1400 |
| 本地 Qwen final-call count | 403 |
| DeepSeek Flash 请求增量 | 1550 |
| DeepSeek Pro 请求增量 | 281 |
| 估算 API 费用增量 | 约 3.09 元 |

说明：

- `regular_network_agent` 中的标签推断题没有走标签工具，避免答案泄露。
- ToolRouter 只处理明确的结构化统计问题和复合 KPI 任务。
- RAG 题使用外部知识证据，不读取 benchmark 标签。
- `analyst-mode auto` 对简单 KPI 场景使用规则摘要，减少不必要的大模型 analyst 调用。

## 常见问题

### 1. 报错：Chroma index not found

说明还没有构建 `knowledge_base/chroma/`。先执行：

```cmd
.\.venv\Scripts\python.exe scripts\build_chroma_index.py ^
  --reset ^
  --batch-size 1 ^
  --max-length 1536 ^
  --device cuda:0 ^
  --use-fp16
```

### 2. 报错：Connection error

如果运行 `hybrid` 或 `local` 模式，通常是本地 Qwen 没启动，或者 `LOCAL_BASE_URL` 端口不对。先跑：

```cmd
.\.venv\Scripts\python.exe scripts\smoke_local_model.py
```

如果没有本地模型，改用 `--backend deepseek`。

### 3. GPU 显存不够

可以降低 reranker batch size：

```cmd
--rag-reranker-batch-size 1
```

或者把 RAG 设备改为 CPU：

```cmd
--rag-embedding-device cpu ^
--rag-reranker-device cpu
```

### 4. 不想消耗线上 API

只运行 smoke test，不运行主实验：

```cmd
.\.venv\Scripts\python.exe scripts\smoke_tool_calling.py
.\.venv\Scripts\python.exe scripts\smoke_signal_router.py
.\.venv\Scripts\python.exe scripts\smoke_budget_router.py
```

## 面试讲解重点

- 这个项目不是单 prompt 问答，而是有 ToolRouter、RAG Tool、Analyst、Solver、Critic 的多智能体流程。
- KPI 统计问题用 Tool Calling 做确定性计算，提升稳定性并减少 token 消耗。
- 通信知识问题用 BGE-M3、Chroma、BM25、RRF 和 reranker 做检索增强。
- 复杂推断问题保留大模型推理，并通过 Critic 校验输出质量。
- 支持本地 Qwen 部署和 DeepSeek Flash/Pro 线上模型调度，能展示本地部署、API 接入和成本控制能力。
