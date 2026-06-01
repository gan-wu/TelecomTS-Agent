# Resume Project Summary

## Project Name

面向 5G 网络的大模型多智能体运维问答系统

## Resume Version

**面向 5G 网络的大模型多智能体运维问答系统**  
`Python / LangGraph / Tool Calling / RAG / Chroma / BGE-M3 / BM25 / RRF / Reranker / Qwen / DeepSeek`

- 负责项目全链路实现，构建面向 5G 网络 KPI、异常检测与运维问答的多智能体系统，设计 `ToolRouter / RAG Tool / Analyst / Solver / Critic` 工作流，覆盖时序统计、网络状态推断、异常诊断和通信知识问答。
- 设计并实现 Tool Calling 工具路由，将显式统计类问题转为本地确定性计算，支持 KPI 均值、方差、趋势、周期、拥塞等任务，降低大模型在长上下文数值计算中的幻觉和格式错误。
- 基于 LangGraph 重构 Agent 数据流，实现 `load_case -> tool_router -> KPI Tool / RAG Tool / Analyst -> Solver -> Critic` 的可追踪编排，并补充 smoke test 验证各分支是否按预期流转。
- 构建通信领域 RAG 知识库，整合 TelecomTS paper、srsRAN、OpenAirInterface、O-RAN SC 文档及自建 KPI/异常排障术语表，采用结构分块、语义合并、上下文增强和 metadata 溯源。
- 实现高质量混合检索链路：`BGE-M3 embedding + Chroma dense retrieval + BM25 + RRF + bge-reranker-v2-m3`，增强 `UL_BLER`、`PRB_Utilization_DL`、`gNB`、`Jamming` 等通信精确术语的召回与排序。
- 接入本地量化 Qwen 模型和 DeepSeek V4 Flash/Pro API，设计 Signal-driven Router + Cascade Escalation，根据任务类型、RAG 置信度、prompt 长度、格式敏感性和输出质量动态选择模型。
- 实现 Budget-Aware Agent Routing，按任务动态控制输出 token 和 RAG 证据长度；在 3000 条综合主实验中，显式工具/多工具路由 1450 条，RAG 路由 150 条，Agent 推理路由 1400 条。
- 完成公平性检查，推断型网络标签题不直接读取 benchmark labels，LLM prompt 中隐藏标签字段，避免把答案泄露给模型；最新 3000 条主实验中 TimeSeries MAE 为 `0.0000`，Network QA Accuracy 为 `80.53%`。

## Interview Explanation

一句话介绍：

> 这是一个面向 5G 网络运维问答的 Agent 系统，我把原来的多智能体问答项目升级成了 Tool Calling + LangGraph + RAG + 本地模型部署 + 智能模型路由的完整工程链路。

面试官问“你做了什么”：

> 我主要做了五块：第一是把显式 KPI 统计问题改成工具调用，避免大模型算错；第二是用 LangGraph 把 Agent 流程显式编排出来；第三是构建通信领域 RAG 知识库，加入 Chroma、BM25、RRF 和 reranker；第四是接入本地 Qwen 和 DeepSeek Flash/Pro，做信号驱动的模型升级路由；第五是做预算感知调度，控制 token 成本和 RAG 证据长度。

面试官问“为什么有效”：

> 因为 benchmark 里一部分问题是均值、方差、周期、趋势这类结构化任务，让大模型直接从长上下文里算数很容易出错，而 Tool Calling 可以确定性计算。通信知识问答则需要外部文档支撑，所以用 RAG 提供证据。对于真正需要推断的网络标签题，我没有用工具直接读标签，而是保留 Agent 推理，这样实验更公平，也更接近真实场景。

面试官问“怎么省钱/提速”：

> 我做了两层优化：一层是工具优先，能确定性计算的问题不调模型；另一层是预算感知路由，高置信 RAG 给少量证据优先走 Flash，低置信或格式修正再升级 Pro，同时用本地 Qwen 承接部分简单分析任务。
