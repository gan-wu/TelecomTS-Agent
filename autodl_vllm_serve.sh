#!/bin/bash
# ==============================================================================
# Step 4.3 (选做): AutoDL 私有化开源大模型微型部署脚本
# ==============================================================================
# 此脚本用于在 AutoDL (单卡 RTX 4090, 24G 显存) 上起一个高吞吐量 vLLM 服务。
# 目标模型：Qwen2.5-14B-Instruct (阿里云出品的最强开源 14B 模型)
# 
# [面试高光包装]：
# 1. 在生产环境中，我们不能完全依赖云厂商 API（存在数据隐私和出海合规问题）。
# 2. 我们通过 vLLM + AutoDL 实现了 "Edge LLM" (边缘级小大模型) 的高并发接入，
#    作为通信机房私有化部署的 Baseline 对照。
#
# 使用方法：
# 1. 在 AutoDL 开一台 Miniconda (Python 3.10) 实例
# 2. 运行 `source autodl_vllm_serve.sh`
# 3. 您的 main_pipeline.py 只需要把 BASE_URL 换成 http://localhost:8000/v1 即可！
# ==============================================================================

set -e

echo "🚀 开始配置 vLLM 和专属 OpenAI-Compatible 服务器..."

# 1. 使用国内镜像加速下载 vllm
pip install vllm transformers accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 启动服务 
#    - tensor-parallel-size=1 (单卡推理)
#    - max-model-len=4096 (限制输入长度，节省显存以塞下整个14B，通信 QA 不吃几万token的长文本)
#    - dtype=bfloat16 (保持最佳原精度，防浮点塌陷)
#    (下载模型可能需要较长时间，建议挂 screen 或 nohup)
echo "📦 正在拉取阿里巴巴 Qwen2.5-14B-Instruct 权重并启动 vLLM Engine..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name qwen-14b

# 上述命令一旦启动成功，终端将打出 `Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`
# 并在您的 `TelecomTS_QA` 中，只需运行：
# python src/main_pipeline.py --base-url "http://localhost:8000/v1" --model "qwen-14b" --api-key "EMPTY" --output "results/predictions_14b_local.csv"
