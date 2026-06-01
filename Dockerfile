# 使用轻量级的 Python 3.10 官方镜像作为底座
FROM python:3.10-slim

# 设置容器内的工作目录
WORKDIR /app

# 优先拷贝环境依赖文件，利用 Docker 缓存加速后续 Build
COPY requirements.txt /app/

# 安装 Python 库，使用清华源加速并禁止缓存以减小镜像体积
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 拷贝项目的源文件和提取好的数据
# 注意：不需要拷贝原先几个GB的 .arrow 原始文件，只需提纯后的 data/
COPY src/ /app/src/
COPY data/ /app/data/

# 预留给大模型调用的 API 环境变量 (可以在 run 容器时传入，防止密钥写死在代码里泄漏)
ENV DEEPSEEK_API_KEY=""
ENV QWEN_API_KEY=""
ENV BASE_URL=""

# 定义入口点：容器启动时默认执行我们的多智能体主管线程序
CMD ["python", "src/main_pipeline.py"]
