FROM python:3.10-slim

WORKDIR /workspace

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libgomp1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    FLAGS_enable_pir_api=0 \
    FLAGS_enable_pir_in_executor=0 \
    FLAGS_use_mkldnn=0

COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /workspace/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 不在 build 阶段初始化 PaddleOCR，避免触发段错误。
# 代码通过 volume 挂载，改代码无需重建镜像。
ENTRYPOINT ["python"]
