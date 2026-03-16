# 1. 提取强大的静态全能 FFmpeg
FROM mwader/static-ffmpeg:7.0.1 AS ffmpeg

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# 2. 安装系统运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgomp1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 3. 复制 FFmpeg 并暴力覆盖所有可能存在的路径！
COPY --from=ffmpeg /ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg /ffprobe /usr/local/bin/ffprobe
# 强行覆盖 conda 默认的残废版 ffmpeg（这一步最致命）
RUN cp /usr/local/bin/ffmpeg /opt/conda/bin/ffmpeg || true
RUN cp /usr/local/bin/ffprobe /opt/conda/bin/ffprobe || true

# 4. 强制提升 /usr/local/bin 的环境变量优先级
ENV PATH="/usr/local/bin:${PATH}"

# 5. 验证：如果此时 grep 找不到 av1，Docker 构建会直接报错停止！
RUN ffmpeg -decoders | grep -i "av1"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    FLAGS_enable_pir_api=0 \
    FLAGS_enable_pir_in_executor=0 \
    FLAGS_use_mkldnn=0

# 6. 安装 Python 依赖
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /workspace/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    
RUN pip install --no-cache-dir open_clip_torch pillow -i https://pypi.tuna.tsinghua.edu.cn/simple

ENTRYPOINT ["python"]