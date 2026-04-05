# 1. Bring in a full static ffmpeg/ffprobe build.
FROM mwader/static-ffmpeg:7.0.1 AS ffmpeg

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ffmpeg /ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg /ffprobe /usr/local/bin/ffprobe
RUN cp /usr/local/bin/ffmpeg /opt/conda/bin/ffmpeg || true
RUN cp /usr/local/bin/ffprobe /opt/conda/bin/ffprobe || true

ENV PATH="/usr/local/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    FLAGS_enable_pir_api=0 \
    FLAGS_enable_pir_in_executor=0 \
    FLAGS_use_mkldnn=0

RUN ffmpeg -decoders | grep -i "av1"

COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /workspace/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir torchvision==0.17.2 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir open_clip_torch pillow -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN python - <<'PY'
import torch
import torchvision
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count())
print("torchvision", torchvision.__version__)
from torchvision.models.optical_flow import raft_small
print("raft_small_import_ok", raft_small is not None)
PY

ENTRYPOINT ["python"]
