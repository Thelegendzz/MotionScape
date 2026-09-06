FROM guozile/cosmos-predict2.5:2b-cu128-nccl227 AS cosmos
FROM guozile/wan2.2-i2v-a14b:20260717-cu128 AS wan22
FROM guozile/longcat-video:20260724-cu128 AS longcat
FROM guozile/cogvideox1.5-5b-i2v:20260714-cu128-r2 AS runtime

LABEL org.opencontainers.image.title="MotionScape unified baseline runtime"
LABEL org.opencontainers.image.description="One container image with isolated Cosmos, CogVideoX, Wan2.2, LongCat, and MAGI Python runtimes"

USER root

# Cosmos is a uv environment with an absolute interpreter link. Preserve both
# its validated /workspace prefix and its standalone CPython distribution.
COPY --from=cosmos /opt/python310 /opt/python310
COPY --from=cosmos /workspace /workspace

# CogVideoX remains the base image's default Python environment. Wan2.2 and
# LongCat keep independent package trees so their Transformers/Diffusers/native
# extension versions cannot overwrite CogVideoX or one another.
COPY --from=wan22 /usr/local/lib/python3.10/dist-packages /opt/motionscape/envs/wan22/site-packages
COPY --from=longcat /usr/local/lib/python3.10/dist-packages /opt/motionscape/envs/longcat/site-packages

# MAGI native extensions were compiled for this exact Conda prefix. The build
# supplies it as a named local context:
#   --build-context magi_env=/path/to/validated/MAGI-1-conda/env
COPY --from=magi_env / /opt/motionscape/envs/magi1/

COPY inference/docker/motionscape-runtime /usr/local/bin/motionscape-runtime

RUN mkdir -p /cache/tmp /workspace/CogVideo /workspace/Wan2.2 /workspace/LongCat-Video /workspace/MAGI-1 \
    && chmod 1777 /cache/tmp \
    && /usr/local/bin/motionscape-runtime list

ENV MOTIONSCAPE_CACHE_ROOT=/cache \
    HF_HOME=/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache/xdg \
    CUDA_CACHE_PATH=/cache/cuda \
    TMPDIR=/cache/tmp \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/motionscape-runtime"]
CMD ["help"]
