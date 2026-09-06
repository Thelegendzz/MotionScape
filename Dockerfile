FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg

WORKDIR /workspace/MotionScape
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY reconstruction ./reconstruction
COPY motion_stratification ./motion_stratification
COPY annotation ./annotation
COPY prompts ./prompts

ENTRYPOINT ["python"]
