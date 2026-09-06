# Unified baseline Docker runtime

This image packages the five validated MotionScape baseline Python runtimes in
one container image while preserving their incompatible package versions in
isolated prefixes. It does **not** merge the five package sets into one
`site-packages` directory.

Supported selectors:

| Selector | Baseline runtime |
|---|---|
| `cosmos` | Cosmos-Predict2.5-2B |
| `cogvideox` | CogVideoX1.5-5B / I2V |
| `wan22` | Wan2.2 T2V / I2V |
| `longcat` | LongCat-Video |
| `magi1` | MAGI-1-24B |

## Build

The tested Dockerfile assembles the already validated baseline images and the
validated MAGI Conda prefix. This preserves the custom Cosmos and MAGI native
extensions instead of recompiling them with different ABIs. The exact input
image tags are recorded in the Dockerfile; equivalent images must be available
locally under those tags.

Run the build from the repository root. Keep Docker's `data-root`, the MAGI
build context, and BuildKit storage on a data disk because the resulting image
has a large virtual size.

```bash
docker buildx build --load \
  --build-context magi_env=/path/to/validated/MAGI-1-conda/env \
  -f inference/docker/Dockerfile \
  -t motionscape-baselines:unified-cu128-cu130 \
  .
```

The five runtimes remain isolated inside the image. Their package sets must not
be concatenated and installed into one Python environment.

## Run

Put cache storage on a data disk and select a runtime before the command:

```bash
docker run --rm --gpus all \
  -v /path/on/data_disk/cache:/cache \
  -v /path/to/MotionScape:/workspace/MotionScape:ro \
  motionscape-baselines:unified-cu128-cu130 \
  cogvideox python3 /workspace/MotionScape/inference/cogvideox/t2v.py --help
```

For Wan2.2, LongCat, and MAGI, also mount the matching upstream source checkout
and set `MOTIONSCAPE_MODEL_SOURCE` to its in-container path. Model weights remain
external and should be mounted read-only from a data disk.

Multiple containers created from this one image may run different selectors
concurrently. Run `docker run --rm IMAGE list` to list all selectors.

## Validation record

The final image was built on an NVIDIA RTX PRO 6000 Blackwell system and checked
for all five selectors using:

- exact PyTorch, CUDA, Transformers, Diffusers, and Hugging Face Hub versions;
- CUDA matrix execution and compute capability `(12, 0)`;
- Wan/LongCat/MAGI FlashAttention imports and MAGI FlashInfer import;
- every MotionScape inference entry point's argument parsing;
- upstream pipeline/model-module imports for all five baselines.

The unified runtime changes only environment selection. It does not change any
model, prompt, sampling, temporal, or evaluation setting.
