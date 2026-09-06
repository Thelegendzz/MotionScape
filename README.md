# MotionScape

[[Hugging Face]](https://huggingface.co/datasets/thelegendzz/MotionScape)
[[Zenodo]](https://doi.org/10.5281/zenodo.21954342)

MotionScape is a real-world first-person UAV-view benchmark for evaluating world models and future video generation under different conditioning settings and visual-motion intensities. The benchmark contains 228 samples. Each reconstructed sample has 275 consecutive frames standardized to `30000/1001` FPS: frames 1–200 are the observed prefix and frames 201–275 are the continuous future target.

This repository contains the benchmark reconstruction, motion stratification, semantic-annotation prompt, baseline adapters, and the final evaluation implementation used for the paper.

## Data access

The release metadata and annotations are available from the
[MotionScape Hugging Face dataset](https://huggingface.co/datasets/thelegendzz/MotionScape)
and the [MotionScape Zenodo record](https://doi.org/10.5281/zenodo.21954342).
The release contains:

- `source_manifest.json`: source URLs, exact source filenames, finalized start timestamps, and acquisition metadata;
- `annotations/`: one flat `{sample_id, weather, environment, caption}` JSON object per sample;
- `dynamicity_buckets.json`: final motion scores and Low/Medium/High assignments.

The original audiovisual content is not redistributed. Source availability and permitted use remain subject to the source platform and copyright holder. Users are responsible for obtaining and using source videos lawfully.

## Installation

Python 3.10 or newer is recommended. Install the core reconstruction, annotation, and motion-stratification environment with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`ffmpeg` and `ffprobe` are required system executables. The release records `yt-dlp==2026.03.17`; the Python package version in `requirements.txt` is pinned accordingly.

A lightweight core [Dockerfile](Dockerfile) is included. It does not contain baseline-model or Table 4 GPU environments. Before building it, configure Docker's `data-root` or BuildKit cache on a data disk; do not rely on a small system partition.

## Benchmark reconstruction

Download each unique source URL while preserving the exact manifest filename:

```bash
python reconstruction/download_sources.py \
  --manifest /path/to/MotionScape/source_manifest.json \
  --output-dir /path/to/MotionScape_raw
```

The downloader uses yt-dlp's default YouTube client and selects each source stream by the resolution, nominal FPS, and codec recorded in the manifest. It fails instead of silently substituting a different codec when the recorded stream is unavailable.

Optional `--proxy`, `--cookies-from-browser`, `--extractor-args`, `--extra-arg`, `--limit`, and `--dry-run` arguments are available. Existing source videos are never overwritten.

Reconstruct all benchmark representations:

```bash
python reconstruction/reconstruct_benchmark.py \
  --input-json /path/to/MotionScape/source_manifest.json \
  --video-root /path/to/MotionScape_raw \
  --output-dir /path/to/MotionScape_reconstructed
```

For every `sample_id`, the reconstruction entry point:

1. seeks to the finalized `start_s` on the source timeline;
2. temporally resamples to `30000/1001` FPS;
3. extracts the first 275 consecutive output frames;
4. writes the 275-frame image sequence and full video;
5. writes frames 1–200 as the observed-prefix video;
6. writes frames 201–275 as the future-target video;
7. verifies the expected 275/200/75 frame counts.

In the public flat-manifest mode, `end_s` is not used as an extraction boundary. Existing complete samples are skipped by default. `--overwrite` is intentionally explicit and should only be used when replacing known outputs.

## Motion stratification

The final paper protocol uses only future-target frames 201–275:

- source FPS: `30000/1001`;
- temporal sampling: every 3 source frames (25 sampled frames, 24 adjacent pairs);
- resize: 854×480 using area interpolation;
- grayscale Farnebäck dense optical flow with OpenCV parameters `(0.5, 3, 15, 3, 5, 1.2, 0)`;
- frame-pair statistic: spatial 75th percentile of flow magnitude;
- clip score: mean of the 24 frame-pair statistics;
- strata: global 33rd and 66th percentiles.

Run:

```bash
python -m motion_stratification.compute_motion_strata \
  --frames-root /path/to/MotionScape_reconstructed/full_frames \
  --manifest-json /path/to/MotionScape/source_manifest.json \
  --output-json /path/to/dynamicity_buckets_detailed.json
```

The final 228-sample split is Low/Medium/High = 75/75/78. The score is a visual-dynamicity or viewpoint-motion-intensity proxy measured in resized image space; it is not physical camera velocity.

## Semantic annotation

The final prompt is [prompts/semantic_annotation.txt](prompts/semantic_annotation.txt). It describes motion with “the camera viewpoint” or “the onboard camera” and does not imply that the camera carrier is visible.

The annotation client preserves temporal order: conditioning frames 196–200 are sent first, followed by sampled future frames 201–275. The API temperature remains fixed at `0.1`.

```bash
export ANNOTATION_API_KEY='...'
python annotation/generate_semantic_annotations.py \
  --input-path /path/to/MotionScape_reconstructed/full_frames \
  --mode prediction \
  --dynamicity-json /path/to/MotionScape/dynamicity_buckets.json \
  --prompt-file prompts/semantic_annotation.txt \
  --output-dir /path/to/new_annotations \
  --response-json
```

`--api-url` and `--model` expose the OpenAI-compatible endpoint and model name. Do not commit `.env` or API keys.

## Generation prompts

- [CLIP-assisted inspection](prompts/clip_assisted_inspection.txt) records the
  negative prompts used during clip inspection, grouped into subtitle text,
  title/logo/intro, transition effect, and third-person drone categories. It is
  preprocessing provenance and is not supplied to the generation models.
- [Text2World](prompts/text2world.txt)
- [Image2World](prompts/image2world.txt)
- [Video2World task-level](prompts/video2world_task_level.txt)
- [Video2World clip-level](prompts/video2world_clip_level.txt)

All clip-level generation prompts concatenate the released per-sample fields in the order
`weather + environment + caption`. The task-level Video2World prompt uses only the fixed
task prompt and observed video context; it does not read per-sample annotations. CLIPSim
continues to use only `caption`, because that is an evaluation input rather than a generation
prompt.

## Baseline inference

Model adapters are retained intact under `inference/`. Model checkpoints and upstream repositories are not redistributed. Model paths are command-line arguments; the remaining configuration and assets are required by the formal inference entry points.

| Baseline | Formal inference entry point |
|---|---|
| Cosmos-Predict2.5-2B | `inference/cosmos/` integration patch and assets |
| CogVideoX1.5-5B / I2V | `inference/cogvideox/t2v.py`, `inference/cogvideox/i2v.py` |
| Wan2.2 T2V / I2V | `inference/wan/t2v.py`, `inference/wan/i2v.py` |
| LongCat-Video | `inference/longcat/v2w.py` |
| MAGI-1-24B | `inference/magi/v2w.py` |

The temporal inference settings used in the benchmark are:

| Model / task | # Cond. Frames | Condition Sampling | Native FPS | # Output Frames |
|---|---:|---|---:|---:|
| Cosmos-Predict2.5-2B (T2W / I2W / V2W) | 0 / 1 / 5 | V2W: consecutive | 30 | 93 / 92 / 88 |
| CogVideoX1.5-5B (T2W / I2W) | 0 / 1 | N/A | 16 | 41 |
| Wan2.2-A14B (T2W / I2W) | 0 / 1 | N/A | 16 | 40 |
| LongCat-Video (V2W) | 13 | Every 2 source frames | 15 | 40 |
| MAGI-1-24B (V2W) | 32 | Every 2 source frames | 16 | 40 |

For Cosmos-Predict2.5-2B, the inference assets set `num_output_frames=77`, while the model's native temporal configuration produces 93 decoded frames. Prediction-only export retains all 93 frames for T2W, drops the single I2W condition frame to produce 92 frames, and drops the five V2W condition frames to produce 88 frames. The table reports the actual saved output lengths used in evaluation; the Video2World setting is therefore 88 frames.

### Model checkpoints

The repository does not redistribute third-party model weights. The following official Hugging Face repositories were used, pinned by `inference/download_model_weights.py` to the recorded experiment revisions:

| Local directory | Official weight repository | Used for |
|---|---|---|
| `Cosmos-Predict2.5-2B` | [nvidia/Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) | Cosmos T2W/I2W/V2W |
| `CogVideoX1.5-5B` | [zai-org/CogVideoX1.5-5B](https://huggingface.co/zai-org/CogVideoX1.5-5B) | CogVideoX T2W |
| `CogVideoX1.5-5B-I2V` | [zai-org/CogVideoX1.5-5B-I2V](https://huggingface.co/zai-org/CogVideoX1.5-5B-I2V) | CogVideoX I2W |
| `Wan2.2-T2V-A14B` | [Wan-AI/Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) | Wan2.2 T2W |
| `Wan2.2-I2V-A14B` | [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) | Wan2.2 I2W |
| `LongCat-Video` | [meituan-longcat/LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video) | LongCat V2W |
| `MAGI-1-24B` | [sand-ai/MAGI-1](https://huggingface.co/sand-ai/MAGI-1) | MAGI V2W, including 24B, T5, and VAE weights |
| `clip-vit-base-patch32` | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | CLIPSim |

The download utility has a small environment of its own. Its `huggingface-hub==0.36.0` pin matches the recorded Cosmos-Predict2.5 Docker environment and must not be used to replace any model-specific inference environment. Install it separately, choose a model root on a sufficiently large data disk, and download one or more snapshots:

```bash
python -m venv .venv-download
source .venv-download/bin/activate
pip install -r inference/requirements-download.txt

# Show all pinned repositories and revisions without downloading.
python inference/download_model_weights.py \
  --output-root /path/on/data_disk/MotionScape_models \
  --model all \
  --dry-run

# Download selected models. Repeat --model as needed.
python inference/download_model_weights.py \
  --output-root /path/on/data_disk/MotionScape_models \
  --model cogvideox1.5-5b \
  --model cogvideox1.5-5b-i2v

# Download all inference and CLIPSim weights.
python inference/download_model_weights.py \
  --output-root /path/on/data_disk/MotionScape_models \
  --model all
```

Cosmos-Predict2.5-2B is gated: accept its NVIDIA license on Hugging Face and authenticate with `hf auth login` or the `HF_TOKEN` environment variable. Never place access tokens in scripts or command-line arguments. Existing partial downloads are resumed by `huggingface_hub`; the script does not delete model files.

Pass the resulting directories to each inference adapter through `--model-path`. For MAGI, pass the downloaded `MAGI-1-24B` directory; the adapter resolves the configuration's relative `ckpt/magi/24B_base`, `ckpt/t5`, and `ckpt/vae` paths against it without modifying the checked-in configuration. The official I3D-FVD weight is not a Hugging Face checkpoint: `evaluation/i3d_fvd.py` resolves [`deepmind/i3d-kinetics-400/1`](https://tfhub.dev/deepmind/i3d-kinetics-400/1) through TensorFlow Hub and verifies the fixed aggregate SHA-256 recorded in the evaluator.

Do not install one baseline's dependencies over another baseline environment. The model adapters were validated with the following key package versions:

| Environment | PyTorch | Transformers | Diffusers | huggingface-hub |
|---|---|---|---|---|
| Cosmos-Predict2.5-2B | `2.7.1+cu128` | `4.57.1` | `0.35.2` | `0.36.0` |
| CogVideoX1.5-5B / I2V | `2.8.0+cu128` | `5.13.1` | `0.39.0` | `1.23.0` |
| Wan2.2 T2V / I2V | `2.8.0+cu128` | `4.51.3` | `0.39.0` | `0.36.2` |
| LongCat-Video | `2.8.0+cu128` | `4.41.0` | `0.35.1` | `0.36.2` |
| Frame metrics / CLIPSim | `2.8.0+cu128` | `5.13.1` | N/A | `1.26.0` |

MAGI uses Python `3.10.20`, PyTorch `2.11.0+cu130`, Transformers `4.42.3`, Diffusers `0.29.2`, FlashAttention `2.8.4`, and FlashInfer `0.6.12`. It uses a dedicated CUDA 13.0 / Blackwell runtime and must not be merged with the environments above.

Unified baseline-container build and usage instructions are in
[`inference/docker/README.md`](inference/docker/README.md).

Cosmos commit, patch, and inference-asset instructions are in
[`inference/cosmos/README.md`](inference/cosmos/README.md).

## Evaluation

The final evaluator is `evaluation/evaluate_video_metrics.py`. Install the frame-metric environment separately:

```bash
pip install -r evaluation/requirements-frame-metrics.txt
```

The official TensorFlow-Hub I3D FVD implementation has a conflicting isolated environment. The final Docker runtime used NVIDIA TensorFlow `2.17.0+nv25.2`, TensorFlow Hub `0.15.0`, NumPy `1.26.4`, and OpenCV `4.10.0.84`; the requirements file records the corresponding public TensorFlow `2.17.0` package:

```bash
python -m venv .venv-i3d
.venv-i3d/bin/pip install -r evaluation/requirements-i3d-fvd.txt
```

Example directory evaluation, storing clip-level and grouped results:

```bash
python evaluation/evaluate_video_metrics.py \
  --gt-dir /path/to/MotionScape_reconstructed/last_75_videos \
  --pred-dir /path/to/model_outputs \
  --dynamicity-buckets-json /path/to/MotionScape/dynamicity_buckets.json \
  --clipsim-caption-dir /path/to/MotionScape/annotations \
  --clipsim-caption-field caption \
  --clipsim-model openai/clip-vit-base-patch32 \
  --output-json /path/to/metrics_per_clip.json \
  --output-dynamicity-json /path/to/metrics_by_dynamicity.json \
  --allow-frame-count-mismatch \
  --frame-alignment timestamp \
  --frame-metric-size 704x1280 \
  --fvd-mode Video2World \
  --model-name MODEL_NAME \
  --i3d-python .venv-i3d/bin/python
```

### Metric preprocessing

- PSNR, SSIM, LPIPS, and Warping Error are computed at clip level after timestamp-aware alignment at each model's native FPS.
- Ground-truth and predicted RGB frames are independently resized to 1280×704. PSNR/SSIM/Warping Error use `[0,1]`; LPIPS uses `[-1,1]`.
- Warping Error estimates flow between adjacent ground-truth frames, warps the preceding predicted frame, and averages valid-pixel errors over adjacent pairs to obtain a clip score.
- Fixed-75 CLIPSim uses up to 75 consecutive available prediction frames. Motion strata are used only for aggregation and never alter CLIPSim frame selection.
- FVD uniformly samples 75 timestamps over the common real-time duration of each GT/pred pair, converts RGB frames to 224×224 and `[-1,1]`, and extracts one official DeepMind I3D Mean embedding per clip.
- FVD is distribution-level: all clip embeddings in each motion stratum are compared jointly, producing one FVD for Low, Medium, and High. It is not an average of clip-level FVD values.

## Repository layout

```text
MotionScape/
├── reconstruction/
├── motion_stratification/
├── annotation/
├── evaluation/
├── inference/                  # final baseline adapters, required assets, weight downloader, unified Docker
└── prompts/
```

## Citation

When using MotionScape, cite the accompanying MotionScape paper and the dataset record:

```text
MotionScape dataset. Zenodo. https://doi.org/10.5281/zenodo.21954342
```

Paper citation metadata will be added after publication.

## License

Original MotionScape code is released under the [MIT License](LICENSE).
`evaluation/evaluate_video_metrics.py` retains its NVIDIA copyright notice and
Apache-2.0 SPDX license header. Released benchmark metadata, annotations, and
motion-stratification assignments are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source videos and
third-party model code or weights remain governed by their respective owners
and licenses.
