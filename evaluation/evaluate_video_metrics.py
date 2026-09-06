# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Evaluate frame-aligned video quality metrics for GT/predicted videos.
Single-video example:
python evaluation/evaluate_video_metrics.py \
    --gt /path/to/gt.mp4 \
    --pred /path/to/pred.mp4 \
    --output-json outputs/video_metrics.json \
    --output-frame-json outputs/video_metrics_frames.json
Directory example:
python evaluation/evaluate_video_metrics.py \
    --gt-dir /path/to/original_videos \
    --pred-dir /path/to/predicted_videos \
    --dynamicity-buckets-json /path/to/dynamicity_buckets.json \
    --output-json outputs/video_metrics_batch.json \
    --output-dynamicity-json outputs/video_metrics_by_dynamicity.json \
    --frame-metric-size 704x1280 \
"""
from __future__ import annotations
import argparse
import copy
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This script requires opencv-python and numpy. "
        "Install evaluation/requirements-frame-metrics.txt first."
    ) from exc
try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This script requires torch. Install "
        "evaluation/requirements-frame-metrics.txt first."
    ) from exc
LOCAL_CLIPSIM_MODEL_CANDIDATES = (
    Path("/dataset/model/MotionScape/clip-vit-base-patch32"),
    Path("/data/model/MotionScape/clip-vit-base-patch32"),
    Path("/model/clip-vit-base-patch32"),
)
def default_clipsim_model() -> str:
    for model_path in LOCAL_CLIPSIM_MODEL_CANDIDATES:
        if (model_path / "config.json").is_file() and (model_path / "preprocessor_config.json").is_file():
            return str(model_path)
    return "openai/clip-vit-base-patch32"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GT/pred video quality metrics frame-by-frame.")
    parser.add_argument("--gt", type=Path, default=None, help="Path to the ground-truth video.")
    parser.add_argument("--pred", type=Path, default=None, help="Path to the predicted video.")
    parser.add_argument("--gt-dir", type=Path, default=None, help="Directory containing ground-truth/original MP4 videos.")
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="Directory containing predicted MP4 videos. Files with an input_clip suffix are skipped.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON summary output path.")
    parser.add_argument(
        "--output-frame-json",
        type=Path,
        default=None,
        help="Optional per-frame JSON output path in single-video mode.",
    )
    parser.add_argument(
        "--output-dynamicity-json",
        type=Path,
        default=None,
        help="Optional JSON output path for directory-mode metrics aggregated by dynamicity bucket.",
    )
    parser.add_argument(
        "--dynamicity-buckets-json",
        type=Path,
        default=None,
        help="JSON file containing samples with sample_id and dynamicity_bucket fields for directory-mode aggregation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device used for frame metrics. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size used for frame-wise GPU metrics such as LPIPS/PSNR/SSIM.",
    )
    parser.add_argument(
        "--frame-metric-size",
        type=str,
        default="704x1280",
        help=(
            "Fixed resolution for PSNR/SSIM/LPIPS/warping metrics. Defaults to `704x1280`, "
            "matching the predicted video size. Use one integer for square resize, such as `512`, "
            "or `HEIGHTxWIDTH`, such as `512x768`."
        ),
    )
    parser.add_argument(
        "--lpips-net",
        type=str,
        default="alex",
        choices=["alex"],
        help="LPIPS is fixed to the standard learned AlexNet implementation.",
    )
    parser.add_argument(
        "--no-fvd",
        action="store_true",
        help="Disable directory-level standard I3D-FVD.",
    )
    parser.add_argument(
        "--i3d-model-path",
        type=Path,
        help=(
            "Downloaded local TF-Hub deepmind/i3d-kinetics-400/1 module. "
            "If omitted, the same fixed model is resolved online."
        ),
    )
    parser.add_argument(
        "--i3d-cache-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "I3D_FVD_CACHE_DIR",
                "/dataset/.cache/i3d_fvd"
                if Path("/dataset").is_dir()
                else ".cache/i3d_fvd",
            )
        ),
        help="Versioned cache for one official I3D embedding per complete video.",
    )
    parser.add_argument(
        "--i3d-python",
        type=Path,
        default=Path(os.environ.get("I3D_FVD_PYTHON", sys.executable)),
        help="Python executable for the isolated TensorFlow I3D-FVD subprocess.",
    )
    parser.add_argument(
        "--i3d-sha256-manifest",
        type=Path,
        help="Required with --i3d-model-path; records every local weight file hash.",
    )
    parser.add_argument(
        "--fvd-mode",
        choices=("Text2World", "Image2World", "Video2World"),
        help="Task mode recorded in standard FVD output.",
    )
    parser.add_argument(
        "--model-name",
        help="Evaluated generation model name recorded in standard FVD output.",
    )
    parser.add_argument("--condition-setting")
    parser.add_argument(
        "--fvd-overall",
        action="store_true",
        help="Also recompute a 228-vs-228 overall FVD; never averages bucket FVDs.",
    )
    parser.add_argument(
        "--allow-frame-count-mismatch",
        action="store_true",
        help="If set, compare only the first min(num_gt_frames, num_pred_frames) frames.",
    )
    parser.add_argument(
        "--frame-alignment",
        choices=("timestamp", "index"),
        default="timestamp",
        help="Pair frames by decoded timestamps (default) or by frame index.",
    )
    parser.add_argument(
        "--prediction-type",
        choices=("auto", "standard", "cogvideox_i2v"),
        default="auto",
        help="auto detects CogVideoX I2V paths; cogvideox_i2v drops its first conditioned frame before metrics.",
    )
    parser.add_argument(
        "--gt-clipsim-only",
        action="store_true",
        help="Only compute CLIPSIM between captions/prompts and GT videos; skips all GT/pred quality metrics.",
    )
    parser.add_argument(
        "--clipsim-caption-json",
        type=Path,
        default=None,
        help="Single-video mode caption annotation JSON. Defaults to --pred with .json if present.",
    )
    parser.add_argument(
        "--clipsim-caption-dir",
        type=Path,
        default=None,
        help="Directory-mode caption annotation JSON root, usually annotations/future_caption.",
    )
    parser.add_argument(
        "--clipsim-caption-field",
        type=str,
        default="caption",
        help="Dot-separated field path for reading the per-video caption annotation.",
    )
    parser.add_argument(
        "--clipsim-model",
        type=str,
        default=default_clipsim_model(),
        help="HuggingFace CLIP model used for CLIPSIM.",
    )
    parser.add_argument(
        "--clipsim-local-files-only",
        action="store_true",
        help="Load the CLIPSIM HuggingFace model from local files only; useful for offline runs.",
    )
    parser.add_argument(
        "--clipsim-batch-size",
        type=int,
        default=16,
        help="Batch size for CLIPSIM frame encoding.",
    )
    parser.add_argument(
        "--clipsim-num-frames",
        type=int,
        default=75,
        help="Maximum number of consecutive CLIPSIM frames (stride 1; default: 75).",
    )
    args = parser.parse_args()
    single_mode = args.gt is not None or args.pred is not None
    directory_mode = args.gt_dir is not None or args.pred_dir is not None
    if single_mode and directory_mode:
        parser.error("Use either --gt/--pred for one pair or --gt-dir/--pred-dir for batch mode, not both.")
    if args.gt_clipsim_only and single_mode and args.gt is None:
        parser.error("--gt-clipsim-only single-video mode requires --gt.")
    if args.gt_clipsim_only and directory_mode and args.gt_dir is None:
        parser.error("--gt-clipsim-only directory mode requires --gt-dir.")
    if not args.gt_clipsim_only and single_mode and (args.gt is None or args.pred is None):
        parser.error("Single-video mode requires both --gt and --pred.")
    if not args.gt_clipsim_only and directory_mode and (args.gt_dir is None or args.pred_dir is None):
        parser.error("Directory mode requires both --gt-dir and --pred-dir.")
    if directory_mode and args.dynamicity_buckets_json is None:
        parser.error("Directory mode requires --dynamicity-buckets-json for dynamicity bucket aggregation.")
    if not single_mode and not directory_mode:
        parser.error("Provide either --gt/--pred or --gt-dir/--pred-dir.")
    args.compute_fvd = directory_mode and not args.gt_clipsim_only and not args.no_fvd
    if single_mode and not args.gt_clipsim_only and not args.no_fvd:
        parser.error(
            "Standard FVD is a distribution metric and is unavailable for a single "
            "video pair; use directory mode or pass --no-fvd."
        )
    if args.compute_fvd and args.fvd_mode is None:
        parser.error("Directory I3D-FVD requires --fvd-mode.")
    if args.compute_fvd and not args.model_name:
        parser.error("Directory I3D-FVD requires --model-name.")
    if args.compute_fvd and not args.i3d_python.is_file():
        parser.error(f"I3D-FVD Python executable does not exist: {args.i3d_python}")
    if args.compute_fvd and args.i3d_model_path is not None and args.i3d_sha256_manifest is None:
        parser.error(
            "Local I3D weights require --i3d-sha256-manifest; no unverified "
            "local weights are accepted."
        )
    if Path(args.clipsim_model).is_dir():
        args.clipsim_local_files_only = True
    if directory_mode and args.clipsim_caption_dir is None:
        default_caption_dir = args.gt_dir.parent / "annotations" / "future_caption"
        if not default_caption_dir.is_dir():
            parser.error(
                "Default CLIPSIM captions were not found at "
                f"{default_caption_dir}; pass --clipsim-caption-dir."
            )
        args.clipsim_caption_dir = default_caption_dir
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.gpu_ids = [str(index) for index in range(torch.cuda.device_count())] if args.device == "cuda" else []
    args.frame_metric_shape = parse_frame_metric_size(args.frame_metric_size)
    args.compute_clipsim = (
        args.gt_clipsim_only
        or directory_mode
        or args.clipsim_caption_json is not None
        or args.clipsim_caption_dir is not None
    )
    if args.compute_clipsim and args.clipsim_batch_size <= 0:
        parser.error("--clipsim-batch-size must be positive.")
    if args.compute_clipsim and args.clipsim_num_frames <= 0:
        parser.error("--clipsim-num-frames must be positive.")
    if args.gt_clipsim_only and single_mode and args.clipsim_caption_json is None:
        parser.error("--gt-clipsim-only single-video mode requires --clipsim-caption-json.")
    if args.gt_clipsim_only and directory_mode and args.clipsim_caption_dir is None:
        parser.error("--gt-clipsim-only directory mode requires --clipsim-caption-dir.")
    return args
def parse_frame_metric_size(frame_metric_size: str | None) -> tuple[int, int] | None:
    if frame_metric_size is None:
        return None
    normalized = frame_metric_size.lower().replace("*", "x")
    if "x" in normalized:
        height_str, width_str = normalized.split("x", maxsplit=1)
        height = int(height_str)
        width = int(width_str)
    else:
        height = width = int(normalized)
    if height <= 0 or width <= 0:
        raise ValueError("--frame-metric-size dimensions must be positive integers.")
    return height, width
def normalize_dynamicity_bucket(dynamicity: str | None) -> str | None:
    if dynamicity is None:
        return None
    normalized = dynamicity.strip().lower()
    if normalized in {"low", "medium", "high"}:
        return normalized
    return None
@dataclass
class VideoInfo:
    path: Path
    frames: list[np.ndarray]
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int
def read_video(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames were decoded from {video_path}")
    actual_frame_count = len(frames)
    duration_sec = actual_frame_count / fps if fps > 0 else 0.0
    return VideoInfo(
        path=video_path,
        frames=frames,
        fps=fps,
        frame_count=actual_frame_count if frame_count <= 0 else actual_frame_count,
        duration_sec=duration_sec,
        width=width or frames[0].shape[1],
        height=height or frames[0].shape[0],
    )
def resize_like(frame: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
def build_lpips_model(net: str, device: str) -> Any:
    if net != "alex":
        raise ValueError("LPIPS is fixed to the standard AlexNet backbone; use `--lpips-net alex`.")
    try:
        import lpips
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Standard LPIPS requires the `lpips` package. No approximate fallback is allowed."
        ) from exc
    try:
        model = lpips.LPIPS(net="alex", version="0.1")
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the standard learned AlexNet LPIPS weights. "
            "No torchvision or alternative-backbone fallback is allowed."
        ) from exc
    return model.to(device).eval()
def frames_to_torch_uint8(frames: list[np.ndarray], device: str) -> torch.Tensor:
    array = np.stack(frames, axis=0)
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().to(device=device, dtype=torch.uint8)
def uint8_tensor_to_unit_range(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float() / 255.0
def unit_range_to_lpips_range(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 2.0 - 1.0
def compute_lpips_batch(lpips_model: Any, gt_tensor: torch.Tensor, pred_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        score = lpips_model(unit_range_to_lpips_range(gt_tensor), unit_range_to_lpips_range(pred_tensor))
    return score.flatten()
def compute_psnr_batch(gt_tensor: torch.Tensor, pred_tensor: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((gt_tensor - pred_tensor) ** 2, dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-12)
    return 10.0 * torch.log10(1.0 / mse)
def gaussian_kernel(kernel_size: int, sigma: float, device: str, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d
def compute_ssim_batch(
    gt_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    channels = gt_tensor.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, gt_tensor.device, gt_tensor.dtype)
    window = kernel.view(1, 1, kernel_size, kernel_size).expand(channels, 1, kernel_size, kernel_size)
    mu_x = F.conv2d(gt_tensor, window, padding=kernel_size // 2, groups=channels)
    mu_y = F.conv2d(pred_tensor, window, padding=kernel_size // 2, groups=channels)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(gt_tensor * gt_tensor, window, padding=kernel_size // 2, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(pred_tensor * pred_tensor, window, padding=kernel_size // 2, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(gt_tensor * pred_tensor, window, padding=kernel_size // 2, groups=channels) - mu_xy
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / torch.clamp(denominator, min=1e-12)
    return ssim_map.mean(dim=(1, 2, 3))
def compute_batched_frame_metrics(
    gt_frames: list[np.ndarray],
    pred_frames: list[np.ndarray],
    device: str,
    batch_size: int,
    lpips_model: Any,
) -> tuple[list[float], list[float], list[float]]:
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    for start in range(0, len(gt_frames), batch_size):
        end = min(start + batch_size, len(gt_frames))
        gt_uint8 = frames_to_torch_uint8(gt_frames[start:end], device)
        pred_uint8 = frames_to_torch_uint8(pred_frames[start:end], device)
        gt_batch = uint8_tensor_to_unit_range(gt_uint8)
        pred_batch = uint8_tensor_to_unit_range(pred_uint8)
        with torch.no_grad():
            psnr_batch = compute_psnr_batch(gt_batch, pred_batch)
            ssim_batch = compute_ssim_batch(gt_batch, pred_batch)
            lpips_batch = compute_lpips_batch(lpips_model, gt_batch, pred_batch)
        psnr_values.extend(float(x) for x in psnr_batch.detach().cpu().tolist())
        ssim_values.extend(float(x) for x in ssim_batch.detach().cpu().tolist())
        lpips_values.extend(float(x) for x in lpips_batch.detach().cpu().tolist())
        del gt_uint8, pred_uint8, gt_batch, pred_batch, psnr_batch, ssim_batch, lpips_batch
    return psnr_values, ssim_values, lpips_values
def estimate_gt_flow(gt_prev: np.ndarray, gt_next: np.ndarray) -> np.ndarray:
    gt_prev_gray = cv2.cvtColor(gt_prev, cv2.COLOR_RGB2GRAY)
    gt_next_gray = cv2.cvtColor(gt_next, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        gt_prev_gray,
        gt_next_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
def warp_frame(frame: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    warped = cv2.remap(
        frame.astype(np.float32) / 255.0,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    return warped, valid
def compute_warping_error(
    gt_prev: np.ndarray,
    gt_next: np.ndarray,
    pred_prev: np.ndarray,
    pred_next: np.ndarray,
) -> float:
    flow = estimate_gt_flow(gt_prev, gt_next)
    warped_pred_prev, valid = warp_frame(pred_prev, flow)
    pred_next_float = pred_next.astype(np.float32) / 255.0
    abs_error = np.abs(warped_pred_prev - pred_next_float).mean(axis=2)
    if not np.any(valid):
        return float("nan")
    return float(abs_error[valid].mean())
def ensure_parent_dir(path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
def write_json(path: Path, data: Any) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=True))
def get_by_field_path(data: Any, field_path: str, *, source_path: Path) -> Any:
    value = data
    traversed: list[str] = []
    for part in field_path.split("."):
        location = ".".join(traversed) or "<root>"
        if isinstance(value, list):
            try:
                index = int(part)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid caption field '{field_path}' for {source_path}: "
                    f"expected a list index at '{location}', got '{part}'."
                ) from exc
            try:
                value = value[index]
            except IndexError as exc:
                raise ValueError(
                    f"Invalid caption field '{field_path}' for {source_path}: "
                    f"index {index} is out of range at '{location}' with list length {len(value)}."
                ) from exc
        elif isinstance(value, dict):
            if part not in value:
                keys = ", ".join(sorted(str(key) for key in value.keys()))
                raise ValueError(
                    f"Invalid caption field '{field_path}' for {source_path}: "
                    f"missing key '{part}' at '{location}'. Available keys: {keys}"
                )
            value = value[part]
        else:
            raise ValueError(
                f"Invalid caption field '{field_path}' for {source_path}: "
                f"cannot read '{part}' at '{location}' from {type(value).__name__}."
            )
        traversed.append(part)
    return value
def format_caption_annotation(value: dict[str, Any], *, source_path: Path, field_path: str) -> str:
    missing = [key for key in ("weather", "environment", "caption") if key not in value]
    if missing:
        raise ValueError(
            f"Expected weather/environment/caption at '{field_path}' in {source_path}; missing: {', '.join(missing)}"
        )
    return (
        f"weather: {str(value['weather']).strip()} "
        f"environment: {str(value['environment']).strip()} "
        f"caption: {str(value['caption']).strip()}"
    )
def read_caption_annotation(annotation_path: Path, field_path: str) -> str:
    data = json.loads(annotation_path.read_text())
    caption = get_by_field_path(data, field_path, source_path=annotation_path)
    if isinstance(caption, dict):
        return format_caption_annotation(caption, source_path=annotation_path, field_path=field_path)
    if not isinstance(caption, str):
        raise ValueError(
            f"Expected a string or weather/environment/caption object at '{field_path}' in {annotation_path}, "
            f"got {type(caption).__name__}"
        )
    return caption.strip()
def build_caption_index(caption_dir: Path | None, field_path: str) -> dict[str, str | None]:
    if caption_dir is None:
        return {}
    if not caption_dir.is_dir():
        raise NotADirectoryError(f"CLIPSIM caption directory does not exist: {caption_dir}")
    caption_index: dict[str, str | None] = {}
    for annotation_path in sorted(caption_dir.rglob("*.json")):
        caption = read_caption_annotation(annotation_path, field_path)
        keys = sample_keys_from_stem(annotation_path.stem)
        for key in keys:
            add_unique_index_value(caption_index, key, caption)
    return caption_index
def find_caption_for_video(path: Path, caption_index: dict[str, str | None]) -> str | None:
    for key in sample_keys_from_stem(path.stem):
        if key in caption_index and caption_index[key] is not None:
            return caption_index[key]
    return None
def resolve_single_caption(args: argparse.Namespace) -> str:
    annotation_path = args.clipsim_caption_json
    if annotation_path is None:
        video_path = args.gt if args.gt_clipsim_only else args.pred
        candidate = video_path.with_suffix(".json")
        if candidate.is_file():
            annotation_path = candidate
    if annotation_path is None:
        raise ValueError("Single-video CLIPSIM requires --clipsim-caption-json or a JSON next to the CLIPSIM video.")
    return read_caption_annotation(annotation_path, args.clipsim_caption_field)
class ClipSimEvaluator:
    def __init__(self, model_name: str, device: str, batch_size: int, *, local_files_only: bool = False) -> None:
        from transformers import CLIPModel, CLIPProcessor
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.batch_size = batch_size
        self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = CLIPModel.from_pretrained(model_name, local_files_only=local_files_only).to(device).eval()
    def compute(
        self,
        video_frames: list[np.ndarray],
        caption: str,
        *,
        num_frames: int,
        video_source: str,
    ) -> dict[str, Any]:
        source_frames = video_frames[:num_frames]
        if not source_frames:
            raise ValueError("No frames available for CLIPSim.")
        text_inputs = self.processor(
            text=[caption],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            text_outputs = self.model.text_model(**text_inputs)
            text_features = self.model.text_projection(text_outputs.pooler_output)
            text_features = F.normalize(text_features, dim=-1)
            similarities = []
            for start in range(0, len(source_frames), self.batch_size):
                image_inputs = self.processor(
                    images=source_frames[start : start + self.batch_size],
                    return_tensors="pt",
                ).to(self.device)
                image_outputs = self.model.vision_model(
                    pixel_values=image_inputs.pixel_values
                )
                image_features = self.model.visual_projection(
                    image_outputs.pooler_output
                )
                image_features = F.normalize(image_features, dim=-1)
                similarities.extend(
                    (image_features @ text_features.T)
                    .squeeze(1)
                    .float()
                    .cpu()
                    .tolist()
                )
        return {
            "score": float(np.mean(similarities)),
            "video_source": video_source,
            "frame_stride": 1,
            "num_sampled_frames": len(similarities),
            "num_source_frames": len(source_frames),
            "num_available_frames": len(video_frames),
            "max_frames": num_frames,
            "model": getattr(self.model, "name_or_path", None)
            or self.model.config.name_or_path,
        }
def evaluate_video_pair(
    gt_path: Path,
    pred_path: Path,
    args: argparse.Namespace,
    lpips_model: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gt_video = read_video(gt_path)
    pred_video = read_video(pred_path)
    prediction_type = args.prediction_type
    if prediction_type == "auto":
        path_text = str(pred_path).lower()
        prediction_type = "cogvideox_i2v" if "cogvideox" in path_text and "i2v" in path_text else "standard"
    pred_start = 1 if prediction_type == "cogvideox_i2v" else 0
    available_pred_frames = pred_video.frames[pred_start:]
    if not available_pred_frames:
        raise ValueError("No predicted frames remain after applying prediction-type handling.")
    if args.frame_alignment == "timestamp":
        if gt_video.fps <= 0 or pred_video.fps <= 0:
            raise ValueError("Timestamp alignment requires valid decoded FPS values for both videos.")
        pred_indices = list(range(pred_start, pred_video.frame_count))
        gt_indices = [int(round(index * gt_video.fps / pred_video.fps)) for index in range(len(available_pred_frames))]
        paired_indices = [(gt_index, pred_index) for gt_index, pred_index in zip(gt_indices, pred_indices, strict=True) if gt_index < gt_video.frame_count]
        if not paired_indices:
            raise ValueError("No overlapping timestamp-aligned frames were found.")
        gt_frames = [gt_video.frames[gt_index] for gt_index, _ in paired_indices]
        pred_frames = [pred_video.frames[pred_index] for _, pred_index in paired_indices]
        alignment_description = "By timestamp: GT index = round(pred-local-index * gt_fps / pred_fps)."
    else:
        if gt_video.frame_count != len(available_pred_frames) and not args.allow_frame_count_mismatch:
            raise ValueError(
                "GT and predicted videos do not have the same usable frame count. "
                "Use --allow-frame-count-mismatch or --frame-alignment timestamp."
            )
        num_index_frames = min(gt_video.frame_count, len(available_pred_frames))
        paired_indices = [(index, pred_start + index) for index in range(num_index_frames)]
        gt_frames = gt_video.frames[:num_index_frames]
        pred_frames = available_pred_frames[:num_index_frames]
        alignment_description = "By frame index after optional I2V condition-frame removal."
    num_frames = len(gt_frames)
    frame_metric_shape = args.frame_metric_shape
    if frame_metric_shape is not None:
        # Resize each source directly to the common metric resolution. Resizing
        # predictions to the (often much larger) native GT resolution first and
        # then back down adds an avoidable interpolation pass and biases all
        # frame-wise metrics.
        metric_gt_frames = [resize_like(frame, frame_metric_shape) for frame in gt_frames]
        metric_pred_frames = [resize_like(frame, frame_metric_shape) for frame in pred_frames]
        pred_resized_to_gt = False
        frame_resize_policy = "gt_and_prediction_directly_to_common_metric_resolution"
    else:
        metric_gt_frames = gt_frames
        metric_pred_frames = [resize_like(frame, gt_frames[0].shape[:2]) for frame in pred_frames]
        pred_resized_to_gt = True
        frame_resize_policy = "prediction_to_native_gt_resolution"
    psnr_values, ssim_values, lpips_values = compute_batched_frame_metrics(
        gt_frames=metric_gt_frames,
        pred_frames=metric_pred_frames,
        device=args.device,
        batch_size=args.batch_size,
        lpips_model=lpips_model,
    )
    frame_rows: list[dict[str, Any]] = []
    warping_errors: list[float] = []
    for frame_idx, (psnr_value, ssim_value, lpips_value) in enumerate(
        zip(psnr_values, ssim_values, lpips_values, strict=True)
    ):
        if frame_idx == 0:
            warping_error = float("nan")
        else:
            warping_error = compute_warping_error(
                metric_gt_frames[frame_idx - 1],
                metric_gt_frames[frame_idx],
                metric_pred_frames[frame_idx - 1],
                metric_pred_frames[frame_idx],
            )
            warping_errors.append(warping_error)
        frame_rows.append(
            {
                "frame_idx": frame_idx,
                "psnr": psnr_value,
                "ssim": ssim_value,
                "lpips": lpips_value,
                "warping_error": warping_error,
            }
        )
    summary = {
        "gt_video": {
            "path": str(gt_video.path),
            "fps": gt_video.fps,
            "frame_count": gt_video.frame_count,
            "duration_sec": gt_video.duration_sec,
            "width": gt_video.width,
            "height": gt_video.height,
        },
        "pred_video": {
            "path": str(pred_video.path),
            "fps": pred_video.fps,
            "frame_count": pred_video.frame_count,
            "duration_sec": pred_video.duration_sec,
            "width": pred_video.width,
            "height": pred_video.height,
        },
        "comparison": {
            "num_compared_frames": num_frames,
            "frame_alignment": alignment_description,
            "prediction_type": prediction_type,
            "dropped_prediction_frames": pred_start,
            "paired_frame_indices": [{"gt": gt_index, "pred": pred_index} for gt_index, pred_index in paired_indices],
            "pred_resized_to_gt": pred_resized_to_gt,
            "frame_resize_policy": frame_resize_policy,
            "frame_metric_size": args.frame_metric_size,
            "frame_metric_resolution": (
                {"height": frame_metric_shape[0], "width": frame_metric_shape[1]}
                if frame_metric_shape is not None
                else {"height": gt_frames[0].shape[0], "width": gt_frames[0].shape[1]}
            ),
            "device": args.device,
            "gpus": args.gpu_ids,
            "batch_size": args.batch_size,
        },
        "metrics": {
            "psnr_mean": float(np.mean(psnr_values)),
            "psnr_std": float(np.std(psnr_values)),
            "ssim_mean": float(np.mean(ssim_values)),
            "ssim_std": float(np.std(ssim_values)),
            "lpips_mean": float(np.mean(lpips_values)),
            "lpips_std": float(np.std(lpips_values)),
            "warping_error_mean": float(np.mean(warping_errors)) if warping_errors else float("nan"),
            "warping_error_std": float(np.std(warping_errors)) if warping_errors else float("nan"),
        },
    }
    return summary, frame_rows
def is_predicted_video(path: Path) -> bool:
    return path.suffix.lower() == ".mp4" and not path.stem.lower().endswith("input_clip")
def normalized_video_id(path: Path) -> str:
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    if match is None:
        return path.stem
    normalized = match.group(1).lstrip("0")
    return normalized or "0"
def sample_keys_from_stem(stem: str) -> list[str]:
    keys = [stem]
    if stem.endswith("_input_clip"):
        keys.append(stem[: -len("_input_clip")])
    if stem.endswith("_gt"):
        keys.append(stem[: -len("_gt")])
    match = re.search(r"(\d+)(?!.*\d)", stem)
    if match is not None:
        normalized = match.group(1).lstrip("0") or "0"
        keys.append(normalized)
    return list(dict.fromkeys(keys))
def add_unique_index_value(index: dict[str, Any | None], key: str, value: Any) -> None:
    if key in index and index[key] != value:
        index[key] = None
    else:
        index[key] = value
def build_gt_video_index(gt_dir: Path) -> dict[str, Path | None]:
    gt_index: dict[str, Path | None] = {}
    for gt_path in sorted(gt_dir.glob("*.mp4")):
        keys = sample_keys_from_stem(gt_path.stem)
        for key in keys:
            add_unique_index_value(gt_index, key, gt_path)
    return gt_index
def find_matching_gt(pred_path: Path, gt_index: dict[str, Path | None]) -> Path | None:
    for key in sample_keys_from_stem(pred_path.stem):
        if key in gt_index and gt_index[key] is not None:
            return gt_index[key]
    return None
def build_dynamicity_bucket_index(dynamicity_buckets_json: Path | None) -> dict[str, str | None]:
    if dynamicity_buckets_json is None:
        return {}
    with dynamicity_buckets_json.open() as f:
        data = json.load(f)
    bucket_index: dict[str, str | None] = {}
    for sample_idx, sample in enumerate(data.get("samples", [])):
        sample_id = sample.get("sample_id")
        bucket = sample.get("dynamicity_bucket") or sample.get("motion_stratum")
        if not sample_id or not bucket:
            continue
        keys = sample_keys_from_stem(str(sample_id))
        keys.extend(sample_keys_from_stem(str(sample_idx).zfill(4)))
        for numeric_key in ("video_index", "segment_index"):
            if numeric_key in sample:
                keys.extend(sample_keys_from_stem(str(sample[numeric_key]).zfill(4)))
        outputs = sample.get("outputs", {})
        if isinstance(outputs, dict):
            for value in outputs.values():
                if isinstance(value, str):
                    keys.extend(sample_keys_from_stem(Path(value).stem))
        keys = list(dict.fromkeys(keys))
        for key in keys:
            add_unique_index_value(bucket_index, key, str(bucket))
    return bucket_index
def find_dynamicity_bucket(path: Path, bucket_index: dict[str, str | None]) -> str | None:
    for key in sample_keys_from_stem(path.stem):
        if key in bucket_index and bucket_index[key] is not None:
            return bucket_index[key]
    return None
def summarize_batch_metrics(pair_summaries: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = [
        "psnr_mean",
        "ssim_mean",
        "lpips_mean",
        "warping_error_mean",
    ]
    aggregate: dict[str, float] = {}
    for metric_name in metric_names:
        values = [
            summary["metrics"][metric_name]
            for summary in pair_summaries
            if isinstance(summary.get("metrics"), dict) and metric_name in summary["metrics"]
        ]
        finite_values = [value for value in values if np.isfinite(value)]
        if finite_values:
            aggregate[f"{metric_name}_mean"] = float(np.mean(finite_values))
            aggregate[f"{metric_name}_std"] = float(np.std(finite_values))
    clipsim_values = [
        summary["clipsim"]["score"]
        for summary in pair_summaries
        if isinstance(summary.get("clipsim"), dict) and "score" in summary["clipsim"]
    ]
    if clipsim_values:
        aggregate["clipsim_score_mean"] = float(np.mean(clipsim_values))
        aggregate["clipsim_score_std"] = float(np.std(clipsim_values))
    return aggregate
def summarize_metrics_by_bucket(pair_summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries_by_bucket: dict[str, list[dict[str, Any]]] = {}
    for summary in pair_summaries:
        bucket = summary.get("dynamicity_bucket", "unknown")
        summaries_by_bucket.setdefault(bucket, []).append(summary)
    bucket_summaries: dict[str, dict[str, Any]] = {}
    for bucket, summaries in sorted(summaries_by_bucket.items()):
        bucket_summaries[bucket] = {
            "count": len(summaries),
            "metrics": summarize_batch_metrics(summaries),
        }
    return bucket_summaries
def make_worker_args(args: argparse.Namespace, gpu_id: str) -> argparse.Namespace:
    worker_args = copy.copy(args)
    worker_args.gpu_ids = [gpu_id]
    worker_args.device = "cuda"
    return worker_args
def evaluate_directory_jobs_on_gpu(
    gpu_id: str,
    jobs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    worker_args = make_worker_args(args, gpu_id)
    lpips_model = build_lpips_model(worker_args.lpips_net, worker_args.device)
    clipsim_evaluator = (
        ClipSimEvaluator(
            worker_args.clipsim_model,
            worker_args.device,
            worker_args.clipsim_batch_size,
            local_files_only=worker_args.clipsim_local_files_only,
        )
        if worker_args.compute_clipsim
        else None
    )
    pair_summaries: list[dict[str, Any]] = []
    for job in jobs:
        gt_path = Path(job["gt_path"])
        pred_path = Path(job["pred_path"])
        print(f"[gpu {gpu_id}] Evaluating {pred_path.name} against {gt_path.name}...")
        summary, _ = evaluate_video_pair(gt_path, pred_path, worker_args, lpips_model)
        summary["video_id"] = job["video_id"]
        summary["dynamicity_bucket"] = job["dynamicity_bucket"]
        if clipsim_evaluator is not None:
            clipsim_frames = read_video(pred_path).frames[int(summary["comparison"]["dropped_prediction_frames"]):]
            summary["clipsim"] = clipsim_evaluator.compute(
                clipsim_frames,
                job["clipsim_caption"],
                num_frames=worker_args.clipsim_num_frames,
                video_source="pred",
            )
        summary["_order"] = job["order"]
        pair_summaries.append(summary)
    return pair_summaries
def run_single_video(args: argparse.Namespace, lpips_model: Any) -> dict[str, Any]:
    summary, frame_rows = evaluate_video_pair(args.gt, args.pred, args, lpips_model)
    if args.compute_clipsim:
        caption = resolve_single_caption(args)
        clipsim_evaluator = ClipSimEvaluator(
            args.clipsim_model,
            args.device,
            args.clipsim_batch_size,
            local_files_only=args.clipsim_local_files_only,
        )
        clipsim_frames = read_video(args.pred).frames
        summary["clipsim"] = clipsim_evaluator.compute(
            clipsim_frames,
            caption,
            num_frames=args.clipsim_num_frames,
            video_source="pred",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))
    if args.output_json is not None:
        write_json(args.output_json, summary)
    if args.output_frame_json is not None:
        write_json(
            args.output_frame_json,
            {
                "gt_video": str(args.gt),
                "pred_video": str(args.pred),
                "frames": frame_rows,
            },
        )
    return summary
def run_i3d_fvd_subprocess(args: argparse.Namespace) -> dict[str, Any]:
    """Run TensorFlow I3D-FVD outside the PyTorch metrics interpreter."""
    script_path = Path(__file__).with_name("i3d_fvd.py").resolve()
    with tempfile.TemporaryDirectory(prefix="i3d_fvd_result_") as temp_dir:
        result_path = Path(temp_dir) / "fvd.json"
        command = [
            str(args.i3d_python),
            str(script_path),
            "--gt-dir",
            str(args.gt_dir),
            "--pred-dir",
            str(args.pred_dir),
            "--dynamicity-buckets-json",
            str(args.dynamicity_buckets_json),
            "--output-json",
            str(result_path),
            "--i3d-cache-dir",
            str(args.i3d_cache_dir),
            "--mode",
            str(args.fvd_mode),
            "--model",
            str(args.model_name),
        ]
        if args.i3d_model_path is not None:
            command.extend(["--i3d-model-path", str(args.i3d_model_path)])
        if args.i3d_sha256_manifest is not None:
            command.extend(
                ["--i3d-sha256-manifest", str(args.i3d_sha256_manifest)]
            )
        if args.condition_setting is not None:
            command.extend(["--condition-setting", str(args.condition_setting)])
        if args.fvd_overall:
            command.append("--overall")
        print(
            f"Running isolated I3D-FVD with {args.i3d_python}...",
            flush=True,
        )
        subprocess_env = os.environ.copy()
        dataset_mount_root = Path("/dataset")
        subprocess_env.setdefault(
            "CUDA_CACHE_PATH",
            str(dataset_mount_root / ".cache/cuda-compute-cache"),
        )
        subprocess_env["TF_CUDNN_USE_AUTOTUNE"] = "0"
        subprocess_env["TF_DETERMINISTIC_OPS"] = "1"
        subprocess.run(command, check=True, env=subprocess_env)
        if not result_path.is_file():
            raise RuntimeError(
                f"I3D-FVD subprocess did not create its result: {result_path}"
            )
        return json.loads(result_path.read_text())
def run_directory_batch(args: argparse.Namespace) -> dict[str, Any]:
    if not args.gt_dir.is_dir():
        raise NotADirectoryError(f"Ground-truth directory does not exist: {args.gt_dir}")
    if not args.pred_dir.is_dir():
        raise NotADirectoryError(f"Predicted directory does not exist: {args.pred_dir}")
    fvd_result = run_i3d_fvd_subprocess(args) if args.compute_fvd else None
    pred_paths = [path for path in sorted(args.pred_dir.glob("*.mp4")) if is_predicted_video(path)]
    gt_index = build_gt_video_index(args.gt_dir)
    bucket_index = build_dynamicity_bucket_index(args.dynamicity_buckets_json)
    caption_index = build_caption_index(args.clipsim_caption_dir, args.clipsim_caption_field) if args.compute_clipsim else {}
    pair_summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    jobs: list[dict[str, Any]] = []
    for order, pred_path in enumerate(pred_paths):
        gt_path = find_matching_gt(pred_path, gt_index)
        if gt_path is None:
            skipped.append({"pred": str(pred_path), "reason": "no matching ground-truth video"})
            continue
        video_id = normalized_video_id(pred_path)
        dynamicity_bucket = normalize_dynamicity_bucket(find_dynamicity_bucket(pred_path, bucket_index))
        if dynamicity_bucket is None and bucket_index:
            dynamicity_bucket = "unknown"
        clipsim_caption = None
        if args.compute_clipsim:
            clipsim_caption = find_caption_for_video(pred_path, caption_index)
            if clipsim_caption is None:
                skipped.append({"pred": str(pred_path), "reason": "no matching CLIPSIM caption annotation"})
                continue
        jobs.append(
            {
                "order": order,
                "gt_path": str(gt_path),
                "pred_path": str(pred_path),
                "video_id": video_id,
                "dynamicity_bucket": dynamicity_bucket,
                "clipsim_caption": clipsim_caption,
            }
        )
    if len(args.gpu_ids) > 1 and args.device.startswith("cuda"):
        job_chunks = [jobs[index :: len(args.gpu_ids)] for index in range(len(args.gpu_ids))]
        with ProcessPoolExecutor(max_workers=len(args.gpu_ids), mp_context=mp.get_context("spawn")) as executor:
            futures = [
                executor.submit(evaluate_directory_jobs_on_gpu, gpu_id, chunk, args)
                for gpu_id, chunk in zip(args.gpu_ids, job_chunks, strict=True)
                if chunk
            ]
            for future in as_completed(futures):
                pair_summaries.extend(future.result())
    else:
        lpips_model = build_lpips_model(args.lpips_net, args.device)
        clipsim_evaluator = (
            ClipSimEvaluator(
                args.clipsim_model,
                args.device,
                args.clipsim_batch_size,
                local_files_only=args.clipsim_local_files_only,
            )
            if args.compute_clipsim
            else None
        )
        for job in jobs:
            gt_path = Path(job["gt_path"])
            pred_path = Path(job["pred_path"])
            print(f"Evaluating {pred_path.name} against {gt_path.name}...")
            summary, _ = evaluate_video_pair(gt_path, pred_path, args, lpips_model)
            summary["video_id"] = job["video_id"]
            summary["dynamicity_bucket"] = job["dynamicity_bucket"]
            if clipsim_evaluator is not None:
                clipsim_frames = read_video(pred_path).frames[int(summary["comparison"]["dropped_prediction_frames"]):]
                summary["clipsim"] = clipsim_evaluator.compute(
                    clipsim_frames,
                    job["clipsim_caption"],
                    num_frames=args.clipsim_num_frames,
                    video_source="pred",
                )
            summary["_order"] = job["order"]
            pair_summaries.append(summary)
    pair_summaries = sorted(pair_summaries, key=lambda summary: summary["_order"])
    for summary in pair_summaries:
        summary.pop("_order", None)
    metrics_by_dynamicity = summarize_metrics_by_bucket(pair_summaries) if pair_summaries else {}
    if fvd_result is not None:
        for bucket, fvd_record in fvd_result["bucket_results"].items():
            bucket_summary = metrics_by_dynamicity.setdefault(
                bucket, {"count": fvd_record["num_generated_videos"], "metrics": {}}
            )
            bucket_summary.setdefault("metrics", {})["fvd"] = fvd_record["fvd"]
            bucket_summary["fvd_details"] = fvd_record
    batch_summary = {
        "pred_dir": str(args.pred_dir),
        "gt_dir": str(args.gt_dir),
        "dynamicity_buckets_json": str(args.dynamicity_buckets_json) if args.dynamicity_buckets_json is not None else None,
        "clipsim_video_source": "pred" if args.compute_clipsim else None,
        "gpus": args.gpu_ids,
        "parallel_workers": len(args.gpu_ids) if len(args.gpu_ids) > 1 and args.device.startswith("cuda") else 1,
        "num_pred_videos": len(pred_paths),
        "num_evaluated_videos": len(pair_summaries),
        "num_skipped_videos": len(skipped),
        "aggregate_metrics": summarize_batch_metrics(pair_summaries) if pair_summaries else {},
        "fvd": fvd_result,
        "metrics_by_dynamicity": metrics_by_dynamicity,
        "videos": pair_summaries,
        "skipped": skipped,
    }
    print(json.dumps(batch_summary, indent=2, ensure_ascii=False, allow_nan=True))
    if args.output_json is not None:
        write_json(args.output_json, batch_summary)
    if args.output_dynamicity_json is not None:
        write_json(args.output_dynamicity_json, metrics_by_dynamicity)
    return batch_summary
def video_info_summary(video: VideoInfo) -> dict[str, Any]:
    return {
        "path": str(video.path),
        "fps": video.fps,
        "frame_count": video.frame_count,
        "duration_sec": video.duration_sec,
        "width": video.width,
        "height": video.height,
    }
def run_gt_clipsim_single(args: argparse.Namespace) -> dict[str, Any]:
    caption = resolve_single_caption(args)
    gt_video = read_video(args.gt)
    clipsim_evaluator = ClipSimEvaluator(
        args.clipsim_model,
        args.device,
        args.clipsim_batch_size,
        local_files_only=args.clipsim_local_files_only,
    )
    clipsim = clipsim_evaluator.compute(
        gt_video.frames,
        caption,
        num_frames=args.clipsim_num_frames,
        video_source="gt",
    )
    summary = {
        "mode": "gt_clipsim_only",
        "gt_video": video_info_summary(gt_video),
        "clipsim": clipsim,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))
    if args.output_json is not None:
        write_json(args.output_json, summary)
    return summary
def run_gt_clipsim_directory(args: argparse.Namespace) -> dict[str, Any]:
    if not args.gt_dir.is_dir():
        raise NotADirectoryError(f"Ground-truth directory does not exist: {args.gt_dir}")
    gt_paths = sorted(path for path in args.gt_dir.glob("*.mp4") if path.is_file())
    bucket_index = build_dynamicity_bucket_index(args.dynamicity_buckets_json)
    caption_index = build_caption_index(args.clipsim_caption_dir, args.clipsim_caption_field)
    clipsim_evaluator = ClipSimEvaluator(
        args.clipsim_model,
        args.device,
        args.clipsim_batch_size,
        local_files_only=args.clipsim_local_files_only,
    )
    video_summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for gt_path in gt_paths:
        caption = find_caption_for_video(gt_path, caption_index)
        if caption is None:
            skipped.append({"gt": str(gt_path), "reason": "no matching CLIPSIM caption annotation"})
            continue
        dynamicity_bucket = normalize_dynamicity_bucket(find_dynamicity_bucket(gt_path, bucket_index))
        print(f"Computing GT CLIPSIM for {gt_path.name}...")
        gt_video = read_video(gt_path)
        clipsim = clipsim_evaluator.compute(
            gt_video.frames,
            caption,
            num_frames=args.clipsim_num_frames,
            video_source="gt",
        )
        video_summaries.append(
            {
                "video_id": normalized_video_id(gt_path),
                "dynamicity_bucket": dynamicity_bucket,
                "gt_video": video_info_summary(gt_video),
                "clipsim": clipsim,
            }
        )
    metrics_by_dynamicity = summarize_metrics_by_bucket(video_summaries) if video_summaries else {}
    batch_summary = {
        "mode": "gt_clipsim_only",
        "gt_dir": str(args.gt_dir),
        "dynamicity_buckets_json": str(args.dynamicity_buckets_json) if args.dynamicity_buckets_json is not None else None,
        "clipsim_video_source": "gt",
        "gpus": args.gpu_ids,
        "parallel_workers": 1,
        "num_gt_videos": len(gt_paths),
        "num_evaluated_videos": len(video_summaries),
        "num_skipped_videos": len(skipped),
        "aggregate_metrics": summarize_batch_metrics(video_summaries) if video_summaries else {},
        "metrics_by_dynamicity": metrics_by_dynamicity,
        "videos": video_summaries,
        "skipped": skipped,
    }
    print(json.dumps(batch_summary, indent=2, ensure_ascii=False, allow_nan=True))
    if args.output_json is not None:
        write_json(args.output_json, batch_summary)
    if args.output_dynamicity_json is not None:
        write_json(args.output_dynamicity_json, metrics_by_dynamicity)
    return batch_summary
def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if args.gt_clipsim_only and args.gt_dir is not None:
        run_gt_clipsim_directory(args)
    elif args.gt_clipsim_only:
        run_gt_clipsim_single(args)
    elif args.gt_dir is not None:
        run_directory_batch(args)
    else:
        lpips_model = build_lpips_model(args.lpips_net, args.device)
        run_single_video(args, lpips_model)
if __name__ == "__main__":
    main()
