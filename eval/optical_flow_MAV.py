#!/usr/bin/env python3
"""
Compute Normalized Mean Optical Flow Magnitude (NMOFM) for Zurich Urban MAV image sequences.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


TARGET_WIDTH = 854
TARGET_HEIGHT = 480
TARGET_DIAGONAL = math.hypot(TARGET_WIDTH, TARGET_HEIGHT)
CUDA_DECODE_PREFERRED_CODECS = {"h264", "hevc", "h265"}
VIDEO_SUFFIXES = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".flv",
    ".mpeg",
    ".mpg",
    ".m4v",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SOURCE_FPS = 24.0
DEFAULT_TARGET_SAMPLE_FPS = 5.0
MAX_SOURCE_FRAMES = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute normalized mean optical flow magnitude for Zurich Urban MAV image sequences."
    )
    parser.add_argument("--input", type=Path, required=True, help="Image-sequence root or a single sequence directory.")
    parser.add_argument(
        "--source-fps",
        type=float,
        default=DEFAULT_SOURCE_FPS,
        help="Source image sequence FPS. Default: 30.",
    )
    parser.add_argument(
        "--target-sample-fps",
        type=float,
        default=DEFAULT_TARGET_SAMPLE_FPS,
        help="Target sampling FPS. Default: 5.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search videos.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/optical_flow_tartanair.json"),
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated logical GPU ids inside the container. Empty means CPU only.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=30,
        help="Worker processes per GPU.",
    )
    parser.add_argument(
        "--ffmpeg-hwaccel",
        choices=["none", "cuda"],
        default="none",
        help="Unused for image sequences. Kept for compatibility.",
    )
    parser.add_argument(
        "--raft-model",
        choices=["small", "large"],
        default="small",
        help="Torchvision RAFT variant.",
    )
    parser.add_argument(
        "--flow-batch-size",
        type=int,
        default=4,
        help="How many frame pairs to run through RAFT at once per process.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Log every N pair results.",
    )
    parser.add_argument(
        "--max-source-frames",
        type=int,
        default=MAX_SOURCE_FRAMES,
        help="Use only the first N source frames from each sequence. Default: 5000.",
    )
    return parser.parse_args()


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | pid=%(process)d | %(message)s",
    )


def align_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return int(math.ceil(value / multiple) * multiple)


def get_model_input_size() -> Tuple[int, int]:
    return align_to_multiple(TARGET_WIDTH, 8), align_to_multiple(TARGET_HEIGHT, 8)


def get_model_input_diagonal() -> float:
    width, height = get_model_input_size()
    return math.hypot(width, height)


def center_crop_to_aspect_ratio(image: Any, target_width: int, target_height: int) -> Any:
    src_width, src_height = image.size
    target_ratio = target_width / target_height
    src_ratio = src_width / src_height if src_height > 0 else target_ratio

    if abs(src_ratio - target_ratio) < 1e-6:
        return image

    if src_ratio > target_ratio:
        crop_width = int(round(src_height * target_ratio))
        crop_height = src_height
        left = max(0, (src_width - crop_width) // 2)
        top = 0
    else:
        crop_width = src_width
        crop_height = int(round(src_width / target_ratio))
        left = 0
        top = max(0, (src_height - crop_height) // 2)

    right = min(src_width, left + crop_width)
    bottom = min(src_height, top + crop_height)
    return image.crop((left, top, right, bottom))


def build_sample_indices(
    total_frames: int,
    source_fps: float,
    target_sample_fps: float,
) -> List[int]:
    if total_frames <= 0:
        return []
    if source_fps <= 0 or target_sample_fps <= 0:
        raise ValueError("source_fps and target_sample_fps must be positive.")

    step = max(source_fps / target_sample_fps, 1.0)
    indices: List[int] = []
    cursor = 0.0
    while True:
        index = min(int(round(cursor)), total_frames - 1)
        if not indices or index != indices[-1]:
            indices.append(index)
        if index >= total_frames - 1:
            break
        cursor += step
    return indices


def warmup_raft_weights(raft_model: str) -> None:
    try:
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch and torchvision are required. Please install torch torchvision."
        ) from exc

    logging.info("Warming up RAFT weights in the main process | raft_model=%s", raft_model)
    if raft_model == "large":
        weights = Raft_Large_Weights.DEFAULT
        raft_large(weights=weights, progress=False)
    else:
        weights = Raft_Small_Weights.DEFAULT
        raft_small(weights=weights, progress=False)
    logging.info("RAFT weights are ready in local cache.")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def list_sequence_images(sequence_dir: Path) -> List[Path]:
    return sorted(path for path in sequence_dir.iterdir() if is_image_file(path))


def discover_sequences(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        raise SystemExit("Input must be an image-sequence directory, not a file.")
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")
    if not input_path.is_dir():
        raise SystemExit(f"Unsupported input path: {input_path}")
    if list_sequence_images(input_path):
        return [input_path]

    pattern = "**/*" if recursive else "*"
    sequences = [
        path for path in input_path.glob(pattern)
        if path.is_dir() and list_sequence_images(path)
    ]
    if not sequences:
        raise SystemExit(f"No image sequences found under: {input_path}")
    return sorted(sequences)


def parse_gpu_ids(raw: str) -> List[int]:
    text = raw.strip()
    if not text:
        return []
    gpu_ids: List[int] = []
    for part in text.split(","):
        value = part.strip()
        if not value:
            continue
        try:
            gpu_ids.append(int(value))
        except ValueError as exc:
            raise SystemExit(f"Invalid GPU id: {value}") from exc
    return gpu_ids


def log_runtime_gpu_info(requested_gpu_ids: Sequence[int]) -> None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    logging.info(
        "Runtime GPU info | requested_gpu_ids=%s | CUDA_VISIBLE_DEVICES=%s",
        list(requested_gpu_ids) if requested_gpu_ids else [],
        visible_devices or "<unset>",
    )


def load_sampled_sequence_frames(
    sequence_dir: Path,
    width: int,
    height: int,
    source_fps: float,
    target_sample_fps: float,
    max_source_frames: int,
) -> Tuple[List[Any], str]:
    import numpy as np
    from PIL import Image

    if max_source_frames <= 0:
        raise ValueError("max_source_frames must be positive.")

    image_paths = list_sequence_images(sequence_dir)
    limited_paths = image_paths[:max_source_frames]
    sample_indices = build_sample_indices(
        total_frames=len(limited_paths),
        source_fps=source_fps,
        target_sample_fps=target_sample_fps,
    )
    selected_paths = [limited_paths[index] for index in sample_indices]
    frames: List[Any] = []
    for image_path in selected_paths:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            rgb = center_crop_to_aspect_ratio(rgb, target_width=16, target_height=9)
            rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
            frames.append(np.asarray(rgb, dtype=np.uint8))
    return frames, "image_sequence"


def get_torch_backend(
    gpu_id: Optional[int],
    args: argparse.Namespace,
) -> Tuple[str, Any, Any, Any, str]:
    try:
        import torch
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch and torchvision are required. Please install torch torchvision."
        ) from exc

    if gpu_id is not None and torch.cuda.is_available():
        visible_count = torch.cuda.device_count()
        if gpu_id < visible_count:
            device = torch.device(f"cuda:{gpu_id}")
            backend = "pytorch_cuda"
            reason = f"Using torch CUDA device cuda:{gpu_id}."
        else:
            device = torch.device("cpu")
            backend = "pytorch_cpu"
            reason = f"Requested GPU id {gpu_id} is not visible. torch sees {visible_count} CUDA device(s)."
    else:
        device = torch.device("cpu")
        backend = "pytorch_cpu"
        reason = "CUDA unavailable or no GPU assigned."

    if args.raft_model == "large":
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=False)
    else:
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=False)

    model = model.to(device)
    model.eval()
    transforms = weights.transforms()
    return backend, device, model, transforms, reason


def compute_pair_means_torch(
    frames: Sequence[Any],
    model: Any,
    transforms: Any,
    device: Any,
    args: argparse.Namespace,
    video_name: str,
) -> List[float]:
    import torch

    pair_means: List[float] = []
    total_pairs = max(len(frames) - 1, 0)
    batch_size = max(1, args.flow_batch_size)

    with torch.inference_mode():
        for start in range(0, total_pairs, batch_size):
            end = min(start + batch_size, total_pairs)
            prev_batch = []
            curr_batch = []
            for pair_index in range(start, end):
                prev_tensor = torch.from_numpy(frames[pair_index]).permute(2, 0, 1).contiguous()
                curr_tensor = torch.from_numpy(frames[pair_index + 1]).permute(2, 0, 1).contiguous()
                prev_batch.append(prev_tensor)
                curr_batch.append(curr_tensor)

            prev = torch.stack(prev_batch, dim=0)
            curr = torch.stack(curr_batch, dim=0)
            prev, curr = transforms(prev, curr)
            prev = prev.to(device, non_blocking=True)
            curr = curr.to(device, non_blocking=True)

            flow_predictions = model(prev, curr)
            flow = flow_predictions[-1]
            magnitude = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)
            batch_means = magnitude.mean(dim=(1, 2)).detach().cpu().tolist()
            pair_means.extend(float(x) for x in batch_means)

            current_pair = end
            if args.progress_every > 0 and (
                current_pair == total_pairs or current_pair % args.progress_every == 0 or start == 0
            ):
                logging.info(
                    "Optical flow progress for %s | pair %d/%d | current_batch_mean=%.6f",
                    video_name,
                    current_pair,
                    total_pairs,
                    sum(batch_means) / max(len(batch_means), 1),
                )

    return pair_means


def compute_nmofm_for_video(
    video_path: Path,
    args: argparse.Namespace,
    gpu_id: Optional[int] = None,
) -> Dict[str, Any]:
    import numpy as np

    backend, device, model, transforms, backend_reason = get_torch_backend(gpu_id, args)
    start_time = time.time()

    image_paths = list_sequence_images(video_path)
    if not image_paths:
        raise RuntimeError(f"No images found in sequence directory: {video_path}")

    from PIL import Image

    with Image.open(image_paths[0]) as first_image:
        width, height = first_image.size

    source_fps = args.source_fps
    frame_count = min(len(image_paths), args.max_source_frames)
    codec = "image_sequence"

    logging.info(
        "Processing video %s | codec=%s | source_fps=%.3f | frame_count=%d | source_resolution=%dx%d | assigned_gpu=%s | flow_backend=%s | flow_backend_reason=%s",
        video_path,
        codec or "unknown",
        source_fps,
        frame_count,
        width,
        height,
        gpu_id,
        backend,
        backend_reason,
    )

    model_width, model_height = get_model_input_size()
    frames, decode_backend = load_sampled_sequence_frames(
        sequence_dir=video_path,
        width=model_width,
        height=model_height,
        source_fps=args.source_fps,
        target_sample_fps=args.target_sample_fps,
        max_source_frames=args.max_source_frames,
    )

    if len(frames) < 2:
        return {
            "video_path": str(video_path),
            "source_fps": source_fps,
            "frame_count": frame_count,
            "source_resolution": [width, height],
            "resized_resolution": [model_width, model_height],
            "sample_fps": args.target_sample_fps,
            "sample_count": len(frames),
            "decoder": "ffmpeg",
            "decode_backend": decode_backend,
            "ffmpeg_hwaccel": args.ffmpeg_hwaccel if gpu_id is not None else "none",
            "gpu_id": gpu_id,
            "flow_backend": backend,
            "flow_backend_reason": backend_reason,
            "codec": codec,
            "pair_count": 0,
            "mean_flow_magnitude_px": 0.0,
            "normalized_mean_optical_flow_magnitude": 0.0,
            "elapsed_s": round(time.time() - start_time, 3),
        }

    total_pairs = len(frames) - 1
    logging.info(
        "Starting optical flow for %s | sampled_frames=%d | pair_count=%d | flow_backend=%s | device=%s | raft_model=%s | model_input=%dx%d",
        video_path.name,
        len(frames),
        total_pairs,
        backend,
        device,
        args.raft_model,
        model_width,
        model_height,
    )

    pair_means = compute_pair_means_torch(
        frames=frames,
        model=model,
        transforms=transforms,
        device=device,
        args=args,
        video_name=video_path.name,
    )

    mean_flow_magnitude_px = float(np.mean(pair_means)) if pair_means else 0.0
    normalized_mean = mean_flow_magnitude_px / get_model_input_diagonal()
    elapsed_s = round(time.time() - start_time, 3)

    logging.info(
        "Finished video %s | decode_backend=%s | flow_backend=%s | sampled_frames=%d | pair_count=%d | mean_flow=%.6f | nmofm=%.8f | elapsed_s=%.3f",
        video_path.name,
        decode_backend,
        backend,
        len(frames),
        len(pair_means),
        mean_flow_magnitude_px,
        normalized_mean,
        elapsed_s,
    )

    return {
        "video_path": str(video_path),
        "source_fps": source_fps,
        "frame_count": frame_count,
        "source_resolution": [width, height],
        "resized_resolution": [model_width, model_height],
        "sample_fps": args.target_sample_fps,
        "sample_count": len(frames),
        "decoder": "ffmpeg",
        "decode_backend": decode_backend,
        "ffmpeg_hwaccel": args.ffmpeg_hwaccel if gpu_id is not None else "none",
        "gpu_id": gpu_id,
        "flow_backend": backend,
        "flow_backend_reason": backend_reason,
        "codec": codec,
        "pair_count": len(pair_means),
        "mean_flow_magnitude_px": mean_flow_magnitude_px,
        "normalized_mean_optical_flow_magnitude": normalized_mean,
        "elapsed_s": elapsed_s,
    }


def summarize_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    import numpy as np

    scores = [
        float(item["normalized_mean_optical_flow_magnitude"])
        for item in results
        if item.get("pair_count", 0) > 0
    ]
    pixel_means = [
        float(item["mean_flow_magnitude_px"])
        for item in results
        if item.get("pair_count", 0) > 0
    ]
    return {
        "video_count": len(results),
        "valid_video_count": len(scores),
        "target_resolution": list(get_model_input_size()),
        "normalization": "mean optical flow magnitude divided by target image diagonal",
        "dataset_mean_flow_magnitude_px": float(np.mean(pixel_means)) if pixel_means else 0.0,
        "dataset_nmofm": float(np.mean(scores)) if scores else 0.0,
    }


def print_results(results: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    for item in results:
        print(
            f"{item['video_path']}\n"
            f"  sample_fps={item['sample_fps']:.3f}, "
            f"sample_count={item['sample_count']}, "
            f"gpu_id={item.get('gpu_id')}, "
            f"flow_backend={item.get('flow_backend')}, "
            f"pair_count={item['pair_count']}, "
            f"mean_flow_magnitude_px={item['mean_flow_magnitude_px']:.6f}, "
            f"nmofm={item['normalized_mean_optical_flow_magnitude']:.8f}"
        )
    print(
        "\nDataset summary:\n"
        f"  video_count={summary['video_count']}, "
        f"valid_video_count={summary['valid_video_count']}, "
        f"dataset_mean_flow_magnitude_px={summary['dataset_mean_flow_magnitude_px']:.6f}, "
        f"dataset_nmofm={summary['dataset_nmofm']:.8f}"
    )


def save_json(output_path: Path, results: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "videos": list(results)}, f, ensure_ascii=False, indent=2)


def _compute_video_task(task: Tuple[str, argparse.Namespace, Optional[int]]) -> Dict[str, Any]:
    video_path_str, args, gpu_id = task
    setup_logging(args.log_level)
    logging.info(
        "Worker started | video=%s | assigned_gpu=%s",
        video_path_str,
        gpu_id,
    )
    return compute_nmofm_for_video(Path(video_path_str), args, gpu_id=gpu_id)


def run_tasks(videos: Sequence[Path], args: argparse.Namespace) -> List[Dict[str, Any]]:
    gpu_ids = parse_gpu_ids(args.gpus)
    log_runtime_gpu_info(gpu_ids)
    logging.info(
        "Starting run | sequence_count=%d | target_resolution=%dx%d | source_fps=%.3f | target_sample_fps=%.3f | gpu_ids=%s | workers_per_gpu=%d | raft_model=%s | flow_batch_size=%d | max_source_frames=%d",
        len(videos),
        get_model_input_size()[0],
        get_model_input_size()[1],
        args.source_fps,
        args.target_sample_fps,
        gpu_ids if gpu_ids else "CPU-only",
        args.workers_per_gpu,
        args.raft_model,
        args.flow_batch_size,
        args.max_source_frames,
    )
    warmup_raft_weights(args.raft_model)
    if not gpu_ids:
        return [compute_nmofm_for_video(video_path, args, gpu_id=None) for video_path in videos]

    slots: List[int] = []
    for gpu_id in gpu_ids:
        for _ in range(max(args.workers_per_gpu, 1)):
            slots.append(gpu_id)
    tasks = [
        (str(video_path), args, slots[index % len(slots)])
        for index, video_path in enumerate(videos)
    ]

    results: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
    logging.info("Launching process pool | worker_count=%d | gpu_slots=%s", len(slots), slots)
    mp_context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(slots),
        mp_context=mp_context,
    ) as executor:
        future_to_index = {
            executor.submit(_compute_video_task, task): index
            for index, task in enumerate(tasks)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
            logging.info(
                "Collected result %d/%d | video=%s",
                index + 1,
                len(tasks),
                results[index]["video_path"],
            )
    return [item for item in results if item is not None]


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    videos = discover_sequences(args.input, args.recursive)
    results = run_tasks(videos, args)
    summary = summarize_results(results)
    print_results(results, summary)
    if args.output_json is not None:
        save_json(args.output_json, results, summary)


if __name__ == "__main__":
    main()
