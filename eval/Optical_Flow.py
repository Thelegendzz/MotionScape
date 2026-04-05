#!/usr/bin/env python3
"""
Compute Normalized Mean Optical Flow Magnitude (NMOFM) for videos with PyTorch RAFT.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import multiprocessing
import os
import shutil
import subprocess
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute normalized mean optical flow magnitude for videos with PyTorch."
    )
    parser.add_argument("--input", type=Path, required=True, help="Video file or directory.")
    parser.add_argument("--sample-fps", type=float, default=5.0, help="Sampling FPS.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search videos.")
    parser.add_argument(
        "--segment-json",
        type=Path,
        default=None,
        help="Optional JSON path containing per-video timestamp segments.",
    )
    parser.add_argument(
        "--segment-key",
        type=str,
        default="valid_timestamps_s",
        help="Segment key to read from --segment-json. Default: valid_timestamps_s.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/optical_flow_sample.json"),
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
        help="Optional ffmpeg hardware acceleration for decode.",
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


def discover_videos(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")
    if not input_path.is_dir():
        raise SystemExit(f"Unsupported input path: {input_path}")
    pattern = "**/*" if recursive else "*"
    videos = [
        path for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]
    if not videos:
        raise SystemExit(f"No video files found under: {input_path}")
    return sorted(videos)


def load_segment_map(segment_json: Optional[Path], segment_key: str) -> Dict[str, List[Tuple[float, float]]]:
    if segment_json is None:
        return {}
    if not segment_json.exists():
        raise SystemExit(f"Segment JSON not found: {segment_json}")

    with segment_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise SystemExit("Segment JSON must contain a top-level 'videos' list.")

    segment_map: Dict[str, List[Tuple[float, float]]] = {}
    for item in videos:
        if not isinstance(item, dict):
            continue
        video_name = str(item.get("video_path") or item.get("video") or item.get("video_name") or "")
        if not video_name:
            continue
        raw_segments = item.get(segment_key) or []
        normalized: List[Tuple[float, float]] = []
        for raw_segment in raw_segments:
            if not isinstance(raw_segment, (list, tuple)) or len(raw_segment) < 2:
                continue
            try:
                start_s = float(raw_segment[0])
                end_s = float(raw_segment[1])
            except (TypeError, ValueError):
                continue
            if end_s < start_s:
                start_s, end_s = end_s, start_s
            normalized.append((start_s, end_s))
        normalized.sort()
        segment_map[Path(video_name).name] = normalized
        segment_map[video_name] = normalized
    return segment_map


def require_ffmpeg_tools() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("Missing ffmpeg/ffprobe in PATH.")


def parse_ffprobe_rate(value: Any) -> float:
    text = str(value or "0")
    if "/" in text:
        left, right = text.split("/", 1)
        try:
            numerator = float(left)
            denominator = float(right)
        except ValueError:
            return 0.0
        if denominator == 0:
            return 0.0
        return numerator / denominator
    try:
        return float(text)
    except ValueError:
        return 0.0


def get_video_codec(video_path: Path) -> str:
    require_ffmpeg_tools()
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logging.warning("ffprobe codec query failed for %s: %s", video_path, result.stderr.strip())
        return ""
    return result.stdout.strip().lower()


def probe_video(video_path: Path) -> Dict[str, Any]:
    require_ffmpeg_tools()
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in: {video_path}")
    stream = streams[0]
    fps = parse_ffprobe_rate(stream.get("avg_frame_rate"))
    frame_count_raw = stream.get("nb_frames")
    frame_count = int(frame_count_raw) if frame_count_raw not in (None, "N/A") else 0
    duration_raw = stream.get("duration")
    duration_s = float(duration_raw) if duration_raw not in (None, "N/A") else 0.0
    if duration_s <= 0.0 and frame_count > 0 and fps > 0:
        duration_s = frame_count / fps
    if frame_count <= 0 and duration_s > 0.0 and fps > 0:
        frame_count = int(round(duration_s * fps))
    return {
        "source_fps": fps,
        "frame_count": frame_count,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration_s": duration_s,
    }


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


def segment_contains_time(segments: Sequence[Tuple[float, float]], t: float) -> bool:
    for start_s, end_s in segments:
        if start_s <= t < end_s:
            return True
        if t < start_s:
            return False
    return False


def iter_ffmpeg_sampled_rgb_frames(
    video_path: Path,
    width: int,
    height: int,
    sample_fps: float,
    gpu_id: Optional[int],
    ffmpeg_hwaccel: str,
) -> Tuple[List[Any], List[float], str]:
    import numpy as np

    require_ffmpeg_tools()
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive.")

    frame_bytes = width * height * 3
    vf = f"fps={sample_fps:.6f},scale={width}:{height},format=rgb24"
    codec = get_video_codec(video_path)
    decode_modes: List[Tuple[str, Optional[int]]] = [("cpu", None)]
    if (
        ffmpeg_hwaccel == "cuda"
        and gpu_id is not None
        and codec in CUDA_DECODE_PREFERRED_CODECS
    ):
        decode_modes = [("cuda", gpu_id), ("cpu", None)]
    if (
        ffmpeg_hwaccel == "cuda"
        and gpu_id is not None
        and codec not in CUDA_DECODE_PREFERRED_CODECS
    ):
        logging.info(
            "Skipping ffmpeg CUDA decode for %s | codec=%s | using CPU decode directly.",
            video_path.name,
            codec or "unknown",
        )

    last_error = ""
    for decode_mode, decode_gpu_id in decode_modes:
        cmd = ["ffmpeg", "-v", "error"]
        if decode_mode == "cuda" and decode_gpu_id is not None:
            cmd.extend(["-hwaccel", "cuda", "-hwaccel_device", str(decode_gpu_id)])
        cmd.extend([
            "-i",
            str(video_path),
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frames: List[Any] = []
        timestamps: List[float] = []
        sample_interval = 1.0 / sample_fps
        try:
            assert proc.stdout is not None
            index = 0
            while True:
                raw = proc.stdout.read(frame_bytes)
                if not raw or len(raw) < frame_bytes:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                frames.append(frame)
                timestamps.append(index * sample_interval)
                index += 1
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            stderr_text = ""
            if proc.stderr is not None:
                stderr_text = proc.stderr.read().decode("utf-8", errors="ignore").strip()
                proc.stderr.close()
            rc = proc.wait()

        if rc == 0 or "Broken pipe" in stderr_text:
            return frames, timestamps, decode_mode

        last_error = stderr_text
        if decode_mode == "cuda":
            logging.warning(
                "ffmpeg CUDA decode failed for %s, falling back to CPU decode. stderr=%s",
                video_path.name,
                stderr_text,
            )

    raise RuntimeError(f"ffmpeg failed for {video_path.name}: {last_error}")


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
    prev_frames: Sequence[Any],
    curr_frames: Sequence[Any],
    model: Any,
    transforms: Any,
    device: Any,
    args: argparse.Namespace,
    video_name: str,
) -> List[float]:
    import torch

    pair_means: List[float] = []
    total_pairs = min(len(prev_frames), len(curr_frames))
    batch_size = max(1, args.flow_batch_size)

    with torch.inference_mode():
        for start in range(0, total_pairs, batch_size):
            end = min(start + batch_size, total_pairs)
            prev_batch = []
            curr_batch = []
            for pair_index in range(start, end):
                prev_tensor = torch.from_numpy(prev_frames[pair_index]).permute(2, 0, 1).contiguous()
                curr_tensor = torch.from_numpy(curr_frames[pair_index]).permute(2, 0, 1).contiguous()
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
    segment_map: Optional[Dict[str, List[Tuple[float, float]]]] = None,
) -> Dict[str, Any]:
    import numpy as np

    backend, device, model, transforms, backend_reason = get_torch_backend(gpu_id, args)
    start_time = time.time()

    metadata = probe_video(video_path)
    source_fps = float(metadata["source_fps"])
    frame_count = int(metadata["frame_count"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    codec = get_video_codec(video_path)

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
    frames, timestamps, decode_backend = iter_ffmpeg_sampled_rgb_frames(
        video_path=video_path,
        width=model_width,
        height=model_height,
        sample_fps=args.sample_fps,
        gpu_id=gpu_id,
        ffmpeg_hwaccel=args.ffmpeg_hwaccel,
    )

    active_segments: List[Tuple[float, float]] = []
    if segment_map is not None:
        active_segments = segment_map.get(video_path.name) or segment_map.get(str(video_path)) or []

    selected_prev_frames: List[Any] = []
    selected_curr_frames: List[Any] = []
    selected_pair_count = 0
    if active_segments:
        for prev_frame, curr_frame, prev_t, curr_t in zip(frames[:-1], frames[1:], timestamps[:-1], timestamps[1:]):
            if segment_contains_time(active_segments, prev_t) and segment_contains_time(active_segments, curr_t):
                selected_prev_frames.append(prev_frame)
                selected_curr_frames.append(curr_frame)
        selected_pair_count = len(selected_prev_frames)
    else:
        selected_prev_frames = list(frames[:-1])
        selected_curr_frames = list(frames[1:])
        selected_pair_count = len(selected_prev_frames)

    if selected_pair_count < 1:
        return {
            "video_path": str(video_path),
            "source_fps": source_fps,
            "frame_count": frame_count,
            "source_resolution": [width, height],
            "resized_resolution": [model_width, model_height],
            "sample_fps": args.sample_fps,
            "sample_count": len(frames),
            "segment_key": args.segment_key if active_segments else None,
            "segment_count": len(active_segments),
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

    total_pairs = selected_pair_count
    logging.info(
        "Starting optical flow for %s | sampled_frames=%d | selected_pair_count=%d | segment_count=%d | flow_backend=%s | device=%s | raft_model=%s | model_input=%dx%d",
        video_path.name,
        len(frames),
        total_pairs,
        len(active_segments),
        backend,
        device,
        args.raft_model,
        model_width,
        model_height,
    )

    pair_means = compute_pair_means_torch(
        prev_frames=selected_prev_frames,
        curr_frames=selected_curr_frames,
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
        "sample_fps": args.sample_fps,
        "sample_count": len(frames),
        "segment_key": args.segment_key if active_segments else None,
        "segment_count": len(active_segments),
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


def _compute_video_task(
    task: Tuple[str, argparse.Namespace, Optional[int], Dict[str, List[Tuple[float, float]]]]
) -> Dict[str, Any]:
    video_path_str, args, gpu_id, segment_map = task
    setup_logging(args.log_level)
    logging.info(
        "Worker started | video=%s | assigned_gpu=%s",
        video_path_str,
        gpu_id,
    )
    return compute_nmofm_for_video(
        Path(video_path_str),
        args,
        gpu_id=gpu_id,
        segment_map=segment_map,
    )


def run_tasks(videos: Sequence[Path], args: argparse.Namespace) -> List[Dict[str, Any]]:
    gpu_ids = parse_gpu_ids(args.gpus)
    segment_map = load_segment_map(args.segment_json, args.segment_key)
    log_runtime_gpu_info(gpu_ids)
    logging.info(
        "Starting run | video_count=%d | target_resolution=%dx%d | sample_fps=%.3f | gpu_ids=%s | workers_per_gpu=%d | ffmpeg_hwaccel=%s | raft_model=%s | flow_batch_size=%d | segment_json=%s | segment_key=%s",
        len(videos),
        get_model_input_size()[0],
        get_model_input_size()[1],
        args.sample_fps,
        gpu_ids if gpu_ids else "CPU-only",
        args.workers_per_gpu,
        args.ffmpeg_hwaccel,
        args.raft_model,
        args.flow_batch_size,
        args.segment_json,
        args.segment_key,
    )
    warmup_raft_weights(args.raft_model)
    if not gpu_ids:
        return [
            compute_nmofm_for_video(video_path, args, gpu_id=None, segment_map=segment_map)
            for video_path in videos
        ]

    slots: List[int] = []
    for gpu_id in gpu_ids:
        for _ in range(max(args.workers_per_gpu, 1)):
            slots.append(gpu_id)
    tasks = [
        (str(video_path), args, slots[index % len(slots)], segment_map)
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
    videos = discover_videos(args.input, args.recursive)
    results = run_tasks(videos, args)
    summary = summarize_results(results)
    print_results(results, summary)
    if args.output_json is not None:
        save_json(args.output_json, results, summary)


if __name__ == "__main__":
    main()
