#!/usr/bin/env python3
"""
Sample video frames every N seconds via ffmpeg, feed frames to CLIP in memory,
and flag non-FPV timestamps or videos. Results are streamed incrementally to JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: torch. Please install PyTorch.") from exc

try:
    from PIL import Image
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: pillow. Please install Pillow.") from exc

try:
    import open_clip
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: open_clip_torch. Please install open_clip_torch.") from exc


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Global flag to avoid dumping too many debug frames.
_DEBUG_IMAGE_SAVED = False
_WORKER_CLASSIFIER: Optional["ClipFPVClassifier"] = None
_WORKER_DEBUG_EVERY = 0


@dataclass
class VideoReport:
    video_path: str
    sample_count: int
    bad_count: int
    bad_timestamps_s: List[List[float]]
    bad_frame_scores: List[dict]
    valid_count: int
    valid_timestamps_s: List[List[float]]


def merge_consecutive_bad_timestamps(
    timestamps: List[float], interval_s: float, precision: int = 3
) -> List[List[float]]:
    if not timestamps:
        return []

    sorted_ts = sorted(set(float(t) for t in timestamps))
    merged: List[List[float]] = []
    start = sorted_ts[0]
    end = sorted_ts[0]
    eps = max(1e-6, interval_s * 0.1)
    for ts in sorted_ts[1:]:
        if ts - end <= interval_s + eps:
            end = ts
        else:
            merged.append([round(start, precision), round(end, precision)])
            start = ts
            end = ts
    merged.append([round(start, precision), round(end, precision)])
    return merged


def drop_single_frame_ranges(
    ranges: List[List[float]], interval_s: float, precision: int = 3
) -> List[List[float]]:
    # 单帧区间长度约为 0，多帧区间长度约为 interval_s 的整数倍
    min_span = max(interval_s * 0.5, 1e-6)
    out: List[List[float]] = []
    for s, e in ranges:
        if (float(e) - float(s)) >= min_span:
            out.append([round(float(s), precision), round(float(e), precision)])
    return out


def complement_ranges(
    excluded: List[List[float]], total_end_s: float, precision: int = 3
) -> List[List[float]]:
    total_end_s = float(total_end_s)
    if total_end_s < 0:
        return []

    if not excluded:
        return [[0.0, round(total_end_s, precision)]]

    eps = 1e-6
    merged = sorted(excluded, key=lambda x: float(x[0]))
    valid: List[List[float]] = []
    cursor = 0.0
    for s, e in merged:
        s = max(0.0, float(s))
        e = min(total_end_s, float(e))
        if s > cursor + eps:
            valid.append([round(cursor, precision), round(s, precision)])
        cursor = max(cursor, e)
    if cursor < total_end_s - eps:
        valid.append([round(cursor, precision), round(total_end_s, precision)])
    if not valid and total_end_s == 0.0:
        valid.append([0.0, 0.0])
    return valid


class IncrementalJsonWriter:
    """
    Stream-write JSON to disk in O(1) per append and flush continuously.
    """

    def __init__(self, path: Path, base_payload: dict):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "w", encoding="utf-8")

        self.file.write("{\n")
        for k, v in base_payload.items():
            self.file.write(f'  "{k}": {json.dumps(v, ensure_ascii=False)},\n')
        self.file.write('  "videos":[\n')

        self.first_item = True

    def append(self, video_data: dict):
        if not self.first_item:
            self.file.write(",\n")

        video_json = json.dumps(video_data, ensure_ascii=False, indent=4)
        indented_json = "\n".join("    " + line for line in video_json.split("\n"))

        self.file.write(indented_json)
        self.file.flush()
        self.first_item = False

    def close(self):
        self.file.write("\n  ]\n}\n")
        self.file.close()


class ClipFPVClassifier:
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: str,
        fpv_prompts: List[str],
        non_fpv_prompts: List[str],
        use_fp16: bool,
    ) -> None:
        self.device = device
        self.use_fp16 = bool(use_fp16 and device.startswith("cuda"))
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.model.eval()
        if self.use_fp16:
            self.model = self.model.half()
        tokenizer = open_clip.get_tokenizer(model_name)
        prompts = fpv_prompts + non_fpv_prompts
        text = tokenizer(prompts).to(device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features
        self.fpvl = len(fpv_prompts)

    def fpv_prob(self, frame_rgb: np.ndarray) -> float:
        return self.fpv_prob_batch([frame_rgb])[0]

    def fpv_prob_batch(self, frames_rgb: Sequence[np.ndarray]) -> List[float]:
        if not frames_rgb:
            return []

        images = [self.preprocess(Image.fromarray(frame_rgb)) for frame_rgb in frames_rgb]
        image_tensor = torch.stack(images, dim=0).to(self.device)
        if self.use_fp16:
            image_tensor = image_tensor.half()
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)

        probs = logits[:, : self.fpvl].sum(dim=-1)
        return [float(prob.item()) for prob in probs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLIP-based FPV filter (ffmpeg pipe) with Streaming JSON.")
    parser.add_argument("--input-path", type=Path, required=True, help="Video file or directory.")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories if input is a directory.")

    parser.add_argument("--interval-s", type=float, default=0.2, help="Sample interval in seconds.")
    parser.add_argument("--size", type=int, default=224, help="Square size for ffmpeg scaling/padding.")
    parser.add_argument("--threshold", type=float, default=0.2, help="FPV probability threshold.")

    parser.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model name.")
    parser.add_argument("--pretrained", type=str, default="openai", help="OpenCLIP pretrained tag.")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda or cpu.")
    parser.add_argument(
        "--devices",
        type=str,
        default="",
        help="Comma-separated devices for parallel workers, e.g. cuda:0,cuda:1. Empty uses --device.",
    )
    parser.add_argument(
        "--workers-per-device",
        type=int,
        default=32,
        help="How many video workers to launch per device.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="How many frames to infer together in one CLIP forward pass.",
    )
    parser.add_argument("--fp16", action="store_true", help="Use FP16 on CUDA.")

    parser.add_argument(
        "--mode",
        type=str,
        default="json",
        choices=["json", "print"],
        help="Output mode: json (timestamps) or print (video path).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/trash_segments_5fps.json"),
        help="Output JSON path when mode=json.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--debug-every", type=int, default=50, help="Log every N frames when debug is on.")

    return parser.parse_args()


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("Missing ffmpeg/ffprobe in PATH.")


def get_video_codec(video_path: Path) -> Optional[str]:
    try:
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
            check=True,
        )
        codec = result.stdout.strip().lower()
        return codec or None
    except Exception:
        return None


def iter_videos(input_path: Path, recursive: bool) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    if recursive:
        for p in input_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                yield p
    else:
        for p in input_path.iterdir():
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                yield p


def iter_ffmpeg_frames(video_path: Path, interval_s: float, size: int) -> Iterable[Tuple[float, np.ndarray]]:
    codec = get_video_codec(video_path)
    if codec:
        logging.debug("Detected codec for %s: %s", video_path, codec)
    logging.debug("Starting ffmpeg decode for %s (interval_s=%.3f, size=%d)", video_path, interval_s, size)
    fps = 1.0 / max(interval_s, 1e-6)
    vf = (
        f"fps={fps:.6f},"
        f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2"
    )

    class FfmpegError(RuntimeError):
        def __init__(self, message: str, stderr: str) -> None:
            super().__init__(message)
            self.stderr = stderr

    def _iter_with_args(extra_args: List[str]) -> Iterable[Tuple[float, np.ndarray]]:
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            *extra_args,
            "-i",
            str(video_path),
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]

        frame_bytes = size * size * 3
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            assert proc.stdout is not None
            idx = 0
            while True:
                raw = proc.stdout.read(frame_bytes)
                if not raw or len(raw) < frame_bytes:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((size, size, 3))
                t = idx * interval_s
                idx += 1
                yield t, frame
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            stderr_text = ""
            if proc.stderr is not None:
                stderr_text = proc.stderr.read().decode("utf-8", errors="ignore").strip()
                proc.stderr.close()
            rc = proc.wait()
            if rc != 0 and "Broken pipe" not in stderr_text:
                raise FfmpegError(f"ffmpeg failed for {video_path.name}: {stderr_text}", stderr_text)
            logging.debug("ffmpeg decode finished for %s (rc=%s)", video_path, rc)

    extra_args: List[str] = []
    if codec == "av1":
        extra_args = ["-hwaccel", "none"]

    try:
        yield from _iter_with_args(extra_args)
    except FfmpegError as exc:
        err = exc.stderr.lower()
        if ("av1" in err) and ("hardware accelerated" in err or "hwaccel" in err):
            logging.warning("AV1 hwaccel failed, retrying with software decode for %s", video_path)
            yield from _iter_with_args(["-hwaccel", "none"])
        else:
            raise


def process_video(
    video_path: Path,
    classifier: ClipFPVClassifier,
    interval_s: float,
    size: int,
    threshold: float,
    debug_every: int,
    batch_size: int,
) -> VideoReport:
    global _DEBUG_IMAGE_SAVED
    bad_times: List[float] = []
    bad_frame_scores: List[dict] = []
    sample_count = 0
    batch_frames: List[np.ndarray] = []
    batch_times: List[float] = []

    def flush_batch() -> None:
        nonlocal sample_count
        global _DEBUG_IMAGE_SAVED
        if not batch_frames:
            return

        probs = classifier.fpv_prob_batch(batch_frames)
        for t, frame, fpv_prob in zip(batch_times, batch_frames, probs):
            sample_count += 1

            if fpv_prob < threshold:
                rounded_t = round(float(t), 3)
                rounded_score = round(float(fpv_prob), 6)
                bad_times.append(rounded_t)
                bad_frame_scores.append(
                    {
                        "timestamp_s": rounded_t,
                        "score": rounded_score,
                    }
                )
                if not _DEBUG_IMAGE_SAVED:
                    debug_img_path = Path("output/debug_first_bad_frame.jpg")
                    debug_img_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(frame).save(debug_img_path)
                    logging.warning(
                        "Saved the first 'bad' frame to %s for inspection. (Score: %.2f)",
                        debug_img_path,
                        fpv_prob,
                    )
                    _DEBUG_IMAGE_SAVED = True

            if debug_every > 0 and sample_count % debug_every == 0:
                logging.debug(
                    "Processed %d frames for %s (t=%.3f, fpv_prob=%.4f)",
                    sample_count,
                    video_path,
                    t,
                    fpv_prob,
                )

        batch_frames.clear()
        batch_times.clear()

    effective_batch_size = max(1, batch_size)
    for t, frame in iter_ffmpeg_frames(video_path, interval_s, size):
        batch_times.append(t)
        batch_frames.append(frame)
        if len(batch_frames) >= effective_batch_size:
            flush_batch()
    flush_batch()
    merged_bad_ranges = merge_consecutive_bad_timestamps(bad_times, interval_s=interval_s, precision=3)
    filtered_bad_ranges = drop_single_frame_ranges(merged_bad_ranges, interval_s=interval_s, precision=3)

    total_end_s = round(max(0, sample_count - 1) * interval_s, 3)
    valid_ranges = complement_ranges(filtered_bad_ranges, total_end_s=total_end_s, precision=3)

    return VideoReport(
        str(video_path),
        sample_count,
        len(filtered_bad_ranges),
        filtered_bad_ranges,
        bad_frame_scores,
        len(valid_ranges),
        valid_ranges,
    )


def resolve_devices(device_arg: str, devices_arg: str) -> List[str]:
    if devices_arg.strip():
        devices = [item.strip() for item in devices_arg.split(",") if item.strip()]
        if not devices:
            raise SystemExit("`--devices` was provided but no valid device was parsed.")
        return devices

    device = device_arg.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and torch.cuda.is_available():
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
    return [device]


def build_worker_devices(devices: Sequence[str], workers_per_device: int) -> List[str]:
    count = max(1, workers_per_device)
    worker_devices: List[str] = []
    for device in devices:
        worker_devices.extend([device] * count)
    return worker_devices


def init_worker(
    model_name: str,
    pretrained: str,
    device: str,
    fpv_prompts: List[str],
    non_fpv_prompts: List[str],
    use_fp16: bool,
    debug_every: int,
) -> None:
    global _WORKER_CLASSIFIER, _WORKER_DEBUG_EVERY
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(device)
    _WORKER_DEBUG_EVERY = debug_every
    _WORKER_CLASSIFIER = ClipFPVClassifier(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        fpv_prompts=fpv_prompts,
        non_fpv_prompts=non_fpv_prompts,
        use_fp16=use_fp16,
    )


def worker_process_video(
    video_path_str: str,
    interval_s: float,
    size: int,
    threshold: float,
    batch_size: int,
) -> VideoReport:
    if _WORKER_CLASSIFIER is None:
        raise RuntimeError("Worker classifier is not initialized.")
    return process_video(
        video_path=Path(video_path_str),
        classifier=_WORKER_CLASSIFIER,
        interval_s=interval_s,
        size=size,
        threshold=threshold,
        debug_every=_WORKER_DEBUG_EVERY,
        batch_size=batch_size,
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("Starting CLIP FPV filter (Streaming JSON Mode)")
    logging.info("Input path: %s", args.input_path)
    ensure_ffmpeg()

    devices = resolve_devices(args.device, args.devices)
    worker_devices = build_worker_devices(devices, args.workers_per_device)
    logging.info("Using device(s): %s", ", ".join(devices))
    logging.info("Launching %d worker(s)", len(worker_devices))

    #v1
    fpv_prompts = [
        "A first-person view (FPV) drone flight over a landscape",
        "An aerial view from a drone flying fast",
        "A drone flying through a forest, building, or city",
        "Aerial photography of scenery from a moving drone",
        "Wide-angle GoPro action camera flying footage",
        "High speed motion blur aerial photography",
    ]

    non_fpv_prompts = [
        "A close-up of a person's face talking to the camera",
        "A text title screen, graphic, or sponsor logo",
        "A person standing on the ground holding a camera",
        "A black, blank screen",
        "A YouTube channel intro graphic",
        "A frame with subtitles or on-screen captions",
        "A nearly blank frame during scene transition",
    ]

    input_path = args.input_path
    if not input_path.exists():
        logging.error("Input path does not exist: %s (cwd=%s)", input_path, Path.cwd())
        return 2

    videos = list(iter_videos(input_path, args.recursive))
    logging.info("Found %d video(s) to process", len(videos))
    if not videos:
        logging.error("No videos matched.")
        return 3

    json_writer = None
    if args.mode == "json":
        base_payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "interval_s": args.interval_s,
            "threshold": args.threshold,
            "output_json": str(args.output_json),
            "model": args.model,
            "pretrained": args.pretrained,
            "device": devices[0] if len(devices) == 1 else ",".join(devices),
            "devices": devices,
            "workers_per_device": args.workers_per_device,
            "batch_size": args.batch_size,
        }
        json_writer = IncrementalJsonWriter(args.output_json, base_payload)

    if len(worker_devices) == 1:
        logging.info("Loading CLIP model: %s / %s", args.model, args.pretrained)
        classifier = ClipFPVClassifier(
            model_name=args.model,
            pretrained=args.pretrained,
            device=worker_devices[0],
            fpv_prompts=fpv_prompts,
            non_fpv_prompts=non_fpv_prompts,
            use_fp16=args.fp16,
        )
        logging.info("Model loaded")

        for video_path in videos:
            logging.info("Processing video: %s", video_path.name)
            report = process_video(
                video_path=video_path,
                classifier=classifier,
                interval_s=args.interval_s,
                size=args.size,
                threshold=args.threshold,
                debug_every=args.debug_every if args.debug else 0,
                batch_size=args.batch_size,
            )
            logging.info(
                "Finished video: %s (samples=%d, bad=%d)",
                video_path.name,
                report.sample_count,
                report.bad_count,
            )

            if args.mode == "print":
                if report.valid_count > 0:
                    print(report.video_path)
            elif args.mode == "json" and json_writer:
                video_dict = {
                    "video_path": report.video_path,
                    "sample_count": report.sample_count,
                    "bad_count": report.bad_count,
                    "bad_timestamps_s": report.bad_timestamps_s,
                    "bad_frame_scores": report.bad_frame_scores,
                    "valid_count": report.valid_count,
                    "valid_timestamps_s": report.valid_timestamps_s,
                }
                json_writer.append(video_dict)
    else:
        ctx = mp.get_context("spawn")
        executors = [
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=ctx,
                initializer=init_worker,
                initargs=(
                    args.model,
                    args.pretrained,
                    worker_device,
                    fpv_prompts,
                    non_fpv_prompts,
                    args.fp16,
                    args.debug_every if args.debug else 0,
                ),
            )
            for worker_device in worker_devices
        ]

        try:
            futures = {}
            for index, video_path in enumerate(videos):
                executor = executors[index % len(executors)]
                logging.info(
                    "Queued video: %s -> worker %d (%s)",
                    video_path.name,
                    index % len(executors),
                    worker_devices[index % len(executors)],
                )
                future = executor.submit(
                    worker_process_video,
                    str(video_path),
                    args.interval_s,
                    args.size,
                    args.threshold,
                    args.batch_size,
                )
                futures[future] = video_path

            for future in as_completed(futures):
                video_path = futures[future]
                report = future.result()
                logging.info(
                    "Finished video: %s (samples=%d, bad=%d)",
                    video_path.name,
                    report.sample_count,
                    report.bad_count,
                )

                if args.mode == "print":
                    if report.valid_count > 0:
                        print(report.video_path)
                elif args.mode == "json" and json_writer:
                    video_dict = {
                        "video_path": report.video_path,
                        "sample_count": report.sample_count,
                        "bad_count": report.bad_count,
                        "bad_timestamps_s": report.bad_timestamps_s,
                        "bad_frame_scores": report.bad_frame_scores,
                        "valid_count": report.valid_count,
                        "valid_timestamps_s": report.valid_timestamps_s,
                    }
                    json_writer.append(video_dict)
        finally:
            for executor in executors:
                executor.shutdown(wait=True, cancel_futures=False)

    if json_writer:
        json_writer.close()
        logging.info("Wrote completely finished JSON: %s", args.output_json)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.info("Interrupted by user. (Data already streamed to output if in JSON mode.)")
        sys.exit(1)
