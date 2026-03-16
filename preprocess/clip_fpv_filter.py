#!/usr/bin/env python3
"""
Sample video frames every N seconds via ffmpeg, feed frames to CLIP in memory,
and flag non-FPV timestamps or videos. Results are streamed incrementally to JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

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

# 全局变量，记录是否已经保存过 debug 图像
_DEBUG_IMAGE_SAVED = False

@dataclass
class VideoReport:
    video_path: str
    sample_count: int
    bad_count: int
    bad_timestamps_s: List[float]


class IncrementalJsonWriter:
    """
    流式追加 JSON 写入器：保证 O(1) 的写入时间，不随文件增大而变慢，同时实时落盘防止崩溃丢失数据。
    """
    def __init__(self, path: Path, base_payload: dict):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "w", encoding="utf-8")
        
        # 写入 JSON 头部
        self.file.write("{\n")
        for k, v in base_payload.items():
            self.file.write(f'  "{k}": {json.dumps(v, ensure_ascii=False)},\n')
        self.file.write('  "videos":[\n')
        
        self.first_item = True

    def append(self, video_data: dict):
        # 如果不是第一个元素，在前面加一个逗号
        if not self.first_item:
            self.file.write(",\n")
        
        # 格式化当前视频的 JSON 并缩进，以保持文件美观
        video_json = json.dumps(video_data, ensure_ascii=False, indent=4)
        indented_json = "\n".join("    " + line for line in video_json.split("\n"))
        
        self.file.write(indented_json)
        self.file.flush()  # 【核心机制】：强制实时写入磁盘，防止程序崩溃丢失数据
        self.first_item = False

    def close(self):
        # 写入 JSON 尾部闭合符号
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
        # 此时传进来的已经是 RGB 格式
        image = Image.fromarray(frame_rgb)
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        if self.use_fp16:
            image_tensor = image_tensor.half()
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        
        probs = logits.squeeze(0)
        # 【核心修正】：累加所有正向 Prompt 的概率
        fpv_prob = float(probs[: self.fpvl].sum().item())
        return fpv_prob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLIP-based FPV filter (ffmpeg pipe) with Streaming JSON.")
    parser.add_argument("--input-path", type=Path, required=True, help="Video file or directory.")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories if input is a directory.")

    parser.add_argument("--interval-s", type=float, default=1.0, help="Sample interval in seconds.")
    parser.add_argument("--size", type=int, default=224, help="Square size for ffmpeg scaling/padding.")
    # 阈值设为 0.5 (正向概率超 50% 即可)
    parser.add_argument("--threshold", type=float, default=0.5, help="FPV probability threshold.")

    parser.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model name.")
    parser.add_argument("--pretrained", type=str, default="openai", help="OpenCLIP pretrained tag.")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda or cpu.")
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
        default=Path("output/trash_segments.json"),
        help="Output JSON path when mode=json.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--debug-every", type=int, default=50, help="Log every N frames when debug is on.")

    return parser.parse_args()


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("Missing ffmpeg in PATH.")


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
    logging.debug("Starting ffmpeg decode for %s (interval_s=%.3f, size=%d)", video_path, interval_s, size)
    fps = 1.0 / max(interval_s, 1e-6)
    vf = (
        f"fps={fps:.6f},"
        f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2"
    )
    cmd =[
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24", # 直接让 FFmpeg 输出 RGB24
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
            raise RuntimeError(f"ffmpeg failed for {video_path.name}: {stderr_text}")
        logging.debug("ffmpeg decode finished for %s (rc=%s)", video_path, rc)


def process_video(
    video_path: Path,
    classifier: ClipFPVClassifier,
    interval_s: float,
    size: int,
    threshold: float,
    debug_every: int,
) -> VideoReport:
    global _DEBUG_IMAGE_SAVED
    bad_times: List[float] =[]
    sample_count = 0
    for t, frame in iter_ffmpeg_frames(video_path, interval_s, size):
        sample_count += 1
        fpv_prob = classifier.fpv_prob(frame)
        
        if fpv_prob < threshold:
            bad_times.append(round(float(t), 3))
            if not _DEBUG_IMAGE_SAVED:
                debug_img_path = Path("output/debug_first_bad_frame.jpg")
                debug_img_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frame).save(debug_img_path)
                logging.warning(f"Saved the first 'bad' frame to {debug_img_path} for inspection. (Score: {fpv_prob:.2f})")
                _DEBUG_IMAGE_SAVED = True

        if debug_every > 0 and sample_count % debug_every == 0:
            logging.debug(
                "Processed %d frames for %s (t=%.3f, fpv_prob=%.4f)",
                sample_count,
                video_path,
                t,
                fpv_prob,
            )
    return VideoReport(str(video_path), sample_count, len(bad_times), bad_times)


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

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    
    # 【抗干扰 Prompt 组】
    fpv_prompts =[
        "A first-person view (FPV) drone flight over a landscape",
        "An aerial view from a drone flying fast",
        "A drone flying through a forest, building, or city",
        "Aerial photography of scenery from a moving drone",
        "Wide-angle GoPro action camera flying footage",     # 新增：广角运动相机视角
        "High speed motion blur aerial photography"
    ]
    non_fpv_prompts =[
        "A close-up of a person's face talking to the camera",
        "A text title screen, graphic, or sponsor logo",
        "A person standing on the ground holding a camera",  # 明确是人站在地上，防止与贴地飞行混淆
        "A completely black, blank, or heavily distorted screen",
        "A YouTube channel intro graphic"
    ]

    logging.info("Loading CLIP model: %s / %s", args.model, args.pretrained)
    classifier = ClipFPVClassifier(
        model_name=args.model,
        pretrained=args.pretrained,
        device=device,
        fpv_prompts=fpv_prompts,
        non_fpv_prompts=non_fpv_prompts,
        use_fp16=args.fp16,
    )
    logging.info("Model loaded")

    input_path = args.input_path
    if not input_path.exists():
        logging.error("Input path does not exist: %s (cwd=%s)", input_path, Path.cwd())
        return 2

    videos = list(iter_videos(input_path, args.recursive))
    logging.info("Found %d video(s) to process", len(videos))
    if not videos:
        logging.error("No videos matched.")
        return 3

    # ================= 核心：初始化流式 JSON 写入器 =================
    json_writer = None
    if args.mode == "json":
        base_payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "interval_s": args.interval_s,
            "threshold": args.threshold,
            "model": args.model,
            "pretrained": args.pretrained,
            "device": device
        }
        json_writer = IncrementalJsonWriter(args.output_json, base_payload)

    for video_path in videos:
        logging.info("Processing video: %s", video_path.name)
        report = process_video(
            video_path=video_path,
            classifier=classifier,
            interval_s=args.interval_s,
            size=args.size,
            threshold=args.threshold,
            debug_every=args.debug_every if args.debug else 0,
        )
        logging.info(
            "Finished video: %s (samples=%d, bad=%d)",
            video_path.name,
            report.sample_count,
            report.bad_count,
        )
        
        # 实时输出处理
        if args.mode == "print":
            if report.bad_count > 0:
                print(report.video_path)
        elif args.mode == "json" and json_writer:
            # 无论好坏都记录，方便追踪进度，如果只想记录有垃圾的，可以加上 if report.bad_count > 0:
            video_dict = {
                "video_path": report.video_path,
                "sample_count": report.sample_count,
                "bad_count": report.bad_count,
                "bad_timestamps_s": report.bad_timestamps_s,
            }
            json_writer.append(video_dict)

    # 优雅关闭文件
    if json_writer:
        json_writer.close()
        logging.info("Wrote completely finished JSON: %s", args.output_json)
        
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.info("Interrupted by user. (Data already streamed to disk is safe!)")
        sys.exit(1)