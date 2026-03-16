#!/usr/bin/env python3
"""
Sample video at N fps, batch M frames per segment, send to Gemini 1.5 Pro API,
and write one JSON output per segment into a single report.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: opencv-python(-headless).") from exc

try:
    from dotenv import load_dotenv
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: python-dotenv. Please run `pip install python-dotenv`.") from exc


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass
class SegmentResult:
    segment_index: int
    start_s: float
    end_s: float
    frame_count: int
    response_text: str
    response_json: Optional[object]
    error: str


@dataclass
class VideoResult:
    video_path: str
    fps: float
    duration_s: float
    sampled_frames: int
    segments: List[SegmentResult]
    error: str


class JsonStreamWriter:
    def __init__(self, path: Path, header: Dict):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("w", encoding="utf-8")
        self.first_video = True
        self.first_segment = True
        header_dump = json.dumps(header, ensure_ascii=False, indent=2)
        # Write header without trailing }
        self.f.write(header_dump[:-2].rstrip())
        self.f.write(',\n  "videos": [\n')

    def start_video(self, video_path: str, fps: float, duration_s: float) -> None:
        if not self.first_video:
            self.f.write(",\n")
        self.first_video = False
        self.first_segment = True
        self.f.write("    {\n")
        self.f.write(f'      "video_path": {json.dumps(video_path, ensure_ascii=False)},\n')
        self.f.write(f'      "fps": {fps},\n')
        self.f.write(f'      "duration_s": {duration_s},\n')
        self.f.write('      "segments": [\n')

    def write_segment(self, segment: Dict) -> None:
        if not self.first_segment:
            self.f.write(",\n")
        self.first_segment = False
        self.f.write("        ")
        self.f.write(json.dumps(segment, ensure_ascii=False))
        self.f.flush()

    def end_video(self, sampled_frames: int, error: str) -> None:
        if not self.first_segment:
            self.f.write("\n")
        self.f.write("      ],\n")
        self.f.write(f'      "sampled_frames": {sampled_frames},\n')
        self.f.write(f'      "error": {json.dumps(error, ensure_ascii=False)}\n')
        self.f.write("    }")
        self.f.flush()

    def close(self) -> None:
        self.f.write("\n  ]\n}\n")
        self.f.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample video, send batches to Gemini 1.5 Pro, output JSON.")
    parser.add_argument("--input-path", type=Path, required=True, help="Video file or directory.")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories if input is a directory.")
    parser.add_argument("--output-json", type=Path, default=Path("output/gemini_actions.json"), help="Output JSON path.")

    parser.add_argument("--sample-fps", type=float, default=5.0, help="Sampling fps.")
    parser.add_argument("--chunk-frames", type=int, default=30, help="Frames per request.")
    parser.add_argument("--model", type=str, default="gemini-2.0-pro-exp", help="Gemini model name.")
    parser.add_argument("--prompt-file", type=Path, default=Path("prompt_action.txt"), help="Path to the text file containing the prompt.")
    parser.add_argument("--timeout", type=int, default=120, help="API timeout (seconds).")
    parser.add_argument("--max-retries", type=int, default=2, help="Retry count on transient errors.")
    parser.add_argument("--min-interval", type=float, default=2.0, help="Minimum seconds between API requests.")
    parser.add_argument("--backoff-base", type=float, default=2.0, help="Retry backoff base (seconds).")
    parser.add_argument("--backoff-max", type=float, default=60.0, help="Retry backoff max (seconds).")
    parser.add_argument("--response-json", action="store_true", help="Ask API to return application/json.")
    parser.add_argument("--max-segments", type=int, default=0, help="Limit segments per video (0 = no limit).")
    return parser.parse_args()


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


def iter_sampled_frames(video_path: Path, sample_fps: float) -> Tuple[Iterable[Tuple[float, object]], float, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0

    sample_interval = 1.0 / max(sample_fps, 1e-6)
    next_sample_time = 0.0
    frame_index = 0

    def generator() -> Iterable[Tuple[float, any]]:
        nonlocal next_sample_time, frame_index
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fps > 0:
                t = frame_index / fps
            else:
                t = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            if t + 1e-6 >= next_sample_time:
                yield t, frame
                next_sample_time += sample_interval
            frame_index += 1
        cap.release()

    return generator(), fps, duration_s


def encode_jpeg_base64(frame, max_dim=720) -> str:
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
    ok, buf = cv2.imencode(".jpg", frame,[int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def call_gemini(
    api_key: str,
    model: str,
    prompt: str,
    images_b64: List[str],
    timeout_s: int,
    want_json: bool,
    max_retries: int,
    min_interval_s: float,
    backoff_base_s: float,
    backoff_max_s: float,
    rate_state: Dict[str, float],
) -> Tuple[str, Dict]:
    # 1. 确保 Key 是纯净的 (去掉任何换行和不可见字符)
    safe_api_key = api_key.strip()
    
    url = "https://api.aiearth.dev/v1/chat/completions"
    
    # 2. 构建符合 OpenAI 兼容协议的多模态 payload
    content_list = [{"type": "text", "text": prompt}]
    for img in images_b64:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img}"}
        })

    payload = {
        "model": model, 
        "messages": [{"role": "user", "content": content_list}],
        "temperature": 0.7
    }
    
    # 3. 最关键的 Header：添加 User-Agent 模拟浏览器，这是解决 403 的核心
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {safe_api_key}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=data, headers=headers, method="POST")

    def rate_limit() -> None:
        if min_interval_s <= 0:
            return
        last_ts = rate_state.get("last_request_ts", 0.0)
        now = time.monotonic()
        wait_s = min_interval_s - (now - last_ts)
        if wait_s > 0:
            time.sleep(wait_s)
        rate_state["last_request_ts"] = time.monotonic()

    def compute_backoff(attempt: int) -> float:
        base = backoff_base_s * (2 ** attempt)
        sleep_s = min(base, backoff_max_s)
        jitter = random.uniform(0.0, min(1.0, sleep_s * 0.1))
        return sleep_s + jitter

    last_err: Optional[Exception] = None
    for attempt in range(max(1, max_retries + 1)):
        try:
            rate_limit()
            with urlrequest.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            out = json.loads(raw)
            # 解析 OpenAI 兼容格式的响应路径
            text = out["choices"][0]["message"]["content"]
            return text, out
        except urlerror.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="ignore")
            last_err = Exception(f"HTTP {exc.code}: {exc.reason} - Body: {err_body}")
            if attempt < max_retries:
                retry_after = exc.headers.get("Retry-After")
                if exc.code == 429 and retry_after:
                    try:
                        sleep_s = float(retry_after)
                    except ValueError:
                        sleep_s = compute_backoff(attempt)
                elif exc.code in (429, 500, 502, 503, 504):
                    sleep_s = compute_backoff(attempt)
                else:
                    sleep_s = compute_backoff(attempt)
                time.sleep(sleep_s)
                continue
            break
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(compute_backoff(attempt))
                continue
            break

    raise RuntimeError(f"Gemini API request failed: {last_err}")

def try_parse_json(text: str) -> Tuple[Optional[object], str]:
    if not text:
        return None, "empty response"
    try:
        return json.loads(text), ""
    except Exception:
        pass
    start_candidates =[i for i, ch in enumerate(text) if ch in "{["]
    if not start_candidates:
        return None, "no json found"
    for start in start_candidates:
        for end in range(len(text) - 1, start, -1):
            if text[end] in "}]":
                snippet = text[start : end + 1].strip()
                try:
                    return json.loads(snippet), ""
                except Exception:
                    continue
        break
    return None, "failed to parse json"


def main() -> int:
    print("[INFO] Loading environment variables...")
    load_dotenv() 
    args = parse_args()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] Missing API key: set GEMINI_API_KEY or GOOGLE_API_KEY in .env.", file=sys.stderr)
        return 2

    print(f"[INFO] Reading prompt from {args.prompt_file}...")
    prompt_text = ""
    if args.prompt_file.exists():
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()
    else:
        print(f"[ERROR] Prompt file '{args.prompt_file}' not found!", file=sys.stderr)
        return 2

    if not args.input_path.exists():
        print(f"[ERROR] Input path not found: {args.input_path}", file=sys.stderr)
        return 2

    header = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "sample_fps": args.sample_fps,
        "chunk_frames": args.chunk_frames,
        "model": args.model,
        "input_path": str(args.input_path),
    }
    writer = JsonStreamWriter(args.output_json, header)
    rate_state: Dict[str, float] = {"last_request_ts": 0.0}
    
    for video_path in iter_videos(args.input_path, args.recursive):
        print(f"\n[INFO] =========================================")
        print(f"[INFO] Processing Video: {video_path.name}")
        sampled_count = 0
        error_msg = ""
        video_started = False
        try:
            frame_iter, fps, duration_s = iter_sampled_frames(video_path, args.sample_fps)
            print(f"[INFO] Video stats: {fps} FPS, {duration_s:.1f} Seconds.")
            writer.start_video(
                video_path=str(video_path),
                fps=round(float(fps or 0.0), 3),
                duration_s=round(float(duration_s or 0.0), 3),
            )
            video_started = True
            
            chunk_images: List[str] = []
            chunk_times: List[float] =[]
            seg_idx = 0
            
            for t, frame in frame_iter:
                sampled_count += 1
                chunk_times.append(t)
                chunk_images.append(encode_jpeg_base64(frame))
                
                # 打印抽帧进度
                if sampled_count % 10 == 0:
                    sys.stdout.write(f"\r[INFO] Extracted {sampled_count} frames...")
                    sys.stdout.flush()
                
                if len(chunk_images) >= args.chunk_frames:
                    print(f"\n[INFO] --- Segment {seg_idx} ready (Frames {sampled_count-args.chunk_frames} to {sampled_count}). Sending to API...")
                    start_s = float(chunk_times[0])
                    end_s = float(chunk_times[-1] + (1.0 / args.sample_fps))
                    try:
                        text, _ = call_gemini(
                            api_key=api_key,
                            model=args.model,
                            prompt=prompt_text, # <--- 修复了 Bug: 之前写的是 args.prompt
                            images_b64=chunk_images,
                            timeout_s=args.timeout,
                            want_json=args.response_json,
                            max_retries=args.max_retries,
                            min_interval_s=args.min_interval,
                            backoff_base_s=args.backoff_base,
                            backoff_max_s=args.backoff_max,
                            rate_state=rate_state,
                        )
                        parsed, parse_err = try_parse_json(text)
                        writer.write_segment(
                            {
                                "segment_index": seg_idx,
                                "start_s": round(start_s, 3),
                                "end_s": round(end_s, 3),
                                "frame_count": len(chunk_images),
                                "response_text": text,
                                "response_json": parsed,
                                "error": parse_err,
                            }
                        )
                        print(f"    [SUCCESS] Segment {seg_idx} parsed. Action: {parsed.get('maneuver', 'N/A') if isinstance(parsed, dict) else 'Unknown'}")
                    except Exception as exc:
                        writer.write_segment(
                            {
                                "segment_index": seg_idx,
                                "start_s": round(start_s, 3),
                                "end_s": round(end_s, 3),
                                "frame_count": len(chunk_images),
                                "response_text": "",
                                "response_json": None,
                                "error": str(exc),
                            }
                        )
                        print(f"    [FAILED] Segment {seg_idx} error: {exc}")

                    seg_idx += 1
                    chunk_images = []
                    chunk_times =[]
                    if args.max_segments > 0 and seg_idx >= args.max_segments:
                        print(f"[INFO] Reached max-segments limit ({args.max_segments}). Stopping video.")
                        break

            # 处理最后遗留的帧 (如果不够一个 Chunk)
            if chunk_images and (args.max_segments == 0 or seg_idx < args.max_segments):
                print(f"\n[INFO] --- Sending final Segment {seg_idx} ({len(chunk_images)} frames) to API...")
                start_s = float(chunk_times[0])
                end_s = float(chunk_times[-1] + (1.0 / args.sample_fps))
                try:
                    text, _ = call_gemini(
                        api_key=api_key, model=args.model, prompt=prompt_text, images_b64=chunk_images,
                        timeout_s=args.timeout, want_json=args.response_json, max_retries=args.max_retries,
                        min_interval_s=args.min_interval,
                        backoff_base_s=args.backoff_base,
                        backoff_max_s=args.backoff_max,
                        rate_state=rate_state,
                    )
                    parsed, parse_err = try_parse_json(text)
                    writer.write_segment(
                        {
                            "segment_index": seg_idx,
                            "start_s": round(start_s, 3),
                            "end_s": round(end_s, 3),
                            "frame_count": len(chunk_images),
                            "response_text": text,
                            "response_json": parsed,
                            "error": parse_err,
                        }
                    )
                except Exception as exc:
                    writer.write_segment(
                        {
                            "segment_index": seg_idx,
                            "start_s": round(start_s, 3),
                            "end_s": round(end_s, 3),
                            "frame_count": len(chunk_images),
                            "response_text": "",
                            "response_json": None,
                            "error": str(exc),
                        }
                    )

        except Exception as exc:
            error_msg = str(exc)
            print(f"\n[ERROR] Fatal error processing video: {exc}")
            if not video_started:
                writer.start_video(
                    video_path=str(video_path),
                    fps=0.0,
                    duration_s=0.0,
                )
                video_started = True

        writer.end_video(sampled_frames=sampled_count, error=error_msg)

    print("\n[INFO] Finalizing JSON output...")
    writer.close()
    print(f"[INFO] Done! Output saved to {args.output_json}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
