#!/usr/bin/env python3
"""
Check single-video resolution, or aggregate valid-scene duration by resolution
from a split_valid.json-like file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check video resolution or aggregate valid-scene durations by resolution."
    )
    parser.add_argument("video", nargs="?", type=Path, help="Path to a single video file.")
    parser.add_argument("--split-json", type=Path, help="Input split_valid.json path.")
    parser.add_argument("--video-root", type=Path, help="Root directory containing videos from split JSON.")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output JSON path for aggregated resolution statistics.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Output JSON indentation (default: 2).",
    )
    args = parser.parse_args()
    if args.split_json is not None:
        if args.video_root is None:
            parser.error("--video-root is required when using --split-json")
        if args.output_json is None:
            parser.error("--output-json is required when using --split-json")
    elif args.video is None:
        parser.error("either a positional video path or --split-json must be provided")
    return args


def get_video_codec(video_path: Path) -> str | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nw=1:nk=1",
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


def ffprobe_video_meta(video_path: Path) -> tuple[int, int, float]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("Missing ffprobe in PATH.")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    if not streams:
        return 0, 0, 0.0
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    duration_s = float(fmt.get("duration") or 0.0)
    return width, height, duration_s


def extract_frame_size_ffmpeg(video_path: Path, target_s: float, force_sw: bool) -> tuple[int, int] | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None

    width, height, duration_s = ffprobe_video_meta(video_path)
    if width <= 0 or height <= 0:
        return None

    seek_s = max(target_s, 0.0)
    if duration_s > 0.0:
        seek_s = min(seek_s, max(duration_s - 1e-3, 0.0))

    extra_args: list[str] = []
    if force_sw:
        extra_args = ["-hwaccel", "none"]

    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        f"{seek_s:.6f}",
        *extra_args,
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    frame_bytes = width * height * 3
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )
    except Exception:
        return None
    if len(proc.stdout) < frame_bytes:
        return None
    return width, height


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def scene_duration(scene: Any) -> float | None:
    if isinstance(scene, (list, tuple)) and len(scene) >= 2:
        try:
            start = float(scene[0])
            end = float(scene[1])
        except (TypeError, ValueError):
            return None
        return max(0.0, end - start)

    if isinstance(scene, dict):
        key_pairs = [
            ("start", "end"),
            ("start_s", "end_s"),
            ("start_time", "end_time"),
        ]
        for start_key, end_key in key_pairs:
            if start_key in scene and end_key in scene:
                try:
                    start = float(scene[start_key])
                    end = float(scene[end_key])
                except (TypeError, ValueError):
                    return None
                return max(0.0, end - start)
    return None


def detect_resolution(video_path: Path) -> tuple[int, int] | None:
    codec = get_video_codec(video_path)
    force_sw = codec == "av1"
    return extract_frame_size_ffmpeg(video_path, target_s=0.0, force_sw=force_sw)


def build_video_index(video_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    if not video_root.exists():
        return index
    for path in video_root.rglob("*"):
        if not path.is_file():
            continue
        index.setdefault(path.name, []).append(path)
    return index


def resolve_video_path(video_root: Path, video_name: str, video_index: dict[str, list[Path]]) -> Path | None:
    direct_path = video_root / video_name
    if direct_path.exists():
        return direct_path
    matches = video_index.get(Path(video_name).name, [])
    if len(matches) == 1:
        return matches[0]
    return None


def aggregate_split_json(split_json: Path, video_root: Path) -> dict[str, Any]:
    with split_json.open("r", encoding="utf-8") as f:
        payload: Any = json.load(f)

    videos = payload.get("videos")
    if not isinstance(payload, dict) or not isinstance(videos, list):
        raise SystemExit("Input JSON format is invalid: missing videos list.")

    resolution_stats: dict[str, dict[str, Any]] = {}
    video_stats: list[dict[str, Any]] = []
    total_valid_duration_s = 0.0
    processed_videos = 0
    missing_videos: list[str] = []
    failed_resolution_videos: list[str] = []
    ambiguous_videos: list[str] = []
    video_index = build_video_index(video_root)

    for item in videos:
        if not isinstance(item, dict):
            continue
        video_name = item.get("video")
        scenes = item.get("scenes")
        if not isinstance(video_name, str) or not video_name or not isinstance(scenes, list) or not scenes:
            continue

        valid_duration_s = 0.0
        for scene in scenes:
            duration = scene_duration(scene)
            if duration is not None:
                valid_duration_s += duration

        if valid_duration_s <= 0:
            continue

        video_path = resolve_video_path(video_root, video_name, video_index)
        if video_path is None:
            if len(video_index.get(Path(video_name).name, [])) > 1:
                ambiguous_videos.append(video_name)
            else:
                missing_videos.append(video_name)
            continue
        if not video_path.exists():
            missing_videos.append(video_name)
            continue

        resolution = detect_resolution(video_path)
        if resolution is None:
            failed_resolution_videos.append(video_name)
            continue

        processed_videos += 1
        total_valid_duration_s += valid_duration_s
        resolution_key = f"{resolution[0]}x{resolution[1]}"
        bucket = resolution_stats.setdefault(
            resolution_key,
            {
                "resolution": resolution_key,
                "video_count": 0,
                "valid_duration_s": 0.0,
                "valid_duration_hhmmss": "00:00:00",
            },
        )
        bucket["video_count"] += 1
        bucket["valid_duration_s"] += valid_duration_s
        bucket["valid_duration_hhmmss"] = format_duration(bucket["valid_duration_s"])

        video_stats.append(
            {
                "video": video_name,
                "video_path": str(video_path),
                "resolution": resolution_key,
                "valid_duration_s": round(valid_duration_s, 6),
                "valid_duration_hhmmss": format_duration(valid_duration_s),
                "valid_segment_count": len(scenes),
            }
        )

    resolution_items = []
    for resolution_key in sorted(
        resolution_stats,
        key=lambda key: (
            int(key.split("x", 1)[0]),
            int(key.split("x", 1)[1]),
        ),
    ):
        bucket = resolution_stats[resolution_key]
        bucket["valid_duration_s"] = round(float(bucket["valid_duration_s"]), 6)
        resolution_items.append(bucket)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "split_json": str(split_json),
        "video_root": str(video_root),
        "processed_video_count": processed_videos,
        "total_valid_duration_s": round(total_valid_duration_s, 6),
        "total_valid_duration_hhmmss": format_duration(total_valid_duration_s),
        "resolution_stats": resolution_items,
        "videos": video_stats,
        "missing_videos": missing_videos,
        "failed_resolution_videos": failed_resolution_videos,
        "ambiguous_videos": ambiguous_videos,
    }


def main() -> int:
    args = parse_args()

    if args.split_json is not None:
        result = aggregate_split_json(args.split_json, args.video_root)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=args.indent)
            f.write("\n")
        print(f"Processed videos: {result['processed_video_count']}")
        print(f"Total valid duration: {result['total_valid_duration_s']:.2f}s ({result['total_valid_duration_hhmmss']})")
        print(f"Resolution buckets: {len(result['resolution_stats'])}")
        print(f"Output: {args.output_json}")
        if result["missing_videos"]:
            print(f"Missing videos: {len(result['missing_videos'])}")
        if result["failed_resolution_videos"]:
            print(f"Resolution failures: {len(result['failed_resolution_videos'])}")
        if result["ambiguous_videos"]:
            print(f"Ambiguous matches: {len(result['ambiguous_videos'])}")
        return 0

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    res = detect_resolution(args.video)
    if res is None:
        raise SystemExit("Failed to read resolution via ffprobe/ffmpeg frame extraction.")
    print(f"{res[0]}x{res[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
