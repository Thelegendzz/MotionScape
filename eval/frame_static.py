#!/usr/bin/env python3
"""
Count how many original-video frames fall inside timestamp segments from JSON files
like split_valid.json or trash_segments_2fps.json.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计时间片段在原始视频中对应的总帧数。"
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path("output/split_valid.json"),
        help="输入 split_valid.json 路径。",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        required=True,
        help="原始视频根目录。",
    )
    parser.add_argument(
        "--segment-key",
        type=str,
        default="valid_timestamps_s",
        help=(
            "待统计的时间区间字段名。"
            "对 split_valid.json 可用 scenes，"
            "对 trash_segments_2fps.json 可用 valid_timestamps_s 或 bad_timestamps_s。"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/frame_statics.json"),
        help="输出统计 JSON 路径。",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="输出 JSON 缩进，默认 2。",
    )
    return parser.parse_args()


def require_ffprobe() -> str:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise SystemExit("Missing ffprobe in PATH.")
    return ffprobe


def parse_rate(value: Any) -> float:
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


def ffprobe_video_meta(video_path: Path) -> tuple[float, int, float]:
    ffprobe = require_ffprobe()
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found: {video_path}")

    stream = streams[0]
    fps = parse_rate(stream.get("avg_frame_rate"))
    frame_count_raw = stream.get("nb_frames")
    duration_raw = stream.get("duration")

    frame_count = int(frame_count_raw) if frame_count_raw not in (None, "N/A") else 0
    duration_s = float(duration_raw) if duration_raw not in (None, "N/A") else 0.0

    if duration_s <= 0.0 and frame_count > 0 and fps > 0:
        duration_s = frame_count / fps
    if frame_count <= 0 and duration_s > 0.0 and fps > 0:
        frame_count = int(round(duration_s * fps))

    return fps, frame_count, duration_s


def build_video_index(video_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    if not video_root.exists():
        return index
    for path in video_root.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, []).append(path)
    return index


def get_video_name(item: dict[str, Any]) -> str:
    for key in ("video_path", "video", "video_name", "name"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def resolve_video_path(video_root: Path, video_name: str, video_index: dict[str, list[Path]]) -> Path | None:
    direct = video_root / video_name
    if direct.exists():
        return direct
    matches = video_index.get(Path(video_name).name, [])
    if len(matches) == 1:
        return matches[0]
    return None


def normalize_scene(scene: Any) -> tuple[float, float] | None:
    if not isinstance(scene, (list, tuple)) or len(scene) < 2:
        return None
    try:
        start_s = float(scene[0])
        end_s = float(scene[1])
    except (TypeError, ValueError):
        return None
    if end_s < start_s:
        start_s, end_s = end_s, start_s
    return start_s, end_s


def segment_to_frame_range(
    start_s: float,
    end_s: float,
    fps: float,
    total_frames: int,
) -> tuple[int, int, int]:
    if fps <= 0 or total_frames <= 0:
        return 0, -1, 0

    start_frame = int(math.floor(start_s * fps + EPS))
    end_exclusive = int(math.floor(end_s * fps + EPS))

    start_frame = max(0, min(start_frame, total_frames))
    end_exclusive = max(start_frame, min(end_exclusive, total_frames))
    frame_count = max(0, end_exclusive - start_frame)

    if frame_count == 0:
        return start_frame, start_frame - 1, 0
    return start_frame, end_exclusive - 1, frame_count


def build_stats(
    split_payload: dict[str, Any],
    video_root: Path,
    segment_key: str,
) -> dict[str, Any]:
    videos = split_payload.get("videos")
    if not isinstance(videos, list):
        raise SystemExit("Input JSON format invalid: missing top-level 'videos' list.")

    video_index = build_video_index(video_root)

    per_video: list[dict[str, Any]] = []
    missing_videos: list[str] = []
    ambiguous_videos: list[str] = []
    total_segment_frames = 0
    total_original_frames = 0
    total_segment_duration_s = 0.0

    for item in videos:
        if not isinstance(item, dict):
            continue
        video_name = get_video_name(item)
        if not video_name:
            continue

        basename = Path(video_name).name
        matches = video_index.get(basename, [])
        if len(matches) > 1 and not (video_root / video_name).exists():
            ambiguous_videos.append(video_name)
            continue

        video_path = resolve_video_path(video_root, video_name, video_index)
        if video_path is None:
            missing_videos.append(video_name)
            continue

        fps, original_frame_count, duration_s = ffprobe_video_meta(video_path)
        scenes = item.get(segment_key) or []

        collected_segments: list[dict[str, Any]] = []
        segment_frame_count = 0
        segment_duration_s = 0.0
        for segment_index, raw_scene in enumerate(scenes):
            normalized = normalize_scene(raw_scene)
            if normalized is None:
                continue
            start_s, end_s = normalized
            duration = max(0.0, end_s - start_s)
            start_frame, end_frame, frame_count = segment_to_frame_range(
                start_s=start_s,
                end_s=end_s,
                fps=fps,
                total_frames=original_frame_count,
            )
            collected_segments.append(
                {
                    "segment_index": segment_index,
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                    "duration_s": round(duration, 3),
                    "original_start_frame": start_frame,
                    "original_end_frame": end_frame,
                    "original_frame_count": frame_count,
                }
            )
            segment_duration_s += duration
            segment_frame_count += frame_count

        total_segment_frames += segment_frame_count
        total_original_frames += original_frame_count
        total_segment_duration_s += segment_duration_s

        per_video.append(
            {
                "video": video_name,
                "resolved_video_path": str(video_path),
                "original_fps": fps,
                "original_frame_count": original_frame_count,
                "original_duration_s": round(duration_s, 3),
                "segment_key": segment_key,
                "segment_count": len(collected_segments),
                "segment_duration_s": round(segment_duration_s, 3),
                "segment_original_frame_count": segment_frame_count,
                "segment_frame_ratio": (
                    round(segment_frame_count / original_frame_count, 6)
                    if original_frame_count > 0
                    else 0.0
                ),
                "segments": collected_segments,
            }
        )

    per_video.sort(key=lambda item: item["video"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_split_json": str(Path(split_payload.get("_input_path", "")).resolve()) if split_payload.get("_input_path") else None,
        "video_root": str(video_root.resolve()),
        "segment_key": segment_key,
        "video_count": len(per_video),
        "missing_video_count": len(missing_videos),
        "ambiguous_video_count": len(ambiguous_videos),
        "total_original_frames": total_original_frames,
        "total_segment_original_frames": total_segment_frames,
        "total_segment_duration_s": round(total_segment_duration_s, 3),
        "segment_frame_ratio_overall": (
            round(total_segment_frames / total_original_frames, 6)
            if total_original_frames > 0
            else 0.0
        ),
        "missing_videos": missing_videos,
        "ambiguous_videos": ambiguous_videos,
        "videos": per_video,
    }


def main() -> None:
    args = parse_args()
    with args.split_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit("Input JSON must be a dict containing a top-level 'videos' list.")
    payload["_input_path"] = str(args.split_json)

    stats = build_stats(payload, args.video_root, args.segment_key)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=args.indent)

    print(f"Output JSON written to: {args.output_json}")
    print(f"Processed videos: {stats['video_count']}")
    print(f"Segment key: {stats['segment_key']}")
    print(f"Total original frames in segments: {stats['total_segment_original_frames']}")
    print(f"Overall segment-frame ratio: {stats['segment_frame_ratio_overall']:.6f}")
    if stats["missing_video_count"] > 0:
        print(f"Missing videos: {stats['missing_video_count']}")
    if stats["ambiguous_video_count"] > 0:
        print(f"Ambiguous videos: {stats['ambiguous_video_count']}")


if __name__ == "__main__":
    main()
