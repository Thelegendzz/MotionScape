#!/usr/bin/env python3
"""
Filter short split scenes from a split.json-like file.

Example:
    python3 preprocess/split_fileter.py \
        --input-json /home/guozile/TransNetV2/output/split.json \
        --output-json /home/guozile/UAV_dataset/output/split_valid.json \
        --min-duration 2.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="删除 split.json 中时长小于阈值的片段，并输出同结构 JSON。"
    )
    parser.add_argument("--input-json", type=Path, required=True, help="输入 split.json 路径")
    parser.add_argument("--output-json", type=Path, required=True, help="输出有效片段 JSON 路径")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="最小时长（秒），小于该值的片段会被删除",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="输出 JSON 缩进（默认 2）",
    )
    return parser.parse_args()


def _scene_duration(scene: Any) -> float | None:
    if isinstance(scene, (list, tuple)) and len(scene) >= 2:
        try:
            start = float(scene[0])
            end = float(scene[1])
            return end - start
        except (TypeError, ValueError):
            return None

    if isinstance(scene, dict):
        key_pairs = [
            ("start", "end"),
            ("start_s", "end_s"),
            ("start_time", "end_time"),
        ]
        for start_key, end_key in key_pairs:
            if start_key in scene and end_key in scene:
                try:
                    return float(scene[end_key]) - float(scene[start_key])
                except (TypeError, ValueError):
                    return None
    return None


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def filter_video_scenes(video: dict[str, Any], min_duration: float) -> tuple[int, int, float, float]:
    scenes = video.get("scenes")
    if not isinstance(scenes, list):
        return 0, 0, 0.0, 0.0

    valid_indices: list[int] = []
    original_duration = 0.0
    kept_duration = 0.0
    for idx, scene in enumerate(scenes):
        duration = _scene_duration(scene)
        if duration is not None:
            original_duration += max(duration, 0.0)

        if duration is None:
            # 无法解析时保守保留，避免误删
            valid_indices.append(idx)
            continue
        if duration >= min_duration:
            valid_indices.append(idx)
            kept_duration += duration

    original_count = len(scenes)
    kept_count = len(valid_indices)
    removed_count = original_count - kept_count

    video["scenes"] = [scenes[i] for i in valid_indices]

    scene_frames = video.get("scene_frames")
    if isinstance(scene_frames, list) and len(scene_frames) == original_count:
        video["scene_frames"] = [scene_frames[i] for i in valid_indices]

    return original_count, removed_count, original_duration, kept_duration


def main() -> None:
    args = parse_args()

    with args.input_json.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if not isinstance(data, dict) or "videos" not in data or not isinstance(data["videos"], list):
        raise SystemExit("输入 JSON 格式不符合预期：缺少 videos 列表。")

    total_scenes = 0
    total_removed = 0
    total_original_duration = 0.0
    total_kept_duration = 0.0
    for video in data["videos"]:
        if not isinstance(video, dict):
            continue
        original_count, removed_count, original_duration, kept_duration = filter_video_scenes(
            video, args.min_duration
        )
        total_scenes += original_count
        total_removed += removed_count
        total_original_duration += original_duration
        total_kept_duration += kept_duration

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=args.indent)
        f.write("\n")

    kept = total_scenes - total_removed
    removed_duration = max(total_original_duration - total_kept_duration, 0.0)
    print(f"Done. min_duration={args.min_duration}s")
    print(f"Scenes: total={total_scenes}, kept={kept}, removed={total_removed}")
    print(
        f"Duration(s): total={total_original_duration:.2f}, "
        f"kept={total_kept_duration:.2f}, removed={removed_duration:.2f}"
    )
    print(
        f"Duration(hh:mm:ss): total={format_duration(total_original_duration)}, "
        f"kept={format_duration(total_kept_duration)}, removed={format_duration(removed_duration)}"
    )
    print(f"Output: {args.output_json}")


if __name__ == "__main__":
    main()
