#!/usr/bin/env python3
"""
恢复/重建视频处理统计 CSV：
- 视频是否还在（输出目录是否存在对应视频）
- 是否被删减（输出帧数 < 输入帧数）
- 被删减了多少（帧数、时长、比例）
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: opencv-python. Install with `pip install opencv-python`."
    ) from exc


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass
class VideoMeta:
    frames: int
    fps: float
    duration_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据输入/输出视频目录重建统计 CSV。")
    parser.add_argument("--input-dir", type=Path, required=True, help="原始视频目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="处理后视频目录")
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="输出 CSV 路径，例如 /data/out/recovered_report.csv",
    )
    parser.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    return parser.parse_args()


def iter_videos(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                yield p
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                yield p


def get_video_meta(path: Path) -> VideoMeta | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (frames / fps) if fps > 0 else 0.0
    cap.release()
    return VideoMeta(frames=frames, fps=fps, duration_s=duration)


def scan_video_map(root: Path, recursive: bool) -> Dict[Path, Path]:
    out: Dict[Path, Path] = {}
    for p in iter_videos(root, recursive):
        key = p.relative_to(root)
        out[key] = p
    return out


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    in_map = scan_video_map(input_dir, args.recursive)
    out_map = scan_video_map(output_dir, args.recursive)

    print(f"[INFO] input videos={len(in_map)} output videos={len(out_map)}")

    rows = []
    for rel, in_path in sorted(in_map.items(), key=lambda x: str(x[0])):
        # 输出目录中默认写 mp4，如果同名同后缀不存在就尝试 .mp4
        out_path = out_map.get(rel)
        if out_path is None:
            mp4_key = rel.with_suffix(".mp4")
            out_path = out_map.get(mp4_key)

        in_meta = get_video_meta(in_path)
        if in_meta is None:
            print(f"[WARN] skip unreadable input: {in_path}")
            continue

        video_exists_in_output = out_path is not None and out_path.exists()

        if video_exists_in_output:
            out_meta = get_video_meta(out_path)  # type: ignore[arg-type]
            if out_meta is None:
                print(f"[WARN] unreadable output: {out_path}, mark as missing")
                video_exists_in_output = False
                out_frames = 0
                out_duration = 0.0
            else:
                out_frames = out_meta.frames
                out_duration = out_meta.duration_s
        else:
            out_frames = 0
            out_duration = 0.0

        cut_frames = max(in_meta.frames - out_frames, 0)
        cut_duration_s = max(in_meta.duration_s - out_duration, 0.0)
        trimmed_ratio = (cut_frames / in_meta.frames) if in_meta.frames > 0 else 0.0
        was_trimmed = cut_frames > 0

        rows.append(
            {
                "video_rel_path": str(rel),
                "video_name": in_path.name,
                "video_exists_in_output": str(video_exists_in_output),
                "was_trimmed": str(was_trimmed),
                "trimmed_frames": str(cut_frames),
                "trimmed_duration_s": f"{cut_duration_s:.4f}",
                "trimmed_ratio": f"{trimmed_ratio:.6f}",
                "input_frames": str(in_meta.frames),
                "output_frames": str(out_frames),
                "input_duration_s": f"{in_meta.duration_s:.4f}",
                "output_duration_s": f"{out_duration:.4f}",
                "input_path": str(in_path),
                "output_path": str(out_path) if out_path else "",
            }
        )

    csv_path = args.csv_path
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_rel_path",
        "video_name",
        "video_exists_in_output",
        "was_trimmed",
        "trimmed_frames",
        "trimmed_duration_s",
        "trimmed_ratio",
        "input_frames",
        "output_frames",
        "input_duration_s",
        "output_duration_s",
        "input_path",
        "output_path",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] rows={len(rows)} csv={csv_path}")


if __name__ == "__main__":
    main()
