#!/usr/bin/env python3
"""
Filter trash segments:
- Keep multi-frame consecutive detections (length >= 2).
- Keep single-frame detections only if they are near the video start or end
  (within the first/last N frames).
- Drop single-frame detections in the middle.
- Merge runs with a 1-frame gap into one interval.
- If all frames except the very first or last are marked, include the missing end.
Output JSON keeps the same structure, but bad_timestamps_s becomes a list of
intervals [start_s, end_s], and counts are re-numbered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter trash segments JSON.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("output/trash_segments.json"),
        help="Input JSON path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/trash_segments_filtered.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=None,
        help="Override interval_s when not present in JSON.",
    )
    parser.add_argument(
        "--edge-frames",
        type=int,
        default=5,
        help="Keep single-frame detections within the first/last N frames.",
    )
    parser.add_argument(
        "--merge-gap-frames",
        type=int,
        default=1,
        help="Merge intervals if the gap between runs is <= N missing frames.",
    )
    return parser.parse_args()


def _to_frame_indices(timestamps: List[float], interval_s: float) -> List[int]:
    if not timestamps:
        return []
    if interval_s <= 0:
        return list(range(len(timestamps)))
    indices = [int(round(ts / interval_s)) for ts in timestamps]
    return sorted(set(indices))


def _consecutive_runs(indices: List[int]) -> List[List[int]]:
    if not indices:
        return []
    runs: List[List[int]] = []
    current = [indices[0]]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            current.append(idx)
        else:
            runs.append(current)
            current = [idx]
        prev = idx
    runs.append(current)
    return runs


def _maybe_fill_ends(indices: List[int], sample_count: Optional[int]) -> List[int]:
    if not indices or sample_count is None or sample_count <= 0:
        return indices
    last_idx = sample_count - 1
    if last_idx <= 0:
        return indices

    idx_set = set(indices)
    # If all frames except the first are marked, include the first.
    if len(idx_set) == last_idx and 0 not in idx_set:
        if min(idx_set) == 1 and max(idx_set) == last_idx:
            idx_set.add(0)
    # If all frames except the last are marked, include the last.
    if len(idx_set) == last_idx and last_idx not in idx_set:
        if min(idx_set) == 0 and max(idx_set) == last_idx - 1:
            idx_set.add(last_idx)
    return sorted(idx_set)


def _filter_timestamps(
    timestamps: List[float],
    interval_s: float,
    sample_count: Optional[int],
    edge_frames: int,
    merge_gap_frames: int,
) -> List[List[float]]:
    if not timestamps:
        return []

    indices = _to_frame_indices(timestamps, interval_s)
    runs = _consecutive_runs(indices)

    last_idx = (sample_count - 1) if sample_count is not None and sample_count > 0 else None
    keep_indices: List[int] = []

    for run in runs:
        if len(run) >= 2:
            keep_indices.extend(run)
            continue
        idx = run[0]
        is_start = idx <= max(edge_frames - 1, 0)
        is_end = last_idx is not None and idx >= max(last_idx - (edge_frames - 1), 0)
        if is_start or is_end:
            keep_indices.append(idx)

    keep_indices = _maybe_fill_ends(sorted(set(keep_indices)), sample_count)
    if not keep_indices:
        return []

    # Merge intervals with a small gap (<= merge_gap_frames missing frames).
    intervals_idx: List[Tuple[int, int]] = []
    start = keep_indices[0]
    end = keep_indices[0]
    for idx in keep_indices[1:]:
        if idx <= end + (merge_gap_frames + 1):
            end = idx
        else:
            intervals_idx.append((start, end))
            start = idx
            end = idx
    intervals_idx.append((start, end))

    intervals_s: List[List[float]] = []
    for s_idx, e_idx in intervals_idx:
        start_s = round(s_idx * interval_s, 3) if interval_s > 0 else float(s_idx)
        end_s = round(e_idx * interval_s, 3) if interval_s > 0 else float(e_idx)
        intervals_s.append([start_s, end_s])

    return intervals_s


def _filter_video(
    video: Dict[str, Any], interval_s: float, edge_frames: int, merge_gap_frames: int
) -> Dict[str, Any]:
    timestamps = list(video.get("bad_timestamps_s") or [])
    sample_count = video.get("sample_count")
    filtered = _filter_timestamps(
        timestamps, interval_s, sample_count, edge_frames, merge_gap_frames
    )

    updated = dict(video)
    updated["bad_timestamps_s"] = filtered
    if "bad_count" in updated:
        updated["bad_count"] = len(filtered)
    if "ad_count" in updated:
        updated["ad_count"] = len(filtered)
    return updated


def main() -> int:
    args = parse_args()
    if not args.input_json.exists():
        raise SystemExit(f"Input JSON not found: {args.input_json}")

    with args.input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    interval_s = args.interval_s if args.interval_s is not None else data.get("interval_s", 1.0)
    videos = data.get("videos") or []

    filtered_videos = [
        _filter_video(video, float(interval_s), args.edge_frames, args.merge_gap_frames)
        for video in videos
    ]
    output = dict(data)
    output["videos"] = filtered_videos
    if "interval_s" not in output and interval_s is not None:
        output["interval_s"] = interval_s

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
