#!/usr/bin/env python3
"""
Select evaluation videos by valid_count buckets with deterministic sampling.

The script reads a JSON file like output/trash_segments_2fps.json and samples
videos from configured valid_count ranges. To keep annotation cost lower, the
sampling prefers shorter videos by restricting the random draw to the shortest
candidate pool in each bucket.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_BUCKETS = ("1-5", "5-10", "10-15")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select videos for manual evaluation by valid_count buckets."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Input JSON path, e.g. output/trash_segments_2fps.json.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path to save the sampling result.",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=None,
        help="Optional text file path to save selected video basenames, one per line.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260330,
        help="Random seed used for deterministic sampling.",
    )
    parser.add_argument(
        "--samples-per-bucket",
        type=int,
        default=5,
        help="How many videos to sample per valid_count bucket.",
    )
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=list(DEFAULT_BUCKETS),
        help="Bucket definitions in the form start-end, inclusive start and exclusive end.",
    )
    parser.add_argument(
        "--prefer-shortest-ratio",
        type=float,
        default=0.5,
        help=(
            "Sample only from the shortest X%% candidates in each bucket while keeping randomness. "
            "Must be in (0, 1]."
        ),
    )
    parser.add_argument(
        "--segment-sample-min",
        type=int,
        default=2,
        help="Minimum number of bad/valid segments to sample per selected video.",
    )
    parser.add_argument(
        "--segment-sample-max",
        type=int,
        default=3,
        help="Maximum number of bad/valid segments to sample per selected video.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_bucket(bucket: str) -> Tuple[int, int]:
    parts = bucket.split("-")
    if len(parts) != 2:
        raise SystemExit(f"Invalid bucket format: {bucket}. Expected start-end.")
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise SystemExit(f"Invalid bucket format: {bucket}. Expected integers.") from exc
    if start < 0 or end <= start:
        raise SystemExit(f"Invalid bucket range: {bucket}")
    return start, end


def get_video_name(item: Dict[str, Any]) -> str:
    raw = item.get("video_path") or item.get("video") or item.get("video_name") or item.get("name")
    if not raw:
        raise SystemExit(f"Video entry missing path/name field: {item}")
    return str(raw)


def get_duration_s(item: Dict[str, Any], interval_s: Optional[float]) -> float:
    if interval_s is not None and item.get("sample_count") is not None:
        sample_count = int(item["sample_count"])
        if sample_count > 0:
            return max(0.0, (sample_count - 1) * interval_s)

    max_end = 0.0
    for key in ("valid_timestamps_s", "bad_timestamps_s", "scenes"):
        intervals = item.get(key) or []
        for interval in intervals:
            if isinstance(interval, Sequence) and len(interval) >= 2:
                max_end = max(max_end, float(interval[1]))
    return max_end


def build_candidate(
    item: Dict[str, Any], interval_s: Optional[float]
) -> Dict[str, Any]:
    full_name = get_video_name(item)
    duration_s = get_duration_s(item, interval_s)
    valid_count = int(item.get("valid_count", 0))
    return {
        "video_path": full_name,
        "video_name": Path(full_name).name,
        "valid_count": valid_count,
        "duration_s": round(duration_s, 3),
        "sample_count": item.get("sample_count"),
        "bad_count": item.get("bad_count"),
        "bad_timestamps_s": item.get("bad_timestamps_s") or [],
        "valid_timestamps_s": item.get("valid_timestamps_s") or [],
    }


def normalize_segments(raw_segments: Any) -> List[List[float]]:
    normalized: List[List[float]] = []
    if not isinstance(raw_segments, list):
        return normalized
    for segment in raw_segments:
        if not isinstance(segment, Sequence) or len(segment) < 2:
            continue
        start = float(segment[0])
        end = float(segment[1])
        if end < start:
            start, end = end, start
        normalized.append([round(start, 3), round(end, 3)])
    return normalized


def sample_segments_for_label(
    segments: List[List[float]],
    min_count: int,
    max_count: int,
    rng: random.Random,
) -> List[List[float]]:
    if not segments:
        return []

    upper = min(max_count, len(segments))
    lower = min(min_count, upper)
    if lower <= 0:
        return []

    sample_size = rng.randint(lower, upper)
    selected = rng.sample(segments, k=sample_size)
    selected.sort(key=lambda item: (float(item[0]), float(item[1])))
    return selected


def attach_segment_samples(
    candidate: Dict[str, Any],
    min_count: int,
    max_count: int,
    rng: random.Random,
) -> Dict[str, Any]:
    bad_segments = normalize_segments(candidate.get("bad_timestamps_s") or [])
    valid_segments = normalize_segments(candidate.get("valid_timestamps_s") or [])

    enriched = dict(candidate)
    enriched["sampled_bad_segments"] = sample_segments_for_label(
        bad_segments,
        min_count=min_count,
        max_count=max_count,
        rng=rng,
    )
    enriched["sampled_valid_segments"] = sample_segments_for_label(
        valid_segments,
        min_count=min_count,
        max_count=max_count,
        rng=rng,
    )
    return enriched


def select_from_bucket(
    candidates: List[Dict[str, Any]],
    sample_size: int,
    rng: random.Random,
    prefer_shortest_ratio: float,
) -> Tuple[List[Dict[str, Any]], int]:
    if not candidates:
        return [], 0

    ordered = sorted(
        candidates,
        key=lambda item: (
            float(item["duration_s"]),
            int(item["valid_count"]),
            item["video_name"],
        ),
    )
    pool_size = min(
        len(ordered),
        max(sample_size, int(math.ceil(len(ordered) * prefer_shortest_ratio))),
    )
    pool = ordered[:pool_size]
    selected = rng.sample(pool, k=min(sample_size, len(pool)))
    selected.sort(
        key=lambda item: (
            float(item["duration_s"]),
            int(item["valid_count"]),
            item["video_name"],
        )
    )
    return selected, pool_size


def main() -> int:
    args = parse_args()
    if not (0.0 < args.prefer_shortest_ratio <= 1.0):
        raise SystemExit("--prefer-shortest-ratio must be in (0, 1].")
    if args.segment_sample_min <= 0:
        raise SystemExit("--segment-sample-min must be positive.")
    if args.segment_sample_max < args.segment_sample_min:
        raise SystemExit("--segment-sample-max must be >= --segment-sample-min.")

    data = load_json(args.input_json)
    videos = data.get("videos")
    if not isinstance(videos, list):
        raise SystemExit("Input JSON must contain a top-level 'videos' list.")

    interval_s = data.get("interval_s")
    interval_s = float(interval_s) if interval_s is not None else None
    video_rng = random.Random(args.seed)
    segment_rng = random.Random(args.seed + 1)

    candidates = [build_candidate(item, interval_s) for item in videos if item.get("valid_count") is not None]

    bucket_specs = [(bucket, *parse_bucket(bucket)) for bucket in args.buckets]
    report_buckets: List[Dict[str, Any]] = []
    selected_names: List[str] = []

    print(f"Seed: {args.seed}")
    print(f"Input JSON: {args.input_json}")
    print(f"Prefer shortest ratio: {args.prefer_shortest_ratio}")

    for bucket_label, start, end in bucket_specs:
        bucket_candidates = [
            item for item in candidates if start <= int(item["valid_count"]) < end
        ]
        selected, pool_size = select_from_bucket(
            bucket_candidates,
            sample_size=args.samples_per_bucket,
            rng=video_rng,
            prefer_shortest_ratio=args.prefer_shortest_ratio,
        )
        selected_with_segments = [
            attach_segment_samples(
                item,
                min_count=args.segment_sample_min,
                max_count=args.segment_sample_max,
                rng=segment_rng,
            )
            for item in selected
        ]

        report_buckets.append(
            {
                "bucket": bucket_label,
                "range": {"start_inclusive": start, "end_exclusive": end},
                "candidate_count": len(bucket_candidates),
                "shortlist_count": pool_size,
                "selected_count": len(selected_with_segments),
                "videos": selected_with_segments,
            }
        )

        print(
            f"\nBucket {bucket_label}: candidates={len(bucket_candidates)}, "
            f"shortlist={pool_size}, selected={len(selected)}"
        )
        for index, item in enumerate(selected_with_segments, start=1):
            selected_names.append(item["video_name"])
            print(
                f"{index}. {item['video_name']} | "
                f"valid_count={item['valid_count']} | duration_s={item['duration_s']}"
            )
            print(
                f"   sampled_bad={len(item['sampled_bad_segments'])}, "
                f"sampled_valid={len(item['sampled_valid_segments'])}"
            )

    report = {
        "input_json": str(args.input_json),
        "seed": args.seed,
        "samples_per_bucket": args.samples_per_bucket,
        "prefer_shortest_ratio": args.prefer_shortest_ratio,
        "segment_sample_min": args.segment_sample_min,
        "segment_sample_max": args.segment_sample_max,
        "buckets": report_buckets,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

    if args.output_txt is not None:
        args.output_txt.parent.mkdir(parents=True, exist_ok=True)
        with args.output_txt.open("w", encoding="utf-8") as f:
            for name in selected_names:
                f.write(name)
                f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
