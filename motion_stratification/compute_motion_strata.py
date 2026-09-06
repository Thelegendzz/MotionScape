#!/usr/bin/env python3
"""Recompute MotionScape dynamicity buckets from only the final 75 frames.

This intentionally preserves the original bucketing protocol (29.97 -> 9.99
FPS sampling, 854x480 Farneback flow, P75 flow magnitude per adjacent sampled
frame pair, mean per clip, and global 33/66 percentile thresholds) while
changing only the temporal scope from all 275 frames to the final 75 frames.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from motion_stratification import motion_scoring as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-root",
        type=Path,
        required=True,
        help="Directory containing one 275-frame image directory per sample.",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Optional source manifest used to enrich detailed output records.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Destination JSON. Existing files are never overwritten.",
    )
    parser.add_argument("--tail-frames", type=int, default=75)
    parser.add_argument("--expected-total-frames", type=int, default=275)
    parser.add_argument("--source-fps", type=float, default=base.DEFAULT_SOURCE_FPS)
    parser.add_argument("--target-fps", type=float, default=base.DEFAULT_TARGET_FPS)
    parser.add_argument("--width", type=int, default=base.DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=base.DEFAULT_HEIGHT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()
    if args.tail_frames < 2:
        parser.error("--tail-frames must be at least 2")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def compute_tail_dynamicity(
    sample_dir: Path,
    tail_frames: int,
    expected_total_frames: int,
    source_fps: float,
    target_fps: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    all_frames = base.list_frames(sample_dir)
    warnings: list[str] = []
    if expected_total_frames > 0 and len(all_frames) != expected_total_frames:
        warnings.append(
            f"expected {expected_total_frames} total frames, found {len(all_frames)}"
        )
    if len(all_frames) < tail_frames:
        return {
            "sample_id": sample_dir.name,
            "frames_dir": str(sample_dir),
            "status": "skipped",
            "reason": f"only {len(all_frames)} frames; need {tail_frames}",
            "source_frame_count": len(all_frames),
            "tail_frame_count": 0,
            "warnings": warnings,
        }

    tail_start_index = len(all_frames) - tail_frames
    target_frames = all_frames[tail_start_index:]
    relative_indices = base.build_sample_indices(
        len(target_frames), source_fps, target_fps
    )
    sampled_paths = [target_frames[index] for index in relative_indices]
    if len(sampled_paths) < 2:
        return {
            "sample_id": sample_dir.name,
            "frames_dir": str(sample_dir),
            "status": "skipped",
            "reason": "not enough sampled target frames for optical flow",
            "source_frame_count": len(all_frames),
            "tail_frame_count": len(target_frames),
            "sampled_frame_count": len(sampled_paths),
            "warnings": warnings,
        }

    previous = base.read_gray_resized(sampled_paths[0], width, height)
    pair_scores: list[float] = []
    for path in sampled_paths[1:]:
        following = base.read_gray_resized(path, width, height)
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            following,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
        pair_scores.append(float(np.percentile(magnitude, 75)))
        previous = following

    absolute_indices = [tail_start_index + index for index in relative_indices]
    return {
        "sample_id": sample_dir.name,
        "frames_dir": str(sample_dir),
        "status": "ok",
        "source_frame_count": len(all_frames),
        "temporal_scope": "final_75_frames_only",
        "tail_frame_count": len(target_frames),
        "tail_start_zero_based_index": tail_start_index,
        "tail_end_zero_based_index": len(all_frames) - 1,
        "sampled_frame_count": len(sampled_paths),
        "sampled_zero_based_indices": absolute_indices,
        "frame_pair_count": len(pair_scores),
        "frame_pair_score_p75_mean": float(np.mean(pair_scores)),
        "frame_pair_scores_p75": pair_scores,
        "warnings": warnings,
    }


def main() -> None:
    args = parse_args()
    base.require_runtime_dependencies()
    sample_dirs = base.list_sample_dirs(args.frames_root)
    if not sample_dirs:
        raise SystemExit(f"No sample directories under {args.frames_root}")
    if args.output_json.exists():
        raise SystemExit(f"Refusing to overwrite existing result: {args.output_json}")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = [
            executor.submit(
                compute_tail_dynamicity,
                sample_dir,
                args.tail_frames,
                args.expected_total_frames,
                args.source_fps,
                args.target_fps,
                args.width,
                args.height,
            )
            for sample_dir in sample_dirs
        ]
        items = [future.result() for future in concurrent.futures.as_completed(futures)]
    items.sort(key=lambda item: item["sample_id"])

    manifest_index = base.load_manifest_index(args.manifest_json)
    base.enrich_with_manifest(items, manifest_index)
    q33, q66 = base.assign_buckets(items)

    bucket_counts = {
        bucket: sum(item.get("dynamicity_bucket") == bucket for item in items)
        for bucket in ("low", "medium", "high")
    }
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frames_root": str(args.frames_root),
        "manifest_json": str(args.manifest_json),
        "temporal_scope": "final_75_frames_only",
        "tail_frames": args.tail_frames,
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
        "resize": {"width": args.width, "height": args.height},
        "frame_pair_score": "dense optical flow magnitude P75",
        "sample_score": "mean of frame-pair P75 scores",
        "bucket_quantiles": {"q33": q33, "q66": q66},
        "summary": {
            "sample_count": len(items),
            "ok_count": sum(item.get("status") == "ok" for item in items),
            "skipped_count": sum(item.get("status") != "ok" for item in items),
            "bucket_counts": bucket_counts,
        },
        "samples": items,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=args.indent),
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote: {args.output_json}")


if __name__ == "__main__":
    main()
