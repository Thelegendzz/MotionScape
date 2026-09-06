#!/usr/bin/env python3
"""Regroup existing MotionScape clip-level metrics using last-75 buckets."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


MODELS = (
    ("Text2World", "Cosmos2.5-2B [13]", "09_Cosmos-Predict2.5-2B-T2W_metrics.json", "09_Cosmos-Predict2.5-2B-T2W_fvd.json"),
    ("Text2World", "Wan2.2 T2V-A14B [14]", "05_Wan2.2-T2V-A14B_per_video_direct_1280x704.json", "05_Wan2.2-T2V-A14B_fvd.json"),
    ("Text2World", "CogVideoX1.5-5B [15]", "03_CogVideoX1.5-5B-T2V_per_video_direct_1280x704.json", "03_CogVideoX1.5-5B-T2V_fvd.json"),
    ("Image2World", "Cosmos2.5-2B", "07_Cosmos-Predict2.5-2B-I2W_metrics.json", "07_Cosmos-Predict2.5-2B-I2W_fvd.json"),
    ("Image2World", "Wan2.2 I2V-A14B", "04_Wan2.2-I2V-A14B_per_video_direct_1280x704.json", "04_Wan2.2-I2V-A14B_fvd.json"),
    ("Image2World", "CogVideoX1.5-5B-I2V", "02_CogVideoX1.5-5B-I2V_per_video_direct_1280x704.json", "02_CogVideoX1.5-5B-I2V_fvd.json"),
    ("Video2World", "Cosmos2.5-2B (Task-level)", "cosmos_v2w_task_metrics.json", "cosmos_v2w_task_fvd.json"),
    ("Video2World", "Cosmos2.5-2B (Clip-level)", "08_Cosmos-Predict2.5-2B-V2W_metrics.json", "08_Cosmos-Predict2.5-2B-V2W_fvd.json"),
    ("Video2World", "LongCat-Video [16]", "01_LongCat-Video-V2W_per_video_direct_1280x704.json", "01_LongCat-Video-V2W_fvd.json"),
    ("Video2World", "MAGI-1-24B [17]", "06_MAGI-1-24B-V2W_per_video_direct_1280x704.json", "06_MAGI-1-24B-V2W_fvd.json"),
)

METRICS = (
    ("psnr", "psnr_mean"),
    ("ssim", "ssim_mean"),
    ("lpips", "lpips_mean"),
    ("warping_error", "warping_error_mean"),
)
BUCKETS = ("low", "medium", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--buckets-json", type=Path, required=True)
    parser.add_argument("--fvd-root", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_json.exists():
        raise SystemExit(f"Refusing to overwrite: {args.output_json}")
    bucket_data = json.loads(args.buckets_json.read_text(encoding="utf-8"))
    bucket_by_id = {
        row["sample_id"]: row.get("dynamicity_bucket") or row.get("motion_stratum")
        for row in bucket_data["samples"]
    }
    if len(bucket_by_id) != 228:
        raise ValueError(f"Expected 228 bucket assignments, got {len(bucket_by_id)}")

    output_models: list[dict[str, Any]] = []
    for mode, model, metric_name, fvd_name in MODELS:
        metric_path = args.metrics_root / metric_name
        data = json.loads(metric_path.read_text(encoding="utf-8"))
        grouped: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in BUCKETS}
        seen: set[str] = set()
        for video in data["videos"]:
            sample_id = Path(video["pred_video"]["path"]).stem
            if sample_id not in bucket_by_id:
                raise KeyError(f"No last75 bucket for {sample_id}")
            if sample_id in seen:
                raise ValueError(f"Duplicate metric record: {sample_id}")
            seen.add(sample_id)
            grouped[bucket_by_id[sample_id]].append(video)
        if seen != set(bucket_by_id):
            missing = sorted(set(bucket_by_id) - seen)
            raise ValueError(f"{model}: missing {len(missing)} metric records")

        fvd_by_bucket: dict[str, float] = {}
        fvd_path = args.fvd_root / fvd_name if args.fvd_root else None
        if fvd_path is not None and fvd_path.is_file():
            fvd_data = json.loads(fvd_path.read_text(encoding="utf-8"))
            fvd_by_bucket = {
                bucket: float(fvd_data["bucket_results"][bucket]["fvd"])
                for bucket in BUCKETS
            }

        rows = []
        for bucket in BUCKETS:
            videos = grouped[bucket]
            row: dict[str, Any] = {"bucket": bucket, "count": len(videos)}
            for output_key, source_key in METRICS:
                values = [float(video["metrics"][source_key]) for video in videos]
                finite = [value for value in values if math.isfinite(value)]
                if len(finite) != len(videos):
                    raise ValueError(f"{model}/{bucket}/{source_key}: non-finite values")
                row[output_key] = statistics.fmean(finite)
            row["fvd"] = fvd_by_bucket.get(bucket)
            row["clipsim"] = None
            rows.append(row)
        output_models.append(
            {
                "mode": mode,
                "model": model,
                "source_metrics_json": str(metric_path),
                "source_fvd_json": str(fvd_path) if fvd_path else None,
                "rows": rows,
            }
        )

    result = {
        "dynamicity_buckets_json": str(args.buckets_json),
        "temporal_scope": bucket_data.get("temporal_scope"),
        "aggregation": "equal-weight mean of existing clip-level metrics within each new bucket",
        "clipsim_status": "excluded; recompute with evaluate_clipsim.py",
        "models": output_models,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote: {args.output_json}")


if __name__ == "__main__":
    main()
