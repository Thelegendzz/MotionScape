#!/usr/bin/env python3
"""
Evaluate CLIP irrelevant-frame detection against human-labeled ground truth.

The script compares predicted bad segments with ground-truth bad segments on a
uniform time grid and reports binary classification metrics:
Precision, Recall, F1, Accuracy.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted irrelevant-frame segments against ground truth."
    )
    parser.add_argument(
        "--pred-json",
        type=Path,
        required=True,
        help="Prediction JSON path, e.g. output/trash_segments_5fps.json.",
    )
    parser.add_argument(
        "--gt-json",
        type=Path,
        required=True,
        help="Ground-truth JSON path with bad_timestamps_s intervals.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the evaluation report as JSON.",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=None,
        help="Override evaluation sampling interval in seconds. Defaults to the finest interval found.",
    )
    parser.add_argument(
        "--positive-key",
        type=str,
        default="bad_timestamps_s",
        help="Interval field treated as the positive class. Default: bad_timestamps_s.",
    )
    parser.add_argument(
        "--negative-key",
        type=str,
        default="relevant_timestamps_s",
        help="Ground-truth interval field treated as the negative class. Default: relevant_timestamps_s.",
    )
    parser.add_argument(
        "--ignore-key",
        type=str,
        default="ambiguous_timestamps_s",
        help="Ground-truth interval field to ignore during evaluation. Default: ambiguous_timestamps_s.",
    )
    parser.add_argument(
        "--video-list",
        type=Path,
        default=None,
        help="Optional text file listing video names/basenames to evaluate, one per line.",
    )
    parser.add_argument(
        "--match-by",
        choices=["basename", "fullpath"],
        default="basename",
        help="How to match videos between prediction and GT JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-video metrics in addition to the aggregate result.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        content = f.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Allow simple JSONC-style line comments in manual annotation files.
        cleaned = re.sub(r"(?m)^\s*//.*$", "", content)
        return json.loads(cleaned)


def normalize_video_id(item: Dict[str, Any], match_by: str) -> str:
    raw = item.get("video_path") or item.get("video") or item.get("video_name") or item.get("name")
    if not raw:
        raise ValueError(f"Video entry missing path/name field: {item}")
    raw_str = str(raw)
    if match_by == "fullpath":
        return raw_str
    return Path(raw_str).name


def load_video_filter(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None
    if not path.exists():
        raise SystemExit(f"Video list not found: {path}")
    selected = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                selected.add(value)
                selected.add(Path(value).name)
    return selected


def collect_videos(data: Dict[str, Any], match_by: str) -> Dict[str, Dict[str, Any]]:
    videos = data.get("videos")
    if not isinstance(videos, list):
        raise SystemExit("Input JSON must contain a top-level 'videos' list.")
    mapped: Dict[str, Dict[str, Any]] = {}
    for item in videos:
        key = normalize_video_id(item, match_by)
        mapped[key] = item
    return mapped


def get_interval_s(data: Dict[str, Any], item: Dict[str, Any]) -> Optional[float]:
    for value in (
        item.get("interval_s"),
        data.get("interval_s"),
        item.get("sample_interval_s"),
    ):
        if value is not None:
            interval = float(value)
            if interval > 0:
                return interval
    fps = item.get("sample_fps") or item.get("fps")
    if fps:
        fps_value = float(fps)
        if fps_value > 0:
            return 1.0 / fps_value
    return None


def get_total_end_s(
    item: Dict[str, Any],
    interval_s: float,
    keys: Sequence[str],
) -> float:
    if "sample_count" in item and item["sample_count"] is not None:
        sample_count = int(item["sample_count"])
        return max(0.0, (sample_count - 1) * interval_s)
    if "frame_count" in item and "fps" in item and item["frame_count"] is not None and item["fps"]:
        frame_count = int(item["frame_count"])
        fps = float(item["fps"])
        if frame_count > 0 and fps > 0:
            return max(0.0, (frame_count - 1) / fps)

    max_end = 0.0
    for key in keys:
        intervals = item.get(key) or []
        for interval in intervals:
            if isinstance(interval, Sequence) and len(interval) >= 2:
                max_end = max(max_end, float(interval[1]))
    return max_end


def normalize_intervals(raw: Any) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    if not raw:
        return intervals
    for entry in raw:
        if not isinstance(entry, Sequence) or len(entry) < 2:
            continue
        start = float(entry[0])
        end = float(entry[1])
        if end < start:
            start, end = end, start
        intervals.append((start, end))
    intervals.sort()
    return intervals


def interval_contains(intervals: Sequence[Tuple[float, float]], t: float) -> bool:
    for start, end in intervals:
        if start - EPS <= t <= end + EPS:
            return True
        if t < start:
            return False
    return False


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + fp + fn + tn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def evaluate_video(
    pred_item: Dict[str, Any],
    gt_item: Dict[str, Any],
    pred_data: Dict[str, Any],
    gt_data: Dict[str, Any],
    eval_interval_s: float,
    positive_key: str,
    negative_key: str,
    ignore_key: str,
) -> Dict[str, Any]:
    pred_intervals = normalize_intervals(pred_item.get(positive_key) or [])
    gt_intervals = normalize_intervals(gt_item.get(positive_key) or [])
    gt_negative_intervals = normalize_intervals(gt_item.get(negative_key) or [])
    gt_ignore_intervals = normalize_intervals(gt_item.get(ignore_key) or [])

    pred_total_end = get_total_end_s(
        pred_item,
        get_interval_s(pred_data, pred_item) or eval_interval_s,
        keys=(positive_key, "valid_timestamps_s", "scenes"),
    )
    gt_total_end = get_total_end_s(
        gt_item,
        get_interval_s(gt_data, gt_item) or eval_interval_s,
        keys=(positive_key, negative_key, ignore_key, "valid_timestamps_s", "scenes"),
    )
    total_end_s = max(pred_total_end, gt_total_end)

    sample_count = int(round(total_end_s / eval_interval_s)) + 1 if total_end_s > 0 else 1
    tp = fp = fn = tn = 0
    skipped_ambiguous = 0
    skipped_unlabeled = 0

    for index in range(sample_count):
        t = round(index * eval_interval_s, 9)
        if interval_contains(gt_ignore_intervals, t):
            skipped_ambiguous += 1
            continue

        pred_positive = interval_contains(pred_intervals, t)
        gt_positive = interval_contains(gt_intervals, t)
        gt_negative = interval_contains(gt_negative_intervals, t)

        if not gt_positive and not gt_negative:
            skipped_unlabeled += 1
            continue

        if pred_positive and gt_positive:
            tp += 1
        elif pred_positive and not gt_positive:
            fp += 1
        elif not pred_positive and gt_positive:
            fn += 1
        else:
            tn += 1

    metrics = compute_metrics(tp, fp, fn, tn)
    return {
        "video": normalize_video_id(gt_item, "basename"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "samples": sample_count,
        "evaluated_samples": tp + fp + fn + tn,
        "skipped_ambiguous": skipped_ambiguous,
        "skipped_unlabeled": skipped_unlabeled,
        "duration_s": round(total_end_s, 6),
        **metrics,
    }


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def main() -> int:
    args = parse_args()
    pred_data = load_json(args.pred_json)
    gt_data = load_json(args.gt_json)

    pred_videos = collect_videos(pred_data, args.match_by)
    gt_videos = collect_videos(gt_data, args.match_by)
    selected_videos = load_video_filter(args.video_list)

    matched_keys = sorted(set(pred_videos) & set(gt_videos))
    if selected_videos is not None:
        matched_keys = [key for key in matched_keys if key in selected_videos or Path(key).name in selected_videos]

    if not matched_keys:
        raise SystemExit("No matched videos found between prediction JSON and ground-truth JSON.")

    inferred_intervals = [
        value
        for value in (
            args.interval_s,
            pred_data.get("interval_s"),
            gt_data.get("interval_s"),
        )
        if value is not None and float(value) > 0
    ]
    if inferred_intervals:
        eval_interval_s = min(float(value) for value in inferred_intervals)
    else:
        raise SystemExit("Unable to infer evaluation interval. Please provide --interval-s.")

    per_video: List[Dict[str, Any]] = []
    total_tp = total_fp = total_fn = total_tn = 0
    total_skipped_ambiguous = 0
    total_skipped_unlabeled = 0
    for key in matched_keys:
        result = evaluate_video(
            pred_item=pred_videos[key],
            gt_item=gt_videos[key],
            pred_data=pred_data,
            gt_data=gt_data,
            eval_interval_s=eval_interval_s,
            positive_key=args.positive_key,
            negative_key=args.negative_key,
            ignore_key=args.ignore_key,
        )
        per_video.append(result)
        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]
        total_tn += result["tn"]
        total_skipped_ambiguous += result["skipped_ambiguous"]
        total_skipped_unlabeled += result["skipped_unlabeled"]

    aggregate_metrics = compute_metrics(total_tp, total_fp, total_fn, total_tn)
    report = {
        "pred_json": str(args.pred_json),
        "gt_json": str(args.gt_json),
        "evaluated_videos": len(per_video),
        "eval_interval_s": eval_interval_s,
        "positive_key": args.positive_key,
        "negative_key": args.negative_key,
        "ignore_key": args.ignore_key,
        "confusion": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
        },
        "skipped": {
            "ambiguous": total_skipped_ambiguous,
            "unlabeled": total_skipped_unlabeled,
        },
        "metrics": aggregate_metrics,
        "per_video": per_video,
    }

    print(f"Evaluated videos: {len(per_video)}")
    print(f"Evaluation interval_s: {eval_interval_s}")
    print(f"Precision: {format_metric(aggregate_metrics['precision'])}")
    print(f"Recall:    {format_metric(aggregate_metrics['recall'])}")
    print(f"F1:        {format_metric(aggregate_metrics['f1'])}")
    print(f"Accuracy:  {format_metric(aggregate_metrics['accuracy'])}")
    print(
        "Confusion: "
        f"TP={total_tp}, FP={total_fp}, FN={total_fn}, TN={total_tn}"
    )
    print(
        "Skipped: "
        f"ambiguous={total_skipped_ambiguous}, unlabeled={total_skipped_unlabeled}"
    )

    if args.verbose:
        print("\nPer-video:")
        for item in per_video:
            print(
                f"{item['video']}: "
                f"P={format_metric(item['precision'])}, "
                f"R={format_metric(item['recall'])}, "
                f"F1={format_metric(item['f1'])}, "
                f"Acc={format_metric(item['accuracy'])}, "
                f"TP={item['tp']}, FP={item['fp']}, FN={item['fn']}, TN={item['tn']}, "
                f"skip_amb={item['skipped_ambiguous']}, skip_unlabeled={item['skipped_unlabeled']}"
            )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
