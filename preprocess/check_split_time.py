import argparse
import json
from pathlib import Path


DEFAULT_BINS = [3, 6, 15, 60, 240, 600]


def format_range_label(start: float, end: float | None) -> str:
    if end is None:
        return f"{start:.0f}s+"
    return f"{start:.0f}-{end:.0f}s"


def get_bucket(duration_s: float, bins: list[float]) -> str:
    for i in range(len(bins) - 1):
        start = bins[i]
        end = bins[i + 1]
        if start <= duration_s < end:
            return format_range_label(start, end)
    return format_range_label(bins[-1], None)


def build_stats(data: dict, bins: list[float]) -> dict:
    bucket_counts = {}
    for i in range(len(bins) - 1):
        bucket_counts[format_range_label(bins[i], bins[i + 1])] = 0
    bucket_counts[format_range_label(bins[-1], None)] = 0

    total_segments = 0
    total_duration_s = 0.0

    for video in data.get("videos", []):
        for scene in video.get("scenes", []):
            if not isinstance(scene, list) or len(scene) != 2:
                continue
            start_s, end_s = scene
            duration_s = max(0.0, float(end_s) - float(start_s))
            bucket = get_bucket(duration_s, bins)
            bucket_counts[bucket] += 1
            total_segments += 1
            total_duration_s += duration_s

    return {
        "source_generated_at": data.get("generated_at"),
        "input_interval_s": data.get("interval_s"),
        "total_valid_segments": total_segments,
        "total_valid_duration_s": round(total_duration_s, 3),
        "duration_ranges": [
            {"range": label, "segment_count": count}
            for label, count in bucket_counts.items()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="统计有效片段时长在不同区间内的数量。")
    parser.add_argument(
        "--input",
        default="output/split_valid.json",
        help="输入的有效片段 JSON 文件路径",
    )
    parser.add_argument(
        "--output",
        default="output/split_valid_duration_statics.json",
        help="输出统计 JSON 文件路径",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    result = build_stats(data, DEFAULT_BINS)
    result["input_json_path"] = str(input_path.resolve())
    result["output_json_path"] = str(output_path.resolve())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Split duration stats written to: {output_path}")


if __name__ == "__main__":
    main()
