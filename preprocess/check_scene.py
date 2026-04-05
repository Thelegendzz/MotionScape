import argparse
import json
from collections import defaultdict
from pathlib import Path


def format_hhmmss(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def normalize_weather(weather: object) -> str:
    if weather is None:
        return "Unknown"
    text = str(weather).strip()
    return text or "Unknown"


def build_weather_stats(data: dict) -> dict:
    weather_stats = defaultdict(lambda: {"total_duration_s": 0.0, "segment_count": 0, "videos": set()})

    for video in data.get("videos", []):
        video_path = video.get("video_path", "")
        for segment in video.get("segments", []):
            start_s = float(segment.get("start_s", 0.0) or 0.0)
            end_s = float(segment.get("end_s", start_s) or start_s)
            duration_s = max(0.0, end_s - start_s)

            response_json = segment.get("response_json") or {}
            weather = normalize_weather(response_json.get("weather"))

            weather_stats[weather]["total_duration_s"] += duration_s
            weather_stats[weather]["segment_count"] += 1
            if video_path:
                weather_stats[weather]["videos"].add(video_path)

    summary = {}
    for weather, stats in sorted(weather_stats.items()):
        total_duration_s = round(stats["total_duration_s"], 3)
        summary[weather] = {
            "total_duration_s": total_duration_s,
            "total_duration_hhmmss": format_hhmmss(total_duration_s),
            "segment_count": stats["segment_count"],
            "video_count": len(stats["videos"]),
        }

    return {
        "input_file": data.get("input_path"),
        "source_created_at": data.get("created_at"),
        "weather_categories": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="统计每个天气类别对应片段的总时长。")
    parser.add_argument(
        "--input",
        default="output/4o_mini_scene.json",
        help="输入的场景识别 JSON 文件路径",
    )
    parser.add_argument(
        "--output",
        default="output/weather_duration_stats.json",
        help="输出统计 JSON 文件路径",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    result = build_weather_stats(data)
    result["input_json_path"] = str(input_path.resolve())
    result["output_json_path"] = str(output_path.resolve())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Weather duration stats written to: {output_path}")


if __name__ == "__main__":
    main()
