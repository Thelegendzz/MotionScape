#!/usr/bin/env python3
"""
Reconstruct MotionScape samples as fixed 29.97 FPS clips and frames.

The public flat source_manifest.json schema is detected from its top-level
items list. Each item is resolved against --video-root by exact source filename
and then by YouTube video ID when available. The legacy reviewed-segment schema
remains supported for provenance.

Each accepted sample is resampled to 30000/1001 FPS and truncated to the first
continuous 275 frames. Existing full outputs are skipped unless --overwrite is
explicitly provided.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_FPS_EXPR = "30000/1001"
TARGET_FPS_FLOAT = 30000 / 1001
TOTAL_FRAMES = 275
FIRST_FRAMES = 200
LAST_FRAMES = 75
DEFAULT_VIDEO_EXTENSIONS = [
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".flv",
    ".wmv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct public or legacy MotionScape manifests as 29.97 FPS "
            "275-frame samples plus 200-frame observed and 75-frame future videos."
        )
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Public source_manifest.json or a legacy reviewed-segment JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination root for reconstructed videos, frames, and recovery manifest.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="可选：原始视频根目录。提供后会按 video 字段在该目录下递归匹配视频路径。",
    )
    parser.add_argument(
        "--after-prefix-letter",
        default="C",
        help="只保留 video 首个字母晚于该字母的视频，默认 C。",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_VIDEO_EXTENSIONS,
        help="使用 --video-root 递归匹配视频时允许的扩展名。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的输出文件/帧目录。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只生成 manifest，不实际调用 ffmpeg。",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="默认丢弃音频；如需完整视频保留音频可启用该选项。前/后片段仍不保留音频。",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="输出清单文件名，默认 manifest.json。",
    )
    parser.add_argument(
        "--item-id",
        action="append",
        default=[],
        help="Flat manifest mode: export only this item id; may be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Flat manifest mode: export at most this many items.",
    )
    return parser.parse_args()


def load_json_with_small_repairs(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        repaired = repair_common_manual_json_edits(text)
        data = json.loads(repaired)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid JSON root object: {path}")
    return data


def repair_common_manual_json_edits(text: str) -> str:
    """Repair small hand-edit artifacts seen in review JSON files.

    This intentionally stays conservative: it only removes standalone empty
    string entries such as a stray line containing "" inside an object, and it
    closes an object immediately before a closing array/object line when manual
    edits left one brace short.
    """

    cleaned_lines = []
    stray_empty_string = re.compile(r'^\s*""\s*,?\s*$')
    for line in text.splitlines():
        if stray_empty_string.match(line):
            continue
        cleaned_lines.append(line)

    repaired_lines: list[str] = []
    stack: list[str] = []
    for line in cleaned_lines:
        stripped = line.lstrip()
        if stripped.startswith("]"):
            while stack and stack[-1] == "{":
                indent = line[: len(line) - len(stripped)] + "  "
                inserted = f"{indent}}}"
                repaired_lines.append(inserted)
                update_bracket_stack(stack, inserted)

        repaired_lines.append(line)
        update_bracket_stack(stack, line)

    return "\n".join(repaired_lines) + "\n"


def update_bracket_stack(stack: list[str], line: str) -> None:
    text = remove_json_strings(line)
    for char in text:
        if char in "{[":
            stack.append(char)
        elif char == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif char == "]":
            if stack and stack[-1] == "[":
                stack.pop()


def remove_json_strings(line: str) -> str:
    result = []
    in_string = False
    escaped = False
    for char in line:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        result.append(char)
    return "".join(result)


def normalize_extensions(extensions: list[str]) -> set[str]:
    normalized = set()
    for ext in extensions:
        ext = ext.strip().lower()
        if ext:
            normalized.add(ext if ext.startswith(".") else f".{ext}")
    return normalized


def build_video_index(video_root: Path | None, extensions: set[str]) -> dict[str, list[Path]]:
    if video_root is None:
        return {}
    if not video_root.is_dir():
        raise SystemExit(f"Video root does not exist or is not a directory: {video_root}")

    index: dict[str, list[Path]] = {}
    for path in video_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            index.setdefault(path.name, []).append(path)
    return index


def first_meaningful_char(value: str) -> str | None:
    for char in value.strip():
        if char.isalnum():
            return char
    return None


def is_video_after_prefix(video_name: str, prefix_letter: str) -> bool:
    char = first_meaningful_char(video_name)
    prefix = prefix_letter.strip()[:1]
    if char is None or not char.isalpha() or not prefix:
        return False
    return char.upper() > prefix.upper()


def get_review_status(segment: dict[str, Any]) -> str:
    review = segment.get("review")
    if not isinstance(review, dict):
        return ""
    return str(review.get("status", "")).strip().lower()


def safe_stem(value: str, max_len: int = 120) -> str:
    stem = Path(value).stem
    stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"_+", "_", stem).strip("._")
    return (stem or "video")[:max_len]


def resolve_video_path(video: dict[str, Any], video_index: dict[str, list[Path]]) -> tuple[Path | None, str | None]:
    name = video.get("video")
    if isinstance(name, str) and name in video_index:
        matches = video_index[name]
        if len(matches) == 1:
            return matches[0], None
        return None, f"ambiguous video_root matches for {name}: {len(matches)}"

    path_value = video.get("video_path")
    if isinstance(path_value, str):
        path = Path(path_value)
        if path.exists():
            return path, None
        if isinstance(name, str) and name in video_index:
            matches = video_index[name]
            if len(matches) == 1:
                return matches[0], None
            return None, f"ambiguous video_root matches for {name}: {len(matches)}"
        return None, f"video_path does not exist: {path}"

    return None, "missing video_path"


def get_export_window(segment: dict[str, Any]) -> tuple[float | None, float | None, str, str | None]:
    review = segment.get("review")
    if review is None:
        return as_float(segment.get("start_s")), as_float(segment.get("end_s")), "original", None
    if not isinstance(review, dict):
        return None, None, "invalid", "review is not an object"

    status = str(review.get("status", "")).strip().lower()
    if status == "reject":
        return None, None, "reject", None
    if status != "shift":
        return None, None, "invalid", f"unsupported review status: {status or '<empty>'}"

    final_window = review.get("final_window")
    if isinstance(final_window, dict):
        return (
            as_float(final_window.get("start_s")),
            as_float(final_window.get("end_s")),
            "final_window",
            None,
        )

    offset = as_float(review.get("final_start_offset_s"))
    start = as_float(segment.get("start_s"))
    end = as_float(segment.get("end_s"))
    if offset is None:
        return None, None, "invalid", "shift review missing final_window or final_start_offset_s"
    if start is None or end is None:
        return None, None, "invalid", "segment missing start_s/end_s"
    return start + offset, end + offset, "offset", None


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def log(message: str) -> None:
    print(message, flush=True)


def run_command(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        log("DRY RUN: " + " ".join(cmd))
        return
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{detail}")


def ffmpeg_base_args(overwrite: bool) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y" if overwrite else "-n"]


def export_frames_from_source(
    input_path: Path,
    frames_dir: Path,
    start_s: float,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if frames_dir.exists() and overwrite and not dry_run:
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        *ffmpeg_base_args(overwrite),
        "-i",
        str(input_path),
        "-ss",
        f"{start_s:.6f}",
        "-map",
        "0:v:0",
        "-vf",
        f"fps={TARGET_FPS_EXPR}",
        "-frames:v",
        str(TOTAL_FRAMES),
        "-q:v",
        "2",
        str(frames_dir / "frame_%06d.jpg"),
    ]
    run_command(cmd, dry_run)
    if not dry_run:
        frame_count = count_exported_frames(frames_dir)
        if frame_count != TOTAL_FRAMES:
            raise RuntimeError(f"expected {TOTAL_FRAMES} frames, got {frame_count}: {frames_dir}")


def export_video_from_frames(
    frames_dir: Path,
    output_path: Path,
    start_number: int,
    frame_count: int,
    overwrite: bool,
    dry_run: bool,
) -> None:
    cmd = [
        *ffmpeg_base_args(overwrite),
        "-framerate",
        TARGET_FPS_EXPR,
        "-start_number",
        str(start_number),
        "-i",
        str(frames_dir / "frame_%06d.jpg"),
        "-an",
        "-frames:v",
        str(frame_count),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        TARGET_FPS_EXPR,
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    run_command(cmd, dry_run)


def count_exported_frames(frames_dir: Path) -> int:
    return sum(1 for path in frames_dir.glob("frame_*.jpg") if path.is_file())


def probe_video_frame_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        return None
    value = result.stdout.strip().splitlines()
    if not value:
        return None
    try:
        return int(value[0])
    except ValueError:
        return None


def is_complete_video(path: Path, expected_frames: int) -> bool:
    return probe_video_frame_count(path) == expected_frames


def describe_existing_outputs(
    full_video_path: Path,
    frames_dir: Path,
    first_video_path: Path,
    last_video_path: Path,
) -> tuple[bool, list[str], int]:
    frame_count = count_exported_frames(frames_dir)
    missing_or_incomplete = []
    full_video_frames = probe_video_frame_count(full_video_path)
    if full_video_frames != TOTAL_FRAMES:
        missing_or_incomplete.append(
            f"expected {TOTAL_FRAMES} frames in full video, found {full_video_frames}: {full_video_path}"
        )
    if frame_count != TOTAL_FRAMES:
        missing_or_incomplete.append(
            f"expected {TOTAL_FRAMES} full frames, found {frame_count}: {frames_dir}"
        )
    first_video_frames = probe_video_frame_count(first_video_path)
    if first_video_frames != FIRST_FRAMES:
        missing_or_incomplete.append(
            f"expected {FIRST_FRAMES} frames in first 200 video, found {first_video_frames}: {first_video_path}"
        )
    last_video_frames = probe_video_frame_count(last_video_path)
    if last_video_frames != LAST_FRAMES:
        missing_or_incomplete.append(
            f"expected {LAST_FRAMES} frames in last 75 video, found {last_video_frames}: {last_video_path}"
        )
    return len(missing_or_incomplete) == 0, missing_or_incomplete, frame_count


def existing_output_names(
    full_video_path: Path,
    frames_dir: Path,
    first_video_path: Path,
    last_video_path: Path,
) -> list[str]:
    existing = []
    if is_complete_video(full_video_path, TOTAL_FRAMES):
        existing.append("full_video")
    if count_exported_frames(frames_dir) == TOTAL_FRAMES:
        existing.append("full_frames")
    if is_complete_video(first_video_path, FIRST_FRAMES):
        existing.append("first_200_video")
    if is_complete_video(last_video_path, LAST_FRAMES):
        existing.append("last_75_video")
    return existing


def make_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "full_videos": output_dir / "full_videos",
        "full_frames": output_dir / "full_frames",
        "first_200_videos": output_dir / "first_200_videos",
        "last_75_videos": output_dir / "last_75_videos",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def get_flat_item_id(item: dict[str, Any]) -> str:
    """Return the identifier used by public and legacy flat manifests."""

    return str(item.get("sample_id") or item.get("id") or "").strip()


def get_youtube_video_id(value: Any) -> str | None:
    """Extract an 11-character YouTube video ID from a manifest URL."""

    if not isinstance(value, str):
        return None
    match = re.search(r"(?:[?&]v=|youtu\.be/)([A-Za-z0-9_-]{11})(?:[&#?/]|$)", value)
    return match.group(1) if match else None


def resolve_flat_video_matches(
    source_item: dict[str, Any],
    video_index: dict[str, list[Path]],
) -> list[Path]:
    """Resolve a flat-manifest source by exact name, then YouTube ID."""

    video_name = str(source_item.get("video") or "").strip()
    exact_matches = video_index.get(video_name, [])
    if exact_matches:
        return exact_matches

    youtube_id = get_youtube_video_id(source_item.get("url"))
    if youtube_id is None:
        return []
    id_marker = f"[{youtube_id}]"
    id_matches = sorted(
        path
        for paths in video_index.values()
        for path in paths
        if id_marker in path.name or path.stem == youtube_id
    )
    expected_suffix = Path(video_name).suffix.lower()
    suffix_matches = [path for path in id_matches if path.suffix.lower() == expected_suffix]
    return suffix_matches or id_matches


def export_flat_manifest(
    args: argparse.Namespace,
    data: dict[str, Any],
    video_index: dict[str, list[Path]],
) -> None:
    """Recover benchmark outputs from the public flat manifest schema."""

    source_items = data.get("items")
    if not isinstance(source_items, list):
        raise SystemExit("Flat manifest is invalid: missing items list.")

    selected_ids = set(args.item_id)
    selected_items = [
        item
        for item in source_items
        if isinstance(item, dict)
        and (not selected_ids or get_flat_item_id(item) in selected_ids)
    ]
    if selected_ids:
        found_ids = {get_flat_item_id(item) for item in selected_items}
        missing_ids = sorted(selected_ids - found_ids)
        if missing_ids:
            raise SystemExit(f"Requested item ids are absent from manifest: {missing_ids}")
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        selected_items = selected_items[: args.limit]

    output_dirs = make_output_dirs(args.output_dir)
    recovery_manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "flat_manifest_recovery",
        "input_json": str(args.input_json),
        "output_dir": str(args.output_dir),
        "target_fps": TARGET_FPS_EXPR,
        "total_frames": TOTAL_FRAMES,
        "first_frames": FIRST_FRAMES,
        "last_frames": LAST_FRAMES,
        "items": [],
        "summary": {
            "items_selected": len(selected_items),
            "items_exported": 0,
            "items_existing": 0,
            "items_failed": 0,
        },
    }

    for source_item in selected_items:
        item_id = get_flat_item_id(source_item)
        video_name = str(source_item.get("video") or "").strip()
        start_s = as_float(source_item.get("start_s"))
        end_s = as_float(source_item.get("end_s"))
        item = dict(source_item)
        item["outputs"] = {
            "full_video": str(output_dirs["full_videos"] / f"{item_id}.mp4"),
            "full_frames": str(output_dirs["full_frames"] / item_id),
            "first_200_video": str(output_dirs["first_200_videos"] / f"{item_id}.mp4"),
            "last_75_video": str(output_dirs["last_75_videos"] / f"{item_id}.mp4"),
        }
        recovery_manifest["items"].append(item)

        if not item_id or not video_name or start_s is None:
            item["status"] = "failed"
            item["reason"] = "item is missing sample_id/id, video or start_s"
            recovery_manifest["summary"]["items_failed"] += 1
            continue
        if start_s < 0:
            item["status"] = "failed"
            item["reason"] = f"invalid start_s: {start_s}"
            recovery_manifest["summary"]["items_failed"] += 1
            continue
        if end_s is not None and end_s <= start_s:
            item["status"] = "failed"
            item["reason"] = f"invalid optional end_s: {start_s} - {end_s}"
            recovery_manifest["summary"]["items_failed"] += 1
            continue

        # If present, end_s records the reviewed source-window end and is not the
        # output truncation boundary. Recovery starts at start_s,
        # resamples to TARGET_FPS_EXPR, and validates that exactly TOTAL_FRAMES
        # were decoded. Rejecting from end_s - start_s would incorrectly exclude
        # valid clips whose final source-frame PTS spans fewer than 275 output
        # frame intervals because of source-FPS timestamp rounding.

        matches = resolve_flat_video_matches(source_item, video_index)
        if len(matches) != 1:
            item["status"] = "failed"
            item["reason"] = (
                f"expected exactly one source named {video_name!r} under --video-root, "
                f"found {len(matches)}"
            )
            recovery_manifest["summary"]["items_failed"] += 1
            continue
        input_path = matches[0]

        full_video_path = Path(item["outputs"]["full_video"])
        frames_dir = Path(item["outputs"]["full_frames"])
        first_video_path = Path(item["outputs"]["first_200_video"])
        last_video_path = Path(item["outputs"]["last_75_video"])
        outputs_complete, output_problems, frame_count = describe_existing_outputs(
            full_video_path,
            frames_dir,
            first_video_path,
            last_video_path,
        )
        if outputs_complete and not args.overwrite:
            item["status"] = "existing"
            item["frame_count"] = frame_count
            recovery_manifest["summary"]["items_existing"] += 1
            log(f"[existing] {item_id}: all recovery outputs complete")
            continue

        try:
            if args.overwrite or frame_count != TOTAL_FRAMES:
                export_frames_from_source(
                    input_path=input_path,
                    frames_dir=frames_dir,
                    start_s=start_s,
                    overwrite=args.overwrite or frames_dir.exists(),
                    dry_run=args.dry_run,
                )
            if args.overwrite or not is_complete_video(full_video_path, TOTAL_FRAMES):
                export_video_from_frames(
                    frames_dir, full_video_path, 1, TOTAL_FRAMES, True, args.dry_run
                )
            if args.overwrite or not is_complete_video(first_video_path, FIRST_FRAMES):
                export_video_from_frames(
                    frames_dir, first_video_path, 1, FIRST_FRAMES, True, args.dry_run
                )
            if args.overwrite or not is_complete_video(last_video_path, LAST_FRAMES):
                export_video_from_frames(
                    frames_dir,
                    last_video_path,
                    FIRST_FRAMES + 1,
                    LAST_FRAMES,
                    True,
                    args.dry_run,
                )
        except RuntimeError as exc:
            item["status"] = "failed"
            item["reason"] = str(exc)
            item["recovery_checks_before_export"] = output_problems
            recovery_manifest["summary"]["items_failed"] += 1
            log(f"[failed] {item_id}: {exc}")
            continue

        item["status"] = "dry_run" if args.dry_run else "exported"
        item["frame_count"] = TOTAL_FRAMES
        recovery_manifest["summary"]["items_exported"] += 1
        log(f"[{item['status']}] {item_id}: flat manifest recovery complete")

    manifest_path = args.output_dir / args.manifest_name
    manifest_path.write_text(
        json.dumps(recovery_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"Recovery manifest written to: {manifest_path}")
    log(json.dumps(recovery_manifest["summary"], ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None and not args.dry_run:
        raise SystemExit("ffmpeg not found in PATH")
    if shutil.which("ffprobe") is None and not args.dry_run:
        raise SystemExit("ffprobe not found in PATH")

    data = load_json_with_small_repairs(args.input_json)
    if isinstance(data.get("items"), list):
        video_index = build_video_index(
            args.video_root, normalize_extensions(args.extensions)
        )
        export_flat_manifest(args, data, video_index)
        return
    videos = data.get("videos")
    if not isinstance(videos, list):
        raise SystemExit("Input JSON is invalid: missing videos list.")

    output_dirs = make_output_dirs(args.output_dir)
    video_index = build_video_index(args.video_root, normalize_extensions(args.extensions))

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_json": str(args.input_json),
        "output_dir": str(args.output_dir),
        "target_fps": TARGET_FPS_EXPR,
        "total_frames": TOTAL_FRAMES,
        "first_frames": FIRST_FRAMES,
        "last_frames": LAST_FRAMES,
        "items": [],
        "summary": {
            "videos_seen": 0,
            "videos_after_prefix": 0,
            "segments_seen": 0,
            "segments_non_shift": 0,
            "segments_exported": 0,
            "segments_existing": 0,
            "segments_completed": 0,
            "segments_rejected": 0,
            "segments_skipped": 0,
        },
    }

    for video_i, video in enumerate(videos):
        if not isinstance(video, dict):
            continue
        manifest["summary"]["videos_seen"] += 1
        video_name = str(video.get("video") or Path(str(video.get("video_path", ""))).name)
        if not is_video_after_prefix(video_name, args.after_prefix_letter):
            continue
        manifest["summary"]["videos_after_prefix"] += 1

        input_path, path_error = resolve_video_path(video, video_index)
        segments = video.get("segments")
        if not isinstance(segments, list):
            continue

        for segment in segments:
            if not isinstance(segment, dict):
                continue
            manifest["summary"]["segments_seen"] += 1
            if get_review_status(segment) != "shift":
                manifest["summary"]["segments_non_shift"] += 1
                continue
            segment_index = int(segment.get("segment_index", len(manifest["items"])))
            start_s, end_s, source, window_error = get_export_window(segment)

            item_id = f"{safe_stem(video_name)}__segment_{segment_index:03d}"
            item = {
                "id": item_id,
                "video_index": video_i,
                "video": video_name,
                "segment_index": segment_index,
                "window_source": source,
                "start_s": start_s,
                "end_s": end_s,
                "outputs": {
                    "full_video": str(output_dirs["full_videos"] / f"{item_id}.mp4"),
                    "full_frames": str(output_dirs["full_frames"] / item_id),
                    "first_200_video": str(output_dirs["first_200_videos"] / f"{item_id}.mp4"),
                    "last_75_video": str(output_dirs["last_75_videos"] / f"{item_id}.mp4"),
                },
            }

            full_video_path = Path(item["outputs"]["full_video"])
            if not args.overwrite and full_video_path.exists():
                manifest["summary"]["segments_existing"] += 1
                item["status"] = "existing"
                item["reason"] = "full target video already exists"
                manifest["items"].append(item)
                log(f"[existing] {item_id}: full target video already exists")
                continue
            if path_error is not None:
                manifest["summary"]["segments_skipped"] += 1
                item["status"] = "skipped"
                item["reason"] = path_error
                manifest["items"].append(item)
                continue
            if window_error is not None or start_s is None or end_s is None:
                manifest["summary"]["segments_skipped"] += 1
                item["status"] = "skipped"
                item["reason"] = window_error or "invalid start_s/end_s"
                manifest["items"].append(item)
                continue
            if start_s < 0 or end_s <= start_s:
                manifest["summary"]["segments_skipped"] += 1
                item["status"] = "skipped"
                item["reason"] = f"invalid export window: {start_s} - {end_s}"
                manifest["items"].append(item)
                continue
            if end_s - start_s < TOTAL_FRAMES / TARGET_FPS_FLOAT:
                manifest["summary"]["segments_skipped"] += 1
                item["status"] = "skipped"
                item["reason"] = "window is shorter than 275 frames at 29.97fps"
                manifest["items"].append(item)
                continue

            assert input_path is not None
            first_video_path = Path(item["outputs"]["first_200_video"])
            last_video_path = Path(item["outputs"]["last_75_video"])
            frames_dir = Path(item["outputs"]["full_frames"])
            outputs_complete, output_problems, frame_count = describe_existing_outputs(
                full_video_path,
                frames_dir,
                first_video_path,
                last_video_path,
            )
            if not args.overwrite and outputs_complete:
                manifest["summary"]["segments_existing"] += 1
                item["status"] = "existing"
                item["frame_count"] = frame_count
                manifest["items"].append(item)
                log(f"[existing] {item_id}: all outputs complete")
                continue
            if output_problems:
                for problem in output_problems:
                    log(f"[missing] {item_id}: {problem}")
            existing_before = existing_output_names(
                full_video_path,
                frames_dir,
                first_video_path,
                last_video_path,
            )
            generated_outputs = []

            try:
                if args.overwrite or frame_count != TOTAL_FRAMES:
                    log(
                        f"[repair] {item_id}: extracting full_frames from source "
                        f"{input_path} start={start_s:.6f}s end={end_s:.6f}s -> {frames_dir}"
                    )
                    export_frames_from_source(
                        input_path=input_path,
                        frames_dir=frames_dir,
                        start_s=start_s,
                        overwrite=args.overwrite or frames_dir.exists(),
                        dry_run=args.dry_run,
                    )
                    generated_outputs.append("full_frames")
                    frame_count = TOTAL_FRAMES
                if args.overwrite or not is_complete_video(full_video_path, TOTAL_FRAMES):
                    log(f"[repair] {item_id}: generating full_video from frames -> {full_video_path}")
                    export_video_from_frames(
                        frames_dir=frames_dir,
                        output_path=full_video_path,
                        start_number=1,
                        frame_count=TOTAL_FRAMES,
                        overwrite=True,
                        dry_run=args.dry_run,
                    )
                    generated_outputs.append("full_video")
                if args.overwrite or not is_complete_video(first_video_path, FIRST_FRAMES):
                    log(f"[repair] {item_id}: generating first_200_video from frames -> {first_video_path}")
                    export_video_from_frames(
                        frames_dir=frames_dir,
                        output_path=first_video_path,
                        start_number=1,
                        frame_count=FIRST_FRAMES,
                        overwrite=True,
                        dry_run=args.dry_run,
                    )
                    generated_outputs.append("first_200_video")
                if args.overwrite or not is_complete_video(last_video_path, LAST_FRAMES):
                    log(f"[repair] {item_id}: generating last_75_video from frames -> {last_video_path}")
                    export_video_from_frames(
                        frames_dir=frames_dir,
                        output_path=last_video_path,
                        start_number=FIRST_FRAMES + 1,
                        frame_count=LAST_FRAMES,
                        overwrite=True,
                        dry_run=args.dry_run,
                    )
                    generated_outputs.append("last_75_video")
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                manifest["summary"]["segments_skipped"] += 1
                item["status"] = "failed"
                if isinstance(exc, subprocess.CalledProcessError):
                    item["reason"] = f"ffmpeg failed with exit code {exc.returncode}"
                else:
                    item["reason"] = str(exc)
                manifest["items"].append(item)
                log(f"[failed] {item_id}: {item['reason']}")
                continue

            if existing_before and generated_outputs and not args.overwrite:
                manifest["summary"]["segments_completed"] += 1
                item["status"] = "completed" if not args.dry_run else "dry_run"
                item["existing_outputs"] = existing_before
            else:
                manifest["summary"]["segments_exported"] += 1
                item["status"] = "exported" if not args.dry_run else "dry_run"
            item["generated_outputs"] = generated_outputs
            if output_problems and generated_outputs:
                item["repaired_outputs"] = output_problems
            if not args.dry_run:
                item["frame_count"] = count_exported_frames(frames_dir)
            manifest["items"].append(item)
            log(f"[{item['status']}] {item_id}: generated {', '.join(generated_outputs) or 'nothing'}")

    manifest_path = args.output_dir / args.manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Manifest written to: {manifest_path}")
    log(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
