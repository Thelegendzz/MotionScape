#!/usr/bin/env python3
"""
字幕检测（不剪辑视频）：
1) 使用 ffmpeg 按指定采样率抽帧（默认 1fps）。
2) 每帧整图送入 PP-OCRv4（PaddleOCR）。
3) 仅当 OCR 结果 score > 阈值 且连续出现时，判定该视频“有字幕”。
4) 输出单个 JSON，包含是否有字幕及字幕出现/消失时刻。

注意：
- 该脚本不修改、不剪辑任何视频。
- 需要 ffmpeg/ffprobe 与 paddleocr 依赖。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: numpy. Install with `pip install numpy`.") from exc

try:
    from paddleocr import PaddleOCR
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: paddleocr. Install with `pip install paddleocr`.") from exc


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_OCR: PaddleOCR | None = None


def configure_paddle_runtime() -> None:
    # 规避部分容器环境下 PIR + oneDNN 的不兼容报错。
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")


@dataclass
class SubtitleSegment:
    start_s: float
    end_s: float


@dataclass
class VideoSubtitleResult:
    video_name: str
    video_path: str
    fps: float
    total_frames: int
    duration_s: float
    has_subtitle: bool
    subtitle_segments: List[SubtitleSegment]
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 PP-OCRv4 的字幕检测（仅输出 JSON）。")
    parser.add_argument("--input-dir", type=Path, required=True, help="输入视频目录")
    parser.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path("subtitle_report.json"),
        help="输出 JSON 路径（默认当前目录 subtitle_report.json）",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.85,
        help="OCR 置信度阈值，只有 score > 该值才算命中",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="采样率（fps），默认 1.0",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(11, max(1, (os.cpu_count() or 2) - 1)),
        help="并行进程数，默认 min(8, CPU核数-1)",
    )
    parser.add_argument("--lang", type=str, default="en", help="PaddleOCR 语言模型，如 ch/en")
    parser.add_argument("--no-angle-cls", action="store_true", help="关闭方向分类以加速")
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


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("Missing ffmpeg/ffprobe in PATH.")


def get_ocr(lang: str, use_textline_orientation: bool) -> PaddleOCR:
    global _OCR
    if _OCR is None:
        configure_paddle_runtime()
        try:
            import paddle

            paddle.set_flags({"FLAGS_enable_pir_api": False, "FLAGS_use_mkldnn": False})
        except Exception:
            pass
        # 兼容不同 paddleocr 版本的构造参数差异。
        candidates = [
            {
                "lang": lang,
                "use_textline_orientation": use_textline_orientation,
                "text_det_limit_side_len": 1920,
            },
            {
                "lang": lang,
                "use_angle_cls": use_textline_orientation,
                "det_limit_side_len": 1920,
            },
            {
                "lang": lang,
            },
        ]
        last_error: Exception | None = None
        for kwargs in candidates:
            try:
                _OCR = PaddleOCR(**kwargs)
                break
            except Exception as exc:
                last_error = exc
        if _OCR is None:
            raise RuntimeError(f"failed to init PaddleOCR: {last_error}")
    return _OCR


def format_hhmmss(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def ffprobe_video_meta(video_path: Path) -> Tuple[float, int, float, int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    stream = data.get("streams", [{}])[0]

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)

    avg_rate = str(stream.get("avg_frame_rate") or "0/1")
    if "/" in avg_rate:
        num, den = avg_rate.split("/", 1)
        fps = (float(num) / float(den)) if float(den) != 0 else 0.0
    else:
        fps = float(avg_rate or 0.0)

    nb_frames_raw = stream.get("nb_frames")
    total_frames = int(nb_frames_raw) if nb_frames_raw not in (None, "N/A") else 0

    duration_raw = stream.get("duration")
    duration_s = float(duration_raw) if duration_raw not in (None, "N/A") else 0.0

    if duration_s <= 0 and total_frames > 0 and fps > 0:
        duration_s = total_frames / fps
    if total_frames <= 0 and duration_s > 0 and fps > 0:
        total_frames = int(round(duration_s * fps))

    return fps, total_frames, duration_s, width, height


def ocr_frame_hit(frame_bgr: np.ndarray, score_threshold: float, lang: str, use_textline_orientation: bool) -> bool:
    ocr = get_ocr(lang=lang, use_textline_orientation=use_textline_orientation)
    result = ocr.ocr(frame_bgr)
    if not result:
        return False

    # 兼容旧版格式：[[[box], [text, score]], ...]
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        lines = result[0]
        for line in lines:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            rec = line[1]
            if not isinstance(rec, (list, tuple)) or len(rec) < 2:
                continue
            try:
                if float(rec[1]) > score_threshold:
                    return True
            except (TypeError, ValueError):
                continue

    # 兼容新版格式：[{..., "rec_scores": [...], ...}, ...]
    if isinstance(result, list):
        for item in result:
            if not isinstance(item, dict):
                continue
            scores = item.get("rec_scores")
            if not isinstance(scores, list):
                continue
            for sc in scores:
                try:
                    if float(sc) > score_threshold:
                        return True
                except (TypeError, ValueError):
                    continue

    return False


def frame_ranges_from_flags(flags: Sequence[bool]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if not f and start is not None:
            ranges.append((start, i - 1))
            start = None
    if start is not None:
        ranges.append((start, len(flags) - 1))
    return ranges


def detect_subtitle_segments(
    sample_times_s: Sequence[float],
    sample_hits: Sequence[bool],
    sample_interval_s: float,
) -> List[SubtitleSegment]:
    if not sample_times_s or not sample_hits:
        return []

    segments: List[SubtitleSegment] = []
    for s, e in frame_ranges_from_flags(sample_hits):
        segments.append(
            SubtitleSegment(start_s=round(sample_times_s[s], 3), end_s=round(sample_times_s[e] + sample_interval_s, 3))
        )
    if not segments:
        return segments

    # 如果两段字幕间隔约等于 1 秒，则合并。
    merged: List[SubtitleSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg.start_s - prev.end_s
        if abs(gap - 1.0) <= 1e-3:
            merged[-1] = SubtitleSegment(start_s=prev.start_s, end_s=seg.end_s)
        else:
            merged.append(seg)
    return merged


def iter_ffmpeg_sampled_frames(
    video_path: Path,
    width: int,
    height: int,
    sample_fps: float,
) -> Iterable[np.ndarray]:
    frame_bytes = width * height * 3
    vf = f"fps={sample_fps:.6f}"
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdout is not None
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        stderr_text = ""
        if proc.stderr is not None:
            stderr_text = proc.stderr.read().decode("utf-8", errors="ignore").strip()
            proc.stderr.close()
        rc = proc.wait()
        if rc != 0 and "Broken pipe" not in stderr_text:
            raise RuntimeError(f"ffmpeg failed for {video_path.name}: {stderr_text}")


def process_video(
    video_path: Path,
    score_threshold: float,
    sample_fps: float,
    lang: str,
    use_textline_orientation: bool,
) -> VideoSubtitleResult:
    try:
        fps, total_frames, duration_s, width, height = ffprobe_video_meta(video_path)
    except Exception as exc:
        return VideoSubtitleResult(video_path.name, str(video_path), 0.0, 0, 0.0, False, [], f"ffprobe_error:{exc}")

    if width <= 0 or height <= 0:
        return VideoSubtitleResult(video_path.name, str(video_path), round(fps, 4), total_frames, round(duration_s, 3), False, [], "invalid_resolution")

    sample_times_s: List[float] = []
    sample_hits: List[bool] = []
    sample_interval_s = 1.0 / max(sample_fps, 1e-6)
    first_ocr_error = ""

    sample_index = 0
    try:
        for frame in iter_ffmpeg_sampled_frames(video_path, width, height, sample_fps=sample_fps):
            try:
                hit = ocr_frame_hit(frame, score_threshold, lang, use_textline_orientation)
            except Exception as exc:
                if not first_ocr_error:
                    first_ocr_error = f"ocr_error:{exc}"
                hit = False
            sample_times_s.append(sample_index * sample_interval_s)
            sample_hits.append(hit)
            sample_index += 1
    except Exception as exc:
        return VideoSubtitleResult(
            video_path.name,
            str(video_path),
            round(fps, 4),
            total_frames,
            round(duration_s, 3),
            False,
            [],
            f"ffmpeg_pipe_error:{exc}",
        )

    segments = detect_subtitle_segments(sample_times_s, sample_hits, sample_interval_s)
    return VideoSubtitleResult(
        video_name=video_path.name,
        video_path=str(video_path),
        fps=round(fps, 4),
        total_frames=total_frames,
        duration_s=round(duration_s, 3),
        has_subtitle=bool(segments),
        subtitle_segments=segments,
        error=first_ocr_error,
    )


def process_video_task(task: Tuple[Path, float, float, str, bool]) -> Tuple[Path, VideoSubtitleResult]:
    vp, score_threshold, sample_fps, lang, use_textline_orientation = task
    return vp, process_video(vp, score_threshold, sample_fps, lang, use_textline_orientation)


def main() -> None:
    configure_paddle_runtime()
    require_ffmpeg()
    args = parse_args()

    videos = list(iter_videos(args.input_dir, args.recursive))
    if not videos:
        print(f"[WARN] no videos found in {args.input_dir}")
        return

    sample_fps = max(0.1, float(args.sample_fps))
    worker_count = max(1, int(args.workers))

    tasks = [
        (
            vp,
            float(args.score_threshold),
            sample_fps,
            str(args.lang),
            not bool(args.no_angle_cls),
        )
        for vp in videos
    ]

    print(f"[INFO] total videos={len(videos)} workers={worker_count} sample_fps={sample_fps}")

    results_map: Dict[Path, VideoSubtitleResult] = {}
    if worker_count == 1:
        for t in tasks:
            vp, result = process_video_task(t)
            results_map[vp] = result
            print(f"[DONE] {vp.name} has_subtitle={result.has_subtitle} segments={len(result.subtitle_segments)} err={result.error}")
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(process_video_task, t): t[0] for t in tasks}
            for future in as_completed(future_map):
                vp = future_map[future]
                try:
                    _, result = future.result()
                except Exception as exc:
                    result = VideoSubtitleResult(vp.name, str(vp), 0.0, 0, 0.0, False, [], f"worker_error:{exc}")
                results_map[vp] = result
                print(f"[DONE] {vp.name} has_subtitle={result.has_subtitle} segments={len(result.subtitle_segments)} err={result.error}")

    ordered_results: List[Dict[str, Any]] = []
    has_subtitle_count = 0
    for vp in sorted(videos, key=lambda x: str(x)):
        r = results_map[vp]
        if r.has_subtitle:
            has_subtitle_count += 1
        ordered_results.append(
            {
                "video_name": r.video_name,
                "video_path": r.video_path,
                "fps": r.fps,
                "total_frames": r.total_frames,
                "duration_s": r.duration_s,
                "has_subtitle": r.has_subtitle,
                "subtitle_segments": [
                    {
                        "start_s": seg.start_s,
                        "end_s": seg.end_s,
                        "start_hhmmss": format_hhmmss(seg.start_s),
                        "end_hhmmss": format_hhmmss(seg.end_s),
                    }
                    for seg in r.subtitle_segments
                ],
                "error": r.error,
            }
        )

    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(args.input_dir),
        "recursive": bool(args.recursive),
        "logic": {
            "sample_fps": sample_fps,
            "crop_region": "full_frame",
            "score_threshold": float(args.score_threshold),
            "ocr": "PP-OCRv4 (PaddleOCR)",
            "decoder": "ffmpeg",
        },
        "summary": {
            "total_videos": len(videos),
            "has_subtitle_videos": has_subtitle_count,
            "no_subtitle_videos": len(videos) - has_subtitle_count,
        },
        "videos": ordered_results,
    }

    json_path = args.json_path
    if not json_path.is_absolute():
        json_path = Path.cwd() / json_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[JSON] {json_path}")


if __name__ == "__main__":
    main()
