#!/usr/bin/env python3
"""
字幕 + 淡入淡出检测（不剪辑视频）：
1) ffmpeg 抽帧（字幕默认 1fps，淡入淡出可独立高采样）。
2) 字幕双通道检测：
   - OCR 识别分数通道（常规字幕）
   - 文本区域通道（仅看检测框面积，覆盖艺术字/水印）
3) 淡入淡出检测（亮度/对比度时序）。
4) 输出单个 JSON。
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
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")


@dataclass
class SubtitleSegment:
    start_s: float
    end_s: float


@dataclass
class FadeSegment:
    fade_type: str
    start_s: float
    end_s: float


@dataclass
class VideoResult:
    video_name: str
    video_path: str
    fps: float
    total_frames: int
    duration_s: float
    has_subtitle: bool
    has_subtitle_ocr: bool
    has_subtitle_region: bool
    subtitle_segments: List[SubtitleSegment]
    has_fade: bool
    fade_segments: List[FadeSegment]
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="字幕+淡入淡出检测（仅输出 JSON）。")
    parser.add_argument("--input-dir", type=Path, required=True, help="输入视频目录")
    parser.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    parser.add_argument("--json-path", type=Path, default=Path("subtitle_report.json"), help="输出 JSON 路径")

    parser.add_argument("--sample-fps", type=float, default=1.0, help="字幕检测采样率（fps）")
    parser.add_argument("--score-threshold", type=float, default=0.85, help="OCR 分数阈值")
    parser.add_argument("--subtitle-min-hit-frames", type=int, default=2, help="字幕判定最小连续命中帧数")
    parser.add_argument("--region-min-area-ratio", type=float, default=0.015, help="文本区域最小面积占比")

    parser.add_argument("--workers", type=int, default=min(11, max(1, (os.cpu_count() or 2) - 1)), help="并行进程数")
    parser.add_argument("--lang", type=str, default="en", help="PaddleOCR 语言模型")
    parser.add_argument("--no-angle-cls", action="store_true", help="关闭方向分类")

    parser.add_argument("--fade-dark-threshold", type=float, default=45.0)
    parser.add_argument("--fade-min-delta", type=float, default=35.0)
    parser.add_argument("--fade-min-duration", type=float, default=0.5)
    parser.add_argument("--fade-max-duration", type=float, default=3.0)
    parser.add_argument("--fade-sample-fps", type=float, default=8.0)
    parser.add_argument("--fade-scale-width", type=int, default=192)
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

            paddle.set_flags(
                {
                    "FLAGS_enable_pir_api": False,
                    "FLAGS_enable_pir_in_executor": False,
                    "FLAGS_use_mkldnn": False,
                }
            )
        except Exception:
            pass

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
            {"lang": lang},
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


def frame_ranges_from_flags(flags: Sequence[bool]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if (not f) and start is not None:
            ranges.append((start, i - 1))
            start = None
    if start is not None:
        ranges.append((start, len(flags) - 1))
    return ranges


def keep_only_long_true_runs(flags: Sequence[bool], min_len: int) -> List[bool]:
    if min_len <= 1:
        return list(flags)
    out = [False] * len(flags)
    for s, e in frame_ranges_from_flags(flags):
        if (e - s + 1) >= min_len:
            for i in range(s, e + 1):
                out[i] = True
    return out


def parse_ocr_result(result: Any) -> Tuple[List[float], List[np.ndarray]]:
    scores: List[float] = []
    boxes: List[np.ndarray] = []

    if not isinstance(result, list):
        return scores, boxes

    # 旧版：result[0] -> [[box, [text, score]], ...]
    if len(result) > 0 and isinstance(result[0], list):
        for line in result[0]:
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                box = line[0]
                rec = line[1]
                try:
                    if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                        scores.append(float(rec[1]))
                except Exception:
                    pass
                try:
                    arr = np.asarray(box, dtype=np.float32)
                    if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] == 2:
                        boxes.append(arr)
                except Exception:
                    pass

    # 新版：dict 结构，可能含 rec_scores / dt_polys
    for item in result:
        if not isinstance(item, dict):
            continue
        rs = item.get("rec_scores")
        if isinstance(rs, list):
            for x in rs:
                try:
                    scores.append(float(x))
                except Exception:
                    pass
        polys = item.get("dt_polys")
        if isinstance(polys, list):
            for p in polys:
                try:
                    arr = np.asarray(p, dtype=np.float32)
                    if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] == 2:
                        boxes.append(arr)
                except Exception:
                    pass

    return scores, boxes


def subtitle_dual_channel_hit(
    frame_bgr: np.ndarray,
    score_threshold: float,
    region_min_area_ratio: float,
    lang: str,
    use_textline_orientation: bool,
) -> Tuple[bool, bool]:
    ocr = get_ocr(lang=lang, use_textline_orientation=use_textline_orientation)
    result = ocr.ocr(frame_bgr)
    if not result:
        return False, False

    scores, boxes = parse_ocr_result(result)
    ocr_hit = any(s > score_threshold for s in scores)

    h, w = frame_bgr.shape[:2]
    frame_area = float(max(h * w, 1))
    region_hit = False
    for b in boxes:
        area = float(abs(np.cross(b[1] - b[0], b[2] - b[0]))) if b.shape[0] >= 3 else 0.0
        if area / frame_area >= region_min_area_ratio:
            region_hit = True
            break

    return ocr_hit, region_hit


def detect_subtitle_segments(
    sample_times_s: Sequence[float],
    sample_hits: Sequence[bool],
    sample_interval_s: float,
) -> List[SubtitleSegment]:
    if not sample_times_s or not sample_hits:
        return []
    segs: List[SubtitleSegment] = []
    for s, e in frame_ranges_from_flags(sample_hits):
        segs.append(SubtitleSegment(round(sample_times_s[s], 3), round(sample_times_s[e] + sample_interval_s, 3)))

    if not segs:
        return segs
    # 保留你之前的规则：间隔 1 秒合并
    merged: List[SubtitleSegment] = [segs[0]]
    for seg in segs[1:]:
        prev = merged[-1]
        if abs((seg.start_s - prev.end_s) - 1.0) <= 1e-3:
            merged[-1] = SubtitleSegment(prev.start_s, seg.end_s)
        else:
            merged.append(seg)
    return merged


def detect_fade_segments(
    sample_times_s: Sequence[float],
    sample_brightness: Sequence[float],
    sample_contrast: Sequence[float],
    sample_interval_s: float,
    dark_threshold: float,
    min_delta: float,
    min_duration: float,
    max_duration: float,
) -> List[FadeSegment]:
    if len(sample_times_s) < 2 or len(sample_brightness) < 2 or len(sample_contrast) < 2:
        return []

    t = np.asarray(sample_times_s, dtype=np.float32)
    b = np.asarray(sample_brightness, dtype=np.float32)
    c = np.asarray(sample_contrast, dtype=np.float32)
    n = min(len(t), len(b), len(c))
    if n < 2:
        return []
    t, b, c = t[:n], b[:n], c[:n]

    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    b_smooth = np.convolve(b, kernel, mode="same")
    diff = np.diff(b_smooth)
    if len(diff) == 0:
        return []

    slope_thr = max(1.0, float(np.std(diff)) * 0.8)
    min_frames = max(1, int(round(min_duration / max(sample_interval_s, 1e-6))))
    max_frames = max(min_frames, int(round(max_duration / max(sample_interval_s, 1e-6))))
    contrast_ref = float(np.percentile(c, 60))
    contrast_dark_limit = max(8.0, contrast_ref * 0.55)

    fades: List[FadeSegment] = []

    # 片头低谷->峰值
    head_end = min(n - 1, int(round(max_frames * 1.5)))
    if head_end >= 2:
        head_curve = b_smooth[: head_end + 1]
        head_contrast = c[: head_end + 1]
        low_idx = int(np.argmin(head_curve))
        if low_idx < head_end:
            high_idx = low_idx + int(np.argmax(head_curve[low_idx:]))
            if high_idx > low_idx:
                delta = float(b_smooth[high_idx] - b_smooth[low_idx])
                dur = float(t[high_idx] - t[low_idx]) + sample_interval_s
                local_diff = diff[low_idx:high_idx]
                up_ratio = float(np.mean(local_diff >= (-slope_thr * 0.2))) if len(local_diff) > 0 else 0.0
                if (
                    delta >= (min_delta * 0.45)
                    and dur >= (min_duration * 0.3)
                    and dur <= (max_duration * 1.2)
                    and up_ratio >= 0.65
                    and float(np.min(head_curve[: max(low_idx + 1, min_frames)])) <= (dark_threshold + 25.0)
                    and float(np.min(head_contrast[: max(low_idx + 1, min_frames)])) <= contrast_dark_limit
                ):
                    fades.append(FadeSegment("fade_in", round(float(t[low_idx]), 3), round(float(t[high_idx] + sample_interval_s), 3)))

    # 片尾高峰->低谷
    tail_start = max(0, n - int(round(max_frames * 1.5)) - 1)
    if tail_start < n - 2:
        high_idx = tail_start + int(np.argmax(b_smooth[tail_start:]))
        if high_idx < n - 1:
            low_idx = high_idx + int(np.argmin(b_smooth[high_idx:]))
            if low_idx > high_idx:
                delta = float(b_smooth[high_idx] - b_smooth[low_idx])
                dur = float(t[low_idx] - t[high_idx]) + sample_interval_s
                local_diff = diff[high_idx:low_idx]
                down_ratio = float(np.mean(local_diff <= (slope_thr * 0.2))) if len(local_diff) > 0 else 0.0
                low_contrast = float(np.min(c[max(high_idx, low_idx - min_frames + 1): low_idx + 1]))
                low_dark = float(np.min(b_smooth[max(high_idx, low_idx - min_frames + 1): low_idx + 1]))
                if (
                    delta >= (min_delta * 0.45)
                    and dur >= (min_duration * 0.3)
                    and dur <= (max_duration * 1.2)
                    and down_ratio >= 0.65
                    and low_dark <= (dark_threshold + 25.0)
                    and low_contrast <= contrast_dark_limit
                ):
                    fades.append(FadeSegment("fade_out", round(float(t[high_idx]), 3), round(float(t[low_idx] + sample_interval_s), 3)))

    # 通用斜率扫描
    i = 0
    while i < len(diff):
        d0 = float(diff[i])
        if abs(d0) < slope_thr:
            i += 1
            continue
        sign = 1.0 if d0 > 0 else -1.0
        j = i
        while j < len(diff) and (j - i + 1) <= max_frames:
            dj = float(diff[j])
            if dj * sign < -slope_thr:
                break
            j += 1
        end_idx = j
        run_len = end_idx - i
        if run_len >= min_frames:
            delta = float(b_smooth[end_idx] - b_smooth[i])
            dur = float(t[end_idx] - t[i]) + sample_interval_s
            if sign > 0:
                if b_smooth[i] <= dark_threshold and c[i] <= contrast_dark_limit and delta >= min_delta and dur <= max_duration:
                    fades.append(FadeSegment("fade_in", round(float(t[i]), 3), round(float(t[end_idx] + sample_interval_s), 3)))
            else:
                if b_smooth[end_idx] <= dark_threshold and c[end_idx] <= contrast_dark_limit and (-delta) >= min_delta and dur <= max_duration:
                    fades.append(FadeSegment("fade_out", round(float(t[i]), 3), round(float(t[end_idx] + sample_interval_s), 3)))
        i = max(i + 1, end_idx)

    if not fades:
        return fades
    fades.sort(key=lambda x: (x.start_s, x.end_s))
    merged: List[FadeSegment] = [fades[0]]
    for seg in fades[1:]:
        prev = merged[-1]
        if seg.fade_type == prev.fade_type and (seg.start_s - prev.end_s) <= sample_interval_s:
            merged[-1] = FadeSegment(prev.fade_type, prev.start_s, seg.end_s)
        else:
            merged.append(seg)
    return merged


def iter_ffmpeg_sampled_frames(video_path: Path, width: int, height: int, sample_fps: float, gray: bool = False) -> Iterable[np.ndarray]:
    if gray:
        frame_bytes = width * height
        pix = "gray"
    else:
        frame_bytes = width * height * 3
        pix = "bgr24"

    vf = f"fps={sample_fps:.6f}"
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-vf", vf,
        "-f", "rawvideo",
        "-pix_fmt", pix,
        "-",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdout is not None
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            arr = np.frombuffer(raw, dtype=np.uint8)
            if gray:
                yield arr.reshape((height, width))
            else:
                yield arr.reshape((height, width, 3))
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


def collect_fade_brightness(video_path: Path, src_w: int, src_h: int, fade_sample_fps: float, fade_scale_width: int) -> Tuple[List[float], List[float], List[float]]:
    w = max(32, int(fade_scale_width))
    h = max(2, int(round(src_h * w / max(src_w, 1))))
    if h % 2 == 1:
        h += 1

    frame_bytes = w * h
    vf = f"fps={fade_sample_fps:.6f},scale={w}:{h},format=gray"
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-vf", vf,
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    times: List[float] = []
    brightness: List[float] = []
    contrast: List[float] = []
    idx = 0
    interval = 1.0 / max(fade_sample_fps, 1e-6)
    try:
        assert proc.stdout is not None
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8)
            brightness.append(float(frame.mean()))
            contrast.append(float(frame.std()))
            times.append(idx * interval)
            idx += 1
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        stderr_text = ""
        if proc.stderr is not None:
            stderr_text = proc.stderr.read().decode("utf-8", errors="ignore").strip()
            proc.stderr.close()
        rc = proc.wait()
        if rc != 0 and "Broken pipe" not in stderr_text:
            raise RuntimeError(f"ffmpeg fade stream failed for {video_path.name}: {stderr_text}")

    return times, brightness, contrast


def process_video(
    video_path: Path,
    score_threshold: float,
    sample_fps: float,
    subtitle_min_hit_frames: int,
    region_min_area_ratio: float,
    lang: str,
    use_textline_orientation: bool,
    fade_dark_threshold: float,
    fade_min_delta: float,
    fade_min_duration: float,
    fade_max_duration: float,
    fade_sample_fps: float,
    fade_scale_width: int,
) -> VideoResult:
    try:
        fps, total_frames, duration_s, width, height = ffprobe_video_meta(video_path)
    except Exception as exc:
        return VideoResult(video_path.name, str(video_path), 0.0, 0, 0.0, False, False, False, [], False, [], f"ffprobe_error:{exc}")

    if width <= 0 or height <= 0:
        return VideoResult(video_path.name, str(video_path), round(fps, 4), total_frames, round(duration_s, 3), False, False, False, [], False, [], "invalid_resolution")

    sample_interval_s = 1.0 / max(sample_fps, 1e-6)
    sample_times_s: List[float] = []
    ocr_hits: List[bool] = []
    region_hits: List[bool] = []
    first_error = ""

    sample_idx = 0
    try:
        for frame in iter_ffmpeg_sampled_frames(video_path, width, height, sample_fps=sample_fps, gray=False):
            try:
                o_hit, r_hit = subtitle_dual_channel_hit(
                    frame,
                    score_threshold=score_threshold,
                    region_min_area_ratio=region_min_area_ratio,
                    lang=lang,
                    use_textline_orientation=use_textline_orientation,
                )
            except Exception as exc:
                if not first_error:
                    first_error = f"ocr_error:{exc}"
                o_hit, r_hit = False, False

            sample_times_s.append(sample_idx * sample_interval_s)
            ocr_hits.append(o_hit)
            region_hits.append(r_hit)
            sample_idx += 1
    except Exception as exc:
        return VideoResult(
            video_path.name,
            str(video_path),
            round(fps, 4),
            total_frames,
            round(duration_s, 3),
            False,
            False,
            False,
            [],
            False,
            [],
            f"ffmpeg_pipe_error:{exc}",
        )

    # 连续帧约束（误检抑制）
    ocr_hits = keep_only_long_true_runs(ocr_hits, max(1, subtitle_min_hit_frames))
    region_hits = keep_only_long_true_runs(region_hits, max(1, subtitle_min_hit_frames))
    subtitle_hits = [a or b for a, b in zip(ocr_hits, region_hits)]

    subtitle_segments = detect_subtitle_segments(
        sample_times_s,
        subtitle_hits,
        sample_interval_s,
    )

    fade_times: List[float] = []
    fade_brightness: List[float] = []
    fade_contrast: List[float] = []
    try:
        fade_times, fade_brightness, fade_contrast = collect_fade_brightness(
            video_path, width, height, fade_sample_fps=fade_sample_fps, fade_scale_width=fade_scale_width
        )
    except Exception as exc:
        if not first_error:
            first_error = f"fade_error:{exc}"

    fade_segments = detect_fade_segments(
        sample_times_s=fade_times,
        sample_brightness=fade_brightness,
        sample_contrast=fade_contrast,
        sample_interval_s=1.0 / max(fade_sample_fps, 1e-6),
        dark_threshold=fade_dark_threshold,
        min_delta=fade_min_delta,
        min_duration=fade_min_duration,
        max_duration=fade_max_duration,
    )

    return VideoResult(
        video_name=video_path.name,
        video_path=str(video_path),
        fps=round(fps, 4),
        total_frames=total_frames,
        duration_s=round(duration_s, 3),
        has_subtitle=bool(subtitle_segments),
        has_subtitle_ocr=any(ocr_hits),
        has_subtitle_region=any(region_hits),
        subtitle_segments=subtitle_segments,
        has_fade=bool(fade_segments),
        fade_segments=fade_segments,
        error=first_error,
    )


def process_video_task(
    task: Tuple[Path, float, float, int, float, str, bool, float, float, float, float, float, int]
) -> Tuple[Path, VideoResult]:
    (
        vp,
        score_threshold,
        sample_fps,
        subtitle_min_hit_frames,
        region_min_area_ratio,
        lang,
        use_textline_orientation,
        fade_dark_threshold,
        fade_min_delta,
        fade_min_duration,
        fade_max_duration,
        fade_sample_fps,
        fade_scale_width,
    ) = task

    return vp, process_video(
        video_path=vp,
        score_threshold=score_threshold,
        sample_fps=sample_fps,
        subtitle_min_hit_frames=subtitle_min_hit_frames,
        region_min_area_ratio=region_min_area_ratio,
        lang=lang,
        use_textline_orientation=use_textline_orientation,
        fade_dark_threshold=fade_dark_threshold,
        fade_min_delta=fade_min_delta,
        fade_min_duration=fade_min_duration,
        fade_max_duration=fade_max_duration,
        fade_sample_fps=fade_sample_fps,
        fade_scale_width=fade_scale_width,
    )


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
            int(args.subtitle_min_hit_frames),
            float(args.region_min_area_ratio),
            str(args.lang),
            not bool(args.no_angle_cls),
            float(args.fade_dark_threshold),
            float(args.fade_min_delta),
            float(args.fade_min_duration),
            float(args.fade_max_duration),
            float(args.fade_sample_fps),
            int(args.fade_scale_width),
        )
        for vp in videos
    ]

    print(f"[INFO] total videos={len(videos)} workers={worker_count} sample_fps={sample_fps}")

    results_map: Dict[Path, VideoResult] = {}
    if worker_count == 1:
        for t in tasks:
            vp, r = process_video_task(t)
            results_map[vp] = r
            print(
                f"[DONE] {vp.name} sub={r.has_subtitle}(ocr={r.has_subtitle_ocr},region={r.has_subtitle_region}) "
                f"sub_segments={len(r.subtitle_segments)} fade={r.has_fade} fade_segments={len(r.fade_segments)} err={r.error}"
            )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(process_video_task, t): t[0] for t in tasks}
            for future in as_completed(future_map):
                vp = future_map[future]
                try:
                    _, r = future.result()
                except Exception as exc:
                    r = VideoResult(vp.name, str(vp), 0.0, 0, 0.0, False, False, False, [], False, [], f"worker_error:{exc}")
                results_map[vp] = r
                print(
                    f"[DONE] {vp.name} sub={r.has_subtitle}(ocr={r.has_subtitle_ocr},region={r.has_subtitle_region}) "
                    f"sub_segments={len(r.subtitle_segments)} fade={r.has_fade} fade_segments={len(r.fade_segments)} err={r.error}"
                )

    ordered: List[Dict[str, Any]] = []
    sub_cnt = 0
    sub_ocr_cnt = 0
    sub_region_cnt = 0
    fade_cnt = 0

    for vp in sorted(videos, key=lambda x: str(x)):
        r = results_map[vp]
        if r.has_subtitle:
            sub_cnt += 1
        if r.has_subtitle_ocr:
            sub_ocr_cnt += 1
        if r.has_subtitle_region:
            sub_region_cnt += 1
        if r.has_fade:
            fade_cnt += 1

        ordered.append(
            {
                "video_name": r.video_name,
                "video_path": r.video_path,
                "fps": r.fps,
                "total_frames": r.total_frames,
                "duration_s": r.duration_s,
                "has_subtitle": r.has_subtitle,
                "has_subtitle_ocr": r.has_subtitle_ocr,
                "has_subtitle_region": r.has_subtitle_region,
                "subtitle_segments": [
                    {
                        "start_s": s.start_s,
                        "end_s": s.end_s,
                        "start_hhmmss": format_hhmmss(s.start_s),
                        "end_hhmmss": format_hhmmss(s.end_s),
                    }
                    for s in r.subtitle_segments
                ],
                "has_fade": r.has_fade,
                "fade_segments": [
                    {
                        "fade_type": f.fade_type,
                        "start_s": f.start_s,
                        "end_s": f.end_s,
                        "start_hhmmss": format_hhmmss(f.start_s),
                        "end_hhmmss": format_hhmmss(f.end_s),
                    }
                    for f in r.fade_segments
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
            "subtitle_min_hit_frames": int(args.subtitle_min_hit_frames),
            "region_min_area_ratio": float(args.region_min_area_ratio),
            "subtitle_detector": "dual_channel(ocr_score + text_region)",
            "ocr": "PP-OCRv4 (PaddleOCR)",
            "decoder": "ffmpeg",
            "fade_dark_threshold": float(args.fade_dark_threshold),
            "fade_min_delta": float(args.fade_min_delta),
            "fade_min_duration": float(args.fade_min_duration),
            "fade_max_duration": float(args.fade_max_duration),
            "fade_sample_fps": float(args.fade_sample_fps),
            "fade_scale_width": int(args.fade_scale_width),
        },
        "summary": {
            "total_videos": len(videos),
            "has_subtitle_videos": sub_cnt,
            "has_subtitle_ocr_videos": sub_ocr_cnt,
            "has_subtitle_region_videos": sub_region_cnt,
            "no_subtitle_videos": len(videos) - sub_cnt,
            "has_fade_videos": fade_cnt,
        },
        "videos": ordered,
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
