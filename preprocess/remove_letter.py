#!/usr/bin/env python3
"""
字幕 + 淡入淡出检测（不剪辑视频）：
1) ffmpeg 按指定采样率抽帧（默认 1fps）。
2) 每帧整图送入 PP-OCRv4（PaddleOCR）做字幕检测。
3) 检测淡入淡出并将结果与字幕一起写入同一个 JSON。

注意：
- 不修改原视频。
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
    fade_type: str  # fade_in / fade_out
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
    has_fade: bool
    fade_segments: List[FadeSegment]
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 PP-OCRv4 的字幕+淡入淡出检测（仅输出 JSON）。")
    parser.add_argument("--input-dir", type=Path, required=True, help="输入视频目录")
    parser.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path("subtitle_report.json"),
        help="输出 JSON 路径（默认当前目录 subtitle_report.json）",
    )
    parser.add_argument("--score-threshold", type=float, default=0.85, help="OCR 置信度阈值")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="采样率（fps），默认 1.0")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(11, max(1, (os.cpu_count() or 2) - 1)),
        help="并行进程数，默认 min(11, CPU核数-1)",
    )
    parser.add_argument("--lang", type=str, default="en", help="PaddleOCR 语言模型，如 ch/en")
    parser.add_argument("--no-angle-cls", action="store_true", help="关闭方向分类以加速")

    # Fade 参数
    parser.add_argument("--fade-dark-threshold", type=float, default=45.0, help="淡入淡出暗场阈值（0~255）")
    parser.add_argument("--fade-min-delta", type=float, default=150.0, help="淡入淡出最小亮度变化")
    parser.add_argument("--fade-min-duration", type=float, default=0.1, help="淡入淡出最短持续时长（秒）")
    parser.add_argument("--fade-max-duration", type=float, default=5.0, help="淡入淡出最长持续时长（秒）")
    parser.add_argument("--fade-sample-fps", type=float, default=10.0, help="淡入淡出检测采样率（fps）")
    parser.add_argument("--fade-scale-width", type=int, default=224, help="淡入淡出检测缩放宽度")
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

    # 原规则保留：两段字幕间隔等于 1 秒时合并
    if not segments:
        return segments
    merged: List[SubtitleSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg.start_s - prev.end_s
        if abs(gap - 1.0) <= 1e-3:
            merged[-1] = SubtitleSegment(start_s=prev.start_s, end_s=seg.end_s)
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
    t = t[:n]
    b = b[:n]
    c = c[:n]

    # 轻量平滑，减少抖动误检
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    b_smooth = np.convolve(b, kernel, mode="same")
    diff = np.diff(b_smooth)
    if len(diff) == 0:
        return []

    slope_thr = max(1.0, float(np.std(diff)) * 0.8)
    min_frames = max(1, int(round(min_duration / max(sample_interval_s, 1e-6))))
    max_frames = max(min_frames, int(round(max_duration / max(sample_interval_s, 1e-6))))

    fades: List[FadeSegment] = []

    # 全局对比度参考：真正淡入淡出靠近黑场时通常对比度会明显变低
    contrast_ref = float(np.percentile(c, 60))
    contrast_dark_limit = max(8.0, contrast_ref * 0.55)

    # 片头特判：在片头窗口内先找低谷，再找后续峰值，适配“不从第0帧开始”的淡入
    head_end = min(n - 1, int(round(max_frames * 1.5)))
    if head_end >= 2:
        head_curve = b_smooth[: head_end + 1]
        head_contrast = c[: head_end + 1]
        low_idx = int(np.argmin(head_curve))
        if low_idx < head_end:
            rise_curve = head_curve[low_idx:]
            high_rel = int(np.argmax(rise_curve))
            high_idx = low_idx + high_rel
            if high_idx > low_idx:
                head_delta = float(b_smooth[high_idx] - b_smooth[low_idx])
                head_dur = float(t[high_idx] - t[low_idx]) + sample_interval_s
                local_diff = diff[low_idx:high_idx]
                if len(local_diff) > 0:
                    up_ratio = float(np.mean(local_diff >= (-slope_thr * 0.2)))
                else:
                    up_ratio = 0.0

                if (
                    head_delta >= (min_delta * 0.45)
                    and head_dur >= (min_duration * 0.3)
                    and head_dur <= (max_duration * 1.2)
                    and up_ratio >= 0.65
                    and float(np.min(head_curve[: max(low_idx + 1, min_frames)])) <= (dark_threshold + 25.0)
                    and float(np.min(head_contrast[: max(low_idx + 1, min_frames)])) <= contrast_dark_limit
                ):
                    fades.append(
                        FadeSegment(
                            "fade_in",
                            round(float(t[low_idx]), 3),
                            round(float(t[high_idx] + sample_interval_s), 3),
                        )
                    )

    # 片尾：在片尾窗口内先找高峰，再找后续低谷，覆盖结尾淡出
    tail_start = max(0, n - int(round(max_frames * 1.5)) - 1)
    if tail_start < n - 2:
        tail_curve = b_smooth[tail_start:]
        tail_contrast = c[tail_start:]
        high_rel = int(np.argmax(tail_curve))
        high_idx = tail_start + high_rel
        if high_idx < n - 1:
            low_rel = int(np.argmin(b_smooth[high_idx:]))
            low_idx = high_idx + low_rel
            if low_idx > high_idx:
                tail_delta = float(b_smooth[high_idx] - b_smooth[low_idx])
                tail_dur = float(t[low_idx] - t[high_idx]) + sample_interval_s
                local_diff = diff[high_idx:low_idx]
                if len(local_diff) > 0:
                    down_ratio = float(np.mean(local_diff <= (slope_thr * 0.2)))
                else:
                    down_ratio = 0.0
                if (
                    tail_delta >= (min_delta * 0.45)
                    and tail_dur >= (min_duration * 0.3)
                    and tail_dur <= (max_duration * 1.2)
                    and down_ratio >= 0.65
                    and float(np.min(tail_curve[max(0, low_rel - min_frames + 1): low_rel + 1])) <= (dark_threshold + 25.0)
                    and float(np.min(tail_contrast[max(0, low_rel - min_frames + 1): low_rel + 1])) <= contrast_dark_limit
                ):
                    fades.append(
                        FadeSegment(
                            "fade_out",
                            round(float(t[high_idx]), 3),
                            round(float(t[low_idx] + sample_interval_s), 3),
                        )
                    )

    # 片尾二次兜底：若最后窗口整体持续下降且末端进入暗场，则视为淡出
    tail2_len = max(min_frames + 1, min(max_frames, n - 1))
    if tail2_len >= 2:
        s2 = n - tail2_len
        e2 = n - 1
        tail2_diff = diff[s2:e2]
        if len(tail2_diff) > 0:
            down_ratio2 = float(np.mean(tail2_diff <= (slope_thr * 0.15)))
            delta2 = float(b_smooth[s2] - b_smooth[e2])
            dur2 = float(t[e2] - t[s2]) + sample_interval_s
            tail2_dark = float(np.min(b_smooth[max(s2, e2 - min_frames + 1): e2 + 1]))
            tail2_contrast = float(np.min(c[max(s2, e2 - min_frames + 1): e2 + 1]))
            if (
                down_ratio2 >= 0.72
                and delta2 >= (min_delta * 0.35)
                and dur2 >= (min_duration * 0.25)
                and dur2 <= (max_duration * 1.2)
                and tail2_dark <= (dark_threshold + 28.0)
                and tail2_contrast <= (contrast_dark_limit * 1.15)
            ):
                fades.append(
                    FadeSegment(
                        "fade_out",
                        round(float(t[s2]), 3),
                        round(float(t[e2] + sample_interval_s), 3),
                    )
                )

    i = 0
    while i < len(diff):
        d0 = float(diff[i])
        if abs(d0) < slope_thr:
            i += 1
            continue

        sign = 1.0 if d0 > 0 else -1.0
        j = i
        strong = 0
        while j < len(diff) and (j - i + 1) <= max_frames:
            dj = float(diff[j])
            if dj * sign < -slope_thr:
                break
            if dj * sign >= slope_thr * 0.6:
                strong += 1
            j += 1

        end_idx = j  # 对应 b_smooth[end_idx]
        run_len = end_idx - i
        if run_len >= min_frames:
            delta = float(b_smooth[end_idx] - b_smooth[i])
            duration = float(t[end_idx] - t[i]) + sample_interval_s
            if sign > 0:
                # fade in：要求起点暗、整体亮度上升
                if (
                    b_smooth[i] <= dark_threshold
                    and c[i] <= contrast_dark_limit
                    and delta >= min_delta
                    and duration <= max_duration
                ):
                    fades.append(FadeSegment("fade_in", round(float(t[i]), 3), round(float(t[end_idx] + sample_interval_s), 3)))
            else:
                # fade out：要求终点暗、整体亮度下降
                if (
                    b_smooth[end_idx] <= dark_threshold
                    and c[end_idx] <= contrast_dark_limit
                    and (-delta) >= min_delta
                    and duration <= max_duration
                ):
                    fades.append(FadeSegment("fade_out", round(float(t[i]), 3), round(float(t[end_idx] + sample_interval_s), 3)))

        i = max(i + 1, end_idx)

    # 合并同类型且间隔很短的段，降低碎片化
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


def collect_fade_brightness(
    video_path: Path,
    src_w: int,
    src_h: int,
    fade_sample_fps: float,
    fade_scale_width: int,
) -> Tuple[List[float], List[float], List[float]]:
    w = max(32, int(fade_scale_width))
    h = max(2, int(round(src_h * w / max(src_w, 1))))
    if h % 2 == 1:
        h += 1

    frame_bytes = w * h
    vf = f"fps={fade_sample_fps:.6f},scale={w}:{h},format=gray"
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
        "gray",
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
    lang: str,
    use_textline_orientation: bool,
    fade_dark_threshold: float,
    fade_min_delta: float,
    fade_min_duration: float,
    fade_max_duration: float,
    fade_sample_fps: float,
    fade_scale_width: int,
) -> VideoSubtitleResult:
    try:
        fps, total_frames, duration_s, width, height = ffprobe_video_meta(video_path)
    except Exception as exc:
        return VideoSubtitleResult(video_path.name, str(video_path), 0.0, 0, 0.0, False, [], False, [], f"ffprobe_error:{exc}")

    if width <= 0 or height <= 0:
        return VideoSubtitleResult(
            video_path.name,
            str(video_path),
            round(fps, 4),
            total_frames,
            round(duration_s, 3),
            False,
            [],
            False,
            [],
            "invalid_resolution",
        )

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
            False,
            [],
            f"ffmpeg_pipe_error:{exc}",
        )

    subtitle_segments = detect_subtitle_segments(sample_times_s, sample_hits, sample_interval_s)
    fade_times: List[float] = []
    fade_brightness: List[float] = []
    fade_contrast: List[float] = []
    try:
        fade_times, fade_brightness, fade_contrast = collect_fade_brightness(
            video_path=video_path,
            src_w=width,
            src_h=height,
            fade_sample_fps=fade_sample_fps,
            fade_scale_width=fade_scale_width,
        )
    except Exception as exc:
        if not first_ocr_error:
            first_ocr_error = f"fade_error:{exc}"

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

    return VideoSubtitleResult(
        video_name=video_path.name,
        video_path=str(video_path),
        fps=round(fps, 4),
        total_frames=total_frames,
        duration_s=round(duration_s, 3),
        has_subtitle=bool(subtitle_segments),
        subtitle_segments=subtitle_segments,
        has_fade=bool(fade_segments),
        fade_segments=fade_segments,
        error=first_ocr_error,
    )


def process_video_task(
    task: Tuple[Path, float, float, str, bool, float, float, float, float, float, int]
) -> Tuple[Path, VideoSubtitleResult]:
    (
        vp,
        score_threshold,
        sample_fps,
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
        vp,
        score_threshold,
        sample_fps,
        lang,
        use_textline_orientation,
        fade_dark_threshold,
        fade_min_delta,
        fade_min_duration,
        fade_max_duration,
        fade_sample_fps,
        fade_scale_width,
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

    results_map: Dict[Path, VideoSubtitleResult] = {}
    if worker_count == 1:
        for t in tasks:
            vp, result = process_video_task(t)
            results_map[vp] = result
            print(
                f"[DONE] {vp.name} has_subtitle={result.has_subtitle} "
                f"sub_segments={len(result.subtitle_segments)} has_fade={result.has_fade} "
                f"fade_segments={len(result.fade_segments)} err={result.error}"
            )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(process_video_task, t): t[0] for t in tasks}
            for future in as_completed(future_map):
                vp = future_map[future]
                try:
                    _, result = future.result()
                except Exception as exc:
                    result = VideoSubtitleResult(vp.name, str(vp), 0.0, 0, 0.0, False, [], False, [], f"worker_error:{exc}")
                results_map[vp] = result
                print(
                    f"[DONE] {vp.name} has_subtitle={result.has_subtitle} "
                    f"sub_segments={len(result.subtitle_segments)} has_fade={result.has_fade} "
                    f"fade_segments={len(result.fade_segments)} err={result.error}"
                )

    ordered_results: List[Dict[str, Any]] = []
    has_subtitle_count = 0
    has_fade_count = 0
    for vp in sorted(videos, key=lambda x: str(x)):
        r = results_map[vp]
        if r.has_subtitle:
            has_subtitle_count += 1
        if r.has_fade:
            has_fade_count += 1

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
                "has_fade": r.has_fade,
                "fade_segments": [
                    {
                        "fade_type": seg.fade_type,
                        "start_s": seg.start_s,
                        "end_s": seg.end_s,
                        "start_hhmmss": format_hhmmss(seg.start_s),
                        "end_hhmmss": format_hhmmss(seg.end_s),
                    }
                    for seg in r.fade_segments
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
            "fade_dark_threshold": float(args.fade_dark_threshold),
            "fade_min_delta": float(args.fade_min_delta),
            "fade_min_duration": float(args.fade_min_duration),
            "fade_max_duration": float(args.fade_max_duration),
            "fade_sample_fps": float(args.fade_sample_fps),
            "fade_scale_width": int(args.fade_scale_width),
        },
        "summary": {
            "total_videos": len(videos),
            "has_subtitle_videos": has_subtitle_count,
            "no_subtitle_videos": len(videos) - has_subtitle_count,
            "has_fade_videos": has_fade_count,
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
