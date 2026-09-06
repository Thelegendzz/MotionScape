"""Shared motion-scoring utilities used by the final MotionScape protocol."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SOURCE_FPS = 30000 / 1001
DEFAULT_TARGET_FPS = 9.99
DEFAULT_WIDTH = 854
DEFAULT_HEIGHT = 480


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def list_sample_dirs(frames_root: Path) -> list[Path]:
    if not frames_root.is_dir():
        raise SystemExit(f"frames root not found or not a directory: {frames_root}")
    return sorted(path for path in frames_root.iterdir() if path.is_dir())


def list_frames(sample_dir: Path) -> list[Path]:
    return sorted(path for path in sample_dir.iterdir() if is_image_file(path))


def require_runtime_dependencies() -> None:
    missing = []
    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError:
        missing.append("opencv-python-headless or opencv-python")
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        missing.append("numpy")
    if missing:
        raise SystemExit(
            "Missing runtime dependency: "
            + ", ".join(missing)
            + ". Please install the dependencies in the environment used to run this script."
        )


def build_sample_indices(total_frames: int, source_fps: float, target_fps: float) -> list[int]:
    if total_frames <= 0:
        return []
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    step = max(source_fps / target_fps, 1.0)
    indices: list[int] = []
    cursor = 0.0
    while cursor < total_frames:
        index = min(int(round(cursor)), total_frames - 1)
        if not indices or index != indices[-1]:
            indices.append(index)
        cursor += step
    return indices


def read_gray_resized(path: Path, width: int, height: int) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def load_manifest_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    index: dict[str, dict[str, Any]] = {}
    for item in data.get("items", []):
        if not isinstance(item, dict):
            continue
        sample_id = item.get("sample_id") or item.get("id")
        if isinstance(sample_id, str):
            index[sample_id] = item
    return index


def assign_buckets(items: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    import numpy as np

    scores = [
        float(item["frame_pair_score_p75_mean"])
        for item in items
        if item.get("status") == "ok" and math.isfinite(float(item.get("frame_pair_score_p75_mean", math.nan)))
    ]
    if not scores:
        return None, None
    q33 = float(np.percentile(scores, 33))
    q66 = float(np.percentile(scores, 66))
    for item in items:
        if item.get("status") != "ok":
            item["dynamicity_bucket"] = None
            item["motion_stratum"] = None
            continue
        score = float(item["frame_pair_score_p75_mean"])
        if score <= q33:
            bucket = "low"
        elif score <= q66:
            bucket = "medium"
        else:
            bucket = "high"
        item["dynamicity_bucket"] = bucket
        item["motion_stratum"] = bucket
    return q33, q66


def enrich_with_manifest(items: list[dict[str, Any]], manifest_index: dict[str, dict[str, Any]]) -> None:
    for item in items:
        manifest_item = manifest_index.get(item["sample_id"])
        if not manifest_item:
            continue
        for key in ("video", "video_index", "segment_index", "window_source", "start_s", "end_s", "outputs"):
            if key in manifest_item:
                item[key] = manifest_item[key]
