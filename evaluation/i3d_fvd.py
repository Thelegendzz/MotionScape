#!/usr/bin/env python3
"""Standard MotionScape I3D-based Fréchet Video Distance.

This module follows the Google Research FVD reference implementation:

* fixed TF-Hub model ``deepmind/i3d-kinetics-400/1``;
* one ``RGB/inception_i3d/Mean`` embedding per 75-frame I3D input;
* GT remains exactly 75 frames; GT and generated clips are sampled over their common real-time interval;
* RGB bilinear resize to 224 x 224 and scaling from [0, 255] to [-1, 1];
* one distribution-level Fréchet distance per motion bucket.

There is deliberately no R3D/S3D fallback. Model or dependency failures are
fatal so non-standard features cannot be mislabeled as FVD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


I3D_MODEL_ID = "deepmind/i3d-kinetics-400/1"
I3D_MODEL_VERSION = "1"
I3D_TFHUB_URL = f"https://tfhub.dev/{I3D_MODEL_ID}"
I3D_OFFICIAL_AGGREGATE_SHA256 = (
    "7ee40f093a7438cdaa48b97b6db358d40825dd5958ea057aa0a8dcc6c62b1b76"
)
I3D_BATCH_SIZE = 16
NUM_FRAMES = 75
INPUT_RESOLUTION = (224, 224)
PREPROCESSING_VERSION = "i3d_fvd_rgb_common_timeline_tf217_gather_bilinear_minus1_1_v7"
EXPECTED_BUCKET_COUNTS = {"low": 75, "medium": 75, "high": 78}
VALID_BUCKETS = tuple(EXPECTED_BUCKET_COUNTS)


@dataclass(frozen=True)
class VideoPair:
    sample_id: str
    motion_bucket: str
    gt_path: Path
    pred_path: Path
    gt_fps: float
    pred_fps: float
    pred_frame_count: int
    gt_frame_indices: tuple[int, ...]
    pred_frame_indices: tuple[int, ...]
    evaluation_duration_sec: float


@dataclass(frozen=True)
class FeatureItem:
    sample_id: str
    path: Path
    source_frame_count: int
    source_fps: float
    frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class ValidatedDataset:
    pairs: tuple[VideoPair, ...]
    by_bucket: dict[str, tuple[VideoPair, ...]]


def _require_tensorflow() -> tuple[Any, Any]:
    try:
        import tensorflow.compat.v1 as tf
        import tensorflow_hub as hub
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Official I3D-FVD requires TensorFlow and TensorFlow Hub. Install "
            "scripts/i3d_fvd_requirements.txt; no R3D or alternate-weight "
            "fallback is permitted."
        ) from exc
    if not hasattr(hub, "Module"):
        raise RuntimeError(
            "The installed tensorflow_hub does not provide the TF1 hub.Module "
            "API required by deepmind/i3d-kinetics-400/1. Install the pinned "
            "scripts/i3d_fvd_requirements.txt environment."
        )
    return tf, hub


def _format_ids(values: Iterable[str], limit: int = 20) -> str:
    items = sorted(values)
    shown = items[:limit]
    suffix = f" ... (+{len(items) - limit} more)" if len(items) > limit else ""
    return ", ".join(shown) + suffix


def load_motion_buckets(path: Path) -> dict[str, str]:
    """Load and strictly validate MotionScape's 75/75/78 bucket assignment."""
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"Failed to read dynamicity bucket JSON: {path}") from exc
    samples = data.get("samples") if isinstance(data, dict) else None
    if not isinstance(samples, list):
        raise ValueError(f"{path} must contain a top-level 'samples' list.")

    assignments: dict[str, str] = {}
    duplicates: set[str] = set()
    invalid_rows: list[str] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            invalid_rows.append(f"row {index}: not an object")
            continue
        sample_id = sample.get("sample_id")
        bucket = str(sample.get("motion_stratum") or "").strip().lower()
        if not isinstance(sample_id, str) or not sample_id:
            invalid_rows.append(f"row {index}: missing sample_id")
            continue
        if bucket not in VALID_BUCKETS:
            invalid_rows.append(
                f"row {index} ({sample_id}): invalid bucket {bucket!r}"
            )
            continue
        if sample_id in assignments:
            duplicates.add(sample_id)
        assignments[sample_id] = bucket

    counts = {
        bucket: sum(value == bucket for value in assignments.values())
        for bucket in VALID_BUCKETS
    }
    errors: list[str] = []
    if invalid_rows:
        errors.append("invalid bucket rows: " + "; ".join(invalid_rows[:20]))
    if duplicates:
        errors.append("duplicate bucket sample ids: " + _format_ids(duplicates))
    for bucket, expected in EXPECTED_BUCKET_COUNTS.items():
        if counts[bucket] != expected:
            errors.append(
                f"{bucket} count is {counts[bucket]}, expected {expected}"
            )
    if len(assignments) != sum(EXPECTED_BUCKET_COUNTS.values()):
        errors.append(
            f"total unique bucket samples is {len(assignments)}, expected 228"
        )
    if errors:
        raise ValueError("Invalid MotionScape bucket definition:\n- " + "\n- ".join(errors))
    return assignments


def _is_generated_video(path: Path) -> bool:
    return (
        path.suffix.lower() == ".mp4"
        and not path.stem.lower().endswith("input_clip")
    )


def index_video_directory(
    directory: Path, *, generated: bool
) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Index videos by exact filename stem and expose duplicate sample ids."""
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    grouped: dict[str, list[Path]] = {}
    for path in sorted(directory.rglob("*.mp4")):
        if generated and not _is_generated_video(path):
            continue
        grouped.setdefault(path.stem, []).append(path.resolve())
    duplicates = {
        sample_id: paths for sample_id, paths in grouped.items() if len(paths) > 1
    }
    index = {
        sample_id: paths[0]
        for sample_id, paths in grouped.items()
        if len(paths) == 1
    }
    return index, duplicates


def probe_video(path: Path) -> tuple[int, float]:
    """Decode-count a video and return its positive container FPS."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    if count <= 0:
        raise ValueError(f"Video contains no decodable frames: {path}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Video has invalid FPS {fps}: {path}")
    return count, fps


def paired_fps_aligned_frame_indices(
    gt_frame_count: int,
    gt_fps: float,
    pred_frame_count: int,
    pred_fps: float,
    target_frame_count: int = NUM_FRAMES,
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    """Sample both videos over their common real-time interval."""
    if gt_frame_count < 2 or pred_frame_count < 2:
        raise ValueError("GT and generated videos must each contain at least two frames.")
    for name, fps in (("GT", gt_fps), ("generated", pred_fps)):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"{name} FPS must be positive, got {fps}.")
    if target_frame_count <= 0:
        raise ValueError("Target frame count must be positive.")

    gt_last_time = (gt_frame_count - 1) / gt_fps
    pred_last_time = (pred_frame_count - 1) / pred_fps
    common_duration = min(gt_last_time, pred_last_time)
    target_times = np.linspace(
        0.0, common_duration, target_frame_count, dtype=np.float64
    )
    gt_indices = np.rint(target_times * gt_fps).astype(np.int64)
    pred_indices = np.rint(target_times * pred_fps).astype(np.int64)
    gt_indices = np.clip(gt_indices, 0, gt_frame_count - 1)
    pred_indices = np.clip(pred_indices, 0, pred_frame_count - 1)
    return (
        tuple(int(index) for index in gt_indices),
        tuple(int(index) for index in pred_indices),
        float(common_duration),
    )


def validate_motion_scape_dataset(
    gt_dir: Path,
    pred_dir: Path,
    dynamicity_buckets_json: Path,
) -> ValidatedDataset:
    """Fail on ID/bucket errors, bad GT length, or invalid generated video metadata."""
    assignments = load_motion_buckets(dynamicity_buckets_json)
    expected_ids = set(assignments)
    gt_index, duplicate_gt = index_video_directory(gt_dir, generated=False)
    pred_index, duplicate_pred = index_video_directory(pred_dir, generated=True)
    gt_ids = set(gt_index)
    pred_ids = set(pred_index)

    errors: list[str] = []
    if duplicate_gt:
        errors.append("duplicate GT ids: " + _format_ids(duplicate_gt))
    if duplicate_pred:
        errors.append("duplicate generated ids: " + _format_ids(duplicate_pred))
    missing_gt = expected_ids - gt_ids
    missing_pred = expected_ids - pred_ids
    unexpected_gt = gt_ids - expected_ids
    unexpected_pred = pred_ids - expected_ids
    if missing_gt:
        errors.append("missing GT: " + _format_ids(missing_gt))
    if missing_pred:
        errors.append("missing generated: " + _format_ids(missing_pred))
    if unexpected_gt:
        errors.append("GT ids absent from buckets: " + _format_ids(unexpected_gt))
    if unexpected_pred:
        errors.append(
            "generated ids absent from buckets: " + _format_ids(unexpected_pred)
        )

    common_ids = expected_ids & gt_ids & pred_ids
    video_metadata: dict[
        str, tuple[
            float, int, float, tuple[int, ...], tuple[int, ...], float
        ]
    ] = {}
    bad_gt_frames: list[str] = []
    bad_generated_videos: list[str] = []
    for sample_id in sorted(common_ids):
        gt_count, gt_fps = probe_video(gt_index[sample_id])
        pred_count, pred_fps = probe_video(pred_index[sample_id])
        if gt_count != NUM_FRAMES:
            bad_gt_frames.append(f"{sample_id}={gt_count}")
            continue
        try:
            gt_indices, pred_indices, duration = paired_fps_aligned_frame_indices(
                gt_count, gt_fps, pred_count, pred_fps, NUM_FRAMES
            )
        except ValueError as exc:
            bad_generated_videos.append(
                f"{sample_id}={pred_count} frames at {pred_fps:.6g} fps ({exc})"
            )
            continue
        video_metadata[sample_id] = (
            gt_fps, pred_count, pred_fps, gt_indices, pred_indices, duration
        )
    if bad_gt_frames:
        errors.append(
            f"GT videos not source video exactly {NUM_FRAMES} frames; sampled over paired common interval: "
            + _format_ids(bad_gt_frames)
        )
    if bad_generated_videos:
        errors.append(
            "invalid generated videos: " + _format_ids(bad_generated_videos)
        )
    if errors:
        raise ValueError(
            "MotionScape I3D-FVD validation failed:\n- " + "\n- ".join(errors)
        )

    pairs = tuple(
        VideoPair(
            sample_id=sample_id,
            motion_bucket=assignments[sample_id],
            gt_path=gt_index[sample_id],
            pred_path=pred_index[sample_id],
            gt_fps=video_metadata[sample_id][0],
            pred_fps=video_metadata[sample_id][2],
            pred_frame_count=video_metadata[sample_id][1],
            gt_frame_indices=video_metadata[sample_id][3],
            pred_frame_indices=video_metadata[sample_id][4],
            evaluation_duration_sec=video_metadata[sample_id][5],
        )
        for sample_id in sorted(expected_ids)
    )
    by_bucket = {
        bucket: tuple(pair for pair in pairs if pair.motion_bucket == bucket)
        for bucket in VALID_BUCKETS
    }
    for bucket, expected in EXPECTED_BUCKET_COUNTS.items():
        actual = len(by_bucket[bucket])
        if actual != expected:
            raise AssertionError(
                f"Validated {bucket} pair count is {actual}, expected {expected}"
            )
    return ValidatedDataset(pairs=pairs, by_bucket=by_bucket)


def decode_rgb_video(item: FeatureItem) -> np.ndarray:
    """Decode a video and select 75 FPS-aligned RGB frames."""
    cap = cv2.VideoCapture(str(item.path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {item.path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != item.source_frame_count:
        raise ValueError(
            f"{item.path} decoded {len(frames)} frames during extraction; "
            f"validation recorded {item.source_frame_count}."
        )
    if len(item.frame_indices) != NUM_FRAMES:
        raise ValueError(
            f"{item.path} has {len(item.frame_indices)} temporal indices; "
            f"expected {NUM_FRAMES}."
        )
    if min(item.frame_indices) < 0 or max(item.frame_indices) >= len(frames):
        raise ValueError(f"Temporal indices are out of range for {item.path}.")
    return np.stack([frames[index] for index in item.frame_indices], axis=0)


class TensorFlowBilinearPreprocessor:
    """GPU bilinear resize and [0,255] -> [-1,1] scaling.

    This is algebraically equivalent to TF1 resize_bilinear with
    align_corners=False, but uses stable gather/arithmetic GPU kernels instead
    of the ResizeBilinear CUDA kernel that faults on some Blackwell inputs.
    """

    def __init__(self) -> None:
        tf, _ = _require_tensorflow()
        self.tf = tf
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device("/GPU:0"):
            self.input = tf.placeholder(
                tf.uint8, shape=[None, None, None, 3], name="rgb_video"
            )
            source = tf.cast(self.input, tf.float32)
            source_height = tf.shape(source)[1]
            source_width = tf.shape(source)[2]

            target_y = (
                tf.cast(tf.range(INPUT_RESOLUTION[0]), tf.float32)
                * tf.cast(source_height, tf.float32)
                / float(INPUT_RESOLUTION[0])
            )
            y0 = tf.cast(tf.floor(target_y), tf.int32)
            y1 = tf.minimum(y0 + 1, source_height - 1)
            y_weight = tf.reshape(
                target_y - tf.cast(y0, tf.float32),
                [1, INPUT_RESOLUTION[0], 1, 1],
            )
            top = tf.gather(source, y0, axis=1)
            bottom = tf.gather(source, y1, axis=1)
            vertical = top + (bottom - top) * y_weight

            target_x = (
                tf.cast(tf.range(INPUT_RESOLUTION[1]), tf.float32)
                * tf.cast(source_width, tf.float32)
                / float(INPUT_RESOLUTION[1])
            )
            x0 = tf.cast(tf.floor(target_x), tf.int32)
            x1 = tf.minimum(x0 + 1, source_width - 1)
            x_weight = tf.reshape(
                target_x - tf.cast(x0, tf.float32),
                [1, 1, INPUT_RESOLUTION[1], 1],
            )
            left = tf.gather(vertical, x0, axis=2)
            right = tf.gather(vertical, x1, axis=2)
            resized = left + (right - left) * x_weight
            self.output = 2.0 * resized / 255.0 - 1.0
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=config)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if frames.ndim != 4 or frames.shape[0] != NUM_FRAMES:
            raise RuntimeError(f"Unexpected decoded video shape: {frames.shape}")
        source_pixels = int(frames.shape[1]) * int(frames.shape[2])
        max_pixels_per_run = 8_000_000
        frame_chunk = max(
            1, min(NUM_FRAMES, max_pixels_per_run // max(source_pixels, 1))
        )
        chunks = [
            self.session.run(
                self.output,
                feed_dict={self.input: frames[start : start + frame_chunk]},
            )
            for start in range(0, NUM_FRAMES, frame_chunk)
        ]
        output = np.concatenate(chunks, axis=0)
        if output.shape != (NUM_FRAMES, *INPUT_RESOLUTION, 3):
            raise RuntimeError(f"Unexpected preprocessed shape: {output.shape}")
        if not np.isfinite(output).all():
            raise RuntimeError("NaN/Inf detected after I3D preprocessing.")
        if output.min() < -1.001 or output.max() > 1.001:
            raise RuntimeError(
                f"I3D input range is [{output.min()}, {output.max()}], "
                "expected [-1,1]."
            )
        return output.astype(np.float32, copy=False)

    def close(self) -> None:
        self.session.close()


def resolve_i3d_model(i3d_model_path: Path | None) -> tuple[str, str]:
    """Resolve only the fixed official TF-Hub model; never select a fallback."""
    _, hub = _require_tensorflow()
    if i3d_model_path is not None:
        resolved = i3d_model_path.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"Local I3D TF-Hub model directory does not exist: {resolved}"
            )
        if not (resolved / "tfhub_module.pb").is_file():
            raise ValueError(
                f"{resolved} is not the expected TF-Hub v1 module: "
                "tfhub_module.pb is missing."
            )
        manifest = sha256_model_directory(resolved)
        if manifest["aggregate_sha256"] != I3D_OFFICIAL_AGGREGATE_SHA256:
            raise ValueError(
                f"Local I3D module hash is {manifest['aggregate_sha256']}, "
                f"expected official {I3D_OFFICIAL_AGGREGATE_SHA256} for "
                f"{I3D_MODEL_ID}. No alternate weights are permitted."
            )
        return str(resolved), "TF-Hub"
    try:
        resolved = hub.resolve(I3D_TFHUB_URL)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to resolve mandatory official model {I3D_TFHUB_URL}. "
            "Pass --i3d-model-path with a valid downloaded TF-Hub module. "
            "No alternate model fallback is permitted."
        ) from exc
    resolved_path = Path(str(resolved)).expanduser().resolve()
    if not resolved_path.is_dir():
        raise RuntimeError(
            f"TF-Hub resolved {I3D_TFHUB_URL} to invalid path {resolved_path}"
        )
    manifest = sha256_model_directory(resolved_path)
    if manifest["aggregate_sha256"] != I3D_OFFICIAL_AGGREGATE_SHA256:
        raise RuntimeError(
            f"Resolved I3D module hash is {manifest['aggregate_sha256']}, "
            f"expected official {I3D_OFFICIAL_AGGREGATE_SHA256} for "
            f"{I3D_MODEL_ID}. No alternate weights are permitted."
        )
    return str(resolved_path), "TF-Hub"


class OfficialI3DExtractor:
    """Exact fixed-batch TF1 graph used by Google Research's FVD reference."""

    def __init__(self, resolved_model_path: str) -> None:
        tf, hub = _require_tensorflow()
        self.tf = tf
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.videos = tf.placeholder(
                tf.float32,
                shape=[
                    I3D_BATCH_SIZE,
                    NUM_FRAMES,
                    INPUT_RESOLUTION[0],
                    INPUT_RESOLUTION[1],
                    3,
                ],
                name="videos",
            )
            checks = [
                tf.Assert(
                    tf.reduce_max(self.videos) <= 1.001,
                    ["I3D input maximum exceeds 1", self.videos],
                ),
                tf.Assert(
                    tf.reduce_min(self.videos) >= -1.001,
                    ["I3D input minimum is below -1", self.videos],
                ),
            ]
            with tf.control_dependencies(checks):
                checked_videos = tf.identity(self.videos)
            module_name = "fvd_kinetics-400_i3d_module"
            module = hub.Module(resolved_model_path, name=module_name)
            module(checked_videos)
            tensor_name = (
                f"{module_name}_apply_default/RGB/inception_i3d/Mean:0"
            )
            try:
                self.embedding = self.graph.get_tensor_by_name(tensor_name)
            except KeyError as exc:
                raise RuntimeError(
                    f"Official I3D Mean embedding tensor is missing: {tensor_name}. "
                    "The supplied model is not deepmind/i3d-kinetics-400/1."
                ) from exc
            self.init_op = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer(),
                tf.tables_initializer(),
            )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=config)
        try:
            self.session.run(self.init_op)
        except Exception as exc:
            self.session.close()
            raise RuntimeError(
                "Failed to initialize official I3D weights. No fallback is allowed."
            ) from exc

    def __call__(self, videos: np.ndarray) -> np.ndarray:
        expected = (
            I3D_BATCH_SIZE,
            NUM_FRAMES,
            INPUT_RESOLUTION[0],
            INPUT_RESOLUTION[1],
            3,
        )
        if videos.shape != expected:
            raise ValueError(f"I3D batch shape is {videos.shape}, expected {expected}")
        try:
            features = self.session.run(
                self.embedding, feed_dict={self.videos: videos}
            )
        except Exception as exc:
            raise RuntimeError(
                "Official I3D feature extraction failed. No fallback is allowed."
            ) from exc
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] != I3D_BATCH_SIZE:
            raise RuntimeError(f"Unexpected I3D embedding shape: {features.shape}")
        if not np.isfinite(features).all():
            raise RuntimeError("NaN/Inf detected in I3D embeddings.")
        return features

    def close(self) -> None:
        self.session.close()


def _cache_metadata(item: FeatureItem) -> dict[str, Any]:
    stat = item.path.stat()
    return {
        "feature_extractor": "I3D",
        "model_id": I3D_MODEL_ID,
        "model_version": I3D_MODEL_VERSION,
        "model_aggregate_sha256": I3D_OFFICIAL_AGGREGATE_SHA256,
        "sample_id": item.sample_id,
        "video_path": str(item.path.resolve()),
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "source_frame_count": item.source_frame_count,
        "source_fps": item.source_fps,
        "frame_indices": list(item.frame_indices),
        "num_frames": NUM_FRAMES,
        "input_resolution": list(INPUT_RESOLUTION),
        "preprocessing_version": PREPROCESSING_VERSION,
    }


def _cache_path(
    cache_dir: Path, kind: str, sample_id: str, metadata: dict[str, Any]
) -> Path:
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    safe_id = hashlib.sha256(sample_id.encode()).hexdigest()[:16]
    return cache_dir / f"{kind}_{safe_id}_{digest[:20]}.npz"


def _read_cached_feature(
    path: Path, expected_metadata: dict[str, Any]
) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            metadata = json.loads(str(cached["metadata"].item()))
            feature = np.asarray(cached["feature"], dtype=np.float64)
    except Exception as exc:
        raise RuntimeError(f"Invalid I3D feature cache: {path}") from exc
    if metadata != expected_metadata:
        return None
    if feature.ndim != 1 or not np.isfinite(feature).all():
        raise RuntimeError(f"Invalid cached I3D feature in {path}")
    return feature


def _write_cached_feature(
    path: Path, feature: np.ndarray, metadata: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        np.savez_compressed(
            temp_path,
            feature=np.asarray(feature, dtype=np.float64),
            metadata=json.dumps(metadata, sort_keys=True),
        )
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def extract_features(
    items: list[FeatureItem],
    *,
    kind: str,
    cache_dir: Path,
    resolved_model_path: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """Extract one feature per video in a fresh GPU process for every batch.

    NVIDIA TensorFlow 25.02 can leave a bad CUDA event behind after repeated
    TF1 I3D session runs on Blackwell. A short-lived process per batch ensures
    that the CUDA context is destroyed between runs without changing any I3D
    inputs, weights, or embeddings.
    """
    features: list[np.ndarray | None] = [None] * len(items)
    pending: list[tuple[int, FeatureItem, dict[str, Any], Path]] = []
    cache_hits = 0
    for index, item in enumerate(items):
        metadata = _cache_metadata(item)
        cache_path = _cache_path(cache_dir, kind, item.sample_id, metadata)
        cached = _read_cached_feature(cache_path, metadata)
        if cached is not None:
            features[index] = cached
            cache_hits += 1
        else:
            pending.append((index, item, metadata, cache_path))

    padding_count = 0
    for start in range(0, len(pending), I3D_BATCH_SIZE):
        batch_items = pending[start : start + I3D_BATCH_SIZE]
        valid_count = len(batch_items)
        if valid_count < I3D_BATCH_SIZE:
            padding_count += I3D_BATCH_SIZE - valid_count
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".i3d_{kind}_batch_", dir=cache_dir
        ) as temporary_directory:
            temporary_path = Path(temporary_directory)
            manifest_path = temporary_path / "manifest.json"
            output_path = temporary_path / "features.npy"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": item.sample_id,
                            "path": str(item.path),
                            "source_frame_count": item.source_frame_count,
                            "source_fps": item.source_fps,
                            "frame_indices": list(item.frame_indices),
                        }
                        for _, item, _, _ in batch_items
                    ],
                    ensure_ascii=False,
                )
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--i3d-batch-worker",
                "--batch-manifest",
                str(manifest_path),
                "--batch-output",
                str(output_path),
                "--resolved-i3d-model-path",
                resolved_model_path,
            ]
            subprocess.run(command, check=True)
            batch_features = np.asarray(
                np.load(output_path, allow_pickle=False), dtype=np.float64
            )
        if batch_features.ndim != 2 or batch_features.shape[0] != valid_count:
            raise RuntimeError(
                f"I3D worker returned shape {batch_features.shape}; "
                f"expected [{valid_count}, D]."
            )
        if not np.isfinite(batch_features).all():
            raise RuntimeError("NaN/Inf detected in I3D worker embeddings.")
        for item, feature in zip(batch_items, batch_features, strict=True):
            index, _, metadata, cache_path = item
            features[index] = feature
            _write_cached_feature(cache_path, feature, metadata)
        print(
            f"I3D {kind}: batch {start // I3D_BATCH_SIZE + 1}/"
            f"{math.ceil(len(pending) / I3D_BATCH_SIZE)}, "
            f"valid={valid_count}, padding={I3D_BATCH_SIZE - valid_count}",
            flush=True,
        )

    if any(feature is None for feature in features):
        raise AssertionError("Internal error: missing I3D features after extraction.")
    array = np.stack([feature for feature in features if feature is not None])
    if array.shape[0] != len(items):
        raise AssertionError(
            f"I3D feature count is {array.shape[0]}, expected {len(items)}"
        )
    return array, {
        "num_videos": len(items),
        "cache_hits": cache_hits,
        "num_extracted": len(pending),
        "padding_count": padding_count,
    }


def run_i3d_preprocess_worker(
    manifest_path: Path,
    item_index: int,
    output_path: Path,
) -> None:
    """Decode and resize one video in an isolated TensorFlow GPU process."""
    rows = json.loads(manifest_path.read_text())
    if not isinstance(rows, list) or not 0 <= item_index < len(rows):
        raise ValueError("Invalid I3D preprocess worker manifest or item index.")
    row = rows[item_index]
    item = FeatureItem(
        sample_id=str(row["sample_id"]),
        path=Path(row["path"]),
        source_frame_count=int(row["source_frame_count"]),
        source_fps=float(row["source_fps"]),
        frame_indices=tuple(int(index) for index in row["frame_indices"]),
    )
    preprocessor = TensorFlowBilinearPreprocessor()
    try:
        video = preprocessor(decode_rgb_video(item))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, video, allow_pickle=False)
    finally:
        preprocessor.close()


def run_i3d_batch_worker(
    manifest_path: Path,
    output_path: Path,
    resolved_model_path: str,
) -> None:
    """Preprocess isolated videos and embed one fixed I3D batch on GPU."""
    rows = json.loads(manifest_path.read_text())
    if not isinstance(rows, list) or not 1 <= len(rows) <= I3D_BATCH_SIZE:
        raise ValueError(
            f"I3D worker manifest must contain 1..{I3D_BATCH_SIZE} videos."
        )

    items = [
        FeatureItem(
            sample_id=str(row["sample_id"]),
            path=Path(row["path"]),
            source_frame_count=int(row["source_frame_count"]),
            source_fps=float(row["source_fps"]),
            frame_indices=tuple(int(index) for index in row["frame_indices"]),
        )
        for row in rows
    ]
    preprocessor = TensorFlowBilinearPreprocessor()
    try:
        videos = [preprocessor(decode_rgb_video(item)) for item in items]
    finally:
        preprocessor.close()

    valid_count = len(videos)
    videos.extend([videos[-1]] * (I3D_BATCH_SIZE - valid_count))
    extractor = OfficialI3DExtractor(resolved_model_path)
    try:
        features = extractor(np.stack(videos, axis=0))[:valid_count]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, features, allow_pickle=False)
    finally:
        extractor.close()


def frechet_distance_from_activations(
    real_activations: np.ndarray, generated_activations: np.ndarray
) -> float:
    """Compute the TF-GAN Fréchet formula between two distributions."""
    real = np.asarray(real_activations, dtype=np.float64)
    generated = np.asarray(generated_activations, dtype=np.float64)
    if real.ndim != 2 or generated.ndim != 2:
        raise ValueError("FVD activations must both have shape [N, D].")
    if real.shape[1] != generated.shape[1]:
        raise ValueError(
            f"FVD feature dimensions differ: {real.shape} vs {generated.shape}"
        )
    if real.shape[0] < 2 or generated.shape[0] < 2:
        raise ValueError("FVD requires at least two videos in each distribution.")
    if not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ValueError("NaN/Inf detected in FVD activations.")

    tf, _ = _require_tensorflow()
    graph = tf.Graph()
    frechet_device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    with graph.as_default(), tf.device(frechet_device):
        real_tensor = tf.constant(real, dtype=tf.float64)
        generated_tensor = tf.constant(generated, dtype=tf.float64)
        real_mean = tf.reduce_mean(real_tensor, axis=0)
        generated_mean = tf.reduce_mean(generated_tensor, axis=0)

        def unbiased_covariance(activations: Any, mean: Any) -> Any:
            centered = activations - mean
            denominator = tf.cast(tf.shape(activations)[0] - 1, tf.float64)
            return tf.matmul(centered, centered, transpose_a=True) / denominator

        def symmetric_matrix_square_root(matrix: Any) -> Any:
            singular_values, left, right = tf.linalg.svd(matrix)
            rooted = tf.where(
                singular_values < 1e-10,
                singular_values,
                tf.sqrt(singular_values),
            )
            return tf.matmul(
                tf.matmul(left, tf.linalg.diag(rooted)),
                right,
                transpose_b=True,
            )

        real_covariance = unbiased_covariance(real_tensor, real_mean)
        generated_covariance = unbiased_covariance(
            generated_tensor, generated_mean
        )
        sqrt_real_covariance = symmetric_matrix_square_root(real_covariance)
        covariance_product_root = symmetric_matrix_square_root(
            tf.matmul(
                sqrt_real_covariance,
                tf.matmul(generated_covariance, sqrt_real_covariance),
            )
        )
        covariance_distance = (
            tf.linalg.trace(real_covariance + generated_covariance)
            - 2.0 * tf.linalg.trace(covariance_product_root)
        )
        mean_distance = tf.reduce_sum(
            tf.math.squared_difference(real_mean, generated_mean)
        )
        distance_tensor = covariance_distance + mean_distance

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as session:
        value = float(session.run(distance_tensor))
    if not np.isfinite(value):
        raise RuntimeError(
            f"TensorFlow returned non-finite FVD {value}; "
            f"real_shape={real.shape}, generated_shape={generated.shape}"
        )
    if value < -1e-6:
        raise RuntimeError(f"TensorFlow returned invalid negative FVD: {value}")
    return max(value, 0.0)


def _value_histogram(values: Iterable[int | float]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for value in values:
        key = str(value) if isinstance(value, int) else f"{value:.12g}"
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def _model_metadata(
    model_source: str, resolved_model_path: str
) -> dict[str, Any]:
    return {
        "metric": "FVD",
        "feature_extractor": "I3D",
        "model_id": I3D_MODEL_ID,
        "model_version": I3D_MODEL_VERSION,
        "model_aggregate_sha256": I3D_OFFICIAL_AGGREGATE_SHA256,
        "model_source": model_source,
        "resolved_model_path": resolved_model_path,
        "num_frames": NUM_FRAMES,
        "input_resolution": list(INPUT_RESOLUTION),
        "input_range": [-1.0, 1.0],
        "preprocessing": "common_timeline_nearest_neighbor_then_bilinear_resize_and_scale_to_minus1_1",
        "temporal_alignment": "gt_and_generated_nearest_neighbor_over_common_real_time_interval",
        "generated_frame_policy": "variable frame count and FPS; at least 2 decodable frames",
        "gt_frame_policy": f"source video exactly {NUM_FRAMES} frames; sampled over paired common interval",
        "preprocessing_version": PREPROCESSING_VERSION,
    }


def sha256_model_directory(model_path: Path) -> dict[str, Any]:
    """Hash every local TF-Hub model file plus a deterministic aggregate."""
    files = sorted(path for path in model_path.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"No files found in local I3D model directory: {model_path}")
    rows: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in files:
        relative = path.relative_to(model_path).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hex_digest = digest.hexdigest()
        rows.append(
            {"path": relative, "size": path.stat().st_size, "sha256": hex_digest}
        )
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(hex_digest.encode())
        aggregate.update(b"\n")
    return {
        "model_id": I3D_MODEL_ID,
        "model_version": I3D_MODEL_VERSION,
        "model_aggregate_sha256": I3D_OFFICIAL_AGGREGATE_SHA256,
        "resolved_model_path": str(model_path.resolve()),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": rows,
    }


def write_sha256_manifest(model_path: Path, output_path: Path) -> dict[str, Any]:
    manifest = sha256_model_directory(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def compute_motion_bucket_fvd(
    dataset: ValidatedDataset,
    *,
    i3d_model_path: Path | None,
    cache_dir: Path,
    mode: str,
    model_name: str,
    condition_setting: str | None = None,
    include_overall: bool = False,
    sha256_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Extract all videos once and calculate one FVD per full bucket."""
    resolved_model_path, model_source = resolve_i3d_model(i3d_model_path)
    resolved_path = Path(resolved_model_path)
    manifest = None
    if i3d_model_path is not None:
        if sha256_manifest_path is None:
            raise ValueError(
                "Local I3D weights require --i3d-sha256-manifest so their "
                "contents are recorded."
            )
        manifest = write_sha256_manifest(resolved_path, sha256_manifest_path)

    gt_items = [
        FeatureItem(
            sample_id=pair.sample_id,
            path=pair.gt_path,
            source_frame_count=NUM_FRAMES,
            source_fps=pair.gt_fps,
            frame_indices=pair.gt_frame_indices,
        )
        for pair in dataset.pairs
    ]
    pred_items = [
        FeatureItem(
            sample_id=pair.sample_id,
            path=pair.pred_path,
            source_frame_count=pair.pred_frame_count,
            source_fps=pair.pred_fps,
            frame_indices=pair.pred_frame_indices,
        )
        for pair in dataset.pairs
    ]
    gt_features, gt_stats = extract_features(
        gt_items,
        kind="gt",
        cache_dir=cache_dir,
        resolved_model_path=resolved_model_path,
    )
    pred_features, pred_stats = extract_features(
        pred_items,
        kind="generated",
        cache_dir=cache_dir,
        resolved_model_path=resolved_model_path,
    )

    pair_index = {pair.sample_id: index for index, pair in enumerate(dataset.pairs)}
    metadata = _model_metadata(model_source, resolved_model_path)
    bucket_results: dict[str, dict[str, Any]] = {}
    for bucket in VALID_BUCKETS:
        bucket_pairs = dataset.by_bucket[bucket]
        indices = [pair_index[pair.sample_id] for pair in bucket_pairs]
        real = gt_features[indices]
        generated = pred_features[indices]
        print(
            f"FVD | model={model_name} | mode={mode} | bucket={bucket} | "
            f"GT={real.shape[0]} | generated={generated.shape[0]} | "
            f"I3D={model_source} ({resolved_model_path}) | "
            f"embeddings={real.shape}/{generated.shape}",
            flush=True,
        )
        fvd = frechet_distance_from_activations(real, generated)
        result = {
            "mode": mode,
            "model": model_name,
            "condition_setting": condition_setting,
            "motion_bucket": bucket,
            "num_real_videos": int(real.shape[0]),
            "num_generated_videos": int(generated.shape[0]),
            "fvd": fvd,
            "feature_extractor": "I3D",
            "model_id": I3D_MODEL_ID,
            "model_version": I3D_MODEL_VERSION,
            "model_aggregate_sha256": I3D_OFFICIAL_AGGREGATE_SHA256,
            "model_source": model_source,
            "resolved_model_path": resolved_model_path,
            "num_frames_per_video": NUM_FRAMES,
            "input_resolution": list(INPUT_RESOLUTION),
            "input_range": [-1.0, 1.0],
            "preprocessing": metadata["preprocessing"],
            "temporal_alignment": metadata["temporal_alignment"],
            "generated_source_frame_count_histogram": _value_histogram(
                pair.pred_frame_count for pair in bucket_pairs
            ),
            "generated_source_fps_histogram": _value_histogram(
                pair.pred_fps for pair in bucket_pairs
            ),
            "gt_fps_histogram": _value_histogram(
                pair.gt_fps for pair in bucket_pairs
            ),
            "evaluation_duration_sec_histogram": _value_histogram(
                pair.evaluation_duration_sec for pair in bucket_pairs
            ),
        }
        bucket_results[bucket] = result
        print(f"FVD | bucket={bucket} | value={fvd:.10f}", flush=True)

    overall_result = None
    if include_overall:
        overall_fvd = frechet_distance_from_activations(
            gt_features, pred_features
        )
        overall_result = {
            "mode": mode,
            "model": model_name,
            "condition_setting": condition_setting,
            "motion_bucket": "overall",
            "num_real_videos": int(gt_features.shape[0]),
            "num_generated_videos": int(pred_features.shape[0]),
            "fvd": overall_fvd,
            **{
                key: metadata[key]
                for key in (
                    "feature_extractor",
                    "model_id",
                    "model_version",
                    "model_aggregate_sha256",
                    "model_source",
                    "resolved_model_path",
                    "input_resolution",
                    "input_range",
                    "preprocessing",
                    "temporal_alignment",
                    "generated_frame_policy",
                    "gt_frame_policy",
                )
            },
            "num_frames_per_video": NUM_FRAMES,
        }

    return {
        **metadata,
        "mode": mode,
        "model": model_name,
        "condition_setting": condition_setting,
        "bucket_results": bucket_results,
        "overall": overall_result,
        "source_video_summary": {
            "generated_frame_count_histogram": _value_histogram(
                pair.pred_frame_count for pair in dataset.pairs
            ),
            "generated_fps_histogram": _value_histogram(
                pair.pred_fps for pair in dataset.pairs
            ),
            "gt_fps_histogram": _value_histogram(
                pair.gt_fps for pair in dataset.pairs
            ),
            "evaluation_duration_sec_histogram": _value_histogram(
                pair.evaluation_duration_sec for pair in dataset.pairs
            ),
        },
        "embedding_shape": {
            "gt": list(gt_features.shape),
            "generated": list(pred_features.shape),
        },
        "cache": {
            "directory": str(cache_dir.resolve()),
            "gt": gt_stats,
            "generated": pred_stats,
        },
        "model_sha256_manifest": (
            {
                "path": str(sha256_manifest_path.resolve()),
                "aggregate_sha256": manifest["aggregate_sha256"],
            }
            if manifest is not None and sha256_manifest_path is not None
            else None
        ),
    }


def run_fvd(
    *,
    gt_dir: Path,
    pred_dir: Path,
    dynamicity_buckets_json: Path,
    i3d_model_path: Path | None,
    cache_dir: Path,
    mode: str,
    model_name: str,
    condition_setting: str | None = None,
    include_overall: bool = False,
    sha256_manifest_path: Path | None = None,
) -> dict[str, Any]:
    dataset = validate_motion_scape_dataset(
        gt_dir, pred_dir, dynamicity_buckets_json
    )
    return compute_motion_bucket_fvd(
        dataset,
        i3d_model_path=i3d_model_path,
        cache_dir=cache_dir,
        mode=mode,
        model_name=model_name,
        condition_setting=condition_setting,
        include_overall=include_overall,
        sha256_manifest_path=sha256_manifest_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--dynamicity-buckets-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--i3d-model-path", type=Path)
    parser.add_argument(
        "--i3d-cache-dir",
        type=Path,
        default=Path(".cache/i3d_fvd"),
    )
    parser.add_argument(
        "--i3d-sha256-manifest",
        type=Path,
        help="Required when --i3d-model-path is used.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("Text2World", "Image2World", "Video2World"),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--condition-setting")
    parser.add_argument("--overall", action="store_true")
    args = parser.parse_args()
    if args.i3d_model_path is not None and args.i3d_sha256_manifest is None:
        parser.error(
            "--i3d-sha256-manifest is required with --i3d-model-path."
        )
    return args


def main() -> None:
    if "--i3d-preprocess-worker" in sys.argv:
        preprocess_parser = argparse.ArgumentParser(
            description="Internal isolated I3D GPU preprocess worker."
        )
        preprocess_parser.add_argument(
            "--i3d-preprocess-worker", action="store_true"
        )
        preprocess_parser.add_argument(
            "--batch-manifest", type=Path, required=True
        )
        preprocess_parser.add_argument("--item-index", type=int, required=True)
        preprocess_parser.add_argument("--batch-output", type=Path, required=True)
        preprocess_args = preprocess_parser.parse_args()
        run_i3d_preprocess_worker(
            preprocess_args.batch_manifest,
            preprocess_args.item_index,
            preprocess_args.batch_output,
        )
        return
    if "--i3d-batch-worker" in sys.argv:
        worker_parser = argparse.ArgumentParser(
            description="Internal isolated I3D GPU batch worker."
        )
        worker_parser.add_argument("--i3d-batch-worker", action="store_true")
        worker_parser.add_argument("--batch-manifest", type=Path, required=True)
        worker_parser.add_argument("--batch-output", type=Path, required=True)
        worker_parser.add_argument(
            "--resolved-i3d-model-path", required=True
        )
        worker_args = worker_parser.parse_args()
        run_i3d_batch_worker(
            worker_args.batch_manifest,
            worker_args.batch_output,
            worker_args.resolved_i3d_model_path,
        )
        return
    args = parse_args()
    result = run_fvd(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        dynamicity_buckets_json=args.dynamicity_buckets_json,
        i3d_model_path=args.i3d_model_path,
        cache_dir=args.i3d_cache_dir,
        mode=args.mode,
        model_name=args.model,
        condition_setting=args.condition_setting,
        include_overall=args.overall,
        sha256_manifest_path=args.i3d_sha256_manifest,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
