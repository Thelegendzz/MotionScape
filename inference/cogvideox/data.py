#!/usr/bin/env python3
"""Shared MotionScape sample reader for CogVideoX1.5-5B-I2V inference.

This module provides the interface used by ``i2v.py``.
The constants and CLI defaults match the surviving CPython bytecode artifact.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


I2W_FRAME_NAME = "frame_000200.jpg"
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".png", ".jpeg", ".webp"}
MAX_HEIGHT = 768
MAX_WIDTH = 1360
SPATIAL_MULTIPLE = 16
DEFAULT_HEIGHT = 704
DEFAULT_WIDTH = 1280

DEFAULT_PROMPT = (
    "Given an observed first-person onboard-camera image and a future-frame description, generate a "
    "plausible future video. Preserve scene layout, appearance, viewpoint, and smooth "
    "camera motion.\n\nFuture description:\n{clip_text_annotation}"
)


def get_by_field_path(value: object, field_path: str, annotation_path: Path) -> object:
    if not field_path:
        return value
    for key in field_path.split("."):
        if isinstance(value, list):
            value = value[int(key)]
        elif isinstance(value, dict):
            value = value[key]
        else:
            raise KeyError(f"Cannot descend into {key!r} in {annotation_path}")
    return value


def read_clip_prompt(annotation_path: Path, field_path: str) -> str:
    value = get_by_field_path(json.loads(annotation_path.read_text()), field_path, annotation_path)
    if isinstance(value, dict):
        value = " ".join(
            str(value[key]).strip()
            for key in ("weather", "environment", "caption")
            if value.get(key)
        )
    elif isinstance(value, list):
        value = " ".join(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    if not text:
        raise ValueError(f"Empty caption field {field_path!r} in {annotation_path}")
    return text


def annotation_path_for(sample_dir: Path, input_root: Path, caption_root: Path) -> Path:
    relative = sample_dir.relative_to(input_root)
    candidates = (caption_root / relative.with_suffix(".json"), caption_root / f"{sample_dir.name}.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No caption JSON for {sample_dir}; tried: " + ", ".join(str(path) for path in candidates)
    )


def sample_name(sample_dir: Path, input_root: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_dir.relative_to(input_root).as_posix()).strip("_")
    return name or sample_dir.name


def iter_samples(input_root: Path, caption_root: Path):
    for sample_dir in sorted(path for path in input_root.rglob("*") if path.is_dir()):
        frame_path = sample_dir / I2W_FRAME_NAME
        if not frame_path.is_file():
            continue
        annotation_path = annotation_path_for(sample_dir, input_root, caption_root)
        yield sample_name(sample_dir, input_root), frame_path, annotation_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--caption-root", type=Path)
    parser.add_argument("--caption-field", default="")
    parser.add_argument("--prompt-template", type=Path)
    parser.add_argument("--num-frames", type=int, default=41)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.width % SPATIAL_MULTIPLE or args.height % SPATIAL_MULTIPLE:
        parser.error(f"--width and --height must be multiples of {SPATIAL_MULTIPLE}.")
    if args.width > MAX_WIDTH or args.height > MAX_HEIGHT:
        parser.error(f"Maximum supported dimensions are {MAX_WIDTH}x{MAX_HEIGHT}.")
    if args.num_frames < 5 or (args.num_frames - 1) % 4:
        parser.error("--num-frames must be at least 5 and have the form 4n+1.")
    if args.fps <= 0:
        parser.error("--fps must be positive.")
    return args
