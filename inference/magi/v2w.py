#!/usr/bin/env python3
"""MotionScape video-to-world inference with MAGI-1-24B.

Launch this program with ``torchrun``.  All ranks collectively serve one MAGI
model-parallel replica and process the pending MotionScape samples in a stable
order.  Final MP4 files are checkpointed atomically and are never overwritten.

The prompt construction is intentionally identical to the LongCat
``v2w.py`` entry point.  The observed prefix uses the latest 32
frames sampled at stride two and ends at ``frame_000200.jpg``.  MAGI returns
only future chunks, so the saved files contain 40 generated frames at 16 FPS
and no conditioning frames.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


CONDITION_END_FRAME_NAME = "frame_000200.jpg"

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "v2w_4gpu.json"
DEFAULT_SOURCE_FPS = 30000 / 1001
DEFAULT_PROMPT = (
    "Given observed first-person onboard-camera video frames and a future-frame description, "
    "generate a plausible future continuation. Preserve scene layout, appearance, "
    "and viewpoint, and continue the observed camera motion smoothly.\n\n"
    "Future description:\n"
    "{clip_text_annotation}"
)


def read_caption(annotation_path: Path, field: str) -> str:
    """Read a dotted JSON field using the same rules as LongCat V2W."""
    value: object = json.loads(annotation_path.read_text(encoding="utf-8"))
    for key in field.split(".") if field else ():
        if isinstance(value, list):
            value = value[int(key)]
        elif isinstance(value, dict):
            value = value[key]
        else:
            raise KeyError(f"Cannot descend into {key!r} in {annotation_path}")
    if isinstance(value, dict):
        value = " ".join(
            str(value[key]).strip()
            for key in ("weather", "environment", "caption")
            if value.get(key)
        )
    elif isinstance(value, list):
        value = " ".join(str(item).strip() for item in value if str(item).strip())
    caption = str(value).strip()
    if not caption:
        raise ValueError(f"Empty caption in {annotation_path} at {field}")
    return caption


def annotation_for_frame(
    frame_path: Path, input_root: Path, caption_root: Path
) -> Path:
    relative_parent = frame_path.parent.relative_to(input_root)
    candidates = (
        caption_root / relative_parent.with_suffix(".json"),
        caption_root / f"{frame_path.parent.name}.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No caption JSON for {frame_path}; tried: "
        + ", ".join(str(path) for path in candidates)
    )


def condition_frame_paths(
    end_frame_path: Path, num_cond_frames: int, frame_stride: int
) -> tuple[Path, ...]:
    match = re.fullmatch(r"(.*?)(\d+)(\.[^.]+)", end_frame_path.name)
    if match is None:
        raise ValueError(f"Cannot parse frame index from {end_frame_path}")
    prefix, index_text, suffix = match.groups()
    end_index = int(index_text)
    start_index = end_index - frame_stride * (num_cond_frames - 1)
    if start_index < 1:
        raise ValueError(
            f"Not enough source frames before {end_frame_path} for "
            f"{num_cond_frames} condition frames at stride {frame_stride}"
        )
    paths = tuple(
        end_frame_path.with_name(f"{prefix}{index:0{len(index_text)}d}{suffix}")
        for index in range(start_index, end_index + 1, frame_stride)
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} condition frame(s) for {end_frame_path}: "
            + ", ".join(str(path) for path in missing[:5])
        )
    return paths


def iter_samples(
    input_root: Path,
    caption_root: Path,
    num_cond_frames: int,
    frame_stride: int,
):
    for end_frame_path in sorted(input_root.rglob(CONDITION_END_FRAME_NAME)):
        relative_parent = end_frame_path.parent.relative_to(input_root)
        yield (
            "__".join(relative_parent.parts),
            condition_frame_paths(end_frame_path, num_cond_frames, frame_stride),
            annotation_for_frame(end_frame_path, input_root, caption_root),
        )


def build_prompt(annotation_path: Path, caption_field: str, template: str) -> str:
    if "{clip_text_annotation}" not in template:
        raise ValueError("Prompt template must include {clip_text_annotation}.")
    return re.sub(
        r"\s+",
        " ",
        template.replace(
            "{clip_text_annotation}",
            read_caption(annotation_path, caption_field),
        ),
    ).strip()


def create_prefix_video(
    frame_paths: tuple[Path, ...], output_path: Path, fps: int, width: int, height: int
) -> None:
    """Encode exact ordered JPEG frames without writing an intermediate frame cache."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-vf",
        f"scale={width}:{height}",
        "-frames:v",
        str(len(frame_paths)),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame_path in frame_paths:
            with frame_path.open("rb") as source:
                shutil.copyfileobj(source, process.stdin, length=1024 * 1024)
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if return_code:
        raise RuntimeError(
            f"ffmpeg failed for {output_path}: {stderr.decode(errors='replace')}"
        )


def probe_frame_count(video_path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--caption-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--magi-root", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help=(
            "MAGI checkpoint root. Relative runtime_config checkpoint paths in "
            "--config-file are resolved against this directory."
        ),
    )
    parser.add_argument(
        "--caption-field",
        default="",
        help=(
            "Dot-separated annotation field. The empty default reads the flat release "
            "object and joins weather, environment, and caption in that order."
        ),
    )
    parser.add_argument("--prompt-template", type=Path)
    parser.add_argument("--source-fps", type=float, default=DEFAULT_SOURCE_FPS)
    parser.add_argument("--num-cond-frames", type=int, default=32)
    parser.add_argument("--frame-stride", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.num_cond_frames <= 0:
        parser.error("--num-cond-frames must be positive.")
    if args.source_fps <= 0:
        parser.error("--source-fps must be positive.")
    if args.frame_stride is not None and args.frame_stride <= 0:
        parser.error("--frame-stride must be positive.")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive.")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards).")
    return args


def rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.caption_root = args.caption_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.config_file = args.config_file.expanduser().resolve()
    args.magi_root = args.magi_root.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()

    for directory, label in (
        (args.input_root, "input root"),
        (args.caption_root, "caption root"),
        (args.magi_root, "MAGI source root"),
        (args.model_path, "MAGI checkpoint root"),
    ):
        if not directory.is_dir():
            raise NotADirectoryError(f"Missing {label}: {directory}")
    if not args.config_file.is_file():
        raise FileNotFoundError(f"Missing MAGI config: {args.config_file}")

    raw_config = json.loads(args.config_file.read_text(encoding="utf-8"))
    runtime = raw_config["runtime_config"]
    engine = raw_config["engine_config"]
    fps = int(runtime["fps"])
    width = int(runtime["video_size_w"])
    height = int(runtime["video_size_h"])
    expected_frames = int(runtime["num_frames"])
    expected_world_size = int(engine["pp_size"]) * int(engine["cp_size"])
    if expected_world_size <= 0:
        raise ValueError(
            f"MAGI config has invalid world size {expected_world_size}."
        )
    for key in ("load", "t5_pretrained", "vae_pretrained"):
        configured_path = Path(runtime[key]).expanduser()
        path = (
            configured_path
            if configured_path.is_absolute()
            else args.model_path / configured_path
        ).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Missing MAGI runtime_config.{key}: {path}")
        runtime[key] = str(path)

    args.frame_stride = args.frame_stride or max(1, round(args.source_fps / fps))
    prompt_template = (
        args.prompt_template.read_text(encoding="utf-8")
        if args.prompt_template
        else DEFAULT_PROMPT
    ).strip()
    samples = list(
        iter_samples(
            args.input_root,
            args.caption_root,
            args.num_cond_frames,
            args.frame_stride,
        )
    )
    if len(samples) != 228:
        raise ValueError(f"Expected 228 MotionScape samples, found {len(samples)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed: list[str] = []
    pending = []
    selected_samples = [
        (sample_index, sample)
        for sample_index, sample in enumerate(samples)
        if sample_index % args.num_shards == args.shard_index
    ]
    for sample_index, sample in selected_samples:
        name = sample[0]
        output_path = args.output_dir / f"{name}.mp4"
        if output_path.exists():
            if output_path.stat().st_size < 100 * 1024:
                raise RuntimeError(
                    f"Refusing to overwrite suspicious existing output: {output_path}"
                )
            completed.append(name)
        else:
            pending.append((sample_index, sample))
    if args.max_samples is not None:
        pending = pending[: args.max_samples]

    first_prompt = build_prompt(
        selected_samples[0][1][2], args.caption_field, prompt_template
    )
    if rank_zero():
        print(
            "MAGI V2W specification: "
            f"samples={len(samples)}, selected={len(selected_samples)}, "
            f"shard={args.shard_index}/{args.num_shards}, "
            f"completed={len(completed)}, pending={len(pending)}, "
            f"condition_frames={args.num_cond_frames}, frame_stride={args.frame_stride}, "
            f"output_frames={expected_frames}, fps={fps}, resolution={width}x{height}, "
            f"pp_size={engine['pp_size']}, cp_size={engine['cp_size']}.",
            flush=True,
        )
        print(f"First prompt: {first_prompt}", flush=True)
    if args.validate_only or not pending:
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != expected_world_size:
        raise RuntimeError(
            f"Launch with torchrun --nproc_per_node={expected_world_size}; "
            f"WORLD_SIZE is {world_size}."
        )

    sys.path.insert(0, str(args.magi_root))
    os.chdir(args.magi_root)

    import torch
    import torch.distributed as dist

    from inference.common import MagiConfig, print_rank_0, set_random_seed
    from inference.infra.distributed import dist_init
    from inference.model.dit import get_dit
    from inference.pipeline.prompt_process import get_txt_embeddings
    from inference.pipeline.video_generate import (
        SampleTransport,
        extract_feature_for_inference,
    )
    from inference.pipeline.video_process import (
        post_chunk_process,
        process_prefix_video,
        save_video_to_disk,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json"
    ) as resolved_config_file:
        json.dump(raw_config, resolved_config_file)
        resolved_config_file.flush()
        config = MagiConfig.from_json(resolved_config_file.name)
    set_random_seed(args.seed)
    dist_init(config)
    rank = dist.get_rank()
    temporary_root = Path(
        os.environ.get(
            "MAGI_PREFIX_TMPDIR",
            str(args.output_dir / ".magi_prefix_tmp"),
        )
    )
    if rank == 0:
        temporary_root.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    model = None
    total_batches = (len(pending) + args.batch_size - 1) // args.batch_size
    for batch_index, batch_start in enumerate(
        range(0, len(pending), args.batch_size), start=1
    ):
        batch = pending[batch_start : batch_start + args.batch_size]
        batch_names = [sample[0] for _, sample in batch]
        prefix_paths = [temporary_root / f"{name}.prefix.mp4" for name in batch_names]

        prefix_error = [None]
        if rank == 0:
            try:
                for (_, sample), prefix_path in zip(batch, prefix_paths):
                    name, condition_paths, _ = sample
                    if prefix_path.exists():
                        prefix_path.unlink()
                    create_prefix_video(condition_paths, prefix_path, fps, width, height)
                    if probe_frame_count(prefix_path) != args.num_cond_frames:
                        raise RuntimeError(
                            f"Condition video frame mismatch for {name}: "
                            f"expected {args.num_cond_frames}"
                        )
            except BaseException as exc:
                prefix_error[0] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(prefix_error, src=0)
        if prefix_error[0] is not None:
            raise RuntimeError(f"Failed to create prefix for {name}: {prefix_error[0]}")
        dist.barrier()

        print_rank_0(
            f"[batch {batch_index}/{total_batches}] Generating "
            f"{len(batch)} input(s): {', '.join(batch_names)}"
        )
        prefix_videos = [
            process_prefix_video(str(prefix_path), config)
            for prefix_path in prefix_paths
        ]
        if model is None:
            model = get_dit(config)
            print_rank_0("MAGI model loaded once for the full pending sample set.")
        text_features = [
            get_txt_embeddings(
                build_prompt(sample[2], args.caption_field, prompt_template), config
            )
            for _, sample in batch
        ]
        transport_inputs = [
            extract_feature_for_inference(model, prefix_video, caption_embs, emb_masks)
            for prefix_video, (caption_embs, emb_masks) in zip(
                prefix_videos, text_features
            )
        ]
        set_random_seed(args.seed + batch[0][0])
        sample_transport = SampleTransport(
            model=model,
            transport_inputs=transport_inputs,
            device=f"cuda:{torch.cuda.current_device()}",
        )
        decoded_chunks = [[] for _ in batch]
        for infer_index, _, chunk in sample_transport.walk():
            decoded_chunks[infer_index].append(post_chunk_process(chunk, config))
        dist.barrier()
        videos = [torch.cat(chunks, dim=0) for chunks in decoded_chunks]

        save_error = [None]
        if rank == 0:
            try:
                for (_, sample), video in zip(batch, videos):
                    name = sample[0]
                    output_path = args.output_dir / f"{name}.mp4"
                    partial_path = args.output_dir / f".{name}.partial.mp4"
                    if output_path.exists():
                        raise FileExistsError(
                            f"Refusing to overwrite output created during run: {output_path}"
                        )
                    if partial_path.exists():
                        partial_path.unlink()
                    actual_tensor_frames = int(video.shape[0])
                    if actual_tensor_frames != expected_frames:
                        raise RuntimeError(
                            f"MAGI returned {actual_tensor_frames} frames for {name}; "
                            f"expected {expected_frames}."
                        )
                    save_video_to_disk(video, str(partial_path), fps=fps)
                    actual_file_frames = probe_frame_count(partial_path)
                    if actual_file_frames != expected_frames:
                        raise RuntimeError(
                            f"Encoded {actual_file_frames} frames for {name}; "
                            f"expected {expected_frames}."
                        )
                    os.replace(partial_path, output_path)
                    print(
                        f"Saved: {output_path} | fps={fps}, "
                        f"saved_prediction_frames={actual_file_frames}, "
                        "saved_condition_frames=0",
                        flush=True,
                    )
            except BaseException as exc:
                save_error[0] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(save_error, src=0)
        if save_error[0] is not None:
            raise RuntimeError(
                f"Failed to save batch {', '.join(batch_names)}: {save_error[0]}"
            )
        dist.barrier()

        del prefix_videos, text_features, transport_inputs, sample_transport
        del decoded_chunks, videos
        gc.collect()
        torch.cuda.empty_cache()
        if rank == 0:
            for prefix_path in prefix_paths:
                if prefix_path.exists():
                    prefix_path.unlink()
        dist.barrier()

    print_rank_0("All selected MAGI MotionScape samples completed.")


if __name__ == "__main__":
    main()
