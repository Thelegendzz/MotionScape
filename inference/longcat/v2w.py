#!/usr/bin/env python3
"""MotionScape video-to-world inference with LongCat-Video.

This entry point uses LongCat-Video's Video-Continuation pipeline. Every
selected GPU loads one complete model replica and processes a disjoint sample
shard. Existing MP4 outputs are skipped by default.

MotionScape frames are 29.97 FPS while LongCat-Video generates at 15 FPS. The
default condition clip therefore takes every second source frame and ends at
``frame_000200.jpg``: frames 176, 178, ..., 200 (13 frames). LongCat internally
produces 53 frames (13 condition + 40 prediction frames). The exported MP4
discards all condition frames and keeps the 40 prediction frames.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path


CONDITION_END_FRAME_NAME = "frame_000200.jpg"

DEFAULT_LONGCAT_ROOT = Path(__file__).resolve().parents[3] / "LongCat-Video"
DEFAULT_SOURCE_FPS = 30000 / 1001
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, paintings, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, deformed, disfigured, still picture, messy background"
)
DEFAULT_PROMPT = (
    "Given observed first-person onboard-camera video frames and a future-frame description, "
    "generate a plausible future continuation. Preserve scene layout, appearance, "
    "and viewpoint, and continue the observed camera motion smoothly.\n\n"
    "Future description:\n"
    "{clip_text_annotation}"
)


def read_caption(annotation_path: Path, field: str) -> str:
    """Read a dotted JSON field and stringify a MotionScape caption object."""
    value: object = json.loads(annotation_path.read_text())
    if field:
        for key in field.split("."):
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
    return str(value).strip()


def annotation_for_frame(
    frame_path: Path, input_root: Path, caption_root: Path
) -> Path:
    """Find the future-caption JSON matching a MotionScape sample directory."""
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
    """Return ordered condition frames ending at ``end_frame_path``."""
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
    """Yield output names, condition-frame paths, and caption paths."""
    for end_frame_path in sorted(input_root.rglob(CONDITION_END_FRAME_NAME)):
        relative_parent = end_frame_path.parent.relative_to(input_root)
        yield (
            "__".join(relative_parent.parts),
            condition_frame_paths(
                end_frame_path, num_cond_frames, frame_stride
            ),
            annotation_for_frame(end_frame_path, input_root, caption_root),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="MotionScape full_frames directory.",
    )
    parser.add_argument(
        "--caption-root",
        type=Path,
        help="Future-caption JSON directory.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="LongCat-Video checkpoint directory.",
    )
    parser.add_argument(
        "--longcat-root",
        type=Path,
        default=DEFAULT_LONGCAT_ROOT,
        help="LongCat-Video source checkout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated MP4 files.",
    )
    parser.add_argument("--caption-field", default="")
    parser.add_argument(
        "--prompt-template",
        type=Path,
        help="Optional template containing {clip_text_annotation}.",
    )
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--resolution", choices=("480p", "720p"), default="480p")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=53,
        help="Internal frames: conditions plus predictions (default: 53; 4n+1).",
    )
    parser.add_argument(
        "--num-cond-frames",
        type=int,
        default=13,
        help="Condition frames at output start (LongCat default: 13).",
    )
    parser.add_argument(
        "--num-output-frames",
        type=int,
        default=40,
        help="Prediction-only frames saved after removing conditions (default: 40).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Output and condition sampling FPS (LongCat default: 15).",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=DEFAULT_SOURCE_FPS,
        help="MotionScape source FPS (default: 30000/1001).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        help="Condition stride; default round(source_fps/fps), i.e. 2.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device-id",
        type=int,
        help="Use only this logical CUDA device (default: all visible).",
    )
    parser.add_argument("--enable-compile", action="store_true")
    parser.add_argument("--offload-kv-cache", action="store_true")
    parser.add_argument(
        "--no-enhance-hf",
        action="store_false",
        dest="enhance_hf",
        help="Disable enhanced high-frequency denoising.",
    )
    parser.set_defaults(enhance_hf=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (args.num_frames - 1) % 4:
        parser.error("--num-frames must have the form 4n+1.")
    if not 0 < args.num_cond_frames <= args.num_frames:
        parser.error("--num-cond-frames must be in 1..num-frames.")
    available_prediction_frames = args.num_frames - args.num_cond_frames
    if not 0 < args.num_output_frames <= available_prediction_frames:
        parser.error(
            "--num-output-frames must be in 1.."
            f"{available_prediction_frames} for the selected frame settings."
        )
    if args.fps <= 0 or args.source_fps <= 0:
        parser.error("--fps and --source-fps must be positive.")
    if args.frame_stride is not None and args.frame_stride <= 0:
        parser.error("--frame-stride must be positive.")
    if args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be positive.")
    return args


def run_worker(
    worker_rank: int,
    device_ids: tuple[int, ...],
    init_method: str,
    args: argparse.Namespace,
    pending: list[tuple[int, tuple[str, tuple[Path, ...], Path]]],
) -> None:
    """Load one V2W model replica and generate this worker's sample shard."""
    device_id = device_ids[worker_rank]
    worker_samples = pending[worker_rank::len(device_ids)]
    if not worker_samples:
        return

    sys.path.insert(0, str(args.longcat_root))
    import numpy as np
    import torch
    import torch.distributed as dist
    from PIL import Image
    from torchvision.io import write_video
    from transformers import AutoTokenizer, UMT5EncoderModel

    from longcat_video.context_parallel import context_parallel_util
    from longcat_video.context_parallel.context_parallel_util import (
        init_context_parallel,
    )
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.longcat_video_dit import (
        LongCatVideoTransformer3DModel,
    )
    from longcat_video.modules.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline

    torch.cuda.set_device(device_id)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=worker_rank,
        world_size=len(device_ids),
    )
    init_context_parallel(
        context_parallel_size=1,
        global_rank=worker_rank,
        world_size=len(device_ids),
    )
    cp_split_hw = context_parallel_util.get_optimal_split(
        context_parallel_util.get_cp_size()
    )
    try:
        print(
            f"[GPU {device_id}] Loading LongCat V2W for "
            f"{len(worker_samples)} sample(s).",
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            subfolder="tokenizer",
            torch_dtype=torch.bfloat16,
        )
        text_encoder = UMT5EncoderModel.from_pretrained(
            args.model_path,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        )
        vae = AutoencoderKLWan.from_pretrained(
            args.model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.model_path,
            subfolder="scheduler",
            torch_dtype=torch.bfloat16,
        )
        dit = LongCatVideoTransformer3DModel.from_pretrained(
            args.model_path,
            subfolder="dit",
            cp_split_hw=cp_split_hw,
            torch_dtype=torch.bfloat16,
        )
        if args.enable_compile:
            dit = torch.compile(dit)
        pipeline = LongCatVideoPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            dit=dit,
        )
        pipeline.to(device_id)
        prompt_template = (
            args.prompt_template.read_text()
            if args.prompt_template
            else DEFAULT_PROMPT
        ).strip()
        if "{clip_text_annotation}" not in prompt_template:
            raise ValueError(
                "Prompt template must include {clip_text_annotation}."
            )
        print(f"[GPU {device_id}] LongCat model loaded.", flush=True)

        for local_index, (
            sample_index,
            (name, condition_paths, annotation_path),
        ) in enumerate(worker_samples, start=1):
            prompt = re.sub(
                r"\s+",
                " ",
                prompt_template.replace(
                    "{clip_text_annotation}",
                    read_caption(annotation_path, args.caption_field),
                ),
            ).strip()
            condition_video = [
                Image.open(path).convert("RGB") for path in condition_paths
            ]
            generator = torch.Generator(device=device_id).manual_seed(
                args.seed + sample_index
            )
            print(
                f"[GPU {device_id}] [{local_index}/{len(worker_samples)}] "
                f"Generating {name} from {len(condition_video)} conditions.",
                flush=True,
            )
            output = pipeline.generate_vc(
                video=condition_video,
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                resolution=args.resolution,
                num_frames=args.num_frames,
                num_cond_frames=args.num_cond_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                use_kv_cache=True,
                offload_kv_cache=args.offload_kv_cache,
                enhance_hf=args.enhance_hf,
            )[0]
            prediction = np.asarray(output)[
                args.num_cond_frames:
                args.num_cond_frames + args.num_output_frames
            ]
            output_tensor = (
                torch.from_numpy(prediction)
                .mul(255)
                .clamp(0, 255)
                .to(torch.uint8)
            )
            output_path = args.output_dir / f"{name}.mp4"
            write_video(
                str(output_path),
                output_tensor,
                fps=args.fps,
                video_codec="libx264",
                options={"crf": "18"},
            )
            actual_frames = int(output_tensor.shape[0])
            del output, prediction, output_tensor, condition_video
            torch.cuda.empty_cache()
            print(
                f"[GPU {device_id}] Saved: {output_path} | fps={args.fps:g}, "
                f"saved_prediction_frames={actual_frames}, "
                "saved_condition_frames=0, "
                f"inference_condition_frames={args.num_cond_frames}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    caption_root = (
        args.caption_root
        or input_root.parent / "annotations" / "future_caption"
    ).expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.longcat_root = args.longcat_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.frame_stride = args.frame_stride or max(
        1, round(args.source_fps / args.fps)
    )

    if not input_root.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_root}")
    if not caption_root.is_dir():
        raise NotADirectoryError(f"Caption directory not found: {caption_root}")
    if not (
        args.longcat_root / "longcat_video" / "pipeline_longcat_video.py"
    ).is_file():
        raise FileNotFoundError(
            f"LongCat source checkout not found: {args.longcat_root}"
        )
    required_checkpoint_entries = (
        "model_index.json",
        "tokenizer",
        "text_encoder",
        "vae",
        "scheduler",
        "dit",
    )
    missing = [
        entry
        for entry in required_checkpoint_entries
        if not (args.model_path / entry).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Incomplete LongCat checkpoint at {args.model_path}; missing: "
            + ", ".join(missing)
        )

    samples = list(
        iter_samples(
            input_root,
            caption_root,
            args.num_cond_frames,
            args.frame_stride,
        )
    )
    if not samples:
        raise ValueError(
            f"No {CONDITION_END_FRAME_NAME} below {input_root}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed_names = {
        name
        for name, _, _ in samples
        if (args.output_dir / f"{name}.mp4").is_file()
    }
    candidates = (
        samples
        if args.overwrite
        else [sample for sample in samples if sample[0] not in completed_names]
    )
    selected = (
        candidates[: args.max_samples]
        if args.max_samples is not None
        else candidates
    )
    pending = list(enumerate(selected))
    print(
        f"LongCat V2W specification: output_fps={args.fps:g}, "
        f"saved_prediction_frames={args.num_output_frames}, "
        "saved_condition_frames=0, "
        f"inference_frames={args.num_frames}, "
        f"inference_condition_frames={args.num_cond_frames}, "
        f"source_fps={args.source_fps:.5g}, "
        f"frame_stride={args.frame_stride}.",
        flush=True,
    )
    print(
        f"Output checkpoint: {len(completed_names)}/{len(samples)} complete; "
        f"generating {len(pending)} this run.",
        flush=True,
    )
    if not pending:
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    visible_gpu_count = torch.cuda.device_count()
    if args.device_id is not None:
        if not 0 <= args.device_id < visible_gpu_count:
            raise ValueError(
                f"--device-id {args.device_id} is outside "
                f"0..{visible_gpu_count - 1}"
            )
        device_ids = (args.device_id,)
    else:
        device_ids = tuple(range(visible_gpu_count))
    active_device_ids = device_ids[: min(len(device_ids), len(pending))]
    print(
        f"Launching {len(active_device_ids)} independent worker(s) on GPU(s) "
        f"{', '.join(map(str, active_device_ids))}; one full model per GPU.",
        flush=True,
    )

    init_fd, init_path = tempfile.mkstemp(
        prefix="longcat_v2w_dist_", dir="/tmp"
    )
    os.close(init_fd)
    os.unlink(init_path)
    init_method = f"file://{init_path}"
    try:
        if len(active_device_ids) == 1:
            run_worker(0, active_device_ids, init_method, args, pending)
        else:
            torch.multiprocessing.spawn(
                run_worker,
                args=(active_device_ids, init_method, args, pending),
                nprocs=len(active_device_ids),
                join=True,
            )
    finally:
        if os.path.exists(init_path):
            os.unlink(init_path)


if __name__ == "__main__":
    main()
