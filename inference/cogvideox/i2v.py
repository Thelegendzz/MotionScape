#!/usr/bin/env python3
"""Multi-GPU, batched CogVideoX1.5-5B-I2V inference for MotionScape.

This entry point reuses the shared MotionScape data reader in ``data.py``. It
starts one process and one full model
replica per GPU visible to PyTorch, round-robins samples between workers, and
processes each worker's samples in batches of ``--batch-size``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import data as base


def parse_args() -> argparse.Namespace:
    """Parse the shared CLI plus the batch-only option without duplicating it."""
    batch_parser = argparse.ArgumentParser(add_help=False)
    batch_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Distinct samples processed together on each GPU (default: 1).",
    )
    batch_args, remaining = batch_parser.parse_known_args()
    if batch_args.batch_size < 1:
        batch_parser.error("--batch-size must be at least 1.")
    sys.argv = [sys.argv[0], *remaining]
    args = base.parse_args()
    args.batch_size = batch_args.batch_size
    return args


def run_worker(
    rank: int,
    world_size: int,
    samples: list[tuple[str, Path, Path]],
    model_path: str,
    output_dir: str,
    prompt_template: str,
    args: argparse.Namespace,
) -> None:
    import torch
    from diffusers import CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image
    from PIL import Image, ImageOps

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    worker_samples = samples[rank::world_size]
    print(
        f"[GPU {rank}/{world_size}] loading one model replica; "
        f"assigned {len(worker_samples)} samples, batch size {args.batch_size}.",
        flush=True,
    )
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    for batch_start in range(0, len(worker_samples), args.batch_size):
        batch_samples = worker_samples[batch_start : batch_start + args.batch_size]
        global_indices = [rank + (batch_start + offset) * world_size for offset in range(len(batch_samples))]
        prompts = [
            re.sub(
                r"\s+",
                " ",
                prompt_template.replace("{clip_text_annotation}", base.read_clip_prompt(annotation_path, args.caption_field)),
            ).strip()
            for _, _, annotation_path in batch_samples
        ]
        result = pipe(
            prompt=prompts,
            image=[load_image(str(frame_path)) for _, frame_path, _ in batch_samples],
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            use_dynamic_cfg=True,
            generator=[torch.Generator(device=device).manual_seed(args.seed + index) for index in global_indices],
        )
        for offset, ((name, _, _), frames) in enumerate(zip(batch_samples, result.frames, strict=True)):
            output_path = Path(output_dir) / f"{name}.mp4"
            export_to_video(frames, str(output_path), fps=args.fps)
            print(
                f"[GPU {rank}] [{batch_start + offset + 1}/{len(worker_samples)}] {name}: {output_path}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    if args.cpu_offload:
        raise ValueError("--cpu-offload cannot be used here; every worker retains its model replica on GPU.")

    input_root = args.input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    model_path = args.model_path.expanduser().resolve()
    if not (model_path / "model_index.json").is_file():
        raise FileNotFoundError(f"CogVideoX checkpoint not found at {model_path}")
    caption_root = (args.caption_root or input_root.parent / "annotations" / "future_caption").expanduser().resolve()
    if not caption_root.is_dir():
        raise NotADirectoryError(f"Caption directory not found: {caption_root}")
    prompt_template = (args.prompt_template.read_text() if args.prompt_template else base.DEFAULT_PROMPT).strip()
    if "{clip_text_annotation}" not in prompt_template:
        raise ValueError("Prompt template must include '{clip_text_annotation}'.")

    samples = list(base.iter_samples(input_root, caption_root))
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError(f"No {base.I2W_FRAME_NAME} files found below {input_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, _, _ in samples:
        output_path = args.output_dir / f"{name}.mp4"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite to allow it.")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select visible GPUs before launching this program.")
    world_size = torch.cuda.device_count()
    print(
        f"Using {world_size} visible GPU(s), {world_size} model replicas, "
        f"and per-GPU batch size {args.batch_size} for {len(samples)} samples.",
        flush=True,
    )
    worker_args = (world_size, samples, str(model_path), str(args.output_dir), prompt_template, args)
    if world_size == 1:
        run_worker(0, *worker_args)
    else:
        torch.multiprocessing.spawn(run_worker, args=worker_args, nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
