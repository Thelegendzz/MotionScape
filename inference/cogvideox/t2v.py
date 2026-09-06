#!/usr/bin/env python3
"""Multi-GPU batched CogVideoX1.5-5B text-to-video inference for MotionScape."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_PROMPT = (
    "Generate a plausible future first-person onboard-camera video from the description below. "
    "Keep the described scene, environment, weather, camera motion, and geometry "
    "coherent, with smooth and realistic temporal progression.\n\n"
    "Future description:\n{clip_text_annotation}"
)


def read_caption(annotation_path: Path, field: str) -> str:
    """Read a dotted JSON field and turn structured MotionScape captions into text."""
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


def iter_samples(caption_root: Path):
    """Yield stable output names and annotation files without requiring video frames."""
    for annotation_path in sorted(caption_root.rglob("*.json")):
        relative = annotation_path.relative_to(caption_root).with_suffix("")
        name = "__".join(relative.parts)
        yield name, annotation_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caption-root", type=Path, required=True, help="Directory containing future-caption JSON files.")
    parser.add_argument("--model-path", type=Path, required=True, help="CogVideoX1.5-5B checkpoint directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated MP4 files.")
    parser.add_argument("--caption-field", default="", help="Dotted JSON field containing the future caption.")
    parser.add_argument("--prompt-template", type=Path, help="Optional template containing {clip_text_annotation}.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--num-frames", type=int, default=41)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.width % 16 or args.height % 16:
        parser.error("--width and --height must be multiples of 16.")
    if (args.num_frames - 1) % 4:
        parser.error("--num-frames must have the form 4n+1 for CogVideoX1.5.")
    return args


def run_worker(rank: int, world_size: int, samples, model_path: str, output_dir: str, prompt_template: str, args) -> None:
    import torch
    from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
    from diffusers.utils import export_to_video

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    worker_samples = samples[rank::world_size]
    print(
        f"[GPU {rank}/{world_size}] loading one T2V model replica; assigned "
        f"{len(worker_samples)} samples, batch size {args.batch_size}.",
        flush=True,
    )
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
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
                prompt_template.replace("{clip_text_annotation}", read_caption(annotation_path, args.caption_field)),
            ).strip()
            for _, annotation_path in batch_samples
        ]
        result = pipe(
            prompt=prompts,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            use_dynamic_cfg=True,
            generator=[torch.Generator(device=device).manual_seed(args.seed + index) for index in global_indices],
        )
        for offset, ((name, _), frames) in enumerate(zip(batch_samples, result.frames, strict=True)):
            output_path = Path(output_dir) / f"{name}.mp4"
            export_to_video(frames, str(output_path), fps=args.fps)
            print(f"[GPU {rank}] [{batch_start + offset + 1}/{len(worker_samples)}] {name}: {output_path}", flush=True)


def main() -> None:
    args = parse_args()
    caption_root = args.caption_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    if not caption_root.is_dir():
        raise NotADirectoryError(f"Caption directory not found: {caption_root}")
    if not (model_path / "model_index.json").is_file():
        raise FileNotFoundError(f"CogVideoX checkpoint not found at {model_path}")
    prompt_template = (args.prompt_template.read_text() if args.prompt_template else DEFAULT_PROMPT).strip()
    if "{clip_text_annotation}" not in prompt_template:
        raise ValueError("Prompt template must include {clip_text_annotation}.")
    samples = list(iter_samples(caption_root))
    if not samples:
        raise ValueError(f"No JSON annotations found below {caption_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_videos = tuple(args.output_dir.glob("*.mp4"))
    completed_names = {name for name, _ in samples if (args.output_dir / f"{name}.mp4").is_file()}
    if args.overwrite:
        candidate_samples = samples
    else:
        candidate_samples = [(name, path) for name, path in samples if name not in completed_names]
    pending_samples = candidate_samples
    if args.max_samples is not None:
        pending_samples = pending_samples[: args.max_samples]
    mode = "regenerating" if args.overwrite else "continuing with"
    print(
        f"Output checkpoint: found {len(existing_videos)} MP4 file(s), including "
        f"{len(completed_names)}/{len(samples)} caption sample(s); {mode} "
        f"{len(pending_samples)} sample(s) this run.",
        flush=True,
    )
    if not pending_samples:
        print("All caption samples already have outputs; nothing to generate.", flush=True)
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select visible GPUs before launching this program.")
    world_size = torch.cuda.device_count()
    print(
        f"Using {world_size} visible GPU(s), {world_size} T2V model replicas, "
        f"and per-GPU batch size {args.batch_size} for {len(pending_samples)} pending samples.",
        flush=True,
    )
    worker_args = (world_size, pending_samples, str(model_path), str(args.output_dir), prompt_template, args)
    if world_size == 1:
        run_worker(0, *worker_args)
    else:
        torch.multiprocessing.spawn(run_worker, args=worker_args, nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
