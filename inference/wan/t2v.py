#!/usr/bin/env python3
"""MotionScape text-to-video inference with Wan2.2-T2V-A14B.

This is a multi-GPU data-parallel entry point. Each logical CUDA device visible
in the container loads one complete Wan T2V model and processes a disjoint
shard of caption JSON files. Existing output MP4 files are skipped by default.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path



DEFAULT_WAN_ROOT = Path(__file__).resolve().parents[3] / "Wan2.2"
DEFAULT_PROMPT = (
    "Generate a plausible future first-person onboard-camera video from the description below. "
    "Keep the described scene, environment, weather, camera motion, and geometry "
    "coherent, with smooth and realistic temporal progression.\n\n"
    "Future description:\n{clip_text_annotation}"
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


def iter_samples(caption_root: Path):
    """Yield stable output names and annotations without requiring condition frames."""
    for annotation_path in sorted(caption_root.rglob("*.json")):
        relative = annotation_path.relative_to(caption_root).with_suffix("")
        yield "__".join(relative.parts), annotation_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caption-root", type=Path, required=True, help="Directory containing future-caption JSON files.")
    parser.add_argument("--model-path", type=Path, required=True, help="Wan2.2-T2V-A14B checkpoint directory.")
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT, help="Wan2.2 source checkout containing the wan package.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated MP4 files.")
    parser.add_argument("--caption-field", default="")
    parser.add_argument("--prompt-template", type=Path, help="Optional template containing {clip_text_annotation}.")
    parser.add_argument("--size", default="1280*720", help="Wan T2V size key, for example 1280*720.")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=40,
        help="Number of generated frames saved to the output video; must be divisible by 4.",
    )
    parser.add_argument("--fps", type=int, default=16, help="MP4 FPS metadata (Wan default: 16).")
    parser.add_argument("--sample-steps", type=int, default=None, help="Defaults to Wan T2V config (40).")
    parser.add_argument("--sample-shift", type=float, default=None, help="Defaults to Wan T2V config (12.0).")
    parser.add_argument("--guide-scale", nargs=2, type=float, metavar=("LOW", "HIGH"), help="Two Wan T2V CFG scales.")
    parser.add_argument("--sample-solver", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-id", type=int, help="Use only this logical CUDA device (default: all visible devices).")
    parser.add_argument("--num-gpus", type=int, help="Use the first N visible logical GPUs (default: all).")
    parser.add_argument("--offload-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--t5-cpu", action="store_true", help="Keep the T5 encoder on CPU to reduce GPU memory.")
    parser.add_argument("--convert-model-dtype", action="store_true")
    parser.add_argument("--max-samples", type=int, help="Maximum pending samples to generate this run.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs that already exist.")
    args = parser.parse_args()
    if args.num_frames <= 0 or args.num_frames % 4:
        parser.error("--num-frames must be a positive multiple of 4.")
    if args.fps <= 0:
        parser.error("--fps must be positive.")
    if args.num_gpus is not None and args.num_gpus <= 0:
        parser.error("--num-gpus must be positive.")
    return args


def run_worker(
    worker_rank: int,
    device_ids: tuple[int, ...],
    args: argparse.Namespace,
    pending: list[tuple[int, tuple[str, Path]]],
) -> None:
    """Load one complete T2V pipeline on one CUDA device and generate its shard."""
    device_id = device_ids[worker_rank]
    worker_samples = pending[worker_rank::len(device_ids)]
    if not worker_samples:
        return

    sys.path.insert(0, str(args.wan_root))
    import torch
    import wan
    from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
    from wan.utils.utils import save_video

    if args.size not in SUPPORTED_SIZES["t2v-A14B"]:
        raise ValueError(f"Unsupported Wan T2V size {args.size}; choose one of: {', '.join(SUPPORTED_SIZES['t2v-A14B'])}")
    torch.cuda.set_device(device_id)
    config = WAN_CONFIGS["t2v-A14B"]
    guide_scale = tuple(args.guide_scale) if args.guide_scale is not None else config.sample_guide_scale
    print(f"[GPU {device_id}] Loading Wan2.2-T2V-A14B for {len(worker_samples)} sample(s) (worker {worker_rank + 1}/{len(device_ids)}).", flush=True)
    pipeline = wan.WanT2V(
        config=config,
        checkpoint_dir=str(args.model_path),
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )
    print(f"[GPU {device_id}] Wan model loaded.", flush=True)
    prompt_template = (args.prompt_template.read_text() if args.prompt_template else DEFAULT_PROMPT).strip()
    if "{clip_text_annotation}" not in prompt_template:
        raise ValueError("Prompt template must include {clip_text_annotation}.")

    for local_index, (sample_index, (name, annotation_path)) in enumerate(worker_samples, start=1):
        prompt = re.sub(r"\s+", " ", prompt_template.replace("{clip_text_annotation}", read_caption(annotation_path, args.caption_field))).strip()
        print(f"[GPU {device_id}] [{local_index}/{len(worker_samples)}] Generating {name}", flush=True)
        video = pipeline.generate(
            prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.num_frames + 1,
            shift=args.sample_shift if args.sample_shift is not None else config.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps if args.sample_steps is not None else config.sample_steps,
            guide_scale=guide_scale,
            seed=args.seed + sample_index,
            offload_model=args.offload_model,
        )
        if video is None:
            raise RuntimeError(f"[GPU {device_id}] WanT2V returned no video; independent workers must use rank=0.")
        output_path = args.output_dir / f"{name}.mp4"
        save_video(
            video[:, : args.num_frames][None],
            str(output_path),
            fps=args.fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        del video
        torch.cuda.empty_cache()
        print(f"[GPU {device_id}] Saved: {output_path}", flush=True)


def main() -> None:
    args = parse_args()
    caption_root = args.caption_root.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.wan_root = args.wan_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not caption_root.is_dir():
        raise NotADirectoryError(f"Caption directory not found: {caption_root}")
    if not args.wan_root.is_dir() or not (args.wan_root / "wan").is_dir():
        raise FileNotFoundError(f"Wan2.2 source checkout not found: {args.wan_root}")
    required_checkpoint_files = ("Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth")
    missing = [name for name in required_checkpoint_files if not (args.model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Wan checkpoint is incomplete at {args.model_path}; missing: {', '.join(missing)}")

    samples = list(iter_samples(caption_root))
    if not samples:
        raise ValueError(f"No JSON annotations found below {caption_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_videos = tuple(args.output_dir.glob("*.mp4"))
    completed_names = {name for name, _ in samples if (args.output_dir / f"{name}.mp4").is_file()}
    candidates = samples if args.overwrite else [sample for sample in samples if sample[0] not in completed_names]
    selected = candidates[:args.max_samples] if args.max_samples is not None else candidates
    pending = list(enumerate(selected))
    print(
        f"Output checkpoint: found {len(existing_videos)} MP4 file(s), including "
        f"{len(completed_names)}/{len(samples)} caption sample(s); generating {len(pending)} this run.",
        flush=True,
    )
    if not pending:
        return

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select GPU visibility before launching this script.")
    visible_gpu_count = torch.cuda.device_count()
    if args.device_id is not None:
        if not 0 <= args.device_id < visible_gpu_count:
            raise ValueError(f"--device-id {args.device_id} is outside 0..{visible_gpu_count - 1}")
        device_ids = (args.device_id,)
    else:
        requested_gpu_count = args.num_gpus or visible_gpu_count
        if requested_gpu_count > visible_gpu_count:
            raise ValueError(f"--num-gpus {requested_gpu_count} exceeds {visible_gpu_count} visible GPU(s)")
        device_ids = tuple(range(requested_gpu_count))
    active_device_ids = device_ids[:min(len(device_ids), len(pending))]
    print(
        f"Launching {len(active_device_ids)} Wan worker(s) on logical GPU(s) "
        f"{', '.join(map(str, active_device_ids))}; each worker loads a complete model.",
        flush=True,
    )
    if len(active_device_ids) == 1:
        run_worker(0, active_device_ids, args, pending)
    else:
        torch.multiprocessing.spawn(
            run_worker,
            args=(active_device_ids, args, pending),
            nprocs=len(active_device_ids),
            join=True,
        )


if __name__ == "__main__":
    main()
