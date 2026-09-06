#!/usr/bin/env python3
"""Download the official model snapshots used by MotionScape.

All Hugging Face revisions are pinned to the snapshots recorded by the
benchmark experiments. Authentication is read from ``HF_TOKEN`` or the local
Hugging Face login; tokens are never accepted as command-line arguments.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str
    directory: str
    purpose: str
    gated: bool = False


MODEL_SPECS = {
    "cosmos-predict2.5-2b": ModelSpec(
        repo_id="nvidia/Cosmos-Predict2.5-2B",
        revision="15a82a2ec231bc318692aa0456a36537c806e7d4",
        directory="Cosmos-Predict2.5-2B",
        purpose="Cosmos-Predict2.5-2B T2W/I2W/V2W",
        gated=True,
    ),
    "cogvideox1.5-5b": ModelSpec(
        repo_id="zai-org/CogVideoX1.5-5B",
        revision="fdc5267c90b5c06492985b966e43aae984e189e0",
        directory="CogVideoX1.5-5B",
        purpose="CogVideoX1.5-5B Text2World",
    ),
    "cogvideox1.5-5b-i2v": ModelSpec(
        repo_id="zai-org/CogVideoX1.5-5B-I2V",
        revision="46c90528707aebbe69066390b4fe7e7d24c9c2a4",
        directory="CogVideoX1.5-5B-I2V",
        purpose="CogVideoX1.5-5B Image2World",
    ),
    "wan2.2-t2v-a14b": ModelSpec(
        repo_id="Wan-AI/Wan2.2-T2V-A14B",
        revision="c8c270b13ee05bfa474194ac9fb07a5868a97cea",
        directory="Wan2.2-T2V-A14B",
        purpose="Wan2.2-A14B Text2World",
    ),
    "wan2.2-i2v-a14b": ModelSpec(
        repo_id="Wan-AI/Wan2.2-I2V-A14B",
        revision="206a9ee1b7bfaaf8f7e4d81335650533490646a3",
        directory="Wan2.2-I2V-A14B",
        purpose="Wan2.2-A14B Image2World",
    ),
    "longcat-video": ModelSpec(
        repo_id="meituan-longcat/LongCat-Video",
        revision="03b55529b1d1d4045f5fbe14d65c8c6e8116b278",
        directory="LongCat-Video",
        purpose="LongCat-Video Video2World",
    ),
    "magi-1-24b": ModelSpec(
        repo_id="sand-ai/MAGI-1",
        revision="ca8bdfb28b1d88a29ec2a3ea423029f801a891bd",
        directory="MAGI-1-24B",
        purpose="MAGI-1-24B Video2World (24B, T5, and VAE weights)",
    ),
    "clip-vit-base-patch32": ModelSpec(
        repo_id="openai/clip-vit-base-patch32",
        revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        directory="clip-vit-base-patch32",
        purpose="CLIPSim evaluation",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Large data-disk directory under which one folder per model is created.",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        choices=("all", *MODEL_SPECS),
        help="Model to download; repeat the option or use 'all'.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum concurrent Hugging Face downloads per snapshot.",
    )
    parser.add_argument(
        "--endpoint",
        help="Optional Hugging Face endpoint. Prefer the official https://huggingface.co endpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pinned repositories and destinations without downloading.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print every recorded model specification as JSON and exit.",
    )
    args = parser.parse_args()
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive.")
    return args


def selected_model_names(values: list[str]) -> list[str]:
    if "all" in values:
        return list(MODEL_SPECS)
    requested = set(values)
    return [name for name in MODEL_SPECS if name in requested]


def printable_specs(output_root: Path | None = None) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, spec in MODEL_SPECS.items():
        item = asdict(spec)
        item["url"] = f"https://huggingface.co/{spec.repo_id}"
        if output_root is not None:
            item["destination"] = str(output_root / spec.directory)
        result[name] = item
    return result


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if args.list:
        print(json.dumps(printable_specs(output_root), indent=2))
        return

    names = selected_model_names(args.model)
    print(json.dumps({name: printable_specs(output_root)[name] for name in names}, indent=2))
    if args.dry_run:
        return

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing huggingface_hub. Install the repository requirements first."
        ) from exc

    output_root.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    for index, name in enumerate(names, start=1):
        spec = MODEL_SPECS[name]
        destination = output_root / spec.directory
        print(
            f"[{index}/{len(names)}] Downloading {spec.repo_id}@{spec.revision} "
            f"to {destination}",
            flush=True,
        )
        if spec.gated and not token:
            print(
                "  This is a gated repository. Ensure its license has been accepted "
                "and run `hf auth login`, or export HF_TOKEN.",
                flush=True,
            )
        snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            local_dir=destination,
            token=token,
            max_workers=args.max_workers,
        )
        print(f"  Complete: {destination}", flush=True)


if __name__ == "__main__":
    main()
