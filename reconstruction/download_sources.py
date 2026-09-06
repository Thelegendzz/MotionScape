#!/usr/bin/env python3
"""Download the unique source videos referenced by MotionScape source_manifest.json.

The downloader preserves the manifest's exact ``video`` filename, deduplicates
items by source URL, refuses filename conflicts, and never overwrites an
existing source video. It selects media using each item's recorded resolution, nominal FPS, and codec,
while retaining the manifest's acquisition block as provenance.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yt-dlp", default="yt-dlp", help="yt-dlp executable")
    parser.add_argument("--archive", type=Path, help="Optional yt-dlp download archive")
    parser.add_argument("--proxy", help="Optional proxy URL passed to yt-dlp")
    parser.add_argument("--cookies-from-browser", help="Optional yt-dlp browser name")
    parser.add_argument(
        "--extractor-args",
        default="youtube:player_client=default",
        help="yt-dlp extractor arguments used for reconstruction downloads.",
    )
    parser.add_argument("--extra-arg", action="append", default=[], help="Additional yt-dlp argument; repeat as needed")
    parser.add_argument("--limit", type=int, help="Download at most N unique sources")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise SystemExit(f"Expected a public MotionScape manifest with an items list: {path}")
    acquisition = payload.get("source_acquisition") or {}
    return acquisition, payload["items"]


def unique_sources(items: list[dict[str, Any]]) -> list[tuple[str, str]]:
    by_url: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        video = str(item.get("video") or "").strip()
        if not url or not video:
            raise SystemExit(f"Manifest item is missing url/video: {item.get('sample_id')}")
        if Path(video).name != video or video in {".", ".."}:
            raise SystemExit(f"Unsafe source filename in manifest: {video!r}")
        previous = by_url.get(url)
        if previous is not None and previous != video:
            raise SystemExit(f"One URL maps to conflicting filenames: {url}: {previous!r} vs {video!r}")
        by_url[url] = video
    return sorted(by_url.items(), key=lambda pair: (pair[1].casefold(), pair[0]))


def manifest_format_selector(item: dict[str, Any]) -> str:
    """Select the recorded resolution, nominal FPS, and codec."""
    sample_id = item.get("sample_id")
    match = re.fullmatch(r"([0-9]+)x([0-9]+)", str(item.get("resolution") or ""))
    if match is None:
        raise SystemExit("Invalid resolution for {}: {!r}".format(sample_id, item.get("resolution")))
    width, height = match.groups()
    try:
        fps = float(Fraction(str(item.get("fps") or "")))
    except (ValueError, ZeroDivisionError) as exc:
        raise SystemExit("Invalid FPS for {}: {!r}".format(sample_id, item.get("fps"))) from exc
    rounded_fps = round(fps)
    if abs(fps - rounded_fps) > 0.1:
        raise SystemExit("Unsupported nominal FPS for {}: {!r}".format(sample_id, item.get("fps")))
    codec_prefixes = {"vp9": "vp9", "av1": "av01", "h264": "avc1"}
    codec = str(item.get("codec") or "").lower()
    if codec not in codec_prefixes:
        raise SystemExit("Unsupported codec for {}: {!r}".format(sample_id, item.get("codec")))
    return (
        f"bv[width={width}][height={height}]"
        f"[fps={rounded_fps}][vcodec^={codec_prefixes[codec]}]+ba"
    )


def main() -> None:
    args = parse_args()
    acquisition, items = load_manifest(args.manifest)
    sources = unique_sources(items)
    if args.limit is not None:
        if args.limit < 0:
            raise SystemExit("--limit must be non-negative")
        sources = sources[: args.limit]

    executable = shutil.which(args.yt_dlp) if not Path(args.yt_dlp).is_file() else args.yt_dlp
    if executable is None:
        raise SystemExit(f"yt-dlp executable not found: {args.yt_dlp}")

    merge_format = str(acquisition.get("merge_output_format") or "mp4")
    extractor_args = args.extractor_args
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.archive or (args.output_dir / "download_archive.txt")

    completed = 0
    skipped = 0
    for index, (url, video) in enumerate(sources, start=1):
        destination = args.output_dir / video
        if destination.is_file():
            print(f"[{index}/{len(sources)}] existing: {destination}")
            skipped += 1
            continue

        output_template = args.output_dir / f"{Path(video).stem}.%(ext)s"
        item = next(item for item in items if str(item.get("url") or "").strip() == url)
        format_selector = manifest_format_selector(item)
        command = [
            str(executable),
            "--no-overwrites",
            "--continue",
            "--download-archive",
            str(archive),
            "--format",
            format_selector,
            "--merge-output-format",
            merge_format,
            "--extractor-args",
            extractor_args,
            "--output",
            str(output_template),
        ]
        if args.proxy:
            command.extend(["--proxy", args.proxy])
        if args.cookies_from_browser:
            command.extend(["--cookies-from-browser", args.cookies_from_browser])
        command.extend(args.extra_arg)
        command.append(url)

        print(f"[{index}/{len(sources)}] {shlex.join(command)}")
        if not args.dry_run:
            subprocess.run(command, check=True)
            if not destination.is_file():
                raise RuntimeError(
                    f"yt-dlp completed but exact manifest filename was not created: {destination}"
                )
        completed += 1

    print(json.dumps({"unique_sources": len(sources), "processed": completed, "existing": skipped}, indent=2))


if __name__ == "__main__":
    main()
