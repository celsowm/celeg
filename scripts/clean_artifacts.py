#!/usr/bin/env python3
"""Safely remove CELEG build artifacts, logs, or one explicit HF model cache.

The targets are deliberately closed: build cleanup only touches the two known
build directories, and model cleanup only touches the exact Hugging Face cache
directory derived from ``owner/repo``. Preview is the default; destructive
work requires --apply.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIRECTORIES = (
    ROOT / "out" / "windows-cpu-release",
    ROOT / "out" / "windows-cuda-release",
)
LOG_PATTERNS = ("celeg-*.log",)
HF_REPO_PATTERN = re.compile(r"^[^/\\:]+/[^/\\:]+$")


def bytes_in(path: pathlib.Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def targets(include_builds: bool, include_logs: bool) -> list[pathlib.Path]:
    result: list[pathlib.Path] = []
    if include_builds:
        result.extend(path for path in BUILD_DIRECTORIES if path.exists())
    if include_logs:
        temporary = pathlib.Path(tempfile.gettempdir()).resolve()
        for pattern in LOG_PATTERNS:
            result.extend(path for path in temporary.glob(pattern) if path.is_file())
    return sorted(set(result))


def model_cache_path(repo: str) -> pathlib.Path:
    if not HF_REPO_PATTERN.fullmatch(repo):
        raise ValueError("--repo must have the exact owner/repo form")
    owner, name = repo.split("/", 1)
    cache_root = os.environ.get("HF_HUB_CACHE")
    if cache_root is None:
        hf_home = os.environ.get("HF_HOME")
        cache_root = str(pathlib.Path(hf_home) / "hub") if hf_home else str(
            pathlib.Path.home() / ".cache" / "huggingface" / "hub"
        )
    return pathlib.Path(cache_root).expanduser().resolve() / f"models--{owner}--{name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--builds", action="store_true", help="remove only known build directories")
    group.add_argument("--logs", action="store_true", help="remove only temporary celeg-*.log files")
    group.add_argument("--all", action="store_true", help="remove both known build directories and logs")
    group.add_argument("--model-cache", action="store_true", help="remove one exact Hugging Face model cache")
    parser.add_argument("--repo", help="exact Hugging Face owner/repo; required with --model-cache")
    parser.add_argument("--apply", action="store_true", help="perform deletion; otherwise print a preview")
    parser.add_argument("--verbose", action="store_true", help="list every exact target")
    args = parser.parse_args()

    if args.model_cache and not args.repo:
        parser.error("--repo is required with --model-cache")
    if not args.model_cache and args.repo:
        parser.error("--repo is only valid with --model-cache")
    paths = ([path for path in (model_cache_path(args.repo),) if path.exists()]
             if args.model_cache else
             targets(args.builds or args.all, args.logs or args.all))
    total = sum(bytes_in(path) for path in paths)
    action = "REMOVE" if args.apply else "PREVIEW"
    print(f"{action}: {len(paths)} CELEG artifact target(s), {format_bytes(total)}")
    if args.verbose:
        for path in paths:
            print(f"  {path} ({format_bytes(bytes_in(path))})")
    elif paths:
        print("Use --verbose to list every exact target.")

    if not args.apply:
        print("No files were deleted. Re-run with --apply to remove these exact targets.")
        return 0

    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    print("CELEG artifact cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
