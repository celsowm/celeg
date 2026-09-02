#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = ROOT / "out" / "darwin-metal-release"
SOURCE = ROOT / "apps" / "benchmark" / "metal" / "rope_table.mm"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def build_backend(build_dir: pathlib.Path, jobs: int) -> None:
    run([
        sys.executable,
        str(ROOT / "scripts" / "dev.py"),
        "build",
        "--backend", "metal",
        "--celeg-tests", "off",
        "--build-type", "Release",
        "--build-dir", str(build_dir),
        "--jobs", str(jobs),
    ])


def compile_harness(build_dir: pathlib.Path) -> pathlib.Path:
    generated = build_dir / "generated" / "metal_inference_source.hpp"
    if not generated.exists():
        raise SystemExit(f"generated Metal inference source does not exist: {generated}")
    binary = build_dir / "celeg-metal-rope-table-benchmark"
    run([
        "xcrun", "--sdk", "macosx", "clang++",
        "-std=c++20", "-fobjc-arc",
        "-I", str(build_dir / "generated"),
        str(SOURCE),
        "-framework", "Foundation",
        "-framework", "Metal",
        "-o", str(binary),
    ])
    return binary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark fused LFM2.5 Metal QK norm/RoPE/KV with a reusable RoPE table."
    )
    parser.add_argument("--build-dir", type=pathlib.Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 4)))
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    build_dir = args.build_dir.expanduser().resolve()

    if not args.no_build:
        build_backend(build_dir, args.jobs)
    binary = compile_harness(build_dir)

    if not args.build_only:
        run([str(binary)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
