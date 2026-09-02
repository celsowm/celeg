#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-attention-tiled-simdgroup"
SOURCE = ROOT / "apps" / "benchmark" / "metal" / "attention_tiled_simdgroup.mm"
BINARY = BUILD_DIR / "celeg-metal-attention-tiled-simdgroup-benchmark"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def build() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    run([
        "xcrun", "--sdk", "macosx", "clang++",
        "-std=c++20", "-fobjc-arc", str(SOURCE),
        "-framework", "Foundation", "-framework", "Metal",
        "-o", str(BINARY),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark tiled simdgroup Metal attention.")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    build()
    if not args.build_only:
        run([str(BINARY)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
