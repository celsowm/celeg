#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = ROOT / "out" / "darwin-metal-relwithdebinfo"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the current Metal benchmark binary, then run benchmarks/compare_metal.py."
    )
    parser.add_argument("--build-dir", type=pathlib.Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 4)))
    args, forwarded = parser.parse_known_args()

    build_dir = args.build_dir.expanduser().resolve()
    run([
        sys.executable,
        str(ROOT / "scripts" / "dev.py"),
        "build",
        "--backend", "metal",
        "--celeg-tests", "off",
        "--build-type", "RelWithDebInfo",
        "--build-dir", str(build_dir),
        "--jobs", str(args.jobs),
    ])

    binary = build_dir / "celeg-metal-bench"
    if not binary.is_file():
        raise SystemExit(f"Metal benchmark binary was not produced: {binary}")
    run([
        sys.executable,
        str(ROOT / "benchmarks" / "compare_metal.py"),
        "--celeg", str(binary),
        *forwarded,
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
