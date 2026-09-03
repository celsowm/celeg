#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = ROOT / "out" / "darwin-metal-relwithdebinfo"


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=capture,
        check=True,
    )


def build(build_dir: pathlib.Path, jobs: int) -> None:
    run([
        sys.executable,
        str(ROOT / "scripts" / "dev.py"),
        "build",
        "--backend", "metal",
        "--celeg-tests", "off",
        "--build-type", "RelWithDebInfo",
        "--build-dir", str(build_dir),
        "--jobs", str(jobs),
    ])


def print_matvec_profile(build_dir: pathlib.Path) -> None:
    binary = build_dir / "celeg-metal-kernel-benchmark"
    if not binary.is_file():
        raise RuntimeError(f"missing kernel benchmark: {binary}")
    result = run([str(binary)], capture=True)
    report = json.loads(result.stdout)
    peak = float(report["copy_roofline_gb_per_second"])
    print("\n=== Metal decode matvec roofline ===")
    print(f"copy_roofline={peak:.1f} GB/s")
    print(f"{'kernel':28s} {'shape':24s} {'ms':>9s} {'GB/s':>9s} {'roof':>8s}")
    for row in report["rows"]:
        if row.get("section") != "matvec":
            continue
        print(
            f"{str(row['kernel']):28s} {str(row['shape']):24s} "
            f"{float(row['ms']):9.4f} {float(row['gb_per_second']):9.1f} "
            f"{float(row['percent_of_roofline']):7.1f}%"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build once and run the focused Metal prefill/decode gap diagnostics."
    )
    parser.add_argument("--build-dir", type=pathlib.Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 4)))
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    build_dir = args.build_dir.expanduser().resolve()

    if not args.no_build:
        build(build_dir, args.jobs)

    bench = build_dir / "celeg-metal-bench"
    if not bench.is_file():
        raise SystemExit(f"missing Metal benchmark binary: {bench}")

    print("\n=== Relaxed prefill storage dispatch profile ===")
    run([
        sys.executable,
        str(ROOT / "scripts" / "metal_prefill_storage_profile.py"),
        "--binary", str(bench),
    ])

    print("\n=== Q4_K relaxed N128 vs N32 ===")
    run([sys.executable, str(ROOT / "scripts" / "metal_q4k_relaxed_n32.py")])

    print_matvec_profile(build_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
