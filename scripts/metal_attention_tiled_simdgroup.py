#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-attention-tiled-simdgroup"
SOURCE = ROOT / "apps" / "benchmark" / "metal" / "attention_tiled_simdgroup.mm"
GENERATED_SOURCE = BUILD_DIR / "attention_tiled_simdgroup_production.mm"
BINARY = BUILD_DIR / "celeg-metal-attention-tiled-simdgroup-benchmark"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def build() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text(encoding="utf-8")
    source = source.replace(
        'text("apps/benchmark/metal/attention_tiled_simdgroup.metal")',
        'text("src/backend/metal/kernels/inference/attention_tiled_simdgroup.metal")',
    )
    source = source.replace(
        '    [e setBytes:&rows length:4 atIndex:4]; [e setBytes:&QH length:4 atIndex:5];\n'
        '    [e setBytes:&KH length:4 atIndex:6]; [e setBytes:&HD length:4 atIndex:7];\n'
        '    [e setBytes:&SCALE length:4 atIndex:8]; [e setThreadgroupMemoryLength:TILED_SHARED atIndex:0];',
        '    const uint32_t pos = 0;\n'
        '    [e setBytes:&rows length:4 atIndex:4]; [e setBytes:&pos length:4 atIndex:5];\n'
        '    [e setBytes:&QH length:4 atIndex:6]; [e setBytes:&KH length:4 atIndex:7];\n'
        '    [e setBytes:&HD length:4 atIndex:8]; [e setBytes:&SCALE length:4 atIndex:9];\n'
        '    [e setBytes:&PAGE length:4 atIndex:10]; [e setThreadgroupMemoryLength:TILED_SHARED atIndex:0];',
    )
    GENERATED_SOURCE.write_text(source, encoding="utf-8")
    run([
        "xcrun", "--sdk", "macosx", "clang++",
        "-std=c++20", "-fobjc-arc", str(GENERATED_SOURCE),
        "-framework", "Foundation", "-framework", "Metal",
        "-o", str(BINARY),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark production tiled simdgroup Metal attention.")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    build()
    if not args.build_only:
        run([str(BINARY)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
