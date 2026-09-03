#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-q4k-relaxed-n32"
HOST_SOURCE = ROOT / "apps" / "benchmark" / "metal" / "relaxed_precision.mm"
GENERATED_SOURCE = BUILD_DIR / "q4k_relaxed_n32.mm"
BINARY = BUILD_DIR / "celeg-metal-q4k-relaxed-n32-benchmark"


def run(command: list[str], *, cwd: pathlib.Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    source = HOST_SOURCE.read_text(encoding="utf-8").replace("4608", "6656")
    source = source.replace(
        "constexpr uint32_t kTokenCounts[] = {128, 256, 512};",
        "constexpr uint32_t kTokenCounts[] = {32};",
    )
    source = source.replace(
        'read_text("apps/benchmark/metal/tensor_relaxed_precision.metal")',
        'read_text("src/backend/metal/kernels/tensor_q4k_relaxed.metal")',
    )
    pipeline_block = (
        '        id<MTLComputePipelineState> baseline =\n'
        '            make_pipeline(device, library, "celeg_matmul_tensor_q4k");\n'
        '        id<MTLComputePipelineState> relaxed =\n'
        '            make_pipeline(device, library, "celeg_matmul_tensor_q4k_relaxed");'
    )
    pipeline_replacement = (
        '        id<MTLComputePipelineState> baseline =\n'
        '            make_pipeline(device, library, "celeg_matmul_tensor_q4k_relaxed_n128");\n'
        '        id<MTLComputePipelineState> relaxed =\n'
        '            make_pipeline(device, library, "celeg_matmul_tensor_q4k_relaxed_n32");'
    )
    if pipeline_block not in source:
        raise RuntimeError("Q4_K relaxed pipeline block changed")
    source = source.replace(pipeline_block, pipeline_replacement)
    source = source.replace(
        "Metal TensorOps relaxed-precision A/B on ",
        "Metal production relaxed Q4_K N128 vs N32 A/B on ",
    )
    source = source.replace(
        "geometry=64x128xK64 threads=128 bit_exact=required\\n\\n",
        "baseline=N128 candidate=N32 K64 threads=128 bit_exact=reported\\n\\n",
    )
    source = source.replace('"strict ms"', '"N128 ms"')
    source = source.replace('"relaxed ms"', '"N32 ms"')

    needle = (
        "                require_bit_exact(queue, baseline, relaxed, weights, input,\n"
        "                                  tokens, shape, row_bytes);"
    )
    replacement = (
        "                try {\n"
        "                    require_bit_exact(queue, baseline, relaxed, weights, input,\n"
        "                                      tokens, shape, row_bytes);\n"
        "                    std::cout << \"bit_exact=yes shape=\" << shape.label\n"
        "                              << \" pp\" << tokens << '\\n';\n"
        "                } catch (const std::exception& mismatch) {\n"
        "                    std::cout << \"bit_exact=no shape=\" << shape.label\n"
        "                              << \" pp\" << tokens << \" detail=\"\n"
        "                              << mismatch.what() << '\\n';\n"
        "                }"
    )
    if needle not in source:
        raise RuntimeError("Q4_K relaxed correctness gate site changed")
    GENERATED_SOURCE.write_text(source.replace(needle, replacement), encoding="utf-8")
    run([
        "xcrun", "--sdk", "macosx", "clang++",
        "-std=c++20", "-fobjc-arc",
        str(GENERATED_SOURCE),
        "-framework", "Foundation",
        "-framework", "Metal",
        "-o", str(BINARY),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare production relaxed Q4_K N128 and N32 TensorOps tiles."
    )
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    build()
    if not args.build_only:
        run([str(BINARY)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
