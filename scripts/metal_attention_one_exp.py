#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-attention-one-exp"
SOURCE = ROOT / "apps" / "benchmark" / "metal" / "attention_one_exp.mm"
GENERATED_SOURCE = BUILD_DIR / "attention_one_exp_production.mm"
BINARY = BUILD_DIR / "celeg-metal-attention-one-exp-benchmark"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def build() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text(encoding="utf-8")
    source = source.replace(
        '        const std::string source =\n'
        '            read_text("src/backend/metal/kernels/inference/common.metal") + "\\n" +\n'
        '            read_text("src/backend/metal/kernels/inference/batch.metal") + "\\n" +\n'
        '            read_text("apps/benchmark/metal/attention_one_exp.metal");',
        '        const std::string source =\n'
        '            std::string("#define celeg_attention_batch celeg_attention_batch_baseline\\n") +\n'
        '            read_text("src/backend/metal/kernels/inference/common.metal") + "\\n" +\n'
        '            read_text("src/backend/metal/kernels/inference/batch.metal") +\n'
        '            "\\n#undef celeg_attention_batch\\n" +\n'
        '            read_text("src/backend/metal/kernels/inference/attention_one_exp.metal");',
    )
    source = source.replace(
        'make_pipeline(device, library, "celeg_attention_batch");\n'
        '        id<MTLComputePipelineState> candidate =\n'
        '            make_pipeline(device, library, "celeg_attention_batch_one_exp");',
        'make_pipeline(device, library, "celeg_attention_batch_baseline");\n'
        '        id<MTLComputePipelineState> candidate =\n'
        '            make_pipeline(device, library, "celeg_attention_batch");',
    )
    GENERATED_SOURCE.write_text(source, encoding="utf-8")
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
        description="Benchmark production bit-exact one-exp Metal causal attention."
    )
    parser.add_argument(
        "--build-only", action="store_true",
        help="compile the Objective-C++ harness without running it",
    )
    args = parser.parse_args()
    build()
    if not args.build_only:
        run([str(BINARY)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
