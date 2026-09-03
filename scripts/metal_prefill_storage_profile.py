#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (
    pathlib.Path.home()
    / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-350M-GGUF"
)
DEFAULT_BINARY = ROOT / "out" / "darwin-metal-relwithdebinfo" / "celeg-metal-bench"
EXPECTED_MODELS = (
    "LFM2.5-350M-BF16.gguf",
    "LFM2.5-350M-F16.gguf",
    "LFM2.5-350M-Q4_0.gguf",
    "LFM2.5-350M-Q4_K_M.gguf",
    "LFM2.5-350M-Q5_K_M.gguf",
    "LFM2.5-350M-Q6_K.gguf",
    "LFM2.5-350M-Q8_0.gguf",
    "LFM2.5-350M-QAD-Q4_0.gguf",
)
PROFILE_LINE = re.compile(r"^\s{2}(\S+)=(\d+)$")


def resolve_models(model_dir: pathlib.Path, requested: list[str] | None) -> list[pathlib.Path]:
    by_name: dict[str, pathlib.Path] = {}
    for path in sorted(model_dir.expanduser().glob("snapshots/*/*.gguf")):
        by_name[path.name] = path
    names = requested or list(EXPECTED_MODELS)
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit("missing GGUF files: " + ", ".join(missing))
    return [by_name[name] for name in names]


def parse_benchmark_json(stdout: str) -> dict[str, object]:
    # celeg-metal-bench historically used std::quoted for JSON strings. That
    # escapes quotes/backslashes but can leave raw control characters emitted
    # by a backend description. Keep the parser strict about the surrounding
    # document while accepting those legacy string contents until the producer
    # is migrated to a full JSON string encoder.
    return json.loads(stdout, strict=False)


def profile(binary: pathlib.Path, model: pathlib.Path, rows: int, repetitions: int) -> tuple[dict[str, object], dict[str, int]]:
    context = max(128, rows + 128)
    env = os.environ.copy()
    env["CELEG_METAL_TENSOR_RELAXED_PRECISION"] = "1"
    env["CELEG_METAL_DISPATCH_PROFILE"] = "1"
    env.pop("CELEG_METAL_GPU_PROFILE", None)
    result = subprocess.run(
        [
            str(binary),
            "--model", str(model),
            "--context", str(context),
            "--prompt-tokens", str(rows),
            "--decode-tokens", "0",
            "--warmup", "1",
            "--repetitions", str(repetitions),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)
    report = parse_benchmark_json(result.stdout)
    histogram: dict[str, int] = {}
    for line in result.stderr.splitlines():
        match = PROFILE_LINE.fullmatch(line)
        if match:
            name, count = match.groups()
            histogram[name] = histogram.get(name, 0) + int(count)
    return report, histogram


def relevant(histogram: dict[str, int]) -> list[tuple[str, int]]:
    prefixes = (
        "celeg_matmul_tensor_",
        "celeg_attention",
        "celeg_swiglu",
        "celeg_qk_",
        "celeg_shortconv_",
    )
    return sorted(
        ((name, count) for name, count in histogram.items() if name.startswith(prefixes)),
        key=lambda item: (-item[1], item[0]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile relaxed Metal prefill dispatch selection across LFM2.5-350M GGUF storages."
    )
    parser.add_argument("--model-dir", type=pathlib.Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", action="append", dest="models", metavar="FILENAME")
    parser.add_argument("--rows", action="append", type=int, dest="row_counts")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()

    binary = args.binary.expanduser()
    if not binary.is_absolute():
        binary = (ROOT / binary).resolve()
    if not binary.is_file():
        raise SystemExit(f"missing Metal benchmark binary: {binary}")
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")

    row_counts = args.row_counts or [32, 512]
    if any(rows < 1 for rows in row_counts):
        raise SystemExit("--rows values must be positive")
    models = resolve_models(args.model_dir, args.models)

    print("Metal relaxed prefill storage profile")
    print(f"binary={binary}")
    print(f"rows={','.join(str(value) for value in row_counts)} repetitions={args.repetitions}\n")

    for model in models:
        print(model.name)
        for rows in row_counts:
            report, histogram = profile(binary, model, rows, args.repetitions)
            rate = float(report["prefill_tokens_per_second"])
            milliseconds = float(report["prefill_ms"])
            print(f"  pp{rows}: {rate:9.1f} t/s  {milliseconds:8.3f} ms")
            entries = relevant(histogram)
            if not entries:
                print("    ERROR: no relevant dispatch-profile entries")
            for name, count in entries:
                print(f"    {name}={count}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
