#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "results" / "metal_dense_stage16_promotion.json"
MODELS = (
    "LFM2.5-350M-BF16.gguf",
    "LFM2.5-350M-F16.gguf",
)


def rate(entry: dict, engine: str, phase: str) -> float:
    phase_name = "prefill_batched" if engine == "llama_cpp" and phase == "prefill" else phase
    return float(entry[engine][phase_name]["median_tokens_per_second"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the official F16/BF16 Metal throughput gate after dense stage16 promotion."
    )
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmup < 0:
        raise SystemExit("--repetitions must be positive and --warmup must be non-negative")

    command = [
        "python3", "benchmarks/compare_metal.py",
        "--workload", "throughput",
        "--repetitions", str(args.repetitions),
        "--warmup", str(args.warmup),
        "--output", str(OUTPUT),
    ]
    for model in MODELS:
        command += ["--model", model]

    subprocess.run(command, cwd=ROOT, check=True)
    document = json.loads(OUTPUT.read_text())
    print("\nDense stage16 production promotion gate")
    print(f"repetitions={args.repetitions} warmup={args.warmup}\n")
    for entry in document["results"]:
        if entry.get("status") != "ok":
            print(f"{pathlib.Path(entry['model']).name}: ERROR {entry.get('error', 'unknown')}")
            continue
        model = pathlib.Path(entry["model"]).name
        print(model)
        for phase in ("prefill", "decode", "combined"):
            celeg = rate(entry, "celeg", phase)
            llama = rate(entry, "llama_cpp", phase)
            print(
                f"  {phase:<8} Celeg {celeg:9.2f} tok/s  "
                f"llama.cpp {llama:9.2f} tok/s  speedup {celeg / llama:6.3f}x"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
