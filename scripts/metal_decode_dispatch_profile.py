#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (
    pathlib.Path.home()
    / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-350M-GGUF"
)
DEFAULT_BINARY = ROOT / "out" / "darwin-metal-relwithdebinfo" / "celeg-metal-bench"
DEFAULT_MODELS = (
    "LFM2.5-350M-Q4_0.gguf",
    "LFM2.5-350M-Q4_K_M.gguf",
    "LFM2.5-350M-Q5_K_M.gguf",
    "LFM2.5-350M-Q6_K.gguf",
    "LFM2.5-350M-Q8_0.gguf",
)
def resolve_models(model_dir: pathlib.Path, requested: list[str] | None) -> list[pathlib.Path]:
    by_name = {
        path.name: path
        for path in sorted(model_dir.expanduser().glob("snapshots/*/*.gguf"))
    }
    names = requested or list(DEFAULT_MODELS)
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit("missing GGUF files: " + ", ".join(missing))
    return [by_name[name] for name in names]


def parse_benchmark_json(stdout: str) -> dict[str, object]:
    # celeg-metal-bench historically used std::quoted for JSON strings. That
    # leaves raw control characters possible inside backend descriptions.
    return json.loads(stdout, strict=False)


def profile(binary: pathlib.Path, model: pathlib.Path, prompt_tokens: int) -> tuple[dict[str, object], list[tuple[str, float]]]:
    process = subprocess.run(
        [
            str(binary),
            "--model", str(model),
            "--context", str(max(128, prompt_tokens + 32)),
            "--prompt-tokens", str(prompt_tokens),
            "--decode-tokens", "1",
            "--numerical-policy", "fast",
            "--warmup", "0",
            "--repetitions", "1",
            "--profile-dispatches", "gpu-stage",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stdout + process.stderr)
    report = parse_benchmark_json(process.stdout)
    dispatch_profile = report.get("profile_dispatches")
    if not isinstance(dispatch_profile, dict) or dispatch_profile.get("mode") != "gpu-stage":
        raise RuntimeError("benchmark did not return a gpu-stage dispatch profile")
    entries = [
        (str(entry["kernel"]), float(entry.get("total_ms", 0.0)))
        for entry in dispatch_profile["decode"]["kernels"]
    ]
    entries.sort(key=lambda item: item[1], reverse=True)
    return report, entries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attribute one real LFM2.5 Metal decode token to GPU dispatch kernels."
    )
    parser.add_argument("--model-dir", type=pathlib.Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", action="append", dest="models", metavar="FILENAME")
    parser.add_argument("--prompt-tokens", type=int, default=32)
    args = parser.parse_args()

    binary = args.binary.expanduser()
    if not binary.is_absolute():
        binary = (ROOT / binary).resolve()
    if not binary.is_file():
        raise SystemExit(f"missing Metal benchmark binary: {binary}")
    if args.prompt_tokens < 1:
        raise SystemExit("--prompt-tokens must be positive")

    print("Metal one-token decode GPU dispatch profile")
    print(f"binary={binary}")
    print(f"prompt_tokens={args.prompt_tokens}\n")
    for model in resolve_models(args.model_dir, args.models):
        report, entries = profile(binary, model, args.prompt_tokens)
        decode_ms = float(report["decode_ms"])
        accounted_ms = sum(milliseconds for _, milliseconds in entries)
        print(f"{model.name}: decode={decode_ms:.3f} ms counters={accounted_ms:.3f} ms")
        for name, milliseconds in entries:
            percent = milliseconds * 100.0 / accounted_ms if accounted_ms else 0.0
            print(f"  {name:<44} {milliseconds:8.4f} ms  {percent:5.1f}%")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
