#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (
    pathlib.Path.home()
    / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-350M-GGUF"
)
DEFAULT_BINARY = ROOT / "out" / "darwin-metal-relwithdebinfo" / "celeg-metal-bench"
DEFAULT_MODELS = (
    "LFM2.5-350M-BF16.gguf",
    "LFM2.5-350M-F16.gguf",
)
TOGGLE = "CELEG_METAL_DENSE_MATVEC_ROWS"


def resolve_models(model_dir: pathlib.Path, requested: list[str] | None) -> list[pathlib.Path]:
    by_name: dict[str, pathlib.Path] = {}
    for path in sorted(model_dir.expanduser().glob("snapshots/*/*.gguf")):
        by_name[path.name] = path
    names = requested or list(DEFAULT_MODELS)
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit("missing GGUF files: " + ", ".join(missing))
    return [by_name[name] for name in names]


def run_case(
    binary: pathlib.Path,
    model: pathlib.Path,
    prompt_tokens: int,
    decode_tokens: int,
    warmup: int,
    repetitions: int,
    enabled: bool,
) -> dict[str, object]:
    context = prompt_tokens + decode_tokens + 128
    env = os.environ.copy()
    if enabled:
        env[TOGGLE] = "1"
    else:
        env.pop(TOGGLE, None)
    process = subprocess.run(
        [
            str(binary),
            "--model", str(model),
            "--context", str(context),
            "--prompt-tokens", str(prompt_tokens),
            "--decode-tokens", str(decode_tokens),
            "--numerical-policy", "fast",
            "--warmup", str(warmup),
            "--repetitions", str(repetitions),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stdout + "\n" + process.stderr).strip()
        raise RuntimeError(detail)
    return json.loads(process.stdout, strict=False)


def combined_rate(document: dict[str, object], prompt_tokens: int, decode_tokens: int) -> float:
    total_ms = float(document["prefill_ms"]) + float(document["decode_ms"])
    return (prompt_tokens + decode_tokens) * 1000.0 / total_ms


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the production Metal dense matvec baseline against the "
            "selective bit-exact row-reuse experiment on Apple M5."
        )
    )
    parser.add_argument("--model-dir", type=pathlib.Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", action="append", dest="models", metavar="FILENAME")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()

    if args.prompt_tokens < 1 or args.decode_tokens < 1 or args.repetitions < 1 or args.warmup < 0:
        raise SystemExit("prompt/decode/repetitions must be positive and warmup non-negative")

    binary = args.binary.expanduser()
    if not binary.is_absolute():
        binary = (ROOT / binary).resolve()
    if not binary.is_file():
        raise SystemExit(f"missing Metal benchmark binary: {binary}")

    models = resolve_models(args.model_dir, args.models)
    print("Metal dense matvec selective row-reuse production A/B")
    print(f"binary={binary}")
    print(
        f"prompt={args.prompt_tokens} decode={args.decode_tokens} "
        f"warmup={args.warmup} repetitions={args.repetitions}"
    )
    print(f"candidate={TOGGLE}=1; policy=fast; Apple M5 only\n")

    for model in models:
        baseline = run_case(
            binary, model, args.prompt_tokens, args.decode_tokens,
            args.warmup, args.repetitions, False,
        )
        candidate = run_case(
            binary, model, args.prompt_tokens, args.decode_tokens,
            args.warmup, args.repetitions, True,
        )

        base_prefill = float(baseline["prefill_tokens_per_second"])
        cand_prefill = float(candidate["prefill_tokens_per_second"])
        base_decode = float(baseline["decode_tokens_per_second"])
        cand_decode = float(candidate["decode_tokens_per_second"])
        base_combined = combined_rate(baseline, args.prompt_tokens, args.decode_tokens)
        cand_combined = combined_rate(candidate, args.prompt_tokens, args.decode_tokens)
        base_decode_ms = float(baseline["decode_ms"])
        cand_decode_ms = float(candidate["decode_ms"])
        base_gpu_ms = float(baseline["decode_gpu_execution_ms"])
        cand_gpu_ms = float(candidate["decode_gpu_execution_ms"])

        print(model.name)
        print(
            f"  prefill   {base_prefill:9.2f} -> {cand_prefill:9.2f} tok/s  "
            f"{cand_prefill / base_prefill:6.3f}x"
        )
        print(
            f"  decode    {base_decode:9.2f} -> {cand_decode:9.2f} tok/s  "
            f"{cand_decode / base_decode:6.3f}x  "
            f"({base_decode_ms:.3f} -> {cand_decode_ms:.3f} ms)"
        )
        print(
            f"  decodeGPU {base_gpu_ms:9.3f} -> {cand_gpu_ms:9.3f} ms     "
            f"{base_gpu_ms / cand_gpu_ms:6.3f}x"
        )
        print(
            f"  combined  {base_combined:9.2f} -> {cand_combined:9.2f} tok/s  "
            f"{cand_combined / base_combined:6.3f}x"
        )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
