#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
from collections import defaultdict


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


def resolve_models(model_dir: pathlib.Path, requested: list[str] | None) -> list[pathlib.Path]:
    by_name: dict[str, pathlib.Path] = {}
    for path in sorted(model_dir.expanduser().glob("snapshots/*/*.gguf")):
        by_name[path.name] = path
    names = requested or list(DEFAULT_MODELS)
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit("missing GGUF files: " + ", ".join(missing))
    return [by_name[name] for name in names]


def family(name: str) -> str:
    if "[ffn_gate " in name:
        return "dense FFN gate"
    if "[ffn_up " in name:
        return "dense FFN up"
    if "[ffn_down " in name:
        return "dense FFN down"
    if name.startswith("celeg_matvec_"):
        return "dense non-FFN"
    if "attention" in name:
        return "attention"
    if "shortconv" in name:
        return "shortconv"
    if "swiglu" in name or "gelu" in name or "activation" in name:
        return "activation"
    if "norm" in name or "rms" in name:
        return "normalization"
    if "rope" in name or "qk_" in name or "kv_" in name:
        return "QK/KV"
    if name.startswith("celeg_"):
        return "other celeg"
    return "other"


def run_profile(
    binary: pathlib.Path,
    model: pathlib.Path,
    prompt_tokens: int,
    decode_tokens: int,
    repetitions: int,
    mode: str,
) -> dict[str, object]:
    context = prompt_tokens + decode_tokens + 128
    process = subprocess.run(
        [
            str(binary),
            "--model", str(model),
            "--context", str(context),
            "--prompt-tokens", str(prompt_tokens),
            "--decode-tokens", str(decode_tokens),
            "--numerical-policy", "fast",
            "--warmup", "1",
            "--repetitions", str(repetitions),
            "--profile-dispatches", mode,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stdout + "\n" + process.stderr).strip()
        raise RuntimeError(detail)
    return json.loads(process.stdout, strict=False)


def print_report(
    model: pathlib.Path,
    baseline: dict[str, object],
    report: dict[str, object],
    decode_tokens: int,
    top: int,
) -> None:
    profile = report.get("profile_dispatches")
    if not isinstance(profile, dict):
        raise RuntimeError("missing gpu-stage profile")
    decode = profile.get("decode")
    if not isinstance(decode, dict):
        raise RuntimeError("missing decode profile")
    kernels = decode.get("kernels", [])
    if not isinstance(kernels, list):
        raise RuntimeError("invalid decode kernel profile")

    runs = int(decode.get("runs", 1))
    base_wall_ms = float(baseline["decode_ms"])
    base_gpu_ms = float(baseline["decode_gpu_execution_ms"])
    wall_ms = float(report["decode_ms"])
    gpu_ms = float(report["decode_gpu_execution_ms"])
    encode_ms = float(report["decode_command_encoding_ms"])
    wait_ms = float(report["decode_command_wait_ms"])
    sampled_total = float(decode.get("sampled_dispatch_ms_total", 0.0)) / max(runs, 1)
    distortion = float(decode.get("distortion_percent", 0.0))

    print(model.name)
    print(
        f"  counts:        wall={base_wall_ms:.3f} ms  gpu={base_gpu_ms:.3f} ms  "
        f"rate={decode_tokens * 1000.0 / base_wall_ms:.2f} tok/s"
    )
    print(
        f"  split+counter: wall={wall_ms:.3f} ms  gpu={gpu_ms:.3f} ms  "
        f"encode={encode_ms:.3f} ms  wait={wait_ms:.3f} ms"
    )
    print(
        f"  mode ratio: wall={base_wall_ms / wall_ms:.3f}x  "
        f"gpu={base_gpu_ms / gpu_ms:.3f}x"
    )
    print(
        f"  sampled_dispatch={sampled_total:.3f} ms/run  "
        f"sample-vs-stage-gpu={distortion:+.1f}%"
    )

    normalized: list[tuple[str, float, float, float, float, float]] = []
    family_ms: dict[str, float] = defaultdict(float)
    for entry in kernels:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("kernel", ""))
        total_ms = float(entry.get("total_ms", 0.0)) / max(runs, 1)
        phase_percent = float(entry.get("phase_percent", 0.0))
        count_per_run = float(entry.get("count_per_run", 0.0))
        median_ms = float(entry.get("median_ms", 0.0))
        p95_ms = float(entry.get("p95_ms", 0.0))
        normalized.append((name, total_ms, phase_percent, count_per_run, median_ms, p95_ms))
        family_ms[family(name)] += total_ms

    print("\n  decode families")
    for name, total_ms in sorted(family_ms.items(), key=lambda item: (-item[1], item[0])):
        if total_ms <= 0.0:
            continue
        percent = total_ms * 100.0 / sampled_total if sampled_total > 0.0 else 0.0
        print(f"    {name:<22} {total_ms:9.3f} ms/run  {percent:6.2f}%")

    print("\n  hottest decode kernels")
    print(
        f"    {'kernel':<72} {'ms/run':>9} {'phase':>8} {'count':>8} "
        f"{'median':>9} {'p95':>9}"
    )
    for name, total_ms, phase_percent, count_per_run, median_ms, p95_ms in sorted(
        normalized, key=lambda item: (-item[1], item[0])
    )[:top]:
        if total_ms <= 0.0:
            continue
        print(
            f"    {name:<72} {total_ms:9.3f} {phase_percent:7.2f}% "
            f"{count_per_run:8.1f} {median_ms:9.4f} {p95_ms:9.4f}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile Metal decode after a pp512 prompt using dispatch GPU-stage timestamps."
    )
    parser.add_argument("--model-dir", type=pathlib.Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", action="append", dest="models", metavar="FILENAME")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    binary = args.binary.expanduser()
    if not binary.is_absolute():
        binary = (ROOT / binary).resolve()
    if not binary.is_file():
        raise SystemExit(f"missing Metal benchmark binary: {binary}")
    if args.prompt_tokens < 1 or args.decode_tokens < 1 or args.repetitions < 1 or args.top < 1:
        raise SystemExit("prompt/decode tokens, repetitions, and top must be positive")

    models = resolve_models(args.model_dir, args.models)
    print("Metal decode GPU-stage profile")
    print(f"binary={binary}")
    print(
        f"prompt={args.prompt_tokens} decode={args.decode_tokens} "
        f"repetitions={args.repetitions}\n"
    )
    for model in models:
        baseline = run_profile(
            binary, model, args.prompt_tokens, args.decode_tokens, args.repetitions, "counts"
        )
        staged = run_profile(
            binary, model, args.prompt_tokens, args.decode_tokens, args.repetitions, "gpu-stage"
        )
        print_report(model, baseline, staged, args.decode_tokens, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
