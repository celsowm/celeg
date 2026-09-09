#!/usr/bin/env python3

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

SHAPE_RE = re.compile(
    r"rows=(?P<rows>\d+) heads=(?P<heads>\d+) dim=(?P<dim>\d+) "
    r"chunks=(?P<chunks>\d+) bitwise=(?P<bitwise>yes|NO) "
    r"baseline_us=(?P<baseline>[0-9.]+) shared_us=(?P<shared>[0-9.]+) "
    r"ratio=(?P<ratio>[0-9.]+)"
)
ATTRIBUTE_RE = re.compile(
    r"(?P<name>baseline|shared)\s+regs=(?P<regs>\d+) local=(?P<local>\d+) "
    r"shared=(?P<shared>\d+) max_threads=(?P<max_threads>\d+)"
)


def parse_output(stdout: str) -> dict:
    device = None
    iterations = None
    attributes = {}
    shapes = []

    for line in stdout.splitlines():
        if line.startswith("device="):
            prefix, _, iteration_text = line.partition(" iterations=")
            device = prefix.removeprefix("device=")
            if iteration_text:
                iterations = int(iteration_text)
            continue

        attribute = ATTRIBUTE_RE.fullmatch(line.strip())
        if attribute:
            values = attribute.groupdict()
            attributes[values["name"]] = {
                "registers": int(values["regs"]),
                "local_bytes": int(values["local"]),
                "shared_bytes": int(values["shared"]),
                "max_threads": int(values["max_threads"]),
            }
            continue

        shape = SHAPE_RE.fullmatch(line.strip())
        if shape:
            values = shape.groupdict()
            shapes.append(
                {
                    "rows": int(values["rows"]),
                    "heads": int(values["heads"]),
                    "head_dim": int(values["dim"]),
                    "chunks": int(values["chunks"]),
                    "bitwise": values["bitwise"] == "yes",
                    "baseline_us": float(values["baseline"]),
                    "shared_us": float(values["shared"]),
                    "ratio": float(values["ratio"]),
                }
            )

    if not shapes:
        raise ValueError("segmented merge benchmark produced no shape results")

    positive_ratios = [shape["ratio"] for shape in shapes if shape["ratio"] > 0.0]
    if len(positive_ratios) != len(shapes):
        raise ValueError("segmented merge benchmark produced a non-positive ratio")

    geometric_mean = math.exp(
        sum(math.log(ratio) for ratio in positive_ratios) / len(positive_ratios)
    )
    max_shape = max(shapes, key=lambda shape: shape["ratio"])

    return {
        "device": device,
        "iterations": iterations,
        "attributes": attributes,
        "all_bitwise": all(shape["bitwise"] for shape in shapes),
        "geomean_ratio": geometric_mean,
        "max_ratio": max_shape["ratio"],
        "max_ratio_shape": {
            key: max_shape[key] for key in ("rows", "heads", "head_dim", "chunks")
        },
        "shapes": shapes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and summarize the CUDA segmented-attention merge benchmark."
    )
    parser.add_argument(
        "binary",
        type=Path,
        help="Path to celeg_segmented_merge_benchmark",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help="Timed iterations per shape (default: 2000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        help="Optional maximum allowed shared/baseline ratio; no default gate is imposed",
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.max_ratio is not None and args.max_ratio <= 0.0:
        parser.error("--max-ratio must be positive")
    if not args.binary.is_file():
        parser.error(f"benchmark binary not found: {args.binary}")

    completed = subprocess.run(
        [str(args.binary), str(args.iterations)],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    sys.stdout.write(completed.stdout)

    try:
        report = parse_output(completed.stdout)
    except ValueError as error:
        print(f"report_error={error}", file=sys.stderr)
        return completed.returncode or 2

    report["max_ratio_limit"] = args.max_ratio
    report["performance_gate_passed"] = (
        args.max_ratio is None or report["max_ratio"] <= args.max_ratio
    )

    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")

    if completed.returncode != 0:
        return completed.returncode
    if not report["all_bitwise"]:
        return 1
    return 0 if report["performance_gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
