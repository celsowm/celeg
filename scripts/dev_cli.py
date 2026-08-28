"""Command-line entrypoint for the celeg developer harness."""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
from typing import Sequence

from dev_build import BuildCoordinator, SmokeCoordinator, TestCoordinator, build_directory
from dev_environment import DevError, discover_environment
from dev_reporting import print_doctor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "build", "test", "smoke", "verify"))
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda", "metal"), default="auto")
    parser.add_argument(
        "--build-type",
        choices=("Release", "RelWithDebInfo", "Debug"),
        default="RelWithDebInfo",
    )
    parser.add_argument(
        "--arch",
        type=normalize_arch,
        default="native",
        help="CUDA architecture, for example native, 86, or 120a",
    )
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--build-dir")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable doctor output")
    parser.add_argument(
        "--celeg-tests",
        dest="celeg_tests",
        choices=("on", "off"),
        default="on",
        help="Register and run tests that require a visible CUDA device",
    )
    parser.add_argument(
        "--asan",
        action="store_true",
        help="Configure CELEG_ASAN=ON (AddressSanitizer) for CPU builds only",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.json and args.command != "doctor":
        parser.error("--json is only valid with doctor")
    return args


def normalize_arch(value: str) -> str:
    normalized = value.lower().removeprefix("compute_").removeprefix("sm_").replace(".", "")
    if normalized == "native":
        return normalized
    if not re.fullmatch(r"\d{2,3}[a-z]?", normalized):
        raise argparse.ArgumentTypeError(f"invalid CUDA architecture: {value}")
    return normalized


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    try:
        environment = discover_environment(args.backend, args.arch)
        print_doctor(environment, args.json)
        if args.command == "doctor":
            return 0 if environment.ok else 1
        if not environment.ok:
            return 1

        directory = build_directory(args, environment)
        BuildCoordinator(args, environment, directory).run()
        if args.command in ("test", "verify"):
            TestCoordinator(args, environment, directory).run()
        if args.command in ("smoke", "verify"):
            SmokeCoordinator(args, environment, directory).run()
        print(f"{args.command}: PASS ({environment.backend}, {args.build_type}, {directory})")
        return 0
    except (DevError, OSError) as error:
        print(f"{args.command}: FAIL: {error}", file=sys.stderr)
        return 1
