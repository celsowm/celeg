#!/usr/bin/env python3
"""Portable build, test, smoke, and environment diagnostics for celeg."""

from __future__ import annotations

import pathlib
import sys

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import dev_environment
from dev_environment import *
from dev_reporting import *
from dev_build import *
from dev_cli import main, normalize_arch, parse_args


if __name__ == "__main__":
    raise SystemExit(main())
