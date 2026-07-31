"""Static dependency checks for the model/backend architecture boundary."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


def files(root: Path, pattern: str):
    return (p for p in root.glob(pattern) if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []

    neutral_model_paths = (
        "include/lfm/model/config",
        "include/lfm/model/execution",
        "include/lfm/model/model.hpp",
        "include/lfm/model/reference.hpp",
        "include/lfm/model/definition.hpp",
        "include/lfm/model/architecture.hpp",
    )
    for relative in neutral_model_paths:
        path_root = root / relative
        candidates = [path_root] if path_root.is_file() else files(path_root, "**/*")
        for path in candidates:
            text = path.read_text(encoding="utf-8")
            if "lfm/backend/cuda" in text or "lfm/detail" in text:
                errors.append(f"backend/internal include in public-neutral header: {path}")

    for relative in ("include/lfm/checkpoint",):
        directory = root / relative
        for path in files(directory, "**/*"):
            text = path.read_text(encoding="utf-8")
            if "lfm/backend/cuda" in text or "lfm/detail" in text:
                errors.append(f"backend/internal include in public-neutral header: {path}")

    legacy = re.compile(r"\b(?:LfmModel|LfmInferenceSession|LfmDiagnostics|LfmPersistence|IPackedSession)\b")
    for relative in ("include/lfm", "src", "tests", "examples"):
        for path in files(root / relative, "**/*"):
            if path.suffix not in {".h", ".hpp", ".c", ".cpp", ".cu", ".inl"}:
                continue
            if legacy.search(path.read_text(encoding="utf-8")):
                errors.append(f"legacy public/interface symbol remains: {path}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("architecture boundary checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
