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

    manifest = root / "MANIFEST.sha256"
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if "  ./" not in line:
                continue
            relative = line.split("  ./", 1)[1].strip()
            if not (root / relative).exists():
                errors.append(f"manifest references obsolete path: {relative}")

    source_files = tuple(
        path for relative in ("include/celeg", "src", "tests")
        for path in files(root / relative, "**/*")
        if path.suffix in {".h", ".hpp", ".c", ".cpp", ".cu", ".cuh", ".inl"}
    )

    for path in source_files:
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bassert\s*\(", text) and path.parts[-2:] != ("support", "assertions.hpp"):
            errors.append(f"raw assert remains in test/source file: {path}")
        if "owner_->impl_" in text or re.search(r"\bnew\s+Impl\b", text):
            errors.append(f"legacy implementation access remains: {path}")
        if re.search(r"(?:legacy_prefill|legacy_sampling|prefill_legacy|--legacy-|--segmented-attention)", text):
            errors.append(f"obsolete compatibility path remains: {path}")

    for path in files(root / "tests", "**/*"):
        if path.suffix in {".h", ".hpp", ".c", ".cpp", ".cu", ".cuh"}:
            if re.search(r"^\s*using\s+namespace\b", path.read_text(encoding="utf-8"), re.MULTILINE):
                errors.append(f"using namespace is not allowed in test translation unit: {path}")

    neutral_roots = (
        root / "include/celeg/model",
        root / "include/celeg/runtime",
        root / "include/celeg/checkpoint",
        root / "include/celeg/text",
        root / "include/celeg/serve",
    )
    cuda_tokens = re.compile(
        r"(?:celeg/backend/cuda|cuda(?:Stream|Event|Graph|Error_t)|"
        r"__nv_bfloat16|cublas(?:Handle_t|LtHandle_t)|CUstream)"
    )
    for directory in neutral_roots:
        for path in files(directory, "**/*"):
            if path.suffix not in {".h", ".hpp", ".c", ".cpp", ".cu"}:
                continue
            text = path.read_text(encoding="utf-8")
            if cuda_tokens.search(text):
                errors.append(f"CUDA dependency in backend-neutral file: {path}")
            if "celeg/detail" in text:
                errors.append(f"backend/internal include in public-neutral header: {path}")

    legacy = re.compile(
        r"\b(?:CelegModel|CelegInferenceSession|CelegDiagnostics|CelegPersistence|"
        r"IPackedSession|InferenceSession|CpuModel::Impl|CudaModel::Impl|"
        r"CpuConcurrentEngine::Impl|ConcurrentEngine::Impl)\b"
    )
    for relative in ("include/celeg", "src", "tests", "examples"):
        for path in files(root / relative, "**/*"):
            if path.suffix not in {".h", ".hpp", ".c", ".cpp", ".cu", ".inl"}:
                continue
            if legacy.search(path.read_text(encoding="utf-8")):
                errors.append(f"legacy public/interface symbol remains: {path}")

    removed = re.compile(
        r"\b(?:ArchitectureKind|ModelConfig|ModelShape|IModelVariant|"
        r"ModelVariantRegistry|ArchitectureRegistry|IArchitectureProvider)\b"
    )
    for relative in ("include/celeg", "src", "tests", "examples", "cmake"):
        for path in files(root / relative, "**/*"):
            if path == Path(__file__).resolve():
                continue
            if path.suffix not in {".h", ".hpp", ".c", ".cpp", ".cu", ".inl", ".cmake"}:
                continue
            if removed.search(path.read_text(encoding="utf-8")):
                errors.append(f"removed architecture interface remains: {path}")

    backend_dispatch = re.compile(r"\b(?:architecture_id|architecture_kind|model_type)\b")
    for relative in ("include/celeg/backend", "src/backend"):
        for path in files(root / relative, "**/*"):
            if path.suffix not in {".h", ".hpp", ".c", ".cpp", ".cu", ".inl"}:
                continue
            if backend_dispatch.search(path.read_text(encoding="utf-8")):
                errors.append(f"architecture dispatch leaked into backend: {path}")
            if re.search(r"\b(?:is_gguf|CheckpointSourceFormat)\b", path.read_text(encoding="utf-8")):
                errors.append(f"format dispatch leaked into backend: {path}")

    removed_paths = (
        root / "include/celeg/model/execution/runtime_types.hpp",
        root / "include/celeg/model/execution/plan.hpp",
        root / "include/celeg/model/weights/loader.hpp",
        root / "include/celeg/model/weights/layout.hpp",
        root / "include/celeg/model/weights/policy.hpp",
        root / "include/celeg/serve/inference_service.hpp",
    )
    # inference_service.hpp is the neutral replacement and must exist; only
    # its removed composite symbol is forbidden below.
    for path in removed_paths[:-1]:
        if path.exists(): errors.append(f"obsolete architecture path remains: {path}")

    forbidden_symbols = re.compile(
        r"\b(?:ModelOptions|ExecutionPlan|IInferenceService|ModelDiagnostics)\b"
    )
    for relative in ("include/celeg", "src", "tests", "examples"):
        for path in files(root / relative, "**/*"):
            if path.suffix not in {".h", ".hpp", ".c", ".cpp", ".cu"}:
                continue
            if forbidden_symbols.search(path.read_text(encoding="utf-8")):
                errors.append(f"removed migration symbol remains: {path}")

    # D12: base_runtime is backend-independent by declaration; it must not
    # compile backend-specific sources. A src/backend/** entry here means the
    # build graph does not match the declared dependency direction.
    base_runtime_manifest = root / "cmake/sources/base_runtime.cmake"
    if base_runtime_manifest.is_file():
        backend_source = re.compile(r"^\s*src/backend/")
        for line in base_runtime_manifest.read_text(encoding="utf-8").splitlines():
            if backend_source.match(line):
                errors.append(
                    f"backend-specific source in CELEG_BASE_RUNTIME_SOURCES: {line.strip()}")

    generic_runtime = re.compile(r"\b(?:Lfm2|LFM2|Granite|GraniteModel)\b")
    for relative in ("include/celeg/runtime", "src/runtime"):
        for path in files(root / relative, "**/*"):
            if path.suffix not in {".h", ".hpp", ".c", ".cpp", ".cu", ".inl"}:
                continue
            if generic_runtime.search(path.read_text(encoding="utf-8")):
                errors.append(f"architecture-specific type leaked into generic runtime: {path}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("architecture boundary checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
