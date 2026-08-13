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
        root / "src/model",
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
            if "include" in directory.parts and "celeg" in directory.parts and "celeg/detail" in text:
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

    # Model-family implementations are an importer concern.  Keep the
    # dependency direction structural so adding the hundredth family never
    # requires extending a family-name denylist.
    for relative in ("src/models", "include/celeg/models"):
        directory = root / relative
        if directory.exists():
            errors.append(f"model-family production directory remains: {relative}")

    family_include = re.compile(r"#\s*include\s*[<\"]celeg/models/")
    neutral_dependency_roots = (
        "include/celeg/model", "src/model", "include/celeg/runtime", "src/runtime",
        "include/celeg/checkpoint", "src/checkpoint", "include/celeg/backend", "src/backend",
        "include/celeg/composition", "src/composition", "include/celeg/text", "src/text",
        "include/celeg/serve", "src/serve",
    )
    for relative in neutral_dependency_roots:
        for path in files(root / relative, "**/*"):
            if path.suffix not in {".h", ".hpp", ".c", ".cpp", ".cu", ".cuh", ".inl"}:
                continue
            if family_include.search(path.read_text(encoding="utf-8")):
                errors.append(f"model-family include leaked across neutral boundary: {path}")

    family_factory = re.compile(r"\bmake_(?:gemma|qwen|lfm|granite|minicpm|smollm|"
                                r"nanbeige|nemotron|muse)[A-Za-z0-9_]*\b", re.IGNORECASE)
    for relative in ("src/composition", "include/celeg/composition"):
        for path in files(root / relative, "**/*"):
            if path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cuh"}:
                continue
            if family_factory.search(path.read_text(encoding="utf-8")):
                errors.append(f"family-prefixed factory remains in composition: {path}")

    builtin_runtime = root / "src/composition/builtin_runtime.cpp"
    if builtin_runtime.is_file():
        text = builtin_runtime.read_text(encoding="utf-8")
        if re.search(r"kDeclarativeChats|model_type|architecture_id|chat_profile|"
                     r"vision_family", text):
            errors.append("builtin_runtime.cpp still contains identity-driven composition")

    # Tokenizer providers and neutral serving/runtime consumers depend on the
    # tokenizer contract, not on the built-in BPE implementation.
    tokenizer_contract_roots = (
        root / "include/celeg/runtime", root / "src/runtime",
        root / "include/celeg/serve", root / "src/serve",
    )
    for directory in tokenizer_contract_roots:
        for path in files(directory, "**/*"):
            if path.suffix not in {".h", ".hpp", ".c", ".cpp", ".cu"}:
                continue
            if path.name == "tokenizer_providers.cpp":
                continue
            if re.search(r"\bBpeTokenizer\b", path.read_text(encoding="utf-8")):
                errors.append(f"concrete BPE tokenizer leaked across contract boundary: {path}")

    # Once a model is expressible by the descriptor vocabulary, its execution
    # and naming policies must not return as family-owned production files.
    # Keep this check filename/structure based so it applies to future models
    # without maintaining a list of family names.
    for relative in ("src/models", "include/celeg/models"):
        for path in files(root / relative, "**/*"):
            if path.name.lower() in {"architecture.cpp", "architecture.hpp",
                                     "naming_policy.cpp", "naming_policy.hpp",
                                     "layer_decoder.cpp", "metadata_decoder.cpp"}:
                errors.append(f"family execution/import file remains: {path}")

    # CELEG's replacement API deliberately has no version-stacked symbols.
    # External protocol/library versions (for example cublas_v2) are outside
    # this rule because the pattern is scoped to CELEG-owned public symbols.
    celeg_versioned = re.compile(r"\bceleg_[A-Za-z0-9_]*_v[23]\b")
    compatibility_identifier = re.compile(
        r"\b(?:legacy|deprecated|compat(?!ible|ibility))[A-Za-z0-9_]*\b",
        re.IGNORECASE)
    for relative in ("include/celeg", "src", "tests", "examples"):
        for path in files(root / relative, "**/*"):
            if path.suffix not in {".h", ".hpp", ".c", ".cpp", ".cu", ".inl"}:
                continue
            text = path.read_text(encoding="utf-8")
            if celeg_versioned.search(text):
                errors.append(f"version-stacked CELEG symbol remains: {path}")
            for line_number, line in enumerate(text.splitlines(), 1):
                code = line.split("//", 1)[0]
                if compatibility_identifier.search(code):
                    errors.append(
                        f"compatibility-named CELEG symbol remains: {path}:{line_number}")
                    break

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

    # The host expert cache is a byte store. Format readers and semantic
    # region names belong to source/binding adapters, never to the generic
    # runtime cache contract.
    generic_cache_tokens = re.compile(
        r"(?:SafeTensor|safetensors|cuda(?:Malloc|Free|Host)|"
        r"\b(?:gate_up|down)_bytes?\b|\b(?:gate_up|down)\s*\()")
    for path in files(root / "include/celeg/runtime/cache", "**/*"):
        if path.suffix not in {".h", ".hpp", ".c", ".cpp", ".cu", ".inl"}:
            continue
        if generic_cache_tokens.search(path.read_text(encoding="utf-8")):
            errors.append(f"format/backend/payload-specific knowledge leaked into generic cache: {path}")

    # The public ABI adapter validates and translates. Concrete service
    # construction is owned by backend factories.
    api_engine = root / "src/api/engine.cpp"
    if api_engine.is_file() and re.search(
            r"(?:CpuInferenceService|CudaInferenceService|make_unique<[^>]*(?:Cpu|Cuda))",
            api_engine.read_text(encoding="utf-8")):
        errors.append(f"concrete backend service construction remains in C API adapter: {api_engine}")

    # Packed execution must consume resources and compiled policy prepared by
    # its owner. These checks keep accidental hot-path plan compilation and
    # direct CUDA allocation out of the orchestration translation unit.
    packed_execution = root / "src/backend/cuda/packed/execution.cu"
    if packed_execution.is_file():
        packed_text = packed_execution.read_text(encoding="utf-8")
        forbidden_packed = (
            (r"CudaExecutionPlan::compile\s*\(", "per-call CUDA plan compilation"),
            (r"\bcudaMalloc(?:Host)?\s*\(", "direct packed CUDA allocation"),
            (r"\bDeviceBuffer\s*<[^>]+>\s*\([^)]*\)",
             "inline packed DeviceBuffer construction"),
        )
        for pattern, description in forbidden_packed:
            if re.search(pattern, packed_text):
                errors.append(f"{description} remains in {packed_execution}")

    # Paged KV storage owns page lifetime and byte movement; attention kernels
    # belong to the attention execution unit. Keep the split structural so a
    # future storage edit cannot quietly reintroduce the execution dependency.
    paged_storage = root / "src/backend/cpu/memory/paged_kv.cpp"
    if paged_storage.is_file():
        storage_text = paged_storage.read_text(encoding="utf-8")
        forbidden_attention = (
            "update_online", "cpu_gqa_decode", "cpu_latent_attention_decode",
            "PartialAttention", "CELEG_CPU_AVX2_TARGET", "cpu_gqa_prefill_paged",
        )
        for symbol in forbidden_attention:
            if symbol in storage_text:
                errors.append(
                    f"attention implementation remains in paged KV storage: {paged_storage} ({symbol})")

    # Descriptor/import code resolves semantic data; primitive execution bodies
    # belong to backend operator/executor translation units.
    for path in files(root / "src/model", "descriptor*.cpp"):
        if re.search(r"\b(?:execute_cpu_|execute_cuda_|launch_[A-Za-z0-9_]+\s*\()",
                     path.read_text(encoding="utf-8")):
            errors.append(f"primitive execution body leaked into descriptor/import file: {path}")

    # Ordinary CUDA decode is orchestration after the attention and mixer
    # extraction. Kernel launch policy must not silently grow back here.
    cuda_decode = root / "src/backend/cuda/model/execution.cu"
    if cuda_decode.is_file() and re.search(
            r"\b(?:launch_gqa_decode|launch_latent_attention|launch_store_kv|"
            r"launch_dynamic_qk_norm_rope)_", cuda_decode.read_text(encoding="utf-8")):
        errors.append(f"CUDA attention execution body remains in orchestration file: {cuda_decode}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("architecture boundary checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
