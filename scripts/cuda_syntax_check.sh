#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
CLANGXX="${CLANGXX:-clang++}"
CUDA_ARCH="${CUDA_ARCH:-sm_80}"

if [[ ! -f "$CUDA_PATH/include/cuda_runtime.h" ]]; then
    echo "cuda_runtime.h not found under CUDA_PATH=$CUDA_PATH" >&2
    exit 2
fi

HOST_COMMON=(
    -std=c++20
    -Wall -Wextra -Wpedantic
    -I"$ROOT/include"
    -I"$ROOT/src"
    -I"$CUDA_PATH/include"
    -fsyntax-only
)

for source in \
    "$ROOT/src/model/execution/runtime_types.cpp" \
    "$ROOT/src/backend/cuda/execution_plan.cpp" \
    "$ROOT/src/backend/cpu/memory/prefix_cache.cpp" \
    "$ROOT/src/runtime/concurrency/request_registry.cpp" \
    "$ROOT/src/runtime/concurrency/batch_planner.cpp" \
    "$ROOT/src/backend/cpu/concurrent.cpp" \
    "$ROOT/apps/cuda/main.cpp"; do
    echo "checking ${source#$ROOT/}"
    "$CLANGXX" "${HOST_COMMON[@]}" "$source"
done

echo "checking examples/api_example.c"
"${CC:-cc}" -std=c11 -Wall -Wextra -Wpedantic \
    -I"$ROOT/include" -fsyntax-only "$ROOT/examples/api_example.c"

COMMON=(
    -std=c++20
    -Wall -Wextra -Wpedantic
    -x cuda
    "--cuda-path=$CUDA_PATH"
    "--cuda-gpu-arch=$CUDA_ARCH"
    -Wno-unknown-cuda-version
    -isystem "$CUDA_PATH/include"
    -I"$ROOT/include"
    -I"$ROOT/src"
    -I"$ROOT/tests"
    -I"$ROOT/src/backend/cuda"
    -I"$ROOT/src/backend/cuda/kernels"
    -fsyntax-only
)

if ! echo '' | "$CLANGXX" "${COMMON[@]}" - >/dev/null 2>&1; then
    echo "cuda_syntax_check: skipping CUDA TUs: $CLANGXX cannot parse" \
        "the toolkit headers at $CUDA_PATH (host checks passed)"
    exit 0
fi

while IFS= read -r source; do
    echo "checking ${source#$ROOT/}"
    "$CLANGXX" "${COMMON[@]}" "$source"
done < <(find \
    "$ROOT/src/backend/cuda" \
    "$ROOT/tests/cuda" \
    -name '*.cu' -print | sort)

echo "cuda_syntax_check: all CUDA translation units parsed"
