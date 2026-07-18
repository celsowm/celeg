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
    -I"$CUDA_PATH/include"
    -fsyntax-only
)

for source in     "$ROOT/src/runtime_types.cpp"     "$ROOT/src/execution_plan.cpp"     "$ROOT/src/concurrent_metrics.cpp"     "$ROOT/src/prefix_cache.cpp"     "$ROOT/src/request_registry.cpp"     "$ROOT/src/batch_planner.cpp"     "$ROOT/src/engine_worker.cpp"; do
    echo "checking ${source#$ROOT/}"
    "$CLANGXX" "${HOST_COMMON[@]}" "$source"
done

echo "checking src/main.cpp"
"$CLANGXX" "${HOST_COMMON[@]}" "$ROOT/src/main.cpp"
echo "checking src/c_api.cpp"
"$CLANGXX" "${HOST_COMMON[@]}" "$ROOT/src/c_api.cpp"
echo "checking src/concurrent.cpp"
"$CLANGXX" "${HOST_COMMON[@]}" "$ROOT/src/concurrent.cpp"
echo "checking src/concurrent_c_api.cpp"
"$CLANGXX" "${HOST_COMMON[@]}" "$ROOT/src/concurrent_c_api.cpp"
echo "checking src/concurrent_benchmark.cpp"
"$CLANGXX" "${HOST_COMMON[@]}" "$ROOT/src/concurrent_benchmark.cpp"
echo "checking src/prefix_cache_benchmark.cpp"
"$CLANGXX" "${HOST_COMMON[@]}" "$ROOT/src/prefix_cache_benchmark.cpp"
echo "checking examples/c_api_example.c"
"${CC:-cc}" -std=c11 -Wall -Wextra -Wpedantic \
    -I"$ROOT/include" -fsyntax-only "$ROOT/examples/c_api_example.c"
echo "checking examples/concurrent_c_api_example.c"
"${CC:-cc}" -std=c11 -Wall -Wextra -Wpedantic \
    -I"$ROOT/include" -fsyntax-only "$ROOT/examples/concurrent_c_api_example.c"

COMMON=(
    -std=c++20
    -Wall -Wextra -Wpedantic
    -x cuda
    "--cuda-path=$CUDA_PATH"
    "--cuda-gpu-arch=$CUDA_ARCH"
    -Wno-unknown-cuda-version
    -I"$ROOT/include"
    -fsyntax-only
)

for source in \
    "$ROOT/src/kernels.cu" \
    "$ROOT/src/model.cu" \
    "$ROOT/src/packed.cu" \
    "$ROOT/src/paged_kv.cu" \
    "$ROOT/tests/cuda_kernels_test.cu"; do
    echo "checking ${source#$ROOT/}"
    "$CLANGXX" "${COMMON[@]}" "$source"
done

echo "cuda_syntax_check: all CUDA translation units parsed"
