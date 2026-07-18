#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
RUN_GPU_TESTS="${RUN_GPU_TESTS:-auto}"

HAS_GPU=OFF
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    HAS_GPU=ON
fi

if [[ "$RUN_GPU_TESTS" == auto ]]; then
    RUN_GPU_TESTS="$HAS_GPU"
fi
if [[ -z "$CUDA_ARCHITECTURES" ]]; then
    CUDA_ARCHITECTURES=$([[ "$HAS_GPU" == ON ]] && echo native || echo 80)
fi

cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
    -DLFM_BUILD_TESTS=ON \
    -DLFM_RUN_CUDA_TESTS="$RUN_GPU_TESTS"
cmake --build "$BUILD_DIR" --parallel
ctest --test-dir "$BUILD_DIR" --output-on-failure

echo "cuda_build_test: build complete (GPU tests: $RUN_GPU_TESTS)"
