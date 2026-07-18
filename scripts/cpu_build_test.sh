#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build-cpu-test}"
rm -rf "$BUILD"
cmake -S "$ROOT" -B "$BUILD" \
  -DLFM_ENABLE_CUDA=OFF \
  -DLFM_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"${JOBS:-2}"
ctest --test-dir "$BUILD" --output-on-failure -j"${JOBS:-2}"
"$BUILD/lfm25-cpu-run" --print-cpu
"$BUILD/lfm25-cpu-kernel-benchmark" 256 256 2 2 32 1 auto none
