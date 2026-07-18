#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build-cpu-sanitize}"
rm -rf "$BUILD"
cmake -S "$ROOT" -B "$BUILD" \
  -DLFM_ENABLE_CUDA=OFF \
  -DLFM_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build "$BUILD" -j"${JOBS:-2}" --target \
  cpu_isa_test cpu_thread_pool_test cpu_topology_test cpu_quantization_test cpu_kernels_test \
  cpu_kv_cache_test cpu_paged_kv_test cpu_parallel_attention_test cpu_prefix_cache_test cpu_numa_test cpu_concurrent_metrics_test cpu_model_options_test \
  cpu_c_api_header_test cpu_c_api_smoke_test
export ASAN_OPTIONS="detect_leaks=1:halt_on_error=1"
export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"
for test in cpu_isa_test cpu_thread_pool_test cpu_topology_test cpu_quantization_test \
            cpu_kernels_test cpu_kv_cache_test cpu_paged_kv_test cpu_parallel_attention_test cpu_prefix_cache_test cpu_numa_test cpu_concurrent_metrics_test \
            cpu_model_options_test cpu_c_api_header_test cpu_c_api_smoke_test; do
  "$BUILD/$test"
done
echo "cpu_sanitizer_test: selected CPU tests passed"
