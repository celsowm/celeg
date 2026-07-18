#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

fail() {
    echo "architecture_check: $*" >&2
    exit 1
}

if grep -q 'lfm/model.hpp' "$ROOT/include/lfm/paged_kv.hpp"; then
    fail "paged_kv.hpp must not depend on LfmModel"
fi
if grep -q 'friend class PhysicalPagedKvCache' "$ROOT/include/lfm/model.hpp"; then
    fail "LfmModel must not expose internals to PhysicalPagedKvCache"
fi
if grep -R -q 'import_from_model' "$ROOT/include" "$ROOT/src"; then
    fail "obsolete contiguous-to-paged import dependency returned"
fi
if ! grep -q 'std::unique_ptr<Impl> impl_' "$ROOT/include/lfm/model.hpp"; then
    fail "LfmModel must remain a PIMPL facade"
fi
if grep -Eq '^#include.*(cuda|cublas|safetensors)' "$ROOT/include/lfm/model.hpp"; then
    fail "public LfmModel facade must not expose CUDA or checkpoint internals"
fi
for interface in LfmInferenceSession LfmDiagnostics LfmPersistence; do
    if ! grep -q "class $interface" "$ROOT/include/lfm/model.hpp"; then
        fail "missing narrow model interface: $interface"
    fi
done
if ! grep -q 'ExecutionPlan plan_' "$ROOT/include/lfm/detail/model_impl.hpp"; then
    fail "model implementation must own a validated ExecutionPlan"
fi
if ! grep -q 'std::variant<AttentionLayer, ConvolutionLayer>' \
        "$ROOT/include/lfm/detail/model_impl.hpp"; then
    fail "Layer must be a typed attention/convolution variant"
fi
if grep -q 'bool attention' "$ROOT/include/lfm/detail/model_impl.hpp"; then
    fail "boolean-discriminated Layer returned"
fi
if ! grep -q 'std::unique_ptr<Impl> impl_' "$ROOT/include/lfm/concurrent.hpp"; then
    fail "ConcurrentEngine must remain a PIMPL facade"
fi
if grep -Eq 'mutex|condition_variable|unordered_map|PhysicalPagedKvCache|PackedDecodeExecutor' \
        "$ROOT/include/lfm/concurrent.hpp"; then
    fail "public ConcurrentEngine facade leaks runtime implementation details"
fi
for component in request_registry batch_planner engine_worker; do
    if [[ ! -f "$ROOT/include/lfm/detail/${component}.hpp" ]]; then
        fail "missing concurrent component: $component"
    fi
done
if ! grep -q 'add_library(lfm25_host' "$ROOT/CMakeLists.txt"; then
    fail "host-only target is missing"
fi
if ! grep -q 'check_language(CUDA)' "$ROOT/CMakeLists.txt"; then
    fail "CUDA must remain optional for CPU-only builds"
fi
if ! grep -q 'add_library(lfm25_cpu_core' "$ROOT/CMakeLists.txt"; then
    fail "CPU runtime target is missing"
fi
if ! grep -q 'add_library(lfm25_cpu SHARED' "$ROOT/CMakeLists.txt"; then
    fail "CPU C ABI library is missing"
fi
if grep -Eq '^#include.*(cuda|cublas)' "$ROOT/include/lfm/cpu_model.hpp"; then
    fail "public CPU model facade must not expose CUDA"
fi
if grep -Eq '^#include.*(cuda|cublas)' "$ROOT/include/lfm/cpu_c_api.h"; then
    fail "CPU C ABI must not expose CUDA"
fi
for component in cpu_isa cpu_thread_pool cpu_quantization cpu_kernels cpu_model; do
    if [[ ! -f "$ROOT/include/lfm/${component}.hpp" ]]; then
        fail "missing CPU component: $component"
    fi
done
if ! grep -q 'requested ISA is detected by the API but its native kernel is not implemented' \
        "$ROOT/src/cpu_model.cpp"; then
    fail "advanced ISA detection must not silently claim an unimplemented kernel"
fi

if grep -Eq '^#include.*(cuda|cublas)' "$ROOT/include/lfm/cpu_concurrent.hpp"; then
    fail "public CPU concurrent facade must not expose CUDA"
fi
for component in cpu_packed cpu_concurrent; do
    if [[ ! -f "$ROOT/include/lfm/${component}.hpp" ]]; then
        fail "missing CPU serving component: $component"
    fi
done
if ! grep -q 'clone_session' "$ROOT/include/lfm/cpu_model.hpp"; then
    fail "CPU sessions must share immutable model weights through clone_session"
fi
if ! grep -q 'std::shared_ptr<Shared> shared' "$ROOT/src/cpu_model_internal.hpp"; then
    fail "CPU model implementation must separate shared weights from session state"
fi
if ! grep -q 'shared.linear.gemm' "$ROOT/src/cpu_packed.cpp"; then
    fail "CPU packed executor must use batched GEMM"
fi
if ! grep -q 'CpuPackedExecutor executor' "$ROOT/src/cpu_concurrent.cpp"; then
    fail "CPU scheduler must route work through CpuPackedExecutor"
fi
if ! grep -q 'LFM25_CPU_C_API_VERSION 5u' "$ROOT/include/lfm/cpu_c_api.h"; then
    fail "CPU C API v5 is missing"
fi
if ! grep -q 'lfm25_cpu_model_options_v4_init' "$ROOT/include/lfm/cpu_c_api.h"; then
    fail "CPU C API v4 compatibility entry points are missing"
fi
if ! grep -q 'lfm25_cpu_engine_options_v3_init' "$ROOT/include/lfm/cpu_c_api.h"; then
    fail "CPU concurrent C API v3 is missing"
fi

if [[ ! -f "$ROOT/include/lfm/cpu_paged_kv.hpp" ]] || [[ ! -f "$ROOT/src/cpu_paged_kv.cpp" ]]; then
    fail "CPU physical paged KV implementation is missing"
fi
if ! grep -q 'forward_chunk' "$ROOT/src/cpu_model_internal.hpp"; then
    fail "CPU layer-major chunked prefill is missing"
fi
if ! grep -q 'cpu_gqa_decode_paged' "$ROOT/src/cpu_paged_kv.cpp"; then
    fail "CPU paged attention kernel is missing"
fi
if ! grep -q 'long_prefill_chunk_tokens' "$ROOT/include/lfm/cpu_concurrent.hpp"; then
    fail "CPU scheduler lacks long-prompt chunk policy"
fi

python - "$ROOT/src/packed.cu" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
lines = path.read_text().splitlines()
limits = {
    "std::vector<int32_t> decode": 80,
    "void prefill_step": 90,
    "void run_transformer_layers": 60,
    "void run_paged_attention_cache": 90,
}
for signature, limit in limits.items():
    matches = [i for i, line in enumerate(lines) if signature in line]
    if not matches:
        raise SystemExit(f"architecture_check: missing packed stage {signature}")
    start = matches[0]
    depth = 0
    opened = False
    for end in range(start, len(lines)):
        depth += lines[end].count("{") - lines[end].count("}")
        opened = opened or "{" in lines[end]
        if opened and depth == 0:
            length = end - start + 1
            if length > limit:
                raise SystemExit(
                    f"architecture_check: {signature} grew to {length} lines (limit {limit})"
                )
            break
    else:
        raise SystemExit(f"architecture_check: unterminated method {signature}")
PY

echo "architecture_check: boundaries passed"
