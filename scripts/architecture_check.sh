#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

fail() {
    echo "architecture_check: $*" >&2
    exit 1
}

require() {
    local pattern="$1"
    local path="$2"
    grep -q -- "$pattern" "$path" || fail "missing $pattern in ${path#$ROOT/}"
}

for obsolete in "$ROOT/include/lfm/api/cpu.h" "$ROOT/include/lfm/api/cuda.h" \
                "$ROOT/include/lfm/backend/cpu/packed.hpp"; do
    [[ ! -e "$obsolete" ]] || fail "obsolete public surface remains: ${obsolete#$ROOT/}"
done

require 'typedef enum lfm25_backend' "$ROOT/include/lfm/api.h"
require 'add_library(lfm25 SHARED' "$ROOT/CMakeLists.txt"

if rg -n 'friend class CpuPackedExecutor|owner_->impl_|new Impl|std::endl' \
        "$ROOT/include" "$ROOT/src" "$ROOT/tests"; then
    fail "forbidden encapsulation or stream pattern found"
fi
if rg -n 'using namespace' "$ROOT/tests"; then
    fail "test translation units must not use using namespace"
fi
if rg -n '\bassert\s*\(' "$ROOT/tests"; then
    fail "tests must use tests/support/assertions.hpp instead of assert"
fi
if rg -n 'reinterpret_cast<[^>]*char\s*\*>\s*\([^)]*&' \
        "$ROOT/src" "$ROOT/tests"; then
    fail "POD serialization through pointer casts is forbidden"
fi

# Only the CpuModel implementation boundary may name its opaque type.
impl_uses=$(rg -l 'CpuModel::Impl' "$ROOT/src/backend/cpu" "$ROOT/include/lfm/backend/cpu" \
    | sed "s#^$ROOT/##" \
    | grep -v -E '^(src/backend/cpu/model\.cpp|src/backend/cpu/packed\.cpp|src/backend/cpu/detail/model_internal\.hpp)$' || true)
[[ -z "$impl_uses" ]] || fail "CpuModel::Impl escaped its implementation boundary: $impl_uses"

echo "architecture_check: boundaries passed"
