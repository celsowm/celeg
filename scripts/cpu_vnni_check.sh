#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build-cpu}"

if [[ ! -x "$BUILD/cpu_linear_kernels_test" ]]; then
  echo "missing $BUILD/cpu_linear_kernels_test" >&2
  exit 1
fi
"$BUILD/cpu_linear_kernels_test"

OBJECT="$(find "$BUILD/CMakeFiles/celeg_cpu_backend.dir/src/backend/cpu/kernels" \
  -name 'quantized_dot.cpp.o' -o -name 'quantized_dot.cpp.obj' | head -n 1)"
if [[ -f "$OBJECT" ]] && command -v objdump >/dev/null 2>&1; then
  disassembly="$(objdump -d "$OBJECT")"
  if ! grep -q 'vpdpbusd' <<<"$disassembly"; then
    echo "VPDPBUSD was not found in quantized_dot.cpp machine code" >&2
    exit 1
  fi
  echo "cpu_vnni_check: VPDPBUSD present"
else
  echo "cpu_vnni_check: object/objdump unavailable; runtime test only"
fi
