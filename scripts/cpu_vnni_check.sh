#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build-cpu}"

if [[ ! -x "$BUILD/cpu_kernels_test" ]]; then
  echo "missing $BUILD/cpu_kernels_test" >&2
  exit 1
fi
"$BUILD/cpu_kernels_test"

OBJECT="$BUILD/CMakeFiles/lfm25_host.dir/src/cpu_kernels.cpp.o"
if [[ -f "$OBJECT" ]] && command -v objdump >/dev/null 2>&1; then
  disassembly="$(objdump -d "$OBJECT")"
  if ! grep -q 'vpdpbusd' <<<"$disassembly"; then
    echo "VPDPBUSD was not found in cpu_kernels.cpp machine code" >&2
    exit 1
  fi
  echo "cpu_vnni_check: VPDPBUSD present"
else
  echo "cpu_vnni_check: object/objdump unavailable; runtime test only"
fi
