#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-./model/LFM2.5-230M}"
PROMPT="${PROMPT:-Explique como a CPU executa uma rede neural.}"
MAX_NEW="${MAX_NEW:-32}"
THREADS="${THREADS:-0}"
KV="${KV:-bf16}"
ISA="${ISA:-auto}"
BUILD="${BUILD:-./build-cpu}"

for requests in 1 2 4 8 16; do
  echo "=== requests=${requests} kv=${KV} isa=${ISA} ==="
  "${BUILD}/celeg-cpu-concurrent-benchmark" \
    "${MODEL}" "${PROMPT}" "${requests}" "${MAX_NEW}" \
    "${THREADS}" "${KV}" "${ISA}"
done
