#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?set MODEL to model.safetensors}"
: "${REFERENCE:?set REFERENCE to the exported reference directory}"
BUILD="${BUILD:-build-cpu}"
KV="${KV:-fp32}"
ISA="${ISA:-scalar}"
THREADS="${THREADS:-1}"
"${BUILD}/celeg-cpu-compare-reference" "$MODEL" "$REFERENCE" "$KV" "$ISA" "$THREADS"
