#!/usr/bin/env bash
set -euo pipefail
MODEL=${MODEL:-./model/LFM2.5-230M/model.safetensors}
TOKENS=${TOKENS:-1024}
KV=${KV:-bf16}
ISA=${ISA:-auto}
THREADS=${THREADS:-0}
for PAGE in 16 32 64; do
  for CHUNK in 64 128 256 512; do
    echo "=== page=${PAGE} chunk=${CHUNK} ==="
    ./build-cpu/lfm25-cpu-prefill-benchmark \
      "$MODEL" "$TOKENS" "$CHUNK" "$PAGE" "$KV" "$ISA" "$THREADS"
  done
done
