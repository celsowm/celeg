#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-./model/LFM2.5-230M}"
PROMPT="${PROMPT:-Explique CUDA em uma frase.}"
REQUESTS="${REQUESTS:-8}"
MAX_NEW="${MAX_NEW:-64}"
BINARY="${BINARY:-./build/celeg-concurrent-benchmark}"
exec "$BINARY" \
  "$MODEL_DIR/model.safetensors" \
  "$MODEL_DIR/tokenizer.json" \
  "$PROMPT" "$REQUESTS" "$MAX_NEW"
