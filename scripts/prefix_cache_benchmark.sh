#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-./model/LFM2.5-230M}"
PROMPT="${PROMPT:-Explique CUDA em uma frase.}"
REPEATS="${REPEATS:-4}"
MAX_NEW="${MAX_NEW:-1}"
SUFFIX="${SUFFIX:-}"
BINARY="${BINARY:-./build/celeg-prefix-cache-benchmark}"
exec "$BINARY" \
  "$MODEL_DIR/model.safetensors" \
  "$MODEL_DIR/tokenizer.json" \
  "$PROMPT" "$REPEATS" "$MAX_NEW" "$SUFFIX"
