#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"
MODEL_DIR="${MODEL_DIR:-$ROOT/model/LFM2.5-230M}"
PROMPT="${PROMPT:-Explique CUDA em uma frase.}"
THREADS="${THREADS:-0}"
GROUP="${GROUP:-32}"
TOKENS="${TOKENS:-32}"

if [[ ! -x "$BUILD/lfm25-cpu-run" ]]; then
  echo "missing $BUILD/lfm25-cpu-run; build the CPU target first" >&2
  exit 1
fi
for file in model.safetensors tokenizer.json config.json; do
  if [[ ! -f "$MODEL_DIR/$file" ]]; then
    echo "missing $MODEL_DIR/$file; set MODEL_DIR" >&2
    exit 1
  fi
done

"$BUILD/lfm25-cpu-run" \
  --model "$MODEL_DIR" \
  --prompt "$PROMPT" \
  --max-new-tokens "$TOKENS" \
  --threads "$THREADS" \
  --cpu-q4-group "$GROUP" \
  --cpu-affinity "${AFFINITY:-compact}" \
  --memory-report
