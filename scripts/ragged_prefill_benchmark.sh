#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:-$ROOT/model/LFM2.5-230M/model.safetensors}"
TOKENIZER="${TOKENIZER:-$ROOT/model/LFM2.5-230M/tokenizer.json}"
PROMPT="${PROMPT:-Explique como CUDA executa kernels e gerencia memória compartilhada.}"
REQUESTS="${REQUESTS:-16}"
MAX_NEW="${MAX_NEW:-1}"
BINARY="${BINARY:-$ROOT/build/lfm25-concurrent-benchmark}"

exec "$BINARY" "$MODEL" "$TOKENIZER" "$PROMPT" "$REQUESTS" "$MAX_NEW"
