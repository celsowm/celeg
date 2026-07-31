#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$ROOT/build/celeg-run}"
COMPARE="${COMPARE:-$ROOT/build/celeg-compare-logits}"
MODEL="${MODEL:-$ROOT/model/LFM2.5-230M}"
PROMPT="${PROMPT:-Explique CUDA em uma frase.}"
CONTEXT="${CONTEXT:-4096}"
OUT="${OUT:-${TMPDIR:-/tmp}/celeg-kv-parity}"

if [[ ! -x "$BIN" || ! -x "$COMPARE" ]]; then
    echo "build celeg-run and celeg-compare-logits first" >&2
    exit 2
fi
if [[ ! -f "$MODEL/model.safetensors" ]]; then
    echo "checkpoint not found under: $MODEL" >&2
    exit 2
fi
mkdir -p "$OUT"

COMMON=(--model "$MODEL" --prompt "$PROMPT" --context "$CONTEXT"
        --max-new-tokens 0 --no-cuda-graph --fast-attention)

"$BIN" "${COMMON[@]}" --kv-cache bf16 --dump-logits "$OUT/kv-bf16.f32" --memory-report
"$BIN" "${COMMON[@]}" --kv-cache int8 --dump-logits "$OUT/kv-int8.f32" --memory-report

"$COMPARE" "$OUT/kv-bf16.f32" "$OUT/kv-int8.f32"
