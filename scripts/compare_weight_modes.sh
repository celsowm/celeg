#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$ROOT/build/celeg-run}"
COMPARE="${COMPARE:-$ROOT/build/celeg-compare-logits}"
MODEL="${MODEL:-$ROOT/model/LFM2.5-230M}"
PROMPT="${PROMPT:-Explique CUDA em uma frase.}"
CONTEXT="${CONTEXT:-4096}"
OUT="${OUT:-${TMPDIR:-/tmp}/celeg-weight-parity}"

if [[ ! -x "$BIN" ]]; then
    echo "binary not found or not executable: $BIN" >&2
    exit 2
fi
if [[ ! -x "$COMPARE" ]]; then
    echo "comparison utility not found or not executable: $COMPARE" >&2
    exit 2
fi
if [[ ! -f "$MODEL/model.safetensors" ]]; then
    echo "checkpoint not found under: $MODEL" >&2
    exit 2
fi
mkdir -p "$OUT"

"$BIN" --model "$MODEL" --prompt "$PROMPT" --context "$CONTEXT" \
    --max-new-tokens 0 --no-cuda-graph \
    --dump-logits "$OUT/bf16.f32" --weight-mode bf16

"$BIN" --model "$MODEL" --prompt "$PROMPT" --context "$CONTEXT" \
    --max-new-tokens 0 --no-cuda-graph \
    --dump-logits "$OUT/int8.f32" --weight-mode int8

"$BIN" --model "$MODEL" --prompt "$PROMPT" --context "$CONTEXT" \
    --max-new-tokens 0 --no-cuda-graph \
    --dump-logits "$OUT/int4.f32" --weight-mode int4

echo "===== BF16 vs INT8 ====="
"$COMPARE" "$OUT/bf16.f32" "$OUT/int8.f32"
echo "===== BF16 vs INT4 ====="
"$COMPARE" "$OUT/bf16.f32" "$OUT/int4.f32"
