#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$ROOT/build/celeg-run}"
MODEL="${MODEL:-$ROOT/model/LFM2.5-230M}"
PROMPT="${PROMPT:-Explique o que é um Tensor Core.}"
CONTEXT="${CONTEXT:-4096}"
DECODE_TOKENS="${DECODE_TOKENS:-128}"
WARMUP_TOKENS="${WARMUP_TOKENS:-16}"

if [[ ! -x "$BIN" ]]; then
    echo "binary not found or not executable: $BIN" >&2
    exit 2
fi
if [[ ! -f "$MODEL/model.safetensors" ]]; then
    echo "checkpoint not found under: $MODEL" >&2
    exit 2
fi

run_case() {
    local name="$1"
    shift
    echo "===== $name ====="
    "$BIN" \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --context "$CONTEXT" \
        --benchmark-warmup "$WARMUP_TOKENS" \
        --benchmark-decode "$DECODE_TOKENS" \
        "$@"
}

run_case strict-cublas \
    --gemm-backend cublas \
    --legacy-sampling

run_case fused-sampling-cublas \
    --gemm-backend cublas

run_case fast-cublas \
    --gemm-backend cublas \
    --fast-attention --fused-projections --fused-residuals

run_case fast-cublaslt \
    --gemm-backend cublaslt \
    --lt-workspace-mb 64 --lt-heuristics 8 \
    --fast-attention --fused-projections --fused-residuals

run_case fast-cublaslt-autotuned \
    --gemm-backend cublaslt \
    --lt-workspace-mb 64 --lt-heuristics 16 --lt-autotune \
    --fast-attention --fused-projections --fused-residuals

run_case int8-w8a16 \
    --weight-mode int8 --memory-report \
    --fast-attention --fused-projections --fused-residuals

run_case int8-w8a16-segmented \
    --weight-mode int8 --memory-report \
    --fast-attention --attention-mode segmented --attention-chunk-tokens 256 \
    --fused-projections --fused-residuals

run_case int4-w4a16 \
    --weight-mode int4 --memory-report \
    --fast-attention --fused-projections --fused-residuals

run_case int4-w4a16-auto-attention \
    --weight-mode int4 --memory-report \
    --fast-attention --attention-mode auto \
    --attention-auto-threshold 4096 --attention-chunk-tokens 256 \
    --fused-projections --fused-residuals

run_case int4-w4a16-kv-int8-auto-attention \
    --weight-mode int4 --kv-cache int8 --memory-report \
    --fast-attention --attention-mode auto \
    --attention-auto-threshold 4096 --attention-chunk-tokens 256 \
    --fused-projections --fused-residuals
