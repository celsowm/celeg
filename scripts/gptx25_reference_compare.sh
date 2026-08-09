#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?set MODEL to the local GPT-X2.5 checkpoint path}"

HF_MODEL="${HF_MODEL:-AxiomicLabs/GPT-X2.5-135M}"
REFERENCE="${REFERENCE:-build/gptx25-reference}"
PROMPT="${PROMPT:-Once upon a time,}"
BUILD="${BUILD:-build-cpu}"
KV="${KV:-fp32}"
ISA="${ISA:-scalar}"
THREADS="${THREADS:-1}"

python scripts/export_cpu_reference.py \
  --model "$HF_MODEL" \
  --prompt "$PROMPT" \
  --output "$REFERENCE" \
  --raw

MODEL="$MODEL" \
REFERENCE="$REFERENCE" \
BUILD="$BUILD" \
KV="$KV" \
ISA="$ISA" \
THREADS="$THREADS" \
  scripts/cpu_reference_compare.sh
