#!/usr/bin/env bash
# Downloads a LiquidAI LFM2.5 checkpoint into a local directory.
#
# Usage:
#   ./scripts/download_model.sh [variant] [destination]
#
# variant:  "230m" (default), "1.2b-instruct", "1.2b-thinking", or "8b-a1b"
# destination: defaults to model/LFM2.5-<variant>
set -euo pipefail

VARIANT="${1:-230m}"
case "$VARIANT" in
    230m)
        REPO="LiquidAI/LFM2.5-230M"
        DEST="${2:-model/LFM2.5-230M}"
        MODEL_SHA256="f630da86651136c9aee893b04b7542007e90fdd718355358e57e7ecc31517cfd"
        ;;
    1.2b-instruct)
        REPO="LiquidAI/LFM2.5-1.2B-Instruct"
        DEST="${2:-model/LFM2.5-1.2B-Instruct}"
        MODEL_SHA256=""
        ;;
    1.2b-thinking)
        REPO="LiquidAI/LFM2.5-1.2B-Thinking"
        DEST="${2:-model/LFM2.5-1.2B-Thinking}"
        MODEL_SHA256=""
        ;;
    8b-a1b)
        REPO="LiquidAI/LFM2.5-8B-A1B"
        DEST="${2:-model/LFM2.5-8B-A1B}"
        MODEL_SHA256=""
        ;;
    *)
        echo "unknown variant: $VARIANT (use '230m', '1.2b-instruct', '1.2b-thinking', or '8b-a1b')" >&2
        exit 2
        ;;
esac

REVISION="${CELEG_MODEL_REVISION:-main}"
BASE="https://huggingface.co/${REPO}/resolve/${REVISION}"

mkdir -p "$DEST"

# Small helper: download one file from the HF resolve endpoint with resume.
download_file() {
  local file="$1"
  echo "downloading $file @ ${REVISION}"
  curl -fL --retry 5 --continue-at - "$BASE/$file?download=true" -o "$DEST/$file"
}

# Config + tokenizer artifacts are always present (sharded or not).
for file in tokenizer.json config.json generation_config.json tokenizer_config.json chat_template.jinja LICENSE; do
  download_file "$file" || true
done

# The weights may be a single `model.safetensors` or a sharded set described by
# `model.safetensors.index.json`. Detect sharding by probing the index file.
INDEX="$DEST/model.safetensors.index.json"
if download_file "model.safetensors.index.json"; then
  # Parse the list of shard filenames referenced by the index. Prefer jq, then
  # python3, then a tolerant grep over the "weight_map" values.
  SHARDS=$( (command -v jq >/dev/null 2>&1 && \
               jq -r '.weight_map | to_entries[].value' "$INDEX") || \
             (command -v python3 >/dev/null 2>&1 && \
               python3 - "$INDEX" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
seen = []
for v in data.get("weight_map", {}).values():
    if v not in seen:
        seen.append(v)
        print(v)
PY
             ) || \
             grep -oE '"[A-Za-z0-9_.-]+-?[0-9]{5,}-of-[0-9]{5,}\.safetensors"' "$INDEX" | tr -d '"' | sort -u )
  if [ -z "$SHARDS" ]; then
    echo "error: could not parse shard list from $INDEX" >&2
    exit 1
  fi
  echo "sharded checkpoint: downloading $(echo "$SHARDS" | wc -l | tr -d ' ') weight shards"
  echo "$SHARDS" | while IFS= read -r shard; do
    [ -n "$shard" ] && download_file "$shard"
  done
else
  # Non-sharded checkpoint.
  download_file "model.safetensors"
  if [ -n "$MODEL_SHA256" ]; then
    echo "$MODEL_SHA256  $DEST/model.safetensors" | sha256sum -c -
  fi
fi
echo "variant $VARIANT ready at $DEST"