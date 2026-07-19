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

REVISION="${LFM_MODEL_REVISION:-main}"
BASE="https://huggingface.co/${REPO}/resolve/${REVISION}"

mkdir -p "$DEST"
for file in model.safetensors tokenizer.json config.json generation_config.json tokenizer_config.json chat_template.jinja LICENSE; do
  echo "downloading $file @ ${REVISION}"
  curl -fL --retry 5 --continue-at - "$BASE/$file?download=true" -o "$DEST/$file"
done

if [ -n "$MODEL_SHA256" ]; then
  echo "$MODEL_SHA256  $DEST/model.safetensors" | sha256sum -c -
fi
echo "variant $VARIANT ready at $DEST"
