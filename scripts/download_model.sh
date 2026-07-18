#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-model/LFM2.5-230M}"
REVISION="${LFM_MODEL_REVISION:-35d5e77449d3a662bc0f7c7b79e1eded6b77006e}"
BASE="https://huggingface.co/LiquidAI/LFM2.5-230M/resolve/${REVISION}"
MODEL_SHA256="f630da86651136c9aee893b04b7542007e90fdd718355358e57e7ecc31517cfd"

mkdir -p "$DEST"
for file in model.safetensors tokenizer.json config.json generation_config.json tokenizer_config.json chat_template.jinja LICENSE; do
  echo "downloading $file @ $REVISION"
  curl -fL --retry 5 --continue-at - "$BASE/$file?download=true" -o "$DEST/$file"
done

echo "$MODEL_SHA256  $DEST/model.safetensors" | sha256sum -c -
