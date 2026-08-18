#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from huggingface_hub import get_safetensors_metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Hugging Face safetensors tensor names, dtypes, and shapes without downloading full weights."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    metadata = get_safetensors_metadata(
        args.model,
        revision=args.revision,
        token=token,
    )

    tensor_details: dict[str, dict[str, object]] = {}
    for filename, file_metadata in metadata.files_metadata.items():
        for name, info in file_metadata.tensors.items():
            tensor_details[name] = {
                "file": filename,
                "dtype": str(getattr(info, "dtype", "")),
                "shape": list(getattr(info, "shape", [])),
            }

    names = sorted(metadata.weight_map)
    records = [
        {
            "name": name,
            **tensor_details.get(name, {"file": metadata.weight_map[name]}),
        }
        for name in names
    ]
    payload = {
        "model": args.model,
        "revision": args.revision,
        "sharded": metadata.sharded,
        "tensor_count": len(names),
        "parameter_count": dict(metadata.parameter_count),
        "tensors": records,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(f"model={args.model}")
    print(f"revision={args.revision or '<default>'}")
    print(f"sharded={metadata.sharded}")
    print(f"tensor_count={len(names)}")
    print(f"parameter_count={dict(metadata.parameter_count)}")

    interesting = re.compile(
        r"(?:^|\.)(?:q_norm|k_norm|input_layernorm|post_attention_layernorm|"
        r"post_attn_norm|pre_feedforward_layernorm|post_feedforward_layernorm|"
        r"post_mlp_norm|norm|embed_tokens|lm_head)(?:\.|$)"
    )
    interesting_names = [name for name in names if interesting.search(name)]
    print("interesting_tensor_names:")
    for name in interesting_names:
        detail = tensor_details.get(name, {})
        print(
            f"  {name} dtype={detail.get('dtype', '')} "
            f"shape={detail.get('shape', '')} file={detail.get('file', metadata.weight_map[name])}"
        )

    layer_zero = [
        name
        for name in names
        if ".layers.0." in name or name.startswith("layers.0.")
    ]
    print("layer_0_tensor_names:")
    for name in layer_zero:
        detail = tensor_details.get(name, {})
        print(
            f"  {name} dtype={detail.get('dtype', '')} "
            f"shape={detail.get('shape', '')} file={detail.get('file', metadata.weight_map[name])}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
