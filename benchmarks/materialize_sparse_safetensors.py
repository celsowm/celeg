#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
from pathlib import Path

from huggingface_hub import get_safetensors_metadata, snapshot_download


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

SMALL_MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "tokenization_*.py",
    "configuration_*.py",
]


def tensor_nbytes(dtype: str, shape: list[int]) -> int:
    if dtype not in DTYPE_BYTES:
        raise RuntimeError(f"unsupported safetensors dtype for sparse fixture: {dtype}")
    return math.prod(shape) * DTYPE_BYTES[dtype]


def write_sparse_safetensors(path: Path, tensors: list[tuple[str, str, list[int]]]) -> int:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, dtype, shape in tensors:
        size = tensor_nbytes(dtype, shape)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size

    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.truncate(8 + len(encoded) + offset)
    return offset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a sparse local safetensors checkpoint carrying real Hub metadata but no downloaded tensor payload."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    metadata = get_safetensors_metadata(
        args.model,
        revision=args.revision,
        token=token,
    )
    if metadata.sharded:
        raise RuntimeError("sparse fixture materialization currently requires one safetensors file")

    snapshot = Path(
        snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            token=token,
            allow_patterns=SMALL_MODEL_FILES,
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for source in snapshot.iterdir():
        if source.is_file():
            shutil.copy2(source, args.output / source.name)

    file_metadata = next(iter(metadata.files_metadata.values()))
    tensors = sorted(
        (
            name,
            str(getattr(info, "dtype", "")),
            [int(dim) for dim in getattr(info, "shape", [])],
        )
        for name, info in file_metadata.tensors.items()
    )
    logical_payload = write_sparse_safetensors(args.output / "model.safetensors", tensors)
    print(f"model={args.model}")
    print(f"revision={args.revision or '<default>'}")
    print(f"tensor_count={len(tensors)}")
    print(f"logical_payload_bytes={logical_payload}")
    print(f"fixture={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
