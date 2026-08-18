#!/usr/bin/env python3
"""Materialize a sparse Hugging Face Safetensors checkpoint from its real header."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_url, snapshot_download


def read_prefix(url: str, size: int) -> bytes:
    response = requests.get(
        url,
        headers={"Range": f"bytes=0-{size - 1}"},
        stream=True,
        timeout=120,
    )
    response.raise_for_status()
    data = bytearray()
    for chunk in response.iter_content(chunk_size=min(1 << 20, size)):
        if not chunk:
            continue
        remaining = size - len(data)
        data.extend(chunk[:remaining])
        if len(data) == size:
            break
    response.close()
    if len(data) != size:
        raise RuntimeError(
            f"expected {size} prefix bytes, received {len(data)}"
        )
    return bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--filename", default="model.safetensors")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    info = HfApi().model_info(
        args.repo,
        revision=args.revision,
        files_metadata=True,
    )
    revision = info.sha
    if not revision:
        raise RuntimeError("Hugging Face did not return a resolved revision")

    snapshot_download(
        repo_id=args.repo,
        revision=revision,
        local_dir=output,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.jinja",
            "*.py",
        ],
    )

    sibling = next(
        (entry for entry in info.siblings if entry.rfilename == args.filename),
        None,
    )
    if sibling is None or sibling.size is None:
        raise RuntimeError(f"cannot determine remote size for {args.filename}")

    url = hf_hub_url(
        repo_id=args.repo,
        filename=args.filename,
        revision=revision,
    )
    prefix8 = read_prefix(url, 8)
    header_size = int.from_bytes(prefix8, "little", signed=False)
    if header_size <= 0 or header_size > 64 * 1024 * 1024:
        raise RuntimeError(f"invalid safetensors header size: {header_size}")
    prefix = read_prefix(url, 8 + header_size)

    target = output / args.filename
    with target.open("wb") as handle:
        handle.write(prefix)
        handle.truncate(int(sibling.size))

    print(f"repo={args.repo}")
    print(f"revision={revision}")
    print(f"file={args.filename}")
    print(f"remote_size={sibling.size}")
    print(f"header_size={header_size}")
    print(f"materialized={target}")


if __name__ == "__main__":
    main()
