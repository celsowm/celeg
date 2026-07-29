#!/usr/bin/env python3
"""Inventory a GGUF file's tensors by quantization type, with decode traffic math.

A "Q4_K_M" file is not uniformly Q4_K -- llama.cpp's mixed recipes promote
selected tensors (commonly the output/embedding matrix, ffn_down, and attn_v) to
Q6_K. Which tensors got promoted decides whether a native-quantized kernel is
worth writing, because a single Q6_K tensor can dominate the weight traffic.

Concretely, for LFM2.5-230M-Q4_K_M this script showed token_embd.weight
[1024, 65536] is Q6_K -- and since that model ties the embedding to the output
head, that one tensor is 29% of all per-token decode weight traffic. Any plan
that assumed "Q4_K only" would have hit a hard ceiling.

Reads the GGUF header directly; no llama.cpp or gguf-py dependency.

Examples
--------
  python scripts/gguf_census.py model.gguf
  python scripts/gguf_census.py model.gguf --all      # every tensor
  python scripts/gguf_census.py model.gguf --grep ffn_down
"""
from __future__ import annotations

import argparse
import math
import struct
from collections import Counter
from pathlib import Path

# ggml_type ordinals -> (name, block_size, bytes_per_block). Block geometry is
# what turns an element count into real bytes moved.
GGML_TYPES = {
    0:  ("F32",    1,   4),
    1:  ("F16",    1,   2),
    2:  ("Q4_0",  32,  18),
    3:  ("Q4_1",  32,  20),
    6:  ("Q5_0",  32,  22),
    7:  ("Q5_1",  32,  24),
    8:  ("Q8_0",  32,  34),
    9:  ("Q8_1",  32,  36),
    10: ("Q2_K", 256,  84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    30: ("BF16",   1,   2),
}


def read_gguf(path: Path):
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise SystemExit(f"{path} is not a GGUF file (magic={magic!r})")
        struct.unpack("<I", f.read(4))  # version
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def rstr() -> str:
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", errors="replace")

        def skip(kind: int) -> None:
            fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            if kind == 8:
                rstr()
            elif kind == 9:
                elem = struct.unpack("<I", f.read(4))[0]
                count = struct.unpack("<Q", f.read(8))[0]
                for _ in range(count):
                    skip(elem)
            else:
                f.read(fixed[kind])

        for _ in range(n_kv):
            rstr()
            skip(struct.unpack("<I", f.read(4))[0])

        tensors = []
        for _ in range(n_tensors):
            name = rstr()
            ndim = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndim)]
            type_id = struct.unpack("<I", f.read(4))[0]
            struct.unpack("<Q", f.read(8))  # offset
            tensors.append((name, dims, type_id))
        return tensors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--all", action="store_true", help="list every tensor")
    ap.add_argument("--grep", help="list tensors whose name contains this substring")
    args = ap.parse_args()

    tensors = read_gguf(args.gguf)

    counts: Counter[str] = Counter()
    elements: Counter[str] = Counter()
    native: Counter[str] = Counter()
    for _, dims, type_id in tensors:
        name, block, block_bytes = GGML_TYPES.get(type_id, (f"type{type_id}", 1, 0))
        n = 1
        for d in dims:
            n *= d
        counts[name] += 1
        elements[name] += n
        native[name] += n // block * block_bytes if block_bytes else 0

    total_elems = sum(elements.values())
    total_bf16 = total_elems * 2
    total_native = sum(native.values())

    print(f"{args.gguf.name}: {len(tensors)} tensors\n")
    print(f"{'type':<8} {'tensors':>8} {'elements':>12} {'bits/wt':>8} "
          f"{'native MB':>11} {'as BF16 MB':>11}")
    for name in sorted(counts, key=lambda k: -elements[k]):
        bits = 8.0 * native[name] / elements[name] if elements[name] else 0.0
        print(f"{name:<8} {counts[name]:>8} {elements[name]:>12,} {bits:>8.3f} "
              f"{native[name] / 1e6:>11.1f} {elements[name] * 2 / 1e6:>11.1f}")
    print(f"{'TOTAL':<8} {len(tensors):>8} {total_elems:>12,} "
          f"{8.0 * total_native / total_elems:>8.3f} "
          f"{total_native / 1e6:>11.1f} {total_bf16 / 1e6:>11.1f}")

    print("\nDense decode reads every weight once per token, so the columns above "
          "are\nalso per-token traffic. Dequantizing everything to BF16 at load "
          f"costs\n{total_bf16 / total_native:.2f}x the memory traffic of keeping "
          "the native blocks.")
    print("Whether that matters depends on whether the GEMV path is actually "
          "bandwidth-bound --\ncheck with scripts/profile_decode.py before "
          "writing a native-quant kernel.")

    if args.all or args.grep:
        print()
        for name, dims, type_id in tensors:
            tname = GGML_TYPES.get(type_id, (f"type{type_id}",))[0]
            if args.grep and args.grep not in name:
                continue
            print(f"{tname:<6} {str(dims):<22} {name}")

    # Call out the biggest single tensors -- these decide kernel priorities.
    def native_bytes(dims: list[int], type_id: int) -> int:
        _, block, block_bytes = GGML_TYPES.get(type_id, ("", 1, 0))
        return math.prod(dims) // block * block_bytes

    print("\nlargest tensors by native bytes:")
    ranked = sorted(tensors, key=lambda t: -native_bytes(t[1], t[2]))
    for name, dims, type_id in ranked[:6]:
        tname = GGML_TYPES.get(type_id, (f"type{type_id}",))[0]
        share = 100.0 * native_bytes(dims, type_id) / total_native if total_native else 0.0
        print(f"  {tname:<6} {native_bytes(dims, type_id) / 1e6:>7.1f} MB native "
              f"({share:>4.1f}%)  {str(dims):<20} {name}")


if __name__ == "__main__":
    main()
