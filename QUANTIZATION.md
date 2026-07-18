# Weight-only quantization

## Available modes

The backend supports three storage modes for two-dimensional linear matrices:

```text
--weight-mode bf16
--weight-mode int8
--weight-mode int4
```

Activations, KV cache, convolution state, normalization vectors and the
short-convolution filter remain BF16. Quantized matrix kernels accumulate in
FP32 and convert the result to BF16.

## W8A16

INT8 uses symmetric per-output-row quantization:

```text
scale[row] = max(abs(weight[row])) / 127
q[row,col] = clamp(round(weight[row,col] / scale[row]), -127, 127)
```

One signed byte is stored per value and one FP32 scale per row.

## W4A16

INT4 uses the analogous symmetric range `[-7, 7]`:

```text
scale[row] = max(abs(weight[row])) / 7
q[row,col] = clamp(round(weight[row,col] / scale[row]), -7, 7)
```

Two two's-complement nibbles are stored per byte, low nibble first. Odd-width
rows reserve the unused high nibble of the final byte. One FP32 scale is kept
per row.

The packed row stride is:

```text
packed_cols = ceil(input_features / 2)
```

Row slicing adjusts pointers by `row_offset * packed_cols`, so concatenated QKV
and gate/up matrices remain independently sliceable.

## Tied embedding and LM head

`model.embed_tokens.weight` is tied to the LM head. In quantized modes the
single stored matrix is packed once:

- embedding lookup dequantizes only the selected vocabulary row;
- the LM head executes the same W8A16 or W4A16 matrix kernel as other linears.

No BF16 duplicate is retained for quantized two-dimensional matrices.

## Expected trade-offs

### INT8

- roughly half the raw matrix bytes of BF16;
- smaller quantization error than INT4;
- custom warp-oriented kernel may improve memory-bound batch-one decode;
- generic prefill path may lose to tensor-core BF16 GEMM.

### INT4

- roughly one quarter of the raw matrix bytes of BF16;
- lowest matrix bandwidth and VRAM use;
- larger quantization error and potentially lower sequence agreement;
- nibble extraction adds integer instructions;
- current kernel does not yet use native low-bit tensor-core MMA instructions.

The row scales slightly reduce the ideal compression ratio, especially for
small matrices.

## Compare modes

```bash
MODEL=./model/LFM2.5-230M \
PROMPT="Explique CUDA em uma frase." \
./scripts/compare_weight_modes.sh
```

The comparison is useful for engineering validation, but a representative
prompt suite and task-level metrics are needed before choosing a production
mode.

## Source validation

Included tests cover:

- BF16 bit conversion;
- INT8 row quantization and dequantization;
- INT4 odd-width packing and row offsets;
- zero rows and scale behavior;
- rejection of non-finite BF16 inputs and overflow-safe sizing;
- known-answer W8A16 and W4A16 CUDA linears;
- INT8 and INT4 embedding lookup.

## INT8 KV cache — v0.0.8

KV precision is selected independently from weight precision:

```bash
--kv-cache bf16|int8
```

For each token and KV head, K and V receive separate symmetric scales:

```text
scale = max(abs(vector)) / 127
q[i] = clamp(round(vector[i] / scale), -127, 127)
```

The cache layout remains `[token][kv_head][head_dim]`. Scale arrays use
`[token][kv_head]`. Attention kernels multiply INT8 values by the scale while
loading, so no full BF16 cache is materialized.

For `head_dim=64`, each K or V vector changes from 128 bytes in BF16 to 68 bytes
in INT8 plus scale. That is a 46.875% reduction. Across six attention layers,
the theoretical cache changes from roughly 12 KiB per token to 6.375 KiB per
token.

This mode introduces quantization error at every stored key and value. Use
`scripts/compare_kv_modes.sh` and workload-specific generation tests before
selecting it as a default.


## CPU groupwise Q4

The CPU backend uses signed Q4 values in `[-7, 7]`, packed two per byte, with
one BF16 scale for each row/group of 32 or 64 input columns. Scalar, AVX2 and
NEON paths consume FP32 activations. AVX-VNNI and AVX-512 VNNI dynamically
quantize each input group to signed INT8 with one FP32 activation scale and one
precomputed group sum.

For VNNI, the four-bit sign bit is toggled so the unsigned byte operand equals
`q4 + 8`. `VPDPBUSD` then computes unsigned-byte × signed-byte products, followed
by the exact `-8 * sum(q8)` correction and FP32 scaling. The Q4 weight format
and `.lfmpack` representation remain unchanged from v0.0.16.
