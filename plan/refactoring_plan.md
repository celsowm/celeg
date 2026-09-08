# Comprehensive Kernel DRY & SOLID Refactoring Plan

This document outlines the systematic refactoring plan to eliminate DRY violations and SOLID antipatterns across CUDA, CPU, and Metal kernel implementations in `celeg`.

## 1. Guiding Principles & Hard Constraints (`AGENTS.md`)
- **Doxygen Only:** All new or updated documentation must use Doxygen (`///`, `/** ... */`). No plain prose `//` or `/\* \*\//` comments.
- **No Backward-Compatibility Shims:** Fully replace old access patterns and interfaces. Do not keep legacy shortcuts, `impl_->` member access, or shims "for compatibility". Delete the old path.
- **Boundary Rules:** 
  - No LFM types in generic runtime directories.
  - No CUDA types in backend-neutral model headers.
  - No new `.inl` aggregation files.
  - No optional interface methods throwing "not supported" by default.
  - No architecture switch statements inside backend operator code.
  - No new quantized formats without quality and performance test coverage (`tests/numerical_compare_test.cpp`).

---

## Phase 1: CUDA Kernel DRY & SOLID Unification

### 1.1 Attention Reduction & Softmax Templates (`src/backend/cuda/kernels/attention_common.cuh`)
- **DRY:** Factor out the strict 3-pass softmax skeleton (`attention_dense.cuh`, `attention_paged.cuh`, `attention_batch_ptrs.cuh`, `attention_block_sparse*.cu`, `attention_dynamic_sparse.cu`) and the online single-pass rescale skeleton into reusable device templates (`strict_three_pass<ScoreFn, ValueFn>` and `online_decode_loop`).
- **SRP/ISP:** Extract segment partial and reduce kernels into clean single-responsibility device helpers. Eliminate duplicated prologue boilerplate via `GqaThreadCtx` inline context builders.

### 1.2 Unified Reductions & RoPE Header (`src/backend/cuda/kernels/reductions.cuh` & `rope_common.cuh`)
- **DRY:** Replace 5 spellings of warp/block reductions (`warp_sum`, `gemv_warp_sum`, `warp_reduce_sum`, local `warp_sum`) with a single `reductions.cuh`.
- **OCP:** Merge `rope.cuh`, `qkv_rope.cuh`, and `rope_pairing.cu` frequency/scaling logic into `rope_common.cuh`. Fix the kind-5 `>` vs `>=` drift and eliminate the duplicated `_for_pairing` functions. Add `#pragma once` guards to all `.cuh` headers.

### 1.3 GGUF Block Structs & Embedding/KV Unification (`src/celeg/checkpoint/gguf_blocks.hpp`)
- **DIP/SRP:** Move duplicated `BlockQ4K`/`BlockQ6K` structs and scale/min decoders from `gguf.cu` and `mmq.cu` into canonical checkpoint headers.
- **OCP:** Unify embedding value/device/batch triplets across dtypes using templated `embedding_gather<DecodeToken, DecodeValue>` and a `dispatch_gguf_type` helper.

---

## Phase 2: CPU Kernel SOLID & DRY Refactoring

### 2.1 Compiler-Boundary Unification (`src/backend/cpu/kernels/quantized_dot.cpp` & `quantized_dot_avx2_msvc.cpp`)
- **DRY:** Eliminate the ~70-line duplication between GCC and MSVC Q4 dot products by extracting a shared `q4_unpack` / `dot32_epi32` device header.
- **OCP/DIP:** Replace static `g_has_avx2_fma` globals and fake-AVX2 scalar stubs with proper capability-table injection via `CpuKernelBackend`.

### 2.2 Breaking Up the `CpuLinearEngine` God Object (`src/backend/cpu/kernels/linear.cpp`)
- **SRP:** Split `CpuLinearEngine` (~713 lines) into single-responsibility format engines (`Q4Engine`, `Int8Engine`, `Bf16Engine`, `GgufEngine`) and a `SegmentRouter`.
- **LSP:** Replace nullable function pointers (`Q4Q8DotFunction`) with the Null-Object pattern. Enforce validation preconditions in constructor paths rather than dispatch-time exceptions.
- **ISP:** Delete the fat umbrella header `kernels.hpp` (~178 lines). Replace with segregated headers (`elementwise.hpp`, `rope.hpp`, `attention.hpp`, `convolution.hpp`, `gated_delta.hpp`, `q4_dot.hpp`).

---

## Phase 3: Metal Kernel DRY & SOLID Cleanups

### 3.1 Shared Metal Decoding & Attention Infrastructure (`src/backend/metal/kernels/inference/decode_common.metal`)
- **DRY:** Consolidate triplicated scale/min decoders, RoPE application, RMSNorm, and SiLU/Softplus activations into a single shared header included by all Metal kernels.
- **OCP:** Unify the 5 attention-span variants (`attention_half`, `one_exp`, `decode_fa`, `gqa_fused`, `tiled_simdgroup`) into a single parameterized `attention_span<KV, Queries, Block, OneExp, Bias>` template.

### 3.2 Matvec and Tensor Policy Consolidation (`src/backend/metal/kernels/inference/matvec_core.metal` & `tensor_matmul.metal`)
- **DRY:** Replace 10 boilerplate wrappers in `matvec_kernels.metal` and redundant cores across `matvec_q4k.metal`, `matvec_q6k.metal`, and `matvec_q80.metal` with a `DEFINE_MATVEC_KERNEL` macro and unified core template.
- **SRP:** Restructure `tensor.metal`, `tensor_q4k_relaxed.metal`, `tensor_dense_relaxed.metal`, and `tensor_q4k_static_stage128.metal` into a single policy-driven `tensor_matmul.metal`.

---

## Phase 4: Verification & Validation Loop

After implementing changes across backends, execute the standard verification workflow:
```text
python scripts/dev.py verify --backend cpu
python scripts/dev.py verify --backend cuda
python scripts/dev.py smoke --backend cuda
```
Run numerical comparison and benchmark harnesses to ensure zero regression:
```text
python benchmarks/run_manifest.py benchmarks/manifests/reference_230m.json
```
