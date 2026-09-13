# Celeg

**Celeg is an architecture-agnostic, backend-agnostic native inference runtime for modern generative models.**

It is written in C++20 and designed around a simple principle:

> **Models describe computation. Backends execute it. Neither should own the runtime.**

Celeg is not built around a particular model family, tensor format, accelerator, or serving stack. Model architectures are translated into common runtime contracts and execution programs, while CPU, CUDA, and Metal implement those contracts independently.

The goal is to make adding a new architecture mostly a matter of describing its semantics and bindings — not rewriting an inference engine.

Celeg currently runs transformer, hybrid attention/recurrent, state-space, Mixture-of-Experts, multimodal-aware, and quantized model families across native CPU, NVIDIA CUDA, and Apple Metal execution paths.

---

## Why Celeg

Most inference runtimes gradually accumulate model-specific execution paths:

```text
if model == X:
    ...
else if model == Y:
    ...
```

Celeg is deliberately designed against that model.

A model family may define:

- layer topology
- tensor bindings
- attention geometry
- recurrent/state-space behavior
- normalization
- positional encoding
- feed-forward structure
- expert routing
- output gating
- chat/template metadata

But execution belongs to reusable backend operators.

The intended architecture is:

```text
                  Checkpoint
                      │
          ┌───────────┴───────────┐
          │                       │
     Safetensors                 GGUF
          │                       │
          └───────────┬───────────┘
                      │
               Model discovery
                      │
               Architecture
                 description
                      │
               Runtime program
                      │
        ┌─────────────┼─────────────┐
        │             │             │
       CPU           CUDA          Metal
        │             │             │
        └─────────────┴─────────────┘
                      │
              Inference engine
                      │
          ┌───────────┼───────────┐
          │           │           │
         CLI       C API       HTTP API
```

The backend does not need to know that a graph came from LFM, Granite, Agnes, Gemma, MiniCPM, or another architecture.

The architecture layer does not need to know whether its operators will execute using AVX, CUDA kernels, Metal compute pipelines, quantized GEMV, paged attention, or another implementation.

That separation is one of Celeg's core design constraints.

---

## Design principles

### Architecture agnostic

Model-specific knowledge is isolated from the generic runtime.

Adding support for a new model should extend architecture description, tensor binding, or semantics without introducing model-name switches throughout the execution engine.

### Backend agnostic

The execution model is shared across backends.

Celeg currently provides native:

- **CPU**
- **NVIDIA CUDA**
- **Apple Metal**

Backends may optimize operations differently while preserving the same semantic contract.

### Format agnostic

Checkpoint format is not the model architecture.

Celeg can discover and execute models from:

- Safetensors
- sharded Safetensors
- GGUF

Format-specific loading is kept separate from execution semantics.

### Quantization is an execution concern

Quantized formats are represented explicitly and can have backend-native implementations rather than forcing every model through a dequantize-first abstraction.

Supported paths include native GGUF quantized execution and backend-specific quantized kernels.

### Semantics before kernels

Shared semantic contracts define behavior before optimized implementations do.

This is particularly important for areas such as:

- attention
- RoPE / scaled RoPE / MRoPE
- recurrent state updates
- online softmax
- MoE routing
- causal masking
- KV-cache behavior
- sampling

Optimized kernels should implement those semantics, not redefine them.

### No compatibility baggage

Celeg favors replacing obsolete abstractions rather than maintaining internal compatibility layers indefinitely.

The runtime is still evolving, and architectural clarity takes precedence over preserving stale internal APIs.

---

## What Celeg supports

Celeg is intended to support model **structures and semantics**, not a hard-coded list of brands.

The repository currently contains validated support for architectures including combinations of:

- dense transformer blocks
- grouped-query attention
- hybrid NoPE/RoPE attention
- scaled RoPE
- MRoPE
- recurrent/state-space blocks
- Mamba-2
- gated-delta style layers
- parallel feed-forward branches
- Mixture-of-Experts
- fused output gates
- dynamic and sparse attention structures
- multimodal-aware model descriptions

These capabilities are exercised today by checkpoints from families such as:

- LFM2 / LFM2.5
- Granite
- MiniCPM5
- SmolLM3
- Nemotron 3 Nano
- Muse Glimmer
- Ling
- Lizzy
- Nanbeige
- Gemma 4
- Agnes

Model families are **validation targets**, not architectural dependencies of the runtime.

Support differs by backend and checkpoint format. See the model sweep and backend documentation for the currently validated combinations.

---

## Execution backends

### CPU

The CPU backend provides native execution without requiring CUDA or Metal.

Depending on the host and operation, Celeg can use optimized execution paths including:

- scalar implementations
- AVX2
- VNNI
- BF16 paths
- quantized kernels
- packed execution
- configurable threading and affinity

### CUDA

The CUDA backend is designed for native GPU inference and includes infrastructure for:

- CUDA Graphs
- cuBLAS / cuBLASLt
- native quantized weights
- paged KV cache
- prefix reuse
- packed prefill
- batched execution
- expert offload
- MoE execution
- backend-native attention kernels
- recurrent and hybrid architectures

### Metal

The Metal backend targets Apple Silicon using native Metal compute kernels.

It is not a wrapper around another inference runtime.

Coverage is actively expanding toward the same architecture-level semantics implemented by CPU and CUDA, including quantized execution, attention variants, recurrent structures, RoPE/MRoPE variants, and MoE paths.

---

## Model loading

Celeg does not bundle model weights.

Models can be loaded from:

- a Hugging Face repository already present in the local cache
- a local Safetensors checkpoint directory
- a local GGUF file

### Hugging Face cache

```bash
celeg-run \
  --repo LiquidAI/LFM2.5-230M \
  --prompt "Explain CUDA in one sentence." \
  --max-new-tokens 32
```

Celeg resolves the checkpoint directly from the local Hugging Face cache.

A model can be downloaded with:

```bash
celeg-download LiquidAI/LFM2.5-230M
```

### Local Safetensors

```bash
celeg-run \
  --model path/to/checkpoint \
  --prompt "Hello"
```

Sharded checkpoints using `model.safetensors.index.json` are supported.

### Local GGUF

```bash
celeg-run \
  --model path/to/model.gguf \
  --prompt "Hello"
```

GGUF metadata, tokenizer information, tensor structure, and quantization information are read directly from the file.

---

## Chat templates

Chat behavior is also model-agnostic.

Celeg can compile Hugging Face/Jinja chat templates obtained from checkpoint metadata or companion template files.

The runtime includes a deterministic restricted Jinja implementation supporting the constructs needed by modern model templates while rejecting unsafe or unsupported behavior.

Template handling includes support for concepts such as:

- system/user/assistant roles
- generation prompts
- thinking modes
- tools
- tool calls
- tool responses
- multiple EOS markers

A template can also be overridden explicitly:

```bash
celeg-run \
  --model path/to/model \
  --chat-template-file path/to/chat_template.jinja
```

---

## OpenAI-compatible server

Celeg includes an HTTP server exposing an OpenAI-compatible interface:

```bash
celeg-serve \
  --model path/to/model \
  --backend cuda \
  --port 8080 \
  --served-model-name celeg
```

The same server can use:

```text
--backend cpu
--backend cuda
--backend metal
```

The serving layer is intentionally separate from model semantics and backend execution.

---

## C API

Celeg exposes a public C ABI through:

```text
include/celeg/api.h
```

The C interface makes the runtime usable from other languages and environments such as:

- C
- C++
- Rust
- Zig
- Node.js native addons
- Python native extensions
- other FFI-capable runtimes

It provides APIs for model creation, tokenization, inference requests, stepping, polling, cancellation, backend capabilities, and diagnostics.

See `docs/API.md` for details.

---

## Build

Requirements:

- CMake 3.24+
- C++20 compiler
- Python 3 for development tooling

CUDA builds additionally require a compatible NVIDIA CUDA Toolkit.

Metal builds require Apple Silicon and the macOS SDK.

The recommended developer entrypoint is:

```bash
python scripts/dev.py doctor
python scripts/dev.py verify --backend cpu
python scripts/dev.py verify --backend cuda
python scripts/dev.py verify --backend metal
```

Or build using CMake presets:

```bash
cmake --preset cpu-relwithdebinfo
cmake --build --preset cpu-relwithdebinfo
ctest --preset cpu-relwithdebinfo
```

CUDA and Metal presets are also provided.

---

## Architecture development

When adding a new model architecture, the preferred direction is:

```text
checkpoint metadata
        ↓
architecture interpretation
        ↓
tensor / semantic bindings
        ↓
shared runtime representation
        ↓
existing backend operators
```

A new model should not normally require:

```text
CPU model-specific forward()
CUDA model-specific forward()
Metal model-specific forward()
```

If a model introduces genuinely new semantics, those semantics should become explicit runtime concepts and then receive backend implementations.

This keeps new architecture support additive rather than multiplying independent inference engines.

---

## Validation

Celeg uses several levels of validation:

- unit tests
- semantic/reference tests
- cross-backend comparisons
- checkpoint smoke tests
- model sweeps
- numerical comparisons
- quantization quality checks
- deterministic benchmark manifests
- real checkpoint parity tests

Optimizations are expected to preserve semantic behavior across backends.

The repository also contains reproducible benchmark manifests under:

```text
benchmarks/manifests/
```

and model/backend validation reports under:

```text
docs/
```

---

## Project status

Celeg is under active development.

The architectural direction is stable — **one model-independent runtime with multiple native execution backends** — but backend coverage and performance are still evolving rapidly.

CPU and CUDA currently have the broadest validated model coverage. Metal support is actively converging on the same semantic surface.

Not every checkpoint × format × backend combination should be assumed to work merely because an architecture is represented in the runtime. Refer to the current model sweep and backend validation reports for tested combinations.

---

## What Celeg is not

Celeg is not:

- a wrapper around llama.cpp
- a wrapper around PyTorch
- a CUDA-only inference engine
- an LFM-specific runtime
- a collection of unrelated model implementations
- tied to GGUF
- tied to Safetensors
- tied to Hugging Face
- tied to the OpenAI API

Those are integrations, formats, validation targets, or execution environments around the runtime — not its architecture.

---

## Long-term direction

Celeg aims to make native model inference composable across:

```text
models × checkpoint formats × quantizations × execution backends
```

without making those dimensions depend on each other.

The long-term goal is straightforward:

> **Describe a model once. Execute it anywhere Celeg has a backend.**
