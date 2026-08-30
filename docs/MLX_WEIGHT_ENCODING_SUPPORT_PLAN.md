# Model-Agnostic MLX Weight Encoding Support Plan

## Status

Proposed implementation plan.

This document defines how Celeg should consume weight encodings produced for the Apple MLX ecosystem without introducing an MLX model family, an MLX runtime dependency, repository-name dispatch, or architecture-specific MLX paths.

The first implementation target is grouped affine quantization as used by common MLX Safetensors checkpoints, initially with 4-bit weights. The design intentionally generalizes the execution contract beyond MLX so that the same normalized representation can be produced by other checkpoint formats in the future.

---

## 1. Executive summary

Celeg should not "support MLX models."

Celeg should support the storage encodings and quantization semantics found in checkpoints that were prepared for MLX.

That distinction is architectural, not cosmetic.

A checkpoint may contain a model architecture that Celeg already understands, while storing linear weights as packed grouped-affine values plus quantization parameters. The checkpoint layer must recognize and validate that physical representation, normalize it into backend-neutral weight facts, and then disappear from the execution decision.

The target flow is:

```text
Safetensors checkpoint
        |
        |  physical tensor inventory + metadata
        v
checkpoint encoding recognition
        |
        |  validated storage description
        v
normalized quantized-weight semantics
        |
        |  architecture-independent WeightPlan
        v
resolved model program
        |
        +----------------+----------------+----------------+
        |                |                |
        v                v                v
       CPU              CUDA             Metal
   reference/opt     native/opt      native/opt
```

MLX provenance may remain available for diagnostics, but it must not select model behavior or backend behavior.

The initial milestone is deliberately primitive-level rather than model-level:

> Correctly load and execute a generic grouped-affine 4-bit linear weight from a Safetensors checkpoint and match an MLX reference.

Only after that primitive is correct should end-to-end checkpoints be used as integration fixtures. Those checkpoints must span at least two different Celeg-supported model architectures to prove that no architecture-specific MLX path was introduced.

---

## 2. Architectural invariants

The implementation must preserve the existing Celeg direction: checkpoint evidence produces semantic facts, semantic facts produce a resolved model/program, and backends execute that program.

The following rules are non-negotiable.

### 2.1 Repository identity must never select behavior

No production code may branch on:

- `mlx-community`;
- a Hugging Face repository owner;
- a Hugging Face repository name;
- a local directory name;
- a model marketing name;
- a filename containing `mlx`;
- a known checkpoint URL.

Renaming or relocating a valid checkpoint must not alter how it is interpreted.

### 2.2 There is no `MlxModel`

Do not introduce concepts such as:

```text
MlxModel
MlxQwenModel
MlxLlamaModel
MlxSmolModel
MlxLfmModel
```

Architecture discovery remains the responsibility of the existing descriptor/inference/model graph path.

### 2.3 There is no `MlxRepository`

MLX-oriented checkpoints are still Safetensors repositories.

`SafeTensorRepository` already provides the correct physical repository abstraction for single-file and sharded Safetensors checkpoints. MLX-specific physical conventions must be recognized above that generic repository interface rather than by adding a parallel repository implementation.

### 2.4 There is no `--mlx` execution mode

Existing model selection remains sufficient:

```text
--repo owner/repository
--model path/to/checkpoint
```

Checkpoint interpretation must be automatic from validated evidence.

A user should not need to know that a checkpoint was produced by MLX tooling in order to run it.

### 2.5 Celeg must not depend on MLX to run the checkpoint

Production inference must not require:

- Python;
- `mlx`;
- `mlx-lm`;
- an MLX subprocess;
- an offline conversion step;
- runtime invocation of MLX kernels.

MLX is allowed as a development-time reference oracle for differential tests.

### 2.6 Source encoding and normalized semantics are different concepts

The checkpoint layer may know that a physical layout is MLX-compatible.

The model program and backend should normally know only the normalized semantics, for example:

```text
quantization = grouped affine
bits = 4
group_size = 64
logical_shape = [out_features, in_features]
scale representation = ...
bias/offset representation = ...
packing = ...
```

This distinction prevents MLX-specific naming from leaking through the runtime.

### 2.7 Unsupported evidence must fail explicitly

Celeg must not guess that an unfamiliar packed tensor layout is MLX-compatible.

Incomplete, contradictory, ambiguous, or unsupported quantization evidence must produce a deterministic diagnostic before inference begins.

---

## 3. Goals

### 3.1 Primary goals

1. Load common MLX-compatible quantized Safetensors checkpoints directly.
2. Preserve Celeg's architecture-agnostic model resolution.
3. Normalize MLX grouped-affine storage into a generic Celeg quantized-weight representation.
4. Support native packed execution without materializing the entire model as BF16/FP16.
5. Make Metal the first optimized backend because the initial use case is Apple Silicon.
6. Keep the normalized contract usable by CPU and CUDA.
7. Establish deterministic differential tests against MLX behavior.
8. Prove agnosticism with more than one model architecture using the same implementation path.
9. Preserve existing command-line, C API, server, tokenizer, and repository behavior.
10. Make future MLX quantization schemes additive rather than requiring another architecture refactor.

### 3.2 Secondary goals

- Improve the general checkpoint codec boundary while implementing the feature.
- Make quantization diagnostics describe semantics instead of checkpoint brands.
- Allow future non-MLX checkpoints to reuse the same grouped-affine normalized representation.
- Keep model architecture code unaware of packing details.

---

## 4. Non-goals

The first implementation does not need to:

- embed or wrap the MLX framework;
- execute arbitrary MLX computation graphs;
- reproduce the MLX Python API;
- recognize checkpoints by repository name;
- support every quantization mode exposed by current or future MLX releases;
- support mixed-bit policies in the first slice;
- support OptiQ policy reconstruction in the first slice;
- add a new model architecture only to demonstrate MLX loading;
- add model-specific kernels;
- add a new public API surface unless an existing generic capability query cannot describe the new weight semantics;
- optimize CPU and CUDA before the storage contract is proven correct;
- silently convert the full model to FP16/BF16 at startup and call that native quantized support.

---

## 5. Current Celeg boundaries relevant to this work

Celeg already has most of the required architecture.

### 5.1 Safetensors repository

`src/checkpoint/repositories/safetensors.cpp` already abstracts:

- a single `model.safetensors` file;
- sharded Safetensors through `model.safetensors.index.json`;
- tensor lookup;
- tensor location;
- deferred reads from shards;
- tensor inventory.

This remains the repository layer for MLX-compatible checkpoints.

### 5.2 Packed checkpoint codecs

The base runtime already separates packed encodings under:

```text
src/checkpoint/packed/
    int8.cpp
    int4.cpp
    fp8.cpp
    nvfp4.cpp
```

This is the right architectural neighborhood for physical encoding adapters.

However, the existing Celeg INT4 codec must not be treated as synonymous with MLX affine quantization.

The current packed INT4 implementation has its own physical sidecar convention and quantization contract. Reusing the word "INT4" does not imply storage compatibility or mathematical equivalence.

### 5.3 Weight planning

`src/model/weight_plan.cpp` already resolves virtual tensor requirements and recognizes checkpoint-packed codecs through generic repository evidence.

This is a useful existing pattern:

```text
model graph requests logical tensor
        |
        v
naming policy resolves virtual name
        |
        v
repository/codec determines physical availability
        |
        v
resolved WeightPlan
```

MLX-compatible grouped-affine weights should extend this idea without adding model-specific probes.

### 5.4 Backend boundary

The model graph describes semantic operations, while backend implementations own execution mechanics.

The Metal backend should therefore receive a normalized quantized-linear weight description rather than MLX repository details.

---

## 6. Terminology

The implementation should use terminology carefully.

### 6.1 `Safetensors`

The container/file format.

### 6.2 `source encoding`

The physical convention observed in the checkpoint, potentially including MLX-specific tensor relationships or metadata.

Source encoding is checkpoint-layer information.

### 6.3 `quantization semantics`

The mathematical interpretation required to recover/use approximate values, for example grouped affine quantization.

This is the useful cross-format concept.

### 6.4 `packing`

The exact bit-level representation of quantized values in storage.

Packing may differ even when mathematical quantization semantics are equivalent.

### 6.5 `provenance`

Optional diagnostic information describing where/how the checkpoint encoding appears to have originated.

Provenance must not choose execution behavior.

---

## 7. Target data model

The central design decision is to avoid a core enum whose meaning is simply "MLX".

Prefer a generic representation similar to the following conceptual API:

```cpp
enum class QuantizationKind {
    None,
    GroupedAffine,
    // Existing/future semantic quantization families.
};

enum class QuantizedPacking {
    // Values here describe physical bit packing, not model families.
    PackedUnsigned4,
    PackedSigned4,
    PackedUnsigned8,
    // Extend only when evidence requires another physical layout.
};

struct GroupedAffineWeightDescriptor {
    TensorLocator packed_weight;
    TensorLocator scales;
    std::optional<TensorLocator> biases;

    std::vector<int64_t> logical_shape;

    int bits = 0;
    int group_size = 0;
    QuantizedPacking packing{};

    TensorDType scale_dtype{};
    std::optional<TensorDType> bias_dtype;
};
```

The final names may differ to fit existing Celeg types, but the separation must remain:

```text
checkpoint provenance != quantization semantics != bit packing
```

If the exact MLX affine formula requires additional parameters, add semantic fields rather than a brand switch.

For example, if zero-point/offset interpretation is needed, encode that mathematical fact explicitly.

---

## 8. Checkpoint evidence and recognition

### 8.1 Recognition must be conjunctive

A checkpoint must not be classified from one weak clue.

Recognition should combine evidence such as:

- quantization metadata from `config.json`;
- Safetensors tensor inventory;
- expected companion tensors for a quantized base weight;
- compatible shapes;
- compatible dtypes;
- supported bit width;
- supported group size;
- optional Safetensors metadata;
- consistency with the logical tensor shape requested by the model graph.

The exact evidence contract must be derived from pinned upstream MLX fixtures during Phase 0.

### 8.2 Repository path is not evidence

The following must have zero semantic value:

```text
mlx-community/foo
foo-mlx-4bit
/Users/me/models/mlx/...
```

### 8.3 Evidence must be validated before backend setup

Do not defer malformed checkpoint discovery until a Metal kernel is launched.

The checkpoint/weight resolution phase should reject, with useful diagnostics:

- missing scale tensors;
- missing required bias/offset tensors;
- invalid bit width;
- invalid group size;
- incompatible packed dimensions;
- incompatible scale dimensions;
- impossible logical shape reconstruction;
- unsupported dtypes;
- contradictory metadata;
- partially quantized tensor groups that cannot be interpreted safely.

### 8.4 Evidence poisoning tests are mandatory

Add tests proving that identity cannot accidentally become behavior.

At minimum:

1. A valid fixture succeeds under two unrelated directory names.
2. A valid fixture succeeds without `mlx` anywhere in its path.
3. An invalid fixture named `mlx-community/...` still fails.
4. A valid MLX-compatible tensor layout under an unrelated repository name succeeds.
5. Changing only repository identity does not change the resolved weight program/fingerprint.

---

## 9. Generic grouped-affine codec

The production codec should be named after semantics or storage, not after MLX, wherever practical.

A likely layout is:

```text
include/celeg/checkpoint/packed/grouped_affine.hpp
src/checkpoint/packed/grouped_affine.cpp
```

MLX-specific recognition can live in a narrowly scoped checkpoint adapter if necessary, but the object returned from that adapter should be generic.

Responsibilities of the grouped-affine codec:

1. Validate the physical tensor relationship.
2. Preserve logical matrix dimensions.
3. Preserve the packed payload without expanding it.
4. Preserve scale/offset parameters with their real dtype.
5. Expose enough information for a backend to execute directly.
6. Provide a deterministic scalar/reference dequantizer for tests and fallback validation.
7. Avoid architecture-specific tensor names.

The codec must not:

- know about attention vs MLP;
- know about Qwen vs Llama vs SmolLM vs LFM;
- inspect Hugging Face repository ownership;
- launch Metal;
- render prompts;
- make model architecture decisions.

---

## 10. Do not overload the existing INT4 meaning

Celeg already has multiple 4-bit concepts.

A common failure mode would be to see "4-bit" and route all 4-bit weights through the existing packed INT4 implementation.

That is unsafe.

Quantization compatibility requires agreement on at least:

- quantized integer domain;
- scale formula;
- offset/bias/zero-point formula;
- group axis;
- group size;
- packed nibble ordering;
- source and scale dtype;
- logical vs physical shape mapping.

Therefore:

```text
same bit width != same encoding
```

The initial implementation must compare the pinned MLX affine behavior against the current Celeg INT4 behavior explicitly and document the differences in code comments/tests where the codecs meet.

If later analysis proves that a low-level packed representation can be shared safely, share only the proven lower-level primitive, not the semantic codec identity.

---

## 11. WeightPlan integration

`WeightPlan` should remain the boundary that turns logical tensor requirements into resolved physical weight access.

The desired behavior is:

```text
TensorRequest
    role = AttentionQuery
    logical shape = [Q, H]
        |
        v
naming policy candidates
        |
        v
repository evidence
        |
        +-- direct dense tensor
        +-- existing packed INT4
        +-- existing FP8
        +-- existing NVFP4
        +-- grouped affine packed weight
        v
resolved weight request
```

The resolved plan should describe the selected storage/quantization semantics exactly once.

Backends must not rediscover the storage type by probing raw tensor names during setup.

### 11.1 Virtual base names

If MLX-compatible checkpoints represent a logical matrix through multiple physical tensors, Celeg should continue to address the matrix through a virtual logical base name.

That preserves the existing model/naming-policy abstraction.

### 11.2 Resolution precedence

Resolution precedence must be explicit and tested when more than one representation appears loadable.

Do not silently prefer one codec based on implementation ordering.

Ambiguous physical representations should either:

- have a documented deterministic precedence grounded in checkpoint semantics; or
- fail as ambiguous.

---

## 12. CPU/reference path

Before writing an optimized Metal kernel, implement a simple correctness path.

This path is not primarily a performance feature. It is the executable specification for the quantization math.

It should support:

```text
packed grouped-affine weight
        |
        v
scalar/reference value reconstruction
        |
        v
reference linear operation
```

The reference path gives us:

- Linux CI coverage without requiring Apple hardware;
- exact unit tests for nibble/byte ordering;
- exact unit tests for group boundaries;
- shape validation tests;
- an oracle for Metal kernel development;
- easier diagnosis of upstream format changes.

For large production checkpoints, the final supported path must not require expanding the entire model into FP32/BF16.

A small reference dequantization allocation is acceptable in unit tests and diagnostic tooling.

---

## 13. Metal execution

Metal is the first optimized backend because grouped-affine MLX checkpoints are especially relevant on Apple Silicon, but the kernel contract must remain generic.

### 13.1 Native packed execution

The finished Metal path should consume packed weights directly.

Conceptually:

```text
activation tile
      +
packed quantized weight tile
      +
scale/offset group parameters
      |
      v
fused/on-demand reconstruction + accumulation
      |
      v
output tile
```

Do not implement production support as:

```text
load entire checkpoint
      |
      v
dequantize entire model to FP16
      |
      v
ordinary FP16 matmul
```

That would defeat the primary memory benefit of the checkpoint representation.

### 13.2 Kernel capability key

Kernel selection should be based on generic facts such as:

```text
operation = linear
quantization = grouped affine
bits = 4
group_size = N
packing = P
scale dtype = S
activation dtype = A
```

It must not be based on:

```text
format = MLX
model = Qwen
repository = mlx-community/...
```

### 13.3 Unsupported combinations

If a valid grouped-affine checkpoint uses a combination the Metal backend does not yet optimize, capability reporting should distinguish:

- recognized checkpoint encoding;
- valid normalized weight semantics;
- unsupported backend execution capability.

This produces a better diagnostic than pretending the checkpoint itself is invalid.

---

## 14. Model agnosticism proof

A single successful model is not sufficient evidence that the architecture is generic.

After primitive correctness is established, end-to-end acceptance must run the same implementation against at least two distinct Celeg-supported model architectures that have compatible grouped-affine checkpoints.

The models are fixtures, not feature branches.

The implementation must contain no code equivalent to:

```cpp
if (model_type == "qwen") use_mlx_weights();
if (model_type == "smollm") use_mlx_weights();
```

The proof should demonstrate:

```text
architecture A ----\
                   > existing model graph + generic grouped-affine weights --> Metal
architecture B ----/
```

Changing the integration fixture later must not require modifying the grouped-affine checkpoint codec.

---

## 15. Differential testing strategy

Correct quantized execution is subtle enough that differential testing is mandatory.

### 15.1 Checked-in synthetic fixture

Create a tiny architecture-independent fixture containing:

- a logical 2-D matrix;
- packed quantized values;
- scales;
- offsets/biases if required by the pinned format;
- quantization metadata;
- expected reconstructed values;
- one or more input vectors;
- expected output vectors.

Keep it small enough for code review and deterministic tests.

The golden output should be produced once with a pinned upstream MLX reference and then stored as test data.

### 15.2 Edge cases

Include groups that exercise:

- minimum and maximum quantized codes;
- negative reconstructed values;
- non-zero affine offset/bias if applicable;
- zero-valued groups;
- multiple groups in one row;
- multiple rows;
- exact group boundaries;
- valid supported scale dtypes;
- malformed payload sizes;
- malformed metadata.

### 15.3 Apple differential test

On Apple Silicon, add an optional/developer differential test that compares Celeg's primitive result with the pinned MLX reference implementation.

The reference tool may use Python/MLX because it is test tooling, not production runtime.

The test should report numeric error explicitly rather than only checking token equality.

Suggested metrics:

- max absolute error;
- mean absolute error;
- max relative error where meaningful;
- output shape;
- finite-value checks.

### 15.4 End-to-end logits

For integration fixtures, compare deterministic prefill/decode outputs with the MLX reference.

Prefer comparisons in this order:

1. selected intermediate primitive outputs during bring-up;
2. final logits for a fixed prompt;
3. greedy next token;
4. short greedy continuation.

Token equality alone is too weak for initial kernel validation because small numeric errors can remain hidden until a later prompt.

---

## 16. Phase 0 — Upstream format census and golden contract

Before changing production execution code:

1. Pin representative upstream MLX affine checkpoints.
2. Record the relevant MLX/`mlx-lm` revision used as the behavioral reference.
3. Inventory quantization metadata.
4. Inventory physical tensor names for several linear layers.
5. Record shapes and dtypes for packed weights and companion tensors.
6. Verify the exact affine reconstruction formula from upstream behavior/source.
7. Verify group axis and group-size interpretation.
8. Verify packed element ordering.
9. Verify whether all model weights are quantized or whether exclusions are normal.
10. Generate the tiny checked-in golden fixture.

### Exit criteria

- The physical contract is documented by tests, not assumptions.
- A standalone reference script can reproduce the expected fixture outputs.
- No production architecture decision depends on a repository name.

---

## 17. Phase 1 — Generic grouped-affine checkpoint codec

Implement the physical codec and normalization layer.

Likely work:

- add a generic grouped-affine packed descriptor;
- add physical validation helpers;
- recognize compatible companion tensors through the repository interface;
- map metadata into `bits`, `group_size`, packing, and parameter dtypes;
- expose the logical matrix through its virtual base name;
- add deterministic diagnostics for malformed evidence.

Potential files:

```text
include/celeg/checkpoint/packed/grouped_affine.hpp
src/checkpoint/packed/grouped_affine.cpp
cmake/sources/base_runtime.cmake
```

If source-specific recognition requires an adapter, keep it narrowly scoped under `checkpoint/` and return the generic grouped-affine descriptor.

### Exit criteria

- Fixture recognition succeeds independent of path/repository identity.
- Invalid evidence fails before backend initialization.
- The descriptor fully captures the semantics needed for reconstruction.
- No model-specific code changed.

---

## 18. Phase 2 — Reference decoder and linear operation

Implement the executable specification.

Tasks:

1. Decode packed 4-bit groups correctly.
2. Apply scale and affine offset/bias semantics correctly.
3. Reconstruct selected values for unit tests.
4. Implement a small reference linear operation.
5. Compare against the checked-in golden outputs.
6. Add malformed-shape and overflow tests.

### Exit criteria

- Linux/CPU CI can validate the codec without MLX installed.
- Golden primitive tests pass exactly within defined tolerances.
- All packing and group-boundary cases are covered.

---

## 19. Phase 3 — WeightPlan and capability integration

Teach the resolved weight path to select grouped-affine storage from evidence.

Tasks:

1. Extend virtual-tensor availability checks.
2. Resolve grouped-affine physical tensors once.
3. Store the normalized descriptor in the resolved weight plan.
4. Ensure backends do not probe MLX-style raw names independently.
5. Add plan/fingerprint tests.
6. Add ambiguous-representation tests.
7. Add backend capability diagnostics.

### Exit criteria

- The resolved model program is stable under checkpoint directory renaming.
- Backend setup receives normalized facts rather than MLX-specific names.
- Existing dense/INT4/FP8/NVFP4 checkpoints remain unchanged.

---

## 20. Phase 4 — Native Metal grouped-affine linear

Implement the first optimized backend.

Tasks:

1. Define Metal buffer binding for packed values and quantization parameters.
2. Implement a correctness-first grouped-affine linear kernel.
3. Match the CPU/reference primitive.
4. Match the upstream MLX differential reference.
5. Add optimized tiling after correctness is locked.
6. Avoid full-weight dequantization.
7. Add memory accounting for packed residency.
8. Add capability checks for supported bit/group/dtype combinations.

### Exit criteria

- The synthetic primitive matches the reference.
- Packed weights remain packed in normal model residency.
- No full-model BF16/FP16 expansion occurs.
- Existing Metal tests continue to pass.

---

## 21. Phase 5 — End-to-end architecture-agnostic proof

Use real checkpoints only after the generic primitive works.

Select at least two model architectures already understood by Celeg and available in compatible grouped-affine Safetensors form.

The exact models should be chosen during implementation from current, pinned fixtures. They are validation assets, not architectural dependencies.

For each fixture:

1. resolve the checkpoint without a special CLI flag;
2. print normalized quantization diagnostics;
3. load without offline conversion;
4. run deterministic prefill;
5. compare final logits with the reference;
6. compare a greedy next token;
7. generate a short greedy continuation;
8. record peak model memory;
9. verify that renaming the local checkpoint directory changes nothing.

### Critical code-review criterion

A diff that adds a model-specific MLX branch fails this phase even if both checkpoints generate sensible text.

### Exit criteria

- Two distinct architectures run through the same generic grouped-affine path.
- No repository identity dispatch exists.
- No model-specific MLX logic exists.
- Differential results satisfy documented tolerances.

---

## 22. Phase 6 — Hardening, diagnostics, and documentation

Once the implementation is real:

1. Update `docs/QUANTIZATION_SUPPORT_MATRIX.md`.
2. Update README support statements.
3. Add benchmark manifests for representative Apple Silicon runs.
4. Add `--print-config` diagnostics for normalized quantization.
5. Document supported bits/group sizes/dtypes precisely.
6. Document failure messages for recognized-but-unsupported combinations.
7. Add regression tests for tensor inventory poisoning.
8. Add checkpoint rename/path-independence tests.
9. Run full CPU/CUDA/Metal verification to catch boundary regressions.

Suggested diagnostic language:

```text
container: safetensors
checkpoint encoding: mlx-compatible grouped affine
quantization: grouped-affine
bits: 4
group-size: 64
backend: metal
execution: native packed
```

The `checkpoint encoding` line is diagnostic provenance. The `quantization` line is the semantic execution contract.

---

## 23. Proposed file-level changes

Exact file placement may evolve during implementation, but ownership should remain approximately:

### Checkpoint/storage layer

```text
include/celeg/checkpoint/packed/grouped_affine.hpp      new
src/checkpoint/packed/grouped_affine.cpp                new
src/checkpoint/formats/safetensors.cpp                  only if metadata exposure is insufficient
src/checkpoint/repositories/safetensors.cpp             minimal/general changes only
cmake/sources/base_runtime.cmake                        register new source
```

### Model/weight resolution

```text
include/celeg/model/weight_plan.hpp                     extend normalized weight description
src/model/weight_plan.cpp                               recognize generic grouped-affine availability
```

If existing types place resolved storage elsewhere, follow those boundaries instead of forcing new ownership into `weight_plan.hpp`.

### Reference implementation

Prefer a generic quantization/reference location, for example:

```text
include/celeg/model/weights/...
src/model/weights/...
```

Do not place the reference math under a model architecture directory.

### Metal backend

Use the existing Metal model/runtime/kernel organization and add only generic quantized-linear support.

No new model-family directory should be created for MLX.

### Tests

Add:

```text
synthetic grouped-affine fixture tests
checkpoint evidence validation tests
WeightPlan resolution tests
identity poisoning tests
CPU/reference math tests
Metal primitive differential tests
end-to-end multi-architecture tests
```

### Documentation after implementation

```text
docs/QUANTIZATION_SUPPORT_MATRIX.md
README.md
BENCHMARK.md or dedicated benchmark documentation if appropriate
```

---

## 24. Public API and CLI compatibility

The feature should require no breaking public API change.

Existing entry points should work:

```bash
celeg-metal-run --repo <compatible-repository> --prompt "Hello"
celeg-metal-run --model <checkpoint-directory> --prompt "Hello"
celeg-serve --model <checkpoint-directory> --backend metal
```

Do not require:

```bash
--mlx
--mlx-model
--mlx-quant
```

If the C API exposes backend capability information, prefer extending generic quantization/capability reporting rather than adding MLX-specific execution functions.

---

## 25. Error model

Errors should identify the failing architectural layer.

### Invalid physical checkpoint

Example:

```text
quantized weight <name>: grouped-affine scale shape is incompatible with logical matrix [R,C] and group_size=N
```

### Recognized semantics but unsupported backend

Example:

```text
grouped-affine weight is valid but Metal does not support bits=8, group_size=128, scale_dtype=...
```

### Ambiguous encoding

Example:

```text
weight <name> resolves to multiple incompatible packed representations
```

### Missing evidence

Example:

```text
checkpoint declares grouped-affine quantization but required companion tensor for <name> is missing
```

Avoid errors such as:

```text
unsupported MLX model
unsupported mlx-community repository
```

Those messages encode the wrong abstraction.

---

## 26. Performance requirements

Correctness comes first, but the final feature should preserve the reason quantized checkpoints exist.

### 26.1 Memory

Production execution must preserve packed model residency wherever the backend advertises native grouped-affine support.

Measure:

- checkpoint bytes;
- host-resident model bytes;
- Metal buffer bytes;
- temporary workspace bytes;
- peak unified-memory use.

### 26.2 Decode

Measure at least:

- one-token latency;
- steady-state decode tokens/s;
- kernel time for representative linear shapes.

### 26.3 Prefill

Measure:

- prompt tokens/s;
- representative short and medium prompt lengths;
- quantized-linear contribution to total runtime.

### 26.4 Comparison

Compare, where reproducible on the same Mac:

```text
Celeg grouped-affine Metal
vs
MLX reference
```

Performance is not an acceptance requirement until numerical correctness and architecture boundaries are proven.

---

## 27. Security and robustness considerations

Checkpoint metadata and tensor dimensions are untrusted input.

All new parsing must use checked arithmetic and reject:

- size overflows;
- negative dimensions;
- impossible packed byte counts;
- group counts that overflow allocation sizes;
- tensor aliases that escape expected repository behavior;
- malformed dtypes;
- contradictory logical/physical dimensions.

Do not allocate based solely on unchecked checkpoint metadata.

The existing packed-codec style of validating shapes and byte counts before decoding should be preserved.

---

## 28. Regression protection

The implementation must not weaken existing behavior.

Run and preserve:

- Safetensors dense loading;
- sharded Safetensors loading;
- existing compressed/packed INT4 loading;
- FP8 loading;
- NVFP4 loading;
- GGUF paths;
- CPU verification;
- CUDA verification where available;
- Metal verification on Apple Silicon;
- architecture-boundary checks;
- public API/symbol checks.

Add specific tests ensuring a dense tensor with a coincidentally similar companion name is not misclassified as grouped-affine.

---

## 29. Future extensions

Once grouped-affine 4-bit is stable, extend by semantic capability rather than by repository family.

Likely future slices include:

### 29.1 Grouped-affine 8-bit

Reuse the same semantic descriptor with a new validated packing/capability combination.

### 29.2 Mixed quantization by tensor/layer

The WeightPlan should already be capable of resolving storage per logical tensor.

A future mixed-bit checkpoint should therefore be a collection of independently resolved weight descriptions, not a model-wide `mlx_mode`.

### 29.3 OptiQ-style policies

Treat the resulting per-weight encoding as evidence already present in the checkpoint. Do not make runtime execution depend on reconstructing a named optimization policy unless the physical format genuinely requires it.

### 29.4 MXFP4

Model it as its own quantization semantics/packing if mathematically distinct from grouped affine.

Do not force it into `GroupedAffine` merely because both can use four bits.

### 29.5 NVFP4 and MXFP8 convergence

Celeg already has packed low-precision concepts. Future refactoring may find shared primitives, but convergence should occur only after exact semantics and layouts are compared.

### 29.6 CPU and CUDA optimized execution

Once the normalized descriptor is stable, optimized CPU/CUDA support should require no checkpoint or model architecture redesign.

That is an important validation of this plan.

---

## 30. Explicit anti-patterns

Reject implementations that introduce any of the following.

### Repository dispatch

```cpp
if (repo.starts_with("mlx-community/")) { ... }
```

### Architecture dispatch for encoding

```cpp
if (model_type == ModelType::Qwen && is_mlx) { ... }
```

### Backend reparsing

```cpp
// Metal backend probes raw `.scales`/`.biases` names itself.
```

### Global mode flag

```cpp
model.is_mlx = true;
```

### Full-model expansion marketed as native support

```text
MLX 4-bit checkpoint -> expand all weights to FP16 -> run FP16
```

### Bit-width conflation

```cpp
if (bits == 4) use_existing_int4_codec();
```

### Silent fallback after contradictory evidence

```text
metadata says quantized, companion tensors are invalid -> pretend checkpoint is dense
```

All of these violate the intended boundary.

---

## 31. Implementation sequence

Recommended order:

```text
0. Pin upstream evidence and generate tiny golden fixture
   |
1. Add generic grouped-affine physical descriptor + validation
   |
2. Add scalar/reference reconstruction and linear math
   |
3. Integrate resolved storage into WeightPlan
   |
4. Add capability reporting
   |
5. Add correctness-first native Metal kernel
   |
6. Differential-test primitive against MLX
   |
7. Run first real checkpoint end to end
   |
8. Run second architecture through the exact same path
   |
9. Optimize Metal kernel
   |
10. Benchmark, harden, document support matrix
```

Do not start by wiring a specific Hugging Face model into the Metal runner.

---

## 32. Definition of done

The first production slice is complete only when all of the following are true.

### Architecture

- [ ] No MLX model class exists.
- [ ] No MLX repository class exists.
- [ ] No repository-name dispatch exists.
- [ ] No architecture-specific MLX branch exists.
- [ ] Existing model inference/graph semantics remain checkpoint-brand agnostic.
- [ ] MLX-specific evidence is normalized before backend execution.

### Checkpoint correctness

- [ ] Grouped-affine 4-bit evidence is recognized from validated metadata/tensors.
- [ ] Malformed and contradictory evidence fails explicitly.
- [ ] Logical shapes are validated against packed physical shapes.
- [ ] Scale/offset dtypes and dimensions are validated.
- [ ] Directory/repository renaming does not affect resolution.

### Numerical correctness

- [ ] Checked-in synthetic fixture matches the pinned reference.
- [ ] CPU/reference decoder passes all group/packing edge cases.
- [ ] Metal primitive matches CPU/reference output within documented tolerance.
- [ ] Apple differential primitive matches MLX reference within documented tolerance.

### Native execution

- [ ] Metal executes directly from packed weights.
- [ ] Normal inference does not expand the whole model to BF16/FP16.
- [ ] Memory accounting reflects packed residency.

### Model agnosticism

- [ ] At least two distinct existing Celeg-supported architectures run compatible grouped-affine checkpoints through the same implementation.
- [ ] Neither integration fixture requires a model-specific grouped-affine code path.

### Compatibility

- [ ] Existing `--repo` and `--model` flows work unchanged.
- [ ] No `--mlx` flag is required.
- [ ] Existing dense Safetensors tests pass.
- [ ] Existing packed INT4/FP8/NVFP4 tests pass.
- [ ] Existing GGUF tests pass.
- [ ] CPU/CUDA/Metal verification remains green where the backend is available.
- [ ] Public API and exported symbols remain compatible unless a separately reviewed generic capability extension is required.

### Documentation

- [ ] Supported grouped-affine combinations are documented precisely.
- [ ] `docs/QUANTIZATION_SUPPORT_MATRIX.md` is updated after implementation.
- [ ] README describes the capability as a weight/checkpoint encoding capability, not an MLX model family.

---

## 33. Final architecture

The desired end state is:

```text
                           CHECKPOINT WORLD

     Dense Safetensors      MLX-compatible ST      GGUF / other packed ST
             |                     |                       |
             +---------- evidence + validation -----------+
                                   |
                                   v
                         normalized weight facts
                                   |
          +------------------------+-------------------------+
          |                        |                         |
       dense                  grouped affine           other codecs
          |                        |                         |
          +------------------------+-------------------------+
                                   |
                                   v
                              WeightPlan
                                   |
                                   v
                         resolved model program
                                   |
                  +----------------+----------------+
                  |                |                |
                 CPU              CUDA             Metal
                  |                |                |
                  +------ generic backend ops ------+
```

The model architecture remains orthogonal:

```text
Qwen --------\
Llama --------\
SmolLM --------> semantic model graph ---> same weight-resolution layer
LFM -----------/
Granite -------/
future --------/
```

This is the core acceptance principle:

> A model architecture describes computation. A checkpoint encoding describes how parameters are stored. A backend describes how the resolved computation is executed. None of those identities should be used as a shortcut for another.

If that boundary is preserved, adding MLX-compatible grouped-affine weights becomes a reusable Celeg capability rather than another model-specific integration.
