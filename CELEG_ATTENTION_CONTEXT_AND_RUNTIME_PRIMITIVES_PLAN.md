# CELEG Attention, Long-Context, and Runtime Primitives Plan

Status: active implementation roadmap  
Date: 2026-08-07  
Scope: model-agnostic attention primitives, positional encodings, long-context extension, KV/cache policies, sparse and linear attention, multimodal attention, backend lowerings, and execution-scale infrastructure

This document is the implementation companion to `CELEG_SOLID_ARCHITECTURE_REFACTORING_PLAN.md`.

The SOLID document defines the architectural north star: CELEG should become a model-agnostic compiler/runtime where a model using already-supported mathematics can be added without new model-family C++ code.

This document defines the primitive vocabulary and backend capabilities required to make that goal realistic for modern 2025-2026 model architectures.

The rule connecting both documents is:

> A model family is not a runtime primitive.

If a new model introduces no new mathematics, support should be descriptor/binding/test work only.

If a new model introduces new mathematics, CELEG adds a generic primitive and backend lowering, not a model-family execution fork.

---

# 1. Current CELEG baseline

At the reviewed baseline, the semantic graph already has useful building blocks.

## 1.1 Attention semantics already expressible

`AttentionSpec` currently represents:

- query-head count;
- key/value-head count;
- head dimension;
- Q/K normalization;
- causal attention;
- sliding causal attention;
- sliding-window size;
- RoPE theta;
- partial rotary fraction;
- shared/published KV groups;
- query scaling;
- query gating;
- positional encoding as either none or RoPE.

Because query heads and KV heads are independent, the current representation can already express the ordinary MHA/GQA/MQA family of projection layouts.

## 1.2 Mixer primitives already present

The graph already has semantic variants for:

- dot-product attention;
- short convolution;
- GatedDeltaNet;
- Mamba2;
- MLP-only mixer blocks.

## 1.3 Feed-forward primitives already present

The graph already supports:

- dense feed-forward blocks;
- Mixture of Experts;
- shared experts;
- grouped routing information;
- routed scaling;
- router softmax / normalization choices;
- per-layer input paths;
- explicit mixer-only layers.

## 1.4 Runtime infrastructure already present

CELEG already contains important long-context infrastructure that must be preserved and generalized rather than reinvented:

- paged KV on CPU;
- paged KV on CUDA;
- paged CUDA attention kernels;
- paged/chunked prefill paths;
- prefix cache infrastructure;
- CPU prefix snapshots/persistence;
- concurrent CPU/CUDA engines;
- CUDA INT8 KV mode;
- CPU FP32/BF16 KV modes.

This means the main missing work is not simply "add paged attention". The main work is to expand the semantic IR, positional policies, state representations, and lowering choices.

---

# 2. Architectural decomposition

Do not keep growing one monolithic `AttentionSpec` indefinitely.

The target should decompose attention semantics into orthogonal concepts.

A possible direction is:

```cpp
struct DotProductAttentionSpec {
    AttentionProjectionSpec projection;
    AttentionPatternSpec pattern;
    PositionSpec position;
    AttentionNormalizationSpec normalization;
    AttentionBiasSpec bias;
    AttentionStateSpec state;
    AttentionOutputSpec output;
};
```

The exact C++ types may change during implementation. The responsibility split should not.

## 2.1 Projection is not masking

Projection semantics include:

- MHA/GQA/MQA dimensions;
- fused vs separate Q/K/V as a binding/lowering detail;
- Q/K normalization;
- query gating;
- latent compression where applicable;
- projection ranks where applicable.

## 2.2 Pattern is not positional encoding

Attention pattern answers:

> Which keys may a query attend to?

Position specification answers:

> How is positional information represented or injected?

These must be independent.

## 2.3 State representation is not model identity

Ordinary KV, latent KV, recurrent state, compressed memories, paged storage, quantized storage, and distributed ownership are execution/state concepts.

They must not be represented as "DeepSeek mode", "Kimi mode", or another family-specific flag.

---

# 3. Phase A - First-class positional encoding and context extension

Priority: P0  
Purpose: remove the current `RoPE theta + optional factor` bottleneck and make context-extension algorithms explicit semantics.

## A1. Replace the current narrow positional enum

Current semantic choice is effectively:

```text
None
RoPE
```

Target direction:

```cpp
using PositionSpec = std::variant<
    NoPositionEncodingSpec,
    RopePositionSpec,
    MultiAxisRopeSpec,
    AlibiPositionSpec,
    RelativePositionBiasSpec
>;
```

ALiBi and relative bias may land later, but the ownership boundary should be prepared now.

## A2. Make RoPE scaling algorithm explicit

Do not represent every extension method as a generic numeric `scaling_factor`.

Target direction:

```cpp
using RopeScalingSpec = std::variant<
    NoRopeScaling,
    LinearRopeScaling,
    DynamicNtkRopeScaling,
    YarnRopeScaling,
    LongRopeScaling,
    Llama3RopeScaling
>;
```

## A3. Linear / Position Interpolation

Represent ordinary linear position scaling explicitly.

Example semantics:

```cpp
struct LinearRopeScaling {
    double factor;
    int original_context;
};
```

Acceptance:

- descriptor can express it without family-specific C++;
- CPU and CUDA produce numerically equivalent frequencies;
- native-context behavior remains unchanged;
- extended-context behavior has reference tests.

## A4. Dynamic / NTK-aware RoPE

Represent dynamic base/frequency adjustment separately from linear interpolation.

Acceptance:

- scaling activates according to the declared semantics rather than model name;
- position-dependent behavior is shared by CPU/CUDA reference logic;
- no `if (family == ...)` in RoPE kernels or model execution.

## A5. YaRN

Add a generic YaRN policy capable of representing at least:

- scaling factor;
- original context length;
- attention factor;
- beta-fast;
- beta-slow;
- any derived frequency-region parameters required by the implementation.

Target example:

```cpp
struct YarnRopeScaling {
    double factor;
    int original_context;
    double attention_factor;
    double beta_fast;
    double beta_slow;
};
```

Acceptance:

- Ministral/Mistral-derived YaRN configurations can be imported as data;
- no Mistral-specific YaRN implementation exists;
- reference vectors validate CPU/CUDA position transforms.

## A6. LongRoPE

LongRoPE must be modeled as more than one scalar factor.

Target example:

```cpp
struct LongRopeScaling {
    int original_context;
    std::vector<float> short_factors;
    std::vector<float> long_factors;
    float attention_factor;
};
```

Acceptance:

- frequency-dependent factors survive descriptor -> IR -> compiled program;
- no Phi-specific branch is required;
- factors are validated against rotary dimension;
- CPU/CUDA use the same resolved semantic tables.

## A7. Llama-3-style frequency-aware scaling

Represent low/high frequency regions explicitly.

Target example:

```cpp
struct Llama3RopeScaling {
    double factor;
    int original_context;
    double low_frequency_factor;
    double high_frequency_factor;
};
```

Acceptance:

- imported configuration is algorithmic data, not a Llama family identity;
- test fixture with identical parameters behaves identically regardless of provenance.

## A8. Partial rotary as first-class semantics

The existing `rotary_fraction` capability is useful and should be preserved, but moved into the new position specification rather than remaining an incidental field of a large attention object.

Acceptance:

- partial rotary works with ordinary, linear, dynamic, YaRN, LongRoPE, and compatible scaling modes;
- validation rejects incompatible rotary dimensions.

## A9. M-RoPE / multi-axis RoPE as first-class IR

CELEG already has M-RoPE implementation paths. The architectural task is to move the semantics into the canonical IR.

Target direction:

```cpp
struct MultiAxisRopeSpec {
    RopePositionSpec base;
    std::vector<int> sections;
    bool interleaved;
    int axes;
};
```

Acceptance:

- M-RoPE is not represented as a Qwen-specific property;
- token and chunk paths use the same position semantics;
- multimodal positions are explicit inputs to the position primitive.

---

# 4. Phase B - Generalize attention patterns

Priority: P0/P1  
Purpose: make masking/connectivity a reusable semantic dimension.

## B1. Replace the two-value mask enum

Current:

```text
Causal
SlidingCausal
```

Target direction:

```cpp
using AttentionPatternSpec = std::variant<
    FullCausalPattern,
    SlidingWindowPattern,
    BidirectionalPattern,
    PrefixLmPattern,
    BlockSparsePattern,
    DynamicSparsePattern
>;
```

## B2. Full causal

Preserve current behavior as the simplest pattern.

## B3. Sliding-window attention

Make sliding-window ownership independent from family and positional policy.

Acceptance:

- per-layer full/sliding schedules are data;
- Ministral-style interleaved schedules require no family execution code;
- window size is validated independently per layer.

## B4. Bidirectional attention

Needed for a genuinely general IR and future encoders/vision towers.

Acceptance:

- causal assumptions are not baked into generic attention kernels;
- backend compiler may choose specialized kernels for causal vs bidirectional.

## B5. Prefix-LM masks

Represent a bidirectional prefix followed by causal generation.

This must be semantic mask data, not chat-template behavior.

## B6. Block-sparse attention

Add an explicit block-level sparse representation.

Avoid a fully arbitrary per-token mask in the hot path unless required; prefer compact declarative structure that can be lowered efficiently.

## B7. Dynamic sparse selection

Reserve an explicit semantic boundary for attention mechanisms where attended blocks/tokens are selected dynamically.

This is needed before Native Sparse Attention or similar mechanisms can be represented cleanly.

---

# 5. Phase C - Multi-head Latent Attention (MLA)

Priority: P0  
Purpose: support modern latent-KV architectures without encoding a DeepSeek family branch.

MLA is not ordinary GQA with different head counts. It changes the representation of attention state and projection structure.

## C1. Add a latent-attention primitive

Possible direction:

```cpp
struct LatentAttentionSpec {
    int query_heads;
    int head_dim;
    int query_rank;
    int kv_rank;
    int rope_head_dim;
    int nope_head_dim;
    bool decoupled_rope;
    AttentionPatternSpec pattern;
    PositionSpec position;
};
```

Exact field names should follow the mathematical decomposition, not one checkpoint's config names.

## C2. Separate latent state from ordinary KV state

Do not force MLA into an ordinary K/V page shape if the compressed latent representation is the persistent state.

Introduce a generic attention-state description from which the backend can derive:

- bytes per token/page;
- persistent vs transient projections;
- page layout;
- quantization eligibility;
- sharing rules.

## C3. CPU reference implementation first

Build a correctness-first CPU/reference lowering before aggressive fusion.

Acceptance:

- reference output matches a trusted implementation on small deterministic fixtures;
- decode state growth matches the declared latent representation;
- model-family identity never reaches the operator.

## C4. CUDA lowering

Add CUDA projection/state/attention lowering with correctness parity before optimization.

Then optimize:

- fused decompression/projection where useful;
- paged latent cache;
- decode-specialized kernels;
- batch-friendly layout.

## C5. Acceptance gate

A DeepSeek-V2/V3-style descriptor should be able to instantiate MLA without adding `models/deepseek/**` execution code.

---

# 6. Phase D - Cross-attention and multi-source attention

Priority: P0/P1  
Purpose: make the IR capable of decoder/encoder and multimodal topologies without family-specific attention variants.

## D1. Make Q source and KV source explicit

Target concept:

```cpp
struct AttentionSourceSpec {
    ValueRef query_source;
    ValueRef key_value_source;
};
```

Self-attention becomes the case where both sources are the current sequence.

Cross-attention becomes the case where K/V come from another graph value or memory.

## D2. Support persistent encoder/vision memory

Backend state planning must distinguish:

- autoregressive KV that grows token by token;
- immutable encoder/vision memory;
- reusable projected memory;
- per-request vs shared memory.

## D3. Acceptance

- a cross-attention layer requires no dedicated model-family operator;
- self- and cross-attention reuse the same semantic primitive where mathematics permits;
- memory ownership is explicit in the compiled backend program.

---

# 7. Phase E - Native Sparse Attention and modern sparse mechanisms

Priority: P1  
Purpose: represent long-context attention whose complexity is reduced by structured sparse access.

## E1. Do not create a `DeepSeekSparseAttention` family type

Model the mathematical components instead.

A possible decomposition is:

```cpp
struct SparseAttentionSpec {
    CompressionSelectionSpec compressed;
    TokenSelectionSpec selected;
    SlidingWindowPattern local;
    PositionSpec position;
};
```

## E2. Separate selection from attention computation

The selection/routing stage may produce a sparse access plan.

The attention stage consumes that plan.

This keeps the design reusable for future sparse mechanisms.

## E3. Backend lowering

CPU:

- correctness/reference selector;
- sparse gather;
- block/tile scheduling.

CUDA:

- fused or staged selection;
- compact block lists;
- coalesced sparse page access;
- paged-KV interoperability.

## E4. Acceptance

A new model using the same sparse mechanism should require only descriptor/binding changes.

---

# 8. Phase F - Kimi Delta Attention and extensible linear attention

Priority: P1  
Purpose: prevent every new linear-attention family from becoming another hard-coded mixer identity.

CELEG already supports GatedDeltaNet. Use that implementation experience to factor shared recurrent/linear-attention semantics.

## F1. Introduce a linear-attention semantic family only where mathematically justified

Do not create inheritance for its own sake.

Identify reusable concepts such as:

- recurrent state shape;
- decay/gating policy;
- feature transforms;
- convolutional preprocessing;
- state update rule;
- output normalization.

## F2. Add Kimi Delta Attention as mathematics

Represent KDA's state/gating/update semantics directly.

No `KimiArchitecture` execution fork.

## F3. Chunk/parallel prefill is a first-class requirement

For recurrent/linear attention, the primitive contract should distinguish:

- token update;
- native chunk/parallel scan;
- generic sequential fallback.

The backend compiler resolves the best available strategy once.

## F4. Acceptance

- token and chunk results are equivalent within tolerance;
- recurrent state is serializable/persistable where appropriate;
- no family checks occur in execution.

---

# 9. Phase G - Attention bias and relative position families

Priority: P2  
Purpose: broaden the IR beyond RoPE-only decoder models.

## G1. ALiBi

Represent ALiBi as an attention-score bias policy, not as a special model.

## G2. Relative position bias

Provide a generic table/bucket-based relative-position bias representation suitable for encoder/decoder architectures that use it.

## G3. Architectural rule

Position encoding and attention bias are separate concepts even when a model uses one as its primary positional mechanism.

---

# 10. Phase H - KV state representation and quantization

Priority: P0/P1  
Purpose: make long context practical rather than merely semantically valid.

## H1. Generalize KV/state format

CPU currently exposes FP32/BF16 KV modes and CUDA exposes BF16/INT8.

Move toward a backend capability model where state storage format can include, where supported:

```text
FP32
BF16
FP16
FP8
INT8
INT4 / low-bit experimental formats
```

Do not promise every format on every backend.

## H2. Per-component quantization policy

Allow keys, values, latent state, and recurrent state to have different storage policy when the algorithm requires it.

Target concept:

```cpp
struct AttentionStateStorageSpec {
    ScalarStorage key;
    ScalarStorage value;
    ScalarStorage latent;
    QuantizationGranularity granularity;
};
```

## H3. Quantization is lowering/storage, not model semantics

The model IR describes mathematical state.

Backend options and capabilities choose physical storage precision unless a checkpoint/runtime contract requires otherwise.

## H4. Paged state must generalize beyond ordinary KV

Extend page planning so it can support:

- ordinary K/V pages;
- latent MLA pages;
- sparse-attention page metadata;
- recurrent-state snapshots where paging is meaningful.

## H5. Acceptance

- page allocator does not assume every attention layer stores two equal-width BF16 matrices;
- memory statistics derive from compiled state layout;
- changing state precision does not change model-family code.

---

# 11. Phase I - Streaming context and attention sinks

Priority: P1  
Purpose: support bounded-memory indefinite sessions independently of a model's trained maximum context.

Streaming context is a runtime policy, not a model architecture.

## I1. Add explicit streaming-cache policy

Possible direction:

```cpp
struct StreamingAttentionStatePolicy {
    int sink_tokens;
    int recent_tokens;
};
```

## I2. Preserve logical positions correctly

Evicting physical KV pages must not incorrectly reset positional semantics.

Position policy and cache retention policy must remain distinct.

## I3. Prefix cache interoperability

Define how prefix snapshots, shared prefixes, and streaming eviction interact.

## I4. Acceptance

- memory remains bounded after arbitrarily long decode in streaming mode;
- retained sink/recent regions are explicit and testable;
- ordinary full-context mode is unchanged.

---

# 12. Phase J - Long-context prefill and memory efficiency

Priority: P0/P1  
Purpose: scale ingestion of large prompts without duplicating semantic execution.

CELEG already has chunked/paged prefill. Refactor and extend it around compiled primitive capabilities.

## J1. Unified chunk capability

Every compiled primitive should declare one of:

```text
native chunk lowering
generic sequential chunk adapter
unsupported by backend
```

Do not discover this by model-family identity.

## J2. Paged prefill should consume compiled attention/state layout

No ordinary-KV assumptions should leak into the generic scheduler once MLA/sparse state exists.

## J3. Ragged batching

Keep long-context batching compatible with requests at different prompt lengths and page boundaries.

## J4. Workspace planning

Large prefill scratch sizes must come from a backend workspace plan derived at compile/load time.

Avoid prompt-size-triggered semantic branching scattered through operators.

---

# 13. Phase K - Context parallelism and distributed attention

Priority: P1/P2  
Purpose: make contexts larger than one device's practical state/compute envelope possible.

This is a backend lowering strategy, not Model IR semantics.

## K1. Add a context-parallel lowering boundary

Target concept:

```text
Attention IR
    |
    +--> single-device lowering
    +--> context-parallel lowering
    +--> ring/distributed lowering
```

## K2. Sequence partitioning

Backend compiler/runtime owns:

- sequence/block partition;
- communication schedule;
- KV ownership;
- collective/reduction strategy;
- overlap of communication and attention computation.

## K3. Do not put world size in the model graph

World size, device mesh, NCCL topology, and placement are runtime/backend deployment concerns.

## K4. Acceptance

The same compiled semantic model can be lowered to one device or multiple devices without changing descriptors.

---

# 14. Phase L - Flash/fused attention as lowering choices

Priority: continuous optimization  
Purpose: keep algorithm and implementation strategy separate.

FlashAttention-like kernels, fused projections, fused RoPE, paged kernels, GEMM-based kernels, tensor-core paths, and decode-specialized kernels are backend implementations of a semantic primitive.

They must not become separate model semantics unless the mathematics actually differs.

Target:

```text
DotProductAttentionSpec
        |
        v
CUDA attention compiler
        |
        +-- decode paged kernel
        +-- prefill flash/fused kernel
        +-- GEMM fallback
        +-- sparse lowering
```

Kernel choice may depend on:

- sequence length;
- head dimension;
- batch shape;
- state format;
- device capability;
- workspace;
- pattern;

but not on model-family name.

---

# 15. Primitive capability matrix

Maintain an executable/documented matrix as the implementation evolves.

Suggested dimensions:

| Primitive / policy | IR | CPU token | CPU chunk | CUDA token | CUDA chunk/prefill | paged state | quantized state |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full causal MHA/GQA/MQA | existing | existing | existing | existing | existing | yes | partial |
| Sliding-window attention | existing | existing | existing | existing | existing/verify | yes | partial |
| Partial RoPE | existing | existing | verify | existing | verify | n/a | n/a |
| M-RoPE | partial/current topology | existing | existing/verify | existing | verify | n/a | n/a |
| Linear RoPE scaling | planned | planned | planned | planned | planned | n/a | n/a |
| Dynamic NTK | planned | planned | planned | planned | planned | n/a | n/a |
| YaRN | planned | planned | planned | planned | planned | n/a | n/a |
| LongRoPE | planned | planned | planned | planned | planned | n/a | n/a |
| Llama3 scaling | planned | planned | planned | planned | planned | n/a | n/a |
| MLA | planned | planned | planned | planned | planned | planned latent | planned |
| Cross-attention | planned | planned | planned | planned | planned | separate memory | optional |
| Block sparse | planned | planned | planned | planned | planned | yes | planned |
| NSA-style sparse | planned | planned | planned | planned | planned | yes | planned |
| GatedDeltaNet | existing | existing | partial/verify | existing | verify | recurrent | n/a |
| KDA | planned | planned | planned | planned | planned | recurrent | n/a |
| Mamba2 | existing | existing | fallback/current | existing/verify | verify | recurrent | n/a |
| Attention sinks | planned runtime policy | planned | n/a | planned | n/a | yes | compatible |
| Context parallel | backend-only | n/a | n/a | planned | planned | distributed | compatible |

Update this matrix only from tests/capability evidence. Do not mark a cell complete merely because a type exists.

---

# 16. Descriptor requirements created by these primitives

The model-agnostic importer must be able to construct every semantic type above without arbitrary scripting.

Descriptor language must support at least:

- metadata lookup;
- typed defaults;
- integer/float/string/bool validation;
- list lookup;
- per-layer selection;
- bounded conditionals;
- arithmetic for dimensions;
- enum/variant construction;
- tensor-role binding patterns;
- repeated layer templates;
- layer schedule generation;
- explicit capability/provenance data.

Examples of legitimate descriptor logic:

```text
if layer_types[layer] == "sliding_attention"
    pattern = sliding(window)
else
    pattern = full_causal
```

```text
rope.scaling = yarn(
    factor = metadata.rope_scaling.factor,
    original_context = metadata.rope_scaling.original_max_position_embeddings,
    beta_fast = metadata.rope_scaling.beta_fast,
    beta_slow = metadata.rope_scaling.beta_slow
)
```

Not legitimate:

- arbitrary C++ callbacks for every family;
- shell execution;
- network access;
- unrestricted scripting;
- family-specific hot-path hooks.

---

# 17. Validation requirements

The IR must reject impossible or ambiguous semantics before backend compilation.

At minimum validate:

- Q heads / KV heads / dimensions;
- rotary dimensions and factor-vector lengths;
- context-extension original/trained context values;
- sliding-window bounds;
- sparse block sizes;
- cross-attention source compatibility;
- MLA rank/projection dimensions;
- recurrent-state dimensions;
- backend-independent MoE invariants;
- position section totals for multi-axis RoPE;
- tensor binding shapes against semantic requirements.

Backend compilation then validates backend-specific support:

- kernel availability;
- supported storage formats;
- device capabilities;
- workspace limits;
- distributed requirements.

---

# 18. Testing strategy

## 18.1 Primitive unit tests

Every new primitive/policy gets deterministic tiny-shape tests independent of real model names.

## 18.2 Reference implementation tests

Prefer a slow, obvious reference implementation for new mathematics before optimized CPU/CUDA paths.

## 18.3 CPU/CUDA parity

For supported primitives compare:

- token outputs;
- chunk outputs;
- logits;
- recurrent/attention state;
- page contents where stable enough;
- position transforms.

## 18.4 Chunk partition invariance

For any primitive supporting chunk execution:

```text
1 + 1 + 1 + ...
4 + 4 + ...
17 + 31 + ...
whole prompt
```

must produce equivalent semantics within tolerance.

## 18.5 Storage-format invariance

Quantized KV/state tests should compare against higher-precision references with explicit tolerances.

## 18.6 Model proof fixtures

Use real models as integration proofs, not as locations for implementation logic.

Recommended sequence:

1. Qwen3-0.6B: prove descriptor/importer path with existing dense-attention mathematics;
2. Ministral-8B-Instruct-2410: prove per-layer full/sliding attention schedules without family C++;
3. a YaRN/Ministral-3-class model: prove explicit RoPE scaling semantics;
4. Phi-4 Mini-class model: prove LongRoPE/partial rotary as generic position primitives;
5. DeepSeek-V2/V3-class model: prove MLA;
6. Kimi Linear-class model: prove modern linear attention primitive extension;
7. multimodal/cross-attention model: prove multi-source attention.

The names above are test targets/provenance only. Production execution must not depend on those names.

---

# 19. Static architecture checks

Extend `scripts/check_architecture_boundaries.py` or successor checks to enforce:

- no family identity in backend execution;
- no `model_type` dispatch in CPU/CUDA;
- no model-family includes from generic IR/runtime/backend roots;
- no new `src/models/<family>` once the descriptor path is ready for that class of model;
- no RoPE algorithm chosen by family name;
- no attention kernel chosen by family name;
- no cache layout chosen by family name;
- no compatibility `_v2` stacking in CELEG-owned APIs.

Eventually add the stronger invariant:

> Adding a model that uses an already-supported primitive set must not modify production C++.

A CI fixture should prove this mechanically.

---

# 20. Recommended implementation order

The order below minimizes confounding variables while moving toward modern model coverage.

## Milestone 1 - Position IR

1. introduce first-class `PositionSpec`;
2. migrate existing RoPE/no-position behavior;
3. move partial rotary into the position specification;
4. move M-RoPE semantics into the canonical IR;
5. keep CPU/CUDA behavior unchanged.

## Milestone 2 - Context scaling

1. linear/position interpolation;
2. dynamic NTK;
3. YaRN;
4. LongRoPE;
5. Llama3 frequency-aware scaling;
6. common reference-frequency generator;
7. CPU/CUDA parity tests.

## Milestone 3 - Pattern IR

1. full causal;
2. sliding window;
3. bidirectional;
4. prefix-LM;
5. block sparse representation;
6. per-layer pattern schedules.

## Milestone 4 - Model-agnostic dense proof

1. generic descriptor/importer path;
2. Qwen3-0.6B with zero family C++;
3. Ministral-8B with zero family C++;
4. delete any temporary family-specific implementation created during experiments.

## Milestone 5 - Attention state abstraction

1. derive persistent state layout from primitive semantics;
2. remove equal-width ordinary-KV assumptions from page planning;
3. generalize state precision/capabilities;
4. preserve prefix/paged infrastructure.

## Milestone 6 - MLA

1. semantic primitive;
2. reference CPU path;
3. paged latent state;
4. CUDA correctness path;
5. optimized lowering;
6. DeepSeek-class descriptor proof.

## Milestone 7 - Cross-attention

1. graph value/source model;
2. persistent encoder/vision memory;
3. CPU/CUDA lowerings;
4. multimodal proof.

## Milestone 8 - Sparse attention

1. block-sparse foundation;
2. dynamic selection plan;
3. NSA-style components;
4. CPU reference;
5. CUDA lowering.

## Milestone 9 - Modern linear attention

1. factor shared recurrent semantics from current GatedDeltaNet/Mamba experience;
2. KDA semantic primitive;
3. token/chunk contracts;
4. Kimi-class proof.

## Milestone 10 - Runtime long-context scale

1. broader KV/state quantization;
2. streaming attention sinks;
3. paged state for all applicable attention forms;
4. context-parallel lowering;
5. distributed/ring-style execution where justified.

---

# 21. Definition of done

This roadmap is complete when all of the following are true.

## Semantic completeness

- modern position-scaling algorithms are explicit IR semantics;
- ordinary, sliding, sparse, latent, recurrent/linear, and cross-attention categories have reusable primitive boundaries;
- M-RoPE is first-class IR rather than family topology state;
- model graph semantics do not encode checkpoint tensor names.

## Backend completeness

- CPU and CUDA compile supported primitives from the same canonical semantics;
- token/chunk paths do not independently reinterpret model meaning;
- paged state supports more than ordinary BF16 K/V;
- state precision is backend capability/policy;
- long-context prefill consumes compiled state/layout plans.

## Open/Closed proof

- Qwen3-class and Ministral-class models using existing mathematics load with no model-family production C++;
- models needing YaRN/LongRoPE add position primitives once, then use descriptors;
- DeepSeek-class MLA support adds MLA once, not DeepSeek execution code;
- Kimi-class support adds KDA once, not Kimi execution code;
- future models composed of those primitives require descriptor/binding/test additions only.

## Runtime scale

- paged KV/state remains supported;
- prefix caching remains supported;
- quantized state can materially reduce long-context memory;
- streaming sessions can use bounded cache policies;
- distributed/context-parallel execution can be added without changing Model IR semantics.

---

# 22. Final invariant

The end state should satisfy this statement:

> CELEG does not support model families. CELEG supports a vocabulary of mathematical primitives, positional policies, state representations, and backend lowerings. External model descriptions are imported into that vocabulary.

That is the implementation foundation required to eventually delete `src/models/**` and keep it deleted.