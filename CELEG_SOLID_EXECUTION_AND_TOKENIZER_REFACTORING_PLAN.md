# CELEG SOLID, CPU Execution, and Tokenizer Refactoring Plan

Status: proposed execution plan  
Baseline: `master` at `05b94cd5dfce98a5be50715ad3c8c591338a1a14`  
Scope: architecture/SOLID review, CPU forward execution, tokenizer architecture, runtime composition, Qwen3.5 vision, and backend extensibility

## 1. Purpose

CELEG has already moved substantially away from the earlier architecture in which model families, backends, checkpoint details, and execution policy were tightly coupled. The current runtime has useful extension boundaries such as `IArchitecture`, `IBackendFactory`, `ITokenizerProvider`, `IVisionProviderFactory`, capability-oriented checkpoint repository interfaces, and focused CPU/CUDA model facades.

The remaining debt is no longer mainly about obvious family-specific switches. It is now concentrated in duplicated semantic representations, large execution units that reimplement model behavior in multiple paths, format/model knowledge inside generic tokenizer code, duplicated built-in composition roots, and a few concrete dependencies that weaken otherwise good dependency inversion.

This document turns the current review into an ordered refactoring program. The order is intentionally correctness-first: places where two execution paths may produce different model semantics must be protected and corrected before large structural changes.

The plan covers three major review rounds:

1. project-wide SOLID architecture and runtime composition;
2. `src/text/tokenizer.cpp` and tokenizer extensibility;
3. `src/backend/cpu/model_forward.cpp` and CPU execution semantics.

It also includes the related findings around `RuntimeTopology`, `ModelGraph`, `CompiledModelProgram`, Qwen3.5 vision, backend API extensibility, and the existing strengths that should not be regressed.

---

## 2. Architectural goals

The target architecture should satisfy the following invariants.

### 2.1 One canonical model semantics representation

A model family should interpret checkpoint metadata once and produce one backend-neutral semantic model. Derived execution information must not become an independently maintained second model description.

Target direction:

```text
Checkpoint
    |
    v
Architecture resolution
    |
    v
ModelGraph / canonical resolved semantics
    |
    v
CompiledModelProgram
    |
    +--------------------+
    |                    |
    v                    v
CPU compiler         CUDA compiler
    |                    |
    v                    v
CPU executable       CUDA executable
```

`RuntimeTopology` may continue to exist where useful, but it should become a derived execution view instead of a second source of truth that architecture modules populate independently from `ModelGraph`.

### 2.2 One semantic implementation per operation

Token decode and chunked prefill may use different kernels and batching strategies, but they must execute the same compiled semantics.

Target direction:

```text
Compiled layer semantics
        |
        v
CPU operator executor
   +----+----+
   |         |
 token     chunk
 kernel    kernel
 path      path
```

The operator owns shared semantics. `forward_token()` and `forward_chunk()` must not independently reconstruct what attention, MoE, Mamba2, GatedDeltaNet, residual ordering, positional encoding, or per-layer input mean.

### 2.3 Generic tokenizer code must not know model-family names

The BPE engine should consume explicit tokenizer behavior, not infer behavior from strings such as `lfm2`, `smaug-bpe`, `gemma4`, or `granite`.

Target direction:

```text
Checkpoint / tokenizer.json
          |
          v
Tokenizer definition resolver
          |
          v
TokenizerDefinition
  - vocabulary
  - merges
  - special tokens
  - normalization
  - pre-tokenization
  - byte encoding/fallback
  - BOS/EOS/PAD policy
          |
          v
BpeTokenizer
```

Adding a new model family must not require adding `if (model == ...)` branches to the generic BPE engine.

### 2.4 Family registration has one composition boundary

A model family should contribute its coherent runtime capabilities from one module rather than being manually added to multiple central lists.

A family module may register:

- architecture resolver;
- chat profile/template;
- tool-call codec;
- tokenizer provider/configuration where family-owned;
- vision provider;
- other family-owned capabilities.

### 2.5 Hot paths stay efficient

SOLID refactoring must not imply virtual dispatch or heap allocation inside every token/layer operation. Prefer compile-time or load-time dispatch construction, function pointers, tagged executable programs, or other zero/low-overhead execution plans where appropriate.

---

## 3. Existing strengths to preserve

The following design improvements are valuable and should be treated as architectural constraints during the refactor.

### 3.1 Capability-oriented checkpoint repositories

Keep the separation between:

- `IWeightRepository`;
- `ILocatableTensorRepository`;
- `IRandomAccessTensorReader`;
- `INativeBlockStorageRepository`;
- `ITokenizerDataRepository`.

Do not collapse these back into one large interface with unsupported/default methods. This is a strong ISP/LSP boundary.

### 3.2 Runtime provider interfaces

Keep the extensibility direction represented by:

- `IArchitecture`;
- `IBackendFactory`;
- `ITokenizerProvider`;
- `IVisionProviderFactory`;
- `IRuntimeModule`.

The work below should remove remaining central registration knowledge rather than replace these interfaces.

### 3.3 Focused CPU/CUDA public facades

Preserve the separation of operational surfaces such as:

- inference/session operations;
- diagnostics;
- persistence.

Avoid re-expanding `CpuModel` or `CudaModel` into monolithic public APIs.

### 3.4 Backend-neutral architecture resolution

Backends should continue to consume resolved semantics and must not inspect architecture IDs to choose family behavior.

---

# Phase 0 - Establish correctness baselines before refactoring

Priority: critical  
Risk: low  
Purpose: make semantic divergence observable before restructuring hot execution code.

## 0.1 Add CPU token-vs-chunk equivalence tests

Create golden/equivalence tests that execute the same prompt through:

1. repeated `forward_token()` calls;
2. one or more `forward_chunk()` calls;
3. different chunk boundaries for the same input.

Compare, where accessible:

- final logits;
- hidden output or another internal deterministic checkpoint in test builds;
- KV state/token counts;
- convolution state;
- GatedDeltaNet recurrent and convolution state;
- Mamba2 state where applicable;
- session position;
- next RoPE/M-RoPE coordinates;
- prefix snapshot/exported state where practical.

Use numerical tolerances appropriate for the same CPU kernels. For paths that should be bit-identical, make the assertion strict.

## 0.2 Cover representative model semantics

At minimum include fixtures for:

- dense attention model;
- short-convolution model;
- GatedDeltaNet model;
- MoE model with ordinary routed experts;
- MoE model with a shared expert;
- model using query gate;
- model using split attention/feed-forward norms;
- model using per-layer input;
- M-RoPE / multimodal position input;
- Mamba2 / Nemotron-H style hybrid execution.

These tests should be architecture-resolution fixtures where possible, not hand-built approximations that can drift away from real resolved programs.

## 0.3 Add chunk-boundary invariance tests

For the same prompt, compare partitions such as:

```text
[N]
[1, N-1]
[2, N-2]
[3, 5, ...]
[1, 1, 1, ...]
```

The semantic result must not depend on the chunking strategy.

## 0.4 Add tokenizer golden tests before moving code

Capture current expected behavior for every supported tokenizer family/profile, including:

- ordinary ASCII;
- multilingual UTF-8;
- punctuation;
- Unicode numeric ranges;
- contractions;
- whitespace/newlines;
- leading spaces;
- byte fallback;
- special/control tokens;
- `<think>` / `</think>` behavior where currently expected;
- BOS/EOS/PAD behavior;
- Gemma `▁` normalization path;
- direct vocabulary hits and BPE merge paths;
- encode/decode round trips where valid.

## 0.5 Acceptance gate

Do not begin structural CPU-forward extraction until these tests exist and failures are understood.

---

# Phase 1 - Correct CPU token/chunk semantic divergence

Priority: critical  
Risk: high if left unresolved, moderate to fix with tests

The current `src/backend/cpu/model_forward.cpp` contains two largely independent implementations of model semantics: `forward_token()` and `forward_chunk()`. The immediate goal is correctness parity before architectural decomposition.

## 1.1 Verify and fix shared-expert MoE parity

The token path executes shared-expert logic after routed experts. The reviewed chunk path performs routed expert grouping/execution but does not show an equivalent shared-expert contribution.

Actions:

- add a failing equivalence test using a resolved model with `MoeLayerProgram::shared`;
- confirm whether chunked prefill currently omits the shared expert;
- implement the same shared-expert semantics in chunk execution;
- preserve the declared combine order and gate semantics;
- verify routed scaling and selected-expert normalization remain identical between token and chunk paths;
- verify both memory-resident and disk-cached expert paths.

Acceptance:

- token and chunk execution match for shared-expert models;
- CPU capability declaration `shared_experts = true` is true for all supported CPU execution modes, not only token decode.

## 1.2 Verify and fix M-RoPE parity

The token path accepts three-axis RoPE coordinates and uses M-RoPE kernels when configured. The reviewed chunk attention path uses scalar positions and ordinary RoPE helpers.

Actions:

- add Qwen3.5 multimodal/M-RoPE token-vs-chunk tests;
- make chunk execution consume per-position three-axis coordinates from `PromptEmbedding`;
- use the same `mrope_section` and interleaving semantics as token execution;
- advance `next_rope_position` consistently after chunk execution;
- verify mixed ordinary-token and visual-token prompts.

Acceptance:

- M-RoPE logits/state are invariant to token vs chunk execution;
- chunking a multimodal prompt cannot silently downgrade M-RoPE to scalar RoPE.

## 1.3 Audit all remaining semantic differences

Create a checklist that compares both paths for:

- query-key normalization;
- query scaling;
- query gating;
- positional encoding `None`;
- sliding attention masks;
- shared KV ownership;
- residual multiplier;
- split attention norms;
- feed-forward activation selection;
- final norm and logits divisor;
- final logit softcap;
- raw prompt embeddings;
- per-layer input;
- Mamba2 state updates;
- GatedDeltaNet state updates;
- short convolution state;
- expert backing initialization and cache behavior;
- profiling instrumentation where it affects control flow.

Any semantic difference must be either removed or documented as an intentional execution-mode restriction enforced before execution.

## 1.4 Remove implicit architecture inference from CPU forward

Current code derives a `nemotron` behavior from the presence of Mamba2 layers. This is not a valid generic semantic rule.

Replace architecture-like inference such as:

```text
mamba2_layer_count > 0 => Nemotron-H block behavior
```

with explicit compiled layer semantics.

The compiled program should state, per layer, whether it has:

- mixer execution;
- post-mixer residual;
- post-mixer normalization;
- feed-forward execution;
- post-feed-forward residual;
- per-layer input processing.

Acceptance:

- no CPU execution rule uses an inferred family identity;
- adding another Mamba2-based architecture cannot accidentally inherit Nemotron-H block ordering.

---

# Phase 2 - Decompose CPU execution into compiled operator executors

Priority: very high  
Risk: moderate after Phase 0/1 tests

## 2.1 Make `model_forward.cpp` an orchestrator

The file should stop implementing detailed math for every mixer and feed-forward type.

Move operator-specific execution into focused units, for example:

```text
src/backend/cpu/operators/
    attention.cpp
    short_convolution.cpp
    gated_delta_net.cpp
    mamba2.cpp
    mlp_only.cpp

src/backend/cpu/feed_forward/
    dense.cpp
    moe.cpp
```

Names may change to fit current project conventions. The important constraint is responsibility, not directory aesthetics.

## 2.2 Introduce CPU compiled layer execution data

Compile resolved semantics into a CPU-specific executable form once.

Possible shape:

```cpp
struct CpuCompiledLayer {
    CpuTokenExecutor token_executor;
    CpuChunkExecutor chunk_executor;
    CpuLayerWeights weights;
    CpuLayerExecutionSpec execution;
};
```

Function pointers or another low-overhead dispatch mechanism are preferred over per-token polymorphic allocation.

The CPU compiler/loader should resolve the operator executor before the hot loop.

## 2.3 Centralize attention semantics

Create one CPU attention semantic implementation shared by token and chunk paths. It should own decisions for:

- Q/K/V projection layout;
- QK norm;
- ordinary RoPE;
- M-RoPE;
- no-positional-encoding mode;
- query scaling;
- query gate;
- KV store ownership;
- sliding/full causal behavior.

Token and chunk methods may call different kernels, but should consume the same `AttentionProgram` / compiled spec.

## 2.4 Centralize MoE semantics

Create a CPU MoE executor shared semantically by token and chunk paths.

It must own:

- router score mode;
- bias application;
- top-K/group selection;
- normalization;
- routed scaling;
- route ordering/grouping;
- resident vs disk-cached expert acquisition;
- expert activation;
- shared expert;
- output combine order.

Do not duplicate router math in token and batch implementations.

Where batch execution needs a different data layout, factor the common semantic decisions from the batching mechanics.

## 2.5 Centralize recurrent mixer semantics

Apply the same pattern to:

- GatedDeltaNet;
- Mamba2;
- short convolution.

State ownership and update order are part of the semantic contract and must be testable independently of `CpuCompiledModel::forward_*`.

## 2.6 Reduce `CpuWorkspace` coupling

`CpuWorkspace` currently knows scratch requirements for all supported operator kinds. After operator extraction, move toward compile-time/load-time workspace requirements.

Possible direction:

```text
CpuCompiledModel
    |
    +-- common scratch
    +-- operator scratch plan
    +-- MoE scratch plan
```

Do not prematurely create per-layer heap allocations. Keep reusable contiguous buffers where performance requires them, but calculate their dimensions from the compiled CPU program rather than from a growing set of `RuntimeTopology::max_*` helpers.

## 2.7 Acceptance gate

- `model_forward.cpp` contains orchestration and lifecycle flow, not implementations of all operator math;
- adding a new mixer does not require implementing its semantics twice in `forward_token()` and `forward_chunk()`;
- all Phase 0 equivalence tests remain green;
- benchmark regression is measured before/after;
- no material decode/prefill regression is accepted without an explicit tradeoff decision.

---

# Phase 3 - Make model semantics canonical and derive topology

Priority: very high  
Risk: high  
Dependency: Phase 1 correctness baseline and preferably Phase 2 execution boundaries

The current architecture can express the same semantics in `RuntimeTopology`, `ModelGraph`, `CompiledModelProgram`, and concrete backend weight-layer variants. This multiplies validation requirements and allows representations to disagree.

## 3.1 Define the canonical ownership rule

Recommended ownership:

- `ModelGraph` / resolved model semantics: canonical backend-neutral semantics;
- `WeightPlan`: canonical requested tensor roles/shapes linked to those semantics;
- `CompiledModelProgram`: derived executable semantic program;
- `RuntimeTopology`: derived convenience/index/layout data only;
- CPU/CUDA executable structures: derived backend-specific implementation data.

Architecture modules should not independently populate two semantic models.

## 3.2 Stop rebuilding graph semantics from topology

Architecture resolution currently often follows a pattern like:

```text
decode_topology(metadata)
build_graph(topology, metadata)
build_weights(topology/graph)
```

Move toward:

```text
resolve canonical model semantics from metadata
        |
        +--> derive topology/index maps
        +--> derive weight plan
```

Each semantic fact should be assigned once.

Examples of facts that should have one owner:

- mixer kind/spec;
- attention head layout;
- positional encoding;
- feed-forward type;
- activation;
- MoE router semantics;
- shared-expert semantics;
- per-layer input;
- residual behavior;
- norm semantics.

## 3.3 Shrink `RuntimeTopology`

`RuntimeTopology` currently contains both shape/index information and semantic policy. Split or derive fields so it no longer acts as a second graph.

Candidate derived/index concerns that can remain in a topology view:

- layer counts;
- attention slot maps;
- KV ownership/index maps where genuinely runtime-derived;
- maximum workspace widths;
- compact execution lookup tables.

Semantic fields should come from the graph/program rather than be duplicated as parallel vectors.

## 3.4 Remove backend reads from both topology and graph for the same decision

A backend operation should not need to ask all of:

```text
RuntimeTopology
CompiledModelProgram
WeightLayer variant
```

to determine one layer's semantics.

By the end of this phase, hot execution should primarily consume backend-compiled layer data produced from `CompiledModelProgram`.

## 3.5 Strengthen validation at conversion boundaries

Validation should be strongest at transitions:

```text
checkpoint -> resolved model
resolved model -> compiled program
compiled program -> backend executable
```

Avoid relying on many cross-checks between independently mutable duplicate representations.

## 3.6 Acceptance gate

- architecture implementations assign each semantic property once;
- `RuntimeTopology` can be regenerated from canonical resolved semantics;
- backend compilation does not reconstruct family semantics from miscellaneous topology flags;
- validation complexity decreases rather than increases;
- all architecture resolution tests remain green.

---

# Phase 4 - Refactor `src/text/tokenizer.cpp` into data-driven tokenizer semantics

Priority: high  
Risk: moderate  
Dependency: Phase 0 tokenizer golden tests

The current `BpeTokenizer` combines JSON parsing, UTF-8 decoding, Unicode categorization, pre-tokenization, BPE merging, byte encoding, special-token rules, normalization, family-specific policy selection, encode, and decode.

## 4.1 Introduce an explicit `TokenizerDefinition`

Replace family-name-driven behavior with explicit behavior data.

Possible model:

```cpp
struct TokenizerDefinition {
    Vocabulary vocabulary;
    MergeTable merges;
    SpecialTokenSet special_tokens;
    NormalizationSpec normalization;
    PreTokenizerSpec pre_tokenizer;
    ByteEncodingSpec byte_encoding;
    TokenIdPolicy token_ids;
};
```

The exact types can remain value types/variants. Do not introduce virtual interfaces unless multiple implementations actually need runtime polymorphism.

## 4.2 Replace historical booleans with behavior

Remove policy fields whose names describe a model rather than behavior, such as:

```text
lfm2_rules
granite_rules
gemma_normalization
```

Represent the actual rule instead, for example:

- contraction case sensitivity;
- optional leading-space attachment;
- maximum numeric run length;
- punctuation grouping;
- newline attachment behavior;
- raw UTF-8 vs GPT-2 byte encoding;
- byte fallback format;
- whitespace marker replacement;
- direct-vocabulary-before-merge behavior.

This prevents invalid boolean combinations and makes new tokenizer variants composable.

## 4.3 Move tokenizer JSON parsing out of `BpeTokenizer`

`BpeTokenizer` should not include or depend directly on a concrete JSON parser.

Introduce a loader/resolver boundary such as:

```text
TokenizerJsonLoader -> TokenizerDefinition -> BpeTokenizer
```

The existing checkpoint tokenizer provider can use this loader.

## 4.4 Upgrade `TokenizerData`

The current format-neutral `TokenizerData` still carries a `pre_tokenizer` string that is interpreted as family/profile magic.

Replace this with normalized semantics or with enough source data for a dedicated resolver to produce `TokenizerDefinition`.

Goal:

```text
GGUF/safetensors/tokenizer.json source differences stop at the provider/loader boundary.
```

## 4.5 Extract UTF-8 and Unicode helpers

Move generic responsibilities out of the tokenizer implementation:

- UTF-8 codepoint iteration/encoding;
- Unicode whitespace classification;
- numeric classification;
- punctuation/symbol classification.

Keep these utilities dependency-light and independently testable.

Do not pretend to implement the entire Unicode standard if CELEG intentionally uses a bounded implementation. Document the supported classification behavior and keep golden coverage.

## 4.6 Extract byte-unicode codec

Move GPT-2 byte-to-Unicode mapping and decode logic into a focused component/value object.

It should be reusable by any tokenizer definition selecting that byte encoding.

## 4.7 Extract BPE merge engine

`bpe_symbols()` and merge-rank execution should become a focused BPE engine whose inputs are vocabulary/merge data and symbols.

It should not know:

- model family;
- BOS/EOS tokens;
- JSON;
- Gemma normalization;
- chat protocol.

## 4.8 Extract special-token matching

Build a dedicated special-token matcher/set with deterministic longest/earliest semantics.

Move hard-coded reasoning delimiter knowledge out of the generic tokenizer. If `<think>` / `</think>` must be treated verbatim for a specific profile, the tokenizer definition/provider should declare that explicitly.

## 4.9 Move BOS/EOS/PAD convention inference to loaders/providers

Generic BPE encode/decode should consume resolved token IDs. It should not infer them from strings such as:

- `<|startoftext|>`;
- `<|begin_of_text|>`;
- `<|im_end|>`;
- `<eos>`;
- `<pad>`.

Those conventions belong to the source-format/profile resolver.

## 4.10 Suggested physical split

A reasonable end state could resemble:

```text
src/text/tokenizer/
    tokenizer.cpp
    bpe.cpp
    byte_unicode.cpp
    unicode.cpp
    pretokenizer.cpp
    normalization.cpp
    special_tokens.cpp
    tokenizer_json_loader.cpp
```

The exact split may be smaller. Avoid one-class-per-file ceremony; split where responsibilities and tests benefit.

## 4.11 Acceptance gate

- `src/text/tokenizer.cpp` contains no family-name string branches;
- adding a new tokenizer behavior is done by resolving a definition/spec, not editing the BPE engine;
- `BpeTokenizer` no longer parses JSON itself;
- all golden tokenizer tests remain green;
- encode/decode performance is benchmarked for regressions.

---

# Phase 5 - Unify built-in family composition

Priority: high  
Risk: low to moderate

The project currently has overlapping central knowledge of built-in architectures in both architecture registration and runtime-module construction.

## 5.1 Make one runtime module the family composition unit

A family should expose one module/factory, for example conceptually:

```cpp
std::unique_ptr<IRuntimeModule> make_qwen35_module();
```

That module registers every family-owned capability.

Do the same for all built-in families.

## 5.2 Remove duplicated architecture lists

Eliminate the need to manually maintain both:

- a central `add_builtin_architectures()` family list;
- a separate `make_builtin_runtime_modules()` architecture-family list.

There should be one built-in module collection/composition root.

## 5.3 Keep generic services separate

Generic built-ins that are not owned by one model family may remain separate modules, for example generic checkpoint formats or truly generic tokenizer providers.

Do not force unrelated capabilities into a family module merely to achieve a single vector.

## 5.4 Add extension tests

Create a test-only runtime module that registers a fake architecture/provider without modifying central production switches/lists.

Acceptance:

- adding a family implementation requires adding its module to one intended composition point only;
- no second architecture registry list must be updated;
- external/custom runtime modules continue to work.

---

# Phase 6 - Decompose Qwen3.5 vision and restore dependency inversion

Priority: medium-high  
Risk: moderate

The current Qwen3.5 vision implementation mixes image decoding, Base64, platform codec behavior, resize, tensor interpretation, CPU math helpers, model-specific vision topology, checkpoint repository construction, and provider behavior.

## 6.1 Separate image ingestion from model encoding

Extract responsibilities such as:

```text
ImageDecoder
ImageResizer
Qwen35VisionEncoder
Qwen35VisionProvider
```

Platform-specific image codecs should not live inside model-specific encoder logic.

## 6.2 Separate generic CPU vision math

Move reusable operations such as simple LayerNorm/linear helpers to an appropriate CPU vision/tensor utility only if reuse or testability justifies it.

Do not create a generic framework prematurely. The goal is to remove unrelated image/platform responsibilities from the Qwen encoder.

## 6.3 Inject repository abstraction

The core Qwen3.5 vision encoder should depend on `IWeightRepository` or another narrow weight-view abstraction rather than directly constructing `SafeTensorRepository`.

The provider/composition boundary should select the concrete repository.

This enables:

- fake repository unit tests;
- future checkpoint storage options;
- cleaner DIP.

## 6.4 Make image codec support explicit

Current platform-specific decoding behavior should become a provider/capability concern. Unsupported codecs must fail at a clear boundary with tests.

## 6.5 Acceptance gate

- Qwen35 encoder tests can run with an injected/fake weight source;
- model-specific code does not own Windows codec plumbing;
- image decoding/resizing has focused tests;
- existing Qwen3.5 vision behavior remains covered.

---

# Phase 7 - Make backend selection extensible beyond CPU/CUDA in the public API

Priority: medium  
Risk: ABI/API design risk  
Dependency: internal backend factory architecture should remain stable first

Internally, `BackendId` and `IBackendFactory` provide a much more extensible direction than the C API, which is still closed around CPU/CUDA enums and unions.

## 7.1 Preserve current API for compatibility

Do not break existing C callers merely to make the design prettier.

Keep the current API stable while designing a v2 extension mechanism.

## 7.2 Design backend-extensible API v2

Evaluate a representation such as:

```c
const char* backend_id;
const void* backend_options;
uint32_t backend_options_size;
```

or a typed property/configuration API.

Requirements:

- ABI versioning/size fields;
- unknown backend reporting;
- backend option validation at the owning backend factory;
- no central CPU/CUDA `if/else` required to instantiate future backends;
- support for potential ROCm, Metal, Vulkan, WebGPU, or other backends without redesigning the core API again.

## 7.3 Move backend-specific option construction behind backend-owned code

The generic service creation path should select a backend factory by ID and delegate option interpretation/construction rather than branching on every backend enum.

## 7.4 Acceptance gate

Demonstrate with a fake/test backend that it can be registered and selected through the new internal/public extension path without editing central backend switches.

---

# Phase 8 - Consolidate semantic policies and eliminate remaining family leakage

Priority: medium  
Risk: low after previous phases

## 8.1 Search generic runtime code for family names

Audit generic directories for direct family checks, including:

```text
src/backend/
src/runtime/
src/model/
src/text/
src/serve/
src/api/
```

Family names are acceptable in:

- family implementations;
- explicit built-in composition;
- tests/fixtures;
- user-facing metadata where identity itself is needed.

They are suspicious in generic execution semantics.

## 8.2 Replace semantic booleans that encode historical families

Prefer explicit semantic enums/specs/value objects over flags whose meaning is effectively "this model behaves like family X".

Examples:

- feed-forward presence/order;
- norm placement;
- positional encoding mode;
- router score/selection/normalization;
- tokenizer pre-tokenization behavior.

## 8.3 Avoid speculative abstraction

Do not replace every enum or `std::variant` with a virtual hierarchy.

`std::variant` is appropriate for a deliberately closed backend executable representation when:

- the compiler resolves the variant once;
- dispatch is centralized;
- consumers do not each reimplement the same switch tree.

The goal is not "zero switches". The goal is to place dispatch at one appropriate boundary.

---

# Phase 9 - Testing, performance, and maintainability gates

Every phase above should be guarded by explicit gates.

## 9.1 Correctness

Required suites should include:

- architecture resolution;
- model validation;
- weight-plan validation;
- tokenizer golden tests;
- CPU token/chunk equivalence;
- prefix snapshot/restore where affected;
- MoE routing/shared-expert tests;
- M-RoPE multimodal tests;
- CPU/CUDA capability tests.

## 9.2 Performance

Track at minimum:

- CPU decode tokens/s;
- CPU prefill tokens/s;
- chunked prefill throughput;
- MoE routed/shared-expert throughput;
- disk-cached expert hit/miss behavior;
- tokenizer encode/decode throughput if tokenizer internals change;
- memory/scratch high-water marks.

Refactoring hot paths must compare against a baseline manifest before merge/commit acceptance.

## 9.3 Compile-time and binary dependency health

As internal headers are split:

- reduce `detail/model_internal.hpp` fan-out where practical;
- avoid exposing all CPU operator weight/state types to unrelated translation units;
- keep CUDA headers out of CPU-only paths;
- keep concrete checkpoint format headers out of generic model/text execution code.

## 9.4 Extension tests

Maintain small fake implementations for:

- architecture;
- runtime module;
- backend factory;
- weight repository capabilities;
- vision weight source;
- tokenizer definition/provider.

The tests should prove that extension points work without editing generic runtime code.

---

## 10. Recommended execution order

The phases are intentionally not ordered by conceptual cleanliness. They are ordered by correctness risk and dependency.

### Milestone A - CPU semantic safety

1. Phase 0 CPU equivalence tests.
2. Phase 1 shared-expert verification/fix.
3. Phase 1 M-RoPE verification/fix.
4. Complete token-vs-chunk semantic audit.
5. Remove implicit `nemotron` inference.

Exit criterion: token and chunk execution are semantically equivalent for every advertised CPU feature.

### Milestone B - CPU execution decomposition

1. Extract attention executor.
2. Extract MoE executor.
3. Extract recurrent/conv executors.
4. Introduce compiled CPU layer dispatch.
5. Shrink `model_forward.cpp` to orchestration.
6. Rework workspace planning from compiled requirements.

Exit criterion: a new mixer/feed-forward semantic implementation is added once, with token/chunk kernel variants behind the same executor contract.

### Milestone C - Canonical model IR

1. Declare canonical semantic ownership.
2. Derive topology/index data.
3. Remove graph/topology duplication family by family.
4. Update CPU compiler.
5. Update CUDA compiler.
6. Delete obsolete duplicate fields after migration.

Exit criterion: architecture resolution cannot create conflicting topology and graph semantics.

### Milestone D - Tokenizer architecture

1. Golden tests.
2. `TokenizerDefinition`.
3. Move JSON loading.
4. Replace family booleans with behavior specs.
5. Extract UTF-8/Unicode, byte codec, BPE, normalization, pre-tokenizer, and special-token responsibilities as justified.
6. Remove all family-name branches from generic tokenizer execution.

Exit criterion: a tokenizer variation is described by data/specs/providers rather than a new conditional in `BpeTokenizer`.

### Milestone E - Composition and peripheral DIP cleanup

1. Unify family runtime modules.
2. Remove duplicate built-in architecture registration lists.
3. Decompose Qwen3.5 vision.
4. Inject weight repository into vision encoder.
5. Add extension tests.

Exit criterion: each family has one coherent composition unit and generic runtime code does not directly construct family storage dependencies.

### Milestone F - Public backend extensibility

1. Design API v2.
2. Add test backend path.
3. Delegate backend option parsing/creation.
4. Preserve v1 compatibility.

Exit criterion: adding a backend does not require widening a central backend enum/union in the new API.

---

## 11. File-level target map

This is a target map, not a requirement to perform all file moves immediately.

### Current hotspot: `src/backend/cpu/model_forward.cpp`

Target responsibilities:

```text
model_forward.cpp
    orchestration only

operators/attention.*
operators/short_convolution.*
operators/gated_delta_net.*
operators/mamba2.*
operators/mlp_only.*

feed_forward/dense.*
feed_forward/moe.*

layer_compiler.*
    resolved/compiled semantics -> CPU executable layer
```

### Current hotspot: `src/backend/cpu/detail/model_internal.hpp`

Target:

- keep shared model/session ownership internals;
- move operator-specific weight/state definitions nearer their executor/compiler owners where this reduces coupling;
- avoid a single header becoming the mandatory dependency for all CPU implementation files.

### Current hotspot: `src/text/tokenizer.cpp`

Target responsibilities:

```text
tokenizer.cpp
    high-level encode/decode orchestration

bpe.*
byte_unicode.*
unicode.*
pretokenizer.*
normalization.*
special_tokens.*
tokenizer_json_loader.*
```

### Current hotspot: `include/celeg/model/resolved.hpp`

Target:

- shrink `RuntimeTopology` to derived/index/runtime information;
- move canonical semantics to the model graph/program layer;
- avoid parallel semantic vectors mirroring graph layers.

### Current hotspot: `src/model/runtime_modules.cpp` and architecture built-ins

Target:

- one built-in family-module list;
- family module owns registration of family capabilities;
- remove duplicate architecture registration composition.

### Current hotspot: `src/models/qwen35/vision.cpp`

Target responsibilities:

```text
image decoder/platform codec
image resize/preprocess
Qwen35 vision encoder
Qwen35 provider/composition
```

with repository abstraction injected into the encoder.

### Current hotspot: `src/api/backend_service_factory.cpp` / `include/celeg/api.h`

Target:

- keep v1 compatible;
- introduce backend-extensible v2 composition;
- remove future need for central CPU/CUDA branches in new API paths.

---

## 12. SOLID-specific completion criteria

### Single Responsibility Principle

Consider SRP materially improved when:

- `model_forward.cpp` orchestrates rather than implementing all kernels/semantics;
- tokenizer loading, normalization, pre-tokenization, BPE, Unicode handling, and special-token rules are separately testable responsibilities;
- Qwen vision encoding does not own platform image codec and repository construction;
- canonical model semantics are not duplicated across topology and graph builders.

### Open/Closed Principle

Consider OCP materially improved when:

- a new model family requires one intended module registration point;
- a new tokenizer behavior does not require family-name branches in generic BPE code;
- a new CPU operator is compiled into one executor pair rather than inserted into duplicate token/chunk decision trees;
- a future backend can use the v2 API without changing a central backend enum/union.

### Liskov Substitution Principle

Preserve the current strong direction:

- optional repository capabilities remain separate interfaces;
- providers advertise and enforce capabilities honestly;
- CPU advertised capabilities must apply to all supported execution paths.

Specifically, a backend must not advertise shared-expert or positional-encoding support if chunked prefill silently implements less functionality than token execution.

### Interface Segregation Principle

Preserve small capability interfaces and focused session/diagnostic/persistence surfaces.

Avoid creating broad new interfaces such as a universal tokenizer or operator interface containing methods that not every implementation can support meaningfully.

### Dependency Inversion Principle

Consider DIP materially improved when:

- generic execution depends on compiled semantic abstractions, not architecture IDs;
- tokenizer engines depend on definitions/specs, not JSON or family names;
- Qwen vision encoding depends on a weight-source abstraction, not `SafeTensorRepository` construction;
- service creation depends on backend factories rather than central backend-specific construction logic in the extensible API path.

---

## 13. Non-goals

This plan deliberately does not require:

- replacing every `std::variant` with virtual classes;
- eliminating every `switch` statement;
- creating a plugin ABI for every internal operator;
- abstracting activations/norms without a demonstrated second implementation need;
- sacrificing hot-path performance for textbook object-oriented purity;
- rewriting CPU and CUDA execution simultaneously when a staged migration is safer;
- breaking the current C API before a compatible migration path exists.

The objective is semantic ownership, extension locality, testability, and low coupling—not maximum abstraction count.

---

## 14. Definition of done

The refactoring program is complete when all of the following are true:

1. CPU token and chunk paths are proven equivalent for every advertised semantic feature, including shared MoE and M-RoPE.
2. `src/backend/cpu/model_forward.cpp` is a small orchestration layer over compiled CPU operator/feed-forward executors.
3. No architecture-specific behavior is inferred in CPU execution from unrelated topology facts such as "Mamba2 exists".
4. Resolved model semantics have one canonical representation; topology is derived rather than independently authored.
5. Generic tokenizer code contains no model-family policy branches and does not parse tokenizer JSON directly.
6. New tokenizer behavior is expressed by explicit normalization/pre-tokenization/byte/BPE/special-token specs.
7. Built-in family registration is maintained in one coherent module composition path.
8. Qwen3.5 vision model code no longer owns platform image decoding and directly constructs concrete checkpoint repositories in its core encoder.
9. Repository capability interfaces remain segregated and substitutable.
10. A backend-extensible public API path exists without central CPU/CUDA-only design assumptions.
11. Correctness suites, equivalence tests, and performance benchmarks remain green.
12. Adding a new model family, tokenizer policy, CPU mixer, or backend requires changes localized to its intended extension boundary rather than coordinated edits across generic runtime files.

At that point CELEG should have moved from "good extension interfaces around several internally duplicated semantic engines" to a runtime where model semantics are resolved once, compiled once per backend, and executed consistently across decode and prefill.