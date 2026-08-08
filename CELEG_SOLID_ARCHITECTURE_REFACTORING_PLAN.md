# CELEG SOLID Model-Agnostic Architecture Refactoring Plan

Status: active execution plan  
Baseline reviewed: `master` at `b7ee89fee117c9bee389793cec8aebd19332b4fe`  
Date: 2026-08-07  
Scope: SOLID architecture, model-agnostic semantic IR, importers, data-driven model descriptors, weight binding, primitive/lowering boundaries, CPU execution, quantized weight extensibility, tokenizer/chat/vision ownership, C API cleanup, and architectural enforcement

This revision supersedes the previous contents of this file and `CELEG_SOLID_EXECUTION_AND_TOKENIZER_REFACTORING_PLAN.md` as the primary CELEG architecture plan.

The strategic direction has changed in one important way:

> `src/models/**` and `include/celeg/models/**` are migration scaffolding, not the desired final architecture.

The final goal is not to make model-family modules perfectly modular. The final goal is to make model-family C++ modules unnecessary for every model that can be expressed using primitives CELEG already knows how to execute.

---

# 1. North-star architecture

CELEG should evolve from:

```text
runtime that supports a list of model families
```

into:

```text
compiler/runtime that imports and executes a model IR composed of supported primitives
```

The target data flow is:

```text
External checkpoint
      |
      v
Checkpoint format reader
      |
      v
Metadata + tensor inventory
      |
      v
Generic model importer
      |
      +-----------------------------+
      |                             |
      v                             v
Model descriptor resolution    Weight binding resolution
      |                             |
      +--------------+--------------+
                     |
                     v
             Canonical CELEG Model IR
                     |
                     v
                IR validation
                     |
                     v
             CompiledModelProgram
                     |
           +---------+---------+
           |                   |
           v                   v
      CPU compiler        CUDA compiler
           |                   |
           v                   v
      CPU program         CUDA program
```

Model-family identity may be useful while importing an external checkpoint or recording provenance, but it must not control execution.

The runtime should eventually be able to execute a model without knowing whether the source checkpoint called itself Qwen, Granite, LFM, Gemma, Nanbeige, Nemotron, MiniCPM, or something that did not exist when CELEG was compiled.

---

# 2. The strongest Open/Closed rule

The architectural success criterion is:

## 2.1 New model, existing mathematics

If a newly released model can be expressed entirely with CELEG primitives that already exist, supporting it should require:

- descriptor/import data;
- test fixtures;
- possibly chat/tokenizer metadata;
- possibly weight-name/binding rules;

and **zero new model-family C++ source files**.

No new:

```text
src/models/foo/architecture.cpp
src/models/foo/naming_policy.cpp
include/celeg/models/foo/architecture.hpp
```

should be necessary.

## 2.2 New mathematical primitive

If a new model introduces genuinely new mathematics, CELEG may need a new primitive such as:

```text
NewMixer
NewAttentionVariant
NewStateSpaceOperator
NewRoutingOperator
```

The new code belongs to the primitive/operator and backend lowering boundaries, not to a model-family runtime directory.

The desired extension is:

```text
new mathematics
    |
    v
new generic primitive
    |
    +--> semantic validation
    +--> CPU lowering/kernel
    +--> CUDA lowering/kernel
```

not:

```text
new mathematics
    |
    v
new model-family execution fork
```

## 2.3 New checkpoint storage representation

A new storage representation such as GPTQ, AWQ, packed INT8, FP8, MXFP, or another future encoding belongs to:

- checkpoint repository capability;
- tensor codec;
- loaded weight representation;
- backend linear kernel selection;

It must not create model-family execution logic.

## 2.4 New checkpoint format

A new container/metadata format belongs to an importer/repository boundary.

It must not require changes in model execution.

## 2.5 New backend

A new backend belongs behind backend factory/compiler/lowering interfaces.

It must not require changes in model descriptors or model semantics.

---

# 3. Non-negotiable project directives

## 3.1 No backward-compatibility commitment

CELEG currently has **no source, ABI, API, configuration, file-layout, or internal compatibility commitment** unless a future explicit stability milestone says otherwise.

Refactors replace obsolete interfaces. They do not stack versions or retain migration shims.

Do not preserve CELEG-owned patterns such as:

- `foo_v2`, `foo_v3`;
- `legacyFoo`;
- `deprecatedFoo`;
- `compatFoo`;
- old/new duplicate entry points;
- transitional aliases;
- compatibility typedefs;
- wrappers that exist only to preserve an obsolete signature.

If an interface is replaced, migrate repository callers and remove the superseded interface in the same refactoring series.

Versioning remains valid where it is intrinsic to an external protocol, external file format, third-party ABI, or explicitly declared future CELEG stability contract.

## 3.2 Performance is an architectural constraint

SOLID must not convert decode or prefill into a dynamic interpreter of descriptors.

Descriptors and generic IR are **load-time/import-time abstractions**.

Before hot execution, CELEG should resolve them into compact backend programs.

Prefer:

- load-time compilation;
- pre-resolved function pointers;
- backend-local typed execution records;
- immutable compiled programs;
- contiguous reusable scratch storage;
- zero-allocation steady-state execution;
- static or load-time kernel dispatch;
- precomputed weight bindings;
- precomputed workspace plans.

Avoid in hot loops:

- metadata lookup;
- JSON parsing;
- descriptor evaluation;
- architecture probing;
- checkpoint-format probing;
- string-based operator dispatch;
- repeated `dynamic_cast`;
- per-layer heap allocation;
- generic attribute-map access for every token.

## 3.3 Architecture identity must disappear before backend compilation

CPU and CUDA consume semantics, never family identity.

No backend execution decision may depend on:

```text
model_type
architecture_id
family
Qwen
Granite
LFM
Gemma
Nanbeige
Nemotron
MiniCPM
```

or any future model-family identifier.

## 3.4 Model semantics and checkpoint binding are different responsibilities

The mathematical graph says what must be computed.

The binding plan says where the parameters come from.

Do not encode tensor file names into model semantics.

Do not encode architecture semantics into tensor storage codecs.

## 3.5 Descriptors must be declarative, deterministic, and constrained

The model descriptor system must not become a second embedded general-purpose programming language.

It should support only the operations needed to interpret checkpoint metadata and construct a model graph.

No arbitrary file I/O, networking, process execution, or unrestricted scripting.

---

# 4. Revision of the previous SOLID plan

The previous plan remains directionally correct in several areas, but some assumptions are now explicitly temporary.

## 4.1 Keep

Keep these goals:

- make `ModelGraph` the canonical semantic source;
- eliminate duplicated `RuntimeTopology` semantics;
- compile token/chunk execution rather than implement semantics twice;
- narrow CPU operator dependencies;
- replace the oversized workspace responsibility with a workspace plan/view;
- make quantized weight dispatch extensible;
- eliminate hard-coded family-name boundary checks;
- remove `v2`/legacy compatibility debt;
- preserve capability-oriented repository interfaces;
- preserve backend-neutral semantics;
- preserve load-time resolution for hot-path performance.

## 4.2 Change

The following are now migration mechanisms, not final abstractions:

### `IArchitecture`

Useful today, but the end state should not require one C++ implementation per model family.

### `ArchitectureCatalog`

Useful while family-specific resolvers exist, but expected to shrink or disappear after generic importers/descriptors can produce the canonical IR.

### one runtime module per model family

This is better than centralized switches, but it is still family-centric.

It is an intermediate organization step only.

### family-extension test

The stronger final test is not "a family can be added by adding one module".

The stronger final test is:

> a model using existing primitives can be added without adding or modifying production C++ code.

## 4.3 Explicitly reject as final state

The following would still be insufficient even if beautifully factored:

```text
models/qwen35/
models/granite/
models/gemma4/
models/new_model_2027/
models/new_model_2028/
```

A directory-per-family architecture scales source organization, but it does not reach the desired Open/Closed boundary.

---

# 5. Current code evidence motivating the new direction

The current model-family sources already show that a large fraction of family code is declarative information encoded manually in C++.

For example, a family resolver commonly performs several separable tasks:

1. recognize metadata;
2. read dimensions/policies from metadata;
3. choose layer primitive kinds from metadata;
4. build semantic layer specifications;
5. declare expected tensor roles/shapes;
6. map tensor roles to checkpoint names;
7. set capabilities/provenance/chat/tokenizer identifiers.

Most of these are not model-family runtime algorithms.

They are import rules.

Similarly, current tensor naming policies frequently reduce to a mapping like:

```text
TensorRole + layer + expert -> one or more candidate checkpoint paths
```

That should become binding data rather than one C++ switch per family.

---

# 6. Target responsibility map

The target architecture should assign responsibilities as follows.

```text
checkpoint/
    parse physical containers
    expose metadata
    expose tensor inventory/data capabilities

model/import/
    match external model descriptions
    evaluate declarative descriptors
    build canonical CELEG Model IR
    build logical weight requirements

model/ir/
    own mathematical semantics
    own validation
    own model-level policies

model/binding/
    map logical weight slots to checkpoint tensors
    describe transforms/aliases/ties

model/primitives/
    define supported semantic operations
    validate primitive attributes

runtime/
    compose importers, descriptors, backends, tokenizer/chat providers
    contain no family execution behavior

backend/cpu/
    lower canonical program to CPU execution records
    own CPU kernels, workspace and caches

backend/cuda/
    lower canonical program to CUDA execution records
    own CUDA kernels, workspace and caches
```

Long-term there should be no production responsibility that naturally requires `src/models/<family>/`.

---

# Phase 0 - Lock the new architectural invariants

Priority: critical  
Risk: low  
Purpose: make the direction executable and prevent regression.

## 0.1 Update architecture-boundary checks

Replace family-name enumeration with structural rules.

At minimum enforce:

```text
src/backend/**             must not include celeg/models/**
src/runtime/**             must not include celeg/models/**
src/checkpoint/**          must not include celeg/models/**
src/model/** generic code  must not include celeg/models/**
include/celeg/model/**      must not include celeg/models/**
include/celeg/runtime/**    must not include celeg/models/**
include/celeg/checkpoint/** must not include celeg/models/**
```

During migration, explicit composition/import glue may temporarily reference old family modules.

That allow-list must shrink over time.

## 0.2 Add a model-agnostic extension test

Introduce a test fixture model descriptor that uses only existing primitives and is not represented by a production C++ architecture class.

The test must prove that CELEG can:

- match the fixture metadata;
- construct valid IR;
- bind weights;
- compile a backend program;
- run a minimal forward path;

without adding model-specific `.cpp` or `.hpp` code.

## 0.3 Add compatibility-debt enforcement

Flag newly introduced CELEG-owned API/runtime symbols containing suspicious compatibility versioning such as:

```text
_v2
_v3
legacy
deprecated
compat
```

with narrow allow-lists only for external format/protocol terminology.

## 0.4 Add forbidden execution identity checks

Backend and compiled-program code must fail CI if execution decisions use family/model identity.

## 0.5 Acceptance gate

- no hard-coded list of family names in the architecture boundary checker;
- fake backend remains externally registrable;
- test-only model descriptor works without production family C++;
- new CELEG API version stacking is rejected;
- backend code contains no family-based execution branch.

---

# Phase 1 - Make canonical semantics real

Priority: critical  
Risk: high if postponed  
Purpose: create the foundation required before family code can become data.

Today semantic information overlaps among:

- `RuntimeTopology`;
- `ModelGraph`;
- `CompiledModelProgram`.

That duplication must be removed before a generic importer can become trustworthy.

## 1.1 `ModelGraph` becomes the semantic authority

Target transition:

```text
current:
checkpoint -> RuntimeTopology -> ModelGraph -> CompiledModelProgram

intermediate target:
checkpoint/import -> ModelGraph -> derived RuntimeShape -> CompiledModelProgram

long-term target:
checkpoint/import -> Model IR -> validation -> CompiledModelProgram
```

## 1.2 Stop reconstructing graph semantics from topology arrays

Remove semantic duplication such as parallel ownership of:

- mixer kinds;
- attention layouts;
- feed-forward kinds;
- FFN intermediate sizes;
- recurrent layouts;
- layer semantic flags.

Architecture/import logic should construct canonical layer/primitive semantics directly.

## 1.3 Reduce `RuntimeTopology`

Replace it with a derived runtime shape/view or progressively delete it.

A derived runtime shape may contain:

- global dimensions;
- counts;
- allocation maxima;
- token policy;
- truly global numerical policy;
- lookup tables derived from canonical semantics.

It must not independently describe what every layer does.

## 1.4 Move logical-to-physical layer mapping out of semantics

Mappings such as repeated physical checkpoint layers are binding/import concerns.

Execution should receive resolved logical weights/programs and not care why two logical layers originated from the same physical checkpoint block.

## 1.5 Move boundary operations into canonical graph semantics

Intermediate/final normalization boundaries, residual behavior, layer scaling and similar semantics need one owner.

## 1.6 Acceptance gate

- each mathematical property has one semantic source;
- graph/program building no longer cross-references duplicated semantic arrays;
- `CompiledModelProgram` can be generated from canonical IR plus binding/model policies;
- adding one semantic attribute does not require updating parallel representations.

---

# Phase 2 - Evolve `ModelGraph` into a CELEG Model IR

Priority: critical  
Risk: moderate  
Purpose: define a model language rather than a model-family catalog.

Do not perform a gratuitous rename immediately. `ModelGraph` can evolve in place first.

The conceptual target is an IR whose nodes describe supported mathematical primitives.

## 2.1 Primitive vocabulary

The IR should be able to represent current CELEG semantics without architecture identity.

Examples include:

- token embedding;
- RMSNorm/other supported norms;
- attention;
- causal/sliding masks;
- RoPE/M-RoPE positional semantics;
- query/key normalization;
- query gating;
- short convolution;
- GatedDeltaNet;
- Mamba2/state-space operations;
- dense gated FFN;
- MoE routing;
- grouped top-K;
- shared experts;
- residual add/scale;
- per-layer external input;
- final normalization;
- language-model head;
- multimodal projection/merge primitives as support matures.

Names such as `QwenAttention` or `GraniteMlp` must not become IR primitives.

## 2.2 Separate semantic primitive IDs from backend implementations

A semantic primitive says what operation means.

A backend lowering says how to execute it.

Target concept:

```cpp
struct PrimitiveNode {
    PrimitiveId primitive;
    AttributeBlock attributes;
    std::vector<ValueId> inputs;
    std::vector<WeightSlotId> weights;
};
```

The exact representation may remain typed variants initially.

The important direction is that family identity is not part of a node.

## 2.3 Do not pay dynamic-IR costs in decode

Generic IDs/attributes may exist during import and validation.

CPU/CUDA compilation must lower them into backend-specific typed records before inference.

## 2.4 Primitive registration

Avoid one giant central execution switch if practical.

A primitive module should own:

- primitive identifier/schema;
- semantic validation;
- shape inference where appropriate;
- CPU lowering registration;
- CUDA lowering registration;
- capability declaration.

Adding new mathematics is allowed to extend the primitive language. OCP does not mean the mathematical language can never grow.

## 2.5 Acceptance gate

- every current family graph can be expressed using family-neutral primitives;
- model-family identity is absent from IR validation/execution;
- new primitive support is organized by primitive, not by every model that uses it;
- backend programs are precompiled from IR before hot execution.

---

# Phase 3 - Introduce a constrained model descriptor system

Priority: very high  
Risk: moderate  
Purpose: move family import rules from C++ into declarative data.

The descriptor system translates external checkpoint metadata into canonical CELEG IR.

## 3.1 Descriptor responsibilities

A descriptor may declare:

- match/probe rules;
- metadata aliases/paths;
- required fields;
- optional fields/defaults;
- global dimensions;
- token policy;
- numerical policy;
- layer count;
- per-layer primitive selection;
- per-layer primitive attributes;
- capabilities;
- provenance labels;
- logical weight roles;
- weight-name candidates;
- safe tensor transforms/bindings;
- optional chat/tokenizer/profile references.

It must not execute model math.

## 3.2 Expression language

Provide a small deterministic expression language supporting operations such as:

- metadata field reference;
- integer/float/string/bool literal;
- `+ - * /` where type-safe;
- `min`, `max`;
- boolean comparisons;
- conditional selection;
- array indexing;
- array length;
- enum/string mapping;
- bounded layer/expert comprehensions;
- explicit assertions;
- fallbacks/defaults.

Avoid unrestricted scripting.

## 3.3 Example direction

A Qwen-like hybrid descriptor could conceptually express:

```yaml
match:
  all:
    - path: model_type
      in: [qwen3_5, qwen3_5_moe]
    - path: text_config.model_type
      in: [qwen3_5_text, qwen3_5_moe_text]

globals:
  hidden: ${text_config.hidden_size}
  layers: ${text_config.num_hidden_layers}
  vocab: ${text_config.vocab_size}
  head_dim: ${text_config.head_dim}

layers:
  count: ${globals.layers}
  mixer:
    switch: ${text_config.layer_types[layer]}
    cases:
      full_attention:
        primitive: attention
        attributes:
          query_heads: ${text_config.num_attention_heads}
          kv_heads: ${text_config.num_key_value_heads}
          head_dim: ${globals.head_dim}
      linear_attention:
        primitive: gated_delta_net
        attributes:
          key_heads: ${text_config.linear_num_key_heads}
          value_heads: ${text_config.linear_num_value_heads}
          key_dim: ${text_config.linear_key_head_dim}
          value_dim: ${text_config.linear_value_head_dim}
```

Exact syntax is not prescribed by this plan.

The important property is that this information no longer requires a model-family C++ resolver.

## 3.4 Descriptor compilation

Parse and validate descriptors at startup/build/load time.

Compile expressions into a simple internal representation so model import is deterministic and testable.

Do not repeatedly parse descriptor text for every layer if avoidable.

## 3.5 Descriptor schema versioning

Descriptor schema versioning is an internal data-format concern, not a reason to create CELEG API `v2` symbols.

Until a stability commitment exists, incompatible descriptor schema changes may simply update bundled descriptors and tests.

## 3.6 Acceptance gate

- one existing simple family can be resolved entirely from descriptor data;
- its old architecture C++ can be deleted;
- malformed descriptors fail with actionable validation errors;
- descriptors cannot perform arbitrary code execution;
- descriptors are evaluated only before backend hot execution.

---

# Phase 4 - Make weight binding data-driven

Priority: very high  
Risk: moderate  
Purpose: eliminate one of the largest sources of per-family source files.

Current naming policies frequently implement a switch from `TensorRole` to checkpoint paths.

That is data.

## 4.1 Logical weight slots

Model IR should reference logical weight slots such as:

```text
token_embedding
final_norm
attention.query
attention.key
attention.value
attention.output
ffn.gate
ffn.up
ffn.down
moe.router
moe.expert.gate_up
moe.expert.down
state_space.*
```

The exact existing `TensorRole` representation may be reused/evolved.

## 4.2 Binding templates

A descriptor should be able to map a logical slot to one or more candidate tensor paths, for example:

```yaml
weights:
  attention.query:
    candidates:
      - "model.language_model.layers.{layer}.self_attn.q_proj.weight"
```

Support placeholders such as:

- `{layer}`;
- `{physical_layer}`;
- `{expert}`;
- `{role}` where appropriate.

## 4.3 Safe binding transforms

Some checkpoints require structural transformations.

Represent a constrained set explicitly, for example:

- transpose;
- reshape;
- concatenate;
- split;
- permutation;
- alias/tie;
- repeated physical-layer binding;
- packed expert binding;
- supported quantized storage decode/attach.

Transforms must be typed and validated.

Do not allow arbitrary descriptor code.

## 4.4 Shape validation

Expected logical shapes should derive from canonical IR.

The binding layer validates physical tensors against those requirements and transforms.

A model-family source file should not manually duplicate shape formulas already derivable from IR.

## 4.5 Delete naming policies incrementally

For each migrated family:

1. express bindings declaratively;
2. prove equivalence with old naming-policy tests;
3. remove the C++ naming policy;
4. remove its registration/build entries.

## 4.6 Acceptance gate

- a new tensor naming convention needs descriptor data only;
- weight binding has no model-family switch in generic code;
- repeated physical layers are represented as binding semantics;
- shapes derive from IR where possible;
- no per-family `TensorNamingPolicy` remains for migrated models.

---

# Phase 5 - Build a generic importer pipeline

Priority: very high  
Risk: moderate  
Purpose: replace `IArchitecture` with import-time composition.

## 5.1 Separate physical format from semantic model import

Target:

```text
Safetensors repository -----+
GGUF repository ------------+--> CheckpointView
future repository ----------+         |
                                      v
                              GenericModelImporter
                                      |
                                      v
                                  Model IR
```

A repository exposes data.

An importer interprets semantics.

## 5.2 Descriptor registry

The generic importer should select descriptors based on declarative probe rules.

Do not add a C++ switch over model types.

## 5.3 Avoid central registration edits

Adding a descriptor should not require editing `builtin_runtime.cpp` or another central C++ table.

Possible implementation directions:

- package descriptors as runtime resources discovered from an installed descriptor directory;
- generate an embedded descriptor table automatically from descriptor files during the build;
- generate a source/resource manifest from a directory scan.

The selected implementation should preserve reproducible packaging without manual family registration.

## 5.4 `IArchitecture` becomes an anti-corruption migration layer

During migration:

```text
old family IArchitecture ----+
                             +--> canonical IR
new generic descriptor ------+
```

Once all supported models use generic import:

- remove family `IArchitecture` implementations;
- remove `ArchitectureCatalog` if no longer needed;
- remove architecture registration tables;
- delete `src/models/**` and `include/celeg/models/**`.

## 5.5 Acceptance gate

- importer can select and resolve descriptors without family C++ factories;
- descriptor addition requires no central C++ registry edit;
- migrated families bypass `IArchitecture` entirely;
- final migration can remove `ArchitectureCatalog` without changing backend code.

---

# Phase 6 - Migrate current families as vertical slices

Priority: very high  
Risk: controlled by equivalence tests  
Purpose: prove that the architecture works on real CELEG diversity.

Do not convert all families in one large rewrite.

## 6.1 Migration order

Use capability complexity, not brand importance.

A good sequence is:

1. simplest currently supported dense transformer family;
2. family with different numerical/normalization policy;
3. family with hybrid mixer selection;
4. MoE family;
5. family with repeated/aliased physical layers;
6. state-space/recurrent family;
7. multimodal family;
8. remaining combinations.

Choose the exact first family after verifying current fixture/test coverage.

## 6.2 Per-family migration checklist

For each existing family:

- capture current probe behavior as fixture tests;
- capture current canonical semantic result;
- capture expected logical weight requests;
- capture tensor binding candidates;
- create descriptor;
- resolve through generic importer;
- compare generated IR against old resolver;
- compare weight bindings;
- compare token/chunk forward numerics within tolerance;
- migrate chat/tokenizer/vision references;
- delete architecture C++;
- delete naming-policy C++;
- delete family registration glue;
- remove empty family directory.

## 6.3 Equivalence is semantic, not structural

The new IR does not need to have byte-identical internal structs to the old path.

It must preserve mathematical behavior, model capabilities and required bindings.

## 6.4 Acceptance gate

A migrated model family has no production C++ directory named after that family.

---

# Phase 7 - Tokenizer, chat, and vision must stop requiring family modules

Priority: high  
Risk: moderate  
Purpose: prevent non-execution features from preserving the `models/` architecture indefinitely.

## 7.1 Tokenizer

The generic tokenizer engine already moved in the right direction by using generic pre-tokenizer behaviors rather than family-specific branches.

Continue toward:

```text
checkpoint tokenizer data
        +
optional descriptor tokenizer hints
        |
        v
Generic tokenizer provider
```

Family names must not be embedded in tokenization mechanics.

## 7.2 Chat

Chat templates/tool codecs are presentation/protocol concerns, not model mathematical semantics.

Represent them as independent profiles/resources selected from checkpoint metadata or descriptor references.

Do not keep an architecture C++ module alive only to register a chat profile.

## 7.3 Vision

Vision requires a distinction.

If multimodal behavior is mathematical model structure, represent it in IR using generic multimodal primitives.

If behavior is input preprocessing such as image resize/normalization/patch preparation, keep it behind generic preprocessing/provider interfaces.

Do not encode `if family == Gemma` or `if family == Qwen` in generic runtime execution.

## 7.4 Acceptance gate

- migrated families need no family C++ for tokenizer registration;
- chat profiles are data/provider resources;
- multimodal model math lowers from IR;
- preprocessing is capability/provider based;
- removal of `src/models/**` does not remove chat/tokenizer/vision functionality.

---

# Phase 8 - Compile CPU layers instead of interpreting semantics twice

Priority: critical  
Risk: moderate after IR equivalence coverage  
Purpose: preserve performance while improving SRP and eliminating token/chunk semantic duplication.

This phase remains from the previous plan and becomes even more important once IR is generic.

## 8.1 Introduce backend-compiled layer records

Possible direction:

```cpp
struct CpuCompiledLayer {
    CpuTokenLayerFn token;
    CpuChunkLayerFn chunk;
    CpuLayerWeights weights;
    CpuLayerExecutionSpec execution;
};
```

The exact type is flexible.

The principle is that generic model semantics are lowered once into CPU execution decisions.

## 8.2 One semantic pipeline, multiple execution shapes

Token and chunk paths may use different kernels, but they must originate from the same compiled semantics.

Do not reconstruct model semantics independently in `forward_token()` and `forward_chunk()`.

## 8.3 Sequential chunk fallback is a lowering capability

If a primitive lacks a native chunk implementation, select a generic sequential adapter at backend compile time.

Do not detect architecture identity or infer behavior from family names.

## 8.4 Equivalence tests

Compare:

- repeated token execution;
- one chunk;
- multiple chunk partitions;
- mixed partitions.

Cover all supported primitive combinations including:

- causal attention;
- sliding attention;
- Q/K norm;
- query gate;
- M-RoPE;
- short convolution;
- GatedDeltaNet;
- Mamba2;
- dense FFN;
- MoE;
- shared expert;
- grouped routing;
- per-layer input;
- special normalization boundaries.

## 8.5 Acceptance gate

- `model_forward.cpp` is orchestration, not a second semantic engine;
- token/chunk semantics compile from the same IR;
- no model-family identity enters CPU compilation or execution.

---

# Phase 9 - Narrow CPU operator dependencies

Priority: high  
Risk: moderate  
Purpose: improve DIP/SRP/testability without introducing hot-path polymorphism.

Current focused operators should not receive the entire concrete `CpuCompiledModel` when they only need a few services/views.

## 9.1 Execution contexts

Possible direction:

```cpp
struct CpuExecutionContext {
    CpuLinearEngine& linear;
    CpuThreadPool& pool;
    CpuSessionView session;
    CpuWorkspaceView workspace;
};
```

Use narrower operator-specific views where useful.

## 9.2 Compiled weights separate from global model object

A layer executor should receive its compiled layer weights/metadata rather than navigating unrelated global state.

## 9.3 Acceptance gate

- operator helpers do not require unrelated `CpuCompiledModel` state;
- operators can be unit-tested with focused fixtures;
- no per-token virtual service graph is introduced.

---

# Phase 10 - Replace workspace god-object behavior with planning

Priority: high  
Risk: moderate  
Purpose: separate memory planning/storage from operator semantics.

## 10.1 `CpuWorkspacePlan`

Derive scratch requirements from compiled CPU program.

Conceptually:

```text
CompiledCpuProgram
      |
      v
CpuWorkspacePlan
      |
      v
CpuWorkspaceStorage
      |
      v
CpuWorkspaceView(s)
```

## 10.2 Operators request views, not global scratch knowledge

Attention, MoE, recurrent and FFN operations should receive the scratch slices they need.

## 10.3 Acceptance gate

- workspace sizing does not manually know every model family;
- new primitive declares/plans its own scratch requirements through backend lowering/planning;
- steady-state inference remains allocation-free where intended.

---

# Phase 11 - Make loaded weight representation extensible

Priority: high  
Risk: moderate  
Purpose: prevent quantization growth from becoming the next giant switch.

The arrival of packed INT8 already demonstrates the pressure point.

## 11.1 Resolve storage/kernel behavior at load time

Possible concept:

```cpp
struct LinearKernelOps {
    GemvFn gemv;
    GemmFn gemm;
    EmbeddingFn embedding;
};

struct CpuLinearSegment {
    const void* data;
    LinearKernelOps ops;
    LinearMetadata metadata;
};
```

Exact implementation is open.

The rule is to avoid repeatedly branching over every supported weight representation in hot GEMV/GEMM paths.

## 11.2 Codec and model semantics remain independent

A descriptor may require a logical matrix.

The checkpoint/binding/codec layers decide how that matrix is physically represented.

No model descriptor should say "execute Nanbeige INT8 math" when the semantics are simply a linear transform whose storage happens to be packed INT8.

## 11.3 Acceptance gate

- a new supported weight encoding does not require model-family code;
- hot-path weight representation dispatch is load-time resolved where practical;
- CPU linear engine does not grow an unbounded central branch tree.

---

# Phase 12 - Collapse the C API to one current design

Priority: high  
Risk: deliberate breaking change  
Purpose: enforce the project's no-backward-compatibility directive and preserve backend OCP.

The current public C API must not retain old enum/union backend selection next to a newer generic backend-ID path.

## 12.1 Keep only backend-ID based creation

Target concept:

```c
typedef struct celeg_engine_options {
    uint32_t struct_size;
    const char* backend_id;
    int32_t max_context;
    const void* backend_options;
    uint32_t backend_options_size;
    celeg_generation_options generation;
} celeg_engine_options;

CELEG_API celeg_engine* celeg_engine_create(
    const char* path,
    const celeg_engine_options* options);
```

## 12.2 Remove compatibility duplicates

Delete rather than deprecate:

- `celeg_engine_create_v2`;
- `celeg_engine_v2_options`;
- `celeg_cpu_backend_v2_options` naming;
- `celeg_cuda_backend_v2_options` naming;
- legacy CPU/CUDA enum/union engine creation path if superseded;
- duplicate `create_service_bundle` paths created only for compatibility.

Rename the chosen current backend option types without version suffixes.

## 12.3 Acceptance gate

- one engine creation model exists;
- adding a backend does not require extending a public CPU/CUDA enum union;
- no CELEG-owned `_v2` engine API remains.

---

# Phase 13 - Delete family-centric runtime composition

Priority: very high after descriptor migration  
Risk: low once all families migrate  
Purpose: remove the final structural reason for `models/`.

## 13.1 Remove built-in family tables

A central list such as:

```text
lfm2
granite
gemma4
qwen35
nanbeige
...
```

must not be required to compose the runtime.

Descriptors/resources should be discovered or generated automatically.

## 13.2 Remove family registration functions

Delete functions whose only purpose is:

```text
register_qwen35_architecture
register_granite_chat_profile
register_gemma4_vision_provider
```

after equivalent generic data/provider mechanisms exist.

## 13.3 Remove `IArchitecture`

Once no supported model depends on a family-specific resolver, remove:

- `IArchitecture` if it serves no remaining generic purpose;
- `ArchitectureCatalog`;
- architecture factory registration;
- migration adapters.

Do not keep them as compatibility shims.

## 13.4 Delete `src/models/**` and `include/celeg/models/**`

This is a planned milestone, not an aspirational comment.

The repository should eventually have no production model-family source tree.

## 13.5 Acceptance gate

```text
src/models/          does not exist
include/celeg/models/ does not exist
```

and all supported model fixtures still load and execute.

---

# Phase 14 - Optional CELEG self-describing model artifact

Priority: strategic, after generic import is mature  
Risk: separate design project  
Purpose: eliminate repeated interpretation of external family conventions and make the runtime maximally model-agnostic.

External Hugging Face/GGUF/etc. checkpoints may remain imperfectly self-describing.

Even after C++ family modules disappear, CELEG may still ship model descriptors that know external family conventions.

A CELEG-native artifact can eliminate that at execution time.

## 14.1 Concept

```text
model.celeg/
    manifest
    model_ir
    weight_bindings
    tokenizer
    chat_profile
    weights or external weight references
```

The artifact records already-resolved model semantics.

## 14.2 Import once, execute generically

```text
External checkpoint
      |
      v
CELEG importer
      |
      v
Self-describing CELEG artifact
      |
      v
Generic CELEG runtime
```

At this point the runtime does not even need the original external family descriptor.

## 14.3 Not required to delete `models/`

Do not block the main refactor waiting for a new file format.

The generic descriptor/import pipeline is sufficient to remove family C++ first.

---

# 15. Descriptor and importer safety rules

A data-driven architecture can become worse than C++ if the descriptor language grows without boundaries.

Enforce these rules.

## 15.1 No Turing-complete scripting

No embedded Python, JavaScript, Lua, shell, or arbitrary callback expressions for normal model descriptors.

## 15.2 Deterministic evaluation

Given:

- descriptor;
- metadata;
- tensor inventory;

resolution must be deterministic.

## 15.3 Resource limits

Descriptor evaluation must bound:

- recursion;
- generated node count;
- generated weight request count;
- expression depth;
- layer/expert comprehensions.

## 15.4 Validation before allocation-heavy execution

Reject invalid model descriptions before backend compilation where possible.

## 15.5 Clear provenance

Record:

- matched descriptor identity;
- external format;
- descriptor revision/hash;
- generated IR fingerprint;
- relevant tokenizer/chat profile identity.

This supports debugging without making provenance part of execution semantics.

---

# 16. Testing strategy

The model-agnostic direction requires stronger tests than architecture-specific happy paths.

## 16.1 Descriptor parser tests

Cover:

- required fields;
- optional defaults;
- type mismatches;
- array indexing;
- conditionals;
- enum mapping;
- arithmetic;
- assertions;
- invalid references;
- bounded comprehensions.

## 16.2 IR validation tests

Cover every primitive schema and cross-node invariant.

## 16.3 Binding tests

Cover:

- candidate selection;
- aliases/tied weights;
- repeated physical layers;
- expert placeholders;
- transforms;
- quantized representations;
- shape mismatches;
- missing required tensors.

## 16.4 Old-vs-new migration equivalence

While a family has both paths temporarily, compare old C++ resolver output with generic descriptor output.

Delete the old path after equivalence is demonstrated.

Do not retain dual paths indefinitely.

## 16.5 Execution equivalence

For migrated fixtures compare:

- logits;
- token decode;
- chunk prefill;
- stateful/recurrent transitions;
- MoE routing where deterministic;
- CPU/CUDA agreement within established tolerances.

## 16.6 Extension tests

CI should include synthetic fixtures proving independently that:

- new model descriptor: no production C++ change;
- new backend: no model/importer change;
- new weight codec: no model semantic change;
- new primitive: no family execution fork.

---

# 17. SOLID interpretation in the target architecture

## 17.1 Single Responsibility Principle

Target responsibility boundaries:

- repository reads physical data;
- descriptor interprets external metadata conventions;
- importer builds IR;
- IR owns mathematical semantics;
- binding maps logical weights to physical tensors;
- primitive validates one mathematical operation;
- backend compiler lowers semantics;
- kernel executes backend-specific math;
- workspace planner owns scratch sizing;
- tokenizer tokenizes;
- chat profile formats conversations.

A model-family class doing several of those simultaneously should disappear.

## 17.2 Open/Closed Principle

The strongest target:

```text
new model using known primitives -> data only
new tensor naming -> data only
new backend -> backend module only
new checkpoint format -> repository/import adapter only
new quantization -> codec/kernel representation only
new mathematics -> new primitive + lowerings
```

This is materially stronger than "add one new model-family subclass".

## 17.3 Liskov Substitution Principle

Keep capability interfaces honest.

Do not create base interfaces with default "unsupported" methods merely to accommodate unrelated repositories/providers/backends.

## 17.4 Interface Segregation Principle

Preserve the existing good direction of narrow repository capabilities.

Apply the same principle to:

- importers;
- descriptor sources;
- backend compilers;
- preprocessing providers;
- weight readers/codecs.

## 17.5 Dependency Inversion Principle

High-level model semantics depend on primitive contracts, not CPU/CUDA kernels.

Backends depend on canonical semantic programs, not model-family classes.

Importers depend on checkpoint metadata/tensor interfaces, not concrete file readers.

Operators depend on focused execution contexts, not whole compiled-model god objects.

---

# 18. Revised SOLID assessment

At the current baseline, the previous numerical score remains broadly useful, but the model-agnostic north star exposes an additional OCP limitation that the older assessment treated as acceptable modularity.

Approximate current state:

| Principle | Score | Reason |
| --- | ---: | --- |
| Single Responsibility | 7.4/10 | Good subsystem separation, but family resolvers still combine import semantics, graph construction, weight planning and provenance; CPU orchestration/workspace remain broad. |
| Open/Closed | 6.9/10 | Backend/provider catalogs are strong, but a new model family still commonly means new architecture/naming C++ plus central composition changes. |
| Liskov Substitution | 8.8/10 | Capability-oriented interfaces are already strong. |
| Interface Segregation | 9.2/10 | Repository capability split is an architectural asset. |
| Dependency Inversion | 8.0/10 | Generic runtime is abstraction-oriented, but family C++ remains above the desired IR/import boundary and CPU operators still depend too broadly on concrete state. |

Target after the complete plan: approximately **9.4-9.7/10** while preserving low-level performance.

The target is not academic interface count. It is a codebase in which unrelated kinds of extension stop forcing edits in the same places.

---

# 19. Execution order

Recommended order to minimize rework:

```text
Phase 0   architectural guards
   |
Phase 1   canonical semantic source
   |
Phase 2   CELEG Model IR vocabulary
   |
Phase 3   descriptor evaluator
   |
Phase 4   generic weight binding
   |
Phase 5   generic importer
   |
Phase 6   migrate model families incrementally
   |\
   | +---- Phase 7 tokenizer/chat/vision decoupling
   |
   +------ Phase 8 CPU compiled-layer execution
   |       Phase 9 narrow CPU contexts
   |       Phase 10 workspace planning
   |
   +------ Phase 11 extensible weight representations
   |
Phase 12  collapse C API / remove v2 compatibility debt
   |
Phase 13  remove family-centric composition and delete models/
   |
Phase 14  optional self-describing CELEG artifact
```

CPU execution work can proceed in parallel after canonical semantics stabilize because it consumes the same target IR/program boundary.

---

# 20. Concrete migration definition of done

The architecture refactor is not complete merely because the code looks cleaner.

It is complete when the following statements are true.

## 20.1 Model extension

A synthetic new model family composed entirely from existing primitives is supported by adding descriptor/resources/tests only.

No production C++ source modification is required.

## 20.2 No production model-family tree

```text
src/models/
include/celeg/models/
```

are gone.

## 20.3 No family resolver hierarchy

There is no required one-class-per-family `IArchitecture` mechanism.

## 20.4 No family execution identity

CPU/CUDA/runtime/model-program code does not branch on family/model identity.

## 20.5 Canonical semantics

One canonical IR owns model mathematics.

Derived backend/runtime structures do not duplicate semantic ownership.

## 20.6 Data-driven binding

Tensor names/layout conventions are import/binding data, not per-family C++ switches.

## 20.7 Primitive-based growth

When genuinely new mathematics arrives, CELEG grows by adding a reusable primitive and backend lowering(s), not by forking a model execution path.

## 20.8 Backend extensibility

A new backend registers through backend abstractions without changes to model semantics or a CPU/CUDA public enum switch.

## 20.9 Quantization extensibility

A new storage/weight representation extends codec/kernel representation boundaries without changing model-family semantics.

## 20.10 No compatibility clutter

There is one current CELEG API design, with no retained `_v2`/legacy path solely for compatibility.

## 20.11 Performance

Descriptor/IR flexibility is resolved before the hot path.

Decode/prefill remain compiled backend execution rather than descriptor interpretation.

---

# 21. Final architectural rule set

The following rules should eventually be enforceable in CI.

```text
RULE 1
Model-family identity may exist in external descriptors and provenance,
but not in generic execution semantics.

RULE 2
A model that uses existing CELEG primitives must not require new model-family C++.

RULE 3
New mathematics extends the primitive language, not a family execution fork.

RULE 4
Model IR owns mathematics; weight binding owns checkpoint tensor mapping.

RULE 5
Checkpoint format and quantization/storage format do not define model semantics.

RULE 6
Descriptors are deterministic, declarative, constrained and evaluated before hot execution.

RULE 7
Backends compile semantic programs and never probe model families.

RULE 8
Token and chunk execution derive from one compiled semantic source.

RULE 9
Focused operators depend on focused execution views, not global god objects.

RULE 10
CELEG has no backward-compatibility debt unless an explicit stability contract creates one.

RULE 11
Central registries must not require manual edits for every new descriptor/model family.

RULE 12
The planned end state contains no production src/models/ or include/celeg/models/ tree.
```

---

# 22. End-state mental model

The desired CELEG architecture is analogous to a compiler.

```text
External model conventions
        |
        v
Importer / descriptor front-end
        |
        v
      CELEG IR
        |
        v
Backend lowering/compiler
      /       \
     v         v
   CPU        CUDA
```

The analogy is intentional:

```text
Qwen --------\
Granite ------\
LFM -----------\
Gemma ----------> CELEG IR ---> CPU
Nanbeige ------/             ---> CUDA
Nemotron -----/
FutureModel --/
```

Just as an LLVM backend should not care whether the source program was originally C, C++, Rust, Swift or Fortran, a CELEG backend should not care which model-family convention produced the semantic IR.

The ultimate product definition should become:

> **CELEG executes any model graph composed of primitives it supports, regardless of the model-family name used by the source checkpoint.**

That is the architectural direction this plan now treats as the SOLID end state.
