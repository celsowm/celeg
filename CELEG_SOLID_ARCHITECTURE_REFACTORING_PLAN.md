# CELEG SOLID Model-Agnostic Architecture Refactoring Plan

Status: active architecture plan  
Date: 2026-08-07  
Scope: SOLID architecture, model-agnostic IR, importers, data-driven descriptors, weight binding, backend compilation, C API cleanup, and removal of model-family runtime code

This is the architectural north-star document for CELEG.

The detailed implementation roadmap for attention mechanisms, positional encodings, context-extension techniques, KV/cache policies, sparse/linear attention, cross-attention, state representations, and CPU/CUDA lowerings lives in:

> **`CELEG_ATTENTION_CONTEXT_AND_RUNTIME_PRIMITIVES_PLAN.md`**

The two documents have deliberately different responsibilities:

- this document defines **architectural ownership, SOLID boundaries, migration strategy, and the end state**;
- `CELEG_ATTENTION_CONTEXT_AND_RUNTIME_PRIMITIVES_PLAN.md` defines **the primitive vocabulary and implementation sequence required to support modern 2025-2026 model mathematics without adding model-family execution code**.

This document supersedes `CELEG_SOLID_EXECUTION_AND_TOKENIZER_REFACTORING_PLAN.md` as the primary architecture plan.

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

Target data flow:

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

Model-family identity may exist while interpreting external metadata or recording provenance, but it must disappear before generic execution and backend compilation decisions.

---

# 2. Strongest Open/Closed criterion

## 2.1 New model, existing mathematics

If a newly released model can be expressed entirely using CELEG primitives that already exist, support should require only:

- descriptor/import data;
- tensor binding rules;
- tokenizer/chat/vision metadata where required;
- fixtures and tests.

It should require **zero new model-family C++ production files**.

No new:

```text
src/models/foo/architecture.cpp
src/models/foo/naming_policy.cpp
include/celeg/models/foo/architecture.hpp
```

should be necessary.

## 2.2 New mathematics

If a new model introduces genuinely new mathematics, CELEG may need a new generic primitive.

The extension should be:

```text
new mathematics
    |
    v
new generic semantic primitive
    |
    +--> validation
    +--> CPU lowering
    +--> CUDA lowering
    +--> tests
```

not:

```text
new model
    |
    v
new family execution fork
```

The detailed catalog and implementation order for these primitives is maintained in `CELEG_ATTENTION_CONTEXT_AND_RUNTIME_PRIMITIVES_PLAN.md`.

## 2.3 New checkpoint storage representation

GPTQ, AWQ, packed INT8, FP8, MXFP, GGUF native blocks, and future weight encodings belong to:

- repository capabilities;
- tensor codecs;
- loaded weight representations;
- backend kernel selection.

They do not create model-family semantics.

## 2.4 New checkpoint format

A new container/metadata format belongs to importer/repository boundaries.

It must not change model execution semantics.

## 2.5 New backend

A new backend belongs behind backend factory/compiler/lowering boundaries.

It must not change descriptors or canonical Model IR.

---

# 3. Non-negotiable directives

## 3.1 No backward-compatibility commitment

CELEG currently has no source, ABI, API, configuration, file-layout, or internal backward-compatibility commitment unless a future explicit stability milestone says otherwise.

Refactors replace obsolete interfaces. They do not stack versions or retain migration shims.

Do not preserve CELEG-owned patterns such as:

- `foo_v2`, `foo_v3`;
- `legacyFoo`;
- `deprecatedFoo`;
- `compatFoo`;
- duplicate old/new entry points;
- transitional aliases;
- compatibility typedefs;
- wrappers that exist only to preserve obsolete signatures.

If an interface is replaced, migrate repository callers and remove the superseded interface in the same refactoring series.

Versioning remains valid where intrinsic to an external protocol, file format, third-party ABI, or an explicitly declared future CELEG stability contract.

## 3.2 Performance is an architectural constraint

Descriptors and generic IR are load/import-time abstractions.

They must not become hot-path interpreters.

Prefer:

- load-time compilation;
- pre-resolved function pointers;
- backend-local typed execution records;
- immutable compiled programs;
- contiguous reusable scratch storage;
- zero-allocation steady-state execution;
- precomputed weight bindings;
- precomputed workspace plans;
- static or load-time kernel dispatch.

Avoid in decode/prefill hot loops:

- JSON parsing;
- metadata lookup;
- descriptor evaluation;
- architecture probing;
- checkpoint-format probing;
- string-based operator dispatch;
- repeated `dynamic_cast`;
- family checks;
- per-layer heap allocation.

## 3.3 Architecture identity disappears before execution

Backends consume mathematical semantics.

No CPU/CUDA execution decision may depend on:

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
Mistral
DeepSeek
Kimi
```

or any future model-family identifier.

## 3.4 Model semantics and checkpoint binding are different responsibilities

The graph says what must be computed.

The binding plan says where parameters come from.

Do not encode tensor file names into semantic primitives.

Do not encode architecture semantics into tensor storage codecs.

## 3.5 Descriptors are declarative and constrained

The descriptor system must support only the operations required to interpret metadata and construct canonical IR/bindings.

It must not become an unrestricted embedded programming language.

No arbitrary:

- filesystem I/O;
- network access;
- process execution;
- runtime C++ callbacks per family;
- unrestricted scripts.

---

# 4. What remains valid from the previous SOLID review

Keep these goals:

- make `ModelGraph` / successor Model IR the canonical semantic source;
- eliminate duplicated semantic state from `RuntimeTopology`;
- compile token/chunk execution rather than implement semantics twice;
- narrow CPU operator dependencies;
- replace oversized workspace ownership with workspace planning/views;
- make quantized weight dispatch extensible;
- eliminate hard-coded family-name boundary checks;
- remove `v2`/legacy compatibility debt;
- preserve capability-oriented repository interfaces;
- preserve backend-neutral semantics;
- preserve load-time resolution for performance.

What changes is the destination.

`IArchitecture`, `ArchitectureCatalog`, and one-runtime-module-per-family are useful migration mechanisms, not the desired permanent extension boundary.

---

# 5. Explicit transitional abstractions

## 5.1 `IArchitecture`

Useful while family-specific C++ resolvers exist.

End-state expectation:

- descriptor/importer infrastructure replaces most family resolvers;
- `IArchitecture` shrinks to a genuinely generic import extension point or disappears.

## 5.2 `ArchitectureCatalog`

Useful during migration.

It should not be required forever just to enumerate model family names.

## 5.3 one module per family

This is better than centralized switches but remains family-centric.

Treat it only as an intermediate organization step.

## 5.4 family extension test

The weaker test is:

> a family can be added by adding one module.

The stronger final test is:

> a model using existing primitives can be added without adding or modifying production C++.

The stronger test is the target.

---

# 6. Canonical Model IR

The current overlapping semantic representations across `RuntimeTopology`, `ModelGraph`, and `CompiledModelProgram` must be reduced to one semantic authority.

Target:

```text
Checkpoint metadata
      |
      v
Importer / descriptor
      |
      v
Canonical Model IR
      |
      +--> derived shape/allocation information
      |
      +--> logical weight requirements
      |
      +--> CompiledModelProgram
```

## 6.1 Stop building graph semantics from topology arrays

Architecture/import code should construct semantic layer specifications directly.

Do not populate parallel arrays such as mixer kinds, attention layouts, FFN kinds, and recurrent layouts and later rebuild the graph from them.

## 6.2 Runtime topology becomes derived shape only

A derived runtime shape may contain:

- dimensions;
- maxima needed for allocation;
- counts;
- lookup tables derived from the graph;
- global token/numerical policies where truly global.

It must not independently describe layer mathematics.

## 6.3 Boundary semantics belong in the IR

Intermediate normalization boundaries, residual policies, per-layer inputs, mixer-only layers, routing policies, and other mathematical behavior must have one canonical semantic owner.

## 6.4 Checkpoint layer reuse belongs to binding

Physical checkpoint-layer mapping is parameter binding, not execution semantics.

Logical layers should execute resolved parameters without caring whether two logical layers came from the same physical checkpoint block.

---

# 7. Primitive vocabulary is a prerequisite for deleting `models/`

The model-agnostic goal only works if the IR can express modern model mathematics generically.

Therefore the following implementation work is a direct dependency of this architecture plan:

> See `CELEG_ATTENTION_CONTEXT_AND_RUNTIME_PRIMITIVES_PLAN.md`.

That companion roadmap owns the detailed design and implementation sequence for:

- decomposed attention semantics;
- MHA/GQA/MQA projection semantics;
- causal/sliding/bidirectional/prefix/block-sparse/dynamic sparse patterns;
- first-class RoPE policies;
- linear position interpolation;
- dynamic/NTK-aware RoPE;
- YaRN;
- LongRoPE;
- Llama-3-style frequency-aware scaling;
- partial rotary;
- M-RoPE/multi-axis positions;
- Multi-head Latent Attention;
- cross-attention and multi-source attention;
- Native Sparse Attention-style components;
- Kimi Delta Attention / modern linear attention;
- ALiBi and relative position biases;
- generalized attention/recurrent state descriptions;
- KV/state quantization;
- attention sinks / streaming cache policy;
- context-parallel/distributed attention lowerings;
- backend flash/fused/paged kernel selection.

Architectural rule:

> These are primitives/policies/lowerings. None of them may be introduced as a model-family execution mode.

---

# 8. Generic model descriptor

Most current family resolver code performs declarative import work:

1. recognize metadata;
2. read dimensions and policies;
3. choose per-layer primitive kinds;
4. build semantic layer specifications;
5. declare logical tensor roles/shapes;
6. map logical roles to checkpoint tensor names;
7. set provenance/tokenizer/chat/vision data.

These should progressively move from family C++ into a declarative descriptor system.

## 8.1 Required descriptor features

Support at least:

- metadata lookup;
- typed defaults;
- validation;
- list lookup;
- per-layer selection;
- bounded conditionals;
- simple arithmetic for dimensions;
- enum/variant construction;
- repeated layer templates;
- layer schedules;
- tensor binding patterns;
- capability/provenance declarations.

## 8.2 Descriptor output

Descriptor evaluation should produce only validated CELEG-owned structures:

```text
Model IR
Logical Weight Plan
Binding Descriptor
Tokenizer/Chat/Vision metadata references
Provenance
```

It should not produce arbitrary execution callbacks.

---

# 9. Weight binding becomes data

Current family naming policies often reduce to:

```text
TensorRole + layer + expert -> candidate checkpoint tensor path(s)
```

Move this into declarative binding rules.

Example concept:

```text
attention.query:
    model.layers.{layer}.self_attn.q_proj.weight

ffn.gate:
    model.layers.{layer}.mlp.gate_proj.weight
```

More advanced binding rules may support:

- multiple candidate names;
- fused tensor sources;
- split/concat transforms;
- expert indexing;
- transposition/layout transforms;
- physical-layer remapping;
- tied embeddings.

Transforms must remain declarative and validated.

---

# 10. Backend compilation

Backends should compile canonical semantics once.

Target CPU shape:

```cpp
struct CpuCompiledLayer {
    CpuTokenLayerFn token;
    CpuChunkLayerFn chunk;
    CpuLayerWeights weights;
    CpuLayerExecutionSpec execution;
};
```

The exact types may differ, but the principle is:

- semantic dispatch is resolved at load time;
- token/chunk paths are execution strategies for one semantic layer;
- model-family identity is absent.

## 10.1 `model_forward.cpp` becomes orchestration

It should prepare input, invoke compiled layers, finalize logits, and update session state.

It should not independently reconstruct the model's semantic decision tree.

## 10.2 Chunk support is a compiled capability

A primitive/lowering declares one of:

```text
native chunk implementation
generic sequential adapter
unsupported by backend
```

Do not infer chunk behavior from architecture or family identity.

---

# 11. Narrow CPU operator dependencies

Focused operators should not receive the whole `CpuCompiledModel&` by default.

Introduce focused views/contexts such as:

```cpp
struct CpuExecutionContext {
    CpuLinearEngine& linear;
    CpuThreadPool& pool;
    CpuSessionView session;
    CpuWorkspaceView workspace;
};
```

Use even narrower operator-specific views where beneficial.

Benefits:

- improved SRP;
- clearer dependencies;
- simpler unit tests;
- better primitive reuse;
- lower risk of hidden cross-operator coupling.

---

# 12. Workspace planning

`CpuWorkspace` should become storage governed by a compile/load-time workspace plan.

Target idea:

```text
Model IR
   |
   v
CPU compiler
   |
   +--> CpuProgram
   +--> CpuWorkspacePlan
```

The plan computes required scratch sizes from compiled primitives.

The workspace owns reusable buffers but does not need to understand every model/operator semantic.

---

# 13. Quantized weights and physical representations

New weight formats must not grow central `if/variant/switch` chains indefinitely.

Prefer load-time resolution to compact operations tables or equivalent backend-local compiled representations.

Target concept:

```cpp
struct LinearKernelOps {
    GemvFn gemv;
    GemmFn gemm;
    EmbeddingFn embedding;
};
```

A loaded segment resolves the correct ops once.

Adding GPTQ/AWQ/FP8/MXFP/other formats should not require model-family edits.

---

# 14. C API cleanup

Because CELEG has no current backward-compatibility commitment, the extensible API replaces the old API instead of becoming `v2`.

Required direction:

- `celeg_engine_v2_options` -> `celeg_engine_options`;
- `celeg_cpu_backend_v2_options` -> `celeg_cpu_backend_options`;
- `celeg_cuda_backend_v2_options` -> `celeg_cuda_backend_options`;
- `celeg_engine_create_v2()` -> `celeg_engine_create()`;
- remove the obsolete enum/union backend-selection path;
- retain `backend_id + backend-owned options payload`;
- migrate repository callers and delete old symbols in the same series.

No deprecated wrappers or compatibility aliases.

---

# 15. Architectural enforcement

Static checks should be structural, not a hard-coded list of family names.

Enforce rules such as:

```text
src/backend/**              must not include celeg/models/**
src/runtime/**              must not include celeg/models/**
src/checkpoint/**           must not include celeg/models/**
include/celeg/model/**      must not include celeg/models/**
include/celeg/runtime/**    must not include celeg/models/**
include/celeg/checkpoint/** must not include celeg/models/**
```

Also reject backend execution dispatch using:

- `architecture_id`;
- `architecture_kind`;
- `model_type`;
- family names.

The checker must scale to the 100th model family without editing a family-name regex.

## 15.1 Compatibility debt checker

Flag suspicious CELEG-owned public/runtime symbols containing patterns such as:

```text
_v2
_v3
legacy
deprecated
compat
```

with allowlists only for legitimate external format/protocol versions.

---

# 16. Migration roadmap

## Phase 0 - Lock invariants

- structural architecture-boundary checks;
- no-compatibility-debt checks;
- backend-extension regression tests;
- descriptor extension test proving production C++ is unnecessary for an existing primitive set.

## Phase 1 - Canonical Model IR

- move semantic ownership into the graph/IR;
- reduce/remove duplicated `RuntimeTopology` semantics;
- derive runtime shape from the IR;
- compile `CompiledModelProgram` from canonical semantics only.

## Phase 2 - Primitive vocabulary

Execute the P0/P1 roadmap in `CELEG_ATTENTION_CONTEXT_AND_RUNTIME_PRIMITIVES_PLAN.md` sufficiently to cover the intended proof models without family hacks.

At minimum before broad family migration:

- first-class position specification;
- explicit RoPE scaling algorithms;
- M-RoPE in canonical IR;
- generalized attention patterns;
- state-layout abstraction capable of ordinary KV and future latent state.

## Phase 3 - Generic descriptor/importer

- descriptor schema/evaluator;
- metadata expressions;
- per-layer schedules;
- IR construction;
- logical weight-plan construction;
- validation.

## Phase 4 - Declarative weight binding

- migrate role/name switches into binding data;
- support candidates, layer/expert indices, transforms, and remapping;
- remove family naming policies as each family migrates.

## Phase 5 - Dense model proof

Recommended proofs:

1. Qwen3-0.6B: small bootstrap model;
2. Ministral-8B-Instruct-2410: per-layer full/sliding attention schedule.

Acceptance:

```text
0 new family .cpp
0 new family .hpp
0 if(model_type == ...)
0 switch(architecture)
```

Production changes are allowed only for genuinely missing generic primitives/importer capabilities.

## Phase 6 - Context-extension proof

Use models requiring YaRN/LongRoPE/related policies to prove that new positional mathematics is added once as a primitive and then selected by data.

See the companion primitives roadmap for sequence and acceptance gates.

## Phase 7 - Advanced-attention proof

Use MLA, sparse-attention, modern linear-attention, and cross-attention models as proofs that genuinely new mathematics extends the primitive set rather than recreating family runtime directories.

## Phase 8 - CPU compiled execution

- compile per-layer token/chunk executors;
- narrow operator contexts;
- introduce workspace plans;
- preserve token/chunk equivalence tests.

## Phase 9 - Weight representation OCP

- resolve physical weight ops at load time;
- avoid format switch growth in hot/shared paths;
- preserve backend-specific optimized kernels.

## Phase 10 - Remove family runtime composition

As descriptors replace family modules, move tokenizer/chat/vision registration toward data/provider capabilities instead of per-family runtime modules.

## Phase 11 - Delete `models/`

Only after coverage proves it is safe:

- delete migrated `src/models/**`;
- delete migrated `include/celeg/models/**`;
- remove architecture catalogs/resolvers that have no remaining generic role;
- add CI rule preventing reintroduction of model-family production directories for models expressible by existing primitives.

---

# 17. Proof ladder

Do not try to prove everything with one hostile model.

Use increasingly demanding integration proofs.

## Proof 1 - Qwen3-0.6B

Goal:

- generic importer;
- descriptor-driven dense transformer;
- generic binding;
- no family C++.

## Proof 2 - Ministral-8B-Instruct-2410

Goal:

- another external family;
- per-layer full/sliding attention schedule;
- prove importer is not accidentally a Qwen importer.

## Proof 3 - YaRN / Ministral-3-class model

Goal:

- explicit context-extension semantics;
- generic position policy;
- no family-specific RoPE branch.

## Proof 4 - Phi-4-Mini-class model

Goal:

- LongRoPE;
- partial rotary;
- position policies composed generically.

## Proof 5 - DeepSeek-V2/V3-class model

Goal:

- MLA as a reusable primitive;
- latent state layout;
- no DeepSeek runtime fork.

## Proof 6 - Kimi-Linear-class model

Goal:

- modern linear attention extension;
- token/chunk/recurrent-state contracts;
- no Kimi runtime fork.

## Proof 7 - multimodal/cross-attention model

Goal:

- multi-source graph values;
- encoder/vision memory;
- cross-attention as semantics, not family code.

---

# 18. SOLID interpretation of the target architecture

## Single Responsibility

- checkpoint reader parses storage/container;
- importer interprets external metadata;
- descriptor declares import rules;
- Model IR owns mathematics;
- binding owns parameter location/transforms;
- backend compiler owns lowering;
- runtime owns scheduling/state;
- kernel owns numerical implementation.

## Open/Closed

Existing mathematics -> descriptor only.

New mathematics -> new generic primitive/lowering only.

New storage -> repository/codec/kernel only.

New backend -> backend compiler/factory only.

## Liskov Substitution

Capability interfaces should remain behaviorally substitutable and avoid fake unsupported operations.

## Interface Segregation

Preserve narrow repository capabilities and introduce similarly narrow primitive/backend capabilities where needed.

## Dependency Inversion

Generic import/model/runtime layers depend on CELEG semantic abstractions, not CPU/CUDA implementations or model-family classes.

---

# 19. Definition of done

The architecture plan is complete when:

- canonical Model IR is the sole semantic authority;
- backend execution contains no model-family identity;
- common new models can be added with descriptors/bindings/tests only;
- missing mathematics extends generic primitives rather than family code;
- weight storage/quantization is orthogonal to model semantics;
- token/chunk execution is compiled from one semantic source;
- CPU operators depend on focused execution views;
- workspace is plan-driven;
- the C API has no unnecessary version stacking;
- architecture checks are structural;
- `src/models/**` and `include/celeg/models/**` are removed when no longer necessary;
- CI prevents their reintroduction for models expressible with existing primitives.

The companion `CELEG_ATTENTION_CONTEXT_AND_RUNTIME_PRIMITIVES_PLAN.md` is considered a direct implementation dependency of this definition of done for modern model coverage.

---

# 20. Final invariant

The intended end state is:

> **CELEG does not support model families. CELEG supports a vocabulary of mathematical primitives, positional policies, state representations, import rules, weight bindings, and backend lowerings. External model descriptions are compiled into that vocabulary.**

Once this invariant is true, adding the next ordinary model should feel like adding data and tests, not adding another runtime subsystem.