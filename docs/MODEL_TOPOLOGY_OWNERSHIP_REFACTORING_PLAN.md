# CELEG Model Topology Ownership Refactoring Plan

## Status

Proposed architectural refactor.

Target repository: `celsowm/celeg`.

Scope: `include/celeg/model/**` and `src/model/**`.

This plan has **no backward-compatibility constraint**. Struct layouts, function
signatures, and public headers may change as clean breaks. No deprecated
wrappers, no transitional shims, no re-exported legacy names.

## Primary goal

> `ModelGraph` is the single semantic owner of execution topology, and the type
> system must make any other owner impossible to express.

The refactor is complete only when a resolution front-end is physically unable
to write a graph-derived field, and every `MixerKind` dispatch is exhaustively
checked by the compiler in exactly one place per concern.

---

# 1. Problem statement

CELEG already has the correct pipeline, and it is already unified. Every
resolution path funnels through `resolve_architecture_stages`
(`src/model/architecture.cpp:9-25`):

```text
stages.topology(checkpoint)          <-- stage 1: front-end builds RuntimeTopology
      |
      v
  topology.validate()
      |
      v
stages.graph(model, checkpoint)      <-- stage 2: front-end builds ModelGraph
      |
      v
derive_runtime_topology_from_graph() <-- stage 3: OVERWRITES most of stage 1
      |
      v
  topology.validate()
      |
      v
stages.weights(model, checkpoint)    <-- stage 4
```

Three front-ends feed this pipeline:

| Front-end | Source | Entry point |
| --- | --- | --- |
| Hand-written architectures | C++ | `resolve_architecture_stages` directly |
| Descriptor-driven | JSON descriptor | `src/model/descriptor/architecture.cpp:456` |
| Descriptorless inference | Tensor names + metadata | `src/model/inference/synthesis.cpp:28-43` |

The design intent is already documented at
`include/celeg/model/resolved.hpp:217-221`:

> *"Importers may use an initial shape while constructing the graph, but
> execution-facing schedules have one semantic owner: ModelGraph."*

**The intent is correct. The type system does not enforce it.**

## 1.1 Root cause: `RuntimeTopology` has two owners and no marker

`RuntimeTopology` (`include/celeg/model/resolved.hpp:40-195`) is a ~60-field
struct whose fields fall into three disjoint categories that are
**indistinguishable at the type level**:

**Category A — graph-owned.** Unconditionally overwritten by
`derive_runtime_topology_from_graph` (`src/model/resolved.cpp:18-127`). Anything
stage 1 writes here is dead computation:

```text
num_hidden_layers            mixer_kinds                  feed_forward_kinds
execute_feed_forward         feed_forward_intermediates   feed_forward_activations
attention_layouts            gated_delta_net_layouts      mamba2_layouts
mlp_only_layouts             attention_slot_for_layer     layer_for_attention_slot
attention_layer_count        conv_layer_count             gated_delta_net_layer_count
mamba2_layer_count           mlp_only_layer_count         num_dense_layers
mamba2_intermediate          max_feed_forward_intermediate dense_intermediate
moe_intermediate             shared_expert_intermediate   num_experts
experts_per_token            normalize_topk               moe_router_softmax
use_expert_bias              routed_scaling_factor        has_per_layer_input
per_layer_input_size         conv_cache                   conv_dim
shared_kv_group_count        has_split_attention_norms
```

The last two are *not* currently derived — see section 1.4. They belong here and
Phase 1 must move them.

**Category B — checkpoint-owned.** Survives stage 3 untouched; stage 1 is the
sole authority:

```text
hidden                       intermediate                 vocab_size
max_position_embeddings      checkpoint_layer_for_layer   token_policy
numerical_policy             mtp_num_hidden_layers
```

**Category C — reset but never derived (defective).** Zeroed at
`src/model/resolved.cpp:47-50`, never repopulated by the MoE derive branch
(`src/model/resolved.cpp:99-115`):

```text
moe_routing_group_count      moe_routing_experts_per_group
moe_routing_groups_per_token moe_routing_group_score_top_k
```

Because the categories are invisible, roughly 615 lines in
`src/model/inference.cpp` and 430 lines in
`src/model/descriptor/architecture.cpp` are a mixture of authoritative work and
discarded work, and no reader can tell which is which.

## 1.2 This has already produced two live defects

### Defect 1 — `double_wide_shared_suffix` is a silent no-op

`src/model/descriptor/architecture.cpp:431-436` doubles
`feed_forward_intermediates` and `max_feed_forward_intermediate` during stage 1.

Both are Category A. They are overwritten at `src/model/resolved.cpp:22`,
`:88`, and `:124-127` from `graph.layers[i].feed_forward.intermediate_size`.
`src/model/descriptor/graph_builder.cpp` never applies the doubling.

The flag is parsed (`src/model/descriptor/parser.cpp:121`) and stored
(`src/model/descriptor/detail.hpp:148`), but has no effect. Any descriptor
declaring it silently receives un-doubled FFN widths.

### Defect 2 — grouped-MoE routing is zeroed in topology

`src/model/inference.cpp:185-203` computes and validates the `moe_routing_*`
fields, including divisibility and range checks.
`src/model/graph_builder.cpp:43-44` correctly carries them into
`MixtureOfExpertsSpec`.

Stage 3 then zeroes them (Category C) and never reads them back, even though
`MixtureOfExpertsSpec` carries them (`include/celeg/model/graph.hpp:322-325`).

After resolution, `topology` reports "no grouped routing" while `graph` reports
grouped routing. Codegen is currently correct only by luck:
`src/model/program.cpp:453-458` happens to read the graph. Any consumer reading
`topology.moe_routing_*` gets wrong answers.

## 1.3 The dual-ownership channel is also an agnosticism leak

`ZERO_MODEL_FAMILIES_REFACTORING_PLAN.md` states the governing constraint:

> *CELEG must support semantics, not model families.*

Stage-1 topology authorship is a channel through which family-specific
knowledge reaches backend control flow **without passing through the semantic
graph**. Two fields currently exploit it.

### `has_split_attention_norms`

Written only from a descriptor flag
(`src/model/descriptor/architecture.cpp:422` reads
`descriptor_.split_attention_norms`). Never derived from the graph. Read by at
least nine backend sites to branch execution:

```text
src/backend/cpu/model_forward_chunk.cpp:380, :402
src/backend/cpu/model_forward_token.cpp:100, :119, :129
src/backend/cpu/packed_execution.cpp:322, :485
src/backend/cpu/weights_loader.cpp:60, :66
src/backend/cuda/model/expert_setup.cpp:27
src/backend/cuda/model/weight_setup.cpp:84
```

Meanwhile `LayerSpec` already carries the real, *per-layer* semantic truth:
`post_attention_norm`, `pre_feed_forward_norm`, `post_feed_forward_norm`
(`include/celeg/model/graph.hpp:337-341`).

So backends branch on a coarse model-wide boolean authored by a descriptor,
while the graph holds a finer-grained answer they ignore. This is lossy — a
checkpoint with split norms on only some layers cannot be expressed — and the
descriptorless path never sets the flag at all, so automatic inference silently
resolves it to `false` regardless of what the tensors show.

### `shared_kv_group_count`

Computed heuristically at `src/model/descriptor/architecture.cpp:420-421` as
`shared_layers > 0 ? (has_attention_variants ? 2 : 1) : 0`, and hardcoded to `0`
at `src/model/inference.cpp:208`.

The graph already carries the authoritative per-layer answer in
`AttentionSpec::kv_sharing` (`KvSharingSpec { group, publishes }`,
`include/celeg/model/graph.hpp:62-67`). The correct value is the count of
distinct non-negative `group` values across attention layers — a derivation, not
a heuristic.

### Consequence for this plan

Both fields belong in `ExecutionTopology`, not `CheckpointDimensions`. Phase 1
must derive them from the graph and delete both write sites. This upgrades the
refactor from *neutral* on agnosticism to *actively closing two leak channels*:
once `ExecutionTopology` has no public constructor, a descriptor cannot inject a
family-shaped flag into backend control flow at all.

## 1.4 Secondary duplication (symptoms, not causes)

These follow from 1.1 and are resolved by the same refactor:

- **`MixerKind` dispatch appears four times**: `src/model/resolved.cpp:60-116`,
  `src/model/graph_builder.cpp:21-48`,
  `src/model/descriptor/graph_builder.cpp:76-115`, and
  `src/model/descriptor/weight_requirements.cpp`.
- **Weight-requirement derivation appears three times and has drifted**:
  `src/model/weight_plan.cpp:30-88` (no query/key norm, no output gate, no
  `kv_sharing`, no MoE), `src/model/descriptor/weight_requirements.cpp:8-77`
  (complete), and inline at `src/model/inference.cpp:544-606`.
- **Pipeline ordering is duplicated**: `src/model/inference/synthesis.cpp:28-43`
  re-implements the stage order of `src/model/architecture.cpp:9-25` instead of
  reusing it.
- **Key selection is forked**: `src/model/descriptor/probe_matcher.cpp:28-32`
  re-derives the `is_gguf()` branch of
  `src/model/descriptor/field_resolver.cpp:7-17` but drops `{architecture}`
  substitution.
- **Ambiguity detection is forked**: `src/model/inference.cpp:45-59`, `:410-421`,
  `:441-455` hand-roll the loop that `inference_detail::find_unique`
  (`src/model/inference/support.hpp`) already provides.
- **Validation errors use four types** for one failure category:
  `std::invalid_argument` (`definition.cpp`, `program.cpp`),
  `std::runtime_error` (`resolved.cpp`), `celeg::ResolutionError`
  (`inference*.cpp`). Only the last carries a machine-readable kind.
- **Quantization row scan is duplicated**:
  `src/model/weights/quantization.cpp:70-75` and `:148-153`.

---

# 2. Target design

## 2.1 Split the god-struct by owner

```cpp
// Authored by the front-end. The checkpoint is the only source.
struct CheckpointDimensions {
    int hidden = 0;
    int intermediate = 0;
    int vocab_size = 0;
    int max_position_embeddings = 0;
    std::vector<int> checkpoint_layer_for_layer;
    TokenPolicy token_policy;
    NumericalPolicy numerical_policy;
    int mtp_num_hidden_layers = 0;

    void validate() const;
};

// Derived exclusively from ModelGraph. No public constructor.
class ExecutionTopology {
 public:
    static ExecutionTopology derive(const ModelGraph& graph);
    // ... existing accessors: attention_layout(), maximum_*(), layer_uses_moe()
 private:
    ExecutionTopology() = default;
    // ... Category A fields
};

struct RuntimeTopology {
    CheckpointDimensions dims;
    ExecutionTopology exec;

    void validate() const;
    std::string fingerprint() const;
    std::string summary() const;
};
```

The private constructor is the keystone. Stage 1 cannot fabricate an
`ExecutionTopology`, so Defect 1 becomes a compile error rather than a silent
no-op, and Category C cannot exist — `derive` either populates a field or the
field does not belong in `ExecutionTopology`.

## 2.2 Collapse the stage contract

`ArchitectureResolutionStages::topology` currently returns a full
`RuntimeTopology`. It becomes:

```cpp
struct ArchitectureResolutionStages {
    std::function<CheckpointDimensions(const CheckpointView&)> dimensions;
    std::function<ModelGraph(const CheckpointView&, const CheckpointDimensions&)> graph;
    std::function<void(ResolvedModel&, const CheckpointView&)> weights;
    // ... capabilities, provenance unchanged
};
```

Note `graph` now **returns** the graph rather than mutating a half-built
`ResolvedModel`. Front-ends stop having write access to a partially-populated
model, which removes the remaining route to accidental ownership.

## 2.3 One dispatch per concern

Front-ends already must construct `LayerSpec` to build the graph, and
`LayerSpec::mixer_kind()` already derives the tag
(`include/celeg/model/graph.hpp:354-364`). Once front-ends emit `LayerSpec`
directly, `MixerKind` dispatch survives in exactly two compiler-checked
visitors:

| Concern | Location | Input |
| --- | --- | --- |
| Topology derivation | `ExecutionTopology::derive` | `const ModelGraph&` |
| Weight requirements | `derive_weight_requirements` | `const ModelGraph&`, `const ITensorNamingPolicy&` |

Adding a mixer kind means extending the `LayerSpec::mixer` variant; both
visitors then fail to compile until updated. No runtime
`"unsupported mixer kind"` throw is needed in either graph builder.

---

# 3. Phases

Each phase must leave the tree building and all tests green.

## Phase 0 — Pin behavior, fix the two live defects

**Goal:** establish a green regression baseline before any structural change,
and stop shipping the two known defects.

**Changes:**

- Add a regression test asserting a descriptor with
  `double_wide_shared_suffix: true` produces doubled FFN widths in the resolved
  graph. Fix by applying the doubling in
  `src/model/descriptor/graph_builder.cpp` when constructing the suffix layers'
  `DenseFeedForwardSpec`/`MixtureOfExpertsSpec`, and delete the dead stage-1
  block at `src/model/descriptor/architecture.cpp:431-436`.
- Add a regression test asserting `moe_routing_*` survives resolution for a
  grouped-routing checkpoint. Fix by populating the four fields from
  `MixtureOfExpertsSpec` in the MoE branch of `src/model/resolved.cpp:99-115`.
- Add characterization tests over the three front-ends asserting the final
  `RuntimeTopology` fingerprint for at least one representative checkpoint each
  (dense attention, hybrid Mamba2, MoE).

**Exit criteria:** both defects covered by failing-then-passing tests;
fingerprints pinned for all three front-ends.

## Phase 1 — Split `RuntimeTopology` (keystone)

**Goal:** make illegal ownership unrepresentable.

**Changes:**

- Introduce `CheckpointDimensions` and `ExecutionTopology` per section 2.1 in
  `include/celeg/model/resolved.hpp`.
- Convert `derive_runtime_topology_from_graph(RuntimeTopology&, const ModelGraph&)`
  into `ExecutionTopology ExecutionTopology::derive(const ModelGraph&)` —
  returning by value, not mutating in place. The in-place mutation signature is
  what allowed the two-owner split to exist.
- Delete every stage-1 write to a Category A field. This is where the bulk of
  `src/model/inference.cpp:32-646` and
  `src/model/descriptor/architecture.cpp:30-459` disappears.
- Derive the two leaked fields identified in section 1.3 inside
  `ExecutionTopology::derive`, and delete their write sites
  (`src/model/descriptor/architecture.cpp:420-422`,
  `src/model/inference.cpp:208`):
  - `shared_kv_group_count` := count of distinct non-negative
    `AttentionSpec::kv_sharing.group` values across attention layers.
  - `has_split_attention_norms` := whether any `LayerSpec` enables
    `post_attention_norm` / `pre_feed_forward_norm`. Retain the model-wide
    boolean in this phase to keep the ~9 backend read sites compiling; converting
    those to the per-layer `LayerSpec` norms is follow-on work, tracked as a
    non-goal below.
- Update all `topology.<field>` readers across backends, `program.cpp`, and
  tests to `topology.dims.<field>` / `topology.exec.<field>`. Mechanical, but
  wide — expect this to touch code outside `src/model/`.

**Exit criteria:** `ExecutionTopology` has no public constructor; the Phase 0
fingerprints are unchanged; no front-end names a Category A field;
`grep -rn "split_attention_norms\|shared_kv_group_count" src/model/` shows reads
only, no writes outside `derive`.

**Note:** deriving `has_split_attention_norms` will change behavior for the
descriptorless path, which currently always resolves it to `false`. This is a
capability fix, not a regression, but it will move a Phase 0 fingerprint — pin
the corrected value and note it in the commit.

## Phase 2 — Front-ends emit `LayerSpec`; collapse dispatch

**Goal:** eliminate the parallel `MixerKind` chains (section 1.4, item 1).

**Changes:**

- Reshape `build_dense_transformer_graph` (`src/model/graph_builder.cpp:8-55`)
  and `build_descriptor_graph` (`src/model/descriptor/graph_builder.cpp:29-123`)
  so each is purely `source -> std::vector<LayerSpec>`, with no `MixerKind`
  if/else chain and no `"unsupported mixer kind"` throw.
- Ensure both visitors in section 2.3 are exhaustive `std::visit` over the
  variant with no `else` fallback, so a new alternative is a compile error.

**Exit criteria:** `grep -rn "MixerKind" src/model/` shows dispatch only in
`ExecutionTopology::derive` and `derive_weight_requirements`.

## Phase 3 — One weight-requirement derivation

**Goal:** eliminate the drifted triplicate (section 1.4, item 2).

**Changes:**

- Add `derive_weight_requirements(const ModelGraph&, const ITensorNamingPolicy&)`
  built from the complete descriptor implementation
  (`src/model/descriptor/weight_requirements.cpp:8-77`, `:139-197`), which
  already handles query/key norm, output gate, `kv_sharing`, and MoE.
- Delete `build_dense_weight_plan` (`src/model/weight_plan.cpp:16-88`) and its
  declaration in `include/celeg/model/weight_plan.hpp:10-11`. Update
  `tests/weight_plan_test.cpp:33`, its only direct caller.
- Delete the inline attention binding at `src/model/inference.cpp:544-606`,
  including the twice-written Q/K-norm binding at `:573-606`.

**Exit criteria:** one implementation of the `AttentionSpec -> TensorRequest`
mapping; a hand-written architecture using query/key norm or MoE produces a
complete weight plan.

## Phase 4 — One resolution pipeline

**Goal:** eliminate the duplicated stage ordering (section 1.4, item 3).

**Changes:**

- Make descriptorless inference an `IArchitecture` implementation whose
  `dimensions`/`graph`/`weights` stages wrap the existing
  `CanonicalModelFacts` production.
- Delete `ResolutionAssembler::assemble` (`src/model/inference/synthesis.cpp:28-43`)
  and route it through `resolve_architecture_stages`.

**Exit criteria:** `derive` is called from exactly one place in the codebase.

## Phase 5 — One error model

**Goal:** resolve the four-way exception inconsistency (section 1.4, item 6).

**Changes:**

- Extend `ResolutionFailureKind` to cover invariant violations currently thrown
  as `std::invalid_argument`/`std::runtime_error`.
- Convert `validate()` throughout `definition.cpp`, `resolved.cpp`,
  `program.cpp` to throw `celeg::ResolutionError` with an appropriate kind and
  evidence.

**Exit criteria:** callers can distinguish bad-checkpoint from
internal-invariant failures without matching on message text.

## Phase 6 — Residual cleanups

**Goal:** close the remaining small forks (section 1.4, items 4, 5, 7).

**Changes:**

- Route `src/model/descriptor/probe_matcher.cpp:28-32` through
  `selected_key` (`src/model/descriptor/field_resolver.cpp:7-17`) so
  `{architecture}` substitution applies to probe conditions. Add a test for a
  probe condition using the placeholder.
- Replace the hand-rolled candidate loops at `src/model/inference.cpp:45-59`,
  `:410-421`, `:441-455` with `inference_detail::find_unique`, and move the
  embedding/head/norm candidate lists into `src/model/inference/support.cpp`
  beside their siblings.
- Extract a `row_abs_max` helper in `src/model/weights/quantization.cpp`, shared
  by `quantize_bf16_rows_into` and `quantize_bf16_rows_int4_into`.

**Exit criteria:** no remaining duplicated candidate-scan or key-selection
logic in `src/model/`.

---

# 4. Non-goals

- Changing execution semantics or numerical output. Phase 0 fingerprints must
  hold through Phase 6, except for the two Phase 0 defect fixes.
- Backend (`CPU`/`CUDA`) refactoring beyond the mechanical field renames forced
  by Phase 1. In particular, converting the ~9 `has_split_attention_norms` read
  sites (section 1.3) to consume per-layer `LayerSpec` norms is **follow-on
  work**. Phase 1 removes the family-knowledge *write* channel; removing the
  coarse model-wide *read* is a separate, backend-scoped change.
- Descriptor schema changes. `double_wide_shared_suffix` keeps its current
  spelling; only its implementation moves.
- Adding new mixer kinds or architectures.

---

# 5. Risks

| Risk | Mitigation |
| --- | --- |
| Phase 1 touches every `topology.*` reader repo-wide | Land Phase 1 as a single mechanical commit after the rename is compiler-verified; the private constructor makes missed cases build failures, not silent bugs |
| Phase 0 fingerprints may encode current defects | Pin fingerprints *after* the two defect fixes, not before |
| Category B list may be incomplete | Phase 1 is compiler-verified: any field neither front-end-written nor `derive`-populated fails to compile into either type |
| Hidden readers of `moe_routing_*` outside `src/model/` | Phase 0 defect-2 fix restores correct values before any structural change, so such readers are fixed regardless |

---

# 6. Sequencing summary

```text
Phase 0  Pin behavior + fix 2 live defects      <-- independently valuable
   |
   v
Phase 1  Split RuntimeTopology                  <-- keystone; forces 2-4
   |
   +--> Phase 2  Collapse MixerKind dispatch
   |
   +--> Phase 3  One weight-requirement derivation
   |
   +--> Phase 4  One resolution pipeline
   |
   v
Phase 5  One error model                        <-- independent
   |
   v
Phase 6  Residual cleanups                      <-- independent
```

Phases 2, 3, and 4 are mutually independent once Phase 1 lands and may proceed
in parallel. Phases 5 and 6 are independent of everything and may be pulled
forward if convenient.
