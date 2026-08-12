# CELEG Model Topology Ownership Refactoring Plan

## Status

Proposed architectural refactor. Reviewed against the tree; all code references
below were verified, and three live defects (section 1.2) were confirmed by
reading the call chains rather than inferred.

Target repository: `celsowm/celeg`.

Scope: `include/celeg/model/**` and `src/model/**`.

This plan has **no backward-compatibility constraint**. Struct layouts, function
signatures, and public headers may change as clean breaks. No deprecated
wrappers, no transitional shims, no re-exported legacy names.

## Primary goal

> `ModelGraph` is the single semantic owner of execution topology, and the type
> system must make any other owner impossible to express.

The refactor is complete only when three things hold:

1. A resolution front-end is physically unable to write a graph-derived field.
2. Every `MixerKind` dispatch is exhaustively checked by the compiler, in
   exactly one place per concern.
3. No backend branches on a model-wide flag where the graph carries a per-layer
   answer.

The third clause is new to this revision. The original plan deferred it as
backend work, which would have left the agnosticism goal unmet at eleven sites
(section 1.3) — a refactor that closes the write channel while leaving the read
channel intact does not actually establish single ownership. Section 1.3 also
shows the conversion is currently free, which removes the argument for
deferring it.

---

# 1. Problem statement

CELEG already has the correct pipeline shape. It is *not* unified: the two
resolution paths reach the same stage order through different code (section 1.6).
Both end up running these stages —
`resolve_architecture_stages` (`src/model/architecture.cpp:9-25`) for the
descriptor path, and a hand-rolled equivalent in
`ResolutionAssembler::assemble` (`src/model/inference/synthesis.cpp:30-64`) for
the automatic path:

```text
stages.topology(checkpoint)          <-- stage 1: front-end builds RuntimeTopology
      |
      v
  topology.validate()
      |
      v
stages.graph(model, checkpoint)      <-- stage 2: front-end builds ModelGraph,
      |                                            READING stage 1's tables
      v
derive_runtime_topology_from_graph() <-- stage 3: OVERWRITES most of stage 1
      |
      v
  topology.validate()
      |
      v
stages.weights(model, checkpoint)    <-- stage 4
      |
      v
  model.validate()
```

The stage-2 arrow is the detail the rest of this document turns on: stage 1's
graph-owned fields are not written and discarded, they are written, *consumed by
stage 2*, and then recomputed from stage 2's output. The cycle is what makes the
ownership question ambiguous, and what makes the fix larger than a struct split.

Two front-ends feed this pipeline:

| Front-end | Source | Entry point | In-tree status |
| --- | --- | --- | --- |
| Descriptorless inference | Tensor names + metadata | `src/model/inference/synthesis.cpp:30-64` | **the only live path** |
| Descriptor-driven | JSON descriptor | `src/model/descriptor/architecture.cpp:456` | extension point, zero instances |

An earlier revision of this plan listed a third front-end, "hand-written
architectures in C++, calling `resolve_architecture_stages` directly". **There is
no such front-end.** `AutomaticArchitecture` (`src/model/automatic_architecture.cpp:14-52`)
is the only architecture registered by
`src/composition/builtin_runtime.cpp:19`, and `build_dense_transformer_graph`
has exactly one caller — `src/model/inference/synthesis.cpp:14`, inside the
descriptorless path. What looked like a separate hand-written front-end is the
descriptorless front-end's own graph builder. See section 1.6 for why this
matters more than a bookkeeping correction.

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
`derive_runtime_topology_from_graph` (`src/model/resolved.cpp:18-127`):

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
has_split_attention_norms
```

Plus `shared_kv_group_count`, which is neither derived nor read and should be
deleted rather than relocated.

`has_split_attention_norms` is *not* currently derived — see section 1.3. They belong here and
Phase 1 must resolve them.

**Category A fields are overwritten, but they are not dead.** Both graph builders
*read* them to construct the graph in the first place:
`build_dense_transformer_graph` (`src/model/graph_builder.cpp:9-54`) and
`build_descriptor_graph` (`src/model/descriptor/graph_builder.cpp:31-120`) each
open with `const RuntimeTopology& topology = model.topology;` and drive their
entire per-layer loop from `mixer_kinds`, `attention_layouts`,
`feed_forward_kinds`, `moe_*`, `conv_cache`/`conv_dim`, and
`feed_forward_intermediates`. The round trip is
`stage 1 -> graph -> stage 3 -> topology`, not a discarded write.

This matters for sequencing: stage-1 Category A writes cannot simply be deleted
(Phase 1) while the graph builders still consume them. The front-ends must first
emit `LayerSpec` directly (Phase 2). See section 3 for the corrected ordering.

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

## 1.2 This has already produced three live defects

### Defect 1 — `double_wide_shared_suffix` is a silent no-op

`src/model/descriptor/architecture.cpp:431-436` doubles
`feed_forward_intermediates` and `max_feed_forward_intermediate` during stage 1.

Both are Category A. They are overwritten at `src/model/resolved.cpp:22`,
`:88`, and `:124-127` from `graph.layers[i].feed_forward.intermediate_size`.
`src/model/descriptor/graph_builder.cpp` never applies the doubling.

The flag is parsed (`src/model/descriptor/parser.cpp:121`) and stored
(`src/model/descriptor/detail.hpp:148`), but has no effect. Any descriptor
declaring it silently receives un-doubled FFN widths.

### Defect 2 — grouped-MoE routing is zeroed in topology, and CUDA reads topology

`src/model/inference.cpp:192-214` computes and validates the `moe_routing_*`
fields, including divisibility and range checks.
`src/model/graph_builder.cpp:43-44` correctly carries them into
`MixtureOfExpertsSpec`.

Stage 3 then zeroes them (Category C, `src/model/resolved.cpp:47-50`) and never
reads them back, even though `MixtureOfExpertsSpec` carries them
(`include/celeg/model/graph.hpp:370-373`).

After resolution, `topology` reports "no grouped routing" while `graph` reports
grouped routing. The semantic program is fine — `src/model/program.cpp:461-477`
reads the graph. **The CUDA backend is not.**
`moe_router_config` (`include/celeg/detail/model/types.hpp:442-455`) builds the
router config from `RuntimeTopology`:

```cpp
cfg.group_count       = shape.moe_routing_group_count;        // always 0
cfg.experts_per_group = shape.moe_routing_experts_per_group;  // always 0
cfg.groups_per_token  = shape.moe_routing_groups_per_token;   // always 0
cfg.group_score_top_k = shape.moe_routing_group_score_top_k;  // always 0
```

It is called from `src/backend/cuda/model/residency.cu:237`, `:310`, and
`src/backend/cuda/model/packed_operators.cu:466` — i.e. the standalone decode,
prefill, and packed paths. `src/backend/cuda/moe/route.cu:66` skips the entire
grouped-selection branch when `group_count == 0`.

So every grouped-routing checkpoint (`noaux_tc`-style: DeepSeek-V3-shaped
routing, and anything setting `topk_group`) **silently falls back to ungrouped
top-K on CUDA**, producing wrong expert selections with no error. This is not a
latent hazard; it is a shipping correctness bug, and it raises the priority of
the Phase 0 fix.

### Defect 3 — descriptor MoE spec is field-shifted

`src/model/descriptor/graph_builder.cpp:105-109` aggregate-initializes
`MixtureOfExpertsSpec` with **12 initializers for a 14-field struct**
(`include/celeg/model/graph.hpp:360-378`), supplying only two of the four
`routing_*` fields:

```cpp
layer.feed_forward = MixtureOfExpertsSpec{
    topology.moe_intermediate, topology.num_experts, topology.experts_per_token,
    topology.normalize_topk, topology.use_expert_bias, topology.routed_scaling_factor,
    0, 0, topology.shared_expert_intermediate > 0,          // <-- shifts here
    topology.shared_expert_intermediate, false, topology.moe_router_softmax};
```

Every field from position 9 onward lands one or two slots early. The resulting
spec is:

| Field | Intended | Actually assigned |
| --- | --- | --- |
| `routing_groups_per_token` | 0 | `shared_expert_intermediate > 0` (0 or 1) |
| `routing_group_score_top_k` | 0 | `shared_expert_intermediate` |
| `has_shared_expert` | `shared_expert_intermediate > 0` | `false` |
| `shared_intermediate_size` | `shared_expert_intermediate` | `moe_router_softmax` (0 or 1) |
| `shared_before_routed` | `false` | default `false` |
| `router_softmax` | `moe_router_softmax` | default `false` |

It compiles because every mismatched pair is `bool`→`int` (an integral
promotion, not narrowing).

Consequences for any descriptor-driven MoE model:

- **Shared experts are silently dropped from the graph.** Meanwhile
  `src/model/descriptor/weight_requirements.cpp:212-222` still requests the
  shared-expert tensors from `topology.shared_expert_intermediate`, so the
  weights are loaded and then never executed.
- **Softmax routing is silently downgraded to sigmoid**, since `router_softmax`
  falls back to its default.

This defect is the mirror image of Defect 2 — there the topology was wrong and
the graph right; here the graph is wrong and the topology right. It is direct
evidence that "the graph is authoritative" is an aspiration the current code
does not hold up, and it means Phase 0 fingerprints cannot be pinned before it
is fixed.

## 1.3 The dual-ownership channel is also an agnosticism leak

`ZERO_MODEL_FAMILIES_REFACTORING_PLAN.md` states the governing constraint:

> *CELEG must support semantics, not model families.*

Stage-1 topology authorship is a channel through which family-specific
knowledge reaches backend control flow **without passing through the semantic
graph**. Two fields currently exploit it.

### `has_split_attention_norms`

Written only from a descriptor flag
(`src/model/descriptor/architecture.cpp:422` reads
`descriptor_.split_attention_norms`). Never derived from the graph. Read by
eleven backend sites to branch execution:

```text
src/backend/cpu/model_forward_chunk.cpp:487, :509
src/backend/cpu/model_forward_token.cpp:100, :119, :129
src/backend/cpu/packed_execution.cpp:324, :487
src/backend/cpu/weights_loader.cpp:60, :66
src/backend/cuda/model/expert_setup.cpp:27
src/backend/cuda/model/weight_setup.cpp:85
```

It is also read inside `src/model/` itself, at
`src/model/descriptor/weight_requirements.cpp:81` and `:187`, where it gates
whether `AttentionPostNorm` / `FfnOutputNorm` tensors are requested at all. That
read has consequences for Phase 3 — see there.

Meanwhile `LayerSpec` already carries the real, *per-layer* semantic truth:
`post_attention_norm`, `pre_feed_forward_norm`, `post_feed_forward_norm`
(`include/celeg/model/graph.hpp:385-389`).

So backends branch on a coarse model-wide boolean authored by a descriptor,
while the graph holds a finer-grained answer they ignore. This is lossy: a
checkpoint with split norms on only some layers cannot be expressed.

**The flag is dead today, which makes the conversion free.** Tracing every
write:

- `descriptor_.split_attention_norms` is parsed
  (`src/model/descriptor/parser.cpp:119`) and stored
  (`src/model/descriptor/detail.hpp:120`), but **no descriptor in the repo sets
  it** and no test exercises it. It is always `false`.
- The per-layer norms are written in exactly one place —
  `src/model/descriptor/graph_builder.cpp:71-75` — under `if
  (descriptor.split_attention_norms)`. The descriptorless front-end
  (`src/model/inference.cpp`) and the hand-written front-end
  (`src/model/graph_builder.cpp`) never set them at all.

So the flag and the per-layer norms are perfectly correlated, and both are
universally disabled. All eleven backend branches are currently dead code.

Two consequences, both good:

1. **Replacing `shape.has_split_attention_norms` with the per-layer
   `LayerSpec` norms is exactly behavior-preserving.** `false` maps to
   "disabled" everywhere. No descriptor audit, no fingerprint movement, no
   capability change to justify in a commit message.
2. **After the conversion the field has no readers, so it is deleted rather
   than derived** — the same disposition as `shared_kv_group_count`.

This is why the backend conversion is folded into this plan rather than
deferred (see section 3). Deferred, it stays a family-shaped boolean that
backends branch on. Done now, while it is provably inert, it costs one
mechanical pass and closes the leak channel permanently. The cost only rises:
the first descriptor that sets the flag turns eleven dead branches live and
makes the conversion a behavior change requiring numerical validation.

The conversion is also nearly one-for-one, because the per-layer spec is
*already in scope at every call site*. From
`src/backend/cpu/weights_loader.cpp:60`:

```cpp
if (shape.has_split_attention_norms) {                       // model-wide flag
    common.post_attention_norm = load_norm(TensorRole::AttentionPostNorm,
                                           layer_program.post_attention_norm);  // per-layer spec
}
```

The gate is redundant with `layer_program.post_attention_norm.enabled()`, which
is already the second argument. The same pattern holds at
`src/backend/cuda/model/weight_setup.cpp:85` (`semantic_layer.post_attention_norm`)
and `src/backend/cpu/model_forward_token.cpp:100` (`semantics.post_attention_norm.epsilon`).

### `shared_kv_group_count` — delete it, do not derive it

Computed heuristically at `src/model/descriptor/architecture.cpp:420-421` as
`shared_layers > 0 ? (has_attention_variants ? 2 : 1) : 0`, and hardcoded to `0`
at `src/model/inference.cpp:215`.

The graph already carries the authoritative per-layer answer in
`AttentionSpec::kv_sharing` (`KvSharingSpec { group, publishes }`,
`include/celeg/model/graph.hpp:62-67`), and
`src/model/descriptor/weight_requirements.cpp:62` already consumes that per-layer
truth to decide whether to request K/V tensors.

**The field has no readers.** A repo-wide search for `shared_kv_group_count`
returns exactly three hits: the declaration
(`include/celeg/model/resolved.hpp:85`) and the two write sites above. Nothing —
no backend, no `program.cpp`, no test — ever reads it.

So the correct treatment is deletion, not derivation. Deriving it would add a
graph traversal to produce a value nobody consumes. Phase 1 should drop the
field from `RuntimeTopology` outright and delete both writes.

### Consequence for this plan

Neither field belongs in `ExecutionTopology`. Both are deleted:
`shared_kv_group_count` because nothing reads it, `has_split_attention_norms`
because its eleven readers convert to a strictly better per-layer source at
zero behavioral cost.

This upgrades the refactor from *neutral* on agnosticism to *actively closing
both leak channels*. Once `ExecutionTopology` has no public constructor and
neither field exists, a descriptor cannot inject a family-shaped flag into
backend control flow at all — not because the type system forbids that
particular flag, but because the only channel that carried it is gone.

## 1.4 Secondary duplication (symptoms, not causes)

These follow from 1.1 and are resolved by the same refactor. Line references
below were verified against the tree as of this revision:

- **`MixerKind` dispatch appears four times**: `src/model/resolved.cpp:60-116`,
  `src/model/graph_builder.cpp:21-36`,
  `src/model/descriptor/graph_builder.cpp:76-99`, and
  `src/model/descriptor/weight_requirements.cpp`. Only the first is a
  `std::visit`; the other three are `MixerKind` if/else chains ending in a
  runtime throw, which is precisely the shape that goes silently stale when a
  mixer kind is added.
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

## 1.5 Blast radius outside `src/model/`

The split renames every topology field access repo-wide. Measured against the
current tree:

| Consumer | Accesses | Files |
| --- | --- | --- |
| Backends (`src/backend/`, `include/celeg/backend/`, `include/celeg/detail/`) | ~1,057 | 36 |
| Tests | 172 | 19 |

Split by destination:

| Destination | Accesses | Notes |
| --- | --- | --- |
| `topology.dims.*` | ~755 | **`hidden` alone is 539** |
| `topology.exec.*` | ~301 | concentrated: 6 files hold ~148 |

Three observations shape the plan.

**`hidden` is half the diff.** 539 of ~1,057 backend accesses are `shape.hidden`.
See section 2.4 for the disposition; it is worth deciding deliberately rather
than absorbing 51% of the churn by default.

**Category A reads are concentrated, not smeared.** Six files hold roughly half:
`src/backend/cpu/weights_loader.cpp` (33), `src/backend/cuda/model/expert_setup.cpp`
(29), `src/backend/cuda/model/resources.cpp` (25),
`src/backend/cuda/model/weight_setup.cpp` (24),
`src/backend/cuda/model/session_resources.cpp` (22),
`src/backend/cuda/model/packed_validation.cu` (15). The remaining 30 files average
five apiece. Weight setup and workspace sizing are the real topology consumers;
the execution paths barely touch it. Review effort should follow that curve.

**No signature churn.** `RuntimeTopology` survives as the pair type, so the ~15
headers taking `const RuntimeTopology&` are untouched and backends keep holding
a `RuntimeTopology shape_` member. This is what keeps ~1,000 of these sites
genuinely mechanical rather than a redesign.

### The part that is not mechanical: test fixtures

Nine test files hand-construct a `RuntimeTopology` and populate Category A
fields directly: `tests/cpu_kv_topology_test.cpp`, `tests/cpu_sampler_test.cpp`,
`tests/cuda_kernels_test.cu`, `tests/expert_offload_test.cpp`,
`tests/expert_residency_test.cu`, `tests/packed_workspace_test.cpp`,
`tests/policy_test.cpp`, and others. **Every one stops compiling the moment
`ExecutionTopology`'s constructor goes private** — which is the keystone working
as designed, but it is real work the original plan did not account for.

From `tests/expert_offload_test.cpp:36-38`:

```cpp
// Derived field used by the KV planner; the 8B-A1B model has 6 attention layers.
shape.attention_layer_count = 6;
```

A test hand-setting a derived field, with a comment acknowledging that it is
derived — exactly the defect class this refactor exists to eliminate, reproduced
in the fixtures that guard it. Each must be rewritten to build a
`std::vector<LayerSpec>` and call `ExecutionTopology::derive`. The result is
strictly better (fixtures stop being able to describe impossible models), but it
is nine rewrites, not a rename.

## 1.6 The descriptor front-end is entirely unexercised

This changes how the defects should be prioritized and how the refactor should
be validated, so it is stated separately from the code smells above.

Descriptors are an optional runtime extension layer. `register_descriptor_architectures`
(`src/model/descriptor/registration.cpp:14-47`) scans a directory of `.json`
files and returns immediately if the directory is absent or empty. The directory
is `CELEG_DESCRIPTOR_DIRECTORY` — `${CMAKE_CURRENT_SOURCE_DIR}/descriptors`
(`CMakeLists.txt:72`), falling back to the install datadir.

Three facts, each verified:

- **`descriptors/` exists and is empty.** No descriptor JSON is tracked by git.
- **No test references descriptors at all.** `grep -rli descriptor tests/`
  returns nothing — not a fixture, not a parse test, not a resolution test.
- **`AutomaticArchitecture` is the only architecture ever registered.**

So the descriptor front-end has zero in-tree instances and zero coverage. It is
a published extension point exercised only by whatever JSON downstream users
supply.

### Consequence 1 — the three defects have very different reach

| Defect | Path | Who is affected today |
| --- | --- | --- |
| 2 — grouped-MoE routing zeroed | descriptorless (`synthesis.cpp:59` runs stage 3) | **everyone**, on every grouped-routing checkpoint, on CUDA |
| 1 — `double_wide_shared_suffix` no-op | descriptor only | external descriptor authors only |
| 3 — MoE spec field-shift | descriptor only | external descriptor authors only |

Defect 2 is the only one producing wrong output for an in-tree user, and it does
so silently on the sole live path. It should be fixed first and on its own — it
is a small, self-contained change to `src/model/resolved.cpp:99-115` that needs
no descriptor infrastructure.

Defects 1 and 3 are real but latent. They are worth fixing in Phase 0 anyway
(they are cheap, and Defect 3 is a one-line correctness trap that will bite the
first descriptor author who uses shared experts), but they are not the emergency
that an earlier revision of this document implied.

### Consequence 2 — descriptor-path changes are unverifiable as things stand

Phases 1a and 2 substantially rewrite `src/model/descriptor/`. With no
descriptors and no tests, **nothing in the build will tell you if that rewrite
breaks the descriptor front-end.** The compiler checks types; there is no
behavioral net whatsoever.

This is a prerequisite, not a caveat. Phase 0 must stand up minimal descriptor
test infrastructure before any structural work touches that directory:

- A small in-tree descriptor JSON fixture directory used by tests.
- Resolution tests over at least three shapes: dense attention, MoE with shared
  experts and softmax routing (covers Defect 3), and a suffix descriptor with
  `double_wide_shared_suffix` (covers Defect 1).
- A descriptor setting `split_attention_norms: true` (covers Phase 1c, which
  currently has nothing exercising the feature it converts).

The original Phase 0 wording — "add a regression test asserting a descriptor
with `double_wide_shared_suffix: true` produces doubled FFN widths" — reads as a
one-line addition. It is not; it requires building the harness that would run
it. Budget for that explicitly.

### Consequence 3 — a note on Phase 1c's safety argument

Section 1.3 argues the split-norm conversion is free because no descriptor sets
the flag. Strictly, that is true *in-tree*; an external descriptor could set it.

The safety argument does not depend on usage. `descriptor.split_attention_norms`
is the sole condition guarding the only writes to the per-layer norms
(`src/model/descriptor/graph_builder.cpp:71-75`), so flag-true and
norms-enabled are the same condition by construction. Reading the per-layer
norms instead of the flag is therefore behavior-preserving for any descriptor,
in-tree or not. What the in-tree emptiness adds is only that the conversion
cannot regress an existing user *today* — which is an argument for doing it
now rather than after the first descriptor ships.

---

# 2. Target design

## 2.1 Split the god-struct by owner

```cpp
// Authored by the front-end. The checkpoint is the only source.
// Note `hidden` is deliberately absent — see section 2.4.
struct CheckpointDimensions {
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

Note what this does *not* buy. Defect 3 is a wrong `MixtureOfExpertsSpec`
written into the graph itself; no amount of topology encapsulation prevents it,
because the graph is the trusted input. Two complementary guards are worth
adopting alongside the split:

- **Designated initializers** (`MixtureOfExpertsSpec{.intermediate_size = ..., ...}`)
  in both graph builders. A field-shift becomes a compile error, and a
  reordered struct stops silently re-binding values.
- **`ExecutionTopology::derive` cross-checks nothing.** After the split,
  `ResolvedModel::validate` (`src/model/resolved.cpp:498-515`) loses its
  graph-vs-topology comparisons — they become tautological, since `exec` is a
  pure function of `graph`. Delete them rather than leaving checks that can
  never fire; the invariants they were guarding must move into
  `ModelGraph::validate`, which is now the only place a wrong graph can be
  caught. In particular, `MixtureOfExpertsSpec` currently has no validator at
  all — Defect 3 would have been caught by one.

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

## 2.4 `hidden` moves to `ModelGraph`

`hidden` is 539 of the ~1,057 backend accesses (section 1.5). Where it lands
decides half the diff, so it deserves an explicit decision rather than a
default.

Three options were considered:

| Option | Backend churn | Verdict |
| --- | --- | --- |
| `CheckpointDimensions::hidden` (original plan) | 539 renames | Consistent, but leaves Phase 2's blocker unsolved |
| Keep `hidden` top-level on `RuntimeTopology` | 0 renames | Cheapest, but `RuntimeTopology` becomes `dims` + `exec` + one loose field — the exact "no marker" shape section 1.1 indicts |
| **`ModelGraph::hidden`, surfaced via `ExecutionTopology`** | 539 renames | **Chosen** |

`hidden` is the residual-stream width. Every `LayerSpec` projection is defined
relative to it — `AttentionSpec::query_width()`, every `TensorRequest` shape in
`weight_requirements.cpp`, the norm widths. It is graph semantics that happens
to be *read from* the checkpoint, which is true of every graph field; sourcing
is not ownership. That is the whole thesis of this document, applied
consistently.

Deciding it this way also dissolves the Phase 2 blocker for free:
`derive_weight_requirements(const ModelGraph&, const ITensorNamingPolicy&)` can
keep its two-argument signature, because the `topology.hidden` it needs is now
reachable from the graph it was already given.

The cost is that the cheapest option is not the chosen one. Accepted
deliberately: keeping `hidden` loose on `RuntimeTopology` would save ~500
mechanical renames by preserving in miniature the exact defect the refactor
exists to remove.

Backends therefore read `shape.exec.hidden`. `CheckpointDimensions` keeps
`intermediate`, which — unlike `hidden` — really is a checkpoint default that
per-layer `feed_forward.intermediate_size` overrides.

---

# 3. Phases

Each phase must leave the tree building and all tests green.

## Phase 0 — Pin behavior, fix the three live defects

**Goal:** establish a green regression baseline before any structural change,
and stop shipping the three known defects. Worth landing on its own schedule
regardless of whether the rest of the refactor proceeds.

Ordered by reach (section 1.6), not by convenience:

**Changes:**

- **Defect 2 first, and alone.** It is the only defect affecting the live path,
  it silently corrupts expert selection on CUDA, and the fix is self-contained:
  populate the four `moe_routing_*` fields from `MixtureOfExpertsSpec` in the MoE
  branch of `src/model/resolved.cpp:99-115`. Land it as its own commit so it can
  be backported independently of everything else in this document. Add a test
  asserting `moe_routing_*` survives resolution for a grouped-routing checkpoint,
  **and** one asserting `moe_router_config(topology)` yields a non-zero
  `group_count` — the second is what actually pins the CUDA symptom.
- **Then stand up descriptor test infrastructure** (section 1.6, consequence 2).
  Nothing below this line can be verified without it, and neither can Phases 1a
  or 2.
- **Then Defect 3.** Rewrite `src/model/descriptor/graph_builder.cpp:105-109` with designated
  initializers, supplying all four `routing_*` fields, `has_shared_expert`,
  `shared_intermediate_size`, and `router_softmax` correctly. Add a regression
  test over a descriptor with a shared expert and softmax routing, asserting
  the resolved `MixtureOfExpertsSpec`. Apply the same designated-initializer
  treatment to `src/model/graph_builder.cpp:40-46` so the two builders cannot
  drift again.
- **Then Defect 1.** Add a regression test asserting a descriptor with
  `double_wide_shared_suffix: true` produces doubled FFN widths in the resolved
  graph. Fix by applying the doubling in
  `src/model/descriptor/graph_builder.cpp` when constructing the suffix layers'
  `DenseFeedForwardSpec`/`MixtureOfExpertsSpec`, and delete the dead stage-1
  block at `src/model/descriptor/architecture.cpp:431-436`.
- Add characterization tests over **both** front-ends (section 1.6) asserting the
  final `RuntimeTopology` fingerprint for representative checkpoints:
  dense attention, hybrid Mamba2, and MoE on the descriptorless path; the three
  descriptor fixtures above on the descriptor path.

**Exit criteria:** all three defects covered by failing-then-passing tests;
descriptor test infrastructure exists and is exercised; fingerprints pinned for
both front-ends, *after* the fixes.

**Caveat:** `RuntimeTopology::fingerprint` (`src/model/resolved.cpp:257-349`)
does not cover the fields any of the three defects touch — it emits
`num_experts`, `experts_per_token`, and `moe_intermediate`, but no `moe_routing_*`,
no shared-expert width, no `router_softmax`, and no norm flags. Fingerprint
pinning is therefore necessary but far from sufficient as a regression net;
the per-defect assertions above are doing the real work. Consider extending
`fingerprint()` to cover the MoE routing and shared-expert fields as part of
this phase.

## Phase 1 — Front-ends emit `LayerSpec`, then split `RuntimeTopology` (keystone)

**Goal:** make illegal ownership unrepresentable.

**Ordering correction.** The original plan split this into Phase 1 (split the
struct) and Phase 2 (front-ends emit `LayerSpec`), with Phase 2 depending on
Phase 1. The dependency runs the other way, and it is hard: as established in
section 1.1, both graph builders *consume* Category A fields. You cannot delete
the stage-1 writes while `build_dense_transformer_graph` and
`build_descriptor_graph` still read `topology.mixer_kinds`,
`topology.attention_layouts`, `topology.feed_forward_kinds`, and the `moe_*`
scalars to decide what to build. Nor can `ExecutionTopology` get a private
constructor while stage 1 must hand a populated topology to stage 2.

So the two are one atomic change, sequenced internally:

1. **1a — front-ends emit `LayerSpec`.** Reshape each front-end so it produces
   `std::vector<LayerSpec>` directly instead of populating topology tables that
   a graph builder then transcribes. This deletes
   `build_dense_transformer_graph` and the per-layer loop of
   `build_descriptor_graph`; the `MixerKind` if/else chains and both
   `"unsupported mixer kind"` throws go with them. This is the bulk of the work
   and the bulk of the risk.
2. **1b — split the struct.** Only once nothing reads Category A from topology
   can the split and the private constructor land.

Land 1a and 1b as separate commits, but treat them as one non-splittable unit
of work: the tree is coherent after 1a (topology is still written, just no
longer read by the builders) and only becomes *enforced* after 1b.

**Changes:**

- Reshape `build_dense_transformer_graph` (`src/model/graph_builder.cpp:8-56`)
  and `build_descriptor_graph` (`src/model/descriptor/graph_builder.cpp:29-123`)
  so each is purely `source -> std::vector<LayerSpec>`, with no `MixerKind`
  if/else chain and no `"unsupported mixer kind"` throw.
- Introduce `CheckpointDimensions` and `ExecutionTopology` per section 2.1 in
  `include/celeg/model/resolved.hpp`.
- Convert `derive_runtime_topology_from_graph(RuntimeTopology&, const ModelGraph&)`
  into `ExecutionTopology ExecutionTopology::derive(const ModelGraph&)` —
  returning by value, not mutating in place. The in-place mutation signature is
  what allowed the two-owner split to exist.
- Delete every stage-1 write to a Category A field. This is where the bulk of
  `src/model/inference.cpp:32-646` and
  `src/model/descriptor/architecture.cpp:30-459` disappears.
- Delete `shared_kv_group_count` from `RuntimeTopology`
  (`include/celeg/model/resolved.hpp:85`) along with both write sites
  (`src/model/descriptor/architecture.cpp:420-421`,
  `src/model/inference.cpp:215`). It has no readers. (`has_split_attention_norms`
  is handled in Phase 1c.)
- Move `hidden` to `ModelGraph` per section 2.4, and rename the 539 backend
  accesses to `shape.exec.hidden`.
- Rewrite the nine hand-built test fixtures (section 1.5) to construct a
  `std::vector<LayerSpec>` and call `ExecutionTopology::derive` instead of
  setting Category A fields directly. These break at 1b by construction; budget
  for them explicitly rather than discovering them in the build log.
- Update all `topology.<field>` readers across backends, `program.cpp`, and
  tests to `topology.dims.<field>` / `topology.exec.<field>`. Mechanical, but
  wide — expect this to touch code outside `src/model/`.

- Ensure both visitors in section 2.3 are exhaustive `std::visit` over the
  variant with no `else` fallback, so a new alternative is a compile error.
  Note the current `derive` visitor (`src/model/resolved.cpp:60-84`, `:85-116`)
  is *not* exhaustive: both end in a bare `else` that silently treats any
  unrecognized alternative as `MlpBlockSpec` / `MixtureOfExpertsSpec`
  respectively. Replacing those with explicit `if constexpr` arms plus a
  `static_assert(always_false<Mixer>)` fallback is a prerequisite for the
  compile-time exhaustiveness this plan promises.

**Exit criteria:** `ExecutionTopology` has no public constructor; the Phase 0
fingerprints are unchanged; no front-end names a Category A field;
`grep -rn "MixerKind" src/model/` shows dispatch only in
`ExecutionTopology::derive` and `derive_weight_requirements`;
`grep -rn "shared_kv_group_count" .` returns nothing; no test constructs a
`RuntimeTopology` field-by-field.

## Phase 1c — Convert the split-norm reads, delete the flag

**Goal:** close the second leak channel from section 1.3.

This is the piece the original plan deferred to "follow-on work". Section 1.3
establishes why deferring it is the wrong call: the flag is provably inert
today — no descriptor sets it, so all eleven backend branches are dead code and
the per-layer norms are never populated. Converting now is a pure refactor.
Converting later, after the first descriptor enables it, is a behavior change
requiring numerical validation on real checkpoints.

**Changes:**

- Replace each `shape.has_split_attention_norms` read with the per-layer spec
  already in scope at that site:
  - `src/backend/cpu/weights_loader.cpp:60`, `:66` →
    `layer_program.post_attention_norm.enabled()` /
    `.post_feed_forward_norm.enabled()`
  - `src/backend/cpu/model_forward_token.cpp:100`, `:119`, `:129` and
    `src/backend/cpu/model_forward_chunk.cpp:487`, `:509` →
    `semantics.post_attention_norm.enabled()` / `.post_feed_forward_norm.enabled()`
  - `src/backend/cpu/packed_execution.cpp:324`, `:487`,
    `src/backend/cuda/model/weight_setup.cpp:85`,
    `src/backend/cuda/model/expert_setup.cpp:27` → the corresponding per-layer
    `LayerSpec`/`CompiledLayer` norm
- Drop the redundant gate at `src/model/descriptor/weight_requirements.cpp:81`
  and `:187`; the per-layer `.enabled()` checks on the following lines already
  express it, and this is a prerequisite for Phase 2's signature.
- Delete `has_split_attention_norms` from `RuntimeTopology`
  (`include/celeg/model/resolved.hpp:86`) and its write at
  `src/model/descriptor/architecture.cpp:422`.
- Keep the `split_attention_norms` descriptor key. Its effect is now entirely
  the per-layer norms at `src/model/descriptor/graph_builder.cpp:71-75`, which
  is where it belonged all along. Add a test descriptor that sets it, since
  nothing currently exercises the feature.

**Exit criteria:** `grep -rn "has_split_attention_norms" .` returns nothing;
Phase 0 fingerprints unchanged; a descriptor with `split_attention_norms: true`
produces per-layer norms in the graph, matching weight requests, and executing
backend branches.

**Ordering:** independent of 1a/1b — it only needs the graph, which already
carries the per-layer truth. Land it first if convenient; it is small,
self-contained, and shrinks the surface 1b has to rename.

## Phase 2 — One weight-requirement derivation

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

**Prerequisites.** The proposed signature takes only
`(const ModelGraph&, const ITensorNamingPolicy&)`, but the descriptor
implementation being promoted reads `RuntimeTopology` in three ways. Two are
already resolved by earlier decisions:

- `topology.hidden` throughout, for tensor shapes — resolved by section 2.4:
  `hidden` lives on `ModelGraph`, so it arrives with the first argument.
- `topology.has_split_attention_norms` at `:81` and `:187` — resolved by
  Phase 1c, which deletes the gate.
- `topology.shared_expert_intermediate` at `:212-222`, for shared-expert
  tensors. **Still open.** Once Defect 3 is fixed,
  `MixtureOfExpertsSpec::has_shared_expert` and `::shared_intermediate_size`
  carry this per-layer, so read the graph instead. Doing so also fixes the
  current mismatch where shared-expert tensors are requested from a model-wide
  topology scalar rather than per-layer — a real bug for any model with shared
  experts on only some layers.

**Exit criteria:** one implementation of the `AttentionSpec -> TensorRequest`
mapping; `derive_weight_requirements` names no `RuntimeTopology` field; a
descriptorless resolution using query/key norm or MoE produces a complete weight
plan (it currently does not — `build_dense_weight_plan` omits both).

## Phase 3 — One resolution pipeline

**Goal:** eliminate the duplicated stage ordering (section 1.4, item 3).

**Changes:**

- Make descriptorless inference an `IArchitecture` implementation whose
  `dimensions`/`graph`/`weights` stages wrap the existing
  `CanonicalModelFacts` production.
- Delete `ResolutionAssembler::assemble` (`src/model/inference/synthesis.cpp:28-43`)
  and route it through `resolve_architecture_stages`.

**Exit criteria:** `ExecutionTopology::derive` is called from exactly one place
in the codebase — inside `resolve_architecture_stages`.

## Phase 4 — One error model

**Goal:** resolve the four-way exception inconsistency (section 1.4, item 6).

**Changes:**

- Extend `ResolutionFailureKind` to cover invariant violations currently thrown
  as `std::invalid_argument`/`std::runtime_error`.
- Convert `validate()` throughout `definition.cpp`, `resolved.cpp`,
  `program.cpp` to throw `celeg::ResolutionError` with an appropriate kind and
  evidence.

**Exit criteria:** callers can distinguish bad-checkpoint from
internal-invariant failures without matching on message text.

## Phase 5 — Residual cleanups

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
  hold through Phase 5, except for the three Phase 0 defect fixes. Every
  subsequent phase — including the Phase 1c backend conversion — is
  behavior-preserving by construction.
- Backend (`CPU`/`CUDA`) refactoring beyond what Phases 1b and 1c require:
  the `dims`/`exec` renames, the `hidden` move (section 2.4), and the split-norm
  read conversion. Backend *execution* logic, kernel selection, memory layout,
  and workspace strategy are all out of scope.

  This is narrower than the original plan's exemption, which deferred the
  split-norm conversion entirely. Section 1.3 explains the reversal: the
  conversion is free today and expensive later, and leaving it undone would
  mean shipping a refactor whose stated goal — no family-shaped knowledge in
  backend control flow — remains unmet at eleven sites.
- Descriptor schema changes. `double_wide_shared_suffix` keeps its current
  spelling; only its implementation moves.
- Adding new mixer kinds or architectures.

---

# 5. Risks

| Risk | Mitigation |
| --- | --- |
| **Phase 1 is much larger than the original plan assumed** — it now subsumes the front-end `LayerSpec` rewrite, because the graph builders consume Category A (section 1.1) | Split internally into 1a (front-ends emit `LayerSpec`) and 1b (struct split), landing as separate commits. 1a is the risk; it is not mechanical and it rewrites both front-ends — one of which has no tests (section 1.6). Budget accordingly and do not attempt 1b first |
| Phase 1b touches ~1,057 backend accesses across 36 files plus 172 in tests (section 1.5) | Land as a single mechanical commit; the private constructor makes missed cases build failures, not silent bugs. No signature churn — `RuntimeTopology` survives as the pair type, so the ~15 headers taking `const RuntimeTopology&` are untouched. Concentrate review on the six files holding half the Category A reads |
| Nine test fixtures hand-build `RuntimeTopology` and stop compiling at 1b | Known and enumerated (section 1.5), not a discovery. Rewrite each to build a `LayerSpec` vector and call `derive`. Do this *first* within 1b, so the fixtures are green before the backend rename lands on top |
| Phase 0 fingerprints may encode current defects | Pin fingerprints *after* all three defect fixes, not before. Note `fingerprint()` does not currently cover the affected fields at all — extend it or rely on the targeted assertions |
| Category B list may be incomplete | Phase 1 is compiler-verified: any field neither front-end-written nor `derive`-populated fails to compile into either type |
| Hidden readers of `moe_routing_*` outside `src/model/` | Confirmed to exist: `moe_router_config` in `include/celeg/detail/model/types.hpp`, reached from three CUDA call sites. The Phase 0 defect-2 fix restores correct values before any structural change, so these are fixed regardless |
| **The descriptor front-end has no tests and no in-tree instances** (section 1.6), yet Phases 1a and 2 rewrite it substantially | Phase 0 builds descriptor test infrastructure *before* any structural work touches `src/model/descriptor/`. Treat this as a hard gate: without it, the compiler is the only check on a rewrite of an entire front-end |
| Defect 3 shows a wrong *graph* can pass every existing validator | `MixtureOfExpertsSpec` has no `validate()`. Add one (`experts_per_token <= num_experts`, grouped-routing consistency, `has_shared_expert` implies `shared_intermediate_size > 0`) and call it from `ModelGraph::validate`. Without this, the split makes the graph the single owner of a value nothing checks |
| `ResolvedModel::validate` graph-vs-topology checks become tautological after the split, giving false assurance | Delete them explicitly in Phase 1 rather than leaving dead checks; migrate the intent into `ModelGraph::validate` |

---

# 6. Sequencing summary

```text
Phase 0  Pin behavior + fix 3 live defects      <-- ship this regardless
   |                                               (defects 2 and 3 are
   |                                                wrong output today)
   |
   +--> Phase 1c  Convert split-norm reads      <-- independent; land early
   |              delete the flag                   (free now, costly later)
   |
   v
Phase 1  1a front-ends emit LayerSpec           <-- the hard part
         1b split RuntimeTopology               <-- keystone; forces 2-3
   |        (test fixtures first, then rename)
   |
   +--> Phase 2  One weight-requirement derivation
   |
   +--> Phase 3  One resolution pipeline
   |
   v
Phase 4  One error model                        <-- independent
   |
   v
Phase 5  Residual cleanups                      <-- independent
```

Phase 1a and 1b are strictly ordered and should not be separated by other work.
Phase 1c is independent of both — it needs only the graph, which already carries
per-layer norm truth — so it can land any time after Phase 0. Landing it early
is preferable: it is small, self-contained, and removes eleven sites from the
surface 1b has to touch.

Phases 2 and 3 are mutually independent once Phase 1 lands and may proceed in
parallel. Phases 4 and 5 are independent of everything and may be pulled
forward if convenient — Phase 5 in particular is small, low-risk, and needs no
prerequisites.

**Natural stopping points.** This plan is deliberately built so it can be
abandoned partway without leaving the tree worse:

- **After Phase 0** — three correctness fixes shipped, no structural change.
  The highest value-per-risk point in the document.
- **After Phase 1c** — both agnosticism leak channels from section 1.3 closed,
  still no structural change. The stated agnosticism goal is met here, before
  the expensive work begins.
- **After Phase 1** — the keystone. Ownership is enforced by the type system;
  everything after this is deduplication.

Phases 2 through 5 are cleanup that pays down section 1.4. Valuable, but the
architectural argument is complete at the end of Phase 1.
