# Metal MRoPE Geometry Plan

## Goal

Bring the Metal RoPE/MRoPE implementation onto the same semantic geometry contract already shared by CPU and CUDA, while preserving current numerical behavior and keeping backend-specific execution details separate.

The concrete capability expansion in this plan is support for both MRoPE axis layouts represented by `MultiAxisRopeSpec::interleaved`:

- `interleaved = true`: axis selection cycles `0, 1, 2, 0, 1, 2, ...`;
- `interleaved = false`: axis selection follows the three declared section lengths.

This is intentionally narrower than a general Metal RoPE expansion. Full-width, unscaled RoPE and the current supported pairing/scaling constraints remain in force unless explicitly relaxed in a later plan.

## Current state

The canonical C++ geometry contract now lives in:

- `src/celeg/model/rope_geometry.hpp`

It defines:

- rotary pair count;
- SplitHalf vs AdjacentPairs component mapping;
- interleaved vs sectioned MRoPE axis selection.

CPU and CUDA generic RoPE paths use that contract.

Metal currently advertises `multi_axis_rope = true`, but `validate_metal_attention_capabilities()` accepts only:

- three axes;
- `interleaved = true`;
- SplitHalf pairing;
- full-width RoPE;
- no RoPE scaling;
- theta `10000`.

The Metal MRoPE shaders currently receive section lengths but choose the axis with `pair % 3`, so the section metadata is used only for shape validation, not axis selection.

Affected Metal shader families include:

- `src/backend/metal/kernels/inference/position.metal`
- `src/backend/metal/kernels/inference/qk_position.metal`
- `src/backend/metal/kernels/inference/mrope_batch.metal`

The runtime binding point is primarily:

- `src/backend/metal/model/runtime/attention.mm`

## Design rule

Share model semantics, not execution mechanisms.

The C++ header remains the canonical host/CUDA semantic definition. Metal should mirror the same tiny pure functions in MSL because MSL cannot consume the C++ header directly.

Do not centralize:

- frequency scaling;
- `pow`, `sin`, `cos` evaluation;
- RMSNorm arithmetic;
- BF16/float conversion policy;
- threadgroup geometry;
- fused KV publication;
- specialized fast kernels.

Those remain backend execution details.

## Stage 0 — Baseline and invariants

Before changing Metal implementation:

1. Record the current `master` commit.
2. Run or inspect the existing RoPE/MRoPE-relevant Metal tests.
3. Preserve the current supported interleaved behavior as the baseline oracle.
4. Record any existing numerical tolerances separately from semantic geometry correctness.

Required invariant: enabling sectioned MRoPE must not change outputs for existing interleaved models.

## Stage 1 — Add a Metal geometry semantics mirror

Create a small MSL semantic helper, preferably:

- `src/backend/metal/kernels/inference/rope_geometry.metal`

It should mirror the semantic operations in `src/celeg/model/rope_geometry.hpp` without pulling in frequency/scaling logic.

Suggested functions:

```metal
struct CelegRopePairComponents {
    uint first;
    uint second;
};

inline CelegRopePairComponents celeg_rope_pair_components(
    uint pair, uint pair_count, uint pairing_mode);

inline uint celeg_mrope_axis_for_pair(
    uint pair, uint section0, uint section1, bool interleaved);
```

Pairing mode should be a narrow Metal ABI value, not a duplicated model enum definition inside shader code.

Add the new source to the Metal shader source list and include it before the position-family shaders in `inference.metal`.

Acceptance criteria:

- no production kernel behavior changes yet;
- MSL helper compiles as part of the generated inference shader;
- helper behavior can be probed independently.

## Stage 2 — Add a Metal geometry probe and parity test

Add a dedicated probe kernel for semantic geometry only. Do not make the production kernels the first place where the new helper is tested.

Add:

- `tests/metal/metal_rope_geometry_semantics_test.mm`

Compare Metal results directly against `src/celeg/model/rope_geometry.hpp` for a table of cases.

Minimum coverage:

### Pair component mapping

- SplitHalf with multiple pair counts;
- AdjacentPairs with multiple pair counts;
- first pair;
- middle pair;
- last valid pair.

### MRoPE axis mapping

Interleaved:

- several consecutive pair indices crossing all three axes.

Sectioned:

- boundary before section 0 ends;
- first pair of section 1;
- last pair of section 1;
- first pair of section 2;
- uneven section sizes such as `{2, 3, 1}`.

Acceptance criteria:

- C++ and MSL geometry results match exactly as integers;
- no floating-point tolerance is involved in this test.

## Stage 3 — Extend the Metal runtime ABI with `interleaved`

Thread the model semantic flag through every MRoPE dispatch that currently passes positions and sections.

Primary runtime file:

- `src/backend/metal/model/runtime/attention.mm`

Represent the flag as a fixed-size Metal-friendly scalar, preferably `uint32_t` on the host and `constant uint&` in MSL.

Do not pass a C++ `bool` directly through `setBytes` because host `bool` size/layout should not become part of the shader ABI.

Affected dispatch families should include all paths that can execute MRoPE:

- non-fused single-token position + KV store;
- fused Q/K norm + MRoPE + KV store;
- batched MRoPE position path.

Acceptance criteria:

- all MRoPE dispatches carry the same explicit layout flag;
- existing interleaved models send `1` and retain the current behavior;
- sectioned models can send `0` without selecting a different pipeline family.

## Stage 4 — Route production MRoPE kernels through the semantic helper

Replace direct `pair % 3` axis selection in the generic MRoPE paths with `celeg_mrope_axis_for_pair()`.

Target files:

- `src/backend/metal/kernels/inference/position.metal`
- `src/backend/metal/kernels/inference/qk_position.metal`
- `src/backend/metal/kernels/inference/mrope_batch.metal`

The only intended semantic change is axis selection when `interleaved == false`.

Preserve exactly:

- current frequency expression;
- multiplication association around normalization and output scaling;
- trigonometric call placement;
- pair traversal order;
- cache store order;
- query vs key scaling behavior.

Do not opportunistically combine the three MRoPE kernels in this stage. First make them semantically correct through one shared helper; structural shader consolidation can follow after parity is proven.

Acceptance criteria:

- interleaved path remains equivalent to old `pair % 3` behavior;
- sectioned path uses declared section boundaries;
- no unrelated shader diff.

## Stage 5 — Expand the declared Metal capability

Only after the runtime and kernels support the flag, update:

- `src/celeg/backend/metal/attention_capabilities.hpp`

Remove the `!multi->interleaved` rejection while retaining the other current constraints initially:

- `axes == 3`;
- SplitHalf pairing;
- full-width rotary dimension;
- no RoPE scaling;
- theta `10000`;
- section sum equals pair count.

This ordering is important: do not advertise a capability before the runtime can execute it.

Add/extend capability tests so that Metal:

- accepts a valid three-axis sectioned MRoPE spec;
- continues accepting a valid interleaved spec;
- rejects invalid section sums;
- rejects unsupported axis counts;
- rejects AdjacentPairs for MRoPE;
- continues rejecting partial-width/scaled MRoPE until separately implemented.

## Stage 6 — Functional Metal/CPU parity tests

Add end-to-end kernel-level parity tests rather than relying only on the integer geometry probe.

Use CPU MRoPE as the semantic oracle because CPU now consumes the canonical geometry helper.

Test both layouts:

1. interleaved;
2. sectioned with uneven sections.

Test at least these execution shapes:

- position-only Q/K transform;
- Q/K norm + MRoPE;
- batch MRoPE;
- fused KV publication where applicable.

For KV-publication tests verify separately:

- transformed query;
- transformed key;
- key cache contents;
- untouched value semantics / value cache contents.

Use tolerances appropriate to the existing CPU-vs-Metal floating-point policy. Geometry itself must already have exact integer parity from Stage 2.

## Stage 7 — Preserve the optimized SplitHalf paths

Do not route specialized ordinary SplitHalf RoPE kernels through a more generic MRoPE abstraction merely for DRY.

In particular, preserve the optimized full-width SplitHalf kernels used by the normal non-MRoPE path unless measurements show no regression.

The shared geometry helper should be essentially zero-cost when used in generic kernels:

- inlined;
- branch resolved from a constant kernel argument when possible;
- no table allocation;
- no additional device memory access.

After functional correctness, inspect generated Metal compiler behavior or benchmark if the branch remains material in hot loops.

If needed, split at dispatch time into interleaved/sectioned specialized pipeline variants only after measurement demonstrates a reason. Do not pre-specialize speculatively.

## Stage 8 — Consolidate duplicated Metal MRoPE helpers

Once parity tests are green, audit the three production shader files for remaining duplicated MRoPE mechanics.

Candidates for consolidation:

- axis resolution;
- pair-index derivation;
- pure rotate-two-values primitive, only if expression ordering can be preserved.

Do not centralize a rotate primitive if doing so changes floating-point contraction/FMA behavior or multiplication association.

The desired final layering is:

```text
Model semantics
    src/celeg/model/rope_geometry.hpp
             |
      CPU / CUDA direct use
             |
       Metal MSL mirror
             |
   Metal execution-specific kernels
```

## Stage 9 — Regression gates

Before declaring the work complete:

1. Run the host `rope_geometry_semantics_test`.
2. Run the new Metal geometry semantic probe test.
3. Run existing Metal semantic tests.
4. Run Metal QK/KV-store tests.
5. Run new interleaved and sectioned MRoPE functional parity tests.
6. Build the Metal inference shader from the normal CMake-generated source path, not only an isolated test string.
7. Run relevant model capability tests.
8. Benchmark ordinary SplitHalf RoPE and MRoPE paths before/after if benchmark infrastructure is available.

No performance claim should be made from this refactor without benchmark evidence.

## Suggested commit slices

Keep commits reviewable and independently understandable:

1. `refactor: mirror RoPE geometry semantics in Metal`
2. `test: compare Metal RoPE geometry with host semantics`
3. `refactor: pass MRoPE layout through Metal runtime`
4. `refactor: use shared MRoPE axis semantics in Metal kernels`
5. `feat: support sectioned MRoPE on Metal`
6. `test: add Metal sectioned MRoPE parity coverage`
7. optional: `refactor: consolidate Metal MRoPE kernel helpers`

Do not combine capability advertisement with the low-level ABI change in the same commit; keeping them separate makes accidental partial support easier to detect.

## Completion criteria

The plan is complete when all of the following are true:

- CPU, CUDA and Metal agree on RoPE pair geometry semantics;
- CPU, CUDA and Metal agree on both interleaved and sectioned three-axis MRoPE axis selection;
- Metal capability validation accepts valid sectioned MRoPE;
- existing interleaved Metal MRoPE output does not regress;
- all Metal MRoPE dispatch paths explicitly carry layout semantics;
- no RoPE scaling or numerical behavior is changed as a side effect;
- specialized ordinary SplitHalf fast paths remain intact unless separately justified by measurements;
- tests cover semantic helpers and real production kernels;
- benchmark claims, if any, are supported by measured data.

## Follow-up work intentionally excluded

These should be separate changes after this plan is complete:

- partial rotary fraction on Metal;
- Linear/DynamicNTK/YaRN/LongRoPE/Llama3/Proportional scaling on Metal;
- MRoPE with AdjacentPairs;
- more than three MRoPE axes;
- changing theta constraints;
- precomputed RoPE table redesign;
- shader specialization solely for performance;
- broader consolidation of ordinary RoPE and MRoPE execution kernels.
