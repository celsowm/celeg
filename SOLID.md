# SOLID refactoring — v0.0.15

v0.0.15 addresses the four architectural debts explicitly left by v0.0.14.
The C ABI and inference behavior are preserved.

## 1. `LfmModel` facade

The public model header now contains only runtime value types, standard-library
ownership and the stable facade. CUDA streams, cuBLAS handles, buffers,
`safetensors` storage and model topology are private to `LfmModel::Impl`.

Three narrow views support interface segregation:

```text
LfmInferenceSession  reset/prefill/decode/session state
LfmDiagnostics       logits, metrics, memory and benchmark data
LfmPersistence       save/load and prefix snapshots
```

Compatibility forwarding methods remain on `LfmModel`, so existing callers and
the C adapters do not break. They contain no inference logic.

## 2. Concurrent serving responsibilities

`ConcurrentEngine` is now a PIMPL facade. Its implementation orchestrates three
host-testable components:

```text
RequestRegistry  request ownership, IDs, outputs and admission queue
BatchPlanner     admission selection and priority ordering
EngineWorker     background thread, wait/notify and stop lifecycle
```

The engine implementation still coordinates CUDA execution, KV and prefix
services, but no longer owns the detailed policies or worker mechanics inside
one public/concrete class.

## 3. Typed model topology

The former structure containing `bool attention` plus nullable attention and
convolution fields was removed. The topology is now:

```cpp
using Layer = std::variant<AttentionLayer, ConvolutionLayer>;
```

Shared norms/MLP weights live in `LayerCommon`. Attention layers alone own QKV,
Q/K norms and KV caches. Convolution layers alone own convolution projections,
weights and recurrent state. This makes invalid mixed states unrepresentable.

## 4. Packed execution stages

`PackedDecodeExecutorImpl::decode()` and `prefill_step()` now orchestrate
focused stages instead of embedding an entire model pass:

```text
validate batch
copy/prepare metadata
sample or mark explicit prompt tokens
embedding
QKV projection and RoPE
paged or local attention
ShortConv operator
MLP
final logits/state commit
```

The transformer loop delegates to dedicated attention, convolution and MLP
methods. Automated architecture checks cap the size of the main orchestration
and largest packed stages.

## SOLID impact

### Single Responsibility

Public facades are separate from implementations; request storage, planning and
worker lifecycle are independent components; model layer variants own only
their applicable state; packed stages have explicit responsibilities.

### Open/Closed

New concurrent planning policies can be developed against registry snapshots
without altering thread ownership. New layer-specific behavior is dispatched
through the typed variant rather than adding nullable fields to a universal
record.

### Liskov and state contracts

Typed layers eliminate behavioral assumptions based on a boolean discriminator.
`SessionPhase` remains the single state contract for a model session.

### Interface Segregation

Callers may use focused model views. Public model and engine headers no longer
force consumers to include CUDA or internal concurrency containers.

### Dependency Inversion

Prefix cache still depends on `IKvPageAllocator`; serving policy components are
host-only and do not depend on CUDA. The public facades depend on private
implementations rather than exposing concrete resources.

## Remaining debt

- `LfmModel::Impl` remains large because it contains the complete standalone
  CUDA runtime; future work may extract weight loading, session serialization
  and matmul planning into separate implementation services.
- `ConcurrentEngine::Impl` remains the orchestration root for admission,
  prefill, decode, KV and metrics; extracting `PrefillCoordinator` and
  `DecodeCoordinator` would further reduce its size.
- the packed paged-attention stage remains hardware-specific and sizeable,
  although it is isolated from decode/prefill orchestration.
- the C ABI remains one umbrella header to avoid compatibility churn.
