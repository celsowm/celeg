# How to Add a Model Architecture

1. Add a concrete `IArchitecture` under `src/models/<architecture>` and expose
   its factory from the composition root.
2. Parse and validate `CheckpointMetadata` without adding architecture-only
   fields to `ModelDefinition`.
3. Build an architecture-owned `ModelGraph`, `WeightPlan`, and semantic tensor
   bindings using `TensorRole`.
4. Compile concrete CPU and CUDA programs from the resolved graph.
6. Keep scheduler, KV paging, sampling, and serving code generic.
7. Add deterministic reference, source-format, lifecycle, and public-header
   tests before enabling the provider in production selection.
8. Register the architecture in the immutable catalog and document unsupported formats or backends
   explicitly.

Architecture-specific branches do not belong in generic operator dispatch.
