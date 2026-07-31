# How to Add a Model Architecture

1. Add checkpoint detection to an `IArchitectureProvider` implementation.
2. Parse and validate architecture metadata without adding architecture-only
   fields to `ModelDefinition`.
3. Implement an architecture-owned tensor naming policy using `TensorRole`.
4. Define typed architecture weights and a materialization plan.
5. Compile concrete CPU and CUDA programs that compose existing operators.
6. Keep scheduler, KV paging, sampling, and serving code generic.
7. Add deterministic reference, source-format, lifecycle, and public-header
   tests before enabling the provider in production selection.
8. Register the provider and document unsupported formats or backends
   explicitly.

Architecture-specific branches do not belong in generic operator dispatch.
