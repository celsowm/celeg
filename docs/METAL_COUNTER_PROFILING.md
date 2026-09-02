# Metal counter profiling on Apple silicon

Celeg's original per-dispatch counter experiment used `MTLCounterSamplingPointAtDispatchBoundary`. Apple silicon is a tile-based deferred architecture and commonly exposes counter sampling at stage boundaries instead of command boundaries, so per-dispatch sampling is not a portable profiling primitive for the Metal backend.

For performance work on Apple silicon:

- use `MTLCommandBuffer.GPUStartTime` / `GPUEndTime` for end-to-end command-buffer timing;
- use stage-boundary counter sampling only when a compute pass is intentionally segmented;
- use `scripts/metal_counter_capabilities.py` to inspect the sampling boundaries exposed by the current device;
- keep counter instrumentation diagnostic-only and out of normal benchmark runs.

The prefill optimization policy comparison therefore remains a throughput benchmark. Fine-grained profiling should use explicitly segmented compute passes rather than assuming dispatch-boundary counters are available.
