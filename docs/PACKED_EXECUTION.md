# Packed execution ownership

`PackedDecodeExecutor` is the public façade. Its reusable device/host state is
owned by `PackedDecodeExecutorImpl`, while request orchestration enters through
the distinct [`PackedDecodePipeline`](../include/celeg/backend/cuda/packed_pipelines.hpp)
and `PackedPrefillPipeline` collaborators. Both paths finish through the
explicit host commit helpers in
[`packed_commit.hpp`](../include/celeg/backend/cuda/packed_commit.hpp).

Decode and ragged prefill are separate operations and must not share host
state transitions. A failed launch or completion wait occurs before commit;
the commit helpers validate all buffers and row indices before mutating session
position, phase, sampled token, or metrics.

The packed executor is compiled for fixed topology maxima. It does not compile
plans or allocate steady-state buffers during a decode/prefill step. Use
`PackedWorkspaceRequirements::derive` and `PackedMetadataCache` when adding a
new packed operator or metadata field.
