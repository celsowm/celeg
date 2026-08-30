# MoE expert storage tiers

Compiled MoE semantics are resolved before storage selection. Sources receive
an `ExpertKey` and immutable payload requirements; they never receive an
architecture name. CPU pack backing owns its indexed reader and cache through
`CpuExpertBackingStore`, while CUDA residency receives a validated
`ExpertResidencyRequest` plus its preallocated `ExpertResidencyWorkspace` and
publishes device pointers transactionally. CUDA source selection is owned by
`CudaExpertSource`; the neutral `HostExpertCache` stores only byte payloads and
leases.

Celeg can keep the complete MoE expert set on storage while retaining only the
working set in RAM and/or VRAM.

## CUDA with SSD backing

```text
celeg-run \
  --repo LiquidAI/LFM2.5-8B-A1B \
  --prompt "Hello" \
  --expert-offload auto \
  --expert-backing disk \
  --expert-cache-policy lfu-lru \
  --expert-host-cache-mib 4096 \
  --expert-cache-per-layer 8
```

The tiers are:

1. Safetensors or an expert sidecar on SSD: authoritative copy of every expert.
2. Frequency-aware host staging cache: warm payloads shared by all layers;
   allocator choice (ordinary, pinned, or mapped) is injected by CUDA.
3. Per-layer VRAM cache: protected hot experts plus probationary/transient
   slots for new routing decisions.

The coordinator discovers a compact unique active set on the device, admits
cold payloads through the source/cache/I/O owners, and pins active slots only
after admission. Before a promotion batch, the cache reserves enough
transient slots for all unique cold experts. Disk backing rejects a batch whose
unique cold set is larger than the entire per-layer GPU cache; increase
`--expert-cache-per-layer` or reduce the prefill chunk in that case.

The fully resident discovery path uses a preallocated cache workspace and a
compact active-list transfer. It does not read storage, submit I/O work, copy
the full router selection, or call `cudaStreamSynchronize`. A not-yet-complete
device discovery is waited through its completion event; cold admission then
uses the bounded host lease/future workspace.

## CPU with SSD backing

```text
celeg-cpu-run \
  --repo LiquidAI/LFM2.5-8B-A1B \
  --prompt "Hello" \
  --cpu-expert-backing disk \
  --cpu-expert-cache-mib 2048 \
  --memory-report
```

For Safetensors checkpoints, the `.lfmpack` file is the persistent SSD backing.
Selected experts are loaded lazily into a shared frequency-aware RAM cache.
Sessions cloned from the same model share that cache, and concurrent misses for
the same expert coalesce into one read. Leases keep weights alive until both
expert GEMVs finish, even if another request evicts the cache entry.

The pack I/O is genuinely storage-backed:

- `CpuPackWriter` writes each matrix directly to the temporary pack instead of
  retaining all entries until `commit()`.
- `CpuPackReader` scans only entry headers and keeps a compact name-to-offset
  index. Matrix payloads remain on SSD until `read_q4_matrix()` is called.
- During the first pack build, disk-backed MoE experts are quantized, written,
  and released one at a time. The model never accumulates the full expert set
  in `weight_store`.
- When an existing pack is opened, expert entries are validated through the
  index without reading their payloads. Only routed experts enter the RAM cache.

The peak attributable to MoE expert preparation is therefore bounded by the
largest expert being quantized plus its temporary conversion buffers, rather
than the sum of every expert in the model. Steady-state expert residency remains
bounded by `--cpu-expert-cache-mib`.

Native GGUF matrices already reference the memory-mapped checkpoint, so the OS
page cache supplies SSD-backed demand paging and Celeg does not add a second
expert cache for that path.
