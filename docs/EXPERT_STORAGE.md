# MoE expert storage tiers

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
2. Frequency-aware pinned-RAM cache: warm experts shared by all layers.
3. Per-layer VRAM cache: protected hot experts plus probationary/transient
   slots for new routing decisions.

Slots used by the active FFN batch are pinned until that FFN completes. Before a
promotion batch, the cache reserves enough transient slots for all unique cold
experts. Disk backing rejects a batch whose unique cold set is larger than the
entire per-layer GPU cache; increase `--expert-cache-per-layer` or reduce the
prefill chunk in that case.

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

Native GGUF matrices already reference the memory-mapped checkpoint, so the OS
page cache supplies SSD-backed demand paging and Celeg does not add a second
expert cache for that path.

### Current first-build limitation

When a Safetensors `.lfmpack` does not exist yet, Celeg still quantizes and
writes all experts while constructing the pack before releasing the eager
expert vectors. Steady-state residency is bounded by `--cpu-expert-cache-mib`,
but the first pack creation can temporarily require the original eager-loading
peak. A later loader refactor can stream each expert directly into the pack to
remove that one-time peak.
