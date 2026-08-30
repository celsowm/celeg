# Study: FreeToken serving DeepSeek-V4-Flash (284B) on this machine

Source examined: `.externals/FreeToken` (FlashML-org/FreeToken, Apache-2.0),
specifically `python/freetoken/models/deepseek_v4/*`, `python/freetoken/moe/*`
(`host_banks.py`, `offload_cache.py`, `expert_banks.py`), and `docs/*.md`.
Numbers below are derived from the checkoint's own
`DeepseekV4Args` defaults (`models/deepseek_v4/args.py`) and from comments/
constants in the offload code itself, not from FreeToken's paper or
marketing copy.

## This machine

```
GPU:  NVIDIA GeForce RTX 5090, 32607 MiB VRAM, driver 580.173.02, CUDA 13.0
RAM:  31 GiB total, ~28 GiB available
CPU:  32 threads
```

This satisfies FreeToken's stated install requirement (`docs/install.md`:
"Linux x86_64, NVIDIA GPU, driver r580+ (CUDA 13)") exactly. The interesting
question is whether it satisfies the *memory* requirement for the specific
model named in the request — it does not, and the reason why is the actual
substance of this study.

## What DeepSeek-V4-Flash is, by the numbers

`DeepseekV4Args` (`args.py`) gives the real shape (this is the checkpoint's
own `inference/config.json` dialect, not a HF `AutoConfig` guess):

| field | value |
|---|---|
| `n_layers` | 43 |
| `dim` | 4096 |
| `moe_inter_dim` | 2048 |
| `n_routed_experts` | 256 |
| `n_shared_experts` | 1 |
| `n_activated_experts` | 6 |
| `head_dim` (MLA latent) | 512 |
| `q_lora_rank` / `o_lora_rank` | 1024 / 1024 |
| default `dtype` / `expert_dtype` | fp8 / fp4 |

Each expert is a SwiGLU MLP: `gate_up` (`2·moe_inter_dim × dim`) + `down`
(`dim × moe_inter_dim`) = `3 · dim · moe_inter_dim` = 25,165,824 params.

- Routed experts: 25.17M × 256 × 43 ≈ **277.1B params**
- Shared expert (1/layer): 25.17M × 43 ≈ **1.08B params**
- Everything else (MLA attention, embed/lm_head, Lightning Indexer, hyper-
  connections, norms): the remainder, ≈ **5.9B params**
- **Total ≈ 284B** — matches the model's own name and the README's "290B+".

Only 6 routed experts + the 1 shared expert (7 of 257) fire per token per
layer. Active compute per token is roughly 7 × 25.17M × 43 ≈ 7.6B (MoE) plus
the ~5.9B always-resident part ≈ **13–14B active params/token** — a dense
~13B-class amount of math riding on top of a 284B-parameter checkpoint. That
gap between "284B stored" and "~13B computed per token" is the entire premise
of FreeToken (`README.md`: "bandwidth-adaptive CPU–GPU co-execution") and of
the model being called "Flash".

## The actual constraint: host RAM, not the GPU

FreeToken's `expert_quant="ds_fp4"` bank layout (`offload_cache.py`,
`_BANK_SCHEMAS["ds_fp4"]`) packs each expert as e2m1 4-bit codes + one e8m0
per-32-element scale byte: **0.53125 bytes/param**. Per expert that's
`gate_up` 8 MiB + `down` 4 MiB + scales ≈ 0.75 MiB → **≈12.75 MiB/expert**.
Over 256 experts × 43 layers that's **≈137 GiB** — which is not my estimate,
it is the number FreeToken's own code names for this exact model. From
`moe/host_banks.py`:

> "registering a lazy mmap first faults+zero-fills every page (**~137 GiB ->
> ~47 s for DSV4**)"

That 137 GiB has to live somewhere before a single token is decoded, and
`HostBank` (`host_banks.py`) is explicit about where: an anonymous `mmap`
sized to the *whole* bank, filled from disk via chunked O_DIRECT reads, then
`cudaHostRegister`-pinned (`pin()`) — a page-locked allocation that cannot be
paged out. This buffer is held "for the process lifetime" (`_LIVE_BUFFERS`,
top-of-file docstring). Every one of FreeToken's four MoE backends reads from
this same host-resident bank set:

- `fused` — needs the same weights resident in VRAM instead (worse, never
  auto-selected for a model this size).
- `offload` — banks in host RAM, LRU expert slots on GPU, misses stream over
  PCIe (`docs/models.md`).
- `cpu` — same host RAM banks, but a `CpuMoeExecutor` computes misses on CPU
  instead of fetching them.
- `hybrid` — a bandwidth-calibrated split of the two above (`ft bench bw`,
  the `hybrid_fetch_fraction` in `OffloadMoeCache`).

None of them relax the RAM requirement — they only change *where the compute
for a miss happens*. There is no per-token disk-streaming path in this
codebase: disk is read exactly once, at load, into the pinned bank
(`read_file_into` in `host_banks.py`); after that the checkpoint file is
never touched again. So the real requirement to serve this model at all,
regardless of `--moe-backend`, is **≈137 GiB of free, pinnable host RAM**.

This machine has **31 GiB**. The `mmap` allocation for the DSV4 ds_fp4 bank
set would either fail outright or, once the loader starts writing real bytes
into it, die to the OOM killer well before the fill completes — the loader
never gets to `cudaHostRegister`, so `ft serve --model deepseek-ai/DeepSeek-
V4-Flash-0731` cannot reach a ready state here. `expert_banks.py` even has a
guard for constrained RAM (`_host_ram_fits_parallel`), but it only decides
*serial vs. parallel reader* for the initial fill — it has no fallback for
"the bank itself doesn't fit," because in this codebase that case isn't
survivable.

## What the GPU side looks like, for contrast

The GPU is comfortably *not* the bottleneck, which is worth stating plainly
since it's the resource this study started out asking about:

- Always-resident weights (attention, embed/lm_head, indexer, hyper-
  connections) at fp8 ≈ 5.9 GB.
- MLA gives a single shared latent KV head (`num_kv_heads=1`,
  `head_dim=512` in `attention.py`/`config.py`) instead of per-head KV, so
  the KV cache is cheap: 43 layers × 512 × 1 byte (fp8) ≈ 21.5 KB/token —
  128K tokens of context costs under 3 GB.
- That leaves the bulk of the 32 GB VRAM for the LRU expert slot cache
  (`--moe-cache-auto`), which at ≈12.75 MiB/slot could hold well over a
  thousand expert slots.

If the 137 GiB pinned host bank existed, this RTX 5090 would run the decode
math easily: RTX 5090 gives ~1,792 GB/s VRAM bandwidth and ~419/838 TFLOPS
FP8 dense/sparse (FP4 roughly double). A purely illustrative, non-measured
bound: a fully-cold decode step still has to cross PCIe for its 6 routed
misses × 43 layers × 12.75 MiB ≈ 3.2 GB (the always-active shared expert
needn't be fetched — it can just stay resident). At a realistic PCIe 5.0 x16
sustained rate (~60–63 GB/s) that's ~51 ms/token, i.e. an LRU-cold *floor* of
roughly ~20 tok/s for single-stream decode, with warm cache hits (this is
exactly what the LRU + `q*` hybrid policy and `ft bench bw` calibration in
`offload_cache.py`/`benchbw.py` exist to maximize) pushing well above that.
None of this is reachable here — it's included only to show that *if* the
RAM existed, the GPU would not be the limiting factor.

## Why this specific box fails: RAM, not compute, not driver

Summary of the gate this model has to pass, in order, on this machine:

| Requirement | Need | Have | OK? |
|---|---|---|---|
| Driver / CUDA | r580+ / CUDA 13 | 580.173.02 / CUDA 13.0 | yes |
| Pinned host RAM for ds_fp4 expert banks | ≈137 GiB | 31 GiB | **no** |
| VRAM for resident weights + KV + expert cache | a few GB + cache | 32 GiB | yes (moot) |

The failure mode is architectural, not tunable: `--moe-backend cpu` doesn't
need the GPU cache, but it still needs the full 137 GiB pinned in RAM before
the CPU executor can read a single expert row from it. There is no flag that
trades "load only part of the model" for "serve at reduced batch/quality" in
this codebase — `HostBank` allocates the whole per-format bank up front.

## Relevance to celeg's own MoE offload design

`docs/EXPERT_STORAGE.md` describes a materially different tiering strategy
already in this repo: `--expert-backing disk` keeps the *authoritative* copy
of every expert on SSD permanently, with only a frequency-aware **working
set** staged into RAM (`--expert-host-cache-mib`) and a smaller working set
in VRAM (`--expert-cache-per-layer`) — the model is never fully materialized
in host RAM. That's the one design choice that would change the outcome of
this whole study: a 284B/6-of-256 MoE model like DeepSeek-V4-Flash is exactly
the case celeg's disk-backed tier was built for, since routing sparsity (6 of
256 experts/token/layer) means the *working set* per step is a small fraction
of the 137 GiB total, even though the full checkpoint never fits in this
box's 31 GiB of RAM. FreeToken's pin-the-whole-quantized-model-in-RAM
approach trades that possibility for simplicity and for the raw copy
bandwidth `cudaHostRegister`+pinned-DMA gives it once the model *does* fit —
a reasonable trade for its stated target (a "gaming PC" with 64–128+ GB of
system RAM), just not one this machine's 31 GiB can pay into.
