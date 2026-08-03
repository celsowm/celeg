# Celeg

Celeg is a native C++20 inference runtime for LFM2, LFM2.5, Granite, MiniCPM5,
and SmolLM3 language models. It provides independent CPU and NVIDIA CUDA
backends, direct checkpoint loading, quantized execution, an OpenAI-compatible
server, and a public C API.

Celeg does not bundle model weights. Supply a Hugging Face repository, a local
Safetensors checkpoint directory, or a local GGUF file.

## Support matrix

| Architecture | Safetensors | GGUF | CPU | CUDA |
| --- | :---: | :---: | :---: | :---: |
| LFM2/LFM2.5 dense | Yes | Yes | Yes | Yes |
| LFM2/LFM2.5 MoE | Yes | Yes | Yes | Yes |
| Granite dense | Yes | No | Yes | Yes |
| MiniCPM5-1B | Yes | Yes | Yes | Yes |
| SmolLM3-3B | Yes | Yes | Yes | Yes |

MiniCPM5-1B uses the standard Llama tensor layout with GQA (16 query heads,
2 KV heads), 131072-token context metadata, and both EOS markers from the
official checkpoint. Its GGUF variants are selected with the `--quant` tag.

## Supported LFM checkpoints

| Variant | Hugging Face repository |
| --- | --- |
| LFM2.5 230M | `LiquidAI/LFM2.5-230M` |
| LFM2.5 1.2B Instruct | `LiquidAI/LFM2.5-1.2B-Instruct` |
| LFM2.5 1.2B Thinking | `LiquidAI/LFM2.5-1.2B-Thinking` |
| LFM2.5 8B-A1B | `LiquidAI/LFM2.5-8B-A1B` |
| LFM2 8B-A1B | `LiquidAI/LFM2-8B-A1B` |

Granite checkpoints are selected from their `config.json`. The runtime
expects the architecture metadata to identify Granite with
`model_type: "granite"`.

## MiniCPM5 checkpoints

The BF16 repository is `openbmb/MiniCPM5-1B`; the GGUF repository is
`openbmb/MiniCPM5-1B-GGUF`, with `Q4_K_M`, `Q8_0`, and `F16` files. After the
repositories are present in the Hugging Face cache, run either format with:

```text
celeg-run --repo openbmb/MiniCPM5-1B --prompt "Hello" --max-new-tokens 32
celeg-run --repo openbmb/MiniCPM5-1B-GGUF:Q4_K_M --prompt "Hello" --max-new-tokens 32
```

For the OpenAI-compatible server, `--repo openbmb/MiniCPM5-1B` selects
Safetensors automatically, while `--repo openbmb/MiniCPM5-1B-GGUF` selects the
GGUF repository and its `--quant` tag. The `minicpm5-instruct` profile renders
the official `<|im_start|>` template, tool definitions, `<function>` calls,
tool responses, and multiple EOS markers.

## SmolLM3 checkpoints

The BF16 checkpoint is `HuggingFaceTB/SmolLM3-3B`; GGUF files are available in
`ggml-org/SmolLM3-3B-GGUF` as `Q4_K_M`, `Q8_0`, and `F16`. SmolLM3 uses a
hybrid NoPE/RoPE attention schedule and defaults to extended thinking. Use
`/no_think` or `/think` in the system message to select the reasoning mode:

```text
celeg-run --repo HuggingFaceTB/SmolLM3-3B --prompt "Explain gravity" --max-new-tokens 32
celeg-run --repo ggml-org/SmolLM3-3B-GGUF:Q4_K_M --prompt "Explain gravity" --max-new-tokens 32
```

The `smollm3-instruct` profile supports the official `<|im_start|>` format,
NoPE-aware execution, XML tool calls, `/system_override`, and `/think` /
`/no_think` system flags.

## Requirements

For CPU builds:

- CMake 3.24 or newer.
- A C++20 compiler.
- Python 3 for the developer helper.

For CUDA builds, add a compatible NVIDIA CUDA Toolkit and GPU. CUDA is
optional; the CPU backend can be built without it.

The repository is developed and tested on Windows and Linux. On Windows,
executables have an `.exe` suffix.

## Build and verify

The portable developer entrypoint discovers the available compiler, CUDA
toolkit, GPU architecture, and runtime dependencies:

```text
python scripts/dev.py doctor
python scripts/dev.py verify --backend cpu
python scripts/dev.py verify --backend cuda
```

Use `--backend auto` to select CUDA when available and CPU otherwise. Other
useful options are `--build-type Release`, `--arch 86`, `--jobs 8`, and
`--build-dir PATH`.

The helper supports these commands:

```text
python scripts/dev.py doctor
python scripts/dev.py build --backend cpu
python scripts/dev.py test --backend cpu
python scripts/dev.py smoke --backend cuda
python scripts/dev.py verify --backend cpu
```

`verify` performs a fresh configure, build, and test run. Build directories are
written under `out/` by default.

For direct CMake builds, use the checked-in presets:

```text
cmake --preset cpu-relwithdebinfo
cmake --build --preset cpu-relwithdebinfo
ctest --preset cpu-relwithdebinfo
```

CUDA presets are named `cuda-release` and `cuda-relwithdebinfo`.

## Obtain a model

### Hugging Face cache

The `--repo` option resolves a repository from the local Hugging Face cache.
This is the preferred workflow when a checkpoint has already been downloaded:

```text
celeg-cpu-run --repo LiquidAI/LFM2.5-230M \
  --prompt "Explain CUDA in one sentence." \
  --max-new-tokens 32
```

The CUDA runner uses the same repository IDs:

```text
celeg-run --repo LiquidAI/LFM2.5-230M \
  --prompt "Explain CUDA in one sentence." \
  --max-new-tokens 32
```

Use `celeg-download` to populate the Hugging Face cache from a repository:

```text
celeg-download LiquidAI/LFM2.5-230M
```

The project script provides convenient LFM2.5 presets and downloads into a
local directory:

```text
./scripts/download_model.sh 230m
./scripts/download_model.sh 1.2b-instruct
./scripts/download_model.sh 1.2b-thinking
./scripts/download_model.sh 8b-a1b
```

### Local Safetensors

For a local Safetensors checkpoint, pass the directory containing
`config.json`, tokenizer files, and either `model.safetensors` or a
`model.safetensors.index.json` plus its shards:

```text
celeg-cpu-run --model path/to/checkpoint-directory \
  --prompt "Write a short welcome message." \
  --max-new-tokens 32
```

This also works for Granite checkpoints whose `config.json` declares
`model_type: "granite"`.

### Local GGUF

GGUF checkpoints are concrete files, not checkpoint directories. Pass the
`.gguf` file directly to the runner:

```text
celeg-cpu-run --model path/to/model.gguf \
  --prompt "Write a short welcome message." \
  --max-new-tokens 32
```

CUDA GGUF inference can select native GGUF weight handling explicitly:

```text
celeg-run --model path/to/model.gguf \
  --weight-mode native \
  --prompt "Write a short welcome message." \
  --max-new-tokens 32
```

Celeg reads GGUF model metadata and tokenizer data directly. The file must
describe an LFM2 or LFM2-MoE checkpoint using the supported GGUF metadata
namespaces; Granite GGUF is not supported at this time.

## Runner options

Both runners support model selection, prompts, context length, maximum output
tokens, sampling controls, and memory reporting. The CUDA runner additionally
supports CUDA Graphs, cuBLAS/cuBLASLt selection, quantized weight modes,
attention modes, paged KV cache controls, session persistence, and LFM2-MoE
expert offload.

Inspect the complete options for the binary produced by your build:

```text
celeg-cpu-run --help
celeg-run --help
```

Typical CPU controls include `--cpu-isa`, `--threads`, `--cpu-kv-cache`,
`--cpu-prefill-chunk`, and `--cpu-affinity`. Typical CUDA controls include
`--weight-mode`, `--kv-cache`, `--attention-mode`, `--no-cuda-graph`, and
`--expert-offload`.

## OpenAI-compatible server

`celeg-serve` exposes an OpenAI-compatible HTTP API. It uses a local model path
and can select the CPU or CUDA backend:

```text
celeg-serve \
  --model path/to/checkpoint-directory \
  --backend cpu \
  --port 8080 \
  --served-model-name celeg
```

Start the CUDA version by changing `--backend cpu` to `--backend cuda`. The
server provides health, model, tokenizer, and chat-completion routes. See the
generated API documentation served by the process for the exact HTTP schema.

## C API

The public C ABI is declared in [`include/celeg/api.h`](include/celeg/api.h).
It uses the `celeg_*` function and type namespace and is suitable for C, Rust,
Zig, Node native addons, and other FFI consumers.

The API supports:

- CPU model creation and direct prefill/decode.
- Request-oriented engine submission, polling, stepping, and cancellation.
- Tokenizer encoding and decoding with caller-provided buffers.
- Backend capability and diagnostic queries.

See [`API.md`](API.md) for initialization rules, handle ownership, status
codes, and short C examples.

## Features

- LFM2/LFM2.5 dense and MoE model variants.
- Granite dense architecture support through the same backend-neutral model
  contracts.
- Safetensors, sharded Safetensors, and LFM2/LFM2-MoE GGUF loading.
- CPU scalar, AVX2, and VNNI execution paths where available.
- CUDA quantized weights, paged KV cache, prefix reuse, packed prefill and
  decode, CUDA Graphs, and cuBLAS/cuBLASLt.
- LFM2-MoE expert residency, host offload, and cache policies.
- Concurrent scheduling, session persistence, diagnostics, and benchmarks.

## Benchmarks and architecture

- [`BENCHMARK.md`](BENCHMARK.md) contains benchmark commands and reproducible
  comparison procedures.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) describes model providers,
  checkpoint contracts, runtime scheduling, and backend boundaries.
- [`docs/ARCHITECTURE_RULES.md`](docs/ARCHITECTURE_RULES.md) records the
  architectural constraints used for changes.
- [`scripts/gguf_census.py`](scripts/gguf_census.py) inventories GGUF tensor
  types and estimated traffic.
- [`scripts/profile_decode.py`](scripts/profile_decode.py) profiles decode
  phases and can compare CUDA configurations.

## Troubleshooting

### CUDA is unavailable

Run `python scripts/dev.py doctor --backend cuda --json` to inspect the CUDA
toolkit, compiler, GPU architecture, and runtime libraries. If CUDA is not
available, build and run the CPU backend explicitly.

### A repository cannot be resolved

`--repo` requires the requested snapshot to exist in the local Hugging Face
cache. Run `celeg-download REPO_ID` or `scripts/download_model.sh VARIANT`,
then retry. Use `--model` when the checkpoint is stored at a known local path.

### A Safetensors checkpoint fails to load

Confirm that the directory contains `config.json`, tokenizer files, and either
an unsharded `model.safetensors` or every shard referenced by
`model.safetensors.index.json`.

### A GGUF checkpoint fails to load

Confirm that the path points to a `.gguf` file and that its metadata describes
an LFM2 or LFM2-MoE model. Granite GGUF files are not supported by the current
loader.

## License

See [`LICENSE`](LICENSE).
