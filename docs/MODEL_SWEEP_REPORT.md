# CPU/CUDA model sweep report

Status: incomplete for the two large checkpoints that could not be acquired.

## Environment

- Platform: Windows 11
- CPU path: AVX2, `out/windows-cpu-release/celeg-cpu-run.exe`
- CUDA path: RTX 3060, 12 GB, existing `out/windows-cuda-relwithdebinfo/celeg-run.exe`
- Prompt: `What is the capital of France?`
- Generation: greedy, temperature 0, top-k 1, maximum 8 new tokens

## Build results

| Backend | Result | Evidence |
|---|---|---|
| CPU Release | PASS | `python scripts/dev.py build --backend cpu --build-type Release --jobs 12` |
| CUDA Release | FAIL | `hf_http.cpp` was fixed for an unmatched brace; the build then stopped because `celeg/backend/cuda/model/detail/compiled_model.hpp` and `celeg/backend/cuda/model/detail/linear_weights.hpp` are absent from `HEAD` |

CPU CTest: 79/80 passed. The only failure was `architecture_boundary_test`, which reports the same nine obsolete `celeg/backend/cuda/model/detail/*` paths.

Existing CUDA build CTest: 86/88 passed. The failures were `architecture_boundary_test` for the obsolete header paths and `architecture_resolution_test` with exit code `0xc0000409`.

The CUDA runtime rows below therefore use the pre-existing CUDA executable and do not represent a current-source CUDA build.

## Runtime results

`OK` means the process exited successfully. `Correct` means the output visibly contained the expected answer; raw prompt formatting and multiple-choice output are retained as observed.

| Model/artifact | CPU | CUDA |
|---|---|---|
| LFM2.5-230M Safetensors | OK, Correct | OK, Garbled (`?`) |
| LFM2.5-350M Safetensors | OK, Correct | OK, Garbled (`??`) |
| MiniCPM5-1B Safetensors | OK, Correct | OK, Garbled (`#`) |
| LFM2.5-230M Q4_K_M | OK, Correct | OK, Empty |
| LFM2.5-230M Q6_K | OK, Correct | OK, Empty |
| LFM2.5-350M Q4_0 | OK, Garbled/empty | FAIL, `execution plan requires INT8 weights` |
| LFM2.5-350M Q4_K_M | OK, Correct | OK, Empty |
| LFM2.5-350M Q5_K_M | OK, Correct | FAIL, unsupported concat |
| LFM2.5-350M Q8_0 | OK, Correct | FAIL, unsupported concat |
| Qwen3.5-0.8B Q4_K_M | OK, Garbled | OK, Garbled |
| MiniCPM5-1B Q4_K_M | OK, Correct | OK, Garbled |
| SmolLM3-3B Q4_K_M | OK, Correct | OK, Empty |
| Nanbeige 3B Q4_K_M | OK, Wrong/non-English | FAIL, missing `tokenizer.ggml.merges` |
| Nemotron 4B Q4_K_M | OK, Correct/partial | OK, Wrong/partial |
| Ling-3.0-tiny-int4 Safetensors | OK, completed at 6.91 decode tok/s | Abnormal termination/no output; consistent with the known 12 GB CUDA limitation |

## Not completed

- `flwrlabs/Lizzy-7B-GGUF` Q4_K_M: Hub download stalled after metadata; only about 5 KB is cached.
- `flwrlabs/Lizzy-7B` Safetensors: not downloaded or tested.
- `LiquidAI/LFM2.5-8B-A1B` Safetensors: not downloaded or tested; expected to exceed 12 GB VRAM.

## Change made to build

Removed the unmatched closing brace after the platform conditional in `src/checkpoint/hf_http.cpp`. This fixed the first CPU and CUDA compiler error. The remaining CUDA header omissions are independent of that fix.
