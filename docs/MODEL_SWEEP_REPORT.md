# CPU/CUDA model sweep report

Status: complete for the 18 acquired checkpoints in this sweep. The CUDA
test suite retains one isolated NVFP4 numerical failure documented below.

## Environment

- Platform: Windows 11
- CPU path: AVX2, `out/windows-cpu-release/celeg-cpu-run.exe`
- CUDA path: RTX 3060, 12 GB; current runner at `out/windows-cuda-release/celeg-run.exe`
- Prompt: `What is the capital of France?`
- Generation: greedy, temperature 0, top-k 1, maximum 8 new tokens

## Build results

| Backend | Result | Evidence |
|---|---|---|
| CPU Release | PASS | `python scripts/dev.py build --backend cpu --build-type Release --jobs 12` |
| CUDA Release | PASS | `python scripts/dev.py build --backend cuda --build-type Release --jobs 12`; 583/583 targets built |

CPU CTest: 80/80 passed.

CUDA CTest: 90/91 passed. The only failure was the NVFP4 comparison at `tests/cuda_kernels_test.cu:679`; it exited `0xc0000409` after reporting `expected 0.410156, got 0.734375`. All other CUDA tests, including architecture-boundary and CUDA model tests, passed.

The runtime rows below use the current `HEAD` executables from `out/windows-cpu-release` and `out/windows-cuda-release`.

## Runtime results

`OK` means the process exited successfully. `Correct` means the output visibly contained the expected answer; raw prompt formatting and multiple-choice output are retained as observed.

| Model/artifact | CPU | CUDA |
|---|---|---|
| LFM2.5-230M Safetensors | OK, Correct | OK, Correct option |
| LFM2.5-350M Safetensors | OK, Correct | OK, Correct |
| MiniCPM5-1B Safetensors | OK, Correct | OK, Correct |
| LFM2.5-230M Q4_K_M | OK, Correct | OK, Correct |
| LFM2.5-230M Q6_K | OK, Correct | OK, Correct option |
| LFM2.5-350M Q4_0 | OK, Empty | OK, Empty |
| LFM2.5-350M Q4_K_M | OK, Correct | OK, Correct option |
| LFM2.5-350M Q5_K_M | OK, Correct | OK, Correct |
| LFM2.5-350M Q8_0 | OK, Correct option | OK, Correct |
| Qwen3.5-0.8B Q4_K_M | OK, Garbled | OK, Garbled |
| MiniCPM5-1B Q4_K_M | OK, Correct | OK, Correct |
| SmolLM3-3B Q4_K_M | OK, Correct | OK, Correct |
| Nanbeige 3B Q4_K_M | OK, Wrong/non-English | OK, Wrong/non-English |
| Nemotron 4B Q4_K_M | OK, Correct/partial | OK, Garbled |
| Ling-3.0-tiny-int4 Safetensors | OK, unclear fragment at 6.91 tok/s | FAIL, `cublasCreate` / VRAM exhausted |
| LFM2.5-8B-A1B Safetensors | OK, Correct option; 16.44 prefill / 17.15 decode tok/s | FAIL, `cublasCreate` / VRAM exhausted |
| Lizzy-7B-GGUF Q4_K_M | OK, Correct; 4.27 prefill / 6.19 decode tok/s | OK, Empty completion |
| Lizzy-7B Safetensors | OK, Correct; 4.18 prefill / 3.48 decode tok/s | OK, Wrong/incomplete: `What is the capital of Spain?` / `The` |

All 18 acquired artifacts were launched on both current-backend executables. The CPU runner reported `backend=cpu-native isa=avx2`; the CUDA runner used the RTX 3060 (compute capability 8.6).

The two Lizzy snapshots were completed through direct resumable transfers after
the standard HF client stalled. Obsolete incomplete fragments were removed;
only the verified Hub blobs and snapshot entries remain.

## Changes made to build

- Restored the nine CUDA model detail headers under `include/celeg/backend/cuda/model/detail/`, matching the include paths used by current CUDA sources.
- Added the CUDA toolkit include directory to `cuda_attention_capability_test` in `CMakeLists.txt` so its `cuda_bf16.h` dependency is visible to the C++ compiler.
- The unmatched-brace fix in `src/checkpoint/hf_http.cpp` is present in `HEAD`.

The LFM2.5-8B-A1B Safetensors snapshot is approximately 16.9 GB and fits
the CPU run, but not the reference GPU's 12 GB VRAM.
